"""HPO entrypoint. Edit and run.

Usage:

    uv run python app.py            # host-side: spawns N parallel worker containers
    python app.py                   # inside a worker container (auto via GYM_DR_WORKER=1)

This file defines:
  - ``base``: the base ``ExperimentConfig`` (everything not swept by HPO).
  - ``search_space(trial)``: returns a dotted-key overrides dict applied per
    trial via ``ExperimentConfig.with_overrides(**overrides)``.

The search includes:
  - PPO hyperparameters (learning_rate, ent_coef, n_steps, batch_size,
    gamma, gae_lambda, clip_range, n_epochs) + frame stacking.
  - **The full policy/value network**, AWS-DeepRacer-faithful:
      * CNN tower: a named DeepRacer arch (shallow/standard/deep) or a
        custom sampled conv stack — see ``gym_dr/networks.py``.
      * FC middleware: policy and value heads sized *independently*.
      * Separate actor/critic towers (``share_features_extractor=False``),
        matching AWS's ``use_separate_networks_per_head=True``.
      * Raw 0-255 grayscale input (``normalize_images=False`` +
        ``time_trial``'s grayscale wrapper) — matching what the car feeds.

To turn this into a single (non-HPO) training run, swap the bottom-of-file
``study(...)`` for ``train(experiment)`` and remove ``search_space``. See
``experiments/hpo_example.py`` for the canonical reference.
"""
from gym_dr import (
    ContinuousActionSpaceConfig,
    ExperimentConfig,
    ObjectAvoidanceConfig,
    Sb3Trainer,
    TrackingConfig,
    TrainingConfig,
    WorldsConfig,
    object_avoidance_aware,
    Study,
    time_trial,
    existing_tracks,
)
from gym_dr.networks import DEEPRACER_CONV_PRESETS, DeepRacerCNN
from gym_dr.rewards import REWARD_VARIANTS


# --------------------------------------------------------------------------- #
# Edit these to control the study. They're consumed only by the `study(...)`
# call at the bottom of the file (host orchestrator); the in-container worker
# reads N_TRIALS_PER_WORKER from env vars set by the host.
# --------------------------------------------------------------------------- #
STUDY_NAME = "object_avoidance_1"
N_TRIALS = 20
N_PARALLEL = 7   # number of concurrent Docker workers (each runs its own simapp)
SEED = 42        # int for reproducibility; None for nondeterministic


base = ExperimentConfig(
    name=STUDY_NAME,
    env_factory=time_trial,
    trainer=Sb3Trainer(
        name="ppo",
        policy="MultiInputPolicy",
        kwargs={
            "n_steps": 256,
            "batch_size": 64,
            "learning_rate": 3.0e-4,
            "ent_coef": 0.01,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "n_epochs": 10,
        },
        device="cuda",
    ),
    reward=object_avoidance_aware,
    # Fixed 4 static obstacles per episode for this study. min_spacing_m
    # caps how close two obstacles can be on the centerline projection;
    # 2.5 m gives the policy room to react on a 25 m track. Leaving
    # terminate_on_collision=True (the AWS-faithful default) — so a crash
    # ends the episode and CRASH_PENALTY shows up exactly once. Set False
    # for safety-style training where the per-step cost persists.
    object_avoidance=ObjectAvoidanceConfig(
        n_obstacles=4,
        min_spacing_m=2.5,
    ),
    action_space=ContinuousActionSpaceConfig(
        steering_low=-30.0,
        steering_high=30.0,
        # speed_low was 0.1 — the policy converged to crawling at that bound
        # in the previous study (mean speeds 0.10-0.16 m/s on 13/21 trials),
        # so trials "stayed on track" by barely moving. Raising the floor
        # makes crawling a non-option: the policy has to actually drive.
        speed_low=1.0,
        speed_high=4.0,
    ),
    # HPO uses worlds.names[0] for every trial; chunk_steps/rotations are
    # only consulted by the non-HPO host orchestrator.
    worlds=WorldsConfig(names=['Oval_track']),
    training=TrainingConfig(
        total_timesteps=250_000,       # per-trial training budget — bumped from
                                       # 20k; DeepRacer policies climb slowly and
                                       # the (now lenient) MedianPruner still kills
                                       # weak trials, so wall-clock is mostly the
                                       # good trials.
        checkpoint_freq=50_000,
        eval_freq=25_000,
        n_eval_episodes=3,
        rtf_override=10,
    ),
    tracking=TrackingConfig(mlflow_experiment=STUDY_NAME),
    #enable_gui=True,   # watch the car: VNC client -> localhost:5900
    seed=SEED,
    use_gpu=True
)


def _sample_conv_layers(trial) -> tuple:
    """Sample a custom DeepRacerCNN conv stack for this trial.

    Shape: two strided *downsampling* layers (big kernels) followed by
    ``cnn_refine_layers`` stride-1 *refinement* layers. DeepRacerCNN pads
    stride-1 layers so they don't collapse the feature map, so depth and
    kernel size sweep freely. Each entry is ``(filters, kernel, stride)``.
    Only used when ``cnn_arch == "custom"`` — otherwise a named DeepRacer
    preset (shallow/standard/deep) is used verbatim.
    """
    base_ch = trial.suggest_categorical("cnn_base_channels", [16, 32, 64])
    first_kernel = trial.suggest_categorical("cnn_first_kernel", [5, 8])
    refine_kernel = trial.suggest_categorical("cnn_refine_kernel", [3, 5])
    n_refine = trial.suggest_int("cnn_refine_layers", 1, 3)

    layers = [
        (base_ch, first_kernel, 4),   # downsample
        (base_ch * 2, 4, 2),          # downsample
    ]
    for _ in range(n_refine):         # stride-1 refinement (channels held)
        layers.append((base_ch * 2, refine_kernel, 1))
    return tuple(layers)


def search_space(trial) -> dict:
    """Per-trial overrides applied through ``ExperimentConfig.with_overrides``.

    Dotted keys walk into dataclasses and dicts; ``trainer.kwargs.*`` lands
    in the SB3 algorithm's constructor, and ``trainer.kwargs.policy_kwargs``
    is replaced wholesale — so the dict below carries *everything* the
    policy network needs: the CNN extractor class + its conv spec, the
    per-head FC middleware, and the AWS-faithful policy flags.
    """
    # --- PPO hyperparameters ------------------------------------------------
    overrides: dict = {
        "trainer.kwargs.learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "trainer.kwargs.ent_coef":      trial.suggest_float("ent_coef", 1e-4, 1e-1, log=True),
        "trainer.kwargs.n_steps":       trial.suggest_categorical("n_steps", [128, 256, 512, 1024]),
        "trainer.kwargs.batch_size":    trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "trainer.kwargs.gamma":         trial.suggest_float("gamma", 0.95, 0.999),
        "trainer.kwargs.gae_lambda":    trial.suggest_float("gae_lambda", 0.9, 0.99),
        "trainer.kwargs.clip_range":    trial.suggest_float("clip_range", 0.1, 0.3),
        "trainer.kwargs.n_epochs":      trial.suggest_int("n_epochs", 4, 12),
        # Frame stacking — DeepRacerEnv emits single frames; stacking gives
        # the policy implicit temporal context. AWS's default is 1.
        "trainer.frame_stack":          trial.suggest_int("frame_stack", 4, 4),
    }

    # --- Reward function -----------------------------------------------------
    # Sweep the *training* reward across the registered variants in
    # gym_dr/rewards.py. Optuna's suggest_categorical only accepts hashable
    # scalars (no function objects), so we sample a name and look up the
    # callable. The *evaluation* reward stays fixed (ExperimentConfig.eval_reward
    # defaults to progress_per_step) so trials trained with different rewards
    # can still be compared fairly on `dr/ep_eval_reward` in MLflow/TB.
    reward_name = trial.suggest_categorical("reward_fn", list(REWARD_VARIANTS))
    overrides["reward"] = REWARD_VARIANTS[reward_name]

    # --- CNN tower: a named DeepRacer arch, or a custom sampled stack -------
    cnn_arch = trial.suggest_categorical("cnn_arch", ["shallow", "standard", "deep", "custom"])
    if cnn_arch == "custom":
        conv_layers = _sample_conv_layers(trial)
    else:
        conv_layers = DEEPRACER_CONV_PRESETS[cnn_arch]
    features_dim = trial.suggest_categorical("features_dim", [256, 512, 1024])

    # --- FC middleware: policy and value heads sized INDEPENDENTLY ----------
    # With share_features_extractor=False each head also gets its own CNN
    # tower, so pi and vf are fully separate networks (AWS-faithful).
    pi_width = trial.suggest_categorical("pi_width", [256, 512, 1024])
    pi_depth = trial.suggest_int("pi_depth", 1, 3)
    vf_width = trial.suggest_categorical("vf_width", [256, 512, 1024])
    vf_depth = trial.suggest_int("vf_depth", 1, 3)

    overrides["trainer.kwargs.policy_kwargs"] = {
        # AWS DeepRacer uses separate actor/critic networks and feeds the
        # model raw 0-255 grayscale (no /255). See gym_dr/networks.py.
        "share_features_extractor": False,
        "normalize_images": False,
        "features_extractor_class": DeepRacerCNN,
        "features_extractor_kwargs": {
            "conv_layers": conv_layers,
            "features_dim": features_dim,
        },
        "net_arch": dict(pi=[pi_width] * pi_depth, vf=[vf_width] * vf_depth),
    }
    return overrides


# Alias so the host-side `prepare-metadata` step (and `inspect`) can read
# the action space and other shared fields off this file.
experiment = base


if __name__ == "__main__":
    Study(
        base,
        search_space,
        study_name=STUDY_NAME,
        n_trials=N_TRIALS,
        n_parallel=N_PARALLEL,
    ).run()
