"""HPO entrypoint. Edit and run.

Usage:

    uv run python app.py            # host-side: spawns N parallel worker containers
    python app.py                   # inside a worker container (auto via GYM_DR_WORKER=1)

This file defines:
  - ``base``: the base ``ExperimentConfig`` (everything not swept by HPO).
  - ``search_space(trial)``: returns a dotted-key overrides dict applied per
    trial via ``ExperimentConfig.with_overrides(**overrides)``.

This study is a **multi-world, multi-seed time-trial HPO**:

  - **Multi-world.** ``world_strategy=OrderedSplit`` trains each trial across an
    ordered list of tracks and *evaluates on a different, held-out list* — so
    the Optuna objective is track *generalisation*, not single-track overfit.
    Because the strategy schedules more than one training chunk, the HPO host
    puts each worker into runtime track-rotation mode: a trial trains
    ``CHUNK_STEPS`` on the first world, hot-swaps the Gazebo track in place via
    ``DeepRacerEnv.set_world`` (no container restart), trains the next, etc.,
    then measures the policy on every held-out eval world.
  - **Multi-seed.** Each trial is seeded from a rotating pool of ``SEEDS``
    (cycled by trial number — see ``search_space``) so the search samples
    configs across several seeds rather than chasing one lucky RNG draw. The
    seed is recorded as a trial user-attr for traceability. (The Optuna TPE
    *sampler* is seeded separately from ``base.seed`` for a reproducible
    search trajectory.)
  - **Time-trial.** ``env_factory=time_trial`` with no ``object_avoidance``
    config = pure time-trial driving. The training reward is swept across the
    time-trial reward variants; the eval reward stays fixed (``progress_safe``)
    so trials trained on different rewards are comparable on the same metric.

The search also covers the PPO hyperparameters and the full
AWS-DeepRacer-faithful policy/value network (CNN tower + per-head FC
middleware, separate actor/critic towers, raw 0-255 grayscale input) — see
``gym_dr/networks.py`` and ``search_space`` below.

To turn this into a single (non-HPO) training run, swap the bottom-of-file
``study(...)`` for ``train(experiment)`` and remove ``search_space``. See
``experiments/ordered_split_example.py`` for the canonical multi-world
``train()`` reference.
"""
from gym_dr import (
    TRACKS,
    ContinuousActionSpaceConfig,
    ExperimentConfig,
    OrderedSplit,
    Sb3Trainer,
    TrackingConfig,
    TrainingConfig,
    center_line,
    study,
    time_trial,
)
from gym_dr.networks import DEEPRACER_CONV_PRESETS, DeepRacerCNN
from gym_dr.rewards import REWARD_VARIANTS


# --------------------------------------------------------------------------- #
# Edit these to control the study. They're consumed only by the `study(...)`
# call at the bottom of the file (host orchestrator); the in-container worker
# reads N_TRIALS_PER_WORKER from env vars set by the host.
# --------------------------------------------------------------------------- #
STUDY_NAME = "tt_multiworld"
N_TRIALS = 20
N_PARALLEL = 7   # number of concurrent Docker workers (each runs its own simapp)

# Per-trial training seeds, cycled by trial number in search_space() so the
# search spreads across seeds instead of trusting one. NOT a TPE-optimised
# dimension on purpose — letting Optuna "tune" the seed would just overfit the
# search to lucky RNG draws.
SEEDS = [382, 829, 17, 720, 233]

# Seed for the Optuna TPE sampler (the search trajectory), kept separate from
# the per-trial training seeds above. The HPO worker offsets it by WORKER_INDEX
# so parallel workers don't sample in lockstep.
SAMPLER_SEED = 42

# --- World schedule: train on these tracks, evaluate generalisation on the ---
# held-out ones. The strategy having >1 training chunk is what flips the HPO
# host into multi-world rotation mode (see gym_dr/app.py::study). All worlds
# below are real, geometrically distinct circuits confirmed present in the
# simapp image (degenerate shapes like Oval_track / Straight_track / H_track
# and the reinvent_base starter track are intentionally excluded). Train and
# eval sets are disjoint so eval reward measures transfer to UNSEEN tracks.
TRAIN_WORLDS = [
    "Spain_track",        # Circuit de Barcelona-Catalunya
    "Monaco",             # European Seaside Circuit
    "Austin",             # American Hills Speedway
    "arctic_pro",         # Hot Rod Super Speedway
    "caecer_gp",          # Vivalas Speedway
    "penbay_pro",         # Po-Chun Super Speedway
]
EVAL_WORLDS = [
    "reInvent2019_track", # Smile Speedway
    "Bowtie_track",       # Bowtie Track
    "jyllandsringen_pro", # Cosmic Circuit
]
CHUNK_STEPS = 100_000      # timesteps trained per world before the track swap
ROTATIONS = 1              # full passes through TRAIN_WORLDS per trial
# Per-trial budget = CHUNK_STEPS x worlds x rotations. The MedianPruner warms
# up as a fraction of this, so it must reflect the *real* per-trial total.
TOTAL_TIMESTEPS = CHUNK_STEPS * len(TRAIN_WORLDS) * ROTATIONS   # 600k / trial

# Fail fast on a typo or an accidental train/eval overlap — a bad world name
# would otherwise only surface deep inside DeepRacerEnv.set_world (ValueError)
# mid-trial, after the container is already up. Validate names against the
# catalog and enforce the held-out split here at import time.
_unknown = sorted(set(TRAIN_WORLDS + EVAL_WORLDS) - set(TRACKS))
assert not _unknown, f"unknown world name(s) not in gym_dr.tracks.TRACKS: {_unknown}"
_overlap = sorted(set(TRAIN_WORLDS) & set(EVAL_WORLDS))
assert not _overlap, f"train/eval worlds must be disjoint; overlap: {_overlap}"


base = ExperimentConfig(
    name=STUDY_NAME,
    env_factory=time_trial,        # pure time-trial (no object_avoidance config)
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
        frame_stack=4,                  # temporal context (DeepRacerEnv emits single frames)
        device="cuda",
    ),
    reward=center_line,                 # time-trial reward (swept in search_space)
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
    # Multi-world train/eval split. OrderedSplit trains across TRAIN_WORLDS in
    # order (hot-swapping the track between CHUNK_STEPS-sized chunks) and scores
    # the policy on the held-out EVAL_WORLDS each evaluation — the per-world
    # means are logged as eval/<world>_mean_reward and their mean is the Optuna
    # objective. The HPO host detects the multi-chunk schedule and rotates worlds
    # within every trial.
    world_strategy=OrderedSplit(
        train_worlds=TRAIN_WORLDS,
        eval_worlds=EVAL_WORLDS,
        chunk_steps=CHUNK_STEPS,
        rotations=ROTATIONS,
    ),
    training=TrainingConfig(
        total_timesteps=TOTAL_TIMESTEPS,   # per-trial budget across all worlds
        checkpoint_freq=50_000,
        # Each eval rolls out n_eval_episodes on EVERY held-out world, so the
        # eval cost scales with len(EVAL_WORLDS). With 3 eval worlds, 50k keeps
        # the eval count (~12/trial) and total eval episodes sane.
        eval_freq=50_000,
        n_eval_episodes=3,
        rtf_override=10,
    ),
    tracking=TrackingConfig(mlflow_experiment=STUDY_NAME),
    #enable_gui=True,   # watch the car: VNC client -> localhost:5900
    seed=SAMPLER_SEED,  # seeds the Optuna sampler; per-trial training seed is set in search_space
    use_gpu=True,
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


# Time-trial reward variants to sweep. Drawn from gym_dr/rewards.py's registry
# minus object_avoidance_aware (that one consumes is_crashed / closest_objects
# params that only exist when object avoidance is enabled — irrelevant to a
# pure time-trial study).
TIME_TRIAL_REWARDS = [
    name for name in REWARD_VARIANTS if name != "object_avoidance_aware"
]


def search_space(trial) -> dict:
    """Per-trial overrides applied through ``ExperimentConfig.with_overrides``.

    Dotted keys walk into dataclasses and dicts; ``trainer.kwargs.*`` lands
    in the SB3 algorithm's constructor, and ``trainer.kwargs.policy_kwargs``
    is replaced wholesale — so the dict below carries *everything* the
    policy network needs: the CNN extractor class + its conv spec, the
    per-head FC middleware, and the AWS-faithful policy flags.
    """
    # --- Per-trial training seed --------------------------------------------
    # Cycle through SEEDS by trial number so the study covers several seeds.
    # Applied as a plain override (experiment.seed) — NOT a trial.suggest_*, so
    # TPE never tries to "optimise" the seed. Recorded as a user-attr so the
    # seed behind each run_name is recoverable from optuna-dashboard.
    seed = SEEDS[trial.number % len(SEEDS)]
    trial.set_user_attr("seed", seed)

    # --- PPO hyperparameters ------------------------------------------------
    overrides: dict = {
        "seed": seed,
        "trainer.kwargs.learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "trainer.kwargs.ent_coef":      trial.suggest_float("ent_coef", 1e-4, 1e-1, log=True),
        "trainer.kwargs.n_steps":       trial.suggest_categorical("n_steps", [128, 256, 512, 1024]),
        "trainer.kwargs.batch_size":    trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "trainer.kwargs.gamma":         trial.suggest_float("gamma", 0.95, 0.999),
        "trainer.kwargs.gae_lambda":    trial.suggest_float("gae_lambda", 0.9, 0.99),
        "trainer.kwargs.clip_range":    trial.suggest_float("clip_range", 0.1, 0.3),
        "trainer.kwargs.n_epochs":      trial.suggest_int("n_epochs", 4, 12),
    }

    # --- Reward function -----------------------------------------------------
    # Sweep the *training* reward across the time-trial variants. Optuna's
    # suggest_categorical only accepts hashable scalars (no function objects),
    # so we sample a name and look up the callable. The *evaluation* reward
    # stays fixed (ExperimentConfig.eval_reward defaults to progress_safe) so
    # trials trained with different rewards can still be compared fairly on
    # eval/mean_reward across the held-out worlds.
    reward_name = trial.suggest_categorical("reward_fn", TIME_TRIAL_REWARDS)
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
    study(
        base,
        search_space,
        study_name=STUDY_NAME,
        n_trials=N_TRIALS,
        n_parallel=N_PARALLEL,
    )
