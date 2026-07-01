"""Phase-2 end-to-end **pure PPO** run — everything EXCEPT the Lagrangian.

The maintainer's first real objective: a single end-to-end training that combines
all the non-safe-RL machinery —
  * the DeepRacer CNN architecture (separate actor/critic towers, P3 action norm),
  * the ACL curriculum (spaced-repetition over the training tracks),
  * **Domain Randomization with ADR** (actuator + observation noise that grows
    automatically as held-out clean-completion improves),
  * clean-completion eval on a DISJOINT held-out split + the live
    `eval/generalization_gap` scalar,
— trained with plain PPO (NO cost/Lagrangian; that's the later W-saferl step).

This is `p1p3_validation.py` (the no-DR baseline = D3) **plus** DR/ADR. Run it
after D3 finishes and (optionally) after the perception net is trained, if we
decide to use it as the actor front-end:

    uv run --no-sync python experiments/end_to_end_ppo.py

ADR note: each knob is a `Range(low, high)` whose `high` is the CEILING. ADR keeps
a live `cur_high` that starts near `low` and expands toward `high` when the held-out
clean-completion rate clears `promote` (0.7), shrinking below `demote` (0.3) — so
robustness ramps to the hardest level the policy can handle, no manual schedule.
random_start/random_direction are ON (need the patched deepracer-env reset modes —
docs/reports/domain-randomization.md).
"""

from gym_dr import (
    ACL,
    ADR,
    ContinuousActionSpaceConfig,
    ExperimentConfig,
    Range,
    Sb3Trainer,
    TraceConfig,
    TrackingConfig,
    TrainingConfig,
    TRACKS,
    centerline_quadratic,
    clean_completion,
    time_trial,
    Study,
)
from gym_dr.networks import DeepRacerCNN

NAME = "end_to_end_ppo"

# Same disjoint sim-only split as the D3 no-DR baseline (physical tracks reserved).
TRAIN_WORLDS = ["Spain_track", "Monaco", "Austin", "arctic_pro", "caecer_gp"]
EVAL_WORLDS = ["Bowtie_track", "jyllandsringen_pro", "penbay_pro"]
CHUNK_STEPS = 100_000
N_CHUNKS = 40                       # 40 * 100k = 4M steps
TOTAL_TIMESTEPS = CHUNK_STEPS * N_CHUNKS

_PHYSICAL = {"reInvent2019_track", "Oval_track", "reinvent_base"}
_bad = (set(TRAIN_WORLDS) | set(EVAL_WORLDS)) & _PHYSICAL
assert not _bad, f"physical/reserved tracks must not be in train/eval: {sorted(_bad)}"
_unknown = sorted((set(TRAIN_WORLDS) | set(EVAL_WORLDS)) - set(TRACKS))
assert not _unknown, f"unknown track name(s): {_unknown}"
assert not (set(TRAIN_WORLDS) & set(EVAL_WORLDS)), "train/eval must be disjoint"


# Automatic Domain Randomization — ranges below are the CEILINGS ADR grows toward.
# random_start/random_direction are deepracer-env reset modes (now supported); they
# need the patched sim — rebuild the image or set GYM_DR_DEEPRACER_ENV_SRC to the
# deepracer-env checkout (gym_dr/docker_runner.py).
DR = ADR(
    steering_noise=Range(0.0, 3.0),  # deg  (±30 steering range)
    speed_noise=Range(0.0, 0.15),    # m/s  (1–4 speed range)
    obs_gaussian=Range(0.0, 10.0),   # 0–255 grayscale additive
    obs_brightness=Range(0.0, 0.2),  # ±20% per-step multiplicative
    drag=Range(0.7, 1.0),            # per-episode throttle->speed factor (sim2real)
    friction=Range(0.8, 1.5),        # per-spawn wheel-mu multiplier (baseline 1.5)
    random_start=True,               # uniform valid start each episode (state coverage)
    random_direction=True,           # random lap direction each episode
    step=0.1, promote=0.7, demote=0.3, seed=42,
)


experiment = ExperimentConfig(
    name=NAME,
    env_factory=time_trial,
    reward=centerline_quadratic,        # smooth centerline + pace + steering smoothness
    eval_reward=clean_completion,       # P1 yardstick (also the default)
    domain_randomization=DR,            # <-- the only addition vs the D3 baseline
    trainer=Sb3Trainer(
        name="ppo",
        policy="MultiInputPolicy",
        kwargs={
            "n_steps": 2048,
            "batch_size": 256,
            "learning_rate": 3.0e-4,    # sane LR (P3) — not the 2e-5 that froze trial_18
            "ent_coef": 0.01,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "n_epochs": 10,
            "target_kl": 0.08,          # collapse/explosion guard (P3)
            "policy_kwargs": {
                "share_features_extractor": False,
                "normalize_images": False,
                "features_extractor_class": DeepRacerCNN,
                "features_extractor_kwargs": {
                    "conv_layers": [[32, 8, 4], [64, 4, 2], [64, 3, 1]],
                    "features_dim": 512,
                },
                "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            },
        },
        frame_stack=4,
        device="cuda",
    ),
    action_space=ContinuousActionSpaceConfig(
        steering_low=-30.0, steering_high=30.0,
        speed_low=1.0, speed_high=4.0,
        normalize_actions=True,         # P3
    ),
    world_strategy=ACL(
        train_worlds=TRAIN_WORLDS,
        eval_worlds=EVAL_WORLDS,
        chunk_steps=CHUNK_STEPS,
        n_chunks=N_CHUNKS,
        unlock_every=4,
        recency_weight=2.0,
        seed=42,
    ),
    training=TrainingConfig(
        total_timesteps=TOTAL_TIMESTEPS,
        checkpoint_freq=CHUNK_STEPS,
        checkpoint_keep_last=3,
        eval_freq=CHUNK_STEPS,          # ADR + generalization-gap update each eval
        n_eval_episodes=3,
        rtf_override=160,
        eval_path_plots=True,
    ),
    tracking=TrackingConfig(mlflow_experiment=NAME),
    trace=TraceConfig(enabled=True),
    seed=42,
    use_gpu=True,
)


if __name__ == "__main__":
    Study(experiment).run()