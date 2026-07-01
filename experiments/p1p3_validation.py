"""Validation run for the P1+P3+D5 fixes — the "best shot" generalization run.

This is the experiment that tests, together, everything the scope review +
maintainer decisions changed (see docs/reports/scope-review.md,
docs/questions-for-maintainer.md):

  * P1 — eval is scored by `clean_completion` (finish the lap WITHOUT leaving
    the track, reasonable speed) and the per-held-out-world
    `eval/<world>_clean_completion_rate` metric. (now the default eval_reward)
  * P3 — `normalize_actions=True` (default): the policy acts in [-1,1] so PPO
    actually explores steering (the raw-Box ~±1° was the trial_18 root cause),
    plus a sane learning rate (3e-4, not the HPO-sampled 2e-5) and a `target_kl`
    guard against the std-blow-up seen at high LR.
  * D5 — `ACL` (Automatic Curriculum Learning): spaced-repetition over the training tracks
    (newer favoured, older always revisited) to fight the catastrophic
    forgetting the strict curriculum showed.
  * A real budget — ~4M steps (the 600k HPO trials were structurally
    under-trained), held out on DISJOINT sim tracks.

Reserved physical tracks (reInvent2019_track, Oval_track) are deliberately NOT
in train or eval — they are scored out-of-loop by scripts/eval_physical_tracks.py
(see docs/eval-protocol.md).

Run (LAST — it saturates the machine):

    uv run python experiments/p1p3_validation.py

Watch real-time-factor: the host log prints the schedule; while it runs, confirm
the sim actually hits rtf_override (or note the throttled effective rate) — the
maintainer observed fps fluctuating for up to ~30 min before settling.
"""

from gym_dr import (
    ContinuousActionSpaceConfig,
    ExperimentConfig,
    Sb3Trainer,
    ACL,
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

NAME = "p1p3_validation"

# Sim-only split — physical tracks (reInvent2019_track, Oval_track) are reserved
# for the out-of-loop evaluator and must NOT appear here.
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


experiment = ExperimentConfig(
    name=NAME,
    env_factory=time_trial,
    reward=centerline_quadratic,        # smooth centerline + pace + steering smoothness
    eval_reward=clean_completion,       # P1 yardstick (also the new default)
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
                # Smaller than trial_18's [1024]*3 heads (scope review R3).
                "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            },
        },
        frame_stack=4,
        device="cuda",
    ),
    action_space=ContinuousActionSpaceConfig(
        steering_low=-30.0, steering_high=30.0,
        speed_low=1.0, speed_high=4.0,
        normalize_actions=True,         # P3 (default; explicit for clarity)
    ),
    world_strategy=ACL(
        train_worlds=TRAIN_WORLDS,
        eval_worlds=EVAL_WORLDS,
        chunk_steps=CHUNK_STEPS,
        n_chunks=N_CHUNKS,
        unlock_every=4,                 # unlock a new track every 4 chunks
        recency_weight=2.0,
        seed=42,
    ),
    training=TrainingConfig(
        total_timesteps=TOTAL_TIMESTEPS,
        checkpoint_freq=CHUNK_STEPS,
        checkpoint_keep_last=3,
        eval_freq=CHUNK_STEPS,
        n_eval_episodes=3,
        rtf_override=160,               # confirm this is actually achieved (D4)
        eval_path_plots=True,
    ),
    tracking=TrackingConfig(mlflow_experiment=NAME),
    trace=TraceConfig(enabled=True),
    seed=42,
    use_gpu=True,
)


if __name__ == "__main__":
    Study(experiment).run()