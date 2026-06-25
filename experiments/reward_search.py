"""Reward search (2026-06-23) — find a training reward that produces clean,
fast laps, after the D3 baseline converged to "floor it, crash at ~28%".

Two-stage approach:
  1. **Offline shape filter** (`scripts/reward_ranking.py`, no sim): only
     progress-normalized rewards (`progress_complete`, `progress_per_step`) rank a
     clean lap above both a fast crash AND a slow crawl. Centerline/per-step
     rewards have a crawl trap. So the search focuses on progress-style rewards.
  2. **This training search** (Optuna, needs the sim): sweeps the reward family +
     its key hyperparameters (and a little PPO lr/entropy, since the fast-crash is
     partly an *optimization* problem — the policy can't yet learn to corner),
     SHORT trials on the held-out split, scored by the clean-completion eval.

Run AFTER D3 frees the sim (uses the 2-worker software-render sweet spot,
`docs/reports/throughput.md`):
    uv run --no-sync python experiments/reward_search.py
"""
from gym_dr import (
    ContinuousActionSpaceConfig,
    ExperimentConfig,
    Sb3Trainer,
    ACL,
    TraceConfig,
    TrackingConfig,
    TrainingConfig,
    clean_completion,
    time_trial,
)
from gym_dr.app import study
from gym_dr.networks import DeepRacerCNN
from gym_dr.rewards import make_progress_reward, make_weighted_reward, progress_complete

STUDY_NAME = "reward_search"
TRAIN_WORLDS = ["Spain_track", "Monaco", "Austin", "arctic_pro", "caecer_gp"]
EVAL_WORLDS = ["Bowtie_track", "jyllandsringen_pro", "penbay_pro"]
CHUNK_STEPS = 100_000
N_CHUNKS = 5                 # 500k steps/trial — enough to rank rewards, not to master
N_TRIALS = 16              # first pass; the Optuna study persists (SQLite) so it's extendable
N_PARALLEL = 2             # sw-render 2-worker sweet spot (docs/reports/throughput.md)
SEEDS = [42, 7, 123]


base = ExperimentConfig(
    name=STUDY_NAME,
    env_factory=time_trial,
    reward=progress_complete,       # swept below; this is the default/seed
    eval_reward=clean_completion,   # fair cross-reward yardstick
    trainer=Sb3Trainer(
        name="ppo", policy="MultiInputPolicy",
        kwargs={
            "n_steps": 2048, "batch_size": 256, "learning_rate": 3e-4,
            "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95,
            "clip_range": 0.2, "n_epochs": 10, "target_kl": 0.08,
            "policy_kwargs": {
                "share_features_extractor": False, "normalize_images": False,
                "features_extractor_class": DeepRacerCNN,
                "features_extractor_kwargs": {
                    "conv_layers": [[32, 8, 4], [64, 4, 2], [64, 3, 1]], "features_dim": 512},
                "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            },
        },
        frame_stack=4, device="cuda",
    ),
    action_space=ContinuousActionSpaceConfig(
        steering_low=-30.0, steering_high=30.0, speed_low=1.0, speed_high=4.0,
        normalize_actions=True),
    world_strategy=ACL(
        train_worlds=TRAIN_WORLDS, eval_worlds=EVAL_WORLDS,
        chunk_steps=CHUNK_STEPS, n_chunks=N_CHUNKS, unlock_every=2,
        recency_weight=2.0, seed=42),
    training=TrainingConfig(
        total_timesteps=CHUNK_STEPS * N_CHUNKS, checkpoint_freq=CHUNK_STEPS,
        eval_freq=CHUNK_STEPS, n_eval_episodes=3, rtf_override=160),
    tracking=TrackingConfig(mlflow_experiment=STUDY_NAME),
    trace=TraceConfig(enabled=True), seed=42, use_gpu=True,
)


def search_space(trial) -> dict:
    seed = SEEDS[trial.number % len(SEEDS)]
    trial.set_user_attr("seed", seed)
    overrides: dict = {
        "seed": seed,
        "trainer.kwargs.learning_rate": trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True),
        "trainer.kwargs.ent_coef": trial.suggest_float("ent_coef", 1e-3, 5e-2, log=True),
    }
    # Reward family + its hyperparameters → build the callable, pass as override.
    family = trial.suggest_categorical("reward_family", ["progress_complete", "weighted_corner"])
    if family == "progress_complete":
        overrides["reward"] = make_progress_reward(
            step_penalty=trial.suggest_float("pc_step_penalty", 0.1, 0.6),
            completion_bonus=trial.suggest_float("pc_completion_bonus", 50.0, 200.0),
            center_bonus=trial.suggest_float("pc_center_bonus", 0.0, 0.2),
        )
    else:
        overrides["reward"] = make_weighted_reward(
            w_center=trial.suggest_float("w_center", 0.5, 1.5),
            w_speed=trial.suggest_float("w_speed", 0.1, 0.8),
            w_corner=trial.suggest_float("w_corner", 0.2, 1.0),
            w_align=trial.suggest_float("w_align", 0.1, 0.5),
            w_pace=trial.suggest_float("w_pace", 0.1, 0.6),
        )
    return overrides


if __name__ == "__main__":
    # Software rendering (Mesa llvmpipe) so the GPU only does NN inference and the
    # 2 workers parallelize rendering across CPU cores instead of contending on the
    # one GPU's render queue. The throughput sweep (docs/reports/throughput.md)
    # found this is the only lever that scales: ~83 steps/s at 2 workers vs the
    # GPU-render ceiling of ~54 that 2 GPU-rendered workers actually fall *below*
    # (~40 aggregate, contended). 2 is the sweet spot (4 oversubscribes the CPU).
    study(base, search_space, study_name=STUDY_NAME, n_trials=N_TRIALS,
          n_parallel=N_PARALLEL,
          extra_env={"LIBGL_ALWAYS_SOFTWARE": "1", "GALLIUM_DRIVER": "llvmpipe"})
