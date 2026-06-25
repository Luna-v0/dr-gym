"""Classic-PPO training on FEATURE observations (the reward elements), with
domain randomization, track changing, and random spawn — no camera, no Lagrangian.

Per the maintainer's request:
  * **feature obs** — the policy sees the ALL_FEATURES vector built from
    reward_params (camera-off path, ``camera_obs=False`` -> ``feature_time_trial``),
    NOT pixels. Fast (no rendering) and the "reward elements we defined as important".
  * **domain randomization** — actuator (steering/speed) noise + ADR that grows the
    noise as held-out clean-completion improves.
  * **track changing** — ACL rotates across the training tracks;
    held-out eval split stays disjoint (generalization signal).
  * **random spawn** — random_start + random_direction (deepracer-env reset modes,
    now NUMBER_OF_RESETS=0 so off-track is immediately terminal — see the multi-car
    impossible-drift fix). Each episode begins at a uniform valid track location.
  * **classic RL** — plain PPO (MlpPolicy over the feature vector), no cost/Lagrangian.

Single car: the feature+DR+random_start+curriculum stack is only wired on the
single-car path (gym_dr/envs/time_trial.py); multi_car doesn't carry DR/random_start
yet. The throughput benchmark (scripts/multicar_throughput.py) covers cars-per-sim
separately.

    GYM_DR_DEEPRACER_ENV_SRC=/home/lunav0/Projects/deepracer-env/deepracer_env \
      uv run --no-sync python experiments/feature_dr_training.py
"""
import os

# 11-feature ACTOR vector (camera-off). Set host+container (the container re-imports
# this module) so dispatch builds the same observation. See oracle_feature_study.py.
os.environ["GYM_DR_FEATURE_SET"] = "actor_extended"

from gym_dr import (                                       # noqa: E402
    ACL,
    ADR,
    ContinuousActionSpaceConfig,
    EnvironmentConfig,
    ExperimentConfig,
    FeatureObs,
    Range,
    Sb3Trainer,
    TraceConfig,
    TrackingConfig,
    TrainingConfig,
    TRACKS,
    centerline_quadratic,
    clean_completion,
    train,
)
from gym_dr.envs.dispatch import build_env                 # noqa: E402
from gym_dr.perception import ACTOR_FEATURES               # noqa: E402

NAME = "feature_dr_training"

# Disjoint sim-only split; physical tracks reserved for out-of-loop eval.
TRAIN_WORLDS = ["Spain_track", "Monaco", "Austin", "arctic_pro", "caecer_gp"]
EVAL_WORLDS = ["Bowtie_track", "jyllandsringen_pro", "penbay_pro"]
CHUNK_STEPS = 50_000
N_CHUNKS = 40                       # 40 * 50k = 2M steps
TOTAL = CHUNK_STEPS * N_CHUNKS

_PHYSICAL = {"reInvent2019_track", "Oval_track", "reinvent_base"}
assert not (set(TRAIN_WORLDS) | set(EVAL_WORLDS)) & _PHYSICAL
assert not sorted((set(TRAIN_WORLDS) | set(EVAL_WORLDS)) - set(TRACKS)), "unknown track"
assert not (set(TRAIN_WORLDS) & set(EVAL_WORLDS)), "train/eval must be disjoint"

# DR (ADR): actuator-noise knobs grow 0->high as held-out clean-completion improves;
# random_start/random_direction spread state coverage; drag + per-spawn friction are
# the sim2real knobs. obs_* are camera-only (no effect on feature obs) so left at 0.
DR = ADR(
    steering_noise=Range(0.0, 3.0),  # deg  (±30 steering range)
    speed_noise=Range(0.0, 0.15),    # m/s  (1–4 speed range)
    drag=Range(0.7, 1.0),            # per-episode throttle->speed factor (sim2real)
    friction=Range(0.8, 1.5),        # per-spawn wheel-mu multiplier (baseline 1.5)
    random_start=True,               # uniform valid start each episode
    random_direction=True,           # random lap direction each episode
    step=0.1, promote=0.7, demote=0.3, seed=42,
)

ENV = EnvironmentConfig(
    observation=FeatureObs(features=tuple(ACTOR_FEATURES)),  # camera-off, 11-feature
    action_space=ContinuousActionSpaceConfig(
        steering_low=-30.0, steering_high=30.0, speed_low=1.0, speed_high=4.0,
        normalize_actions=True),
    curriculum=ACL(
        train_worlds=TRAIN_WORLDS, eval_worlds=EVAL_WORLDS,
        chunk_steps=CHUNK_STEPS, n_chunks=N_CHUNKS,
        unlock_every=4, recency_weight=2.0, seed=42),
    domain_randomization=DR,
    n_cars=1, reward=centerline_quadratic, eval_reward=clean_completion,
)

experiment = ExperimentConfig(
    name=NAME,
    environment=ENV,
    env_factory=build_env,          # dispatches (1, feature) -> feature_time_trial
    trainer=Sb3Trainer(
        name="ppo",
        policy="MlpPolicy",         # low-dim feature vector -> MLP, no CNN
        kwargs={
            "n_steps": 2048, "batch_size": 256, "learning_rate": 3.0e-4,
            "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95,
            "clip_range": 0.2, "n_epochs": 10,
            "target_kl": 0.08,      # P3 collapse/explosion guard
            "policy_kwargs": {"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
        },
        frame_stack=1, device="cpu",
    ),
    training=TrainingConfig(
        total_timesteps=TOTAL, checkpoint_freq=CHUNK_STEPS, checkpoint_keep_last=3,
        eval_freq=CHUNK_STEPS, n_eval_episodes=3, rtf_override=60,
        eval_path_plots=True),
    tracking=TrackingConfig(mlflow_experiment=NAME),
    trace=TraceConfig(enabled=True),
    seed=42, use_gpu=False,
)


if __name__ == "__main__":
    train(experiment)
