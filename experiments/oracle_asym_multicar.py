"""Asymmetric-critic feature oracle — MULTI-CAR (n=12) for proper DR aggregation.

Domain randomization wants MANY variations aggregated per PPO update, not one
variation per episode. Here the **12 cars each drive a different train track with
their own per-episode DR** (drag / steering+speed bias / actuator + feature noise),
so a single gradient step averages over 12 conditions at once — a far stronger,
less-biased robustness signal than the single-car oracle (oracle_asym_robust.py),
which sees one variation per episode. It's also ~6x faster on the 22-core laptop
(feature obs scales ~linearly; 12 = one car per train track, under the Gazebo
~12-13 separate-track-instance spawn cap).

LEARNABILITY (why a first n=18 attempt flatlined at 5% progress / 2-step episodes):
multi_car applied every DR magnitude at FULL strength from step 0 (it has no in-loop
held-out eval to drive feedback-ADR), so the policy was stuck in an unlearnable POMDP
— a ±15° per-episode steering bias is unobservable in a single frame, and no single
correction works for both signs. Two fixes here:
  1. DR WARMUP — multi_car ramps all DR magnitudes 0->full over the first ~20% of
     training (GYM_DR_DR_WARMUP_STEPS), so it learns to drive first, then to counter
     the perturbations as they grow.
  2. FRAME-STACK MEMORY (frame_stack=4) — stacking the last 4 feature vectors exposes
     the drift signature so the net can infer + counter the bias online (system-id
     without a recurrent policy; ports to a Lagrangian/Safety-Gym PPO unchanged).

Same study as the single-car oracle: actor sees the NOISED 11-feature vector, the
asymmetric critic sees the TRUE one (gym_dr.asymmetric.AsymmetricActorCriticPolicy);
feature_noise + actuator/drag/bias/friction DR; 18 train / 8 held-out tracks chosen
by max-min over the wobble x tightness map.

The 18 cars cover ALL 18 train tracks every step (GYM_DR_DEMO_WORLDS), so no in-sim
track rotation is needed. The HELD-OUT generalization eval is run SEPARATELY,
single-car (multi-car can't hot-swap tracks): after/while this trains, evaluate a
checkpoint on the 8 held-out worlds with scripts/evaluate.py (proven set_world).

    GYM_DR_DEEPRACER_ENV_SRC=.../deepracer_env uv run --no-sync python experiments/oracle_asym_multicar.py
"""
import os

os.environ["GYM_DR_FEATURE_SET"] = "actor_extended"        # 11-feature actor vector

from gym_dr import (                                       # noqa: E402
    ADR, ContinuousActionSpaceConfig, EnvironmentConfig, ExperimentConfig,
    FeatureObs, OrderedSplit, Range, Sb3Trainer, TraceConfig, TrackingConfig,
    TrainingConfig, TRACKS, centerline_quadratic, clean_completion, train,
)
from gym_dr.asymmetric import AsymmetricActorCriticPolicy   # noqa: E402
from gym_dr.envs.dispatch import build_env                  # noqa: E402
from gym_dr.perception import ACTOR_FEATURES                # noqa: E402

NAME = "oracle_asym_multicar"

# Diverse train tracks (max-min over the wobble x tightness map). Gazebo's spawn
# service can't reliably create more than ~12-13 SEPARATE track instances in one
# world (n=18 timed out at racetrack_13), so we use 12 distinct tracks / 12 cars —
# 12 track + DR variations aggregated per PPO update. (The 18-set's other 6 tracks
# are spare; rotate them in via chunks later if wanted.)
_TRAIN_18 = [
    "Tokyo_Training_track", "hamption_pro", "2022_march_open", "Albert", "2022_july_open",
    "2022_summit_speedway_mini", "caecer_loop", "thunder_hill_pro", "dubai_open",
    "Virtual_May19_Train_track", "hamption_open", "2022_september_pro", "2022_march_pro",
    "H_track", "2022_august_pro", "2022_summit_speedway", "morgan_open", "jyllandsringen_pro",
]
TRAIN_WORLDS = _TRAIN_18[:12]   # 12 distinct tracks -> one car each (under the spawn cap)
EVAL_WORLDS = [   # held-out (single-car eval, separate): physical + diverse
    "reinvent_base", "Oval_track", "morgan_pro", "New_York_Track",
    "Mexico_track", "Monaco", "Canada_Training", "2022_august_open",
]
N_CARS = len(TRAIN_WORLDS)                                  # 12 (spawn cap)
TOTAL_STEPS = int(os.getenv("GYM_DR_ORACLE_STEPS", "3000000"))  # +warmup -> a bit longer
# DR WARMUP (multi-car ADR substitute): ramp every DR magnitude (the unobservable
# ±bias, feature noise, actuator noise) 0 -> full over the first ~20% of training so
# the frame-stacked policy first learns to DRIVE (near-clean, survivable episodes),
# then learns to COUNTER the perturbations as they grow. Without this, full-strength
# DR from step 0 left the policy in an unlearnable POMDP (flat 5% progress, 2-step
# episodes). multi_car reads this; app.py forwards it into the container.
DR_WARMUP_STEPS = int(os.getenv("GYM_DR_DR_WARMUP_STEPS", str(TOTAL_STEPS // 5)))
os.environ["GYM_DR_DR_WARMUP_STEPS"] = str(DR_WARMUP_STEPS)
_PHYSICAL = {"reinvent_base", "reInvent2019_track", "Oval_track"}
assert not (set(TRAIN_WORLDS) & _PHYSICAL), "physical tracks must stay held-out"
assert not (set(TRAIN_WORLDS) & set(EVAL_WORLDS)), "train/eval must be disjoint"
assert not sorted((set(TRAIN_WORLDS) | set(EVAL_WORLDS)) - set(TRACKS)), "unknown track"

# Each car drives a different train track (one per car). app.py forwards
# GYM_DR_DEMO_WORLDS; multi_car assigns names[i] to car i. WORLD_NAME (the base
# .world) must equal worlds[0], which the curriculum's first_world sets.
os.environ["GYM_DR_DEMO_WORLDS"] = ",".join(TRAIN_WORLDS)
os.environ["GYM_DR_N_CARS"] = str(N_CARS)
print(f"[oracle-mc] {N_CARS} cars / {N_CARS} train tracks (one each), {len(EVAL_WORLDS)} "
      f"held-out (single-car eval, separate); asym critic + feature_noise + frame_stack=4 "
      f"memory; DR warmup={DR_WARMUP_STEPS} steps; features={len(ACTOR_FEATURES)}")

# Heavy DR — every car samples its OWN per-episode drag / bias, and per-step
# actuator + feature noise, so one rollout aggregates 18 distinct conditions.
DR = ADR(
    feature_noise=Range(0.0, 0.30),            # Gaussian on the actor's feature vector
    steering_noise=Range(0.0, 3.0), speed_noise=Range(0.0, 0.15),  # per-step jitter
    steering_bias=15.0, speed_bias=0.5,        # per-EPISODE constant lean (miscalibration)
    drag=Range(0.5, 1.0), friction=Range(0.8, 1.5),
    random_start=True, random_direction=True,
    step=0.1, promote=0.7, demote=0.3, seed=42,
)

ENV = EnvironmentConfig(
    observation=FeatureObs(features=tuple(ACTOR_FEATURES), asymmetric_critic=True),
    action_space=ContinuousActionSpaceConfig(
        steering_low=-30.0, steering_high=30.0, speed_low=1.0, speed_high=4.0,
        normalize_actions=True),
    # Single training chunk on the base world (WORLD_NAME=TRAIN_WORLDS[0]); the 18
    # cars' actual tracks come from GYM_DR_DEMO_WORLDS. eval_worlds EMPTY -> the
    # in-sim eval callback never calls set_world (which multi-car can't do); the
    # held-out generalization is measured separately, single-car.
    curriculum=OrderedSplit(train_worlds=[TRAIN_WORLDS[0]], eval_worlds=[],
                            chunk_steps=TOTAL_STEPS, rotations=1),
    domain_randomization=DR,
    n_cars=N_CARS, reward=centerline_quadratic, eval_reward=clean_completion,
)

experiment = ExperimentConfig(
    name=NAME,
    environment=ENV,
    env_factory=build_env,
    trainer=Sb3Trainer(
        name="ppo", policy=AsymmetricActorCriticPolicy,   # actor=noised, critic=true
        # n_steps=1024 keeps the rollout batch sane at 12 cars (12 * 1024 = 12.3k).
        kwargs={"n_steps": 1024, "batch_size": 512, "learning_rate": 3.0e-4,
                "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95,
                "clip_range": 0.2, "n_epochs": 10, "target_kl": 0.08,
                "policy_kwargs": {"net_arch": {"pi": [128, 128], "vf": [128, 128]}}},
        # frame_stack=4 = OBSERVATION-LEVEL MEMORY. The per-episode actuator bias is
        # unobservable in a single frame, so a memoryless MLP can't counter it (no one
        # steering offset works for both a +15° and a -15° episode). Stacking the last
        # 4 feature vectors exposes the DRIFT signature (sliding off-center despite
        # centering commands) so the net can infer the bias and apply a standing
        # correction — online system-id WITHOUT a recurrent net. Deliberately NOT
        # RecurrentPPO: frame-stacking is just VecFrameStack obs preprocessing, so it
        # ports unchanged to a Lagrangian/safe-RL PPO on Safety-Gymnasium later (there
        # is no recurrent Lagrangian PPO to migrate to). VecFrameStack stacks the
        # asym Dict{actor,critic} per key; the policy adapts its input dim at build.
        frame_stack=4, device="cpu"),
    training=TrainingConfig(
        total_timesteps=TOTAL_STEPS, checkpoint_freq=100_000,
        checkpoint_keep_last=5, eval_freq=10 ** 12, n_eval_episodes=5,
        rtf_override=60),
    tracking=TrackingConfig(mlflow_experiment=NAME),
    trace=TraceConfig(enabled=False),
    seed=42, use_gpu=False,
)


if __name__ == "__main__":
    train(experiment)
