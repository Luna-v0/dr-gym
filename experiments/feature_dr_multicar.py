"""Multi-car FEATURE-obs PPO with domain randomization + per-car track diversity.

The multi-car payoff the maintainer wanted: N cars in ONE Gazebo (the VecEnv), so a
SINGLE process carries central DR + curriculum/track diversity + decorrelated parallel
rollouts — which separate containers can't do cleanly. Now possible because:
  * feature obs uses an EMPTY sensor list (no camera to render/block) — multi_car
    scales (n=4 ~107 steps/s vs n=1 ~46), and
  * DR (random_start/random_direction + actuator noise) is wired into multi_car
    (gym_dr/envs/multi_car.py), plus the zero-quaternion guard (deepracer-env).

Track diversity = each car on its OWN track instance (GYM_DR_DEMO_WORLDS, one per
car), so the N cars sample N different tracks every step — the generalization engine,
in parallel. random_start spreads episodes across each track; actuator noise + ADR
ceilings add environmental robustness. Classic PPO (MlpPolicy), no Lagrangian.

    GYM_DR_DEEPRACER_ENV_SRC=/home/lunav0/Projects/deepracer-env/deepracer_env \
    GYM_DR_DEMO_NCARS=4 GYM_DR_DEMO_SPACING=300 \
    GYM_DR_DEMO_WORLDS=Spain_track,Monaco,Austin,arctic_pro \
      uv run --no-sync python experiments/feature_dr_multicar.py
"""
import os

# 11-feature ACTOR vector (camera-off). Set host+container so dispatch agrees.
os.environ["GYM_DR_FEATURE_SET"] = "actor_extended"

from gym_dr import (                                       # noqa: E402
    ContinuousActionSpaceConfig, DomainRandomization, EnvironmentConfig,
    ExperimentConfig, FeatureObs, FixedWorlds, Range, Sb3Trainer, TraceConfig,
    TrackingConfig, TrainingConfig, centerline_quadratic, Study,
)
from gym_dr.envs.dispatch import build_env                 # noqa: E402
from gym_dr.perception import ACTOR_FEATURES               # noqa: E402

N_CARS = int(os.getenv("GYM_DR_DEMO_NCARS") or os.getenv("GYM_DR_N_CARS") or "4")
# Per-car tracks (the parallel diversity). app.py forwards GYM_DR_DEMO_WORLDS; the
# multi_car factory assigns names[i % len] to car i. WORLD_NAME (the base .world)
# must equal worlds[0], so car 0 reuses the loaded track.
WORLDS = os.getenv("GYM_DR_DEMO_WORLDS", "Spain_track,Monaco,Austin,arctic_pro")
BASE_WORLD = WORLDS.split(",")[0]
STEPS = int(os.getenv("GYM_DR_DEMO_STEPS", "2000000"))

# Static DR (the ADR controller isn't wired to the multi-car VecEnv yet, so the noise
# is sampled at its fixed Range ceiling per episode). drag + friction are sim2real.
DR = DomainRandomization(
    steering_noise=Range(0.0, 3.0), speed_noise=Range(0.0, 0.15),
    drag=Range(0.7, 1.0), friction=Range(0.8, 1.5),
    random_start=True, random_direction=True, seed=42,
)

ENV = EnvironmentConfig(
    observation=FeatureObs(features=tuple(ACTOR_FEATURES)),  # camera-off -> no sensors
    action_space=ContinuousActionSpaceConfig(
        steering_low=-30.0, steering_high=30.0, speed_low=1.0, speed_high=4.0,
        normalize_actions=True),
    curriculum=FixedWorlds(names=[BASE_WORLD], chunk_steps=STEPS, rotations=1),
    domain_randomization=DR,
    n_cars=N_CARS, reward=centerline_quadratic,
    enable_gui=os.getenv("GYM_DR_DEMO_GUI", "0") != "0",
)

experiment = ExperimentConfig(
    name=f"feature_dr_multicar_{N_CARS}",
    environment=ENV,
    env_factory=build_env,          # (n>1, feature) -> multi_car (feature)
    trainer=Sb3Trainer(
        name="ppo", policy="MlpPolicy",
        kwargs={"n_steps": 512, "batch_size": 256, "learning_rate": 3.0e-4,
                "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95,
                "clip_range": 0.2, "n_epochs": 10, "target_kl": 0.08,
                "policy_kwargs": {"net_arch": {"pi": [128, 128], "vf": [128, 128]}}},
        frame_stack=1, device="cpu"),
    training=TrainingConfig(total_timesteps=STEPS, eval_freq=STEPS,
                            checkpoint_freq=100_000, rtf_override=60),
    tracking=TrackingConfig(mlflow_experiment="feature_dr_multicar"),
    trace=TraceConfig(enabled=False),
    seed=42, use_gpu=False,
)


if __name__ == "__main__":
    Study(experiment).run()