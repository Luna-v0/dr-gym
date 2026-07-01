"""Oracle-feature PPO v2 — feature-noise robustness with an ASYMMETRIC value net.

Builds on the oracle (state-based feature policy, the camera teacher / upper bound)
with two maintainer-requested additions:

  1. **Feature-vector noise (DR)** — `DomainRandomization.feature_noise` adds Gaussian
     noise to the ACTOR's feature vector each step, under domain randomization so it's
     variable (here an ADR `Range` that grows with held-out clean-completion). Tests how
     robust the policy is to noisy/imperfect features — i.e. how much degradation a
     real camera->features extractor can have before driving suffers.
  2. **Asymmetric value network** — the critic receives the **true** (un-noised) feature
     vector while the actor sees the noised one (`FeatureObs.asymmetric_critic=True` ->
     Dict obs `{actor:noised, critic:true}` + `AsymmetricActorCriticPolicy`). The value
     function learns from a clean, low-variance signal; only the deployable actor must
     cope with noise.

More tracks + a better-distributed split than the first oracle: the 18 train / 8 held-out
tracks were chosen by **max-min (k-center) selection over the wobble x tightness geometry
map** (`scripts/track_geometry.py`), so the training set spans flowing<->zig-zag and
open<->hairpin instead of clustering. Physical `reinvent_base` + `Oval_track` stay held-out
(sim2real). Classic PPO, no Lagrangian.

    GYM_DR_DEEPRACER_ENV_SRC=.../deepracer_env uv run --no-sync python experiments/oracle_asym_robust.py
"""
import os

os.environ["GYM_DR_FEATURE_SET"] = "actor_extended"   # 11-feature actor vector (host+container)

from gym_dr import (                                       # noqa: E402
    ACL, ADR, ContinuousActionSpaceConfig, EnvironmentConfig, ExperimentConfig,
    FeatureObs, Range, Sb3Trainer, TraceConfig, TrackingConfig, TrainingConfig,
    TRACKS, centerline_quadratic, clean_completion, OfftrackRate,
)
from gym_dr.asymmetric import AsymmetricActorCriticPolicy            # noqa: E402
from gym_dr.envs.dispatch import build_env                          # noqa: E402
from gym_dr.perception import ACTOR_FEATURES                        # noqa: E402

NAME = "oracle_asym_robust"

# Diverse spanning split (max-min over the wobble x tightness map; see module docstring).
TRAIN_WORLDS = [
    "Tokyo_Training_track", "hamption_pro", "2022_march_open", "Albert", "2022_july_open",
    "2022_summit_speedway_mini", "caecer_loop", "thunder_hill_pro", "dubai_open",
    "Virtual_May19_Train_track", "hamption_open", "2022_september_pro", "2022_march_pro",
    "H_track", "2022_august_pro", "2022_summit_speedway", "morgan_open", "jyllandsringen_pro",
]
EVAL_WORLDS = [
    "reinvent_base", "Oval_track",                       # physical / sim2real held-out
    "morgan_pro", "New_York_Track", "Mexico_track", "Monaco", "Canada_Training",
    "2022_august_open",
]
CHUNK_STEPS = 50_000
N_CHUNKS = int(os.getenv("GYM_DR_DEMO_CHUNKS", "80"))      # 80 * 50k = 4M steps (early-stop ends sooner)
_PHYSICAL = {"reinvent_base", "reInvent2019_track", "Oval_track"}
assert not (set(TRAIN_WORLDS) & _PHYSICAL), "physical tracks must stay held-out"
assert not (set(TRAIN_WORLDS) & set(EVAL_WORLDS)), "train/eval must be disjoint"
assert not sorted((set(TRAIN_WORLDS) | set(EVAL_WORLDS)) - set(TRACKS)), "unknown track"
print(f"[oracle2] {len(TRAIN_WORLDS)} train / {len(EVAL_WORLDS)} held-out; "
      f"asymmetric critic + feature_noise DR; features={len(ACTOR_FEATURES)}")

# ADR: feature_noise is the robustness knob (grows 0->0.3 as held-out clean-completion
# improves); actuator/drag/friction + random start/direction are the usual sim2real DR.
DR = ADR(
    feature_noise=Range(0.0, 0.30),   # additive Gaussian std on the actor's feature vector
    steering_noise=Range(0.0, 3.0), speed_noise=Range(0.0, 0.15),
    drag=Range(0.7, 1.0), friction=Range(0.8, 1.5),
    random_start=True, random_direction=True,
    step=0.1, promote=0.7, demote=0.3, seed=42,
)

ENV = EnvironmentConfig(
    observation=FeatureObs(features=tuple(ACTOR_FEATURES), asymmetric_critic=True),
    action_space=ContinuousActionSpaceConfig(
        steering_low=-30.0, steering_high=30.0, speed_low=1.0, speed_high=4.0,
        normalize_actions=True),
    curriculum=ACL(train_worlds=TRAIN_WORLDS, eval_worlds=EVAL_WORLDS,
                   chunk_steps=CHUNK_STEPS, n_chunks=N_CHUNKS,
                   unlock_every=4, recency_weight=2.0, seed=42),
    domain_randomization=DR,
    n_cars=1, reward=centerline_quadratic, eval_reward=clean_completion,
)

experiment = ExperimentConfig(
    name=NAME,
    environment=ENV,
    env_factory=build_env,
    trainer=Sb3Trainer(
        name="ppo", policy=AsymmetricActorCriticPolicy,   # actor=noised, critic=true
        kwargs={"n_steps": 2048, "batch_size": 256, "learning_rate": 3.0e-4,
                "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95,
                "clip_range": 0.2, "n_epochs": 10, "target_kl": 0.08,
                "policy_kwargs": {"net_arch": {"pi": [128, 128], "vf": [128, 128]}}},
        frame_stack=1, device="cpu"),
    training=TrainingConfig(
        total_timesteps=CHUNK_STEPS * N_CHUNKS, checkpoint_freq=CHUNK_STEPS,
        checkpoint_keep_last=3, eval_freq=CHUNK_STEPS, n_eval_episodes=5,
        rtf_override=60, eval_path_plots=True,
        early_stop=OfftrackRate(max_offtrack_rate=0.10, patience=4)),
    tracking=TrackingConfig(mlflow_experiment=NAME),
    trace=TraceConfig(enabled=True),
    seed=42, use_gpu=False,
)


if __name__ == "__main__":
    from gym_dr import train
    train(experiment)
