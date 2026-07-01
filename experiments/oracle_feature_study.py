"""Oracle-feature PPO study (Test 1 from docs/reports/feature-based-policy.md).

Trains a STATE-BASED policy on the ground-truth feature vector — no camera, no CNN.
It answers "are these features sufficient to drive well?" and becomes the teacher /
upper bound for the camera student (g(camera)->features). Per the maintainer's
feature decision (CNN-learnability is the gate):

  ACTOR vector = 11 features (gym_dr.perception.ACTOR_FEATURES):
    core perception (1-2 frames):  lateral_offset (how-central), heading_error,
        dist_left_edge, dist_right_edge, speed_mps (RAW m/s, NOT sim-max-normalised
        — proprioceptive + sim2real-stable), yaw_rate
    dynamics (3-4 frame stack):    long_accel (acceleration), lateral_velocity,
        edge_closing_rate
    learnable extras:              curvature_ahead (short FOV lookahead, K=3),
        nearest_object_dist
  EXCLUDED: progress_frac (perceptual aliasing — not recoverable from one frame;
            stays critic-only/privileged).

Domain randomization (maintainer-requested):
  * random_start      — every episode begins at a uniform valid track location.
  * random_direction  — random clockwise / counter-clockwise each episode.
  * drag (drag_min)   — per-episode throttle->speed factor ~U[0.7,1.0], so a given
        throttle reaches different speeds (the sim-vs-real motor/drag mismatch).
  * actuator noise + ADR ceilings (environmental robustness).
Track changing via StochasticCurriculum over the train split (held-out eval split
stays disjoint -> live generalization signal). Classic PPO (MlpPolicy), no Lagrangian.

    GYM_DR_DEEPRACER_ENV_SRC=/home/lunav0/Projects/deepracer-env/deepracer_env \
      uv run --no-sync python experiments/oracle_feature_study.py
"""
import os

# Select the 11-feature actor vector in feature_time_trial (host + container, since
# the container re-imports this module). Default elsewhere stays the validated 9.
os.environ["GYM_DR_FEATURE_SET"] = "actor_extended"

from gym_dr import (                                       # noqa: E402
    ACL, ADR, ContinuousActionSpaceConfig, EnvironmentConfig, ExperimentConfig,
    FeatureObs, Range, Sb3Trainer, TraceConfig, TrackingConfig, TrainingConfig,
    TRACKS, centerline_quadratic, clean_completion, Study,
)
from gym_dr.envs.dispatch import build_env                          # noqa: E402
from gym_dr.perception import ACTOR_FEATURES                        # noqa: E402

NAME = "oracle_feature_study_v5"   # FRESH. v2/v3 collapsed (resumed value-head mismatch);
                                   # v4 looked "fresh-healthy" but every dr/* metric was 0,
                                   # progress=0, and eval paths were EMPTY because the
                                   # EnvironmentConfig refactor made __post_init__ clobber the
                                   # metrics-wrapped reward injected by install_metrics (the env
                                   # ran the RAW reward, so record_step never fired). Fixed in
                                   # config.py (__post_init__ no longer overwrites explicit
                                   # overrides). v5 is the first run with working metrics+trace+paths.
RESUME_CKPT = None                 # train from scratch with the bounded/fast-eval config
TRAIN_WORLDS = ["Spain_track", "Monaco", "Austin", "arctic_pro", "caecer_gp"]
EVAL_WORLDS = ["Bowtie_track", "jyllandsringen_pro", "penbay_pro"]
CHUNK_STEPS = 50_000
N_CHUNKS = int(os.getenv("GYM_DR_DEMO_CHUNKS", "60"))      # 60 * 50k = 3M steps
_PHYSICAL = {"reInvent2019_track", "Oval_track", "reinvent_base"}
assert not (set(TRAIN_WORLDS) | set(EVAL_WORLDS)) & _PHYSICAL
assert not sorted((set(TRAIN_WORLDS) | set(EVAL_WORLDS)) - set(TRACKS)), "unknown track"
assert not (set(TRAIN_WORLDS) & set(EVAL_WORLDS)), "train/eval must be disjoint"
print(f"[oracle] actor features ({len(ACTOR_FEATURES)}): {ACTOR_FEATURES}")

# Domain randomization (ADR): noise knobs grow 0->high as clean-completion improves;
# drag + per-spawn friction + random start/direction are sim2real knobs.
DR = ADR(
    steering_noise=Range(0.0, 3.0), speed_noise=Range(0.0, 0.15),
    drag=Range(0.7, 1.0),            # per-episode throttle->speed factor (sim2real)
    friction=Range(0.8, 1.5),        # per-spawn wheel-mu multiplier (baseline 1.5)
    random_start=True, random_direction=True,         # random location + CW/CCW
    step=0.1, promote=0.7, demote=0.3, seed=42,
)

ENV = EnvironmentConfig(
    observation=FeatureObs(features=tuple(ACTOR_FEATURES)),   # camera-off, 11-feature
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
    env_factory=build_env,
    trainer=Sb3Trainer(
        name="ppo", policy="MlpPolicy",
        kwargs={"n_steps": 2048, "batch_size": 256, "learning_rate": 3.0e-4,
                "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95,
                "clip_range": 0.2, "n_epochs": 10, "target_kl": 0.08,
                "policy_kwargs": {"net_arch": {"pi": [128, 128], "vf": [128, 128]}}},
        frame_stack=1, device="cpu"),
    training=TrainingConfig(
        total_timesteps=CHUNK_STEPS * N_CHUNKS, checkpoint_freq=CHUNK_STEPS,
        checkpoint_keep_last=3, eval_freq=CHUNK_STEPS, n_eval_episodes=5,
        resume_from=RESUME_CKPT,                # resume the v1 policy
        rtf_override=60, eval_path_plots=True),
    tracking=TrackingConfig(mlflow_experiment=NAME),
    trace=TraceConfig(enabled=True),
    seed=42, use_gpu=False,
)


if __name__ == "__main__":
    Study(experiment).run()