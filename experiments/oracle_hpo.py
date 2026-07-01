"""Optuna HPO for the asymmetric-critic feature oracle.

Searches the PPO hyperparameters + the OBSERVATION-MEMORY depth (``frame_stack``) +
network width + the feature-noise DR level that best learn the oracle's robust
state-based driving policy. The winning config is then transplanted into the
multi-car production run (``experiments/oracle_asym_multicar.py``): every knob
searched here is config-driven (``trainer.*`` / ``environment.domain_randomization.*``)
and so applies unchanged to the 12-car DR-aggregated run.

WHY SINGLE-CAR FOR THE SEARCH (not the multi-car oracle):
  The multi-car oracle can't run the in-loop held-out eval (it has no in-process
  ``set_world``), so it produces no clean generalisation objective for Optuna. The
  SINGLE-car asym oracle DOES (ACL curriculum hot-swaps the held-out worlds each
  chunk via ``set_world``), so each trial returns a real held-out clean-completion
  score — the right HPO objective. It's also cheaper per trial. The PPO/arch/
  frame-stack/DR knobs transfer directly to the multi-car run.

Same study as the oracle: actor sees the NOISED 11-feature vector, the asymmetric
critic sees the TRUE one (``AsymmetricActorCriticPolicy``); feature_noise +
actuator/drag/friction DR + a MILD per-episode steering bias (so ``frame_stack``
memory has an unobservable per-episode quantity to infer — the thing it buys the
production run, where the bias is strong).

Host (spawns ``n_parallel`` worker containers sharing one SQLite Optuna study):

    GYM_DR_DEEPRACER_ENV_SRC=.../deepracer_env uv run --no-sync python experiments/oracle_hpo.py

Knobs (env): GYM_DR_HPO_TRIALS (default 40), GYM_DR_HPO_PARALLEL (default 2),
GYM_DR_HPO_CHUNK (default 40000), GYM_DR_HPO_NCHUNKS (default 6 -> 240k steps/trial;
short on purpose — rank configs by early held-out learning, Optuna prunes the laggards).
Each single-car trial is multi-hour; run with n_parallel>1 on a many-core box.
"""
import os

os.environ["GYM_DR_FEATURE_SET"] = "actor_extended"        # 11-feature actor vector (host+container)

from gym_dr import (                                       # noqa: E402
    ACL, ADR, ContinuousActionSpaceConfig, EnvironmentConfig, ExperimentConfig,
    FeatureObs, Range, Sb3Trainer, TraceConfig, TrackingConfig, TrainingConfig,
    TRACKS, centerline_quadratic, clean_completion, Study, OfftrackRate,
)
from gym_dr.asymmetric import (                             # noqa: E402
    AsymmetricActorCriticPolicy, asymmetric_recurrent_policy)
from gym_dr.envs.dispatch import build_env                  # noqa: E402
from gym_dr.perception import ACTOR_FEATURES                # noqa: E402

# ARCHITECTURE-ROBUSTNESS study (not a pure oracle): compares a memoryless MLP vs an
# LSTM (sb3-contrib RecurrentPPO) — both asymmetric (actor sees NOISED features, critic
# sees CLEAN) — on a task with an unobservable per-episode steering bias. NO frame
# stacking on either (it's a hack; the LSTM IS the memory, the MLP is the control).
NAME = "arch_robust_hpo"

# Same spanning split as the oracle (max-min over the wobble x tightness map);
# the held-out set drives the per-chunk generalisation objective. Physical
# reinvent_base + Oval stay held-out (sim2real).
TRAIN_WORLDS = [
    "Tokyo_Training_track", "hamption_pro", "2022_march_open", "Albert", "2022_july_open",
    "2022_summit_speedway_mini", "caecer_loop", "thunder_hill_pro", "dubai_open",
    "Virtual_May19_Train_track", "hamption_open", "2022_september_pro", "2022_march_pro",
    "H_track", "2022_august_pro", "2022_summit_speedway", "morgan_open", "jyllandsringen_pro",
]
EVAL_WORLDS = [
    "reinvent_base", "Oval_track", "morgan_pro", "New_York_Track",
    "Mexico_track", "Monaco", "Canada_Training", "2022_august_open",
]
CHUNK_STEPS = int(os.getenv("GYM_DR_HPO_CHUNK", "40000"))
N_CHUNKS = int(os.getenv("GYM_DR_HPO_NCHUNKS", "6"))       # 6 * 40k = 240k steps / trial
_PHYSICAL = {"reinvent_base", "reInvent2019_track", "Oval_track"}
assert not (set(TRAIN_WORLDS) & _PHYSICAL), "physical tracks must stay held-out"
assert not (set(TRAIN_WORLDS) & set(EVAL_WORLDS)), "train/eval must be disjoint"
assert not sorted((set(TRAIN_WORLDS) | set(EVAL_WORLDS)) - set(TRACKS)), "unknown track"

# THE TASK (option A): a FIXED per-episode unobservable steering bias (±BIAS deg) — a
# hidden latent only MEMORY can infer (POMDP for a memoryless MLP) — plus per-step
# feature noise. Same task for every trial/arch so the MLP-vs-LSTM comparison is fair.
# GYM_DR_HPO_BIAS sets the magnitude (deg); the search tunes only feature_noise's ceiling.
BIAS = float(os.getenv("GYM_DR_HPO_BIAS", "10.0"))
DR = ADR(
    feature_noise=Range(0.0, 0.20),   # per-step perception noise (ceiling tuned per trial)
    steering_noise=Range(0.0, 3.0), speed_noise=Range(0.0, 0.15),
    steering_bias=BIAS, speed_bias=0.5,   # the hidden per-episode latent (fixed across trials)
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
                   unlock_every=2, recency_weight=2.0, seed=42),
    domain_randomization=DR,
    n_cars=1, reward=centerline_quadratic, eval_reward=clean_completion,
)

base = ExperimentConfig.from_environment(ENV,
    name=NAME,
    env_factory=build_env,
    trainer=Sb3Trainer(
        name="ppo", policy=AsymmetricActorCriticPolicy,   # default = the MLP arm
        kwargs={"n_steps": 2048, "batch_size": 256, "learning_rate": 3.0e-4,
                "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95,
                "clip_range": 0.2, "n_epochs": 10, "target_kl": 0.08,
                "policy_kwargs": {"net_arch": {"pi": [128, 128], "vf": [128, 128]}}},
        frame_stack=1, device="cpu"),   # NO stacking — search_space overrides per arch
    training=TrainingConfig(
        total_timesteps=CHUNK_STEPS * N_CHUNKS, checkpoint_freq=CHUNK_STEPS,
        checkpoint_keep_last=1, eval_freq=CHUNK_STEPS, n_eval_episodes=5,
        rtf_override=60, eval_path_plots=True,   # render held-out trajectory overlays to TB IMAGES
        # mild early-stop so a clearly-solved trial frees the worker for the next
        early_stop=OfftrackRate(max_offtrack_rate=0.10, patience=3)),
    tracking=TrackingConfig(mlflow_experiment=NAME),
    trace=TraceConfig(enabled=False),
    seed=42, use_gpu=False,
)

# Host-side metadata pre-gen reads the action space off `experiment`.
experiment = base


def search_space(trial) -> dict:
    """Per-trial overrides for the MLP-vs-LSTM ARCHITECTURE comparison. Objective
    (maximised) = held-out clean-completion (eval_reward=clean_completion, evaluated
    recurrent-aware for the LSTM). NO frame stacking on either arch — the MLP is the
    memoryless control, the LSTM IS the memory. Optuna's `arch` categorical lets it
    compare both head-to-head; it'll spend most trials on the better arch (the MLP is
    expected to flat-line under the unobservable bias — that's the finding)."""
    arch = trial.suggest_categorical("arch", ["mlp", "lstm"])
    width = trial.suggest_categorical("net_width", [64, 128, 256])
    common = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "ent_coef": trial.suggest_float("ent_coef", 1e-4, 3e-2, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [1024, 2048]),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256]),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.90, 0.99),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
        "n_epochs": trial.suggest_categorical("n_epochs", [5, 10]),
        "target_kl": trial.suggest_categorical("target_kl", [0.05, 0.08, 0.15]),
    }
    net = {"pi": [width, width], "vf": [width, width]}
    ov: dict = {
        "trainer.frame_stack": 1,                                   # NO stacking, either arch
        "domain_randomization.feature_noise": Range(
            0.0, trial.suggest_float("feature_noise_high", 0.1, 0.4)),
    }
    if arch == "mlp":
        ov["trainer.name"] = "ppo"
        ov["trainer.policy"] = AsymmetricActorCriticPolicy
        ov["trainer.kwargs"] = {**common, "policy_kwargs": {"net_arch": net}}
    else:  # lstm — sb3-contrib RecurrentPPO, recurrence instead of stacking
        hidden = trial.suggest_categorical("lstm_hidden_size", [64, 128, 256])
        ov["trainer.name"] = "recurrent_ppo"
        ov["trainer.policy"] = asymmetric_recurrent_policy()
        ov["trainer.kwargs"] = {**common, "policy_kwargs": {
            "net_arch": net, "lstm_hidden_size": hidden, "enable_critic_lstm": True}}
    return ov


if __name__ == "__main__":
    Study(
        base,
        search_space,
        study_name=NAME,
        n_trials=int(os.getenv("GYM_DR_HPO_TRIALS", "40")),
        n_parallel=int(os.getenv("GYM_DR_HPO_PARALLEL", "2")),
    ).run()
