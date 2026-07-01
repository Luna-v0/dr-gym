"""Visual (VNC) demo for the composable multi-car + camera-off work.

Configurable via env vars so the same file serves every stage of the visual
validation (connect a VNC client to vnc://localhost:5900):

    GYM_DR_DEMO_NCARS    number of cars (default 1; >1 needs MC-3)
    GYM_DR_DEMO_CAMERA   "1"/"0" — camera obs vs feature obs (default 1)
    GYM_DR_DEMO_WORLD    track name (default Spain_track)

Run (with the deepracer-env overlay so the camera-off toggle / sim edits apply):
    GYM_DR_DEEPRACER_ENV_SRC=/home/lunav0/Projects/deepracer-env/deepracer_env \
      uv run --no-sync python experiments/multicar_demo.py
"""
import os

from gym_dr import (
    ContinuousActionSpaceConfig,
    ExperimentConfig,
    Sb3Trainer,
    TraceConfig,
    TrackingConfig,
    TrainingConfig,
    WorldsConfig,
    centerline_quadratic,
    Study,
)
from gym_dr.networks import DeepRacerCNN

# The container re-runs this script to rebuild the experiment, so the choices
# must round-trip: the host reads the GYM_DR_DEMO_* vars, and the orchestrator
# propagates GYM_DR_N_CARS / GYM_DR_CAMERAS into the container (gym_dr/app.py).
# Read the demo var first (host), else the propagated var (container).
N_CARS = int(os.getenv("GYM_DR_DEMO_NCARS") or os.getenv("GYM_DR_N_CARS") or "1")
# camera off iff GYM_DR_DEMO_CAMERA=0 (host) or GYM_DR_CAMERAS="" (container).
CAMERA = os.getenv("GYM_DR_DEMO_CAMERA", "1") != "0" and os.getenv("GYM_DR_CAMERAS", "x") != ""
WORLD = os.getenv("GYM_DR_DEMO_WORLD", "Spain_track")
STEPS = int(os.getenv("GYM_DR_DEMO_STEPS", "50000"))  # raise for a long VNC watch
RTF = int(os.getenv("GYM_DR_DEMO_RTF", "20"))         # Gazebo real-time-factor hint
# policy device: default cuda for camera (CNN), cpu for feature obs (small MLP);
# override with GYM_DR_DEMO_DEVICE for the throughput benchmark's NN-device axis.
DEVICE = os.getenv("GYM_DR_DEMO_DEVICE", "cuda" if CAMERA else "cpu")

# Feature obs is low-dim — no CNN needed; camera obs uses the DeepRacerCNN.
policy_kwargs = (
    {
        "share_features_extractor": False,
        "normalize_images": False,
        "features_extractor_class": DeepRacerCNN,
        "features_extractor_kwargs": {
            "conv_layers": [[32, 8, 4], [64, 4, 2], [64, 3, 1]], "features_dim": 512},
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
    }
    if CAMERA
    else {"net_arch": {"pi": [128, 128], "vf": [128, 128]}}
)

experiment = ExperimentConfig(
    name=f"multicar_demo_{N_CARS}car_{'cam' if CAMERA else 'feat'}",
    n_cars=N_CARS,
    camera_obs=CAMERA,
    enable_gui=os.getenv("GYM_DR_DEMO_GUI", "1") != "0",   # VNC; set 0 for headless benchmark
    reward=centerline_quadratic,
    trainer=Sb3Trainer(
        name="ppo", policy="MultiInputPolicy" if CAMERA else "MlpPolicy",
        kwargs={"n_steps": 512, "batch_size": 128, "learning_rate": 3e-4,
                "ent_coef": 0.01, "policy_kwargs": policy_kwargs},
        frame_stack=4 if CAMERA else 1, device=DEVICE,
    ),
    action_space=ContinuousActionSpaceConfig(
        steering_low=-30.0, steering_high=30.0, speed_low=1.0, speed_high=4.0,
        normalize_actions=True),
    worlds=WorldsConfig(names=[WORLD], chunk_steps=50_000, rotations=1),
    training=TrainingConfig(total_timesteps=STEPS, eval_freq=STEPS,
                            checkpoint_freq=50_000, rtf_override=RTF),
    tracking=TrackingConfig(mlflow_experiment="multicar_demo"),
    trace=TraceConfig(enabled=False),
    seed=42, use_gpu=(DEVICE == "cuda"),
)


if __name__ == "__main__":
    print(f"[demo] n_cars={N_CARS} camera_obs={CAMERA} world={WORLD} — "
          f"connect VNC to vnc://localhost:5900")
    Study(experiment).run()