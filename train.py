from __future__ import annotations

import json
import os
import shutil
import signal
import time
from importlib.util import find_spec
from datetime import datetime, timezone
from pathlib import Path

from deepracer_env.environments.deepracer_env import DeepRacerEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from reward import reward_function


PROJECT_ROOT = Path("/workspace")
MODEL_METADATA_PATH = PROJECT_ROOT / "model_metadata.json"
REWARD_SOURCE_PATH = PROJECT_ROOT / "reward.py"


def get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def get_env_optional_int(name: str) -> int | None:
    value = os.getenv(name)
    return int(value) if value else None


def current_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def build_run_paths() -> dict[str, Path]:
    artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", str(PROJECT_ROOT / "artifacts")))
    run_name = os.getenv("RUN_NAME", f"deepracer_cpu_{current_timestamp()}")
    run_dir = artifacts_dir / run_name

    return {
        "artifacts_dir": artifacts_dir,
        "run_dir": run_dir,
        "checkpoints_dir": run_dir / "checkpoints",
        "tensorboard_dir": run_dir / "tensorboard",
        "export_dir": run_dir / "export_bundle",
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def prepare_export_bundle(paths: dict[str, Path], config: dict) -> None:
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    shutil.copy2(MODEL_METADATA_PATH, paths["run_dir"] / "model_metadata.json")
    shutil.copy2(MODEL_METADATA_PATH, paths["export_dir"] / "model_metadata.json")
    shutil.copy2(REWARD_SOURCE_PATH, paths["run_dir"] / "reward_function.py")
    shutil.copy2(REWARD_SOURCE_PATH, paths["export_dir"] / "reward_function.py")

    export_notes = (
        "This run directory contains the Stable-Baselines3 checkpoints and the\n"
        "DeepRacer-facing metadata/reward files.\n\n"
        "Important: the saved SB3 .zip files are not the same artifact format as\n"
        "an AWS DeepRacer console import bundle. They are suitable for resuming\n"
        "training here, but a later conversion/retraining step is still required\n"
        "before deploying through the standard physical-car workflow.\n"
    )
    (paths["export_dir"] / "README.txt").write_text(export_notes, encoding="utf-8")

    write_json(paths["run_dir"] / "run_config.json", config)
    write_json(
        paths["run_dir"] / "training_status.json",
        {
            "status": "initialized",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def update_training_status(run_dir: Path, status: str, extra: dict | None = None) -> None:
    payload = {
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    write_json(run_dir / "training_status.json", payload)


def save_model(model: PPO | None, path_without_suffix: Path) -> None:
    if model is None:
        return
    model.save(str(path_without_suffix))
    print(f"Saved model: {path_without_suffix}.zip", flush=True)


def install_signal_handlers() -> None:
    def _raise_interrupt(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGINT, _raise_interrupt)
    signal.signal(signal.SIGTERM, _raise_interrupt)


class TrainingStatusCallback(BaseCallback):
    def __init__(
        self,
        run_dir: Path,
        started_at: float,
        update_interval_steps: int,
        update_interval_seconds: int,
        max_train_seconds: int | None,
    ) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.started_at = started_at
        self.update_interval_steps = update_interval_steps
        self.update_interval_seconds = update_interval_seconds
        self.max_train_seconds = max_train_seconds
        self._last_report_step = 0
        self._last_report_time = started_at

    def _on_step(self) -> bool:
        now = time.monotonic()
        steps_since_report = self.num_timesteps - self._last_report_step
        seconds_since_report = now - self._last_report_time
        if (
            steps_since_report < self.update_interval_steps
            and seconds_since_report < self.update_interval_seconds
        ):
            return True

        elapsed_seconds = int(now - self.started_at)
        payload = {
            "timesteps_completed": self.num_timesteps,
            "elapsed_seconds": elapsed_seconds,
        }
        if self.max_train_seconds is not None:
            payload["time_limit_seconds"] = self.max_train_seconds
            payload["time_remaining_seconds"] = max(0, self.max_train_seconds - elapsed_seconds)

        update_training_status(self.run_dir, "running", payload)
        self._last_report_step = self.num_timesteps
        self._last_report_time = now
        return True


class WallClockLimitCallback(BaseCallback):
    def __init__(self, run_dir: Path, started_at: float, max_train_seconds: int) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.started_at = started_at
        self.max_train_seconds = max_train_seconds
        self.time_limit_reached = False

    def _on_step(self) -> bool:
        elapsed_seconds = int(time.monotonic() - self.started_at)
        if elapsed_seconds < self.max_train_seconds:
            return True

        self.time_limit_reached = True
        update_training_status(
            self.run_dir,
            "time_limit_reached",
            {
                "timesteps_completed": self.num_timesteps,
                "elapsed_seconds": elapsed_seconds,
                "time_limit_seconds": self.max_train_seconds,
            },
        )
        print(
            f"Wall-clock training limit reached after {elapsed_seconds}s at {self.num_timesteps} timesteps",
            flush=True,
        )
        return False


def main() -> None:
    install_signal_handlers()

    total_timesteps = get_env_int("TOTAL_TIMESTEPS", 500_000)
    checkpoint_freq = get_env_int("CHECKPOINT_FREQ", 1_000)
    n_steps = get_env_int("N_STEPS", 256)
    batch_size = get_env_int("BATCH_SIZE", 64)
    learning_rate = get_env_float("LEARNING_RATE", 3e-4)
    ent_coef = get_env_float("ENT_COEF", 0.01)
    device = os.getenv("SB3_DEVICE", "cpu")
    resume_from = os.getenv("RESUME_FROM")
    max_train_seconds = get_env_optional_int("MAX_TRAIN_SECONDS")
    status_update_steps = get_env_int("STATUS_UPDATE_STEPS", 1_000)
    status_update_seconds = get_env_int("STATUS_UPDATE_SECONDS", 30)

    paths = build_run_paths()
    config = {
        "run_name": paths["run_dir"].name,
        "world_name": os.getenv("WORLD_NAME", "reinvent_base"),
        "total_timesteps": total_timesteps,
        "checkpoint_freq": checkpoint_freq,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "device": device,
        "resume_from": resume_from,
        "rtf_override": os.getenv("RTF_OVERRIDE"),
        "max_train_seconds": max_train_seconds,
        "status_update_steps": status_update_steps,
        "status_update_seconds": status_update_seconds,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    prepare_export_bundle(paths, config)

    print(f"Artifacts directory: {paths['run_dir']}", flush=True)
    print(f"TensorBoard directory: {paths['tensorboard_dir']}", flush=True)

    tensorboard_log = str(paths["tensorboard_dir"])
    if find_spec("tensorboard") is None:
        tensorboard_log = None
        print("tensorboard is not installed; disabling tensorboard logging", flush=True)

    env = DeepRacerEnv(reward_fn=reward_function)
    model: PPO | None = None
    started_at = time.monotonic()
    wall_clock_callback: WallClockLimitCallback | None = None

    try:
        if resume_from:
            print(f"Resuming model from: {resume_from}", flush=True)
            model = PPO.load(
                resume_from,
                env=env,
                device=device,
                tensorboard_log=tensorboard_log,
            )
        else:
            model = PPO(
                policy="MultiInputPolicy",
                env=env,
                verbose=1,
                n_steps=n_steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                ent_coef=ent_coef,
                tensorboard_log=tensorboard_log,
                device=device,
            )

        save_model(model, paths["run_dir"] / "initial_model")
        update_training_status(paths["run_dir"], "running")

        callbacks: list[BaseCallback] = [
            CheckpointCallback(
                save_freq=max(1, checkpoint_freq),
                save_path=str(paths["checkpoints_dir"]),
                name_prefix="ppo_checkpoint",
            ),
            TrainingStatusCallback(
                run_dir=paths["run_dir"],
                started_at=started_at,
                update_interval_steps=max(1, status_update_steps),
                update_interval_seconds=max(1, status_update_seconds),
                max_train_seconds=max_train_seconds,
            ),
        ]
        if max_train_seconds is not None:
            wall_clock_callback = WallClockLimitCallback(
                run_dir=paths["run_dir"],
                started_at=started_at,
                max_train_seconds=max_train_seconds,
            )
            callbacks.append(wall_clock_callback)

        callback = CallbackList(callbacks)

        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=not bool(resume_from),
        )

        save_model(model, paths["run_dir"] / "final_model")
        elapsed_seconds = int(time.monotonic() - started_at)
        final_status = "time_limit_reached" if wall_clock_callback and wall_clock_callback.time_limit_reached else "completed"
        update_training_status(
            paths["run_dir"],
            final_status,
            {
                "timesteps_completed": model.num_timesteps,
                "elapsed_seconds": elapsed_seconds,
                "time_limit_seconds": max_train_seconds,
            },
        )
    except KeyboardInterrupt as exc:
        save_model(model, paths["run_dir"] / "interrupted_model")
        update_training_status(
            paths["run_dir"],
            "interrupted",
            {"reason": str(exc)},
        )
        raise
    except Exception as exc:
        save_model(model, paths["run_dir"] / "crash_recovery_model")
        update_training_status(
            paths["run_dir"],
            "failed",
            {"reason": repr(exc)},
        )
        raise
    finally:
        save_model(model, paths["run_dir"] / "latest_model")
        env.close()


if __name__ == "__main__":
    main()
