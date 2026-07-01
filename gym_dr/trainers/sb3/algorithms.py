from __future__ import annotations

from typing import Any


OFF_POLICY = {"sac", "td3", "ddpg"}


def import_algos() -> dict[str, type]:
    from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

    algos = {"ppo": PPO, "sac": SAC, "td3": TD3, "a2c": A2C, "ddpg": DDPG}
    try:  # sb3-contrib is optional (only the LSTM architecture arm needs it)
        from sb3_contrib import RecurrentPPO

        algos["recurrent_ppo"] = RecurrentPPO
    except ImportError:
        pass
    return algos


def make_model(env, *, name: str, policy: str, kwargs: dict[str, Any], device: str, tensorboard_log: str | None):
    algos = import_algos()
    name = name.lower()
    if name not in algos:
        raise KeyError(f"unknown algorithm {name!r}; known: {sorted(algos)}")
    if name in OFF_POLICY and policy in {"MultiInputPolicy", "CnnPolicy"} and "buffer_size" not in kwargs:
        raise ValueError(
            f"{name.upper()} with policy={policy!r} requires an explicit `buffer_size` "
            f"in Sb3Trainer.kwargs (default 1e6 OOMs on image-dict observations; "
            f"use <= 50_000 for the DeepRacer camera obs)."
        )
    return algos[name](
        policy=policy,
        env=env,
        tensorboard_log=tensorboard_log,
        device=device,
        verbose=1,
        **kwargs,
    )


def load_model(path: str, env, *, name: str, device: str, tensorboard_log: str | None):
    algos = import_algos()
    name = name.lower()
    if name not in algos:
        raise KeyError(f"unknown algorithm {name!r}; known: {sorted(algos)}")
    return algos[name].load(path, env=env, device=device, tensorboard_log=tensorboard_log)
