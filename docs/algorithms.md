# Algorithm choice

`AlgorithmConfig.name` selects the Stable-Baselines3 class. The trainer is algorithm-agnostic.

| Name | Class | Notes |
|---|---|---|
| `ppo` | `stable_baselines3.PPO` | Default; on-policy, robust, `MultiInputPolicy` works with the dict-obs DeepRacer env. |
| `a2c` | `stable_baselines3.A2C` | On-policy, similar to PPO. |
| `sac` | `stable_baselines3.SAC` | Off-policy. Requires explicit `buffer_size` in `algorithm.kwargs` (see below). |
| `td3` | `stable_baselines3.TD3` | Off-policy; same buffer caveat as SAC. |
| `ddpg` | `stable_baselines3.DDPG` | Off-policy; same buffer caveat as SAC. |

## Off-policy + dict-obs caveat

SB3 SAC / TD3 / DDPG do support `MultiInputPolicy` with dict observations, but the replay buffer keeps full image observations and OOMs at the default `buffer_size=1_000_000` on the 120×160×3 DeepRacer camera obs.

`gym_dr/algorithms.py` raises a clear error if you pick an off-policy algorithm without an explicit `buffer_size`. Reasonable starting point:

```python
algorithm=AlgorithmConfig(
    name="sac",
    policy="MultiInputPolicy",
    kwargs={"buffer_size": 50_000, "batch_size": 256, "learning_rate": 3e-4},
    device="cpu",
)
```

PPO/A2C remain the recommended path for the standard DeepRacer setup.

## Discrete vs continuous action spaces

The action-space config drives both what gets written to `model_metadata.json` and which SB3 policy is appropriate. For a discrete action space, you would typically use a discrete-action algorithm (e.g. PPO with a `MultiInputPolicy` on a categorical action head, or DQN). This is straightforward to wire up by extending the algorithm registry in `gym_dr/algorithms.py` and choosing a compatible policy in the config.
