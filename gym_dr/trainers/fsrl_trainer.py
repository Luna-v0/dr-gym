"""FSRL PPO-Lagrangian (PID) trainer backend — D9 (adopt FSRL ``PPOLagAgent``).

FSRL's ``PPOLagAgent`` is PPO + PID-Lagrangian in one algorithm
(`docs/reports/safe-rl-backend.md`, **validated** 2026-06-22 on Safety-Gymnasium).
This plugs it in as a `Trainer` backend so the rest of the pipeline (Docker
dispatch, MLflow, artifacts, curriculum, held-out eval) is unchanged. The
constraint **cost** comes from our graded `gym_dr/costs.py` surfaced as
``info["cost"]`` by ``CostInfoWrapper`` — the *same* contract proven on the
Safety-Gym side (a `_CostToInfo` wrapper in `scripts/validate_fsrl_safetygym.py`).

Two obs regimes, two code paths (kwargs verified against the installed FSRL,
tianshou 0.5.1):

  * **Vector obs** (Safety-Gym, or a future privileged-vector DeepRacer): the
    high-level ``PPOLagAgent`` builds MLP actor/critics — this is the *validated*
    path. Use ``fit`` with ``cnn=False``.
  * **Dict camera obs** (real DeepRacer): ``PPOLagAgent``'s MLPs would flatten the
    4×120×160 stack — wrong. So we build a Tianshou **CNN preprocess_net** shared
    by an ``ActorProb`` (steering/speed) + a reward ``Critic`` + a cost ``Critic``,
    wrap them in the ``PPOLagrangian`` policy, and drive ``OnpolicyTrainer``. This
    is ``cnn=True`` (default). The CNN mirrors ``gym_dr/networks.py``'s DeepRacer
    conv stack; the asymmetric cost-critic (privileged obs) is the W-perception
    hook (`docs/reports/perception.md`). **Frame stacking** here is Tianshou-side
    (collector ``stack_num``), not SB3 ``VecFrameStack`` — set it on the buffer.

Install (its OWN venv — Tianshou pins clash with SB3): see the validation script.
Everything heavy is lazy-imported so this module imports without FSRL present.
"""
from __future__ import annotations

from typing import Any, Sequence

from gym_dr.networks import DEFAULT_CONV, ConvSpec
from gym_dr.trainers.base import Trainer, TrainingContext, TrainResult


def _build_cnn_preprocess(obs_space, conv_layers: ConvSpec, features_dim: int, device: str):
    """A Tianshou-compatible CNN ``preprocess_net`` over an image obs.

    Tianshou's contract: ``forward(obs, state=None, info={}) -> (features, state)``
    and an ``.output_dim`` attribute. Mirrors ``gym_dr/networks.py``'s DeepRacer
    conv stack (valid padding on strided layers, ``k//2`` on stride-1), raw uint8
    in (divide by 255 here so the net is self-contained off SB3's preprocessing).
    """
    import gymnasium as gym
    import numpy as np
    import torch
    import torch.nn as nn

    # locate the single image subspace (Dict) or use the Box directly.
    if isinstance(obs_space, gym.spaces.Dict):
        image_key = next(
            k for k, s in obs_space.spaces.items()
            if isinstance(s, gym.spaces.Box) and len(s.shape) == 3
        )
        img_space = obs_space.spaces[image_key]
    else:
        image_key = None
        img_space = obs_space
    # channels-first (C,H,W) expected; deepracer-env grayscale stack is (4,120,160).
    c, h, w = img_space.shape if img_space.shape[0] <= 4 else (
        img_space.shape[2], img_space.shape[0], img_space.shape[1]
    )

    class _CNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            prev = c
            for filters, kernel, stride in conv_layers:
                padding = kernel // 2 if stride == 1 else 0
                layers += [nn.Conv2d(prev, filters, kernel, stride, padding), nn.ReLU()]
                prev = filters
            layers.append(nn.Flatten())
            self.conv = nn.Sequential(*layers)
            with torch.no_grad():
                n_flat = self.conv(torch.zeros(1, c, h, w)).shape[1]
            self.fc = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())
            self.output_dim = features_dim
            self._image_key = image_key
            self._device = device

        def forward(self, obs, state=None, info=None):
            if self._image_key is not None:
                obs = obs[self._image_key]
            x = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self._device)
            if x.dim() == 3:  # (C,H,W) -> (1,C,H,W)
                x = x.unsqueeze(0)
            return self.fc(self.conv(x / 255.0)), state

    return _CNN().to(device)


class FsrlTrainer(Trainer):
    """Adopt FSRL ``PPOLagrangian`` (PID-Lagrangian PPO) behind the Trainer contract."""

    def __init__(self, *, epoch: int = 100, cost_limit: float = 10.0,
                 n_train_envs: int = 1, hidden_sizes: Sequence[int] = (256, 256),
                 lr: float = 5e-4, target_kl: float = 0.02, gamma: float = 0.99,
                 seed: int = 10, conv_layers: ConvSpec = DEFAULT_CONV,
                 features_dim: int = 512, cnn: bool = True,
                 step_per_epoch: int = 10000, episode_per_collect: int = 20,
                 device: str | None = None) -> None:
        self.epoch = epoch
        self.cost_limit = cost_limit          # the CMDP budget d (set empirically — dr/ep_mean_cost)
        self.n_train_envs = n_train_envs       # 1 = one sim/container; raise only with N-cars/parallel sim
        self.hidden_sizes = tuple(hidden_sizes)
        self.lr = lr
        self.target_kl = target_kl
        self.gamma = gamma
        self.seed = seed
        self.conv_layers = conv_layers
        self.features_dim = features_dim
        self.cnn = cnn                         # True = Dict camera obs; False = vector obs (validated path)
        self.step_per_epoch = step_per_epoch
        self.episode_per_collect = episode_per_collect
        self.device = device

    def fit(self, env: Any, ctx: TrainingContext) -> TrainResult:
        import torch
        from tianshou.env import DummyVectorEnv
        from fsrl.utils import TensorboardLogger

        from gym_dr.envs.wrappers import CostInfoWrapper

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Surface the graded cost as info["cost"]; reuse the shared metrics tap.
        def make_env():
            return CostInfoWrapper(env, ctx.metrics_state)

        train_envs = DummyVectorEnv([make_env for _ in range(max(1, self.n_train_envs))])
        test_envs = DummyVectorEnv([make_env])
        logger = TensorboardLogger(str(ctx.run_dir / "tensorboard"), name="fsrl_ppolag")

        if not self.cnn:
            # ---- validated vector-obs path: high-level agent (MLP nets) --------
            from fsrl.agent import PPOLagAgent

            agent = PPOLagAgent(
                make_env(), logger, cost_limit=self.cost_limit, device=device,
                seed=self.seed, lr=self.lr, hidden_sizes=self.hidden_sizes,
                target_kl=self.target_kl, gamma=self.gamma,
            )
            try:
                agent.learn(train_envs, test_envs, epoch=self.epoch,
                            step_per_epoch=self.step_per_epoch,
                            episode_per_collect=self.episode_per_collect)
            finally:
                self._persist(ctx, agent.policy)
            return self._result(extra={"path": "vector"})

        # ---- DeepRacer Dict-camera path: custom CNN actor + reward/cost critics ----
        from tianshou.utils.net.continuous import ActorProb, Critic
        from torch.distributions import Independent, Normal
        from fsrl.policy import PPOLagrangian
        from fsrl.trainer import OnpolicyTrainer

        sample_env = make_env()
        obs_space = sample_env.observation_space
        act_space = sample_env.action_space
        act_shape = act_space.shape or (1,)
        max_action = float(act_space.high[0]) if hasattr(act_space, "high") else 1.0

        # one shared CNN front-end is cheapest; AWS uses separate towers, so we
        # build one preprocess per head (actor / reward-critic / cost-critic) to
        # match DeepRacerCNN's `share_features_extractor=False` discipline.
        def cnn():
            return _build_cnn_preprocess(obs_space, self.conv_layers, self.features_dim, device)

        actor = ActorProb(cnn(), act_shape, hidden_sizes=self.hidden_sizes,
                          max_action=max_action, device=device,
                          unbounded=True, conditioned_sigma=True).to(device)
        reward_critic = Critic(cnn(), hidden_sizes=self.hidden_sizes, device=device).to(device)
        cost_critic = Critic(cnn(), hidden_sizes=self.hidden_sizes, device=device).to(device)
        optim = torch.optim.Adam(
            list(actor.parameters()) + list(reward_critic.parameters())
            + list(cost_critic.parameters()), lr=self.lr,
        )

        def dist_fn(*logits):
            return Independent(Normal(*logits), 1)

        policy = PPOLagrangian(
            actor=actor, critics=[reward_critic, cost_critic], optim=optim,
            dist_fn=dist_fn, logger=logger, target_kl=self.target_kl,
            cost_limit=self.cost_limit, gamma=self.gamma,
            observation_space=obs_space, action_space=act_space,
        )

        # Tianshou-side frame stacking lives on the collector buffer (stack_num),
        # NOT SB3 VecFrameStack — the raw deepracer-env emits a single frame.
        from tianshou.data import Collector, VectorReplayBuffer

        buffer = VectorReplayBuffer(
            total_size=self.step_per_epoch, buffer_num=len(train_envs), stack_num=4,
        )
        train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
        test_collector = Collector(policy, test_envs)

        trainer = OnpolicyTrainer(
            policy=policy, train_collector=train_collector,
            test_collector=test_collector, max_epoch=self.epoch,
            step_per_epoch=self.step_per_epoch,
            episode_per_collect=self.episode_per_collect, logger=logger,
            cost_limit=self.cost_limit,
        )
        try:
            for _ in trainer:
                pass
        finally:
            self._persist(ctx, policy)
        return self._result(extra={"path": "cnn-camera"})

    # ------------------------------------------------------------------ #
    def _persist(self, ctx: TrainingContext, policy) -> None:
        import torch

        def _save(p):
            torch.save(policy.state_dict(), p)

        ctx.save_model(_save, name="latest_model")
        ctx.save_model(_save, name="final_model")

    def _result(self, *, extra: dict) -> TrainResult:
        return TrainResult(
            final_eval_reward=float("nan"),
            extra={"backend": "fsrl-ppolag", "epoch": self.epoch,
                   "cost_limit": self.cost_limit, **extra},
        )
