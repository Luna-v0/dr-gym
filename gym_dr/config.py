"""Typed configuration dataclasses.

``ExperimentConfig`` is the single object the user composes in ``app.py``.
It carries everything ``gym_dr.train(experiment)`` needs to run a training:
which env to build, which trainer to use, which reward function, which
action space, which world(s), how long to train, and where to log.

All dataclasses are ``frozen=True`` so they hash; HPO mutates them through
``with_overrides(**flat_dotted_keys)`` which uses ``dataclasses.replace``
to return a new instance.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from gym_dr.action_space import ActionSpaceConfig, ContinuousActionSpaceConfig
from gym_dr.domain_randomization import DomainRandomization
from gym_dr.object_avoidance import ObjectAvoidanceConfig
from gym_dr.worlds import FixedWorlds, WorldStrategy

if TYPE_CHECKING:
    from gym_dr.early_stopping import EarlyStopStrategy
    from gym_dr.environment import EnvironmentConfig
    from gym_dr.trainers.base import Trainer


@dataclass(frozen=True)
class TrainingConfig:
    """Per-chunk training control.

    A *chunk* is one ``model.learn`` call: one container, one ``WORLD_NAME``.
    Multi-world runs string several chunks together — see ``WorldsConfig``.

    Fields
    ------
    - ``total_timesteps``  — how long a single chunk trains for.
    - ``checkpoint_freq``  — how often to drop a checkpoint into ``checkpoints/``.
    - ``max_train_seconds`` — wall-clock cap for the chunk.
    - ``status_update_steps`` / ``status_update_seconds`` — debounce knobs for
      ``training_status.json`` writes.
    - ``resume_from``      — checkpoint zip to resume from.
    - ``rtf_override``     — Gazebo real-time-factor hint.
    - ``eval_freq``        — how often the eval callback rolls out the policy.
    - ``n_eval_episodes``  — episodes per eval.
    - ``early_stop_*``     — stop a chunk early once the car masters the track
      (stays on it during evaluation); advances the rotation / ends the run.
    """

    total_timesteps: int = 500_000
    """Number of environment timesteps for this chunk. The multi-world host
    orchestrator overrides this per chunk to ``WorldsConfig.chunk_steps``."""

    checkpoint_freq: int = 1_000
    """Save a periodic checkpoint every N timesteps to
    ``artifacts/<chunk>/checkpoints/<prefix>_<step>_steps.zip``. Each
    checkpoint gets a sibling ``.model_metadata.json`` so any one of them is
    shippable to the physical car as-is."""

    checkpoint_keep_last: int | None = None
    """Cap on how many periodic checkpoints to retain on disk. ``None`` (default)
    keeps every checkpoint — fine for short runs, but a long run at a small
    ``checkpoint_freq`` can hoard hundreds of multi-hundred-MB zips and fill the
    disk. Set e.g. ``3`` to keep only the most recent 3 (each older checkpoint +
    its metadata sidecar is deleted after a new one is written). ``best_model``,
    ``final_model`` and ``latest_model`` live outside ``checkpoints/`` and are
    never pruned, so resuming and shipping still work."""

    max_train_seconds: int | None = None
    """Optional wall-clock limit. When reached, the chunk saves a final model
    and exits with status ``time_limit_reached``. ``None`` = no cap (train
    until ``total_timesteps``)."""

    status_update_steps: int = 1_000
    """Minimum number of timesteps between consecutive ``training_status.json``
    rewrites. Lower = more frequent writes (mild I/O cost)."""

    status_update_seconds: int = 30
    """Minimum wall-clock seconds between ``training_status.json`` rewrites.
    Combined with ``status_update_steps`` via OR (whichever triggers first)."""

    resume_from: str | None = None
    """**Container path** to a previously-saved checkpoint zip. The next
    chunk in a multi-world rotation gets this set automatically to the
    previous chunk's ``latest_model.zip``. Set explicitly to resume a brand-
    new training from a previous one."""

    rtf_override: int | None = None
    """Requested Gazebo real-time factor. Passed through as the
    ``RTF_OVERRIDE`` env var; the simapp treats it as a hint and may ignore
    high values. Typical: ``100``."""

    eval_freq: int = 5_000
    """Run the eval callback every N timesteps. Each eval calls
    ``ctx.report_eval`` which (a) logs to MLflow and (b) reports to Optuna
    for pruning if this is an HPO trial."""

    n_eval_episodes: int = 3
    """Episodes per eval rollout. Higher = lower-variance eval reward at the
    cost of wall-clock during eval."""

    eval_path_plots: bool = False
    """When ``True``, each evaluation renders the car's driven trajectory over a
    skeleton of the track and logs it to TensorBoard's *Images* tab — one
    overlay chart per eval world (all ``n_eval_episodes`` traces, colour + legend
    per episode) plus one chart per individual episode. Off by default: image
    logging is heavier than scalars and buffers each eval episode's ``(x, y)``
    path. The geometry comes straight from the env's reward params
    (``x``/``y``/``waypoints``/``track_width``) — no DeepRacerEnv change needed."""

    early_stop: "EarlyStopStrategy | None" = None
    """Interchangeable early-stopping strategy (``gym_dr.early_stopping``), or
    ``None`` (default) to train the full ``total_timesteps`` / ``chunk_steps``.

    A strategy is a frozen, HPO-sweepable object that decides — from an eval
    round's aggregate metrics (``offtrack_rate``, ``clean_completion_rate``,
    ``mean_reward``, …) — whether to end the current chunk early (advancing a
    multi-world rotation, or ending a single-track run). The historical
    track-mastery default is ``OfftrackRate(max_offtrack_rate=0.0, patience=1)``
    (stop the first eval round the car completes without leaving the track).
    Others: ``CleanCompletion(min_rate=1.0, patience=2)``,
    ``RewardThreshold(min_reward=...)``,
    ``MetricThreshold(metric=..., threshold=..., mode="max"|"min")``, and the
    ``AllOf``/``AnyOf`` combinators. The eval callback owns an
    ``EarlyStopController`` that requires the strategy's ``patience`` consecutive
    qualifying rounds and resets the streak at the start of each chunk (so
    mastering one track never pre-credits the next). Sweep e.g.
    ``training.early_stop.max_offtrack_rate`` via HPO overrides."""


@dataclass(frozen=True)
class TrackingConfig:
    """MLflow + TensorBoard settings.

    Fields
    ------
    - ``mlflow_tracking_uri`` — where MLflow stores runs.
    - ``mlflow_experiment``   — MLflow experiment name (groups runs in the UI).
    - ``tensorboard``         — enable per-run TB event writing.
    - ``tags``                — extra tags applied to every MLflow run.
    """

    mlflow_tracking_uri: str = "file:./mlruns"
    """MLflow store URI. The default is a **relative** file URI so it
    resolves consistently on both sides of the host/container boundary:

    - On the host, ``python app.py`` runs from the project dir; ``./mlruns``
      lands at ``<project_dir>/mlruns``.
    - Inside the container, the Dockerfile CMD does ``cd /workspace`` first,
      so ``./mlruns`` resolves to ``/workspace/mlruns`` — the same dir, via
      the ``-v <project_dir>/mlruns:/workspace/mlruns`` bind mount.

    Override only if you want a remote MLflow server (e.g.
    ``http://mlflow.internal:5000``)."""

    mlflow_experiment: str = "gym-dr"
    """MLflow experiment name. All chunks of a multi-world run + all HPO
    trials of a study share this experiment; use one experiment per
    project area to keep the UI tidy."""

    tensorboard: bool = True
    """When ``True``, SB3 writes per-run TB events under
    ``artifacts/<chunk>/tensorboard/``."""

    tags: dict[str, str] = field(default_factory=dict)
    """Free-form key/value tags applied to every MLflow run. Useful for
    grouping runs from the same campaign in the UI."""


@dataclass(frozen=True)
class WorldsConfig:
    """Worlds to rotate through during a single training run.

    Multi-world runs use *sequential rotation with shared policy* inside a
    single container: the trainer trains ``chunk_steps`` timesteps on the
    first world, then calls ``DeepRacerEnv.set_world`` to swap the Gazebo
    track in place (no container restart, no gzserver restart) and continues.
    The policy weights and PPO optimizer state stay in memory across swaps
    (off-policy replay buffers, if any, would be lost — PPO has none).

    Example::

        worlds = WorldsConfig(
            names=["reinvent_base", "Bowtie_track"],
            chunk_steps=20_000,
            rotations=3,
        )

    runs 6 chunks of 20k timesteps each:
    reinvent_base -> Bowtie_track -> reinvent_base -> ... -> Bowtie_track.

    For valid world names see ``.deepracer-env-upstream/tracks.txt``. Upstream
    swaps the world at runtime via ``DeepRacerEnv.set_world``, so the whole
    rotation runs in one container.
    """

    names: list[str] = field(default_factory=lambda: ["reinvent_base"])
    """Ordered list of world names. A single-element list = single-world
    training. The list order is the rotation order within each pass."""

    chunk_steps: int = 50_000
    """Timesteps to train per ``(rotation, world)`` chunk before swapping the
    track at runtime. All chunks run in one container with one persistent
    policy + Gazebo process (see the class docstring)."""

    rotations: int = 1
    """How many full passes through ``names``. With ``rotations=1`` and a
    list of 3 worlds, 3 chunks run total (3 × 1 = 3). With ``rotations=2``
    and 3 worlds, 6 chunks (3 × 2)."""

    def __post_init__(self) -> None:
        # Guard the easy mistake: `names="Oval_track"` (a bare str) instead
        # of `names=["Oval_track"]`. A str is iterable, so it would silently
        # "work" — iterating into single characters as world names. Coerce
        # it to a one-element list so the intent (one world) is honoured.
        if isinstance(self.names, str):
            object.__setattr__(self, "names", [self.names])


@dataclass(frozen=True)
class TraceConfig:
    """Per-step Tier-1 trace sink (see ``docs/trace-contract.md``).

    When ``enabled``, the metrics wrapper writes one row per env step to
    per-episode Parquet shards under ``artifacts/<chunk>/trace/steps/`` — the
    simtrace-equivalent that the analysis layer reads via
    ``gym_dr.trace.load_steps``. Off by default: a long HPO study would emit a
    shard per episode across hundreds of trials, so turn it on for the analysis
    runs you actually want to dissect.
    """

    enabled: bool = False
    """Write the per-step trace. Default ``False`` (HPO-safe)."""

    compression: str = "snappy"
    """Parquet codec passed to ``DataFrame.to_parquet``. ``snappy`` (fast,
    default), ``zstd`` (smaller), or ``None`` for uncompressed."""


def _default_env_factory():
    # The dispatcher routes on (n_cars, camera_obs); for the default
    # (1, True) it is exactly time_trial, so existing single-car configs are
    # unaffected. See gym_dr/envs/dispatch.py.
    from gym_dr.envs import build_env

    return build_env


def _default_trainer():
    from gym_dr.trainers import Sb3Trainer

    return Sb3Trainer()


def _default_reward():
    from gym_dr.rewards import center_line

    return center_line


def _default_eval_reward():
    # clean_completion is the eval-only yardstick that matches the maintainer's
    # success criterion: finish the lap WITHOUT leaving the track, at a
    # reasonable (non-minimum) speed. It replaced progress_safe as the default
    # because progress_safe is dominated by speed^2 and barely penalises
    # off-track, so it couldn't tell a 0.2%-progress policy from a 50% one (see
    # docs/reports/scope-review.md, docs/eval-protocol.md). Eval-only — never a
    # training reward. progress_safe stays importable for back-compat.
    from gym_dr.rewards import clean_completion

    return clean_completion


@dataclass(frozen=True)
class ExperimentConfig:
    """A full training experiment definition.

    Compose one of these in your ``app.py``, then call
    ``gym_dr.train(experiment)``. The orchestrator handles host-vs-container
    mode dispatch, multi-world rotation, MLflow tracking, and artifact
    layout — your code only has to declare *what* to train.

    Plug-in points
    --------------
    - ``env_factory``: swap the env. Default ``gym_dr.envs.time_trial`` builds
      a single-agent time-trial ``DeepRacerEnv`` and conditionally enables
      static-obstacle Object Avoidance when ``object_avoidance`` is set.
      To use a different upstream race type (head-to-head, F1) or a future
      env version, write a sibling factory under ``gym_dr/envs/`` and
      reference it here.
    - ``trainer``: swap the RL algorithm/library. Default
      ``gym_dr.trainers.Sb3Trainer()`` wraps SB3 PPO/SAC/TD3/A2C/DDPG. Any
      object with ``fit(env, ctx) -> TrainResult`` satisfies the protocol.
    - ``reward``: plain ``(params: dict) -> float`` callable. Receives the
      upstream DeepRacer reward params dict (see ``gym_dr/rewards.py`` for
      the key list and example functions).
    - ``action_space``: continuous bounds or a discrete action list.
    - ``worlds``: list of world names to rotate through.
    """

    name: str
    """Identifier for this experiment. Per-chunk artifact dirs are
    ``artifacts/<name>_rot<r>_<world>/``; MLflow runs use ``<name>`` as the
    parent run name."""

    env_factory: Callable[["ExperimentConfig"], Any] = field(default_factory=_default_env_factory)
    """Callable ``(experiment) -> gym.Env``. Default: ``gym_dr.envs.time_trial``.
    Replace to plug in a different race type or env version."""

    trainer: "Trainer" = field(default_factory=_default_trainer)
    """Any object with ``fit(env, ctx) -> TrainResult``. Default:
    ``gym_dr.trainers.Sb3Trainer()`` (SB3 PPO; switch to SAC/TD3/A2C/DDPG via
    ``Sb3Trainer(name="sac", ...)``)."""

    reward: Callable[[dict], float] = field(default_factory=_default_reward)
    """``(params: dict) -> float`` — the *training* reward. ``params`` is the
    upstream DeepRacer reward-params dict (``track_width``,
    ``distance_from_center``, ``progress``, ``speed``, ``all_wheels_on_track``,
    ``waypoints``, ...). See ``gym_dr/rewards.py`` for variants."""

    eval_reward: Callable[[dict], float] = field(default_factory=_default_eval_reward)
    """``(params: dict) -> float`` — the *evaluation* reward, computed in
    parallel to the training reward and logged per-episode as
    ``dr/ep_eval_reward``. Default ``clean_completion``: rewards finishing the
    lap without leaving the track, at a reasonable (non-minimum) speed —
    matching the success criterion (see ``docs/eval-protocol.md``). Invariant
    to the training reward chosen per trial, so HPO trials that sweep different
    training rewards can still be ranked fairly. Doesn't affect what the policy
    optimizes (that's ``reward``)."""

    cost: Callable[[dict], float] | None = None
    """Optional CMDP **cost** — graded *risk* of nearing a bad state
    (``gym_dr/costs.py``), logged every episode as ``dr/ep_mean_cost`` /
    ``dr/ep_max_cost`` so even *unconstrained* PPO characterises the cost level
    (to pick a constraint budget empirically). ``None`` ⇒ monitored with
    ``cost_near_edge``. A constrained (safe-RL) trainer keeps E[discounted cost]
    ≤ a budget."""

    action_space: ActionSpaceConfig = field(default_factory=ContinuousActionSpaceConfig)
    """Continuous bounds (steering and speed ranges) or a discrete action
    list. Controls both the env's gym action space and what gets written to
    ``model_metadata.json`` (the DeepRacer-compatible sidecar)."""

    worlds: WorldsConfig = field(default_factory=WorldsConfig)
    """List of worlds to rotate through. Single-world runs use a list of one
    (the default: ``["reinvent_base"]``). Ignored when ``world_strategy`` is
    set — see :meth:`effective_strategy`."""

    world_strategy: WorldStrategy | None = None
    """Optional world-scheduling strategy (``gym_dr.worlds``). When set it
    *supersedes* ``worlds``, deciding both the training world order and the
    (possibly held-out) evaluation worlds. ``None`` (default) falls back to a
    :class:`~gym_dr.worlds.FixedWorlds` built from ``worlds`` — so
    existing configs behave exactly as before. Use
    :class:`~gym_dr.worlds.OrderedSplit` to train on one ordered list and
    evaluate on another."""

    domain_randomization: DomainRandomization | None = None
    """Opt-in domain randomization — ``DomainRandomization`` / ``ADR`` with
    ``Range``/``Choice`` knobs (``gym_dr.domain_randomization``). Default None.
    Prefer authoring via ``environment=EnvironmentConfig(domain_randomization=...)``."""

    object_avoidance: ObjectAvoidanceConfig | None = None
    """Optional static-obstacle Object Avoidance settings. ``None`` (the
    default) keeps training pure time-trial. Set to an
    :class:`ObjectAvoidanceConfig` instance to spawn obstacles each
    episode — the ``time_trial`` env factory will translate it to upstream
    and forward to ``DeepRacerEnv(object_avoidance=...)``. Pair with
    :func:`gym_dr.rewards.object_avoidance_aware` (or your own reward) to
    consume the resulting ``is_crashed`` / ``closest_objects`` reward
    params."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    """Per-chunk training control: timesteps, eval/checkpoint frequencies,
    wall-clock cap, resume target. See ``TrainingConfig`` for each field."""

    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    """MLflow + TensorBoard settings."""

    trace: TraceConfig = field(default_factory=TraceConfig)
    """Per-step Tier-1 trace sink. Off by default; enable to dump the
    simtrace-equivalent Parquet shards for offline analysis (see
    ``docs/trace-contract.md`` and ``gym_dr/trace.py``)."""

    enable_gui: bool = False
    """When ``True``, the simapp boots Gazebo with its GUI/VNC enabled. The
    host orchestrator passes ``ENABLE_GUI=True`` and publishes VNC port
    5900 (or ``5900 + worker_idx`` for parallel HPO workers). Connect any
    VNC client to ``localhost:5900`` to watch the car drive in real time.
    Adds Gazebo rendering overhead — leave off for long unattended runs."""

    use_gpu: bool = False
    """When ``True`` the host orchestrator passes ``--gpus all`` to
    ``docker run`` so the container can see host GPUs. You separately need
    a CUDA-capable image (``./bootstrap.sh -a gpu``) and a CUDA-aware
    trainer config (``Sb3Trainer(device="cuda")``) — flipping this flag
    alone is not enough. ``Sb3Trainer.fit`` checks ``torch.cuda.is_available``
    at start and fails fast with a clear message if the pieces don't line
    up, so misconfigurations crash on the host instead of mid-rollout."""

    n_cars: int = 1
    """Number of racecars to spawn in a **single** Gazebo world (multi-agent).
    ``1`` (default) = the classic single-car env. ``> 1`` makes the env factory
    build a ``MultiAgentDeepRacerEnv`` presented to SB3 as a ``VecEnv`` with
    ``num_envs = n_cars`` — one physics step advances all cars, so the per-step
    sim cost amortizes and PPO gets decorrelated parallel samples. All cars share
    the world's track (one world = one track). See ``gym_dr/envs/multi_car.py``
    and ``docs/reports/multi-car.md``."""

    camera_obs: bool = True
    """When ``True`` (default) the policy observes the grayscale camera (vision).
    When ``False`` the policy observes a low-dim **feature vector** built from the
    privileged ``reward_params`` (``gym_dr.perception.all_targets``) and the
    camera is **not rendered** at all (much cheaper per step). Composable with
    ``n_cars``: single/multi × camera/feature. The reward operates on
    ``reward_params`` either way, so a reward/policy transfers across the two."""

    seed: int | None = None
    """Random seed plumbed everywhere we control:

    - Python ``random``, NumPy, and ``torch.manual_seed`` /
      ``torch.cuda.manual_seed_all`` are set in the orchestrator before
      the env is built.
    - SB3 receives ``seed=`` as a kwarg; internally it re-seeds the same
      three RNGs and forwards to the first ``env.reset(seed=...)`` for
      policy + rollout determinism.
    - Optuna's TPE sampler is seeded as ``base.seed + worker_idx`` so
      parallel workers don't sample in lockstep.

    ``None`` = nondeterministic. Note Gazebo physics is not deterministic
    even at a fixed seed — expect some run-to-run variance from the
    simulator regardless."""

    @classmethod
    def from_environment(
        cls,
        environment: "EnvironmentConfig",
        *,
        name: str,
        env_factory: "Callable[[ExperimentConfig], Any] | None" = None,
        trainer: "Trainer | None" = None,
        training: "TrainingConfig | None" = None,
        tracking: "TrackingConfig | None" = None,
        trace: "TraceConfig | None" = None,
        seed: "int | None" = None,
        use_gpu: bool = False,
    ) -> "ExperimentConfig":
        """Build an ``ExperimentConfig`` from a typed :class:`EnvironmentConfig`.

        **The single authoring path.** The environment's fields (observation →
        ``camera_obs``, ``action_space``, curriculum → ``world_strategy``,
        ``domain_randomization``, ``object_avoidance``, ``safe_rl`` → ``cost``,
        ``n_cars``, ``reward``, ``eval_reward``, ``enable_gui``) are read into the
        flat config **once, here** — not re-derived on every ``dataclasses.replace``.
        That eliminates the old dual-source-of-truth: because nothing re-unpacks, a
        ``with_overrides`` (e.g. the metrics-wrapped reward ``install_metrics``
        injects) can never be silently undone. Pass the training / tracking / trainer
        concerns as keyword arguments.
        """
        import os

        from gym_dr.environment import FeatureObs

        # Feature-obs vector selection (dispatch reads GYM_DR_FEATURE_SET) +
        # asymmetric-critic Dict obs (GYM_DR_ASYM_CRITIC). Env vars because the
        # container RE-IMPORTS the experiment module, re-running this builder at
        # module load, so the flags ride along without explicit forwarding.
        if isinstance(environment.observation, FeatureObs):
            from gym_dr.perception import ACTOR_FEATURES

            if tuple(environment.observation.features) == tuple(ACTOR_FEATURES):
                os.environ["GYM_DR_FEATURE_SET"] = "actor_extended"
            if environment.observation.asymmetric_critic:
                os.environ["GYM_DR_ASYM_CRITIC"] = "1"

        kwargs: "dict[str, Any]" = dict(
            name=name,
            action_space=environment.action_space,
            world_strategy=environment.curriculum,
            domain_randomization=environment.domain_randomization,
            object_avoidance=environment.object_avoidance,
            n_cars=environment.n_cars,
            reward=environment.reward,
            eval_reward=environment.eval_reward,
            enable_gui=environment.enable_gui,
            camera_obs=environment.camera_obs,
            use_gpu=use_gpu,
        )
        if environment.safe_rl is not None:
            kwargs["cost"] = environment.safe_rl.cost
        if env_factory is not None:
            kwargs["env_factory"] = env_factory
        if trainer is not None:
            kwargs["trainer"] = trainer
        if training is not None:
            kwargs["training"] = training
        if tracking is not None:
            kwargs["tracking"] = tracking
        if trace is not None:
            kwargs["trace"] = trace
        if seed is not None:
            kwargs["seed"] = seed
        return cls(**kwargs)

    def effective_strategy(self) -> WorldStrategy:
        """The world schedule actually used for this run.

        Returns ``world_strategy`` when set, else a
        :class:`~gym_dr.worlds.FixedWorlds` derived from ``worlds`` —
        so the strategy pattern is the single source of truth for world order
        and evaluation worlds, whether or not the user opted into a custom
        strategy.
        """
        if self.world_strategy is not None:
            return self.world_strategy
        return FixedWorlds(
            names=list(self.worlds.names),
            chunk_steps=self.worlds.chunk_steps,
            rotations=self.worlds.rotations,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON dump / MLflow logging.

        Callables serialize as ``module.qualname`` strings. The trainer is
        special-cased: if it's a dataclass we ``asdict()`` it so its kwargs
        survive the round-trip.
        """
        return {
            "name": self.name,
            "env_factory": _describe_callable(self.env_factory),
            "trainer": _describe(self.trainer),
            "reward": _describe_callable(self.reward),
            "eval_reward": _describe_callable(self.eval_reward),
            "cost": _describe_callable(self.cost) if self.cost is not None else None,
            "action_space": {
                **dataclasses.asdict(self.action_space),
                "action_space_type": self.action_space.action_space_type,
            },
            "worlds": dataclasses.asdict(self.worlds),
            "n_cars": self.n_cars,
            "camera_obs": self.camera_obs,
            "world_strategy": (
                _describe(self.world_strategy) if self.world_strategy is not None else None
            ),
            "object_avoidance": (
                dataclasses.asdict(self.object_avoidance)
                if self.object_avoidance is not None
                else None
            ),
            "domain_randomization": (
                dataclasses.asdict(self.domain_randomization)
                if self.domain_randomization is not None
                else None
            ),
            "training": dataclasses.asdict(self.training),
            "tracking": dataclasses.asdict(self.tracking),
            "trace": dataclasses.asdict(self.trace),
            "enable_gui": self.enable_gui,
            "use_gpu": self.use_gpu,
            "seed": self.seed,
        }

    def flat_params(self) -> dict[str, Any]:
        """Flatten ``to_dict()`` into dotted keys for ``mlflow.log_params``."""
        flat: dict[str, Any] = {}

        def walk(prefix: str, val: Any) -> None:
            if isinstance(val, dict):
                if not val:
                    flat[prefix] = "{}"
                    return
                for k, v in val.items():
                    walk(f"{prefix}.{k}" if prefix else str(k), v)
            elif isinstance(val, (list, tuple)):
                flat[prefix] = json.dumps(val)
            elif val is None:
                flat[prefix] = ""
            else:
                flat[prefix] = val

        walk("", self.to_dict())
        return flat

    def with_overrides(self, **overrides: Any) -> ExperimentConfig:
        """Return a new ExperimentConfig with dotted-key overrides applied.

        Walks dataclass fields and dict-typed fields. Examples::

            cfg.with_overrides(name="trial_3")
            cfg.with_overrides(**{"trainer.kwargs.learning_rate": 1e-4})

        Used by HPO to mutate a base experiment per trial, and by the
        in-container chunk dispatcher to apply per-chunk env-var overrides.
        """
        return _apply_overrides(self, overrides)


def _describe(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        d = dataclasses.asdict(obj)
        d["__class__"] = f"{obj.__class__.__module__}.{obj.__class__.__qualname__}"
        return d
    return _describe_callable(obj)


def _describe_callable(obj: Any) -> str:
    mod = getattr(obj, "__module__", "?")
    name = getattr(obj, "__qualname__", repr(obj))
    return f"{mod}.{name}"


def _apply_overrides(obj: Any, overrides: dict[str, Any]) -> Any:
    grouped: dict[str, dict[str, Any]] = {}
    leaves: dict[str, Any] = {}
    for key, val in overrides.items():
        if "." in key:
            top, rest = key.split(".", 1)
            grouped.setdefault(top, {})[rest] = val
        else:
            leaves[key] = val

    replacements: dict[str, Any] = dict(leaves)
    for top, sub in grouped.items():
        current = getattr(obj, top)
        if dataclasses.is_dataclass(current):
            replacements[top] = _apply_overrides(current, sub)
        elif isinstance(current, dict):
            new_dict = dict(current)
            for sub_key, sub_val in sub.items():
                _set_nested(new_dict, sub_key.split("."), sub_val)
            replacements[top] = new_dict
        else:
            raise ValueError(
                f"Cannot apply nested override {top}.{next(iter(sub))} to non-dataclass field"
            )
    return dataclasses.replace(obj, **replacements)


def _set_nested(d: dict, path: list[str], value: Any) -> None:
    cursor = d
    for key in path[:-1]:
        nxt = cursor.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[key] = nxt
        cursor = nxt
    cursor[path[-1]] = value


def load_config(path: str | Path) -> ExperimentConfig:
    """Import a Python file and return its ``experiment`` module attribute.

    Used by the in-container worker to load the same script the host ran.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    spec = importlib.util.spec_from_file_location(p.stem, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module from {p}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "experiment"):
        raise ValueError(f"{p} must export `experiment: ExperimentConfig`")
    cfg = module.experiment
    if not isinstance(cfg, ExperimentConfig):
        raise TypeError(f"{p} `experiment` must be ExperimentConfig, got {type(cfg)}")
    return cfg


def load_search_space(path: str | Path):
    """Import a Python file and return its ``search_space`` module attribute.

    The function should take an Optuna trial and return a flat dotted-key
    overrides dict consumable by ``ExperimentConfig.with_overrides``.
    """
    p = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(p.stem, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module from {p}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "search_space", None)
    if fn is None:
        raise ValueError(f"{p} must export `search_space(trial) -> dict` for HPO")
    return fn
