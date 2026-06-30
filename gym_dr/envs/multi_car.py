"""Multi-car (N agents in one Gazebo world) -> SB3 ``VecEnv``.

ONE Gazebo world steps physics once for all N namespaced racecars, so the
per-step sim cost amortizes across agents and PPO gets decorrelated parallel
samples. The raw N-agent orchestration is ``deepracer_env``'s
``MultiAgentDeepRacerEnv`` (free-running step, per-car reset); this module adapts
it to the SB3 ``VecEnv`` interface (``num_envs = n_cars``), applying per-car:

  * **action transform** — policy acts in ``[-1,1]`` (when ``normalize_actions``)
    mapped to engineering units and clipped to the action bounds, mirroring the
    single-car ``NormalizeActions`` + ``ActionBounds`` wrappers;
  * **observation transform** — grayscale camera frame (``camera_obs=True``) or
    the ``ALL_FEATURES`` vector from ``reward_params`` (``camera_obs=False``);
  * **auto-reset** — a car whose episode ended is reset on its own (terminal obs
    in ``info['terminal_observation']``), VecEnv convention, others keep driving.

Frame stacking stays the trainer's job (``VecFrameStack`` wraps this VecEnv). The
``backend`` is injectable so the orchestration logic is unit-tested against a mock
Gazebo (``tests/test_multi_car_vecenv.py``). See ``docs/reports/multi-car.md``.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from gym_dr.action_space import ContinuousActionSpaceConfig
from gym_dr.perception import ALL_FEATURES, all_targets

_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)  # BT.601, matches GrayscaleObs


def _find_image_key(space: gym.spaces.Dict) -> Optional[str]:
    for key, sub in space.spaces.items():
        if isinstance(sub, gym.spaces.Box) and len(sub.shape) == 3:
            return key
    return None


class MultiCarVecEnv(VecEnv):
    """SB3 ``VecEnv`` over N cars sharing one Gazebo world."""

    def __init__(self, backend, *, camera_obs: bool,
                 action_cfg: ContinuousActionSpaceConfig,
                 actuator_steering_std: float = 0.0,
                 actuator_speed_std: float = 0.0,
                 noise_seed: Optional[int] = None,
                 obs_gaussian: float = 0.0, obs_brightness: float = 0.0,
                 obs_contrast: float = 0.0, obs_gamma: float = 0.0,
                 drag_spec=None, steer_bias_max: float = 0.0, speed_bias_max: float = 0.0,
                 feature_targets=None, feature_dim: int = 0,
                 feature_noise: float = 0.0, feature_asym: bool = False,
                 dr_warmup_steps: int = 0,
                 recorder=None, car_tracks: Optional[Sequence[str]] = None,
                 dr_meta: Optional[dict] = None) -> None:
        self._backend = backend
        self._camera = camera_obs
        self._cfg = action_cfg
        self._normalize = bool(getattr(action_cfg, "normalize_actions", True))
        n = backend.n_cars
        # Perception dataset recorder (camera frames + feature targets). Only the
        # camera path produces frames; ``recorder`` is None unless GYM_DR_PERCEPTION_OUT
        # is set. car_tracks/dr_meta tag each episode shard (which track, what DR).
        self._rec = recorder if camera_obs else None
        self._car_tracks = list(car_tracks) if car_tracks else [""] * n
        self._dr_meta = dict(dr_meta or {})
        # Actuator-noise DR (engineering units) applied per-car in the action
        # transform — the single-car ActuatorNoise wrapper can't reach this VecEnv
        # (metrics.wrap passes it through), so we replicate it here. 0 => off.
        self._act_noise = (
            np.array([actuator_steering_std, actuator_speed_std], dtype=np.float32)
            if (actuator_steering_std or actuator_speed_std) else None)
        self._noise_rng = np.random.default_rng(noise_seed)
        # Per-episode speed regime ("drag"): each episode multiplies the COMMANDED
        # speed by a factor ~U[low, 1.0] so some whole episodes drive slow and others
        # fast — that's what makes the dataset's executed-speed distribution span the
        # full range (slow→peak) rather than bunching where the policy likes to drive.
        # The single-car DragRandomization wrapper can't reach this VecEnv, so we
        # replicate it per-car here (resampled on each car's episode reset).
        from gym_dr.randomization import sample_spec, spec_bounds
        self._samp = sample_spec
        self._drag_spec = drag_spec
        self._has_drag = drag_spec is not None and spec_bounds(drag_spec)[0] < 1.0
        self._drag_factor = np.ones(n, dtype=np.float32)
        # Per-episode ACTUATOR BIAS: a CONSTANT lean held for the whole episode (a
        # miscalibrated actuator) — e.g. a steering trim where "0 command -> -20 deg",
        # or a motor that runs hot/cold (a speed offset). Resampled per car per episode
        # from U[-max, +max]; the policy must detect the drift and compensate. Distinct
        # from the per-STEP symmetric jitter (_act_noise). 0 => off.
        self._steer_bias_max = float(steer_bias_max)
        self._speed_bias_max = float(speed_bias_max)
        self._has_bias = self._steer_bias_max > 0.0 or self._speed_bias_max > 0.0
        self._act_bias = np.zeros((n, 2), dtype=np.float32)
        # Photometric observation DR for the camera path (brightness/contrast/gamma/
        # gaussian). The single-car ObservationNoise wrapper can't reach this VecEnv,
        # so we apply the SAME jitter here. The recorded dataset frame is the jittered
        # one (what the policy sees); labels stay ground-truth (from reward_params).
        self._obs_jitter = None
        if camera_obs and (obs_gaussian or obs_brightness or obs_contrast or obs_gamma):
            self._obs_jitter = dict(gaussian_std=float(obs_gaussian),
                                    brightness=float(obs_brightness),
                                    contrast=float(obs_contrast), gamma=float(obs_gamma))

        # engineering-unit action bounds (what the sim executes)
        self._low = np.array([action_cfg.steering_low, action_cfg.speed_low], dtype=np.float32)
        self._high = np.array([action_cfg.steering_high, action_cfg.speed_high], dtype=np.float32)
        # the action space the POLICY sees: [-1,1] when normalizing, else eng units
        if self._normalize:
            action_space: gym.spaces.Space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        else:
            action_space = gym.spaces.Box(self._low, self._high, dtype=np.float32)

        # Feature-obs config (camera-off path): which target builder (actor_targets 11
        # vs all_targets 9), the dim, per-step feature noise on the ACTOR vector, and
        # the asymmetric Dict obs {actor:noised, critic:true} — mirrors the single-car
        # FeatureObsWrapper so the asym oracle can run multi-car at high car counts.
        self._feat_targets = feature_targets or all_targets
        self._feat_dim = int(feature_dim) or len(ALL_FEATURES)
        self._feat_noise = float(feature_noise)
        self._feat_asym = bool(feature_asym) and not camera_obs

        # DR WARMUP (the multi-car ADR substitute). Multi-car can't run the in-loop
        # held-out eval that feedback-ADR needs (set_world is disabled here), so
        # instead of applying every DR magnitude at full strength from step 0 — which
        # left the policy in an unlearnable POMDP (unobservable ±bias + full
        # feature/actuator noise) and flatlined learning — we ramp ALL magnitude
        # knobs (bias, feature_noise, actuator + obs noise) by a factor that grows
        # linearly 0 -> 1 over the first ``dr_warmup_steps`` TIMESTEPS. Early
        # episodes are near-clean (learnable + survivable, so the frame-stacked
        # policy can observe the drift and infer the bias); the perturbations reach
        # full strength only once it can drive. Self-counted from the VecEnv's own
        # step count — no callback/eval signal needed. 0 => off (full strength).
        self._dr_warmup_steps = int(dr_warmup_steps)
        self._dr_steps = 0

        if camera_obs:
            self._image_key = _find_image_key(backend.single_observation_space)
            if self._image_key is None:
                raise ValueError("camera_obs=True but backend obs has no image space")
            h, w, _ = backend.single_observation_space.spaces[self._image_key].shape
            self._hw = (h, w)
            # Dict obs (grayscale single frame) — matches the single-car GrayscaleObs
            # output so MultiInputPolicy + DeepRacerCNN + VecFrameStack work unchanged.
            obs_space: gym.spaces.Space = gym.spaces.Dict(
                {self._image_key: gym.spaces.Box(0, 255, (h, w, 1), np.uint8)})
        else:
            self._image_key = None
            _fbox = gym.spaces.Box(-1.0, 1.0, (self._feat_dim,), np.float32)
            obs_space = (gym.spaces.Dict({"actor": _fbox, "critic": _fbox})
                         if self._feat_asym else _fbox)

        super().__init__(num_envs=n, observation_space=obs_space, action_space=action_space)
        self._actions: Optional[np.ndarray] = None
        self._prev_params: List[Optional[dict]] = [None] * n
        # Per-car episode metrics: the SAME ``_EpisodeMetrics`` the single-car wrapper
        # uses, one per car, attached by ``gym_dr.metrics.install_metrics``. They make
        # multi-car stamp ``info[i]["dr_episode"]`` (+ path), so the existing vec-aware
        # callbacks (dr/* logging, clean_completion eval, path plots) light up exactly
        # like the single-car path — the unified pipeline. ``None`` until attached.
        self._metrics: List[Any] = [None] * n
        self._eval_reward_fn: Optional[Any] = None
        self._use_eval = False
        # The multi-agent backend has NO set_world (each car's track + offset TrackData
        # is fixed at construction). The eval callback reads this to avoid "evaluating"
        # every held-out world on the current training tracks (a silent set_world no-op
        # that would fake per-world metrics + a ~0 generalization gap). Held-out eval is
        # a separate single-car pass. See gym_dr/trainers/sb3/callbacks.py.
        self.can_set_world: bool = hasattr(backend, "set_world")

    # ---- action / obs transforms ------------------------------------- #
    def _dr_scale(self) -> float:
        """Current DR magnitude multiplier in [0, 1] — the linear warmup factor.

        Grows 0 -> 1 over the first ``dr_warmup_steps`` timesteps, then stays 1.
        ``0`` warmup steps => always 1.0 (full strength, the legacy behaviour).
        """
        if self._dr_warmup_steps <= 0:
            return 1.0
        return min(1.0, self._dr_steps / float(self._dr_warmup_steps))

    def _resample_drag(self, car: int) -> None:
        """Resample this car's per-EPISODE actuator DR: speed regime (drag) + the
        constant steering/speed bias (held for the whole episode). The bias is scaled
        by the current DR-warmup factor, so early (near-clean) episodes stay
        survivable and the lean grows to full ±max only once the policy can drive."""
        if self._has_drag:
            self._drag_factor[car] = float(self._samp(self._drag_spec, self._noise_rng))
        if self._has_bias:
            scale = self._dr_scale()
            self._act_bias[car, 0] = self._noise_rng.uniform(
                -self._steer_bias_max, self._steer_bias_max) * scale
            self._act_bias[car, 1] = self._noise_rng.uniform(
                -self._speed_bias_max, self._speed_bias_max) * scale

    def _to_engineering(self, action: np.ndarray, car: int = 0) -> np.ndarray:
        a = np.asarray(action, dtype=np.float32).copy()
        if self._normalize:  # [-1,1] -> [low, high]
            a = self._low + (a + 1.0) * 0.5 * (self._high - self._low)
        if self._has_drag:   # per-episode speed regime: scale commanded speed
            a[1] = a[1] * self._drag_factor[car]
        if self._has_bias:   # per-episode constant lean (steering trim / motor offset)
            a = a + self._act_bias[car]   # already DR-warmup-scaled at resample
        if self._act_noise is not None:  # additive per-step (symmetric) Gaussian jitter
            a = a + (self._act_noise * self._dr_scale()
                     * self._noise_rng.standard_normal(2).astype(np.float32))
        return np.clip(a, self._low, self._high)

    def _obs_from(self, raw_obs, info, car: int):
        if self._camera:
            img = np.asarray(raw_obs[self._image_key], dtype=np.uint8)
            gray = (img[..., :3].astype(np.float32) @ _LUMA).astype(np.uint8)
            if self._obs_jitter is not None:
                from gym_dr.envs.wrappers import apply_image_jitter
                gray = apply_image_jitter(gray, self._noise_rng, **self._obs_jitter)
            return {self._image_key: gray[..., None]}  # Dict{key: (H, W, 1)}
        params = (info or {}).get("reward_params", {}) if info else {}
        clean = self._feat_targets(params, self._prev_params[car]).astype(np.float32)
        if params:
            self._prev_params[car] = dict(params)
        feat_noise = self._feat_noise * self._dr_scale()  # DR-warmup ramp
        if feat_noise > 0:         # actor-robustness DR: noise the feature vector
            noised = np.clip(clean + self._noise_rng.normal(0.0, feat_noise, clean.shape),
                             -1.0, 1.0).astype(np.float32)
        else:
            noised = clean
        if self._feat_asym:        # actor sees noised, critic sees the TRUE vector
            return {"actor": noised, "critic": clean}
        return noised

    def _stack(self, obs_list: Sequence[Any]):
        """Batch per-car obs to num_envs-first. Dict obs (camera image, OR the
        asymmetric {actor,critic} feature obs) stack per key; a plain feature array
        stacks directly."""
        if isinstance(obs_list[0], dict):
            return {k: np.stack([o[k] for o in obs_list], axis=0) for k in obs_list[0]}
        if self._camera:
            return {self._image_key: np.stack([o[self._image_key] for o in obs_list], axis=0)}
        return np.stack(obs_list, axis=0)

    # ---- VecEnv interface -------------------------------------------- #
    def _record_step(self, car: int, obs, params: Optional[dict],
                     action: Optional[np.ndarray] = None) -> None:
        """Capture one camera frame + its feature target (+ the policy's action)
        for the dataset (no-op unless a recorder is attached). ``obs`` is this
        car's Dict camera obs; ``action`` is the engineering-unit [steer, speed]."""
        if self._rec is None or not params:
            return
        try:
            self._rec.record(car, obs[self._image_key], params, action)
        except Exception:  # noqa: BLE001 — recording must never break the rollout
            pass

    def _start_episode(self, car: int) -> None:
        if self._rec is None:
            return
        track = self._car_tracks[car] if car < len(self._car_tracks) else ""
        self._rec.start_episode(car, track=track, dr_meta=self._dr_meta)

    # ---- per-car episode metrics (unified pipeline) ------------------- #
    def attach_metrics(self, *, cost_fn=None, eval_reward_fn=None,
                       capture_path: bool = False) -> None:
        """Build one ``_EpisodeMetrics`` per car (called by ``install_metrics``).

        Mirrors the single-car ``_MetricsEnvWrapper`` so multi-car produces the same
        ``info[i]["dr_episode"]`` the dr/* + eval callbacks consume. Trace sinks are
        not attached here (multi-car trace is a follow-up; the camera run has trace
        off and uses the perception recorder for its dataset)."""
        from gym_dr.metrics import _EpisodeMetrics
        self._eval_reward_fn = eval_reward_fn
        self._metrics = []
        for _ in range(self.num_envs):
            st = _EpisodeMetrics()
            st.cost_fn = cost_fn
            st.capture_path = bool(capture_path)
            self._metrics.append(st)

    def _record_metrics(self, car: int, params: Optional[dict], reward: float) -> float:
        """Feed this car's step into its metrics state; return the eval-reward value."""
        st = self._metrics[car]
        if st is None or not params:
            return 0.0
        er = float(self._eval_reward_fn(params)) if self._eval_reward_fn else 0.0
        try:
            st.record_step(params, float(reward), er)
        except Exception:  # noqa: BLE001 — metrics must never break the rollout
            pass
        return er

    def reset(self) -> np.ndarray:
        self._prev_params = [None] * self.num_envs
        for i in range(self.num_envs):             # fresh per-car speed regime
            self._resample_drag(i)
        for st in self._metrics:                   # drop partial episodes' metrics
            if st is not None:
                st.reset()
        if self._rec is not None:
            self._rec.flush_all()  # drop any partial episodes from a prior rollout
        raw = self._backend.reset()
        obs = [self._obs_from(raw[i], None, i) for i in range(self.num_envs)]
        if self._rec is not None:
            for i in range(self.num_envs):
                self._start_episode(i)
                # first frame has no reward_params from reset(); the first recorded
                # frame comes on the next step, which is fine (stack warms up).
        return self._stack(obs)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self):
        self._dr_steps += self.num_envs       # advance the DR-warmup ramp (timesteps)
        eng = [self._to_engineering(self._actions[i], i) for i in range(self.num_envs)]
        raw_obs, rewards, dones, infos = self._backend.step(eng)
        obs_out, info_out, rew_out = [], [], []
        for i in range(self.num_envs):
            info = dict(infos[i]) if infos[i] else {}
            o = self._obs_from(raw_obs[i], infos[i], i)
            params = info.get("reward_params")
            # Record THIS step's frame + target before any auto-reset (the obs `o`
            # here is the real step observation; ``params`` its label).
            self._record_step(i, o, params, eng[i])
            # Per-car episode metrics (unified pipeline): accumulate this step.
            eval_r = self._record_metrics(i, params, rewards[i])
            # During eval, return the eval-reward (matches the single-car wrapper);
            # otherwise the training reward.
            rew_out.append(eval_r if (self._use_eval and self._eval_reward_fn) else rewards[i])
            if dones[i]:
                info["terminal_observation"] = o
                self._prev_params[i] = None
                st = self._metrics[i]
                if st is not None:                 # stamp the summary the callbacks read
                    info["dr_episode"] = st.summary()
                    if st.capture_path:
                        info["dr_episode_path"] = st.path_payload()
                    st.reset()
                if self._rec is not None:          # close the shard, open the next
                    self._rec.flush_episode(i)
                    self._start_episode(i)
                self._resample_drag(i)             # new episode's speed regime
                reset_raw = self._backend.reset_one(i)
                o = self._obs_from(reset_raw, None, i)
            obs_out.append(o)
            info_out.append(info)
        return (self._stack(obs_out), np.asarray(rew_out, np.float32),
                np.asarray(dones, dtype=bool), info_out)

    def close(self) -> None:
        self._backend.close()

    # SB3 plumbing (single-process VecEnv; these touch the shared backend) ----
    def get_attr(self, attr_name: str, indices=None) -> List[Any]:
        return [getattr(self, attr_name, getattr(self._backend, attr_name, None))
                for _ in self._idx(indices)]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> List[Any]:
        # The eval callback toggles the recorder's phase tag (train/eval) around
        # evaluation so eval-rollout shards are labelled — reached through the
        # VecFrameStack wrapper's env_method passthrough.
        if method_name == "set_recorder_phase":
            if self._rec is not None and args:
                self._rec.set_phase(args[0])
            return [None for _ in self._idx(indices)]
        # The eval callback flips this around evaluation so the per-car metrics report
        # the eval reward (mirrors the single-car state.use_eval_reward toggle).
        if method_name == "set_metrics_eval_mode":
            self._use_eval = bool(args[0]) if args else False
            for st in self._metrics:
                if st is not None:
                    st.use_eval_reward = self._use_eval
            return [None for _ in self._idx(indices)]
        fn = getattr(self._backend, method_name, None)
        return [fn(*args, **kwargs) if callable(fn) else None for _ in self._idx(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None) -> List[bool]:
        return [False for _ in self._idx(indices)]

    def _idx(self, indices):
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            return [indices]
        return indices


def multi_car(experiment) -> MultiCarVecEnv:
    """Build the N-car VecEnv for ``experiment`` (n_cars > 1)."""
    from deepracer_env.environments.multi_agent_env import MultiAgentDeepRacerEnv

    cfg = experiment.action_space
    if not isinstance(cfg, ContinuousActionSpaceConfig):
        raise TypeError("multi-car requires a ContinuousActionSpaceConfig action space")
    import os
    # Camera multi-car is capped at 2 because the LAUNCH only spawns 2 car bodies
    # (racetrack_with_racecar.launch hardcodes racecar_0/1 + car_node.py args="2");
    # nothing spawns a 3rd body, so racecar_2's camera topic advertises but has NO
    # publisher (0 Hz) and the agent's blocking sensor read log_and_exit()s ~120s in.
    # This is a LAUNCH/CONFIG limit, NOT "Gazebo renders only 2 cameras" (that earlier
    # claim was a misdiagnosis — see docs/reports/status-2026-06-28.md). Feature obs
    # (camera_obs=False) scales to n=8 only because a missing model's STATE read
    # doesn't block (phantom agents). Raising it needs generated racecar_2..N launch
    # blocks; the real residual limit is then Gazebo's single OGRE render thread
    # (graceful fps/RTF degradation past 2, not a crash). Fail fast meanwhile.
    if bool(experiment.camera_obs) and int(experiment.n_cars) > 2 \
            and os.getenv("GYM_DR_ALLOW_CAMERA_NCARS") != "1":
        raise ValueError(
            f"camera_obs multi-car is capped at n_cars=2: the launch only spawns 2 car "
            f"bodies (racetrack_with_racecar.launch), so car {experiment.n_cars - 1}'s "
            f"camera has no publisher and the blocking read aborts. Add racecar_2..N "
            f"launch blocks to raise it, use camera_obs=False (feature scales to n=8), "
            f"run >2 camera cars as separate processes, or set "
            f"GYM_DR_ALLOW_CAMERA_NCARS=1 to override.")
    # Per-car track list (the generalization engine): GYM_DR_DEMO_WORLDS is a
    # comma-separated list of track names, one per car — each car drives its own
    # track instance, so N cars train across N different tracks in one world.
    # Shorter-than-n_cars lists cycle; empty falls back to the experiment world
    # (all cars same track = parallel sampling). Forwarded by app.py.
    worlds_env = os.getenv("GYM_DR_DEMO_WORLDS", "").strip()
    worlds = None
    if worlds_env:
        names = [w.strip() for w in worlds_env.split(",") if w.strip()]
        n = int(experiment.n_cars)
        worlds = [names[i % len(names)] for i in range(n)]
    # Feature obs reads reward_params (car pose / track position), NOT pixels, so
    # the camera sensor is dead weight — and worse, each agent's camera sensor
    # blocks on a Gazebo image at reset; with N cars a camera can miss its frame and
    # DoubleBuffer.get() calls log_and_exit, killing the whole run (seen at n>=4).
    # Empty sensor list => CompositeSensor returns {} (no blocking) and nothing to
    # render. Camera obs keeps the camera sensor.
    sensors = list(cfg.sensor) if bool(experiment.camera_obs) else []
    # Domain randomization for the multi-car VecEnv (mirrors the single-car
    # time_trial wiring): random_start/random_direction are deepracer-env reset
    # modes passed through the controller config to every car; actuator noise is
    # applied per-car inside MultiCarVecEnv (the wrapper stack can't reach it).
    dr = getattr(experiment, "domain_randomization", None)
    reset_config: dict = {}
    if dr is not None and getattr(dr, "random_start", False):
        reset_config["random_start"] = True
    if dr is not None and getattr(dr, "random_direction", False):
        reset_config["random_direction"] = True
    # Use the RAW reward in the backend: ``install_metrics`` hands us a reward whose
    # closure records into a single shared ``_EpisodeMetrics``. With N cars that state
    # would accumulate every car's steps forever (wrong + a memory leak). The unified
    # per-car metrics (attach_metrics) do the recording instead, so the backend needs
    # only the plain reward value. ``__wrapped__`` is the original (metrics.py).
    reward_fn = getattr(experiment.reward, "__wrapped__", experiment.reward)
    backend = MultiAgentDeepRacerEnv(
        n_cars=int(experiment.n_cars),
        reward_fn=reward_fn,
        sensors=sensors,
        worlds=worlds,
        config=reset_config or None,
        # metres between separated track instances. Default 300 (cars can't see
        # each other); set GYM_DR_DEMO_SPACING small (e.g. 50) to fit both in one
        # VNC view for visual validation. Forwarded into the container by app.py.
        spacing=float(os.getenv("GYM_DR_DEMO_SPACING", "300")),
    )
    from gym_dr.randomization import spec_bounds
    # Feature-obs vector: actor_targets (11) when GYM_DR_FEATURE_SET=actor_extended,
    # else all_targets (9) — mirrors feature_time_trial so the multi-car feature path
    # matches the single-car one (incl. the asym oracle).
    if os.getenv("GYM_DR_FEATURE_SET") == "actor_extended":
        from gym_dr.perception import ACTOR_FEATURES as _AF, actor_targets as _ftargets
        _fdim = len(_AF)
    else:
        _ftargets, _fdim = all_targets, len(ALL_FEATURES)
    # Perception dataset recorder (camera frames + feature targets) — only active
    # when GYM_DR_PERCEPTION_OUT is set; captures both training and eval rollouts
    # (eval reuses this VecEnv). Tag shards with each car's track + the DR settings.
    from gym_dr.perception import ACTOR_FEATURES
    from gym_dr.perception_recorder import recorder_from_env
    recorder = recorder_from_env(int(experiment.n_cars), ACTOR_FEATURES)
    dr_meta = {
        "visual_dr": os.getenv("GYM_DR_VISUAL_DR", "0"),
        "friction_mu": os.getenv("GYM_DR_FRICTION_MU", ""),
        "obs_gaussian_hi": spec_bounds(getattr(dr, "obs_gaussian", 0.0))[1] if dr else 0.0,
        "steering_noise_hi": spec_bounds(dr.steering_noise)[1] if dr else 0.0,
    }
    return MultiCarVecEnv(
        backend, camera_obs=bool(experiment.camera_obs), action_cfg=cfg,
        actuator_steering_std=spec_bounds(dr.steering_noise)[1] if dr else 0.0,
        actuator_speed_std=spec_bounds(dr.speed_noise)[1] if dr else 0.0,
        noise_seed=getattr(dr, "seed", None) if dr else None,
        obs_gaussian=spec_bounds(dr.obs_gaussian)[1] if dr else 0.0,
        obs_brightness=spec_bounds(dr.obs_brightness)[1] if dr else 0.0,
        obs_contrast=spec_bounds(getattr(dr, "obs_contrast", 0.0))[1] if dr else 0.0,
        obs_gamma=spec_bounds(getattr(dr, "obs_gamma", 0.0))[1] if dr else 0.0,
        drag_spec=getattr(dr, "drag", None) if dr else None,
        steer_bias_max=spec_bounds(getattr(dr, "steering_bias", 0.0))[1] if dr else 0.0,
        speed_bias_max=spec_bounds(getattr(dr, "speed_bias", 0.0))[1] if dr else 0.0,
        feature_targets=_ftargets, feature_dim=_fdim,
        feature_noise=spec_bounds(getattr(dr, "feature_noise", 0.0))[1] if dr else 0.0,
        feature_asym=(os.getenv("GYM_DR_ASYM_CRITIC") == "1" and not bool(experiment.camera_obs)),
        # Linear DR warmup (the multi-car ADR substitute): ramp every magnitude knob
        # 0 -> full over the first N timesteps so the policy can learn to drive before
        # the unobservable bias / feature noise reach full strength. Forwarded by app.py.
        dr_warmup_steps=int(os.getenv("GYM_DR_DR_WARMUP_STEPS", "0") or 0),
        recorder=recorder, car_tracks=list(backend.worlds), dr_meta=dr_meta,
    )
