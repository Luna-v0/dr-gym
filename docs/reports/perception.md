# Perception net + asymmetric actor-critic — W-perception · `[REAL]` · 2026-06-22

The bridge from sim to the physical car: the deployed policy sees **only the front camera**, but in sim we
have privileged ground truth for free. This workstream builds (1) a supervised perception head that learns
to read *frame-local driving features* from the camera, and (2) the asymmetric actor-critic arrangement that
lets the critic exploit privileged state at train time without the actor ever depending on it.

**Status (2026-06-22):** net + target builder + data collector + supervised trainer + tests **built**
(sim-free parts green: `uv run pytest tests/test_perception.py` = 12 passed). Data collection + the
supervised fit are **queued** (need a live Gazebo). The asymmetric critic is **designed below**, pending the
constrained-RL backend (FSRL/Tianshou) where it slots in cleanly.

## Why these features, and which we refuse to learn
Labels come free from the 26-key `reward_params`. The key design distinction is **label availability vs
observability**: a *label* is always cheap (a value or a finite difference of `reward_params`), but whether
the camera can *observe* it depends on how many frames the quantity needs. The 4-frame stack is what makes
the dynamic quantities learnable — **2 frames to see speed (optical flow), 3 to see acceleration (speed
changing), 4 to see jerk.**

**Core six** (`perception_targets`, validated) — static + first-order, all ego-relative:

| feature | from `reward_params` | range | frames to observe |
|---|---|---|---|
| `lateral_offset` | `distance_from_center` signed by `is_left_of_center`, ÷ half-width | [-1,1] | 1 |
| `heading_error` | `heading` − local tangent from `waypoints[closest_waypoints]`, ÷180 | [-1,1] | 1 |
| `dist_left_edge` / `dist_right_edge` | half-width ± `distance_from_center`, ÷ width | [0,1] | 1 |
| `speed_norm` | `speed` ÷ 4.0 | [0,1] | 2 (also proprioceptive) |
| `yaw_rate` | Δ`heading` per step, ÷30°/step | [-1,1] | 2 (also IMU) |

**Dynamic candidates** (`DYNAMIC_FEATURES`, MAE-gated) — what the 4-stack unlocks:

| feature | label | frames to observe | why |
|---|---|---|---|
| `long_accel` | Δ`speed` per step | 3 | momentum / throttle response |
| `lateral_velocity` | Δ`lateral_offset` per step | 3 | drifting toward an edge |
| `edge_closing_rate` | −Δ(nearest-edge dist) per step | 3 | **safety-critical** — this *is* the graded risk `gym_dr/costs.py` encodes |

**We deliberately do NOT regress global `x`/`y`/`heading` or `progress`.** Two different places on a track
look identical to a forward camera (perceptual aliasing), so those labels are unlearnable from one frame and
would just train the net to hallucinate a position (they live in `privileged_state`, critic-only).

**Methodology — let the data decide.** We collect labels for the *full* candidate set (`all_targets` =
core ⊕ dynamic) and let the **held-out per-feature MAE table** (`experiments/train_perception.py`) say which
are actually recoverable from the camera. Low MAE ⇒ keep on the actor; high MAE ⇒ drop (the actor must not
lean on a feature it can't see). This *is* the W-perception deliverable — an empirical learnability ranking,
not a guess. The net's output head adapts automatically: `signed_indices_for(features)` picks tanh (signed)
vs sigmoid ([0,1]) channels from whatever feature set the dataset was built with.

## Pipeline (all built, sim-gated runs queued)
1. **Collect** — `scripts/collect_perception_data.py`: drives the privileged pure-pursuit controller (reused
   from `scripts/scripted_baseline.py`) with **ε-random** actions so the set covers near-edge / off-heading
   states (where `gym_dr/costs.py` fires — exactly where perception must be accurate). Captures the *same*
   grayscale 4-frame stack the policy sees (built via the real `time_trial` factory, so wrappers match) paired
   with `perception_targets`. Writes `obs (N,4,120,160) uint8` + `targets (N,6) f32` to `.npz`. Run across
   several worlds and concatenate — the net should see the same track variety the policy will.
2. **Train** — `experiments/train_perception.py`: CPU-friendly, fits `gym_dr.perception.PerceptionNet`
   (DeepRacer "shallow" conv encoder → 256 FC → 6 outputs, tanh on signed channels / sigmoid on bounded),
   SmoothL1, reports per-feature held-out MAE, saves `perception_net.pt` (+ feature names, input shape).
3. **Deploy** — the trained encoder is the camera front-end on the car (W-deploy); export through the same
   ONNX path. Because targets are ego-relative and the net divides by 255 internally, it is self-contained.

## Asymmetric actor-critic (design — slots into the FSRL/Tianshou backend)
Standard SB3 `MultiInputPolicy` feeds actor and critic the **same** observation, so it can't give the critic
privileged state. Two ways to get asymmetry, in increasing integration cost:

- **(A) Perception-as-frozen-frontend (cheap, SB3-compatible).** Pretrain `PerceptionNet` supervised, then
  use its encoder as a (optionally frozen) `features_extractor` for the *actor*, while the *critic* keeps the
  full `DeepRacerCNN` or — better — a small MLP over the **privileged feature vector** exposed through `info`.
  This needs the critic to receive privileged state, which SB3 won't route — so in SB3 we approximate by
  giving *both* heads the perception features (symmetric) and rely on the pretrained features for sample
  efficiency. Real asymmetry needs (B).
- **(B) True asymmetric critic (the FSRL/Tianshou path).** Tianshou's actor/critic are independent modules
  with independent inputs, so the critic can take a **privileged observation** (the `perception_targets`
  vector itself, or raw pose) while the actor takes only the camera stack + perception features. This is the
  right home: the constrained-RL backend (`gym_dr/trainers/fsrl_trainer.py`, D9) already builds separate
  nets, and the cost-critic there benefits from privileged state for the *same* reason the value-critic does
  (reward/cost have different scales and the privileged signal is denser). Plan: extend `FsrlTrainer` so the
  env emits both the camera stack (actor obs) and the privileged vector (critic-only obs, via `info` or a
  Tuple obs the actor masks), and build the actor on the pretrained `PerceptionNet` encoder.

**Guardrail (non-negotiable, `docs/` guardrails):** the deployed actor **never** touches privileged sim
state. The asymmetry lives only in the critic, which is discarded at deployment. "Passes in sim with a
privileged critic" is not a deployment claim — only the camera-only actor ships.

## Domain randomization coupling
Train perception **with** the DR wrappers (`gym_dr/domain_randomization.py`, ADR): observation noise /
brightness jitter on the frames, actuator noise on the collection controller. A perception net that survives
DR is the one that transfers to the car's real camera — this is where W-dr and W-perception meet, and why the
collector uses the real wrapper stack rather than raw frames.

## Data sourcing — dedicated collector vs rosbag (two complementary sources)
1. **Dedicated collector** (`scripts/collect_perception_data.py`, built) — scripted + ε-random driving for
   **controlled coverage**, especially near-edge / off-heading states where perception must be accurate. On
   demand, sim only.
2. **Rosbag from a normal training run** (designed, not built) — record the camera topic
   `/<racecar>/camera/zed/rgb/image_rect_color` + `/clock` to a bag during any RL run; the **labels come free**
   from the parquet trace (`reward_params` per step). Join **on `sim_time`** offline (the trace contract's
   `sim_time` column was reserved for exactly this *"bag→trace path"* — `gym_dr/trace.py`) → `(4-frame stack,
   ALL_FEATURES label)` → `npz` → `train_perception.py`. The bag's temporal order also yields the dynamic
   features for free.
   - **Pros:** free on-policy data as a training byproduct; the *actual* published frames (real sensor-pipeline
     fidelity); **same topic name on the physical car**, so one pipeline serves sim *and* real.
   - **Real-car use:** a car bag has frames but **no privileged labels** → feeds self-supervised
     representation learning, sim2real distribution-gap measurement, and on-real evaluation of `g`. The bridge
     to `[REAL]`.
   - **Caveats:** image bags are large (record `CompressedImage` / subsample / cap); a *converged* policy
     under-samples the dangerous near-edge states (record across the whole run + keep the ε-random collector);
     align on `sim_time`, never wall-clock (RTF-accelerated sim would misalign frames vs labels).
   - **Built (2026-06-22):** the host-side join script `scripts/bag_to_perception.py` — its pure join/stack
     core (sim_time nearest-join + 4-frame stacking) is tested (`tests/test_bag_to_perception.py`, 7 tests);
     the bag reader uses `rosbags` (installable on py3.8, lazy-imported). **Precondition found:** the current
     trace stores raw geometry but **not** `is_left_of_center` / `waypoints`, so `lateral_offset`'s sign and
     `heading_error` can't be recomputed offline — the trace must be **extended with the derived `ALL_FEATURES`
     columns** at write time (`gym_dr/metrics.py`) first. **Still to build:** that trace extension + a
     deepracer-env launch flag to record `camera + /clock`.

## Next (queued, needs a free Gazebo)
- Collect ~5 worlds × ~20 episodes, concatenate, train, read the per-feature MAE learnability table.
- Decide actor input from that table (drop any feature with MAE > ~0.2 from the actor's diet).
- Wire option (B) into `FsrlTrainer` once the FSRL camera CNN is finalized (D9).

## Files
- `gym_dr/perception.py` — `perception_targets`, `PerceptionNet`, `PERCEPTION_FEATURES`.
- `scripts/collect_perception_data.py` — privileged + ε-random collector → `.npz`.
- `experiments/train_perception.py` — supervised fit + per-feature MAE table.
- `tests/test_perception.py` — target-builder correctness + net shape/range/overfit (sim-free, green).
