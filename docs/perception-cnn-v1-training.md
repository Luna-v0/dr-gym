# Perception CNN v1 — training spec (camera frame → feature vector)

> How to train v1 of the supervised net `g: camera → feature-vector` that distills the
> privileged actor features from vision, so the camera policy can run `π(g(camera))` on the
> real car. Exhaustive dump from the 2026-06-29 session; maintainer will refine.
> Sim-free, CPU-friendly — this is offline supervised regression on already-collected data.

---

## 1. Goal & where it fits

The oracle/arch policies consume a **privileged feature vector** (track geometry the sim
knows exactly). The real car has no such oracle — it has a camera. v1 trains a CNN to
**regress those features from a single grayscale frame**, producing the per-feature
**held-out MAE learnability table** (the W-perception deliverable): which quantities are
recoverable from vision, which are not. v2 then wires `g` in as the actor's
features-extractor (`π(g(camera))`, Test 2) and exports ONNX→OpenVINO for the car.

**v1 is NOT end-to-end RL.** It's plain supervised regression. Keep it boring and measurable.

---

## 2. The dataset (already collected — know its exact shape)

**Format:** per-episode `.npz` shards. Verified keys/shapes (one shard):
| key | shape | dtype | meaning |
|---|---|---|---|
| `frames` | `(T, 120, 160)` | `uint8` [0,255] | grayscale frames, T≈14 steps/shard |
| `targets` | `(T, 11)` | `float32` | the 11 ACTOR_FEATURES, **already normalized** to [-1,1]/[0,1] |
| `features` | `(11,)` | str | ordered target names (use to align the head) |
| `diag` | `(T, 6)` | `float32` | `progress, speed_mps, is_offtrack, x, y, heading` (sanity only) |
| `diag_cols` | `(6,)` | str | diag names |
| `meta` | scalar | str(JSON) | `visual_dr, friction_mu, obs_gaussian_hi, steering_noise_hi, track, car, phase, episode, ts` |

**Location & split (IMPORTANT):**
- **TRAIN:** `mlruns/**/perception_out/train/<track>/ep*_car*.npz` — **129,680 shards across
  114 tracks** on the main PC (≈ 1.8M frame/target pairs at T≈14).
- **VAL / TEST:** **0 shards in this mlruns tree.** They were captured *separately* by
  `experiments/perception_capture_heldout.py` (frozen-rollout, lr=0, on held-out tracks)
  and offloaded (recorder → NVMe → Pi; see memory `camera-cnn-dataset-run`). **LOCATE them
  (laptop / NVMe / Pi) or re-run the held-out capture before reporting generalization.**
- Split discipline is **BY-TRACK** (held-out *tracks*, no variant leakage — see memory
  `camera-cnn-dataset-run` and `docs/...` camera split notes). Report MAE on held-out
  **tracks**, never held-out frames of a trained track.

**Visual DR is baked into `frames`** (track/ground/wall/sky color, contrast/gamma, Gaussian
noise — see `meta.visual_dr`, `obs_gaussian_hi`). That's deliberate — the net must be robust
to it. Do NOT add more pixel aug that fights the recorded DR; light aug (small crops/jitter)
is fine.

---

## 3. The 11 target features — and which are actually learnable

Order (from `features`): `lateral_offset, heading_error, dist_left_edge, dist_right_edge,
speed_mps, yaw_rate, long_accel, lateral_velocity, edge_closing_rate, curvature_ahead,
nearest_object_dist`.

| group | features | v1 expectation |
|---|---|---|
| **Vision geometry** (the real prize) | `lateral_offset, heading_error, dist_left_edge, dist_right_edge` | **LOW MAE** — single forward frame shows these directly |
| **Visible map/obstacle** | `curvature_ahead` (short FOV lookahead), `nearest_object_dist` | low-ish MAE if the corner/object is in frame |
| **Proprioceptive** | `speed_mps` (wheel encoders), `yaw_rate` (IMU) | HIGH MAE from one frame is FINE — on the real car these come from sensors, not vision; keep them proprioceptive at deploy |
| **Temporal deltas** | `long_accel, lateral_velocity, edge_closing_rate` | HIGH MAE for single-frame v1 — they need motion → use the 4-frame stack (v1.1) or finite-difference consecutive CNN outputs |

**So a "good" v1 = low MAE on the 4 geometry features + curvature/object, with the
proprioceptive/temporal ones expected-high.** The MAE table is the deliverable; a high-MAE
feature is one the actor must NOT lean on (or must get from proprioception). Signed channels
(tanh head): `lateral_offset, heading_error, yaw_rate, long_accel, lateral_velocity,
edge_closing_rate` — use `signed_indices_for(features)`; the rest are [0,1] (sigmoid).

---

## 4. The net — `gym_dr.perception.PerceptionNet`

Already implemented. Conv stack → `Flatten` → `Linear(features_dim)` → `ReLU` →
`Linear(n_outputs)`, with `tanh` on the signed channels and `sigmoid`/identity on the rest
so outputs stay in the target ranges. Construct with the dataset's feature list:

```python
from gym_dr.perception import PerceptionNet, signed_indices_for
features = [...]  # from shard["features"], len 11
net = PerceptionNet(
    in_channels=1,                       # v1: single frame. v1.1: 4 (frame stack)
    n_outputs=len(features),             # 11
    input_hw=(120, 160),
    signed_indices=signed_indices_for(features),
)
```
- **v1: `in_channels=1`** — single frame, simplest. The targets are per-frame, so this is
  valid. Loader yields `(frames[i]/255.0)[None] -> (1,120,160)`, target `targets[i]`.
- **v1.1: `in_channels=4`** — sliding window of 4 consecutive frames (channel-stacked),
  target = the latest frame's `targets`. Needed to learn the temporal deltas + match the
  policy's CameraObs (grayscale 4-stack) at deploy.

---

## 5. The code gap — the existing trainer reads the WRONG format

`experiments/train_perception.py` expects a **single legacy `.npz` with keys `obs`,
`targets`** (from the old `collect_perception_data.py`). The real dataset is **129k
per-episode shards with keys `frames`, `targets`** in a by-track tree — too big to
`np.concatenate` into RAM. **v1 needs a streaming tree loader.** Sketch:

```python
class ShardFrameDataset(torch.utils.data.Dataset):
    """Streams (frame, target) pairs from perception_out/<split>/<track>/*.npz.
    Build an index of (shard_path, t) once; load shards lazily (LRU-cache a few)."""
    def __init__(self, roots, stack=1):
        self.index = []                      # [(path, t), ...]
        for p in sorted(glob(f"{root}/**/*.npz", recursive=True) for root in roots):
            T = int(np.load(p, mmap_mode="r")["frames"].shape[0])  # or read a sidecar count
            self.index += [(p, t) for t in range(stack-1, T)]
        self.stack = stack
    def __getitem__(self, k):
        p, t = self.index[k]; d = np.load(p)          # + small LRU cache on p
        fr = d["frames"][t-self.stack+1 : t+1].astype(np.float32) / 255.0   # (stack,120,160)
        return torch.from_numpy(fr), torch.from_numpy(d["targets"][t])
```
Notes: index-build over 129k shards is the slow part — cache the index (pickle) keyed by
the dataset dir mtime; LRU-cache open shards; `num_workers>0`. Don't reopen per item naively.

---

## 6. Loss / optimizer / schedule (v1 defaults)

- **Loss:** per-feature **Smooth L1** (Huber) or MSE, summed/averaged over channels. Targets
  are pre-normalized so no per-channel scaling needed. **Optionally weight the loss** to
  downweight (or mask) the expected-unlearnable channels (`speed_mps, yaw_rate`, the 3
  temporal deltas) so the net spends capacity on the vision features — but still REPORT
  their MAE (don't hide them).
- **Optimizer:** Adam, `lr=1e-3`, `batch_size=256`, **20–30 epochs**. Cosine or step LR
  decay optional. CPU is fine; GPU faster (the net is small, the data load dominates).
- **Seed everything**; deterministic split by track.

---

## 7. Evaluation = the deliverable

Print/emit a **per-feature held-out MAE table** (MAE in normalized units), on **held-out
TRACKS** (val, then once on test). Also a few qualitative overlays: predicted vs true
`lateral_offset`/`heading_error` over an episode. The verdict: which features clear a usable
MAE bar (e.g. < ~0.05–0.10 normalized) → those are what the camera actor may consume; the
rest stay proprioceptive (speed/yaw) or critic-only.

---

## 8. Output & next steps

- Save `artifacts/perception/perception_net_v1.pt` (state_dict + the `features` list + a
  `meta` with input_hw/in_channels/stack so deploy reconstructs the head exactly).
- Emit the MAE table to `docs/reports/perception-v1-mae.md`.
- **v2 (Test 2, task #16):** wire `g` as the actor's `features_extractor` →
  `π(g(camera))`, asymmetric (critic keeps the privileged vector), add the perception
  penalty. Then **ONNX → OpenVINO IR** for the car (`gym_dr/optimize.py`,
  `docs/onnx-support-status.md`, memory `dr-gym-onnx-openvino-ir` — note the OpenVINO bf16
  gotcha + two-venv setup).

---

## 9. Concrete command (after the loader in §5 is added)

```bash
# v1, single-frame, train split here + held-out val (once located/regenerated):
uv run python experiments/train_perception.py \
    --train-root 'mlruns/**/perception_out/train' \
    --val-root   '<held-out val root>/perception_out/val' \
    --stack 1 --epochs 30 --batch-size 256 --lr 1e-3 \
    --out artifacts/perception/perception_net_v1.pt
# (current train_perception.py CLI is --data <npz>; it must be updated to the tree loader.)
```

## 10. Checklist
1. [ ] Add the streaming tree loader (§5) to `train_perception.py` (or a new module).
2. [ ] Locate or regenerate the **val/test** held-out shards (§2) — by-track, no leakage.
3. [ ] Train v1 (`in_channels=1`); emit the per-feature held-out MAE table (§7).
4. [ ] (v1.1) repeat with `stack=4` to recover temporal features + match the policy obs.
5. [ ] Decide the deployable feature subset from the MAE table.
6. [ ] (v2) wire `π(g(camera))` + perception penalty; ONNX→OpenVINO for the car.
