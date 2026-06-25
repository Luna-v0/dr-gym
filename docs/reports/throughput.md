# Throughput — threads vs RTF · `[BOTH]` · 2026-06-21

## Method
`scripts/throughput_benchmark.py`: launch K detached sim containers at
`rtf_override=R` for 120 s each; measure aggregate **agent env-steps/s** (from
`training_status.json`) and **effective RTF = sim-time/wall-time** (sampled off
ROS `/clock`). Workstation: RTX 4060 Ti (shared with other containers).

## Results
| workers | rtf_set | effective RTF | per-worker steps/s | aggregate steps/s |
|--:|--:|--:|--:|--:|
| 1 | 160 | 4.56 | 54.4 | 54.4 |
| 1 | 10  | 4.45 | 54.1 | 54.1 |
| 4 | 40  | 2.31 | 14.3 | 57.1 |
| 7 | 10  | n/a  | 0.1  | 0.8  |

## Findings
1. **`rtf_override` is not honored — the sim hard-caps at ~4.5× real-time.** Asking
   for 160 vs 10 gives the *same* effective RTF (~4.5) and ~54 steps/s. The setting
   is a ceiling/hint, not the achieved clock; the sim is **render/compute-bound**.
   → *the set RTF is NOT achieved; it throttles to ~4.5×.*
2. **fps ≠ sim clock (quantified).** SB3 `time/fps` ≈ 54 steps/s at ~4.5× RTF ⇒ the
   control rate is ≈ 12 Hz of *sim* time. fps is training throughput; RTF is the
   sim clock; they differ by the control period.
3. **Separate parallel containers don't scale on one GPU.** 1 worker = 54 steps/s;
   4 workers = 57 aggregate (per-worker collapses 54→14); 7 workers thrash to 0.8.
   → **1 instance is the sweet spot** for separate Gazebo processes here.

## Implications
- **Single-instance training is optimal** on this machine; multi-process parallelism
  (Q3, separate containers) does **not** pay off — it's GPU-render-bound.
- The untested lever is **N-cars-in-one-world** (one Gazebo, shared physics/render —
  the maintainer's preferred design). It may scale where N containers don't; that's
  the **next benchmark**.
- **Training ETA:** ~54 steps/s ⇒ 4M steps ≈ **~20 h**, and parallelism won't shorten
  it. Options: accept an overnight run, reduce the budget (2M ≈ ~10 h), or pursue real
  speedup via N-cars-in-one-world or a lighter renderer (`[DISS]`).

## Device sweep — 2×2 (render device × inference device), 1 worker, rtf 30
| config | effective RTF | steps/s |
|---|--:|--:|
| GPU-render + CUDA NN | 4.68 | 55.3 |
| GPU-render + CPU NN | 2.43 | 10.3 |
| CPU-render + CUDA NN | 5.23 | 55.0 |
| CPU-render + CPU NN (pure CPU) | 2.18 | 8.3 |

**Inference device dominates; render device barely matters** (corrects the "render-bound" claim above):
- CUDA inference → ~55 steps/s whether rendering is GPU or software; CPU inference → ~9 steps/s (≈5–6×
  slower). The per-step cost is the **network**, not the camera render.
- **CPU-render + CUDA-NN is the best cell** (55.0, RTF 5.23) — software rendering matched GPU rendering, so
  **GPU passthrough for Gazebo rendering is unnecessary; only CUDA for the NN matters.**
- The small-net hypotheses both fail here: the GPU↔host copy does **not** swamp the benefit, and CPU vector
  units are **not** enough — CUDA is ~5× faster even for this tiny CNN.
- The ~5× RTF ceiling is therefore the **Gazebo physics + ROS step loop**, not rendering and not (with GPU
  NN) the network.

**Implications:**
- **Train with `device=cuda`.** CPU training is ~5× slower (4M steps ≈ ~100 h vs ~20 h). D3 stays on GPU.
- **New multi-instance hypothesis to test:** with **software rendering + GPU inference**, the GPU only does
  light NN work, so several instances may scale across CPU cores (rendering parallelizes) where the earlier
  *GPU-rendered* multi-container test contended on the GPU. Re-run the threads sweep with `sw_render=True` +
  `device=cuda` before concluding "1 instance only" — alongside N-cars-in-one-world.

## Software-render multi-instance sweep (2026-06-22) — the hypothesis CONFIRMED, with a 2-worker sweet spot
Re-ran the threads sweep with `--sw-render` + `device=cuda` (rendering on CPU, NN on GPU), 120 s/point:

| workers | per-worker steps/s | **aggregate steps/s** | effective RTF |
|---|---|---|---|
| 1 | 55.5 | 55.5 | 5.71 |
| 2 | 41.6 | **83.3** | 3.71 |
| 4 | 9.1 | 36.4 (collapse) | 1.83 |

- **Software rendering DOES scale where GPU rendering did not.** The earlier GPU-rendered multi-container
  test flatlined (1→54, 4→57 aggregate) because every instance contended on the one GPU's render queue. With
  rendering moved to CPU and only the (tiny) NN on the GPU, **2 instances give ~83 steps/s — a real ~1.5×
  over a single env** — the first lever that has actually beaten the single-instance ceiling on this machine.
- **But it's CPU-bound and oversubscribes fast.** 4 workers *collapse* to 36 steps/s (below a single env):
  software rendering is CPU-heavy, so 4 render loops thrash the cores. **The sweet spot is 2 workers.**
- **Operating recommendation:** for parallel data collection / HPO on this box, run **2 software-rendered
  instances with GPU inference** (~83 steps/s aggregate), not 1 and not 4. D3 (a single long run) still wants
  the single GPU-rendered env at max per-worker rate; the 2-worker point is for *throughput-bound* studies
  (HPO, DR data collection) where wall-clock across many short runs matters more than one run's latency.

## CORRECTION (2026-06-23) — the sw-render gain does NOT transfer to full training
The 2-worker sw-render sweet spot above was measured on **lightweight env-stepping** (forward inference only).
On the **full PPO+CNN+trace+eval** workload (the reward search), sw-render gave **14.4 steps/s/worker**
(steady, two independent windows) — *no better than, likely worse than*, the GPU-render run (~20 steps/s/worker,
and that 20 *included* eval/boot overhead so its pure-collection rate was higher). So: **do not use sw-render
for real training.** The benchmark over-promised because training adds GPU backward passes and the CPU now
carries software rendering for 2 workers.

**Implication:** throughput on one machine is effectively capped (~14–20 steps/s/worker; more processes don't
scale on GPU-render, sw-render doesn't help). The wins are NOT in steps/s on this box — they're architectural:

| lever | mechanism | payoff | effort |
|---|---|---|---|
| more GPU-render processes | parallel containers | none (GPU render-queue contention) | n/a |
| software rendering | render on CPU | none for full training (above) | n/a |
| **camera-off feature policy** | no rendering + low-dim MLP | **big**: cheaper per-step (skip render) + far fewer steps to learn control | moderate |
| **N-cars-in-one-world** | one physics/render ctx steps N agents | **big**: ~N× data per physics step | large (deepracer-env is single-agent) |
| camera-off **+** N-cars | N feature agents, no render, shared physics | **largest** for control-side RL | large |
| GPU-heavy server | — | ~none (GPU isn't the bottleneck) | $$ |
| many-core server | more parallel sim instances | modest | $$ |

The camera renders via a `libgazebo_ros_camera` plugin + `<sensor type="camera">` in the racecar URDF
(`simulation/urdf/deepracer/deepracer_single_cam.urdf`); camera-off = disable that sensor (bind-mountable sim
asset change) + a dr-gym feature-obs env. N-cars is blocked by deepracer-env's explicit single-agent design
(`rollout_agent_ctrl.py`: "we are not supporting multi-agent training for now"). **Plan:** camera-off first
(smaller; measures the rendering cost), N-cars as a larger spike if still needed.

## Caveats
- Workstation shared with other containers; absolute numbers may shift, but the
  qualitative conclusions (RTF cap ~4.5×, GPU-render no-scaling, sw-render 2-worker sweet spot) are robust.
- The 2-worker gain is CPU-core-bound: on a box with more free cores the sweet spot likely shifts higher;
  re-run the sweep there. 7-worker effective RTF remains unmeasured (sim barely started within 120 s).

## Next
- **Done:** software-render multi-instance sweep ⇒ 2-worker sweet spot (~83 steps/s, +50%).
- Benchmark **N-cars-in-one-world** (multi-robot namespacing) vs the 2-instance sw-render point — does one
  Gazebo world with N namespaced cars beat 2 separate sw-render stacks (shared physics/render context)?
- Then the Backlog-#3 auto-tuner over `(rtf × instances/cars × render-device)`.
