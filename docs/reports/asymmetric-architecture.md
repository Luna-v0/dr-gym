# Asymmetric value network & image→policy distillation — feature study · `[REAL]` · 2026-06-22

Answers the question: *which features go where?* in the two privileged-information architectures for getting
a **camera-only** DeepRacer policy out of a sim that hands us **ground truth for free**. These are two
distinct paradigms — they are often conflated, so this report keeps them separate and gives the **concrete
feature partition** over our 26-key `reward_params`.

The hard guardrail throughout: the **deployed actor may only consume what the physical car can sense**
(camera + onboard proprioception). Everything privileged lives in modules that are *discarded at deployment*
(the critic) or used only to *produce a deployable student* (the teacher). Code: `gym_dr/perception.py`
(`perception_targets` = deployable, `privileged_state` = critic-only, `critic_state` = the concat).

---

## What the physical car can actually sense (the deployability line)
| signal | on the real car? | source |
|---|---|---|
| front camera frames (grayscale stack) | **yes** | ZED camera → the policy input |
| forward speed | **yes (proprioceptive)** | wheel encoders / VESC |
| yaw rate, accel | **yes (proprioceptive)** | onboard IMU |
| last action | **yes** | the controller commands it |
| lateral offset, heading error, edge distances | **no** — must be *inferred* from the camera | sim ground truth only |
| global pose / progress / waypoints / object geometry | **no, ever** | sim map only |

This table is the whole game: `speed` and `yaw_rate` are **proprioceptive**, so the actor keeps them at
deploy *without vision* — the genuinely vision-distilled quantities are **lateral offset, heading error, and
edge distances**. Global/map/contact state is never recoverable from a forward camera and is the privileged
set.

---

## Paradigm A — Asymmetric actor-critic (one-stage; Pinto et al. 2017)
Train actor and critic **together** with RL. The **actor** sees the deployable observation; the **critic(s)**
additionally see privileged ground-truth state. Because the critic is only used to compute advantages during
training and is **thrown away at deployment**, giving it privileged state is free and lowers value-estimation
variance → faster, more stable RL. This is the cheapest win and slots straight into our FSRL/Tianshou
backend, which already builds independent actor/critic nets (`gym_dr/trainers/fsrl_trainer.py`).

**Feature partition (implemented):**

- **Actor input** (`perception_targets`, deployable, 6 dims) — camera stack + proprioception + distilled:
  `lateral_offset`, `heading_error`, `dist_left_edge`, `dist_right_edge` (vision-distilled),
  `speed_norm`, `yaw_rate` (proprioceptive). Ego-relative, non-aliasing.
- **Critic input** (`critic_state` = `perception_targets ⊕ privileged_state`, 12 dims) — adds the 6 privileged
  extras the camera can't give:
  `progress_frac` (where on the lap), `curvature_ahead` (the bend before it's visible — map knowledge),
  `nearest_object_dist`, `offtrack`, `crashed`, `wheels_on_track` (exact contact/terminal flags).

**Why the COST critic benefits most.** Our safety costs (`gym_dr/costs.py`) are *defined on* near-edge /
near-object distances. Giving the **cost** critic the exact privileged distances (`dist_*_edge` ground truth,
`nearest_object_dist`) makes the constraint-value estimate far less noisy than reading them through a lossy
perception net — directly improving the FSRL PID-Lagrangian dual update. So in `FsrlTrainer` the two critics
(reward + cost, already separate) take `critic_state`; the actor takes the camera + deployable vector.

---

## Paradigm B — Privileged teacher → camera student distillation (two-stage; "Learning by Cheating")
The "**separated model that distills the information of the images to the policy**." Two stages:

1. **Teacher** — train a policy *directly on privileged state* (`critic_state` / full ground truth). It has no
   perception bottleneck, so it learns the *control* problem fast and well (this is also exactly what our
   `scripts/scripted_baseline.py` pure-pursuit controller is, in miniature — a privileged driver).
2. **Student** — a **camera-only** network distilled from the teacher: supervised/DAgger on the teacher's
   actions, with a **perception encoder** that compresses the image stack into the state-like features. Two
   flavours of "distill image → policy":
   - **Representation distillation** (what `PerceptionNet` does): train the encoder to *reproduce the
     privileged features* (`perception_targets` as labels), then feed those features to the policy. Modular,
     interpretable, and the per-feature MAE tells you what's learnable (`experiments/train_perception.py`).
   - **Policy distillation** (end-to-end): train the student to *match the teacher's action distribution*
     directly from pixels (DAgger to fix covariate shift). Higher ceiling, less interpretable.

**Why two stages help here:** the sim is the throughput bottleneck (~5× RTF cap, `docs/reports/throughput.md`),
and a privileged teacher trains far faster than camera RL — so Stage 1 is cheap, and Stage 2 distillation is
supervised (no sim rollouts needed beyond data collection, which `scripts/collect_perception_data.py` already
does). It also cleanly separates *control* failures from *perception* failures.

**Which to use?** Start with **A (asymmetric critic)** — near-zero extra cost, immediate variance reduction,
already fits the FSRL backend. Add **B (teacher→student)** if camera-only RL is too sample-hungry or the
generalization gap stays large after A. They compose: a privileged teacher can also warm-start the asymmetric
actor.

---

## Mapping the 26 `reward_params` keys to the partition
| key(s) | actor (deployable) | critic / teacher (privileged) | note |
|---|---|---|---|
| camera frames | ✅ raw input | (critic uses features, not pixels) | the only actor image input |
| `speed` | ✅ (proprioceptive) | ✅ | encoder/VESC on the car |
| `steering_angle` (last action) | ✅ (own command) | ✅ | feed as last-action |
| `distance_from_center`,`is_left_of_center`,`track_width` | → **distilled** to `lateral_offset`,`dist_*_edge` | ✅ exact | vision-learnable ego-relative |
| `heading` + `waypoints`/`closest_waypoints` (tangent) | → **distilled** to `heading_error` | ✅ exact | vanishing-point cue is learnable |
| `waypoints` ahead (curvature) | partial (visible road bend) | ✅ `curvature_ahead` | map knowledge beyond view |
| `progress`,`projection_distance`,`track_length`,`steps`,`x`,`y` | ❌ (aliased) | ✅ `progress_frac` | global pose unlearnable from 1 frame |
| `all_wheels_on_track`,`is_offtrack`,`is_crashed`,`is_reversed` | ❌ | ✅ exact flags | sharpen value near terminals |
| `objects_*` (location/distance/speed/heading/left_of_center) | ❌ (OA: partial via camera later) | ✅ `nearest_object_dist` + full | privileged object geometry |

`yaw_rate` is not a raw `reward_params` key — we derive it as a finite difference of `heading` for the label,
but on the car it comes from the **IMU**, so it stays on the actor side.

---

## Status & next
- **Built (this session):** the deployable/privileged feature split is now concrete code —
  `perception_targets` (actor, 6), `privileged_state` (critic extras, 6), `critic_state` (concat, 12), with
  `curvature_ahead` + object/contact privileged signals; tests in `tests/test_perception.py`.
- **Wired:** `FsrlTrainer` builds separate reward/cost critics — the asymmetric input (`critic_state`) is the
  small remaining wiring (emit it via `info`/a Tuple obs the actor masks) once we run on the live Dict env.
- **Next (sim-gated, after D3):** collect data, fit `PerceptionNet`, read the per-feature MAE to finalise the
  *actor's* feature diet (drop any with MAE > ~0.2), then choose A-only vs A+B from the post-A generalization
  gap.

## Sources
- Pinto, Andrychowicz, Welinder, Zaremba, Abbeel (2017) — *Asymmetric Actor Critic for Image-Based Robot Learning*.
- Chen, Zhou, Koltun, Krähenbühl (2019) — *Learning by Cheating* (privileged teacher → sensorimotor student).
- Lee, Hwangbo, Wellhausen, Koltun, Hutter (2020) — *Learning quadrupedal locomotion over challenging terrain* (teacher–student privileged).
- Miki et al. (2022) — *Learning robust perceptive locomotion* (belief-state encoder over privileged + exteroception).
- Kumar, Fu, Pathak, Malik (2021) — *RMA: Rapid Motor Adaptation* (online distillation of privileged env factors).
