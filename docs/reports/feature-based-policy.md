# Feature-based policy & the sim2real-by-extractor decomposition · `[REAL]` · 2026-06-22

A clean architecture (maintainer's framing): instead of one camera→action network, split the system at the
**feature vector**:

```
   camera stack ──g(·)──►  features  ──π(·)──►  action
                 extractor            policy
   (the ONLY sim2real-sensitive part)   (state-based: sim==real by construction)
```

- `π(features) → action` is a **state-based** policy. A feature like "lateral offset = 0.2 m" or "edge
  closing rate" means the *same thing* in sim and on the real car, so **π transfers perfectly** — provided it
  is fed accurate features. It never sees a pixel.
- `g(camera) → features` is the extractor (the supervised/`PerceptionNet` from `docs/reports/perception.md`).
  It is the **only** component that touches raw pixels, so the **sim-to-real appearance gap is fully
  localized to g** — and can be hardened independently (more data, DR, the multi-view contrastive mode) without
  ever retraining π.

This is the modular form of teacher→student distillation, with the **feature vector as the contract** between
perception and control.

## The two tests to add (future work)

### Test 1 — oracle-feature policy (the upper bound + the teacher)
Train PPO with the **ground-truth feature vector as the observation** (built from `reward_params` via
`gym_dr.perception.enrich_reward_params` / `all_targets` — the same `ALL_FEATURES` the extractor predicts).
No camera, no CNN.

- **Answers:** *are these features sufficient to drive well?* If π can't drive on **perfect** features, the
  feature *set* is incomplete and no extractor will rescue it → go back and add features (this is the direct
  test of the feature design in `docs/reports/perception.md`).
- **Bonus:** trains ~orders of magnitude faster (low-dim obs, no CNN) → cheap HPO / many seeds, and it
  becomes the **teacher** for distillation.
- **Ceiling caveat:** π's performance is capped by the feature set, by construction — which is exactly the
  signal we want.

### Test 2 — extractor-in-the-loop (the perception penalty)
Run the *same* π fed `g`'s **predicted** features from the camera (`π(g(camera))`).

- **Answers:** *how much does imperfect perception cost?* The gap `score(Test 1) − score(Test 2)` is the
  perception penalty. Improve `g` independently and re-measure — π is fixed.
- **Robustness refinement (bake in):** train π in Test 1 with **noise injected on the feature observation,
  calibrated to g's per-feature held-out MAE** (from `experiments/train_perception.py`). Then π is already
  robust to realistic perception error before it meets `g`, so Test 2 doesn't collapse from distribution
  shift. This is domain randomization applied to the *feature* observation rather than the pixels — cheap and
  principled (the MAE table gives the exact noise scale per feature).

## Why this is worth it
- **Decouples perception from control** — debug each separately; a control failure and a perception failure
  no longer look alike.
- **Sim2real isolated to g** — the riskiest `[REAL]` component is the only thing the multi-view contrastive
  mode / DR needs to harden; π is invariant.
- **Throughput** — the feature-based policy sidesteps the rendering bottleneck entirely (no camera needed once
  features are computed from `reward_params`), so control-side HPO is fast (`docs/reports/throughput.md`).

## Enabling pieces — status
- **Built (this session):** `enrich_reward_params(params, prev)` exposes the derived features
  (`ALL_FEATURES`) as **reward-function arguments** and as the feature-based observation vector; tested.
  `all_targets` / `signed_indices_for` already adapt the label/observation set.
- **To build (the tests):** a `feature_state` env mode (swap the Dict camera obs for a `Box` feature vector,
  read from the reward tap like `scripts/collect_perception_data.py` does, tracking `prev_params` for the
  dynamic features) + feature-noise DR + the two experiment configs. **Gated on Phase 1** (need g and its MAE
  to set the noise) and a free sim.

## Sequencing in the broader plan
1. D3 (no-DR baseline) finishes → frees the sim.
2. **Phase 1** — collect data, fit `g`, read the MAE learnability table (`docs/reports/perception.md`).
3. **Test 1** — oracle-feature PPO with MAE-calibrated feature noise (fast, sim-light).
4. **Test 2** — `π(g(camera))`; report the perception penalty.
5. Then the `[REAL]` hardening of `g` (multi-view contrastive, `docs/reports/asymmetric-architecture.md`).
