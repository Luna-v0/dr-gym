# Evaluation protocol — held-out generalization & the clean-completion yardstick

`[DISS]` · the answer to scope-review finding #1 ("the metric can't tell good from bad").

## Success criterion
A policy that drives to the **end of every held-out eval track without leaving the track**, at a
**reasonable (non-minimum) speed**. Every "good/not good" verdict is measured against this.

## The yardstick: clean-completion rate
Read these (logged to TensorBoard + MLflow each evaluation):

| Metric | Meaning | Use |
|---|---|---|
| `eval/<world>_clean_completion_rate` | fraction of eval episodes that **finished the lap with zero off-track steps** | **primary** — the success criterion, per held-out world |
| `eval/clean_completion_rate` | mean of the above across held-out worlds | **primary aggregate** |
| `eval/<world>_completion_rate` | finished the lap (off-track allowed) | secondary — separates "didn't finish" from "finished dirty" |
| `eval/<world>_offtrack_resets` | eval episodes that ended off-track | diagnoses *why* clean-rate is low |
| `dr/ep_mean_speed` (eval) | average speed | the "reasonable, non-minimum speed" axis |
| `eval/<world>_mean_reward` | sum of `eval_reward` (use `clean_completion`) | model-selection / Optuna signal |

Per training rollout, `dr/ep_completed` and `dr/ep_completed_clean` give the same signal on the *training*
tracks.

**Do not** rank policies by `progress_safe` reward alone — it is dominated by `speed²` and penalizes
off-track by only −1.0, so it can't discriminate (see `docs/reports/scope-review.md`).

## Generalization gap
`gap = mean(train-track clean_completion_rate) − mean(held-out clean_completion_rate)`. Report it every run.
A small gap with a high held-out rate is the goal; a large gap means it overfits the training tracks.

## Held-out split
Use `OrderedSplit(train_worlds=..., eval_worlds=...)` (or `StochasticCurriculum`) with **disjoint,
geometrically distinct** sets. Record the exact split per run. Single-world runs (`SequentialRotation`)
evaluate on the *current* training world — **not** a held-out measurement; prefer a split strategy for any
generalization claim.

### Reserved physical tracks — do NOT train or in-loop-eval on these
The maintainer physically owns only **`reInvent2019_track`** and **`Oval_track`**. These (and similar
reInvent variants / `reinvent_base`) are **reserved for the out-of-loop physical-track evaluator**
(`scripts/eval_physical_tracks.py`) — they must **not** appear in `train_worlds` or `eval_worlds` of any sim
run, so the physical-track numbers stay a true held-out sim-to-real signal. Draw sim splits from the rest,
e.g. train = `Spain_track, Monaco, Austin, arctic_pro, caecer_gp`; held-out =
`Bowtie_track, jyllandsringen_pro, penbay_pro`.

## The eval reward
Set `eval_reward=clean_completion` so `best_model` selection and the Optuna objective track clean,
reasonably-fast laps. It is opt-in; making it the default is decision **D1**
(`docs/questions-for-maintainer.md`).

## Procedure
1. Fix a train/held-out split; train with `trace.enabled=True`, multi-seed.
2. Headline = `eval/clean_completion_rate` (held-out); also report the gap vs train tracks.
3. **Confirm by watching** — `eval_path_plots` overlays, trace path plots (`deepracer-utils`
   `plot_episode_path`), or VNC. The metric guides; it does not replace looking at the driving.
4. Compare configs/seeds with `rliable` (`deepracer-utils/deepracer/logs/rliable_utils.py`) on the
   completion-rate / progress metric.

## How it runs (where the numbers come from)
`MultiWorldEvalCallback` (`gym_dr/trainers/sb3/callbacks.py`) swaps the shared Gazebo env to each held-out
world, runs `n_eval_episodes`, mines each episode's `dr_episode` summary for completion/clean/off-track
counts, logs per-world + aggregate rates, then restores the training world. The episode flags come from
`gym_dr/metrics.py` (`dr/ep_completed`, `dr/ep_completed_clean`).
