"""Phase 2 — the experiment that runs AFTER the reward search.

Takes the **best reward** found by `experiments/reward_search.py` (the Optuna
study `reward_search`) and runs the full end-to-end PPO training with it:
full 4M-step budget + `ACL` + Domain Randomization/ADR +
random valid-start/direction. This is `end_to_end_ppo.py` with `reward` replaced
by the search winner — closing the loop reward-search → end-to-end.

Why this is the right "next experiment": the offline filter showed the D3
fast-crash is partly an *optimization* problem (the policy can't explore
cornering), not only reward shape. So the fix is the **combination** — the best
progress-normalized reward (from the search) PLUS DR + random-start (state
coverage so the policy actually learns to corner). This run tests that combo.

    uv run --no-sync python experiments/phase2_from_search.py
    # needs GYM_DR_DEEPRACER_ENV_SRC set for random_start/direction (patched sim)
"""
from __future__ import annotations

import optuna

from gym_dr import Study
from gym_dr.rewards import make_progress_reward, make_weighted_reward
from experiments.end_to_end_ppo import experiment as base_experiment

STUDY_NAME = "reward_search"
STORAGE = "sqlite:///optuna.db"


def reward_from_params(params: dict):
    """Rebuild the reward callable from a trial's params (mirrors
    reward_search.search_space)."""
    family = params.get("reward_family", "progress_complete")
    if family == "progress_complete":
        return make_progress_reward(
            step_penalty=params.get("pc_step_penalty", 0.3),
            completion_bonus=params.get("pc_completion_bonus", 100.0),
            center_bonus=params.get("pc_center_bonus", 0.1),
        )
    return make_weighted_reward(
        w_center=params.get("w_center", 1.0), w_speed=params.get("w_speed", 0.5),
        w_corner=params.get("w_corner", 0.5), w_align=params.get("w_align", 0.3),
        w_pace=params.get("w_pace", 0.3),
    )


def build_experiment():
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not completed:
        raise SystemExit("reward_search has no COMPLETE trials yet — let it finish first.")
    best = study.best_trial
    print(f"[phase2] best trial #{best.number}: value={best.value:.4f}")
    print(f"[phase2] params: {best.params}")

    reward = reward_from_params(best.params)
    overrides = {
        "name": "phase2_end_to_end",
        "reward": reward,
    }
    # carry the tuned PPO knobs the search found
    if "learning_rate" in best.params:
        overrides["trainer.kwargs.learning_rate"] = best.params["learning_rate"]
    if "ent_coef" in best.params:
        overrides["trainer.kwargs.ent_coef"] = best.params["ent_coef"]
    return base_experiment.with_overrides(**overrides)


if __name__ == "__main__":
    Study(build_experiment()).run()