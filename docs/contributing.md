# Contributing — how to work on this project

For humans and agents. Read `docs/open-questions.md` (the living decision log) at the start of a session.

## Setup

- **dr-gym** has its own Python 3.8 venv managed by `uv`. See `README.md` and `pyproject.toml` for the
  exact install; analysis tooling lives in the `analysis` dep group, on-device tooling in `optimize`.
- **deepracer-env** is the ROS 1 Noetic + Gazebo 11 simulator; it runs inside Docker. The project image is
  `my-deepracer-project:{cpu,gpu}`, built on the base `awsdeepracercommunity/deepracer-env:0.1-<arch>`.
  `bootstrap.sh` builds the project image; if the deepracer-env source advanced, rebuild the **base** first
  (otherwise the container's `DeepRacerEnv` silently lags).
- **deepracer-utils** is installed into dr-gym's `analysis` group from GitHub; it provides the trace
  readers and plots.
- **Deploy** (`[REAL]`) uses two dedicated py3.8 venvs, `.venv-ov-modern` and `.venv-ov-legacy` (their
  OpenVINO pins conflict with the main env). Setup is documented in the `optimize` group comment in
  `pyproject.toml`.

## Running

- **A training run:** write an experiment script that builds an `ExperimentConfig` and ends with
  `if __name__ == "__main__": train(experiment)`; run it with `uv run python app.py` (or
  `uv run python experiments/<name>.py`). The host orchestrator `docker run`s the sim container; you do not
  launch Gazebo by hand. See `docs/configuration.md` and `docs/system-overview.md`.
- **HPO:** a script with `base: ExperimentConfig` + `search_space(trial)` ending in `study(...)`. See
  `docs/hpo.md` and `experiments/*.py`.
- **Dry-run a config (no Docker):** `inspect(experiment)` in `gym_dr/app.py`.
- **Tests:** `uv run pytest`. ⚠️ Do **not** run two PPO-exercising suites concurrently — they thrash CPU.
- **Analysis:** `uv run --group analysis jupyter lab notebooks/trace_analysis.ipynb`.

## How to add things (reuse the existing patterns)

| To add… | Do this |
|---|---|
| a **reward** | Write a plain `def reward(params: dict) -> float` and pass it as `ExperimentConfig.reward`. No registry. See `gym_dr/rewards.py`. |
| a **trainer backend** | Implement the `Trainer` interface in `gym_dr/trainers/base.py` (see the SB3 impl). This is the extensibility contract (W-extensibility). |
| a **world strategy** | Subclass `WorldStrategy` in `gym_dr/worlds.py` (`training_chunks`, `evaluation_worlds`, `first_world`). Curriculum belongs here. |
| an **env / race type** | Add a factory under `gym_dr/envs/` that passes the right `config` to `DeepRacerEnv`, and re-export it. See `gym_dr/envs/time_trial.py`. |
| an **experiment** | A script with `ExperimentConfig` + `train`/`study`, tagged `[DISS]`/`[REAL]`/`[BOTH]`. |

## Conventions

- **Tag every doc, report, and experiment** `[DISS]` / `[REAL]` / `[BOTH]`, and name the
  sim-fidelity-vs-throughput and privileged-vs-deployable tradeoffs whenever they appear.
- **Docs** live in `docs/` (keep them current — stale docs are a bug). **Reports** (findings + one
  recommendation) live in `docs/reports/`. **Architecture decisions** live in `docs/adr/`.
- **Reproducibility:** pin versions; change one variable per experiment; multi-seed; enable the trace.
  Gazebo physics is non-deterministic even at a fixed seed — treat seeds as best-effort.
- **Guardrail:** the deployed actor must never depend on privileged sim state. "Passes on training tracks"
  ≠ "generalizes" — always report the held-out gap.

## Sign-off gates

Proceed autonomously on investigation, documentation, and low-risk fixes. **Get maintainer sign-off before
changing** the simulator stack, the default training configuration, the reward, or the architecture.

## Report format

```
# <Title> · <[DISS]/[REAL]/[BOTH]> · <date>
## Question / goal
## What I did
## Evidence
## Findings
## Recommendation
## Risks / open questions
## Next steps
```
