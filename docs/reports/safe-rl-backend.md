# Safe-RL backend — OmniSafe vs alternatives · `[DISS]` · 2026-06-22

Decision input for **D9**: which constrained-RL backend, and does OmniSafe let us change the network
architecture cleanly?

**DECIDED (2026-06-22): adopt FSRL `PPOLagAgent`** (PID-Lagrangian PPO in one algorithm). **Built so far:**
the cost→`info["cost"]` bridge (`CostInfoWrapper`, reusing the graded `gym_dr/costs.py` tap, tested), an
`FsrlTrainer` backend scaffold (`gym_dr/trainers/fsrl_trainer.py`, against the Trainer contract), and the
Safety-Gymnasium validation script (`scripts/validate_fsrl_safetygym.py`).

**VALIDATED (2026-06-22): PPO-Lag runs end-to-end on Safety-Gymnasium and exhibits correct constrained
behaviour.** Setup that worked: a *separate* `.venv-safe` on **Python 3.10** (3.11 fails — safety-gymnasium
pins `pygame==2.1.0`, which has no 3.11 wheel and won't build from source here; 3.10 gets the prebuilt
wheel), `uv pip install fast-safe-rl safety-gymnasium` (tianshou 0.5.1). **One integration bug, fixed:**
Safety-Gymnasium tasks return a CMDP **6-tuple** `(obs, reward, cost, terminated, truncated, info)`, but
Tianshou/FSRL (and gymnasium's passive checker) expect the 5-tuple with cost in `info["cost"]`. Added a
`_CostToInfo` wrapper in the validation script + built envs via `safety_gymnasium.make` (avoids the passive
checker rejecting the 6-tuple). **This is the same `info["cost"]` contract as our DeepRacer
`CostInfoWrapper`** — so the bridge is proven on both sides. Result on `SafetyPointGoal1-v0` (`cost_limit=10`, 20 epochs, ~10 min on CPU):
the PID multiplier drives episode cost down under the limit while reward rises — **the ideal CMDP outcome.
Final: `best_reward` 17.5 (from negative early on) with `best_cost` 9.0 (≤ limit 10)**; mid-run the multiplier
visibly trades off (cost dipped to 3 at epoch 5 when reward was still negative, then both re-balanced).
Textbook Lagrangian dynamics ⇒ **the algorithm is trustworthy**. (`PPOLagAgent` verified kwargs:
`cost_limit, device, seed, lr, hidden_sizes, target_kl, gamma`; the DeepRacer camera path uses the lower-level
`PPOLagrangian` policy with a Tianshou CNN `preprocess_net` + separate reward/cost `Critic`s — see
`gym_dr/trainers/fsrl_trainer.py`.)

**Next:** finalize the Tianshou CNN for camera obs in `FsrlTrainer` (+ the asymmetric cost-critic, see
`docs/reports/perception.md`) → DeepRacer constrained run with `cost_limit` from the empirically-logged
`dr/ep_mean_cost` (D3 is logging it now).

## Direct answer on OmniSafe + architecture
OmniSafe is modular (Adapter/Wrapper patterns) and configures actor/critic via its **own** `ModelConfig` +
model registry. Standard MLP/CNN nets are config-driven, but a **bespoke feature extractor like our
`DeepRacerCNN`** (separate actor/critic towers, Dict camera obs, 4-frame stack, `normalize_images=False`)
needs a custom *registered* OmniSafe actor/critic class. That is **more** work than our current SB3
`policy_kwargs={features_extractor_class=DeepRacerCNN}`, which already works. Adopting OmniSafe means
re-expressing the whole stack — env, the custom CNN, world hot-swap, curriculum, the trace/metrics, the eval
callbacks — in OmniSafe's abstractions. That re-port (plus dependency weight / rigidity for non-standard
envs) is the likely source of the "problematic things" you've heard.

## The technical crux (from the research)
Safe RL benefits from **separate optimizers** for policy / reward-critic / cost-critic: reward and cost have
different scales, so a single optimizer (SB3's design) lets the cost-value gradient dominate and destabilise
updates. OmniSafe / SafePO use separate optimizers; an SB3-based Lagrangian must add a **separate cost
critic + its own optimizer** (a custom algorithm subclass).

## Options
| Library | Maturity | Algorithms | Safety-Gymnasium | Custom-arch for *our* CNN | Reuses our SB3 stack |
|---|---|---|---|---|---|
| **OmniSafe** (PKU) | strong, JMLR'24, good docs | many (PPO/TRPO/SAC/DDPG/TD3-Lag, CPO, FOCOPS, P3O…) | native | medium — custom registered model | **No** (re-port env+model+curriculum+trace) |
| **FSRL** (liuzuxin, PyTorch/Tianshou) | good, lighter | PPO-Lag, CPO, … | yes | medium | No (Tianshou) |
| **SafePO** (PKU) | benchmark/pedagogical, 16 algos | many | native | medium | No |
| **SB3 + custom Lagrangian** | we own it | PPO-Lag / PID-Lag (build) | via a thin adapter | **trivial — already works** | **Yes** (everything reused) |
| safety-starter-agents (OpenAI) | unmaintained, TF1 | PPO-Lag/CPO/TRPO-Lag | — | — | No |

## PID-Lagrangian, turnkey — who joins it with PPO
- **FSRL `PPOLagrangian` = PPO + PID-Lagrangian in ONE algorithm** (adaptive PID multiplier, Stooke et al.) —
  confirmed in its docs. So for turnkey PID-Lagrangian-PPO it's a single agent; nothing to assemble.
- **OmniSafe splits them:** `PPOLag` (plain Lagrangian = integral-only multiplier) vs `PPOPID` (the PID
  variant) — a one-line config switch.
- **TorchRL / RLlib:** no prominent turnkey PID-Lagrangian-PPO.
So the realistic *adopt* options for PID-Lagrangian-PPO are **FSRL (`PPOLagrangian`)** or **OmniSafe
(`PPOPID`)**; the *build* path (SB3) means implementing the PID multiplier ourselves.

## Recommendation — hybrid (adopt where cheap, build where integration friction is high)
1. **Validate the *algorithm* on Safety-Gymnasium with OmniSafe (turnkey).** Confirms PPO-Lag / PID-Lag
   reproduces known safe-RL behaviour (trust), with zero custom code — exactly what a standard, well-tested
   library is best for. This is the base-prompt's "Safety-Gymnasium study."
2. **For DeepRacer, build an SB3-based PPO-/PID-Lagrangian `Trainer` backend** — a `SafeSb3Trainer` that
   reuses our `DeepRacerCNN`, Dict obs, frame stack, world hot-swap, `StochasticCurriculum`, trace, eval
   callbacks (zero re-port), adding a **cost critic + separate optimizer + a PID-Lagrangian dual update** on
   our graded `gym_dr/costs.py`. Full architecture control (your requirement), and it slots into the
   existing `Trainer` ABC (Backlog #4 / W-extensibility).

This keeps OmniSafe as the **trusted reference/benchmark** and avoids re-porting our heavily-customised stack.
If you'd rather not maintain a custom Lagrangian at all, the alternative is going full OmniSafe and paying the
re-port — but given how custom our CNN/env/curriculum already are, and your wish for clean architecture
control, the hybrid is the better fit.

## Effort
- Safety-Gymnasium validation via OmniSafe: **low** (install + run a turnkey PPO-Lag on `SafetyPointGoal1`).
- `SafeSb3Trainer` (cost-critic head + separate optimizer + PID dual): **moderate**, but reuses all our
  callbacks/metrics/curriculum; the dual-update + separate optimizer is the only genuinely new code.

## Other libraries surveyed (2025–26)
None displace the recommendation, but for completeness:
- **FSRL** (online, Tianshou) + **OSRL** (offline) — actively maintained, lighter than OmniSafe; the
  runner-up if you'd rather *adopt* than build the SB3 Lagrangian.
- **SafePO** (PKU) — 16 algorithms, benchmark/pedagogical.
- **safe-control-gym** (utiasDSL) — safe learning-based *control* + RL (more control-theoretic).
- **Bullet-Safety-Gym** — constrained-RL envs (PyBullet), supports CPO.
- **SafeRL-Kit** (5 methods) and **chauncygu/Safe-RL-Baselines** (curated impls) — reference.
- **HASARD** — a **vision-based** safe-RL benchmark for embodied agents; closer to camera-based DeepRacer
  than Safety-Gymnasium's point-mass tasks, so a useful *second* validation target.

Net: the hybrid stands. If "build an SB3 Lagrangian" is unwanted, **FSRL** is the lightest adopt-alternative;
validate on **Safety-Gymnasium** (+ optionally **HASARD** for vision) before DeepRacer.

## Sources
- [OmniSafe (JMLR 2024)](https://www.jmlr.org/papers/v25/23-0681.html) · [arXiv](https://arxiv.org/abs/2305.09304) · [GitHub/docs](https://github.com/PKU-Alignment/omnisafe) · [features](https://www.omnisafe.ai/en/latest/start/features.html)
- [FSRL — fast safe RL (PyTorch/Tianshou)](https://github.com/liuzuxin/FSRL)
- [Optimizer architecture in SB3 for safe RL (single vs separate optimizers)](https://medium.com/@kwon1122/optimizer-architecture-in-stable-baselines3-for-safe-reinforcement-learning-64e3560749f2)
- [Safety-Gymnasium (benchmark)](https://openreview.net/forum?id=WZmlxIuIGR)
