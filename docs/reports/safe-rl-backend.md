# Safe-RL backend — OmniSafe vs alternatives · `[DISS]` · 2026-06-22

Decision input for **D9**: which constrained-RL backend, and does OmniSafe let us change the network
architecture cleanly?

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
