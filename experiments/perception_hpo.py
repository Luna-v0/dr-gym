"""Optuna HPO for the supervised perception net (W-perception) — search the CNN
**architecture, kernel sizes, width, and optimization** that best regress the camera
4-frame stack to the 11 actor features, scored by **held-out (canonical VAL-track)
prize-feature MAE**.

This is the supervised counterpart to ``experiments/oracle_hpo.py`` (which HPOs the RL
policy). It does NOT use the RL ``gym_dr.study`` helper or sim — it is plain offline
torch regression, so it uses Optuna directly. Data plumbing is shared with the training
notebook via ``gym_dr.perception_data`` (canonical by-track split, windowed in-RAM loader).

What it searches (``sample_config``)
------------------------------------
* **architecture** — one of the DeepRacer presets (``shallow``/``standard``/``deep``) or a
  ``custom`` conv stack with searched **first-layer kernel & stride, depth, and filters**.
* ``features_dim`` (FC width), ``learning_rate``, ``batch_size``, ``weight_decay`` (AdamW).
* ``signed_curvature`` — whether to put ``curvature_ahead`` on a **tanh** head (the v1 bug
  fix: it is ~57% negative but ``signed_indices_for`` leaves it on a sigmoid → unfittable).
* ``proprio_weight`` — loss weight on the proprioceptive/temporal channels (the prize
  geometry channels are always weight 1.0).

Objective (minimised): mean held-out VAL-track MAE over the four vision-geometry prizes
(``lateral_offset, heading_error, dist_left_edge, dist_right_edge``). Per-epoch val MAE is
reported for a ``MedianPruner`` so weak configs die early. Frames are loaded to GPU **once**
and reused across all trials.

Run (NOT run automatically — create-only). Knobs are env vars:

    uv run python experiments/perception_hpo.py

    GYM_DR_PERC_HPO_TRIALS    (default 40)   total Optuna trials
    GYM_DR_PERC_HPO_SEARCH_EP (default 10)   epochs per trial during the search
    GYM_DR_PERC_HPO_FINAL_EP  (default 25)   epochs to retrain the winning config
    GYM_DR_PERC_HPO_SUBSAMPLE (default 40)   max TRAIN shards/track per trial (0/"" = all)
    GYM_DR_PERC_HPO_STACK     (default 4)    camera frame-stack depth (matches deploy)
    GYM_DR_PERC_HPO_JOBS      (default 1)    Optuna n_jobs (keep 1 — single GPU)
    GYM_DR_PERC_HPO_DB        (default artifacts/perception/perception_hpo.db)

The study is SQLite-backed and resumable (``load_if_exists``). The winner is retrained at
full scale and saved to ``artifacts/perception/perception_net_hpo.pt`` with a report at
``docs/reports/perception-hpo.md`` (best config + full per-feature MAE table + top trials).
"""
from __future__ import annotations

import copy
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "experiments")

from gym_dr.networks import DEEPRACER_CONV_PRESETS                       # noqa: E402
from gym_dr.perception import (ACTOR_FEATURES, PerceptionNet,           # noqa: E402
                               signed_indices_for)
from gym_dr.perception_data import (bucket_paths, cap_per_track,        # noqa: E402
                                    load_frames, make_windows)

# --------------------------------------------------------------------------- #
# config (env-tunable; create-only — see module docstring)
# --------------------------------------------------------------------------- #
TRIALS    = int(os.getenv("GYM_DR_PERC_HPO_TRIALS", "40"))
SEARCH_EP = int(os.getenv("GYM_DR_PERC_HPO_SEARCH_EP", "10"))
FINAL_EP  = int(os.getenv("GYM_DR_PERC_HPO_FINAL_EP", "25"))
SUBSAMPLE = int(os.getenv("GYM_DR_PERC_HPO_SUBSAMPLE", "40") or "0") or None
STACK     = int(os.getenv("GYM_DR_PERC_HPO_STACK", "4"))
JOBS      = int(os.getenv("GYM_DR_PERC_HPO_JOBS", "1"))
DB        = os.getenv("GYM_DR_PERC_HPO_DB", "artifacts/perception/perception_hpo.db")
STUDY     = os.getenv("GYM_DR_PERC_HPO_STUDY", "perception_cnn_hpo")
SEED      = int(os.getenv("GYM_DR_PERC_HPO_SEED", "0"))
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

FEATURES = list(ACTOR_FEATURES)
N_OUT = len(FEATURES)
PRIZE = ["lateral_offset", "heading_error", "dist_left_edge", "dist_right_edge"]
PRIZE_IDX = [FEATURES.index(f) for f in PRIZE]
PROPRIO_TEMPORAL = ["speed_mps", "yaw_rate", "long_accel", "lateral_velocity", "edge_closing_rate"]
OUT_MODEL = "artifacts/perception/perception_net_hpo.pt"
OUT_REPORT = "docs/reports/perception-hpo.md"

_DATA: dict = {}


# --------------------------------------------------------------------------- #
# data: load frames to GPU ONCE, reuse across trials
# --------------------------------------------------------------------------- #
def get_data() -> dict:
    """Load TRAIN/VAL/TEST frames once (TRAIN optionally subsampled per track), move the
    uint8 frame tensors to the device, and pre-build the stack-N window indices/targets.
    Cached in ``_DATA`` so every trial reuses the same resident tensors."""
    if _DATA:
        return _DATA
    print(f"[hpo] loading data once (stack={STACK}, subsample={SUBSAMPLE}/track) ...", flush=True)
    bk = bucket_paths()
    tr = cap_per_track(bk["TRAIN"], SUBSAMPLE, SEED)
    va = sorted(bk["VAL"])
    te = sorted(bk["TEST"])
    print(f"[hpo] shards: TRAIN={len(tr)} VAL={len(va)} TEST={len(te)}", flush=True)
    F_tr, Yf_tr, b_tr = load_frames(tr, tag="train")
    F_va, Yf_va, b_va = load_frames(va, tag="val")
    F_te, Yf_te, b_te = load_frames(te, tag="test")

    def pack(F, Yf, bounds):
        # frames stay in CPU RAM (works on small-VRAM GPUs); batches move per-step
        s, ti = make_windows(bounds, STACK)
        return dict(F=torch.from_numpy(F), starts=torch.from_numpy(s),
                    Y=torch.from_numpy(Yf[ti]), Yf=Yf, bounds=bounds)

    _DATA.update(train=pack(F_tr, Yf_tr, b_tr), val=pack(F_va, Yf_va, b_va),
                 test=pack(F_te, Yf_te, b_te),
                 base=np.abs(Yf_va[make_windows(b_va, STACK)[1]]
                             - Yf_tr[make_windows(b_tr, STACK)[1]].mean(0)).mean(0))
    gb = sum(_DATA[k]["F"].element_size() * _DATA[k]["F"].nelement()
             for k in ("train", "val", "test")) / 1e9
    print(f"[hpo] frames resident in CPU RAM: {gb:.2f} GB (batches move to {DEVICE})", flush=True)
    return _DATA


def _gather_gpu(F, starts, idx, stack):
    """Gather a ``(B, stack, 120, 160)`` float batch on DEVICE from CPU-resident frames:
    index uint8 on CPU, move uint8 (4x smaller than float) to GPU, then cast."""
    rows = starts[idx][:, None] + torch.arange(stack)[None, :]
    return F[rows].to(DEVICE, non_blocking=True).float()


@torch.no_grad()
def eval_mae(net, split, bs=2048) -> np.ndarray:
    net.eval()
    F, starts, Y = split["F"], split["starts"], split["Y"]
    n = starts.shape[0]
    se = torch.zeros(N_OUT, device=DEVICE)
    for i in range(0, n, bs):
        idx = torch.arange(i, min(i + bs, n))
        se += (net(_gather_gpu(F, starts, idx, STACK)) - Y[idx].to(DEVICE)).abs().sum(0)
    return (se / max(n, 1)).cpu().numpy()


# --------------------------------------------------------------------------- #
# search space
# --------------------------------------------------------------------------- #
def build_conv_spec(trial) -> tuple:
    """A DeepRacer preset, or a ``custom`` conv stack with searched first-layer kernel &
    stride, depth, and base filters. Only the first (one or two) layers downsample, so the
    120x160 input never collapses for sane choices; build_net still guards the rest."""
    kind = trial.suggest_categorical("arch", ["shallow", "standard", "deep", "custom"])
    if kind != "custom":
        return tuple(tuple(layer) for layer in DEEPRACER_CONV_PRESETS[kind])
    n_conv = trial.suggest_int("n_conv", 3, 4)
    kernel0 = trial.suggest_categorical("kernel0", [3, 5, 7, 8])
    stride0 = trial.suggest_categorical("stride0", [2, 4])
    filters0 = trial.suggest_categorical("filters0", [16, 32])
    kernels = [kernel0, 4, 3, 3]
    strides = [stride0, 2, 1, 1]
    return tuple((min(filters0 * (2 ** i), 128), kernels[i], strides[i]) for i in range(n_conv))


def sample_config(trial) -> dict:
    conv = build_conv_spec(trial)
    cfg = dict(
        conv_spec=conv,
        features_dim=trial.suggest_categorical("features_dim", [128, 256, 512]),
        lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
        weight_decay=trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
        signed_curvature=trial.suggest_categorical("signed_curvature", [True, False]),
        proprio_weight=trial.suggest_categorical("proprio_weight", [0.0, 0.1, 0.3, 1.0]),
    )
    # record the resolved arch so the retrain doesn't have to re-derive conditional params
    trial.set_user_attr("conv_spec", [list(l) for l in conv])
    return cfg


def build_net(cfg):
    """Construct PerceptionNet for a config; signed indices optionally include
    ``curvature_ahead`` (the v1 head fix). Raises on a degenerate conv stack."""
    signed = list(signed_indices_for(FEATURES))
    if cfg["signed_curvature"]:
        signed.append(FEATURES.index("curvature_ahead"))
    net = PerceptionNet(in_channels=STACK, conv_layers=tuple(cfg["conv_spec"]),
                        features_dim=cfg["features_dim"], n_outputs=N_OUT,
                        input_hw=(120, 160), signed_indices=tuple(sorted(set(signed))))
    return net


def loss_weights(cfg) -> "torch.Tensor":
    w = {f: 1.0 for f in FEATURES}
    for f in PROPRIO_TEMPORAL:
        w[f] = cfg["proprio_weight"]
    w["nearest_object_dist"] = 0.3
    return torch.tensor([w[f] for f in FEATURES], dtype=torch.float32, device=DEVICE)


# --------------------------------------------------------------------------- #
# train one config (best-by-prize ckpt); optionally return full per-feature MAE
# --------------------------------------------------------------------------- #
def train_eval(cfg, epochs, trial=None, full=False):
    import optuna
    data = get_data()
    tr, va = data["train"], data["val"]
    try:
        net = build_net(cfg).to(DEVICE)
    except Exception as exc:  # degenerate architecture -> prune
        raise optuna.TrialPruned(f"invalid arch {cfg['conv_spec']}: {exc}")

    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    huber = torch.nn.SmoothL1Loss(reduction="none")
    w = loss_weights(cfg)
    F, starts, Y = tr["F"], tr["starts"], tr["Y"]
    n, bs = starts.shape[0], cfg["batch_size"]
    g = torch.Generator()                                  # CPU generator (frames on CPU)
    best = (1e9, None)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=g.manual_seed(SEED + ep))
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = (huber(net(_gather_gpu(F, starts, idx, STACK)), Y[idx].to(DEVICE)) * w).mean()
            loss.backward()
            opt.step()
        sched.step()
        mae = eval_mae(net, va)
        prize = float(mae[PRIZE_IDX].mean())
        if prize < best[0]:
            best = (prize, copy.deepcopy(net.state_dict()))
        if trial is not None:
            trial.report(prize, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
    if not full:
        return best[0]
    net.load_state_dict(best[1])
    return best[0], net, eval_mae(net, va), eval_mae(net, data["test"])


def objective(trial) -> float:
    return train_eval(sample_config(trial), SEARCH_EP, trial=trial)


# --------------------------------------------------------------------------- #
# study + winner retrain/report
# --------------------------------------------------------------------------- #
def _table(val, test, base) -> str:
    L = ["| feature | val MAE | skill | test MAE | usable |", "|---|---|---|---|---|"]
    for i, f in enumerate(FEATURES):
        sk = f"{1 - val[i]/base[i]:+.2f}" if base[i] > 1e-6 else "n/a"
        ok = "✅" if (val[i] < 0.10 and base[i] > 1e-6 and (1 - val[i]/base[i]) > 0.15) else "—"
        L.append(f"| `{f}` | {val[i]:.4f} | {sk} | {test[i]:.4f} | {ok} |")
    return "\n".join(L)


def write_report(study, val, test, base, best_cfg):
    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
    bt = study.best_trial
    top = sorted([t for t in study.trials if t.value is not None], key=lambda t: t.value)[:10]
    rows = ["| # | prizeMAE | arch | feat_dim | lr | batch | wd | signedΩ | propW |",
            "|---|---|---|---|---|---|---|---|---|"]
    for t in top:
        p = t.params
        rows.append(f"| {t.number} | {t.value:.4f} | {p.get('arch')} | "
                    f"{p.get('features_dim')} | {p.get('lr'):.1e} | {p.get('batch_size')} | "
                    f"{p.get('weight_decay'):.1e} | {p.get('signed_curvature')} | "
                    f"{p.get('proprio_weight')} |")
    md = (f"# Perception CNN — HPO results\n\n"
          f"Optuna study `{STUDY}` ({len([t for t in study.trials if t.value is not None])} "
          f"complete / {len(study.trials)} trials). Objective = held-out **VAL-track** mean MAE "
          f"over the four vision-geometry prizes. Search epochs={SEARCH_EP}, "
          f"subsample={SUBSAMPLE}/track; winner retrained {FINAL_EP} epochs at full TRAIN.\n\n"
          f"## Best config (prize MAE {study.best_value:.4f})\n```\n"
          + "\n".join(f"{k} = {v}" for k, v in best_cfg.items()) + "\n```\n\n"
          f"## Winner — full held-out per-feature MAE\n\n" + _table(val, test, base) + "\n\n"
          f"*skill = 1 − MAE/base; ✅ = MAE<0.10 & skill>0.15.*\n\n"
          f"## Top trials\n\n" + "\n".join(rows) + "\n")
    with open(OUT_REPORT, "w") as fh:
        fh.write(md)
    print(f"[hpo] report -> {OUT_REPORT}", flush=True)


def main() -> int:
    import optuna
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(os.path.dirname(DB) or ".", exist_ok=True)
    get_data()  # load once up front (fail fast before the study)

    study = optuna.create_study(
        study_name=STUDY, direction="minimize",
        storage=f"sqlite:///{DB}", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))
    print(f"[hpo] study={STUDY} storage={DB} trials={TRIALS} device={DEVICE}", flush=True)
    study.optimize(objective, n_trials=TRIALS, n_jobs=JOBS, gc_after_trial=True)

    print(f"\n[hpo] best prize MAE={study.best_value:.4f}\n[hpo] best params={study.best_params}",
          flush=True)
    # retrain the winner at full scale and save
    best_cfg = dict(
        conv_spec=tuple(tuple(l) for l in study.best_trial.user_attrs["conv_spec"]),
        features_dim=study.best_params["features_dim"], lr=study.best_params["lr"],
        batch_size=study.best_params["batch_size"], weight_decay=study.best_params["weight_decay"],
        signed_curvature=study.best_params["signed_curvature"],
        proprio_weight=study.best_params["proprio_weight"])
    print(f"[hpo] retraining winner {FINAL_EP} epochs ...", flush=True)
    prize, net, val_mae, test_mae = train_eval(best_cfg, FINAL_EP, full=True)
    print(f"[hpo] winner full-retrain prize MAE={prize:.4f}", flush=True)

    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    signed = build_net(best_cfg)._signed_mask.nonzero().flatten().tolist()
    torch.save({"state_dict": net.state_dict(), "features": FEATURES, "in_channels": STACK,
                "stack": STACK, "input_hw": (120, 160), "conv_spec": best_cfg["conv_spec"],
                "features_dim": best_cfg["features_dim"], "signed_indices": signed,
                "val_mae": dict(zip(FEATURES, val_mae.tolist())),
                "test_mae": dict(zip(FEATURES, test_mae.tolist())),
                "hpo": {"study": STUDY, "best_params": study.best_params,
                        "best_prize_mae": study.best_value}}, OUT_MODEL)
    print(f"[hpo] model -> {OUT_MODEL}", flush=True)
    write_report(study, val_mae, test_mae, get_data()["base"], best_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
