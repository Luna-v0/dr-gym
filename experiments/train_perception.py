"""Train the supervised perception net (W-perception) on a collected dataset.

CPU-friendly, sim-free: consumes the `.npz` from scripts/collect_perception_data.py
(obs frame stacks + frame-local labels) and fits gym_dr.perception.PerceptionNet,
reporting **per-feature MAE** so we can judge which quantities are actually
learnable from the camera (edges/lateral offset should be; yaw_rate is hardest).

    uv run python experiments/train_perception.py \
        --data artifacts/perception/oval.npz --epochs 30 \
        --out artifacts/perception/perception_net.pt

The learnability table this prints IS the W-perception deliverable's evidence: a
feature with high held-out MAE is one the actor should not lean on.
"""
from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", nargs="+", required=True,
                    help="one or more .npz datasets (concatenated)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/perception/perception_net.pt")
    args = ap.parse_args()

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from gym_dr.perception import PerceptionNet, signed_indices_for

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    obs_list, tgt_list, feat_names = [], [], None
    for path in args.data:
        d = np.load(path, allow_pickle=True)
        obs_list.append(d["obs"])
        tgt_list.append(d["targets"])
        names = [str(x) for x in d["features"]] if "features" in d else None
        if feat_names is None:
            feat_names = names
        elif names is not None and names != feat_names:
            raise SystemExit(f"feature mismatch across shards: {feat_names} vs {names}")
    obs = np.concatenate(obs_list, axis=0)
    tgt = np.concatenate(tgt_list, axis=0).astype(np.float32)
    if feat_names is None:  # legacy shard without names
        feat_names = [f"f{i}" for i in range(tgt.shape[1])]
    print(f"[train] dataset: obs {obs.shape}, targets {tgt.shape}")
    print(f"[train] features: {feat_names}")

    # deterministic shuffle + split
    idx = np.random.default_rng(args.seed).permutation(len(obs))
    obs, tgt = obs[idx], tgt[idx]
    n_val = max(1, int(len(obs) * args.val_frac))
    x_tr = torch.as_tensor(obs[n_val:], dtype=torch.float32)
    y_tr = torch.as_tensor(tgt[n_val:], dtype=torch.float32)
    x_va = torch.as_tensor(obs[:n_val], dtype=torch.float32)
    y_va = torch.as_tensor(tgt[:n_val], dtype=torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_channels = obs.shape[1]
    input_hw = (obs.shape[2], obs.shape[3])
    net = PerceptionNet(in_channels=in_channels, input_hw=input_hw,
                        n_outputs=len(feat_names),
                        signed_indices=signed_indices_for(feat_names)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = torch.nn.SmoothL1Loss()

    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=args.batch_size,
                        shuffle=True)
    for epoch in range(args.epochs):
        net.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
        net.eval()
        with torch.no_grad():
            pred = net(x_va.to(device)).cpu()
            mae = (pred - y_va).abs().mean(dim=0)
        print(f"[epoch {epoch:3d}] train_loss={running/len(x_tr):.4f}  "
              f"val_MAE_mean={mae.mean().item():.4f}")

    # final per-feature learnability table — THE deliverable: which features are
    # actually recoverable from the camera stack (low MAE) vs not (high MAE).
    print("\n[per-feature held-out MAE]  (lower = more learnable from camera)")
    for name, m in zip(feat_names, mae.tolist()):
        flag = "  <- hard" if m > 0.2 else ""
        print(f"  {name:18s} {m:.4f}{flag}")

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": net.state_dict(),
                "features": list(feat_names),
                "in_channels": in_channels, "input_hw": input_hw}, args.out)
    print(f"\n[train] saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
