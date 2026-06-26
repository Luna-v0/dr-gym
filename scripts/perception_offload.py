#!/usr/bin/env python3
"""Disk monitor + offload for the perception dataset (camera_cnn_dataset run).

The recorder writes per-episode .npz shards under the run's artifacts dir (on the
project/OS disk). This mover keeps that disk clear and the data safe by relocating
finished shards to fast NVMe, then archiving to the Pi over the LAN when NVMe fills.
NO Google Drive (unreliable rclone mount).

Pipeline per shard (only fully-written ``*.npz``; the recorder writes ``*.npz.tmp``
then renames, so we never grab a partial file):

  artifacts/<run>/perception_out  --move-->  /mnt/models/dr_perception
                                  --rsync-->  deepracer@<pi>:~/dr_perception   (when NVMe low)

Prompts (prints a loud banner) only if BOTH NVMe and the Pi are full/unreachable —
otherwise fully autonomous. Run in the background alongside the training:

    nohup uv run --no-sync python scripts/perception_offload.py \
        --src artifacts/camera_cnn_dataset/perception_out > /tmp/perception_offload.log 2>&1 &
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

PI = "deepracer@192.168.15.5"
PI_DIR = "~/dr_perception"


def free_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return -1.0


def _finished_shards(src: Path):
    # only fully-renamed shards (skip the ".tmp.npz" the recorder is mid-write on)
    return [p for p in sorted(src.rglob("*.npz")) if not p.name.endswith(".tmp.npz")]


def _pi_ok() -> bool:
    try:
        return subprocess.run(["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
                               PI, "true"], timeout=10).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True, help="recorder output dir to drain")
    ap.add_argument("--nvme", type=Path, default=Path("/mnt/models/dr_perception"))
    ap.add_argument("--interval", type=int, default=60, help="seconds between sweeps")
    ap.add_argument("--nvme-min-gb", type=float, default=25.0,
                    help="archive oldest NVMe shards to the Pi below this free space")
    args = ap.parse_args()
    args.nvme.mkdir(parents=True, exist_ok=True)
    print(f"[offload] src={args.src} -> nvme={args.nvme} -> pi={PI}:{PI_DIR}", flush=True)

    moved = archived = 0
    while True:
        args.src.mkdir(parents=True, exist_ok=True)
        # 1. drain the capture dir -> NVMe (keeps the OS disk clear)
        for shard in _finished_shards(args.src):
            rel = shard.relative_to(args.src)
            dst = args.nvme / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(shard), str(dst))
                moved += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[offload] move failed {shard}: {exc}", flush=True)

        # 2. NVMe low -> archive OLDEST shards to the Pi, then delete locally
        if free_gb(args.nvme) < args.nvme_min_gb:
            if _pi_ok():
                oldest = sorted(args.nvme.rglob("*.npz"), key=lambda p: p.stat().st_mtime)
                batch = oldest[:200]
                for shard in batch:
                    rel = shard.relative_to(args.nvme)
                    ptarget = f"{PI_DIR}/{rel.as_posix()}"
                    r = subprocess.run(
                        ["ssh", PI, f"mkdir -p {Path(ptarget).parent.as_posix()}"],
                        timeout=20)
                    if r.returncode != 0:
                        break
                    r = subprocess.run(["rsync", "-q", "--remove-source-files",
                                        str(shard), f"{PI}:{ptarget}"], timeout=120)
                    if r.returncode == 0:
                        archived += 1
                    else:
                        break
            elif free_gb(args.nvme) < 5.0:
                print("\n" + "!" * 70 +
                      f"\n[offload] NVMe < 5 GB free AND Pi unreachable — capture will "
                      f"start DROPPING shards.\n  Free space on /mnt/models or fix the Pi "
                      f"({PI}).\n" + "!" * 70, flush=True)

        if moved % 50 == 0 or archived:
            print(f"[offload] moved={moved} archived_to_pi={archived} "
                  f"nvme_free={free_gb(args.nvme):.0f}GB", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
