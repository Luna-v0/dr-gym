#!/usr/bin/env bash
# Phase-1 perception data collection (W-perception) — sweep the sim worlds and
# write one dataset shard per world, then train_perception.py concatenates them.
#
# Lane-reading is track-agnostic, so we collect across BOTH the curriculum's
# train worlds AND the held-out eval worlds: the perception net seeing held-out
# geometry doesn't leak any *policy* advantage (it's a separate supervised model
# that just learns to read the lane). Reserved physical tracks (reInvent2019,
# Oval) are excluded per docs/eval-protocol.md.
#
# Needs the sim (a free Gazebo). Run AFTER D3 finishes:
#     bash scripts/collect_perception_all.sh
# Then:
#     uv run python experiments/train_perception.py \
#         --data artifacts/perception/*.npz --epochs 40 \
#         --out artifacts/perception/perception_net.pt
set -euo pipefail
cd "$(dirname "$0")/.."

WORLDS=(Spain_track Monaco Austin arctic_pro caecer_gp Bowtie_track jyllandsringen_pro penbay_pro)
EPISODES="${EPISODES:-20}"
EPSILON="${EPSILON:-0.25}"
OUTDIR="${OUTDIR:-artifacts/perception}"
mkdir -p "$OUTDIR"

for w in "${WORLDS[@]}"; do
  echo "=== collecting $w ($EPISODES episodes, eps=$EPSILON, DR on) ==="
  uv run --no-sync python scripts/collect_perception_data.py \
    --world "$w" --episodes "$EPISODES" --epsilon "$EPSILON" \
    --out "$OUTDIR/${w}.npz"
done

echo "=== done. Train with: uv run python experiments/train_perception.py --data $OUTDIR/*.npz ==="
