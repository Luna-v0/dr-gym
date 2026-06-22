#!/usr/bin/env bash
# Clean up artifacts/ — which the Docker sim writes as ROOT, so deletion needs
# root. This uses a throwaway alpine container for that, so you do NOT need sudo.
#
# Safe by default: no args = DRY RUN (reports only). Pick a mode to actually delete.
#
#   bash scripts/clean_artifacts.sh                       # dry run: sizes, families, largest runs
#   bash scripts/clean_artifacts.sh --prune              # per run: KEEP best_model/, final_model.*,
#                                                         #   run_config/model_metadata/training_status,
#                                                         #   reward_function.py, trace/, tensorboard/ —
#                                                         #   DROP initial/latest model, checkpoints/, export_bundle/, eval/
#   bash scripts/clean_artifacts.sh --delete GLOB...     # rm whole run dirs, e.g.:
#                                                         #   --delete 'object_avoidance_1_*' 'hpo_*' 'time_trail_min_speed_1_*'
set -u
ART="$(cd "$(dirname "$0")/.." && pwd)/artifacts"
[ -d "$ART" ] || { echo "no artifacts dir at $ART"; exit 1; }
root_rm() { docker run --rm -v "$ART:/a" alpine sh -c "$1"; }

case "${1:-}" in
  --prune)
    echo "before: $(du -sh "$ART" | cut -f1)"
    root_rm '
      for d in /a/*/; do
        [ -d "$d" ] || continue
        rm -rf "$d/checkpoints" "$d/export_bundle" "$d/eval" 2>/dev/null
        rm -f "$d/initial_model.zip" "$d/initial_model.model_metadata.json" \
              "$d/latest_model.zip"  "$d/latest_model.model_metadata.json" 2>/dev/null
      done'
    echo "after:  $(du -sh "$ART" | cut -f1)  (kept best_model + final_model + config + trace + TB per run)"
    ;;
  --delete)
    shift
    [ "$#" -gt 0 ] || { echo "usage: --delete GLOB [GLOB...]"; exit 1; }
    for g in "$@"; do echo "  $g -> $(ls -d "$ART"/$g 2>/dev/null | wc -l) dir(s)"; done
    printf "Delete these entirely? [y/N] "; read -r ans
    [ "$ans" = y ] || { echo "aborted"; exit 0; }
    echo "before: $(du -sh "$ART" | cut -f1)"
    for g in "$@"; do root_rm "rm -rf /a/$g"; done
    echo "after:  $(du -sh "$ART" | cut -f1)"
    ;;
  *)
    echo "=== artifacts total: $(du -sh "$ART" | cut -f1) ==="
    echo "--- run families (count) ---"
    ls -1 "$ART" 2>/dev/null | sed -E 's/_trial_[0-9]+$//; s/_seed[0-9]+$//; s/_rot[0-9].*$//' \
      | sort | uniq -c | sort -rn
    echo "--- 15 largest runs ---"
    du -sh "$ART"/* 2>/dev/null | sort -h | tail -15
    echo
    echo "DRY RUN — nothing deleted. Then run one of:"
    echo "  bash scripts/clean_artifacts.sh --prune"
    echo "  bash scripts/clean_artifacts.sh --delete 'object_avoidance_1_*' 'hpo_*'"
    ;;
esac
