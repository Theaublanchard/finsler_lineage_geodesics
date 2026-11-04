#!/usr/bin/env bash
# Run all sweep configs sequentially against one dataset.
#
# Usage:
#   bash scripts/run_sweeps.sh
#   bash scripts/run_sweeps.sh --dataset cite
#   bash scripts/run_sweeps.sh --dataset zebrafish_cns --count 20
#   bash scripts/run_sweeps.sh --dataset mouse_blood --count 20
#   bash scripts/run_sweeps.sh cfm_finsler cfm_baseline   # run specific configs only
#
# Positional args (optional): names of configs to run (without .yaml extension).
# If none given, all configs in SWEEPS are run.
#
# The sweep is invoked via `uv run` so a pre-activated venv is not required.

set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
DATASET="zebrafish"
COUNT_ARG=""
SWEEP_DIR="/home/tblanchard/phd/finsler_lineage_geodesics/configs/sweeps"

# All available configs, in run order
ALL_SWEEPS=(cfm cfm_finsler mfm mfm_finsler)

# ── parse flags ───────────────────────────────────────────────────────────────
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --count)   COUNT_ARG="--count $2"; shift 2 ;;
        --*)       echo "Unknown flag: $1" >&2; exit 1 ;;
        *)         POSITIONAL+=("$1"); shift ;;
    esac
done

# Use positional args as the config list, or fall back to all
if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
    SWEEPS=("${POSITIONAL[@]}")
else
    SWEEPS=("${ALL_SWEEPS[@]}")
fi

# ── run ───────────────────────────────────────────────────────────────────────
echo "Dataset : $DATASET"
echo "Configs : ${SWEEPS[*]}"
echo "────────────────────────────────────────────────────────────────────────"

for name in "${SWEEPS[@]}"; do
    cfg="$SWEEP_DIR/$name.yaml"
    echo ""
    echo "=== $name ($cfg) ==="
    uv run python -m fingeo_supplement.scripts.sweep \
        --sweep_config "$cfg" \
        --dataset "$DATASET" \
        $COUNT_ARG
done

echo ""
echo "All sweeps complete."
