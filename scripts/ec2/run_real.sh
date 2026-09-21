#!/usr/bin/env bash
# Run the real-data evaluation on EC2: smoke first, then the full paper run.
# Run from the repository root on the server:
#   bash scripts/ec2/run_real.sh smoke   # capped, fast plumbing check
#   bash scripts/ec2/run_real.sh full    # the real run (Tables 5/6)
set -euo pipefail

MODE="${1:-full}"
REPO_ROOT="${AEGIS_REPO:-$(pwd)}"
CONFIG="${AEGIS_CONFIG:-configs/ec2_real_evaluation.yaml}"
OUT="${AEGIS_OUT:-outputs/real}"

cd "$REPO_ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

if [ ! -f "$CONFIG" ]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

case "$MODE" in
  smoke)
    python scripts/run_real_experiment.py evaluate \
      --config "$CONFIG" --output "$OUT.smoke" --backend dummy --limit 40
    echo "Smoke outputs in $OUT.smoke (dummy judge — do NOT report these)."
    ;;
  full)
    python scripts/run_real_experiment.py evaluate \
      --config "$CONFIG" --output "$OUT"
    python scripts/make_plots.py \
      --input "$OUT/real_evaluation_summary.csv" \
      --output "$OUT/asr_vs_f.png" --real
    echo "Real outputs + provenance in $OUT. is_paper_result=False until verified."
    echo "Next: fill audits/result_integrity_audit.md and the Table 5/6 placeholders."
    ;;
  *)
    echo "Usage: bash scripts/ec2/run_real.sh [smoke|full]" >&2
    exit 1
    ;;
esac