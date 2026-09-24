#!/usr/bin/env bash
# Run the real-data evaluation on a GPU server: smoke first, then the full paper run + ablation.
# Run from the repository root on the server:
#   bash scripts/ec2/run_real.sh smoke   # capped, fast plumbing check (dummy judge)
#   bash scripts/ec2/run_real.sh full    # the real run (Tables 5/6)
#   bash scripts/ec2/run_real.sh ablate  # component ablations on real verdicts (Table 6)
#
# Give every benchmark its own AEGIS_OUT: all runs write the same file names, so a shared
# directory overwrites the previous benchmark's results.
#   AEGIS_OUT=outputs/real_dan bash scripts/ec2/run_real.sh full
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

# Preflight before spending GPU time. A FAIL means the run would abort or produce an
# uninterpretable table (no benign slice for calibration, payloads longer than the context window,
# wrong served model, ...). Override deliberately with AEGIS_SKIP_PREFLIGHT=1.
run_preflight() {
  if [ "${AEGIS_SKIP_PREFLIGHT:-0}" = "1" ]; then
    echo "==> Preflight skipped (AEGIS_SKIP_PREFLIGHT=1 — record why)."
    return 0
  fi
  echo "==> Preflight: python scripts/check_ready.py --config $CONFIG"
  python scripts/check_ready.py --config "$CONFIG"
}

case "$MODE" in
  smoke)
    python scripts/run_real_experiment.py evaluate \
      --config "$CONFIG" --output "$OUT.smoke" --backend dummy --limit 40
    echo "Smoke outputs in $OUT.smoke (dummy judge — do NOT report these; their verdicts are"
    echo "cached under <cache_dir>_dummy, never in the cache a real run reuses)."
    ;;
  full)
    run_preflight
    python scripts/run_real_experiment.py evaluate \
      --config "$CONFIG" --output "$OUT"
    python scripts/make_plots.py \
      --input "$OUT/real_evaluation_summary.csv" \
      --output "$OUT/asr_vs_f.png" --real
    echo "Real outputs + provenance in $OUT. is_paper_result=False until verified."
    echo "NEXT, BEFORE READING THE TABLE: check attack_effect in $OUT/real_evaluation_summary.csv."
    echo "  attack_effect = 0 for every Aegis rule means the attack never changed a decision, so"
    echo "  the comparison cannot separate the methods - raise experiment.attack_kwargs and re-run."
    echo "Then: bash scripts/ec2/run_real.sh ablate"
    ;;
  ablate)
    python scripts/run_real_experiment.py ablate \
      --config "$CONFIG" --output "$OUT"
    echo "Ablation rows + provenance in $OUT/real_ablation.csv. Reuses the honest cache."
    ;;
  *)
    echo "Usage: bash scripts/ec2/run_real.sh [smoke|full|ablate]" >&2
    exit 1
    ;;
esac
