#!/usr/bin/env bash
# One-time provisioning of a GPU server for the real-data runs.
# Run from the repository root on the server:
#
#   bash scripts/ec2/setup.sh
#   PYTHON=python3.12 bash scripts/ec2/setup.sh     # force a specific interpreter
#
# Works on Ubuntu 22.04 and 24.04. The directory name says "ec2" for historical reasons; nothing
# in this script is AWS-specific.
#
# E blocker fixed here: the previous version ran `apt-get install python3.11`, which does NOT
# exist in the default Ubuntu 22.04 (3.10) or 24.04 (3.12) archives, so provisioning failed on a
# fresh server. The interpreter is now detected first and only installed if genuinely missing.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MIN_MINOR=11   # pyproject requires-python >= 3.11

version_ok() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)' >/dev/null 2>&1
}

pick_python() {
  local candidate
  # ${PYTHON:-} is intentionally unquoted so an empty value expands to nothing.
  for candidate in ${PYTHON:-} python3.12 python3.13 python3.11 python3; do
    [ -n "$candidate" ] || continue
    if command -v "$candidate" >/dev/null 2>&1 && version_ok "$candidate"; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

install_python() {
  echo "==> No Python >= 3.${MIN_MINOR} found; installing one"
  sudo apt-get update
  if apt-cache policy python3.12 2>/dev/null | grep -q Candidate; then
    # Ubuntu 24.04+: 3.12 is in the default archive.
    sudo apt-get install -y --no-install-recommends python3.12 python3.12-venv
  else
    # Ubuntu 22.04: the archive only has 3.10, so 3.11 comes from the deadsnakes PPA.
    echo "==> python3.12 not in the archive; adding the deadsnakes PPA for python3.11"
    sudo apt-get install -y --no-install-recommends software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends python3.11 python3.11-venv
  fi
}

report_host() {
  echo "==> Host preflight"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
    nvidia-smi 2>/dev/null | grep -i "CUDA Version" || true
  else
    echo "    nvidia-smi: NOT FOUND - install the NVIDIA driver before serving a backbone"
  fi
  free -g | head -n 2 || true
  df -h "$REPO_ROOT" | tail -n 1 || true
  grep -E '^(NAME|VERSION)=' /etc/os-release || true
  echo "    needed for a real run: GPU >= 24 GB VRAM, RAM >= 32 GB, >= 60 GB free disk,"
  echo "    and ports 8001-8003 free (one per backbone; homogeneous runs need only 8001)."
}

report_host

PY_BIN="$(pick_python || true)"
if [ -z "${PY_BIN:-}" ]; then
  install_python
  PY_BIN="$(pick_python || true)"
fi
if [ -z "${PY_BIN:-}" ]; then
  echo "ERROR: no Python >= 3.${MIN_MINOR} available and automatic install failed." >&2
  echo "       Install one, then re-run:  PYTHON=/path/to/python3 bash scripts/ec2/setup.sh" >&2
  exit 1
fi
echo "==> Using $PY_BIN ($("$PY_BIN" -V 2>&1))"

# Debian/Ubuntu ship the venv module as a separate package.
if ! "$PY_BIN" -m venv --help >/dev/null 2>&1; then
  echo "==> Installing the venv module for $(basename "$PY_BIN")"
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends "$(basename "$PY_BIN")-venv"
fi

echo "==> Creating virtualenv (.venv)"
[ -d .venv ] || "$PY_BIN" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

echo "==> Installing the package with the full GPU stack (downloads torch + vLLM, several GB)"
pip install -e ".[ec2]"

mkdir -p outputs
# Freeze what actually resolved: the dependency spec has no upper bounds, so this snapshot is the
# only way to reinstall the same stack later.
python -m pip freeze > "outputs/pip-freeze-$(date -u +%Y%m%dT%H%M%SZ).txt"
echo "==> Wrote outputs/pip-freeze-*.txt (keep it with the run provenance)"

# Data location. The shipped configs use the in-repo data/benchmarks, so /data is optional and is
# only created when passwordless sudo is available.
if [ "${AEGIS_MAKE_DATA_MOUNT:-1}" = "1" ] && command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
  sudo mkdir -p /data/benchmarks
  sudo chown -R "$USER" /data
  echo "==> Created /data/benchmarks (optional; configs default to the in-repo data/benchmarks)"
else
  echo "==> Skipped /data/benchmarks (no passwordless sudo, or AEGIS_MAKE_DATA_MOUNT=0)."
  echo "    The default configs read the in-repo data/benchmarks, so this is not a problem."
fi

echo "==> Sanity check: mechanism + adapters (offline, no GPU, no network)"
python -m pytest -q

cat <<'EOF'

Provisioning done. Next steps:
  1. Validate the run config before spending GPU time:
       python scripts/check_ready.py --config configs/ec2_real_evaluation.yaml
  2. Start the judge backbone:
       HF_TOKEN=... bash scripts/ec2/serve_vllm.sh          # one backbone on :8001
       bash scripts/ec2/serve_backbones.sh                  # RQ4: three backbones, 3 GPUs
  3. Smoke first (dummy judge, plumbing only - never report its numbers):
       bash scripts/ec2/run_real.sh smoke
  4. Then the real run:
       AEGIS_OUT=outputs/real_harmbench bash scripts/ec2/run_real.sh full
       AEGIS_OUT=outputs/real_harmbench bash scripts/ec2/run_real.sh ablate
  5. Back up and finalise (after the run):
       bash scripts/ec2/backup_results.sh
       python scripts/finalize_run.py --run outputs/real_harmbench

Notes
  - meta-llama/Llama-3.1-8B-Instruct is a gated HF repo: export HF_TOKEN=... before
    serve_vllm.sh (or use a model you can access).
  - Datasets ship in the repository under data/benchmarks; uploading to /data/benchmarks is only
    needed if you deliberately point data.root there. universal_injection is NOT shipped and
    requires the genuine gradient-optimised suffix (see docs/universal_injection_runbook.md).
  - The first judge run also downloads the sentence-transformer embedding model (~90 MB).
  - See docs/ec2_experiment_guide.md for the full runbook and cost estimates.
EOF
