#!/usr/bin/env bash
# One-time provisioning of an EC2 instance for the real-data runs.
# Run from the repository root on the server:
#   bash scripts/ec2/setup.sh
set -euo pipefail

PY="${PYTHON:-python3.11}"

echo "==> Updating system packages"
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  "$PY" "$PY-venv" git curl

echo "==> Creating virtualenv (.venv) with the full EC2 stack"
if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[ec2]"

echo "==> Data and output mount points"
sudo mkdir -p /data/benchmarks
sudo chown -R "$USER" /data
mkdir -p outputs

echo "==> Sanity check: synthetic mechanism"
python -m pytest -q

cat <<'EOF'

Provisioning done. Next steps:
  1. Upload benchmarks: scripts/ec2/upload_data.ps1  (from your Windows dev machine)
  2. Start the judge backbone: bash scripts/ec2/serve_vllm.sh
  3. Run: bash scripts/ec2/run_real.sh smoke   # then: bash scripts/ec2/run_real.sh full

Notes
  - meta-llama/Llama-3.1-8B-Instruct is a gated HF repo: export HF_TOKEN=... before
    serve_vllm.sh (or use a model you can access).
  - The first judge run also downloads the sentence-transformer embedding model (~90 MB).
  - See docs/ec2_experiment_guide.md for the full runbook and cost estimates.
EOF