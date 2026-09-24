#!/usr/bin/env bash
# Bring up a DIVERSE judge committee for RQ4: one backbone per port.
# Run from the repository root on a machine with (ideally) one GPU per backbone.
#
#   HF_TOKEN=... bash scripts/ec2/serve_backbones.sh
#   AEGIS_BACKBONES="8002|Qwen/Qwen2.5-7B-Instruct|Qwen/Qwen2.5-7B-Instruct" bash scripts/ec2/serve_backbones.sh
#
# The default set matches configs/ec2_rq4_diverse.yaml. Keep the two in sync: a backbone that is
# served but not mapped in the config is never used, and a mapped endpoint that is not served
# makes every judge call fail.
#
# With only ONE 24 GB GPU you cannot serve three 7-8B backbones: use hosted API backbones
# (judges.backend: anthropic, or an OpenAI-compatible endpoint you already have) instead, or run
# the homogeneous arm and report RQ4 as not measured.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# <port>|<served-model-name>|<hf-model-id>, one per line. blank lines ignored.
DEFAULT_BACKBONES="8001|meta-llama/Llama-3.1-8B-Instruct|meta-llama/Llama-3.1-8B-Instruct
8002|Qwen/Qwen2.5-7B-Instruct|Qwen/Qwen2.5-7B-Instruct
8003|mistralai/Mistral-7B-Instruct-v0.3|mistralai/Mistral-7B-Instruct-v0.3"
BACKBONES="${AEGIS_BACKBONES:-$DEFAULT_BACKBONES}"

N_GPUS=0
if command -v nvidia-smi >/dev/null 2>&1; then
  N_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
fi
N_BACKBONES="$(printf '%s\n' "$BACKBONES" | grep -c '|' || true)"
echo "==> GPUs detected: $N_GPUS | backbones requested: $N_BACKBONES"
if [ "$N_GPUS" -lt "$N_BACKBONES" ]; then
  echo "WARNING: fewer GPUs than backbones. Each 7-8B model wants ~16 GB of weights, so sharing"
  echo "         one GPU will OOM. Either free up GPUs, reduce the backbone list, or use APIs." >&2
fi

INDEX=0
while IFS='|' read -r PORT SERVED MODEL; do
  [ -n "${PORT:-}" ] || continue
  if [ "$N_GPUS" -ge "$N_BACKBONES" ] && [ "$N_GPUS" -gt 0 ]; then
    # Give each backbone its own GPU so vLLM does not fight for memory.
    export CUDA_VISIBLE_DEVICES="$INDEX"
    echo "==> Backbone $((INDEX + 1))/$N_BACKBONES on GPU $INDEX (:${PORT}, ${SERVED})"
  else
    unset CUDA_VISIBLE_DEVICES || true
    echo "==> Backbone $((INDEX + 1))/$N_BACKBONES (:${PORT}, ${SERVED}, all visible GPUs)"
  fi
  VLLM_MODEL="$MODEL" VLLM_SERVED_MODEL="$SERVED" VLLM_PORT="$PORT" \
    bash scripts/ec2/serve_vllm.sh
  INDEX=$((INDEX + 1))
done <<< "$BACKBONES"

echo
echo "All requested backbones are up. Validate the matching config before running:"
echo "    python scripts/check_ready.py --config configs/ec2_rq4_diverse.yaml"
echo "Stop them with:  kill \$(cat outputs/vllm-*.pid)"
