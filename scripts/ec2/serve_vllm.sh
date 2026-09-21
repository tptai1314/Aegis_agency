#!/usr/bin/env bash
# Start the self-hosted judge backbone (vLLM OpenAI-compatible server on :8001).
# Run from the repository root. Idempotent: no-op if a vLLM server is already up.
#
#   HF_TOKEN=... bash scripts/ec2/serve_vllm.sh         # gated repos (Llama-3)
#   VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct bash scripts/ec2/serve_vllm.sh
set -euo pipefail

MODEL="${VLLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
SERVED="${VLLM_SERVED_MODEL:-$MODEL}"
PORT="${VLLM_PORT:-8001}"
MAX_LEN="${VLLM_MAX_LEN:-8192}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.92}"
RUNTIME="${VLLM_RUNTIME:-vllm}"

# shellcheck disable=SC1091
source .venv/bin/activate
mkdir -p outputs

if pgrep -f "vllm.entrypoints" >/dev/null 2>&1; then
  echo "A vLLM server is already running; pid(s): $(pgrep -f 'vllm.entrypoints' | tr '\n' ' ')"
  exit 0
fi

echo "==> Starting vLLM: model=$MODEL port=$PORT max_len=$MAX_LEN"
if [ "$RUNTIME" = "vllm" ]; then
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$SERVED" \
    --host 127.0.0.1 --port "$PORT" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    > outputs/vllm.log 2>&1 &
elif [ "$RUNTIME" = "sglang" ]; then
  python -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --mem-fraction-static 0.9 \
    > outputs/vllm.log 2>&1 &
else
  echo "Unknown VLLM_RUNTIME=$RUNTIME (use vllm or sglang)." >&2
  exit 1
fi
echo $! > outputs/vllm.pid

echo "==> Waiting for $SERVED to become healthy on :$PORT (up to 6 min)"
for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "vLLM ready on http://127.0.0.1:$PORT/health"
    exit 0
  fi
  sleep 2
done
echo "vLLM did not become healthy; see outputs/vllm.log" >&2
exit 1