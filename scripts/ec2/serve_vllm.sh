#!/usr/bin/env bash
# Start one self-hosted judge backbone as an OpenAI-compatible vLLM server.
# Run from the repository root.
#
#   HF_TOKEN=... bash scripts/ec2/serve_vllm.sh                     # Llama-3.1-8B on :8001
#   VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct VLLM_PORT=8002 bash scripts/ec2/serve_vllm.sh
#   VLLM_RUNTIME=sglang bash scripts/ec2/serve_vllm.sh
#
# Idempotent per PORT (not globally): a second backbone on another port is a normal RQ4 setup, so
# the guard checks the port's own health endpoint instead of bailing out when any vLLM process
# exists. Use scripts/ec2/serve_backbones.sh to bring up a whole diverse set.
set -euo pipefail

MODEL="${VLLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
SERVED="${VLLM_SERVED_MODEL:-$MODEL}"
PORT="${VLLM_PORT:-8001}"
# 8192 covers every shipped benchmark except the longest `dan` payloads; the runner shortens those
# via judges.max_prompt_chars. Raise this if you prefer truncating less and have the VRAM.
MAX_LEN="${VLLM_MAX_LEN:-8192}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.92}"
RUNTIME="${VLLM_RUNTIME:-vllm}"
LOG="outputs/vllm-$PORT.log"
PIDFILE="outputs/vllm-$PORT.pid"
HEALTH="http://127.0.0.1:$PORT/health"

# shellcheck disable=SC1091
source .venv/bin/activate
mkdir -p outputs

if curl -sf "$HEALTH" >/dev/null 2>&1; then
  echo "A vLLM server is already healthy on :$PORT; nothing to do."
  exit 0
fi

echo "==> Starting $RUNTIME: model=$MODEL served=$SERVED port=$PORT max_len=$MAX_LEN gpu_util=$GPU_UTIL"
if [ "$RUNTIME" = "vllm" ]; then
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$SERVED" \
    --host 127.0.0.1 --port "$PORT" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    > "$LOG" 2>&1 &
elif [ "$RUNTIME" = "sglang" ]; then
  python -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --mem-fraction-static 0.9 \
    > "$LOG" 2>&1 &
else
  echo "Unknown VLLM_RUNTIME=$RUNTIME (use vllm or sglang)." >&2
  exit 1
fi
echo $! > "$PIDFILE"

echo "==> Waiting for :$PORT to become healthy (up to 6 min; log: $LOG)"
for _ in $(seq 1 180); do
  if curl -sf "$HEALTH" >/dev/null 2>&1; then
    # A healthy port is not enough: the served model name must match judges.model in the config.
    if curl -sf "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q "$SERVED"; then
      echo "Ready: http://127.0.0.1:$PORT/v1 serving '$SERVED'"
      echo "Config check: judges.endpoint must be http://127.0.0.1:$PORT/v1 and"
      echo "              judges.model (or judges.models[<backbone>]) must be '$SERVED'."
      exit 0
    fi
    echo "Server is healthy on :$PORT but does not list '$SERVED' under /v1/models;" >&2
    echo "check VLLM_SERVED_MODEL and the config's judges.model." >&2
    exit 1
  fi
  sleep 2
done
echo "vLLM did not become healthy on :$PORT; see $LOG" >&2
exit 1
