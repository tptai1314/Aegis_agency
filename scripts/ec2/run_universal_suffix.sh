#!/usr/bin/env bash
# Produce the GENUINE gradient-optimized universal injected content S for the
# universal_injection benchmark (liu2024universal, arXiv:2403.04957).
#
# Runs the official optimizer (SheltonLiu-N/Universal-Prompt-Injection) against a
# white-box Llama-2-7B-Chat on this GPU instance and writes a results JSON whose
# last-step `final_suffix` is S. Fetch that JSON back and rebuild the benchmark:
#
#   scp -i ~/.ssh/aegis.pem ubuntu@<host>:"/path/results/eval/.../*.json" ./
#   python scripts/prepare_universal_injection.py \
#       --source <path/to/Universal-Prompt-Injection> \
#       --output D:/Data/benchmarks/universal_injection \
#       --suffix-file <results.json>
#
# Prereqs (see docs/ec2_experiment_guide.md):
#   - CUDA GPU instance (>= ~14 GB VRAM for llama-2-7b-chat-hf FP16)
#   - HF_TOKEN exported with an account granted Llama-2 access (gated repo)
#
# Usage:
#   HF_TOKEN=hf_... bash scripts/ec2/run_universal_suffix.sh \
#       --injection semi-dynamic --tokens 150 --num-steps 500
set -euo pipefail

REPO_R="$HOME/Universal-Prompt-Injection"
VENV_R="$REPO_R/.venv"
REPO_URL="https://github.com/SheltonLiu-N/Universal-Prompt-Injection.git"

# ---- args ---------------------------------------------------------------
INJECTION="semi-dynamic"
TOKENS=150
NUM_STEPS=500
BATCH_SIZE=256
START=0
END=5
MODEL="llama2"
SAVE_SUFFIX="normal"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --injection)  INJECTION="$2";  shift 2 ;;
    --tokens)     TOKENS="$2";     shift 2 ;;
    --num-steps)  NUM_STEPS="$2";  shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --start)      START="$2";      shift 2 ;;
    --end)        END="$2";        shift 2 ;;
    --model)      MODEL="$2";      shift 2 ;;
    --save-suffix) SAVE_SUFFIX="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "==> ERROR: HF_TOKEN not set (required for meta-llama/Llama-2 gated access)." >&2
  exit 1
fi

# ---- checkout + env ------------------------------------------------------
if [ ! -d "$REPO_R" ]; then
  echo "==> Cloning official repo"
  git clone "$REPO_URL" "$REPO_R"
else
  echo "==> Repo already present: $REPO_R"
fi
cd "$REPO_R"

if [ ! -d "$VENV_R" ]; then
  echo "==> Creating venv $VENV_R"
  python3 -m venv "$VENV_R"
fi
# shellcheck disable=SC1091
source "$VENV_R/bin/activate"

echo "==> Installing requirements"
python -m pip install --upgrade pip
if pip install -r requirements.txt; then
  :
else
  echo "==> requirements install failed; retrying fresh torch/CUDA wheels (cu118)."
  pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
  pip install -r requirements.txt
fi

# ---- model ---------------------------------------------------------------
echo "==> Downloading meta-llama/Llama-2-7b-chat-hf to ./models/llama2/ (gated, needs approval)"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
mkdir -p models/llama2
python - <<'EOF'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
m = "meta-llama/Llama-2-7b-chat-hf"
tok = AutoTokenizer.from_pretrained(m, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(m, device_map="auto",
                                             torch_dtype=torch.float16,
                                             low_cpu_mem_usage=True, use_cache=False)
model.save_pretrained("models/llama2/llama-2-7b-chat-hf", from_pt=True)
tok.save_pretrained("models/llama2/llama-2-7b-chat-hf", from_pt=True)
print("model saved")
EOF

# ---- optimize ------------------------------------------------------------
echo "==> Running universal_prompt_injection.py"
python universal_prompt_injection.py \
  --model "$MODEL" \
  --injection "$INJECTION" \
  --tokens "$TOKENS" \
  --num-steps "$NUM_STEPS" \
  --batch_size "$BATCH_SIZE" \
  --start "$START" --end "$END" \
  --save_suffix "$SAVE_SUFFIX"

OUT="results/eval/$MODEL/$INJECTION/momentum_1.0/token_length_$TOKENS/target_0/${START}_${END}_20_${SAVE_SUFFIX}.json"
cat <<EOF

Done. The optimized universal injected content S is the last-step final_suffix of:
  $OUT
Fetch it back and rebuild using scripts/prepare_universal_injection.py --suffix-file.
(Optional sanity: python get_responses_universal.py --evaluate sentiment_analysis --path "$OUT")
EOF