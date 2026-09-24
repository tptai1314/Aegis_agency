#!/usr/bin/env bash
# Produce the GENUINE gradient-optimised universal injected content S for the
# universal_injection benchmark (liu2024universal, arXiv:2403.04957).
#
# Runs the official optimizer (SheltonLiu-N/Universal-Prompt-Injection) against a white-box
# Llama-2-7B-Chat and writes a results JSON whose last-step `final_suffix` is S. Fetch that JSON
# back to the dev machine and rebuild the benchmark there:
#
#   python scripts/prepare_universal_injection.py \
#       --source <path/to/Universal-Prompt-Injection> \
#       --output data/benchmarks/universal_injection \
#       --suffix-file <results.json>
#
# Prereqs:
#   - a dedicated GPU with >= ~14 GB VRAM free (llama-2-7b-chat-hf in FP16); this job must not
#     share the GPU with a vLLM judge server
#   - HF_TOKEN exported for an account granted Llama-2 access (gated repo)
#
# Usage:
#   HF_TOKEN=hf_... bash scripts/ec2/run_universal_suffix.sh \
#       --injection semi-dynamic --tokens 150 --num-steps 500
#
# Note on cost: the paper uses 1000 steps; the default here is 500 (hours on a mid GPU). The
# suffix is only "genuine" if the run completed the optimisation loop, so the number of steps is
# recorded and must be stated in the benchmark provenance.
set -euo pipefail

REPO_R="${AEGIS_UPI_REPO:-$HOME/Universal-Prompt-Injection}"
VENV_R="$REPO_R/.venv"
REPO_URL="https://github.com/SheltonLiu-N/Universal-Prompt-Injection.git"
MIN_VRAM_MB=14000

# ---- args ---------------------------------------------------------------
INJECTION="semi-dynamic"
TOKENS=150
NUM_STEPS=500
BATCH_SIZE=256
START=0
END=5
MODEL="llama2"
SAVE_SUFFIX="normal"

while [ $# -gt 0 ]; do
  case "$1" in
    --injection)  INJECTION="$2";  shift 2 ;;
    --tokens)     TOKENS="$2";     shift 2 ;;
    --num-steps)  NUM_STEPS="$2";  shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --start)      START="$2";      shift 2 ;;
    --end)        END="$2";        shift 2 ;;
    --model)      MODEL="$2";      shift 2 ;;
    --save-suffix) SAVE_SUFFIX="$2"; shift 2 ;;
    -h|--help)    sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [ -z "${HF_TOKEN:-}" ]; then
  echo "==> ERROR: HF_TOKEN not set (required for meta-llama/Llama-2 gated access)." >&2
  exit 1
fi

# ---- GPU preflight ------------------------------------------------------
# Fail before the multi-GB download if the GPU cannot hold the model, and refuse to fight a
# running judge server for VRAM (that would OOM both).
if command -v nvidia-smi >/dev/null 2>&1; then
  FREE_MB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -nr | head -n 1)"
  echo "==> Free GPU memory: ${FREE_MB} MiB (need >= ${MIN_VRAM_MB} MiB)"
  if [ "${FREE_MB:-0}" -lt "$MIN_VRAM_MB" ]; then
    echo "==> ERROR: not enough free VRAM. Stop any vLLM/SGLang server first" >&2
    echo "         (kill \$(cat outputs/vllm-*.pid) 2>/dev/null) and re-run." >&2
    exit 1
  fi
  if pgrep -f 'vllm.entrypoints' >/dev/null 2>&1; then
    echo "==> ERROR: a vLLM server is running; the optimizer needs the GPU exclusively." >&2
    exit 1
  fi
else
  echo "==> WARNING: nvidia-smi not found; cannot verify VRAM. Continuing anyway." >&2
fi

# ---- checkout + env ------------------------------------------------------
if [ ! -d "$REPO_R" ]; then
  echo "==> Cloning official repo"
  git clone "$REPO_URL" "$REPO_R"
else
  echo "==> Repo already present: $REPO_R"
fi
cd "$REPO_R"
UPI_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "==> Optimizer repo commit: $UPI_COMMIT"

if [ ! -d "$VENV_R" ]; then
  echo "==> Creating venv $VENV_R"
  python3 -m venv "$VENV_R"
fi
# shellcheck disable=SC1091
source "$VENV_R/bin/activate"

echo "==> Installing requirements"
python -m pip install --upgrade pip
if ! pip install -r requirements.txt; then
  echo "==> requirements install failed; retrying with fresh torch/CUDA wheels (cu118)."
  pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
  pip install -r requirements.txt
fi

# ---- model ---------------------------------------------------------------
# The token stays in the environment only: writing it into ~/.huggingface or the git credential
# store persists a credential on a machine that may be borrowed or shared.
echo "==> Downloading meta-llama/Llama-2-7b-chat-hf to ./models/llama2/ (gated, needs approval)"
mkdir -p models/llama2
python - <<'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
m = "meta-llama/Llama-2-7b-chat-hf"
tok = AutoTokenizer.from_pretrained(m, use_fast=False)
tok.save_pretrained("models/llama2/llama-2-7b-chat-hf")
for name in ("pytorch_model.bin", "model.safetensors"):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            m, device_map="cpu", low_cpu_mem_usage=True, use_cache=False
        )
        model.save_pretrained("models/llama2/llama-2-7b-chat-hf", safe_serialization=True)
        del model
        break
    except Exception as exc:  # noqa: BLE001
        print(f"download attempt ({name}) failed: {exc}")
else:
    raise SystemExit("could not download the Llama-2 checkpoint")
print("model saved to models/llama2/llama-2-7b-chat-hf")
EOF

if [ ! -f models/llama2/llama-2-7b-chat-hf/config.json ]; then
  echo "==> ERROR: the local checkpoint was not created; check HF access approval." >&2
  exit 1
fi

# ---- optimize ------------------------------------------------------------
echo "==> Running universal_prompt_injection.py (${NUM_STEPS} steps; this takes hours)"
python universal_prompt_injection.py \
  --model "$MODEL" \
  --injection "$INJECTION" \
  --tokens "$TOKENS" \
  --num-steps "$NUM_STEPS" \
  --batch_size "$BATCH_SIZE" \
  --start "$START" --end "$END" \
  --save_suffix "$SAVE_SUFFIX"

# ---- locate the result ---------------------------------------------------
# The upstream output path encodes momentum/token length/target and a fixed "20"; rather than
# guessing it, take the newest results JSON under results/eval/.
SUFFIX_JSON="$(find results/eval -name '*.json' -type f -printf '%T@ %p\n' 2>/dev/null \
  | sort -nr | head -n 1 | cut -d' ' -f2- || true)"

if [ -z "${SUFFIX_JSON:-}" ]; then
  echo "==> ERROR: no results JSON under results/eval; check the optimizer log." >&2
  exit 1
fi

# Record the run identity next to the JSON so the benchmark provenance can cite it.
RUN_RECORD="$(dirname "$SUFFIX_JSON")/aegis_suffix_run.json"
cat > "$RUN_RECORD" <<EOF
{
  "optimizer_repo": "$REPO_URL",
  "optimizer_commit": "$UPI_COMMIT",
  "model": "meta-llama/Llama-2-7b-chat-hf",
  "injection": "$INJECTION",
  "tokens": $TOKENS,
  "num_steps": $NUM_STEPS,
  "batch_size": $BATCH_SIZE,
  "start": $START,
  "end": $END,
  "save_suffix": "$SAVE_SUFFIX",
  "results_json": "$SUFFIX_JSON",
  "gradient_optimized_suffix": true,
  "paper_reference": "liu2024universal uses 1000 steps; state the step count actually used"
}
EOF

cat <<EOF

==> Done. Optimised universal suffix written by:
      $SUFFIX_JSON
    run record:
      $RUN_RECORD

Fetch both to the dev machine (run this ON WINDOWS, from the repo root):

    mkdir -p .\\upi_results
    scp -i "\$env:USERPROFILE\\.ssh\\<key>" <user@host>:"$SUFFIX_JSON" .\\upi_results\\
    scp -i "\$env:USERPROFILE\\.ssh\\<key>" <user@host>:"$RUN_RECORD" .\\upi_results\\

Then build the benchmark and CHECK that it is flagged genuine:

    python scripts/prepare_universal_injection.py \\
        --source <path/to/Universal-Prompt-Injection> \\
        --output data/benchmarks/universal_injection \\
        --suffix-file .\\upi_results\\$(basename "$SUFFIX_JSON")

    python -c "import json;print(json.load(open('data/benchmarks/universal_injection/provenance.json'))['gradient_optimized_suffix'])"
    # must print True; anything else is the SIMULATED stand-in and must NOT be reported
EOF
