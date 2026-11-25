#!/usr/bin/env bash
set -euo pipefail

# ---------- Config (override via flags or env) ----------
HF_TOKEN_FILE="${HF_TOKEN_FILE:-./huggingface_api_key.txt}"
MODEL="${MODEL:-meta-llama/Llama-3.1-70B-Instruct}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
DTYPE="${DTYPE:-bfloat16}"
MAX_LEN="${MAX_LEN:-4096}"
GPU_UTIL="${GPU_UTIL:-0.95}"

# Optional: avoid proxies for local server calls
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}"

# ---------- CLI flags ----------
# Examples:
#   ./start_llama.sh --gpus 0
#   ./start_llama.sh --gpus "0 1 2 3"
#   ./start_llama.sh --gpus "0,1,2,3"
#   ./start_llama.sh --model meta-llama/Llama-3.1-70B-Instruct
EXTRA_VLLM_ARGS=()
GPU_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)         GPU_ARG="${2:?}"; shift 2 ;;
    --model)        MODEL="${2:?}"; shift 2 ;;
    --port)         PORT="${2:?}"; shift 2 ;;
    --host)         HOST="${2:?}"; shift 2 ;;
    --dtype)        DTYPE="${2:?}"; shift 2 ;;
    --max-len)      MAX_LEN="${2:?}"; shift 2 ;;
    --gpu-util)     GPU_UTIL="${2:?}"; shift 2 ;;
    --)             shift; break ;;
    *)              EXTRA_VLLM_ARGS+=("$1"); shift ;;
  esac
done

# ---------- Read HF token ----------
if [[ ! -f "$HF_TOKEN_FILE" ]]; then
  echo "ERROR: HF token file not found at: $HF_TOKEN_FILE" >&2
  echo "Create it with your Hugging Face token on a single line." >&2
  exit 1
fi
HF_TOKEN="$(tr -d ' \t\r\n' < "$HF_TOKEN_FILE")"
if [[ -z "$HF_TOKEN" ]]; then
  echo "ERROR: HF token file is empty: $HF_TOKEN_FILE" >&2
  exit 1
fi
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export HF_TOKEN="$HF_TOKEN"

# ---------- GPU index handling ----------
# Accepts inputs like:
#   "0"
#   "0 1 2"
#   "0,1,2"
#   "0, 1  2"
norm_csv() {
  local s="$1"
  # Replace all whitespace with commas
  s="${s//[[:space:]]/,}"
  # Collapse multiple commas
  s="$(echo "$s" | tr -s ',')"
  # Trim leading/trailing commas
  s="${s#,}"
  s="${s%,}"
  echo "$s"
}

# Precedence:
# 1. --gpus argument
# 2. existing CUDA_VISIBLE_DEVICES
# 3. auto-detect via nvidia-smi
if [[ -n "$GPU_ARG" ]]; then
  CUDA_VISIBLE_DEVICES="$(norm_csv "$GPU_ARG")"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(norm_csv "$CUDA_VISIBLE_DEVICES")"
else
  # Auto-detect: pick all GPUs 0..N-1 (if nvidia-smi present), else default to 0
  if command -v nvidia-smi >/dev/null 2>&1; then
    TOTAL="$(nvidia-smi -L | wc -l | tr -d ' ')"
    if [[ "$TOTAL" =~ ^[0-9]+$ ]] && (( TOTAL > 0 )); then
      CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((TOTAL-1)))"
    else
      CUDA_VISIBLE_DEVICES="0"
    fi
  else
    CUDA_VISIBLE_DEVICES="0"
  fi
fi

# Validate format "d(,d)*"
if ! [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "ERROR: GPU list must be a list of indexes, e.g. '0 1 3' or '0,1,3' but got '$CUDA_VISIBLE_DEVICES'." >&2
  exit 1
fi

# Count GPUs directly from the visible indexes
IFS=',' read -r -a GPU_ARR <<< "$CUDA_VISIBLE_DEVICES"
NUM_VISIBLE="${#GPU_ARR[@]}"

if (( NUM_VISIBLE <= 0 )); then
  echo "ERROR: No GPUs selected (CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES')." >&2
  exit 1
fi

# Tensor parallel size: always use all visible GPUs
TP_SIZE="$NUM_VISIBLE"

export CUDA_VISIBLE_DEVICES

# ---------- Summary ----------
echo "============================================================"
echo "Launching vLLM OpenAI API server"
echo "  Model:                   $MODEL"
echo "  Host:                    $HOST"
echo "  Port:                    $PORT"
echo "  DType:                   $DTYPE"
echo "  Max model len:           $MAX_LEN"
echo "  GPU memory util:         $GPU_UTIL"
echo "  CUDA_VISIBLE_DEVICES:    $CUDA_VISIBLE_DEVICES"
echo "  Tensor parallel size:    $TP_SIZE"
if ((${#EXTRA_VLLM_ARGS[@]})); then
  echo "  Extra vLLM args:         ${EXTRA_VLLM_ARGS[*]}"
fi
echo "============================================================"

# ---------- No proxy for local addresses ----------
for var in NO_PROXY no_proxy; do
  current="${!var:-}"
  if [[ -n "$current" ]]; then
    export "$var"="$current,127.0.0.1,localhost,::1"
  else
    export "$var"="127.0.0.1,localhost,::1"
  fi
done

# ---------- Launch ----------
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --tensor-parallel-size "$TP_SIZE" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enforce-eager \
  --host "$HOST" --port "$PORT" \
  "${EXTRA_VLLM_ARGS[@]}"