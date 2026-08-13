#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

MODE="${1:-render}"
: "${MODEL_PATH:?Set MODEL_PATH to the mounted checkpoint directory}"

export VLLM_ADAPTIVE_SPEC_DEPTHS="${VLLM_ADAPTIVE_SPEC_DEPTHS:-2,4,5}"
export VLLM_MTP_INSTRUMENT="${VLLM_MTP_INSTRUMENT:-1}"
export VLLM_MTP_INSTRUMENT_WINDOW="${VLLM_MTP_INSTRUMENT_WINDOW:-32}"
# The measured live O14 wrapper imports Build A v0 directly. Do not set
# VLLM_BUILDA_VER here; that selector exists only inside the unwired v1 module.
export VLLM_BUILDA_BMM="${VLLM_BUILDA_BMM:-1}"
export VLLM_MOE_MARLIN_ATOMIC_ADD="${VLLM_MOE_MARLIN_ATOMIC_ADD:-1}"
export VLLM_USE_B12X_SPARSE_INDEXER="${VLLM_USE_B12X_SPARSE_INDEXER:-1}"
export R17_DRAFT_TEMP_SCALE="${R17_DRAFT_TEMP_SCALE:-1.0}"
export KV_FP8_ROPE="${KV_FP8_ROPE:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-4}"

spec="$(python3 - "${MODEL_PATH}" <<'PY'
import json, sys
print(json.dumps({
    "model": sys.argv[1],
    "method": "mtp",
    "quantization": "compressed-tensors",
    "num_speculative_tokens": 5,
    "moe_backend": "marlin",
    "attention_backend": "B12X_MLA_SPARSE",
    "draft_sample_method": "probabilistic",
    "adaptive_speculative_tokens_window": 32,
}, separators=(",", ":")))
PY
)"

args=(
  python3 -m vllm.entrypoints.openai.api_server
  --model "${MODEL_PATH}" --tokenizer "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME:-glm-5.2}"
  --trust-remote-code --load-format auto
  --quantization compressed-tensors
  --distributed-executor-backend ray
  --tensor-parallel-size 4
  --decode-context-parallel-size 1
  --pipeline-parallel-size 1
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
  --max-model-len "${MAX_MODEL_LEN:-399000}"
  --max-num-seqs "${MAX_NUM_SEQS:-4}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}"
  --generation-config vllm
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":40}'
  --port "${PORT:-8211}" --host "${HOST:-0.0.0.0}"
  --no-enable-log-requests
  --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES:-12792889000}"
  --kv-cache-dtype nvfp4_ds_mla
  --attention-backend B12X_MLA_SPARSE
  --moe-backend marlin
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice
  --speculative-config "${spec}"
  --long-prefill-token-threshold "${LONG_PREFILL_TOKEN_THRESHOLD:-2048}"
  --async-scheduling
  --enable-decode-aware-prefill
  --decode-prefill-token-budget "${DECODE_PREFILL_TOKEN_BUDGET:-256}"
  --idle-prefill-token-budget "${IDLE_PREFILL_TOKEN_BUDGET:-2048}"
  --max-long-prefills-per-step "${MAX_LONG_PREFILLS_PER_STEP:-1}"
  --compilation-config '{"cudagraph_capture_sizes":[6,12,18,24]}'
  --enable-prefix-caching
)

if [[ "${MODE}" == "render" ]]; then
  printf '%q ' "${args[@]}"
  printf '\n'
  exit 0
fi

if [[ "${MODE}" != "serve" || "${O14_EXECUTE:-0}" != "1" ]]; then
  echo "Refusing execution. Use 'render', or set O14_EXECUTE=1 and invoke 'serve'." >&2
  exit 64
fi

exec "${args[@]}"
