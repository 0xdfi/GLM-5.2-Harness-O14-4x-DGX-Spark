#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

MODE="${1:-render}"
: "${MODEL_PATH:?Set MODEL_PATH to the mounted checkpoint directory}"

export VLLM_ADAPTIVE_SPEC_DEPTHS="${VLLM_ADAPTIVE_SPEC_DEPTHS:-2,4,5}"
export VLLM_MTP_INSTRUMENT="${VLLM_MTP_INSTRUMENT:-1}"
export VLLM_MTP_INSTRUMENT_WINDOW="${VLLM_MTP_INSTRUMENT_WINDOW:-32}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-1}"
# The measured live O14 wrapper imports Build A v0 directly. Do not set
# VLLM_BUILDA_VER here; that selector exists only inside the unwired v1 module.
export VLLM_BUILDA_BMM="${VLLM_BUILDA_BMM:-1}"
export VLLM_MOE_MARLIN_ATOMIC_ADD="${VLLM_MOE_MARLIN_ATOMIC_ADD:-1}"
export VLLM_MARLIN_USE_ATOMIC_ADD="${VLLM_MARLIN_USE_ATOMIC_ADD:-1}"
export VLLM_USE_B12X_SPARSE_INDEXER="${VLLM_USE_B12X_SPARSE_INDEXER:-1}"
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-256}"
export KV_FP8_ROPE="${KV_FP8_ROPE:-1}"

export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_121a}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-4}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

# API-driver value; rank containers use a separate measured base value.
export CUDA_DEVICE_MAX_CONNECTIONS="${O14_DRIVER_CUDA_DEVICE_MAX_CONNECTIONS:-32}"

o14_cache_root="${O14_JIT_CACHE_ROOT:-${XDG_CACHE_HOME:-${HOME:-/tmp}/.cache}/o14}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${o14_cache_root}/cuda}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${o14_cache_root}/xdg}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${o14_cache_root}/vllm}"
export B12X_CUTE_COMPILE_CACHE_DIR="${B12X_CUTE_COMPILE_CACHE_DIR:-${o14_cache_root}/b12x-cute}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${o14_cache_root}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${o14_cache_root}/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${o14_cache_root}/torch-extensions}"

# The runtime works with stock/modified QuantTrio checkpoints and with the
# O14 W8-head lineage. Exact-head rescoring is an optional checkpoint feature,
# not a prerequisite for the runtime.
lmhead_profile="${O14_LMHEAD_PROFILE:-auto}"
lmhead_default="${MODEL_PATH%/}/lmhead_w8v2_sidecar.safetensors"
lmhead_candidate="${VLLM_LMHEAD_V2_SIDECAR:-${lmhead_default}}"
case "${lmhead_profile}" in
  auto)
    if [[ -f "${lmhead_candidate}" ]]; then
      export VLLM_LMHEAD_V2_SIDECAR="${lmhead_candidate}"
      export VLLM_LMHEAD_V2_TOPM="${VLLM_LMHEAD_V2_TOPM:-64}"
      export VLLM_LMHEAD_V2_REQUIRE=1
    else
      unset VLLM_LMHEAD_V2_SIDECAR VLLM_LMHEAD_V2_TOPM
      export VLLM_LMHEAD_V2_REQUIRE=0
    fi
    ;;
  off)
    unset VLLM_LMHEAD_V2_SIDECAR VLLM_LMHEAD_V2_TOPM
    export VLLM_LMHEAD_V2_REQUIRE=0
    ;;
  required)
    if [[ ! -f "${lmhead_candidate}" ]]; then
      echo "O14_LMHEAD_PROFILE=required but sidecar is missing: ${lmhead_candidate}" >&2
      exit 66
    fi
    export VLLM_LMHEAD_V2_SIDECAR="${lmhead_candidate}"
    export VLLM_LMHEAD_V2_TOPM="${VLLM_LMHEAD_V2_TOPM:-64}"
    export VLLM_LMHEAD_V2_REQUIRE=1
    ;;
  *)
    echo "Invalid O14_LMHEAD_PROFILE=${lmhead_profile}; use auto, off, or required." >&2
    exit 64
    ;;
esac

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
    "rejection_sample_method": "block",
    "adaptive_speculative_tokens_window": 32,
}, separators=(",", ":")))
PY
)"

args=(
  python3 -m vllm.entrypoints.openai.api_server
  --model "${MODEL_PATH}" --tokenizer "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME:-glm-5.2}"
  --trust-remote-code --download-dir "${MODEL_PATH}" --load-format auto
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
  --hf-overrides '{"use_index_cache":true,"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}'
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
