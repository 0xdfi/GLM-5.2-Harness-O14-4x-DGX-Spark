# O14 operational performance envelope

This document defines the public operator-supplied boundary around **O14 Fast — 250K total KV, READY**. The current capacity defaults in `recipe/o14.env.example` and `recipe/serve-o14.sh` are `MAX_MODEL_LEN=249000` and `KV_CACHE_MEMORY_BYTES=7995534848` bytes per rank. The remaining runtime settings come from the source-complete O14 package and preserved operating envelope; repository tests bind the published values. No public O14 Fast OCI image is claimed.

## Per-rank environment

Before starting each rank container, the operator must supply values appropriate to that rank and fabric for all of these variables:

- `RAY_ADDRESS`
- `VLLM_HOST_IP`
- `NCCL_SOCKET_IFNAME`
- `GLOO_SOCKET_IFNAME`
- `NCCL_IB_HCA`
- `NCCL_IB_GID_INDEX`
- `NCCL_IB_TC`
- `UCX_NET_DEVICES`
- `UCX_TLS`

The repository intentionally provides no defaults or example values for those deployment-specific settings. The image also leaves `CUDA_DEVICE_MAX_CONNECTIONS` unset. Set `CUDA_DEVICE_MAX_CONNECTIONS=4` in every rank-container environment before starting Ray or model processes.

## Container and host contract

Each rank container requires:

- GPU access;
- host networking and host IPC;
- 16 GiB of shared memory;
- unlimited memlock;
- operator-selected RDMA device mappings;
- a writable persistent cache mount; and
- verification from runtime communications logs that NCCL selected the intended RDMA/IB path rather than a socket fallback.

Host networking, host IPC, device mappings, and unlimited memlock materially reduce container isolation. They describe the measured configuration and must be scoped deliberately by the operator.

Set `O14_JIT_CACHE_ROOT` explicitly to the persistent cache mount inside the container. The wrapper derives CUDA, XDG, vLLM, B12X CUTE, Triton, TorchInductor, and PyTorch-extension cache directories from that root. If only `XDG_CACHE_HOME` is preset, the wrapper retains that XDG location and derives the other caches below its `o14` subdirectory; the XDG cache itself is not relocated. The example environment avoids that asymmetry by setting `O14_JIT_CACHE_ROOT`. The host-side mount source remains operator-selected.

`docker/Dockerfile.repro` also pins `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` inside the reconstructed image. An operator running the wrapper in a different image must provide an equivalent Triton/PTXAS toolchain.

## Published runtime defaults

The capacity values are the current O14 Fast recipe. Other defaults reconstruct the preserved operating envelope. A variable's presence records the recipe; it is not, by itself, proof that a runtime branch fired.

| Variables | Consumer / classification | Evidence boundary |
|---|---|---|
| `VLLM_USE_V2_MODEL_RUNNER`, `VLLM_WORKER_MULTIPROC_METHOD` | pinned vLLM runtime | The checked-in vLLM overlay reads these settings. `spawn` is retained as compatibility/safety configuration and is not claimed as a causal speed lever under the Ray executor. |
| `VLLM_ADAPTIVE_SPEC_DEPTHS`, `VLLM_MTP_INSTRUMENT`, `VLLM_MTP_INSTRUMENT_WINDOW`, `VLLM_BUILDA_BMM`, `VLLM_MOE_MARLIN_ATOMIC_ADD`, `VLLM_USE_B12X_SPARSE_INDEXER`, `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`, `KV_FP8_ROPE` | O14/vLLM/B12X overlay | The public source contains the consumers. `VLLM_MOE_MARLIN_ATOMIC_ADD` is the O14 Marlin overlay switch. |
| `VLLM_USE_FLASHINFER_SAMPLER`, `VLLM_MARLIN_USE_ATOMIC_ADD`, `SAFETENSORS_FAST_GPU` | preserved launch compatibility knobs | No checked-in O14 overlay consumes these names; they are not used as sole proof of sampler, Marlin, or loader selection. `VLLM_MARLIN_USE_ATOMIC_ADD` is distinct from the O14 overlay switch above. |
| `CUTE_DSL_ARCH` | B12X / FlashAttention CUTE source | Targets `sm_121a` in this recipe. |
| `CUDA_DEVICE_ORDER`, `CUDA_MODULE_LOADING`, `PYTORCH_CUDA_ALLOC_CONF`, `TORCH_NCCL_ASYNC_ERROR_HANDLING` | CUDA / PyTorch | Process-environment defaults from the preserved recipe. |
| `NCCL_CUMEM_ENABLE`, `NCCL_MIN_NCHANNELS`, `NCCL_MAX_NCHANNELS` | NCCL | The fixed four-channel setting reflects the measured fabric envelope; other fabrics may require retuning before validation. |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | vLLM runtime | Preserved request-execution timeout; not a performance acceptance result. |
| `O14_DRIVER_CUDA_DEVICE_MAX_CONNECTIONS`, `O14_JIT_CACHE_ROOT` and derived cache variables | public wrapper | Driver/rank split and persistent compilation/cache layout described here. |

The recovered wrapper intentionally omits `R17_DRAFT_TEMP_SCALE`, `VLLM_NVFP4_GEMM_BACKEND`, `VLLM_NVFP4_ALLOW_SLOW_FALLBACK`, and `VLLM_QUANTIZATION_DISABLE_FUSED_MOE`: the source-complete O14 runtime published here contains no consumer for the three `VLLM_*` names, while the draft-temperature source default is already `1.0` (an exact no-op). Compact KV and active backend selection are instead explicit in the rendered argv (`nvfp4_ds_mla`, `B12X_MLA_SPARSE`, and `marlin`).

## Driver/rank CUDA connection split

The historical campaign setup showed two `CUDA_DEVICE_MAX_CONNECTIONS` layers, retained in the current Fast operating recipe:

1. the preserved rank-container environment records the base value `4` before Ray or model processes start;
2. `recipe/serve-o14.sh` sets the API-driver process from `O14_DRIVER_CUDA_DEVICE_MAX_CONNECTIONS`, whose default is `32`. Setting `CUDA_DEVICE_MAX_CONNECTIONS` directly before invoking the wrapper does not override this driver value; use the `O14_` variable.

Reading the executor source, workers launched through pre-existing rank containers and Ray daemons should preserve the rank-container value of `4`; driver-spawned local workers would instead inherit `32`. This is an inference from source, not an observation of worker process environments. The evidence does not include a direct per-worker process-environment capture, so the documented-versus-observed reconciliation remains open pending such a capture.

## Checkpoint and architecture contract

`MODEL_PATH` must name an operator-prepared local checkpoint directory. `--download-dir` is pinned to that same operator-selected mount so any dependency cache lookup does not fall back to a private home directory; the recipe does not authorize an agent to synthesize a model reference or download weights during launch. Model weights remain outside the image.

The fixed `--hf-overrides` value encodes the measured GLM-5.2 78-layer full/sparse attention schedule. The stock, modified, and obliterated QuantTrio checkpoints described by this repository share that architecture. A checkpoint with a different layer count or attention schedule is outside this recipe and must not be launched by changing the pattern speculatively.

## Render-first launch sequence

Use `recipe/o14.env.example` as a starting point, supply the per-rank variables above through the operator's deployment system, and render the command before execution:

```bash
bash recipe/serve-o14.sh render
```

Rendering is the default. The Fast-only wrapper refuses every other profile before rendering or serving. Starting the API server requires `O14_PROFILE_NAME=o14-fast`, `O14_PROFILE_STATUS=READY`, the `serve` argument, and `O14_EXECUTE=1`. Rank formation, checkpoint placement, RDMA selection, health checks, and rollback remain operator-controlled.
