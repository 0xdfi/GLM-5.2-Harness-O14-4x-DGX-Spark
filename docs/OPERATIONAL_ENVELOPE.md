# O14 operational performance envelope

This document defines the public operator-supplied boundary around the measured O14 recipe. It records the required shape of the rank environment without publishing deployment-specific topology values. A clean public image build, live cluster launch, and benchmark rerun remain separate, unproven steps.

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

Set `O14_JIT_CACHE_ROOT` to the persistent cache mount inside the container. The wrapper derives separate CUDA, vLLM, B12X CUTE, Triton, TorchInductor, and PyTorch-extension cache directories below that root. The example environment uses a generic container path only; the host-side mount source remains operator-selected.

## Driver/rank CUDA connection split

The measured setup deliberately used two `CUDA_DEVICE_MAX_CONNECTIONS` layers:

1. rank containers, Ray daemons, and their inherited node-local environment use the base value `4`;
2. `recipe/serve-o14.sh` sets the API-driver process from `O14_DRIVER_CUDA_DEVICE_MAX_CONNECTIONS`, whose default is `32`.

The executor applies driver variables with non-overwriting defaults, so model workers are source-inferred to preserve the rank-container value. The evidence does not include a direct per-worker process-environment capture; this source inference must not be represented as direct process evidence.

## Safe launch sequence

Use `recipe/o14.env.example` as a starting point, supply the per-rank variables above through the operator's deployment system, and render the command before execution:

```bash
bash recipe/serve-o14.sh render
```

Rendering is the default. Starting the API server still requires both the `serve` argument and `O14_EXECUTE=1`. Rank formation, checkpoint placement, RDMA selection, health checks, and rollback remain operator-controlled.
