# Complete O14 build and serve recipe

This directory closes the missing public runtime recipe.

## Included

- **74-file vLLM runtime overlay** recovered from the four identical O14 ranks: 67 modified and 7 added files over the exact clean R17 stage-0 wheel.
- **3-file B12X overlay** over public B12X commit `334a2d75d166becea0aa640b402d521ea0a290eb`; the other 123 installed B12X files match that commit.
- **14-path native NVFP4 cache-op patch** over public vLLM commit `e232d262369b8c918cf478a7a96a0fcf8127cf65`.
- **Hash-gated FlashInfer 0.6.15 compatibility patch** with exact preimage and result hashes.
- **Observed runtime dependency snapshot**, immutable public input hashes, and the complete Docker assembly recipe at [`docker/Dockerfile.repro`](../docker/Dockerfile.repro).
- **Exact serving command and environment** in [`recipe/serve-o14.sh`](../recipe/serve-o14.sh) and [`recipe/o14.env.example`](../recipe/o14.env.example).
- **Safe operational envelope** for operator-supplied rank topology, container capabilities, persistent caches, and the measured driver/rank CUDA connection split in [`docs/OPERATIONAL_ENVELOPE.md`](../docs/OPERATIONAL_ENVELOPE.md).

Run the static closure check:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 reproducibility/verify.py
```

Build after downloading the two release assets listed in `runtime-input-release.json`:

```bash
docker build -f docker/Dockerfile.repro -t glm52-o14:public-repro .
```

Render the exact serving command without starting a server:

```bash
docker run --rm -e MODEL_PATH=/models glm52-o14:public-repro render
```

## Boundary

Model weights are not bundled. Stock QuantTrio and weight-modified/obliterated QuantTrio checkpoints use the same runtime recipe; the optional W8 sidecar is auto-detected. Reproducing the published benchmark additionally requires the same checkpoint/W8 sidecar, four GB10 nodes, and equivalent RoCE/NCCL topology. The repository does not claim that a clean public image build or benchmark rerun has already been completed.
