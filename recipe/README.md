# Recipe

`docker/Dockerfile.repro` is the complete public runtime assembly recipe. `serve-o14.sh` is the render-first launch command matching the measured O14 envelope.

The operator-supplied per-rank topology, container requirements, persistent caches, and measured driver/rank CUDA connection split are defined in [`docs/OPERATIONAL_ENVELOPE.md`](../docs/OPERATIONAL_ENVELOPE.md).

The build uses the checked-in 74-file vLLM overlay, 3-file B12X overlay, 14-path native patch, exact binary input hashes, and the FlashInfer compatibility patch. Model weights stay outside the image. The runtime accepts stock QuantTrio and weight-modified/obliterated QuantTrio checkpoints; it does not identify or reject a checkpoint based on tensor hashes.

Exact W8-head rescoring is optional. `O14_LMHEAD_PROFILE=auto` is the default: it enables the sidecar path only when `lmhead_w8v2_sidecar.safetensors` exists under `MODEL_PATH`. Use `off` to force ordinary checkpoint behavior or `required` to fail closed when reproducing the measured W8-head profile.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 reproducibility/verify.py
docker build -f docker/Dockerfile.repro -t glm52-o14:public-repro .
cp recipe/o14.env.example .env.o14
set -a; source .env.o14; set +a
bash recipe/serve-o14.sh render
```

Execution requires both `O14_EXECUTE=1` and the `serve` argument. Four-node Ray/RoCE formation, checkpoint placement, and rollback remain operator-controlled.
