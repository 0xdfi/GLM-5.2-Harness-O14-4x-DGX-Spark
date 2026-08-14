# Recipe

`docker/Dockerfile.repro` is the complete public runtime assembly recipe. `serve-o14.sh` is the render-first launch command matching the measured O14 envelope.

The build uses the checked-in 74-file vLLM overlay, 3-file B12X overlay, 14-path native patch, exact binary input hashes, and the FlashInfer compatibility patch. Model weights stay outside the image.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 reproducibility/verify.py
docker build -f docker/Dockerfile.repro -t glm52-o14:public-repro .
cp recipe/o14.env.example .env.o14
set -a; source .env.o14; set +a
bash recipe/serve-o14.sh render
```

Execution requires both `O14_EXECUTE=1` and the `serve` argument. Four-node Ray/RoCE formation, checkpoint placement, and rollback remain operator-controlled.
