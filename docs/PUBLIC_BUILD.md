# Complete public build recipe

The repository now includes the full O14 runtime assembly path in [`docker/Dockerfile.repro`](../docker/Dockerfile.repro).

The separate [`OPERATIONAL_ENVELOPE.md`](OPERATIONAL_ENVELOPE.md) defines the operator-supplied rank topology, container capabilities, persistent caches, and driver/rank CUDA connection split needed around that image.

It pins and assembles:

- public ARM64 CUDA 13.0.2 base digest;
- exact R17 stage-0 vLLM wheel and native NVFP4 wheel release assets by SHA-256;
- 74-file vLLM runtime overlay;
- B12X public commit `334a2d75d166becea0aa640b402d521ea0a290eb` plus the exact 3-file live delta;
- FlashInfer 0.6.15 Python, cubin, and CUDA 13.0 JIT-cache wheels by public URL and SHA-256;
- the hash-gated FlashInfer compatibility patch;
- the observed dependency snapshot;
- the exact O14 serve command and environment.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 reproducibility/verify.py
docker build -f docker/Dockerfile.repro -t glm52-o14:public-repro .
```

The static closure check has been run. A clean public image build and benchmark rerun are not claimed yet. Model weights remain external. The W8 sidecar is optional for ordinary stock or modified QuantTrio serving and required only when reproducing the measured exact-head profile.
