# O14 profiles and recipe

`docker/Dockerfile.repro` is the complete public runtime assembly recipe. `serve-o14.sh` is the render-first O14 Fast launch command. Named status and selection rules are machine-readable in [`profiles/o14-profiles.json`](../profiles/o14-profiles.json); automation must also follow [`AGENTS.md`](../AGENTS.md).

## O14 Fast — 250K total KV, READY

O14 Fast is TP4/DCP1/PP1 with 250K total KV and an exact allocator total of 250023 tokens:

| Field | Exact value |
|---|---:|
| `KV_CACHE_MEMORY_BYTES` | `7995534848` bytes per rank |
| `MAX_MODEL_LEN` | `249000` |
| `MAX_NUM_SEQS` | `4` |
| `MAX_NUM_BATCHED_TOKENS` | `2048` |

The geometry is 79 main entries × 368 bytes plus 22 index entries × 132 bytes = 31976 bytes per local token per rank. A 64-token block is 2046464 bytes per rank. The selected 3907 blocks and 3891 blocks per max request advertise 250023 tokens; one block lower advertises 249959.

There is no published Fast OCI image. Agents may build and deploy Fast only from a pinned repository commit after verifying `SHA256SUMS`; model weights stay outside the image. Historical 399K campaign receipts remain preserved under [`evidence/`](../evidence/) and [`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md).

## O14 Balanced — TESTING / DO NOT DEPLOY

Balanced is a non-deployable TP4/DCP2/PP1 placeholder targeting 500K total logical KV: allocator expectation 500237, `KV_CACHE_MEMORY_BYTES=8000000000` bytes per rank, and `MAX_MODEL_LEN=490000`. Capacity and speed measured fields remain `TBD`.

Balanced requires separate v3 runtime source changes and, only after live acceptance, its own immutable OCI image, source/build manifest, `sha256:` digest, and anonymous-pull proof. Never reuse or overwrite Fast's image, and never synthesize a Balanced image tag from this placeholder.

## Public operational boundary

The operator-supplied per-rank topology, container requirements, persistent caches, and driver/rank CUDA connection split are defined in [`docs/OPERATIONAL_ENVELOPE.md`](../docs/OPERATIONAL_ENVELOPE.md). The public topology assumes a 100GbE switch; 100000 Mb/s per active RDMA link is expected. Addresses, interfaces, device mappings, host paths, and checkpoint locations remain operator-supplied.

The build uses the checked-in 74-file vLLM overlay, 3-file B12X overlay, 14-path native patch, exact binary input hashes, and the FlashInfer compatibility patch. The runtime accepts stock QuantTrio and weight-modified/obliterated QuantTrio checkpoints that retain the documented GLM-5.2 78-layer attention schedule; it does not identify or reject a checkpoint based on tensor hashes.

Exact W8-head rescoring is optional. `O14_LMHEAD_PROFILE=auto` is the default: it enables the sidecar path only when `lmhead_w8v2_sidecar.safetensors` exists under `MODEL_PATH`. Use `off` to force ordinary checkpoint behavior or `required` to fail closed when reproducing the measured W8-head profile.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 reproducibility/verify.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate.py
python3 -m unittest discover -s tests -v
shasum -a 256 -c SHA256SUMS

docker build -f docker/Dockerfile.repro -t glm52-o14-fast:local .
cp recipe/o14.env.example .env.o14-fast
set -a; source .env.o14-fast; set +a
bash recipe/serve-o14.sh render
```

Execution requires `O14_PROFILE_NAME=o14-fast`, `O14_PROFILE_STATUS=READY`, `O14_EXECUTE=1`, and the `serve` argument. Four-node Ray/RDMA formation, checkpoint placement, health checks, and rollback remain operator-controlled.
