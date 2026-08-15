# GLM-5.2 Harness O14 on 4× DGX Spark

> **Current named profiles:** **O14 Fast — 250K total KV, READY** on TP4/DCP1/PP1. **O14 Balanced** is a TP4/DCP2/PP1 500K target in `TESTING / DO NOT DEPLOY` status. Automation must read [`profiles/o14-profiles.json`](profiles/o14-profiles.json) and deploy only entries whose status is exactly `READY`, whose `deployable` field is `true`, and whose profile-specific artifact gates pass.
>
> **Historical benchmark record:** the retained speed rows below belong to the historical 399K O14 campaign. Balanced speed is `TBD`. The historical prose battery includes two C1 and two C4 runs with cache-busted prompts and a 1,024-output-token cap per stream. See the [full protocol and raw values](docs/BENCHMARKS.md).

This repository is the source-complete public reconstruction pack for the O14 runtime family. It now publishes an exact O14 Fast recipe and a fail-closed O14 Balanced phase-1 placeholder while preserving the historical O14 receipts unchanged.

The historical 2026-08-13 campaign measured a TP4/DCP1/PP1, 399K-context GLM-5.2 stack built on the vLLM 0.27 line. The complete public recipe contains **74 vLLM runtime files + 3 B12X files + a 14-path native patch**, exact binary-input hashes, and the launch contract. It includes sparse MLA, compact NVFP4 KV, adaptive MTP, full CUDA graphs, a custom Triton MLA BMM, an exact-rescore W8 head path, and an env-gated Marlin MoE change.

This repository contains the complete runtime assembly recipe, pinned public bases and wheels, exact manifests, the publishable serving command, sanitized benchmark protocols, and a provenance boundary. The two exact local wheel inputs are distributed as checksum-bound GitHub Release assets. Checkpoint and sidecar weights are not bundled. No public O14 Fast or O14 Balanced OCI image is claimed.

## Current profiles

| profile | status | topology | total logical KV | max request | image / build |
|---|---|---|---:|---:|---|
| **O14 Fast** | **READY** | TP4 / DCP1 / PP1 | **250K total KV** | 249000 | no public OCI image; source build from pinned commit |
| **O14 Balanced** | **TESTING / DO NOT DEPLOY** | TP4 / DCP2 / PP1 | 500K target; 500237 allocator expectation | 490000 | no image or build release; separate v3 image required after acceptance |

O14 Fast's exact machine recipe is allocator `250023`, `KV_CACHE_MEMORY_BYTES=7995534848` bytes per rank, `MAX_MODEL_LEN=249000`, `MAX_NUM_SEQS=4`, and `MAX_NUM_BATCHED_TOKENS=2048`. Its block arithmetic is documented in [`recipe/README.md`](recipe/README.md). Balanced uses an expected `8000000000` bytes per rank and `MAX_MODEL_LEN=490000`; measured capacity and speed remain `TBD` until live acceptance.

The public topology uses a 100GbE switch; 100000 Mb/s per active RDMA link is expected. Operators supply all addresses, interface/HCA names, device mappings, and host paths.

## Historical 399K O14 campaign results

The retained battery used cache-busted prompts and server counters to prevent prefix-cache inflation:

| workload | historical result |
|---|---:|
| cold prose, C1 decode | **25.40 / 25.61 tok/s** |
| cold prose, C4 aggregate decode | **54.68 / 53.62 tok/s** |
| predictable/code-class peak, C1 | **36.63 tok/s**, 3.265 accepted/step |
| predictable/code-class peak, C4 aggregate | **80.56 tok/s**, 3.934 accepted/step |
| cold prefill, 187,022 tokens | **661.1 tok/s** over 282.9 s |

Compared with the R15 starting point, the historical campaign reported roughly **+6–8% prose C1**, **+4–5% prose C4**, and **+2.5% cold prefill**. O14, o10, and o12-A remained inside the campaign's ±5% noise band at only one or two repeated batteries. This package does not claim that each O14 micro-change independently caused a statistically resolved end-to-end gain.

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) and [evidence/final-o14-battery.jsonl](evidence/final-o14-battery.jsonl).

## Historical runtime identity captured for the 399K campaign

This identity belongs to the historical benchmark evidence. The tag and digest below are not a public pullable image and must not be used as an image release claim.

| item | historical identity |
|---|---|
| hardware | 4× NVIDIA DGX Spark / GB10, one rank per node, RoCEv2 |
| parallelism | TP4 / DCP1 / PP1 |
| vLLM package | `0.27.2.dev0+g6e448d0ea.d20260812` |
| source line | vLLM `v0.27.1`, custom port at source commit `6e448d0ea` |
| PyTorch | `2.11.0+cu130` |
| Ray | `2.56.0` |
| FlashInfer | `0.6.15` |
| image tag | `glm52-r17:o14-20260813` |
| image digest | `sha256:d79f7410c475782bc00ee970c39a357e74206106ef247e57aea19607d7ebcb67` |
| model envelope | compressed-tensors W4A16/Int8Mix-derived hybrid; weights not included |
| serving envelope | 399,000 max model length, C≤4, 2,048 batch-token cap |
| KV | `nvfp4_ds_mla`, 12,792,889,000 bytes per rank |
| attention / MoE | `B12X_MLA_SPARSE` / Marlin |
| MTP | probabilistic draft; adaptive depths 2/4/5, window 32 |
| CUDA graphs | FULL capture sizes 6/12/18/24 |

All four observed ranks reported the same image digest and identical pre-publication hashes for the live source files. One private-path comment in `mla_attention.py` was redacted in the public copy without changing executable source. The health, model, metrics, source, environment, and worker-log evidence were inspected read-only. No running process was stopped, restarted, or changed.

## Important post-handoff correction

The Fable 5 handoff and campaign chart described O14 as using **Build A v1**. Direct inspection of all four live ranks proves that the deployed `mla_attention.py` imports `builda_bmm_v0`; `VLLM_BUILDA_BMM=1` is set and no v1 selector is present. Triton cache evidence also identifies v0 compilation. Therefore:

- **Build A v0 is the kernel active in the measured O14 battery.**
- Build A v1 exists in the image and passed separate graph-replay correctness and microbenchmark testing, but it was **not wired into the live O14 call path**.
- The v1 microbenchmarks are retained as measured kernel-development evidence, not credited as a cause of O14's end-to-end numbers.

The exact W8 top-64 rescore path was separately verified in worker logs: `FIRST RESCORE FIRED` appeared once on each of the four TP ranks.

## What is custom

The source-complete manifest replaces the campaign's historical release-count shorthand. It publishes a 74-file vLLM runtime overlay, a 3-file B12X overlay over a public base matching the other 123 installed files, and a 14-path native source/build patch, including:

- B12X sparse MLA, indexer integration, and the int64 sparse-index repair;
- compact `nvfp4_ds_mla` KV dispatch, accounting, and attention guards;
- adaptive MTP depth 2/4/5 with telemetry and block rejection sampling;
- decode-aware prefill and full-graph shape support;
- compressed/quantized draft checkpoint loading repairs;
- the W8 lm-head top-64 exact-rescore path, verified firing on four ranks;
- Build A v0, the active custom Triton MLA-absorb BMM;
- Build A v1, a graph-retuned candidate included and tested but not live-wired;
- an env-gated Marlin MoE atomic-add selection;
- an exactness-preserving draft-temperature knob, shipped at its no-op `1.0` setting;
- a FlashInfer compatibility shim for the rebuilt runtime.

The campaign also tested and rejected several paths instead of hiding them: a correct but slow CUTE-DSL BMM, a dense W8A16 Triton replacement that passed **71/71 correctness checks**, won 0/7 relevant shapes, and projected a regression, an MTP controller that lost in causal replay, and an unloadable `eh_proj` INT8 arm.

The exact live source delta, B12X delta, native patch, manifests, build lock, and fail-closed verifier are in [`reproducibility/`](reproducibility/). The legacy [`overlays/`](overlays/), [`kernels/`](kernels/), [`runtime/`](runtime/), and [`patches/`](patches/) paths remain convenient views of six already-published O14 files.

## Upstream contribution status

See [`docs/VLLM_UPSTREAMING.md`](docs/VLLM_UPSTREAMING.md) for the frozen
O14-to-vLLM feature catalogue, attribution boundaries, ranked contribution
sequence, evidence requirements, and no-spam engagement plan.

## Repository map

- [`recipe/serve-o14.sh`](recipe/serve-o14.sh): render-first server command for O14 Fast.
- [`recipe/o14.env.example`](recipe/o14.env.example): required and performance-relevant environment.
- [`docker/Dockerfile.o14-overlay`](docker/Dockerfile.o14-overlay): legacy six-file convenience layer; not the source-complete route.
- [`docker/Dockerfile.repro`](docker/Dockerfile.repro): complete checksum-bound public runtime assembly recipe.
- [`reproducibility/`](reproducibility/): exact 74/3/14 source pack, manifests, build lock, dependency snapshot, and verifier.
- [`docs/PUBLIC_BUILD.md`](docs/PUBLIC_BUILD.md): build procedure and claim boundary.
- [`kernels/builda_bmm_v0.py`](kernels/builda_bmm_v0.py): exact active live Build A source.
- [`kernels/builda_bmm_v1.py`](kernels/builda_bmm_v1.py): tested graph-retuned candidate present in the image but not live-wired.
- [`overlays/`](overlays/): exact source captured from the live O14 image, retaining upstream SPDX attribution.
- [`runtime/r17_shim.py`](runtime/r17_shim.py): idempotent FlashInfer compatibility shim.
- [`benchmarks/`](benchmarks/): sanitized final-battery probe protocols.
- [`docs/PORT_LINEAGE.md`](docs/PORT_LINEAGE.md): vLLM 0.27 migration surfaces and rejected work.
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md): Fable 5 campaign results, microbenchmarks, and limits.
- [`evidence/o14-results.json`](evidence/o14-results.json): machine-readable runtime and measurements.

## Use

```bash
PYTHONDONTWRITEBYTECODE=1 python3 reproducibility/verify.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate.py
python3 -m unittest discover -s tests -v
shasum -a 256 -c SHA256SUMS

docker build -f docker/Dockerfile.repro -t glm52-o14-fast:local .

cp recipe/o14.env.example .env.o14-fast
# Fill MODEL_PATH and supply your own rank/fabric topology.
set -a; source .env.o14-fast; set +a
bash recipe/serve-o14.sh render
```

The serving script renders by default. It executes only when `O14_PROFILE_NAME=o14-fast`, `O14_PROFILE_STATUS=READY`, and `O14_EXECUTE=1` are combined with the `serve` argument. It refuses unknown or non-READY profiles. The runtime accepts the documented GLM-5.2 QuantTrio checkpoint family; `O14_LMHEAD_PROFILE=auto` makes the W8 sidecar optional and enables exact-head rescoring only when that file is present. Four-node Ray/RDMA formation, checkpoint placement, health checks, and rollback remain operator responsibilities.

## Claim boundary

Runtime recipe publication is complete for the known O14 surfaces: exact public bases and target hashes reconstruct all 74 vLLM runtime files, 3 B12X files, the 14-path native patch, and the FlashInfer compatibility edit. **O14 Fast — 250K total KV, READY** is source-built from this repository; no public OCI image or anonymous-pull claim exists. O14 Balanced remains TESTING with no released build or image. Exact historical benchmark replication further requires the same checkpoint family and W8 sidecar, four GB10 nodes, equivalent topology, TP4, the historical 399K NVFP4 KV envelope, C≤4, and the stated sampling/graph policy. The retained speed numbers are historical campaign evidence, not a general vLLM benchmark.

Apache-2.0. Third-party projects and checkpoint weights retain their own licenses.
