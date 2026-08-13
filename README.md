# GLM-5.2 Harness O14 on 4× DGX Spark

Public reconstruction pack for the O14 configuration measured and left serving on a four-node DGX Spark/GB10 cluster on 2026-08-13.

O14 is a TP4, 399K-context GLM-5.2 serving stack built on the vLLM 0.27 line with a 26-file custom runtime port, sparse MLA, compact NVFP4 KV, adaptive MTP, full CUDA graphs, a custom Triton MLA BMM, an exact-rescore W8 head path, and an env-gated Marlin MoE change.

This repository contains the publishable recipe, live-derived O14 overlay sources, sanitized benchmark protocols, raw final-battery values recovered from the Fable 5 session transcript, and a provenance boundary. It does **not** contain checkpoint weights, the private base image, compiled artifacts, host addresses, credentials, or a turnkey cluster controller.

## Measured O14 results

The final battery used cache-busted prompts and server counters to prevent prefix-cache inflation:

| workload | result |
|---|---:|
| cold prose, C1 decode | **25.40 / 25.61 tok/s** |
| cold prose, C4 aggregate decode | **54.68 / 53.62 tok/s** |
| predictable/code-class peak, C1 | **36.63 tok/s**, 3.265 accepted/step |
| predictable/code-class peak, C4 aggregate | **80.56 tok/s**, 3.934 accepted/step |
| cold prefill, 187,022 tokens | **661.1 tok/s** over 282.9 s |

Compared with the R15 starting point, the campaign reports roughly **+6–8% prose C1**, **+4–5% prose C4**, and **+2.5% cold prefill**. O14, o10, and o12-A remain inside the campaign's ±5% noise band at only one or two repeated batteries. This package does not claim that each O14 micro-change independently caused a statistically resolved end-to-end gain.

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) and [evidence/final-o14-battery.jsonl](evidence/final-o14-battery.jsonl).

## Live runtime identity captured for this upload

| item | identity |
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

The vLLM 0.27 rebuild carried a campaign-documented 26-file runtime port, including:

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

The exact live sources, kernels, reconstruction layer, and fail-closed applicator are in [`overlays/`](overlays/), [`kernels/`](kernels/), [`runtime/`](runtime/), [`docker/`](docker/), and [`patches/`](patches/). The broader 0.27 port is documented rather than falsely presented as a clean stock-vLLM patch: several surfaces were rewritten against changed APIs, and the private pre-O14 base image is not published.

## Repository map

- [`recipe/serve-o14.sh`](recipe/serve-o14.sh): render-first server command matching the live envelope.
- [`recipe/o14.env.example`](recipe/o14.env.example): required and performance-relevant environment.
- [`docker/Dockerfile.o14-overlay`](docker/Dockerfile.o14-overlay): public O14 reconstruction layer, parameterized by a compatible base image.
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
cp recipe/o14.env.example .env.o14
# Fill MODEL_PATH and confirm your compatible image/runtime.
set -a; source .env.o14; set +a
bash recipe/serve-o14.sh render
```

The script renders by default. It executes only with `O14_EXECUTE=1` and refuses to run if the model path is missing. A four-node Ray cluster, RDMA device mapping, image distribution, checkpoint placement, and rollback controller remain operator responsibilities.

## Claim boundary

These numbers apply to this checkpoint family, four GB10 nodes, TP4, 399K NVFP4 KV, C≤4, and the stated sampling/graph policy. They are not a general vLLM benchmark. Final values are preserved both in the append-only Fable 5 campaign ledger at source commit `698b9085c3aed47de5513a720204ed788b607f6a` and as exact tool-output values recovered from its session transcript. Where the prose handoff conflicted with the deployed source, the four-rank live source inspection controls.

Apache-2.0. Third-party projects and checkpoint weights retain their own licenses.
