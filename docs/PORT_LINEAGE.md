# vLLM 0.27 port lineage

## Rebuild base

O14 runs `vllm 0.27.2.dev0+g6e448d0ea.d20260812`, rebuilt from the vLLM 0.27.1 line. The migration moved the prior GLM-5.2 runtime onto changed scheduler, speculative-decode, CUDA-graph, attention, and KV APIs instead of freezing the old fork.

The campaign's initial port map examined 23 payload surfaces:

| area | source surfaces audited or ported |
|---|---|
| adaptive MTP | acceptance length, depth ladder, metrics, base/autoregressive speculators, and spec-decode utilities |
| scheduler | core/async scheduler, scheduler output, config, and argument plumbing |
| full CUDA graphs | graph utilities, warmup, model runner, and vLLM config |
| sparse MLA | B12X backend, MLA indexer, and attention wiring |
| distributed execution | DCP all-to-all, context-parallel utilities, and parallel state/config |
| structured output | initialization and scheduler compatibility |

A final port lane handled additional load-bearing surfaces:

- `_custom_ops.py`: fail-closed `nvfp4_ds_mla` cache operation;
- `mla_attention.py`: compact-KV reinterpretation guard and Build A integration;
- `deepseek_mtp.py`: packed-module mapping for the quantized draft path;
- `kv_cache_interface.py`: NVFP4 quant-mode recognition and 368-byte/token page accounting;
- DFlash speculator signature compatibility.

The campaign's consolidated **26-file** release count and a separate **27-Python-file** audit were historical, overlapping scopes and cannot be compared directly. The exact public inventory supersedes those labels: **74 vLLM files (67 modified + 7 added), 3 modified B12X files over a public base matching the other 123 installed files, and 14 native source/build paths**. Every target has a base and target hash in a fail-closed manifest.

## Custom runtime work that reached O14

### B12X sparse MLA and indexer

The custom B12X stack is Python/CUTE-DSL source JIT-compiled for the target GPU rather than an opaque binary blob. The 0.27 port registered `B12X_MLA_SPARSE`, rewrote indexer integration against the changed sparse-MLA API, preserved compact KV, and carried the int64 sparse-index repair.

### NVFP4 compact KV

`nvfp4_ds_mla` required more than a command-line dtype. The port added custom cache-op dispatch, quantized-KV classification, page-size accounting at 368 bytes/token, and attention guards preventing compact layout reinterpretation as generic fp8. O14 reserves 12,792,889,000 bytes per rank and reports roughly 400K-token capacity for a 399K request limit.

### Build A Triton BMM

The active live path is `kernels/builda_bmm_v0.py`, imported by the exact deployed `overlays/mla_attention.py`. V1 was separately graph-retuned, correctness-tested, and present in the image, but direct four-rank inspection proved it was not selected in the measured O14 battery. See `docs/BENCHMARKS.md` for both the valid v1 microbenchmarks and the attribution correction.

### Marlin MoE path

A trace measured Marlin MoE at 32.1 ms/step. O14 exposes two WNA16 Marlin call sites through the import-time `VLLM_MOE_MARLIN_ATOMIC_ADD` switch and runs it enabled. The exact live overlay and fail-closed applicator are included.

### Exact W8 head path

The checkpoint lineage uses an INT8/W8 lm-head sidecar with top-64 exact rescoring. The v2.1 implementation is fail-loud and emits one first-fire record. Publication-time inspection recovered `FIRST RESCORE FIRED` with nonzero changed-entry counts on all four TP ranks. The sidecar weights remain private checkpoint artifacts.

### Adaptive MTP and block rejection

The runtime uses probabilistic drafting, adaptive depths 2/4/5, a 32-step controller window, and block rejection sampling. Full CUDA graphs cover the resulting verification shapes. The draft-temperature scale is present but set to `1.0`; no gain is attributed to it.

## Work tested and rejected

- The CUTE-DSL SIMT BMM was correct and capture-safe but slower than Triton and cuBLAS.
- A dense W8A16 Triton candidate passed 71/71 correctness tests but won 0/7 relevant shapes and projected a regression.
- Humming and TensorRT-LLM MoE paths rejected the checkpoint's group-128 compressed-tensors layout.
- A causal throughput-argmax MTP controller lost to the incumbent in replay.
- `eh_proj` INT8 was not loadable with the existing ignore-list/layout and was removed.
- The trace showed NCCL around 10.6 ms/step, invalidating an earlier theory that communication alone explained fixed decode cost.

## Public reconstruction boundary

The complete known O14 runtime source surfaces are now in `reproducibility/`. The runtime overlay is bound to the exact R17 stage-0 wheel from public vLLM commit `6e448d0e`; the B12X overlay is bound to public commit `334a2d75`; and the native patch is bound to public vLLM commit `e232d262`. The two vLLM lines overlap on seven paths and remain separate.

`docker/Dockerfile.repro` is the complete pinned assembly recipe, not a build receipt. The checkpoint, W8 sidecar, four-node topology, and runtime qualification required to replicate the benchmark remain outside the public source pack.
