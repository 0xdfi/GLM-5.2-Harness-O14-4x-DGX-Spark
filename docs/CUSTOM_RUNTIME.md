# Custom vLLM 0.27 runtime and kernels

## Version lineage

The live package reports `vllm 0.27.2.dev0+g6e448d0ea.d20260812`. Its runtime Python/source line is reconstructed from public vLLM 0.27.1 commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` plus the exact checked-in 74-file overlay.

The port was not a blind file copy. The initial campaign map classified 23 payload files as five copy/new-file cases, eight re-anchored diffs, and ten rewrites against changed 0.27 APIs; a final lane handled five additional load-bearing surfaces, with overlap between inventories. The campaign's 26-file release count and a separate 27-Python-file audit were historical, incomparable scopes. The exact public manifest supersedes both: **74 vLLM files (67 modified + 7 added), 3 modified B12X files over a public base matching the other 123 installed files, and 14 native source/build paths**.

The native writer/build line is independently pinned to public vLLM commit `e232d262369b8c918cf478a7a96a0fcf8127cf65`. It overlaps the runtime overlay on seven paths and is kept as a separately hashed patch instead of being presented as one clean Git queue.

## Functional surfaces

- **Sparse attention:** `B12X_MLA_SPARSE`, sparse indexer plumbing, custom MLA backend registration, and the int64 sparse-index repair.
- **KV format:** `nvfp4_ds_mla` cache-op wiring, quant-mode recognition, 368-byte/token page accounting, and compact-layout attention guards.
- **Speculative decode:** adaptive MTP 2/4/5, a 32-step controller window, probabilistic drafting, block rejection sampling, and telemetry.
- **Scheduling:** decode-aware prefill with a 256-token decode-time budget and a 2,048-token idle budget.
- **CUDA graphs:** full-graph support for adaptive verification shapes, with capture sizes 6/12/18/24.
- **Checkpoint/runtime repair:** quantized-draft packed-module mapping and a W8 lm-head exact-rescore path.

## Build A v0: active in measured O14

`kernels/builda_bmm_v0.py` is byte-identical to the module on every live rank. `overlays/mla_attention.py` imports this v0 module and gates its two MLA-absorb batched matrix multiplications with `VLLM_BUILDA_BMM=1`. The live environment had that flag enabled.

The kernel uses bf16 inputs/outputs, fp32 accumulation, arbitrary A/output strides, static shape dispatch, and no host synchronization in the launch path. Calls outside its verified shape/dtype envelope fall back to `torch.bmm`.

## Build A v1: tested, present, not live-wired

`kernels/builda_bmm_v1.py` re-tunes the same two tiny BMMs under CUDA-graph replay:

- BMM1 `(16,B,192) × (16,192,512)`: `BLOCK_N=32`, `BLOCK_K=32`, 2 warps, 2 stages.
- BMM2 `(16,B,512) × (16,512,256)`: `BLOCK_N=32`, `BLOCK_K=64`, 4 warps, 3 stages.

It passed B=3..6 contiguous/strided correctness and separate graph-replay microbenchmarks. The module is present in the live image, but the deployed wrapper imports v0 and no `VLLM_BUILDA_VER` selector exists in the live launch environment. The original Fable handoff calling O14 “v1” was therefore incorrect. This repository preserves v1 as tested development work without assigning it causal credit for the final O14 battery.

## W8 exact-rescore head

`overlays/logits_processor.py` implements a top-M exact rescore over a quantized full-vocabulary lm-head. O14 uses top 64 candidates per rank, recomputes those rows from the bf16 sidecar with fp32 accumulation, and scatters exact values before sampling/argmax. `VLLM_LMHEAD_V2_REQUIRE=1` turns missing worker environment or a bypassing dtype branch into a hard failure.

Read-only worker-log inspection recovered one `FIRST RESCORE FIRED` line from each of the four TP ranks, with nonzero changed-entry counts of 84, 100, 112, and 114. Unlike the Build A version label, this component's final-battery activation is directly evidenced.

## Marlin MoE atomic-add overlay

`overlays/marlin_moe.py`, captured directly from the live image, changes two literal `use_atomic_add=False` calls to a module-level, import-time switch:

```text
VLLM_MOE_MARLIN_ATOMIC_ADD=1
```

Default is off. O14 runs it on. `patches/apply_o14_overlays.py` reproduces the same delta and requires exactly two matching call sites, aborting on any base-source mismatch.

## Draft-temperature overlay

`overlays/speculator.py` adds `R17_DRAFT_TEMP_SCALE`, read once at import and applied to draft sampling temperature. The live value is `1.0`, an exact no-op. Rejection sampling continues to use the target distribution; this knob was staged for acceptance-economics experiments and receives no O14 speed credit.

## FlashInfer compatibility shim

`runtime/r17_shim.py` idempotently adds a no-op `set_autotune_process_group` only when the older FlashInfer package lacks it. The consequence is per-rank tactic selection rather than cross-rank timing averaging. It is a compatibility repair, not a speed claim.

## Custom-kernel work rejected during the campaign

- A CUTE-DSL SIMT version of the tiny BMM was correct and capture-safe but measured 22.9/56.0 µs, so Triton remained the candidate.
- A dense W8A16 Triton replacement passed 71/71 correctness checks but won 0 of 7 relevant shapes against Marlin. The campaign projected a 2.9 ms/step regression and killed it.
- Humming and TensorRT-LLM MoE paths rejected the group-128 compressed-tensors layout.
- `eh_proj` INT8 could not load under the existing quantization ignore-list/layout and offered an estimated prize of only about 0.3%.

## Public reconstruction and build boundary

`reproducibility/verify.py` fail-closes the complete checked-in source surfaces: 74 vLLM targets, 3 B12X targets, and 14 native-build targets. `docker/Dockerfile.repro` is the public ARM64 assembly recipe and leaves a clean image build explicitly unproven. The legacy `Dockerfile.o14-overlay` remains a convenience view of the six previously published top-layer files.

No native wheel or container image was built while preparing this pack. A future build must reconcile the separate runtime/native vLLM lines, resolve dependencies, prove package-owned native operators, and pass numerical, graph, long-context, and distributed gates. Exact benchmark replication additionally requires the same private checkpoint and W8 sidecar plus the same four-node hardware/topology.
