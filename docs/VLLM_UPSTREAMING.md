# O14 → vLLM upstream contribution catalogue

This document separates reusable upstream candidates from O14 deployment glue,
third-party-derived work, superseded ports, and negative experiments. It is an
inspection snapshot, not a claim about vLLM after the frozen revision and not a
promise that any proposal will be accepted.

## Frozen scope

- **Research snapshot:** 2026-08-14 02:03 UTC (2026-08-13 22:03 EDT).
- **O14 repository:**
  [`df71121ba10859643057ca0eb5a84597d81505ec`](https://github.com/0xdfi/GLM-5.2-Harness-O14-4x-DGX-Spark/commit/df71121ba10859643057ca0eb5a84597d81505ec).
- **O14 vLLM base:**
  [`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`](https://github.com/vllm-project/vllm/commit/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac),
  packaged as `0.27.2.dev0+g6e448d0ea.d20260812`.
- **vLLM `main` inspected:**
  [`fe4c5dcd4cc71c2c1dc05375be497e413425aad7`](https://github.com/vllm-project/vllm/commit/fe4c5dcd4cc71c2c1dc05375be497e413425aad7).

The O14 base and frozen `main` had diverged. Status below means “observed at the
frozen vLLM SHA”; every branch, issue, PR, owner, and CI requirement must be
rechecked immediately before a human acts.

## vLLM contribution rules at the frozen revision

The authoritative frozen contribution entry points are vLLM's
[`docs/contributing/README.md`](https://github.com/vllm-project/vllm/blob/fe4c5dcd4cc71c2c1dc05375be497e413425aad7/docs/contributing/README.md),
[AI-assisted contributor instructions in `AGENTS.md`](https://github.com/vllm-project/vllm/blob/fe4c5dcd4cc71c2c1dc05375be497e413425aad7/AGENTS.md),
and the required fields in the
[pull-request template](https://github.com/vllm-project/vllm/blob/fe4c5dcd4cc71c2c1dc05375be497e413425aad7/.github/PULL_REQUEST_TEMPLATE.md).

- **DCO/signoff:** every commit requires a `Signed-off-by:` line (`git commit
  -s`). No separate repository CLA was found in the frozen tree; DCO was the
  operative contribution agreement.
- **CI and PR evidence:** the required branch checks were `DCO`,
  `buildkite/ci/pr`, and `pre-commit`. CI is not universally automatic; a
  trusted reviewer normally triggers `/ci run`. The PR must include purpose,
  issue linkage, a test plan with commands, actual results, and before/after or
  end-to-end evidence where applicable. Documentation updates are required when
  the change modifies user-facing behavior.
- **AI assistance:** pure code-agent PRs are **not allowed**. The human submitter
  must understand and defend the change end-to-end, review every changed line,
  and run the relevant tests. An AI-assisted PR description must explain why
  the work does not duplicate an existing PR, list test commands and results,
  include model-evaluation results when output, accuracy, or serving is
  affected, and clearly disclose the AI assistance. Non-trivial AI-generated
  code also needs commit-trailer attribution such as `Co-authored-by:` or
  `Assisted-by:`, alongside the DCO `Signed-off-by:` trailer. AI assistance is
  not, by itself, a claim of authorship or copyright ownership.
- **RFC threshold:** architectural changes **>500 LOC**, excluding kernel, data,
  config, and test code, are expected to have a prior
  [RFC issue](https://github.com/vllm-project/vllm/blob/fe4c5dcd4cc71c2c1dc05375be497e413425aad7/.github/ISSUE_TEMPLATE/750-RFC.yml).
  Smaller changes may still warrant an RFC when they introduce a public format,
  scheduler policy, or cross-subsystem contract.
- **Kernel bar:** a custom kernel needs registered schema and implementation,
  fake/meta support for Tensor outputs, matching signatures, `torch.library`
  `opcheck`, correctness and fallback tests, and documentation when behavior is
  user-facing. Relevant title prefixes include `[Kernel]`, `[Core]`, `[Model]`,
  `[Bugfix]`, and `[CI/Build]`.

## Catalogue

Action labels are deliberately strict: **New PR**, **RFC after evidence**,
**Watch / no comment now**, **No contact without new evidence**, **Hold**, or
**No**. The **Public O14 source / evidence** column distinguishes source availability
from execution evidence. The exact 74-file vLLM, 3-file B12X, and 14-path
native manifests are public; some targeted traces, checkpoint fixtures, and
causal A/B evidence remain unavailable.

| Feature family | Ownership / provenance | Public O14 source / evidence | Active in measured O14? | Frozen vLLM-main status | Upstream action and public thread | Missing proof before action |
|---|---|---|---|---|---|---|
| **vLLM 0.27 custom-port shell** | vLLM-derived source; 0.27 re-port and reconciliation by 0xdfi/Nous; individual features retain the donor lineages below. | Public source/evidence: the exact [74-file vLLM manifest](../reproducibility/manifests/vllm-overlay.json), [runtime overlay](../reproducibility/overlays/vllm/vllm), [offline materializer](../reproducibility/verify.py), [port lineage](PORT_LINEAGE.md), [serve recipe](../recipe/serve-o14.sh), and [runtime record](../evidence/o14-results.json). The historical 26-release-file and 27-Python-file counts were incomparable scopes; the exact public manifest supersedes them. | **Yes**, as the runtime shell; source identity is now exact, while public build/runtime replay remains unproven. | Not a comparable feature on main. The old APIs and the frozen main have diverged, so a wholesale file copy would be stale. | **No** monolithic PR. Decompose and reimplement only current defects. | Public build/import receipt, dependency closure, current-main reproduction, and upstream-style tests. |
| **Scheduler async-preemption race repair** | Original 0xdfi/Nous correctness repair in a vLLM-derived scheduler. | The exact O14 [scheduler source](../reproducibility/overlays/vllm/vllm/v1/core/sched/scheduler.py) is public in the 74-file manifest. The targeted race trace and a current-main reproducer are not published. | **Yes.** The guarded request-index lookup was present; runtime survival is supporting evidence, not a targeted race proof. | **Unresolved:** scheduler structure changed, and the audit did not establish that the race still reproduces in the [frozen scheduler](https://github.com/vllm-project/vllm/blob/fe4c5dcd4cc71c2c1dc05375be497e413425aad7/vllm/v1/core/sched/scheduler.py). | **New PR** only after current-main reproduction. | Minimal failing unit test, current-main RED/GREEN proof, concurrency stress test, and proof that request snapshots do not already close the race. |
| **Build A v0 / v1 tiny MLA BMM** | Original 0xdfi/Don/Nous campaign kernel work. | Public source: [`builda_bmm_v0.py`](../kernels/builda_bmm_v0.py) (`_bmm_tiny_kernel`, `bmm_v0`, `builda_bmm`), [`builda_bmm_v1.py`](../kernels/builda_bmm_v1.py), and [`mla_attention.py`](../overlays/mla_attention.py); activation/hash evidence: [`o14-results.json`](../evidence/o14-results.json) and [benchmarks](BENCHMARKS.md#build-a-v1-kernel-microbenchmarks). | **v0 yes; v1 no.** v0 was live-wired on all ranks. v1 has better isolated B=3–6 graph-replay timings but was present only as tested development work. | Main still used `torch.bmm` at the two relevant call sites ([first](https://github.com/vllm-project/vllm/blob/fe4c5dcd4cc71c2c1dc05375be497e413425aad7/vllm/model_executor/layers/attention/mla_attention.py#L919), [second](https://github.com/vllm-project/vllm/blob/fe4c5dcd4cc71c2c1dc05375be497e413425aad7/vllm/model_executor/layers/attention/mla_attention.py#L1188)). Neither O14 file is upstream-ready unchanged. | **New PR**, rewritten as a generic guarded custom op. It may use v1 tuning data and v0 integration proof, but not claim v1 powered O14. **Comment now** on [`vLLM#36297`](https://github.com/vllm-project/vllm/pull/36297) only with adjacent tiny-BMM data that distinguishes live v0 from microbenchmarked v1 and states the branch-validation boundaries. | Current-main/Torch rerun; full dtype/device/shape/stride/architecture guards; guaranteed fallback; fake/meta and `opcheck`; numerical, canary, compile, and full-CG replay tests; raw repeated benchmarks; paired end-to-end A/B. |
| **W8 `lm_head` top-64 selected-row rescore** | Original 0xdfi/Don/Nous logic in modified vLLM logits processing. | Public source: [`logits_processor.py`](../overlays/logits_processor.py) (`_lmhead_v2_init`, `_lmhead_v2_rescore`); four-rank fire evidence: [`o14-results.json`](../evidence/o14-results.json) and the [custom-runtime record](CUSTOM_RUNTIME.md#w8-exact-rescore-head). Sidecar weights are **not published**. | **Yes.** First-fire records appeared on all four TP ranks, and selected candidate rows changed. | The frozen [`LogitsProcessor`](https://github.com/vllm-project/vllm/blob/fe4c5dcd4cc71c2c1dc05375be497e413425aad7/vllm/model_executor/layers/logits_processor.py#L90-L128) had no selected-row rescore hook. The O14 sidecar and environment contract are checkpoint-specific. | **Watch / no comment now** on [`vLLM#48870`](https://github.com/vllm-project/vllm/pull/48870): it is a different fused tied-head mechanism, and O14 lacks a public sidecar fixture and candidate-recall proof. Reconsider after both exist or if a maintainer asks. | Public sidecar fixture and format; TP/sharding and sampling tests; graph warmup/replay; causal A/B; adversarial candidate-recall tests. Exactness is only for selected rows, not proof of a full-vocabulary winner. |
| **Marlin fused-MoE atomic add** | Original O14 policy switch around an operation already present in vLLM; modified file retains vLLM lineage. | Public source: [`marlin_moe.py`](../overlays/marlin_moe.py) and fail-closed [`apply_o14_overlays.py`](../patches/apply_o14_overlays.py); [runtime/trace record](CUSTOM_RUNTIME.md#marlin-moe-atomic-add-overlay). No atomic-off/on evidence exists. | **Yes.** Both call sites ran with the switch enabled. No causal on/off result exists. | Main still hardcoded `False` at the [first](https://github.com/vllm-project/vllm/blob/fe4c5dcd4cc71c2c1dc05375be497e413425aad7/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py#L145-L160) and [second](https://github.com/vllm-project/vllm/blob/fe4c5dcd4cc71c2c1dc05375be497e413425aad7/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py#L202-L228) sites, while [`vLLM#48569`](https://github.com/vllm-project/vllm/pull/48569) already proposed a safer shape/device/dtype-aware policy. [`vLLM#45660`](https://github.com/vllm-project/vllm/pull/45660) tracks accumulator initialization relevant to graph safety. | **No contact without new evidence** on [`vLLM#48569`](https://github.com/vllm-project/vllm/pull/48569): it already has stronger GB10 atomic-off/on and paired-layer evidence, while O14's reachability/stability-only result adds no causal evidence. Do not submit a competing PR. | Numerical and overlapping-write safety matrix, full-CG zero-initialization proof, supported-shape coverage, and matched atomic-off/on A/B. |
| **Adaptive MTP depths 2/4/5 and telemetry** | Controller lineage: Aiden Le; local-inference-lab work by Luke Alonso and Martin Vit; CosmicRaisins forward port/tuning and telemetry; 0xdfi/Nous parser/API reconciliation and 0.27 wiring. | Public config/evidence: [`serve-o14.sh`](../recipe/serve-o14.sh), [`o14-results.json`](../evidence/o14-results.json), and [port lineage](PORT_LINEAGE.md#adaptive-mtp-and-block-rejection); [`speculator.py`](../overlays/speculator.py) publishes a related speculator surface. The adaptive [acceptance controller](../reproducibility/overlays/vllm/vllm/v1/spec_decode/dynamic/acceptance_length.py), [depth ladder](../reproducibility/overlays/vllm/vllm/v1/spec_decode/dynamic/depth_ladder.py), scheduler policy, and telemetry implementation are public in the exact 74-file vLLM overlay; a fixed-depth causal control remains absent. | **Yes.** Ladder 2/4/5, window 32, telemetry, and full-graph shapes were active. No fixed-depth control was retained. | Main already had dynamic speculative scheduling and [adaptive verification](https://github.com/vllm-project/vllm/blob/fe4c5dcd4cc71c2c1dc05375be497e413425aad7/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py); the exact O14 rolling policy was absent and should not become a parallel framework. | **Watch / no comment now** on [`vLLM#47111`](https://github.com/vllm-project/vllm/pull/47111): it needs a rebase and is inactive, while O14 now publishes the controller source but still lacks a fixed-K A/B. Reconsider only if the PR reactivates and O14 has material public source or evidence. | Static K2/K4/K5 controls; stability/oscillation and multi-request fairness tests; per-position metric semantics; telemetry overhead/privacy review; current-MRV2 rebase; repeated goodput evidence. |
| **O14 probabilistic draft sampling** | Native/vLLM-derived draft-sampling mode; O14 selected it and carried the 0.27 integration. It is not an O14-original sampling algorithm, and the separate O14 temperature-scale modification is catalogued below. | Public source/config: [`speculator.py`](../overlays/speculator.py) (`DraftModelSpeculator.sample_draft`, `_copy_request_inputs`) and [`serve-o14.sh`](../recipe/serve-o14.sh) (`draft_sample_method`); the [runtime identity](../README.md#live-runtime-identity-captured-for-this-upload) and [port lineage](PORT_LINEAGE.md#adaptive-mtp-and-block-rejection) record the selected mode. | **Yes.** The measured profile selected `draft_sample_method=probabilistic`; no greedy control or branch-level first-fire receipt was retained. | Already native at the frozen SHA: [`SpeculativeConfig`](https://github.com/vllm-project/vllm/blob/fe4c5dcd4cc71c2c1dc05375be497e413425aad7/vllm/config/speculative.py) exposed `draft_sample_method={greedy,probabilistic}`. O14 did not establish a missing-main feature. | **No** contribution, issue, or comment; there is no O14-specific thread to open. Do not fold this config-only claim into adaptive-MTP [`#47111`](https://github.com/vllm-project/vllm/pull/47111) or block-rejection [`#46781`](https://github.com/vllm-project/vllm/pull/46781). | Public resolved-runtime/first-fire receipt; greedy-versus-probabilistic A/B; quality and acceptance evidence; and proof that the exact shaped draft distribution `q` reaches acceptance/residual math on the tested branch. |
| **Full-CUDA-graph adaptive verification shapes** | vLLM-derived dynamic-speculative/CUDA-graph plumbing ported and reconciled to the 0.27 runtime by 0xdfi/Nous; the exact capture-size list is O14 deployment tuning, not a new generic graph algorithm. | Public config/evidence: [`serve-o14.sh`](../recipe/serve-o14.sh) (`cudagraph_capture_sizes`), [`o14-results.json`](../evidence/o14-results.json), the [runtime identity](../README.md#live-runtime-identity-captured-for-this-upload), and [custom-runtime inventory](CUSTOM_RUNTIME.md#functional-surfaces). The exact [graph utilities](../reproducibility/overlays/vllm/vllm/v1/worker/gpu/cudagraph_utils.py), [warmup](../reproducibility/overlays/vllm/vllm/v1/worker/gpu/warmup.py), and [model runner](../reproducibility/overlays/vllm/vllm/v1/worker/gpu/model_runner.py) source is public. | **Yes:** FULL mode with exact active capture sizes **6/12/18/24**. O14 was TP4/DCP1/PP1; DCP1 does **not** validate data parallelism (DP). | Superseded as reusable graph plumbing: dynamic speculative decoding gained full-CUDA-graph support in [`07516fda67d2133e26c0fd7386c0b0c8641e2a6e`](https://github.com/vllm-project/vllm/commit/07516fda67d2133e26c0fd7386c0b0c8641e2a6e). At the frozen SHA, dynamic speculative decoding nevertheless remained disabled under DP by [`93e2ab71119ff08805adc93be75196450382b088`](https://github.com/vllm-project/vllm/commit/93e2ab71119ff08805adc93be75196450382b088); O14 DCP1 is not evidence against that limitation. | **No** fixed-shape PR or comment; merged [`#45953`](https://github.com/vllm-project/vllm/pull/45953) already carries dynamic-speculation full-CG work. **Hold** any DP-enablement proposal until it is reproduced and designed on current main under actual DP. | Public O14 graph-plumbing source; current-main capture/replay over every reachable `(depth, request-count)` shape; actual DP>1 multi-rank evidence; fallback/recapture and shape-closure tests. DCP1 is not DP proof. |
| **Decode-aware prefill budgeting** | Third-party-derived: `penguinchang` concept, `ciprianveg` packaging/distribution, OsakaTX port/validation; 0xdfi/Nous 0.27 re-port. | Public config/docs: [`serve-o14.sh`](../recipe/serve-o14.sh) (256/2,048/one-long-prefill flags), [custom-runtime inventory](CUSTOM_RUNTIME.md#functional-surfaces), and [port lineage](PORT_LINEAGE.md#rebuild-base). The exact [scheduler implementation](../reproducibility/overlays/vllm/vllm/v1/core/sched/scheduler.py) is public; matched causal workload evidence remains absent. | **Yes:** active budget 256, idle budget 2,048, one long prefill per step. | The frozen [scheduler](https://github.com/vllm-project/vllm/blob/fe4c5dcd4cc71c2c1dc05375be497e413425aad7/vllm/v1/core/sched/scheduler.py) had related chunking/throttling controls but not this dual active/idle policy. [`vLLM#33743`](https://github.com/vllm-project/vllm/pull/33743) was closed; [`vLLM#44794`](https://github.com/vllm-project/vllm/pull/44794) uses a different mechanism. | **RFC after evidence.** No initial comment on adjacent threads. | Matched mixed-load A/B with decode TPOT/ITL, TTFT, prefill throughput, and fairness/starvation; faithful waiting-queue behavior; chunked-prefill, prefix-cache, TP/PP/DP interaction tests. |
| **B12X sparse MLA, sparse indexer, and int64/direct-K repair** | B12X/local-inference-lab and Luke Alonso; vLLM 0.27 integration by 0xdfi/Nous. The int64 repair is Luke Alonso/local-inference-lab work, not Nous-original. | Public selection/guard/evidence: [`serve-o14.sh`](../recipe/serve-o14.sh), [`mla_attention.py`](../overlays/mla_attention.py), [`o14-results.json`](../evidence/o14-results.json), and [port lineage](PORT_LINEAGE.md#b12x-sparse-mla-and-indexer). The exact vLLM integration and [3-file B12X overlay](../reproducibility/overlays/b12x/b12x) are public with base/target hashes. | **Yes**, through the selected B12X path; DCP was only 1. The retained evidence does not prove the overflow boundary fired. | Broadly superseded by main's [FlashInfer sparse-MLA backend](https://github.com/vllm-project/vllm/blob/fe4c5dcd4cc71c2c1dc05375be497e413425aad7/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py) and [sparse indexer](https://github.com/vllm-project/vllm/blob/fe4c5dcd4cc71c2c1dc05375be497e413425aad7/vllm/model_executor/layers/sparse_attn_indexer.py). [`vLLM#47629`](https://github.com/vllm-project/vllm/pull/47629) already carried the relevant int64 pointer-width repair and long-offset test. | **No** duplicate PR and no initial comment. Add only a wrapper-level regression if a current-main gap is independently reproduced. | Exact B12X source receipt, public backend source/tests, DCP>1, branch-specific validation, long-offset boundary proof, and isolated correctness/performance evidence. |
| **Compact NVFP4 MLA KV (`nvfp4_ds_mla`)** | Third-party-derived chain: tonyd2wild, danielwoz, drowzeys, B12X/local-inference-lab, jasl and related vLLM work; O14 deployment integration and 0.27 port by 0xdfi/Nous. | Public guard/config/evidence: [`mla_attention.py`](../overlays/mla_attention.py), [`serve-o14.sh`](../recipe/serve-o14.sh), [`o14-results.json`](../evidence/o14-results.json), and [port lineage](PORT_LINEAGE.md#nvfp4-compact-kv). The runtime integration, exact [B12X reader sources](../reproducibility/overlays/b12x/b12x/attention/mla), and [14-path native writer/build patch](../reproducibility/patches/native/exp1-r4-native.patch) are public; no public build or GPU execution is claimed. | **Yes.** The custom format, 368-byte accounting, dispatch, and guards were active. | Main had generic `nvfp4`/`nvfp4_4over6` but rejected generic NVFP4 with MLA in the audited path. Open [`vLLM#51724`](https://github.com/vllm-project/vllm/pull/51724) proposed an overlapping 352-byte format, with layout refactoring in [`vLLM#51704`](https://github.com/vllm-project/vllm/pull/51704). | **Hold.** Reconcile formats and dependency contracts; do not submit a competing private ABI. [`vLLM#46654`](https://github.com/vllm-project/vllm/issues/46654) remains catalogued, but there is **no umbrella comment now**; reconsider only if a later consolidated public recipe supplies material new evidence. | Public writer/reader implementation and vectors; 368-versus-352 reconciliation; allocator/page accounting tests; long-context corruption tests; public base patch; exact donor source/license receipts, including the unresolved danielwoz repository license. |
| **Quantized MTP packed-module loading** | CosmicRaisins donor patch; 0xdfi/Nous re-anchored only the retained mapping. | The exact [runtime source](../reproducibility/overlays/vllm/vllm/model_executor/models/deepseek_mtp.py) and native-line [regression patch/tests](../reproducibility/patches/native/exp1-r4-native.patch) are public. Checkpoint mapping bytes and a distributable quantized-draft fixture are not published. | **Partially proven active:** the quantized draft loaded and served, but no first-fire proof isolates the mapping branch. | Already present or superseded in frozen main's [`DeepSeekMTP.load_weights`](https://github.com/vllm-project/vllm/blob/fe4c5dcd4cc71c2c1dc05375be497e413425aad7/vllm/model_executor/models/deepseek_mtp.py). More reproducible work already existed in [`#49552`](https://github.com/vllm-project/vllm/issues/49552), [`#49553`](https://github.com/vllm-project/vllm/pull/49553), and [`#49900`](https://github.com/vllm-project/vllm/pull/49900). | **No** O14 port PR and no initial comment. Reopen only for a distinct current-main loader failure. | Tiny public quantized-draft fixture, branch-level failure proof, constructor-order and non-target tests, and proof of which older repair machinery remains necessary. |
| **Block rejection sampling** | Native vLLM configuration, not custom O14 code. | The full pinned vLLM base plus exact runtime overlay is materializable, and the [port lineage](PORT_LINEAGE.md#adaptive-mtp-and-block-rejection) records selection. No block-vs-standard result is published. | **Yes**, configured as `block`; no isolated comparison. | Native in the frozen speculative config; MRV2 support merged in [`vLLM#46781`](https://github.com/vllm-project/vllm/pull/46781). | **No** contribution or comment. | No O14 evidence distinguishes block from standard rejection; none is needed to claim authorship because O14 should make no such claim. |
| **Draft-temperature scale** | Original O14 modification to a vLLM-derived speculator. | Public source/config: [`speculator.py`](../overlays/speculator.py) (`_R17_DRAFT_TEMP_SCALE`, `DraftModelSpeculator._copy_request_inputs`), [`apply_o14_overlays.py`](../patches/apply_o14_overlays.py), and [`serve-o14.sh`](../recipe/serve-o14.sh); no-op evidence in [benchmarks](BENCHMARKS.md#negative-results-retained). | Source installed, but **functionally inactive**: the measured value was `1.0`, an exact no-op. | The exact independent knob was absent; the frozen speculator copied request temperature directly. | **Hold.** No PR, issue, or comment from O14 evidence. | Non-1.0 execution; proof that the same shaped proposal `q` is used in acceptance/residual math; distribution-preservation tests; validation; matched quality/acceptance/throughput A/B. |
| **FlashInfer autotuner shim** | Original 0xdfi/Nous compatibility shim around a FlashInfer API. | Public source: [`r17_shim.py`](../runtime/r17_shim.py) (conditional `set_autotune_process_group` injection) and [custom-runtime documentation](CUSTOM_RUNTIME.md#flashinfer-compatibility-shim). | **Yes**, on the older bundled dependency; no speed claim. | Obsolete for frozen main's newer FlashInfer requirement and current integration. This is not a vLLM runtime feature. | **No** vLLM PR. Fix dependency alignment or contribute at the FlashInfer layer if a supported-version defect remains. | Exact affected-version matrix and a reproducer in the owning dependency. |
| **DFlash, DCP/CP, and structured-output compatibility** | Primarily vLLM-derived 0.27 API re-anchoring by 0xdfi/Nous; several initial payloads were absent from the final port tree. | Public docs/config only: [port lineage](PORT_LINEAGE.md#rebuild-base), [custom-runtime inventory](CUSTOM_RUNTIME.md#functional-surfaces), and DCP1 in [`serve-o14.sh`](../recipe/serve-o14.sh). The exact DFlash/autoregressive, DCP/parallel-state, and related 0.27 compatibility sources are public in the 74-file vLLM overlay; DCP1 is not DCP>1 or DP evidence. | Structured-output compatibility was installed. DCP-specific behavior was inert at DCP1. DFlash/autoregressive paths were not the active proposer. | DFlash and structured output were superseded by current implementations and tests. DCP sparse-MLA support remained partial; related work included [`#47779`](https://github.com/vllm-project/vllm/pull/47779), [`#50382`](https://github.com/vllm-project/vllm/pull/50382), and [`#46514`](https://github.com/vllm-project/vllm/pull/46514). | **No** DFlash/structured-output port. **Hold** DCP unless a DCP>1 current-main defect is reproduced. No initial comments. | Public final patch boundaries, DCP>1 execution, branch-specific failing tests, and proof for the five initial-but-not-final payload files. |
| **Launch and benchmark packaging** | Original 0xdfi/Nous reconstruction recipe, probes, evidence packaging, and documentation. | Public artifacts: [`serve-o14.sh`](../recipe/serve-o14.sh), [`o14.env.example`](../recipe/o14.env.example), the [source-complete reconstruction](../reproducibility/README.md), [`benchmarks/README.md`](../benchmarks/README.md), [`o14-results.json`](../evidence/o14-results.json), [`final-o14-battery.jsonl`](../evidence/final-o14-battery.jsonl), and the [benchmark protocol](BENCHMARKS.md). | The source record is exact, but no public native/image build was performed and exact benchmark replay still depends on private checkpoint/sidecar and topology. | Not applicable to vLLM core. | **No** runtime PR. Retain out of tree as reproducibility documentation. | Public base/checkpoint/sidecar, captured resolved launch manifest, topology, and reconciliation of the documented-versus-observed `CUDA_DEVICE_MAX_CONNECTIONS` value. |
| **Negative and rejected experiments** | Original campaign experiments unless an external backend is named. | Public summaries/evidence: [negative benchmark record](BENCHMARKS.md#negative-results-retained), [custom-runtime record](CUSTOM_RUNTIME.md#custom-kernel-work-rejected-during-the-campaign), and [port-lineage record](PORT_LINEAGE.md#work-tested-and-rejected). Most local experiment implementations/raw artifacts are **not published**. | **No:** dense W8A16 Build B; CUTE-DSL BMM; causal throughput-argmax MTP controller; `eh_proj` INT8; W8 head v1 without exact rescore; Humming/TensorRT-LLM MoE attempts. | No missing-main claim follows from a negative O14 result. | **No.** Preserve as negative evidence; do not upstream or market as shipped features. | Public raw/reproducible artifacts and per-file license headers are incomplete for some local-only experiments, but no contribution should be opened merely to fill that archive gap. |

## Ranked contribution sequence

This sequence ranks useful human work, not ownership claims or predicted
acceptance.

1. **Scheduler race — New `[Bugfix][Core]` PR after reproduction.** Produce a
   minimal current-main failing test and stress test first. If the race no
   longer exists, close the item without a PR.
2. **Build A — New two-step kernel contribution.** First add a generic guarded
   tiny MLA BMM custom op and benchmark; then route only the two eligible
   current-main call sites. The rewrite may adopt v1's better tuning table and
   v0's live integration evidence. It must not copy either file unchanged or
   claim that v1 powered the O14 battery. `#36297` is the only comment-now
   thread, limited to adjacent tiny-BMM evidence and explicit validation bounds.
3. **Marlin atomic add — Evidence first; no contact now.** Do not comment on
   `#48569` or open a competing PR without new causal evidence beyond O14's
   reachability/stability-only result.
4. **W8 selected-row rescore — Watch; no immediate comment.** `#48870` uses a
   different fused tied-head mechanism. Reconsider an RFC or thread comment
   only after a public sidecar fixture and candidate-recall proof exist, or if a
   maintainer asks.
5. **Adaptive MTP — Watch; no immediate comment.** Reconsider `#47111` only if
   it reactivates and O14 has material new controller evidence, including a
   fixed-K A/B. Preserve all named donor attribution.
6. **Decode-aware prefill — RFC only after matched evidence.** Freeze other
   changes and report decode latency, TTFT, prefill throughput, starvation, and
   fairness under the same workload.
7. **Compact NVFP4 MLA KV — Hold.** Reconcile `#51724`/`#51704`, format size,
   dependency contracts, test vectors, and licensing before any code proposal.

All remaining catalogue rows are either already represented upstream,
superseded, dependency-local, deployment-only, inactive, or insufficiently
proven.

## No-spam engagement plan

No remote action is performed by this repository update. Any later human
engagement is gated by thread activity and public O14 evidence:

1. **COMMENT NOW — [`vLLM#36297`](https://github.com/vllm-project/vllm/pull/36297):**
   share only the adjacent GB10 tiny-BMM graph-replay data. Explicitly
   distinguish live v0 from microbenchmarked v1 and state that O14 did not
   validate that PR's fused FP8 branch.
2. **WATCH / NO COMMENT NOW — [`vLLM#47111`](https://github.com/vllm-project/vllm/pull/47111):**
   the PR needs a rebase and is inactive. O14 now publishes controller source but
   still has no fixed-K A/B. Reconsider only if the PR reactivates and O14 has
   material public source or evidence.
3. **WATCH / NO COMMENT NOW — [`vLLM#48870`](https://github.com/vllm-project/vllm/pull/48870):**
   it is a different fused tied-head mechanism, and O14 has neither a public
   sidecar fixture nor candidate-recall proof. Reconsider after both exist or if
   a maintainer asks.
4. **NO CONTACT WITHOUT NEW EVIDENCE — [`vLLM#48569`](https://github.com/vllm-project/vllm/pull/48569):**
   that PR already has stronger GB10 atomic-off/on and paired-layer evidence.
   O14's reachability/stability-only result adds no causal evidence.
5. **NO UMBRELLA COMMENT NOW — [`vLLM#46654`](https://github.com/vllm-project/vllm/issues/46654):**
   keep it catalogued, but do not post a consolidated O14 recipe comment without
   material new evidence.

Do not cross-post the same story to adjacent B12X, NVFP4, DCP, quantized-MTP,
scheduler, build, or GLM threads.

## Out-of-tree and superseded boundaries

### Remain out of tree

- The monolithic 0.27 port as a single upstream change; its exact historical
  source is public but remains unsuitable for a wholesale current-main copy.
- The 368-byte `nvfp4_ds_mla` ABI remains out of upstream until its now-public
  implementation/tests are reconciled with current formats and dependencies.
- O14's checkpoint-specific W8 sidecar contract until a generic public design
  and artifact exist.
- Both Build A files **unchanged**; only a rewritten guarded custom op is a
  candidate.
- O14's environment-driven adaptive controller, decode-budget port,
  draft-temperature no-op, FlashInfer shim, and deployment recipe.
- All rejected/negative experiment implementations.

### Superseded or already represented upstream at the frozen SHA

- Native block rejection configuration.
- Quantized MTP packed-module loading covered by current model loading work.
- DFlash signature/API and structured-output compatibility ports.
- Fixed full-CUDA-graph verification shape plumbing as a reusable source
  feature; O14's exact shape list remains deployment tuning.
- Most historical B12X backend identity/indexer plumbing and the core int64
  repair; only independently reproduced current-main gaps merit new work.
- The old FlashInfer compatibility shim for the frozen main dependency set.

## Attribution and licensing

This repository is Apache-2.0, and modified vLLM files retain their vLLM SPDX
and history. That repository license does **not** relicense checkpoint weights,
dependencies, unpublished source, or any other third-party artifact. Before
extracting any contribution, preserve the applicable upstream notices and
confirm the exact donor source and license.

The audited lineages are:

- **vLLM contributors:** all vLLM-derived files, native probabilistic draft
  sampling, and native block rejection.
- **B12X/local-inference-lab and Luke Alonso:** sparse MLA/indexer work and the
  int64/direct-K repair; 0xdfi/Nous contributed the documented 0.27 integration.
- **Aiden Le, Luke Alonso, Martin Vit, and CosmicRaisins:** adaptive MTP
  controller lineage; CosmicRaisins also supplied policy/telemetry work;
  0xdfi/Nous reconciled parsing/APIs and wired the 0.27 runtime.
- **penguinchang, ciprianveg, and OsakaTX:** decode-aware prefill lineage;
  0xdfi/Nous carried the 0.27 re-port.
- **tonyd2wild, danielwoz, drowzeys, B12X/local-inference-lab, jasl, and
  applicable vLLM contributors:** compact NVFP4 MLA KV lineage; 0xdfi/Nous
  supplied earlier deployment integration and the documented 0.27 port.
- **CosmicRaisins:** quantized-draft packed-module mapping donor; 0xdfi/Nous
  re-anchored the retained mapping.
- **Claude Fable 5:** attribution is limited to co-authorship of the final
  live-battery commit and evidence recovery from that session
  transcript/campaign, as documented in [provenance](PROVENANCE.md). This does
  not attribute assistance with CosmicRaisins' donor patch or donor lineage.
- **0xdfi/Don and Nous Research contributors:** O14-original scheduler repair,
  Build A work, selected-row rescore logic, Marlin policy switch,
  draft-temperature experiment, compatibility shim, packaging, benchmark
  harness, and documented integration/reconciliation work—without absorbing
  any third-party ownership listed above.

The public B12X 0.30.2 wheel is Apache-2.0 and now has an exact SHA-bound
3-file overlay. Remaining provenance caution applies to external donor
lineages and local rejected experiments not included in this source pack; it
is not an invitation to label third-party work Nous-original.

## Evidence boundaries

- **Bundled profile:** the final battery demonstrates the assembled O14 stack;
  it does not causally attribute throughput to any one feature.
- **Sample size/noise:** most end-to-end cells are `n=1–2`, inside the campaign's
  approximately ±5% noise band. See the [benchmark record](BENCHMARKS.md).
- **Non-source dependencies:** no compiled wheel or measured image is
  published. The checkpoint and W8 sidecar weights remain private, and exact
  four-node benchmark replication is not proven by source reconstruction.
- **Selected-row exactness:** W8 rescoring recomputes selected rows exactly; it
  does not prove that the full-vocabulary BF16 winner was in the quantized
  top-64 candidate set.
- **Atomic add:** O14 has reachability and stability evidence, but no causal
  atomic-off/atomic-on A/B or complete numerical/graph-safety matrix.
- **Build A identity:** v0 was active in the measured O14 path. V1 has better
  isolated timings but was not live-wired. Neither checked-in implementation is
  merge-ready unchanged.
- **Source boundary:** facts are measured, source-inspected, or inferred as
  labeled in [provenance](PROVENANCE.md). Private implementation gaps are not
  filled by inference.
