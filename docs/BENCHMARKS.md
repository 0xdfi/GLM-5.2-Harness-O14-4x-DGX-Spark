# O14 benchmark record

> **Historical record:** these receipts belong to the historical 399K O14 campaign. O14 Balanced remains TESTING and has no measured speed row.

## Final historical O14 battery

The values below are exact tool outputs recovered from the final Claude Fable 5 session transcript. The protocol used cache-busted prompts and server counters to avoid prefix-cache inflation. The sanitized probe sources are published in `benchmarks/`; the recovered records are in `evidence/final-o14-battery.jsonl`.

| class | concurrency | run | tok/s | steps/s | accepted/step | contamination |
|---|---:|---|---:|---:|---:|---|
| cold unpredictable prose | C1 | A | 25.40 | 10.78 | 1.353 | false |
| cold unpredictable prose | C1 | B | 25.61 | 11.14 | 1.296 | false |
| cold unpredictable prose | C4 aggregate | A | 54.68 | 23.34 | 1.344 | false |
| cold unpredictable prose | C4 aggregate | B | 53.62 | 22.23 | 1.412 | false |
| predictable/code-class peak | C1 | final | 36.63 | n/a | 3.265 | n/a |
| predictable/code-class peak | C4 aggregate | final | 80.56 | n/a | 3.934 | n/a |

Peak C1 generated 900 tokens; peak C4 generated 3,515 total tokens. The deep cold-prefill gate passed with 187,022 prompt tokens in 282.9 seconds, or 661.1 tok/s.

## Campaign comparison

| metric | R15 starting point | o10 | o12-A | o12-B | O14 live |
|---|---:|---:|---:|---:|---:|
| prose C1 | 23.6 / 24.1 | 26.1 / 25.5 | 25.7 / 26.2 | 24.3 / 24.4 | **25.40 / 25.61** |
| prose C4 aggregate | 52.1 | 54.4 | 54.6 | 52.5 / 53.8 | **54.68 / 53.62** |
| peak C1 | n/m | n/m | 36.3 | 35.8 | **36.63** |
| peak C4 aggregate | n/m | n/m | 77.4 | 83.7 | **80.56** |
| cold prefill | 644.7 | 663.7 | n/m | 658.5 | **661.1** |

The source chart characterizes O14 versus R15 as roughly +6–8% prose C1, +4–5% prose C4, and +2.5% prefill.

## Build A v1 kernel microbenchmarks

Build A v1 was swept separately under CUDA-graph replay across B=3/4/5/6, including contiguous and production-like strided operands.

| batch | BMM1 cuBLAS | BMM1 v1 | BMM2 cuBLAS | BMM2 v1 |
|---:|---:|---:|---:|---:|
| 3 | 7.30 µs | 3.10 µs | 11.36 µs | 4.80 µs |
| 4 | 9.25 µs | 3.44 µs | 9.53 µs | 4.21 µs |
| 5 | 8.88 µs | 3.08 µs | 8.95 µs | 4.08 µs |
| 6 | 7.67 µs | 3.93 µs | 9.48 µs | 4.09 µs |

The quiet-window sweep observed 3.02–3.07 µs for BMM1 and 3.89–3.99 µs for BMM2. Correctness passed for B=3..6, both shapes, contiguous and strided patterns, with maximum relative error ≤3.2e-3 against fp32 `torch.bmm` reference. The campaign estimated 0.75–1.0 ms/step potential versus the cuBLAS path.

**Attribution correction:** these v1 measurements are valid isolated tests, but final live source inspection proved the measured O14 battery called Build A v0. V1 was present in the image and not wired. No end-to-end O14 gain is attributed to v1 here.

## Trace-informed work

A 171-step trace on the vLLM 0.27 runtime attributed approximately:

| kernel family | ms/step |
|---|---:|
| Marlin MoE W4A16 | 32.1 |
| dense Marlin | 23.9 |
| cuBLAS/CUTLASS bf16 BMM fallback | 11.1 |
| NCCL ring LL all-reduce | 10.6 |
| bf16 GEMV family | 4.6 |

That trace motivated Build A and corrected an early hypothesis that communication alone dominated the fixed cost.

## Negative results retained

- The CUTE-DSL SIMT BMM was correct and capture-safe but much slower at 22.9/56.0 µs.
- The dense W8A16 Triton candidate passed 71/71 correctness checks, won 0/7 relevant shapes, and projected a 2.9 ms/step regression.
- A causal throughput-argmax MTP controller replay over 1,275 usable windows scored 12.65% below the incumbent and was killed.
- The `eh_proj` INT8 arm was dropped because its packed layout was rejected by the loader; the estimated prize was about 0.3%.
- The draft-temperature knob shipped at 1.0, so no O14 gain is attributed to it.
- Peak C4 reached 83.7 tok/s on an earlier O12-B battery; O14's 80.56 is in the same broad noise band, not categorically faster.

## Statistical boundary

The campaign assigned a ±5% noise band to n=1–2 batteries. O14 was retained operationally because it combined the repaired 0.27 runtime, exactness gates, current kernel/env set, deep-prefill pass, and live stability. Separating sub-5% component effects requires at least three paired repeated batteries and was not completed before publication.
