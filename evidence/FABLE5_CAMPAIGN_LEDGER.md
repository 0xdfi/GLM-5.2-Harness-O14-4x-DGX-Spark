# Fable 5 campaign ledger extract

- Source commit: `698b9085c3aed47de5513a720204ed788b607f6a`
- Final-battery commit: `c3e3f4cdfb7a8d4f56a4f66380900ff48c30a35d`
- Commit co-author: `Claude Fable 5 <noreply@anthropic.com>`

The final session recorded:

```text
prose C1 25.40 / 25.61 tok/s
prose C4 aggregate 54.68 / 53.62 tok/s
peak C1 36.63 tok/s, accepted/step 3.265
peak C4 80.56 tok/s, accepted/step 3.934
cold prefill 661.1 tok/s at 187,022 prompt tokens — PASS
```

The campaign handoff described the live configuration as “Build A v1.” Post-handoff source inspection corrected that label: the same `mla_attention.py` on all four ranks imports `builda_bmm_v0`; `VLLM_BUILDA_BMM=1` was active and no v1 selector was present. Build A v1 remained in the image and its separate microbenchmarks remain valid, but it was not the live final-battery call path.

The exact-rescore head activation was independently recovered from worker logs on all four TP ranks, each with a nonzero `FIRST RESCORE FIRED` changed-entry count.

This is a sanitized extract. Private host paths, topology addresses, checkpoint names, and incident details are deliberately excluded. Exact recovered tool-output records are in `final-o14-battery.jsonl`.
