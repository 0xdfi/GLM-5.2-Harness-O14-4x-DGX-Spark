# Benchmark protocols

These sanitized probes reproduce the protocol shape used for the final O14 battery without publishing private host or topology details.

```bash
export O14_BASE_URL=http://127.0.0.1:8211
python3 benchmarks/prose_probe.py 1 1024 prose-c1
python3 benchmarks/prose_probe.py 4 1024 prose-c4
python3 benchmarks/peak_probe.py 1 peak-c1
python3 benchmarks/peak_probe.py 4 peak-c4
python3 benchmarks/deep_prefill_probe.py prefill
```

Safety and interpretation:

- `prose_probe.py` refuses to start unless the server reports idle and marks foreign counter contamination.
- Prompts are nonce-busted to prevent prefix-cache inflation.
- C4 is aggregate decode throughput across four concurrent requests.
- Peak probes deliberately use predictable code-class content and temperature 0 to characterize deep-MTP behavior; they are not representative prose.
- Deep prefill sends a roughly 187K-token cache-busted request and may take minutes.
- Running these scripts creates live inference load. Do not execute them against a shared or production endpoint without authorization.
- The publication validation compiles these scripts but does not execute benchmark traffic.

Exact values recovered from the final Fable 5 session are preserved in `evidence/final-o14-battery.jsonl`.
