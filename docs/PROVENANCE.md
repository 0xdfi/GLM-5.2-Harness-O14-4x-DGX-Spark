# Provenance and publication boundary

## Source authority

- Campaign repository HEAD at packaging: `698b9085c3aed47de5513a720204ed788b607f6a`.
- Final live-battery commit: `c3e3f4cdfb7a8d4f56a4f66380900ff48c30a35d`, co-authored `Claude Fable 5 <noreply@anthropic.com>`.
- Exact final-battery values were recovered from raw tool-result events in the final Fable session transcript and sanitized into `evidence/final-o14-battery.jsonl`.
- This is the complete public Fable attribution boundary: the repository does not evidence Fable assistance with CosmicRaisins' donor patch or donor lineage.
- Live runtime identity, environment, source imports, four-rank source hashes, and exact-rescore first-fire records were inspected read-only immediately before publication.
- The source reconstruction is pinned to public vLLM commits `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` and `e232d262369b8c918cf478a7a96a0fcf8127cf65`, plus the public `b12x==0.30.2` wheel SHA-256 `081f6391f9a375dea8810bebf6a9d0276cacecde229b28715924f411e04bb38b`.
- The exact public source manifests contain 74 vLLM runtime files, 3 B12X files, and 14 native-build paths. The historical 26/27 counts are retained only as incomparable campaign scopes.

## Evidence classes

- **Measured:** final prose/peak/prefill values, Build A v1 isolated microbenchmarks, correctness outcomes, and trace budget.
- **Source-inspected:** vLLM package/image identity, runtime arguments/environment, active Build A v0 import, custom-port map, overlays, and four-rank exact-rescore fire proof.
- **Source-reconstructed:** all known runtime source targets materialized offline from exact public bases with fail-closed base/target hashes and deterministic receipts.
- **Estimated:** Build A v1's 0.75–1.0 ms/step projected impact and broader remaining-headroom estimates. These are labeled estimates and not promoted to measured gains.

## Reconciliation rule and corrected handoff claim

The Fable campaign prose called O14 “Build A v1.” The deployed source on every rank imports `builda_bmm_v0`, the live environment has only `VLLM_BUILDA_BMM=1`, and cache evidence identifies v0 compilation. The live source controls over the prose handoff. This repository identifies v0 as active and v1 as separately tested, present, and not wired.

Pre-publication live source SHA-256 values (the public `mla_attention.py` differs only by a redacted private-path comment):

| file | SHA-256 |
|---|---|
| `kernels/builda_bmm_v0.py` | `965b6aeaf3a0c41abefec89a31592f6b8061a053053831be7c5e6f7560857515` |
| `kernels/builda_bmm_v1.py` | `ee3145bee53a67dbff35f0556dfdcc21ac89a7038f3bd22cd0687928e38053e4` |
| `overlays/mla_attention.py` | `ef38f75c7aa57958ab3ced69f8a1fd613bf1c29d2210bc177db15c45660eb044` |
| `overlays/logits_processor.py` | `5ab8890051bf82da012a2c7357bda60fb011469ebf42ff647fffba8416fe4dfa` |

These four hashes matched all four observed ranks.

## Omitted from the public repository

- model weights and derived checkpoint shards;
- compiled vLLM/B12X wheels and the measured private container image;
- private hostnames, fabric addresses, filesystem paths, service tunnels, and orchestration receipts;
- credentials and provider configuration;
- production lifecycle and rollback controller;
- the failed `eh_proj` derived checkpoint;
- raw profiler traces and logs containing private topology/path data.

## Limits

The published benchmark records preserve exact values but not private request bodies, timestamps, host paths, or topology. The sanitized probe programs document the protocol and default to an operator-supplied `O14_BASE_URL`. Re-running them is a new experiment and may differ with checkpoint, hardware, sampling, load, or runtime state.

Source-complete means the exact known source deltas and native build plumbing are public and reconstructable. It does not mean the public scaffold was compiled or that a runtime/image was validated. Exact benchmark replication still requires the same checkpoint and W8 sidecar, four GB10 nodes with equivalent fabric/topology, and the resolved serving environment.
