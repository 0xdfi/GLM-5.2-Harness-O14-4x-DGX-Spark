# Agent instructions

This repository is public source material for named O14 profiles. Treat [`profiles/o14-profiles.json`](profiles/o14-profiles.json) as the selector of record.

## Fail-closed selection

1. Read the requested profile from `profiles/o14-profiles.json`.
2. Continue to build or serve only when `status` is exactly `READY`, `deployable` is `true`, and every profile-specific artifact gate below is satisfied.
3. For unknown, `CANDIDATE`, `TESTING`, rejected, or superseded profiles, return a plan-only response. Do not build, pull, launch, or synthesize missing values.
4. Pin the exact Git commit. Verify `SHA256SUMS`, run `python3 scripts/validate.py`, `python3 reproducibility/verify.py`, and the unit tests before building.
5. Model weights are separate and operator-supplied. Never infer a checkpoint path, host address, fabric interface, image tag, or registry namespace.

## O14 Fast — 250K total KV, READY

`o14-fast` is the READY TP4/DCP1/PP1 profile with 250K total KV. Its exact allocator recipe is:

- `KV_CACHE_MEMORY_BYTES=7995534848` bytes per rank
- `MAX_MODEL_LEN=249000`
- `MAX_NUM_SEQS=4`
- `MAX_NUM_BATCHED_TOKENS=2048`
- allocator total logical KV: `250023` tokens

There is no published O14 Fast OCI image. Agents may build and deploy Fast only from the pinned canonical source commit with `docker/Dockerfile.repro`; do not invent or pull a `fast`, `latest-fast`, or similar tag. The render-first wrapper requires `O14_PROFILE_NAME=o14-fast`, `O14_PROFILE_STATUS=READY`, `O14_EXECUTE=1`, and the `serve` argument to execute.

## O14 Balanced

`o14-balanced` is `TESTING / DO NOT DEPLOY`. It is a machine-readable placeholder for TP4/DCP2/PP1, a 500K target, a 500237-token allocator expectation, `8000000000` KV bytes per rank, and `MAX_MODEL_LEN=490000`.

Do not build, pull, or launch Balanced. Do not derive a Balanced tag from Fast or reuse Fast's image. Balanced requires a separate immutable v3 runtime image because its DCP2 path has source changes. Refuse Balanced until a later accepted selector has `status: READY`, `deployable: true`, a complete source/build manifest, `image.public_image_available: true`, and a verified immutable `sha256:` image digest. The current selector satisfies none of those release conditions.

## Public topology boundary

The documented fleet class uses a 100GbE switch; 100000 Mb/s per active RDMA link is expected. Operators must supply their own addresses, interface/HCA names, device mappings, and paths. Never replace these public parameters with private defaults.
