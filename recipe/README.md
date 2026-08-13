# Recipe boundary

`serve-o14.sh` reproduces the publishable API-server argument and environment envelope observed in O14. It assumes:

1. four Ray-visible GPU workers, one GB10 per node;
2. a vLLM 0.27-compatible base containing the documented custom port;
3. the published O14 overlays and active Build A v0 integration baked into that image;
4. the same checkpoint family mounted at `MODEL_PATH`;
5. RoCE/NCCL wiring chosen for the operator's own interfaces and GID indices.

Build A v1 is included as tested development evidence but is intentionally not selected by this recipe because direct live inspection proved O14 imported v0. The exact-head sidecar path is configurable and `VLLM_LMHEAD_V2_REQUIRE=1` preserves the observed fail-loud behavior.

The script does not create/destroy containers, form the Ray cluster, modify routes, unload a model, or infer network interfaces. Those omissions are intentional: the original launcher contains private topology and lifecycle controls that do not belong in a public recipe.

Run `bash serve-o14.sh render` first. Execution requires both the `serve` argument and `O14_EXECUTE=1`.
