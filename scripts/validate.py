#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
REQUIRED = {
    ".gitignore",
    "AGENTS.md",
    "README.md",
    "LICENSE",
    "NOTICE",
    "recipe/o14.env.example",
    "recipe/serve-o14.sh",
    "recipe/README.md",
    "profiles/o14-profiles.json",
    "docker/Dockerfile.repro",
    "docker/Dockerfile.o14-overlay",
    "kernels/builda_bmm_v0.py",
    "kernels/builda_bmm_v1.py",
    "overlays/marlin_moe.py",
    "overlays/speculator.py",
    "overlays/mla_attention.py",
    "overlays/logits_processor.py",
    "runtime/r17_shim.py",
    "patches/apply_o14_overlays.py",
    "benchmarks/README.md",
    "benchmarks/prose_probe.py",
    "benchmarks/peak_probe.py",
    "benchmarks/deep_prefill_probe.py",
    "docs/BENCHMARKS.md",
    "docs/CUSTOM_RUNTIME.md",
    "docs/OPERATIONAL_ENVELOPE.md",
    "docs/PORT_LINEAGE.md",
    "docs/PROVENANCE.md",
    "docs/VLLM_UPSTREAMING.md",
    "evidence/o14-results.json",
    "evidence/final-o14-battery.jsonl",
    "evidence/FABLE5_CAMPAIGN_LEDGER.md",
    "tests/test_overlay_applicator.py",
    "tests/test_o14_evidence.py",
}
FORBIDDEN = (
    "/" + "Users/",
    "/home/" + "dfi",
    "192." + "168.",
    "169." + "254.",
    "spark-" + "nord",
    "spark-" + "sud",
    "spark-" + "ost",
    "spark-" + "west",
    "127.0.0.1:" + "18211",
)
EXACT_HASHES = {
    "kernels/builda_bmm_v0.py": "965b6aeaf3a0c41abefec89a31592f6b8061a053053831be7c5e6f7560857515",
    "kernels/builda_bmm_v1.py": "ee3145bee53a67dbff35f0556dfdcc21ac89a7038f3bd22cd0687928e38053e4",
    "overlays/logits_processor.py": "5ab8890051bf82da012a2c7357bda60fb011469ebf42ff647fffba8416fe4dfa",
}

missing = sorted(path for path in REQUIRED if not (ROOT / path).is_file())
if missing:
    raise SystemExit(f"missing required files: {missing}")

for path in ROOT.rglob("*"):
    if (
        not path.is_file()
        or ".git" in path.parts
        or "__pycache__" in path.parts
        or path.suffix == ".pyc"
        or path.name == "SHA256SUMS"
    ):
        continue
    text = path.read_text(encoding="utf-8")
    hits = [needle for needle in FORBIDDEN if needle in text]
    if hits:
        raise SystemExit(f"private-data pattern in {path.relative_to(ROOT)}: {hits}")

for relative, expected in EXACT_HASHES.items():
    actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"source hash mismatch for {relative}: {actual}")

data = json.loads((ROOT / "evidence/o14-results.json").read_text())
assert data["schema"] == "glm52-o14-public-evidence/2"
assert data["runtime"]["vllm_package"] == "0.27.2.dev0+g6e448d0ea.d20260812"
assert data["runtime"]["parallelism"] == {"tp": 4, "dcp": 1, "pp": 1}
assert data["runtime"]["max_model_len"] == 399000
assert data["runtime"]["builda_active_version"] == "v0"
assert data["runtime"]["builda_v1_wired_into_live_wrapper"] is False
assert data["runtime"]["lmhead_exact_rescore"]["first_rescore_fired_all_four_ranks"] is True
assert len(data["measurements"]) == 5

records = [json.loads(line) for line in (ROOT / "evidence/final-o14-battery.jsonl").read_text().splitlines()]
assert len(records) == 7
assert [records[index]["tok_s"] for index in (0, 1)] == [25.4, 25.61]
assert [records[index]["aggregate_tok_s"] for index in (2, 3)] == [54.68, 53.62]
assert records[4]["tok_s"] == 36.63 and records[5]["aggregate_tok_s"] == 80.56
assert records[6]["prompt_tokens"] == 187022 and records[6]["prefill_tok_s"] == 661.1

profile_index = json.loads((ROOT / "profiles/o14-profiles.json").read_text())
assert profile_index["schema"] == "glm52-o14-profile-selector/1"
assert profile_index["selector_policy"]["deployable_statuses"] == ["READY"]
assert profile_index["selector_policy"]["unknown_profile_action"] == "PLAN_ONLY"
assert profile_index["canonical_source"]["commit_pin_source"] == "consumer"
assert profile_index["canonical_source"]["canonical_commit"] is None
assert profile_index["fabric"] == {
    "switch_class_gbps": 100,
    "expected_active_rdma_link_mbps": 100000,
    "operator_supplied_topology": True,
}
fast = profile_index["profiles"]["o14-fast"]
balanced = profile_index["profiles"]["o14-balanced"]
assert fast["status"] == "READY" and fast["deployable"] is True
assert fast["status_label"] == "O14 Fast — 250K total KV, READY"
assert fast["topology"] == {
    "tensor_parallel_size": 4,
    "decode_context_parallel_size": 1,
    "pipeline_parallel_size": 1,
}
fast_capacity = fast["capacity"]
assert fast_capacity["public_total_kv_label"] == "250K total KV"
assert fast_capacity["allocator_total_logical_kv_tokens"] == 250023
assert fast_capacity["kv_cache_memory_bytes_per_rank"] == 7995534848
assert fast_capacity["max_model_len"] == 249000
assert fast_capacity["max_num_seqs"] == 4
assert fast_capacity["max_num_batched_tokens"] == 2048
g = fast_capacity["geometry"]
assert g["main_cache_entries"] * g["main_bytes_per_token"] + g["index_cache_entries"] * g["index_bytes_per_token"] == 31976
assert g["bytes_per_local_token_per_rank"] == 31976
assert g["all_layer_block_stride_bytes_per_rank"] == 31976 * g["block_tokens"] == 2046464
assert g["selected_blocks"] * g["all_layer_block_stride_bytes_per_rank"] == 7995534848
assert g["blocks_per_max_request"] == (249000 + 63) // 64 == 3891
assert g["selected_blocks"] * 249000 // g["blocks_per_max_request"] == 250023
assert (g["selected_blocks"] - 1) * 249000 // g["blocks_per_max_request"] == g["one_block_lower_total_kv_tokens"] == 249959
assert fast["image"] == {
    "public_image_available": False,
    "reference": None,
    "digest": None,
}
assert fast["build"]["method"] == "SOURCE_BUILD"
for referenced_path in (
    fast["historical_evidence"]["receipt"],
    fast["build"]["dockerfile"],
    fast["build"]["checksum_manifest"],
):
    relative = pathlib.PurePosixPath(referenced_path)
    assert not relative.is_absolute() and ".." not in relative.parts, referenced_path
    assert (ROOT / relative).is_file(), referenced_path
assert balanced["status"] == "TESTING" and balanced["deployable"] is False
assert balanced["status_label"] == "TESTING / DO NOT DEPLOY"
assert balanced["topology"] == {
    "tensor_parallel_size": 4,
    "decode_context_parallel_size": 2,
    "pipeline_parallel_size": 1,
}
assert balanced["capacity"] == {
    "target_total_logical_kv_tokens": 500000,
    "allocator_expected_tokens": 500237,
    "evidence_class": "TESTING_UNTIL_ACCEPTED",
    "kv_cache_memory_bytes_per_rank": 8000000000,
    "max_model_len": 490000,
}
assert balanced["image"]["public_image_available"] is False
assert balanced["image"]["reference"] is None and balanced["image"]["digest"] is None
assert balanced["image"]["required_digest_algorithm_after_acceptance"] == "sha256"
assert balanced["image"]["must_not_overwrite_profile"] == "o14-fast"
assert balanced["build"]["method"] is None
assert balanced["benchmark"]["status"] == "TBD"
assert balanced["benchmark"]["capacity_measured_tokens"] is None
assert balanced["benchmark"]["prefill_tokens_per_second"] is None
assert balanced["benchmark"]["decode_tokens_per_second"] is None

serve = (ROOT / "recipe/serve-o14.sh").read_text()
env = (ROOT / "recipe/o14.env.example").read_text()
agents = (ROOT / "AGENTS.md").read_text()
mla = (ROOT / "overlays/mla_attention.py").read_text()
dockerfile = (ROOT / "docker/Dockerfile.o14-overlay").read_text()
dockerfile_repro = (ROOT / "docker/Dockerfile.repro").read_text()
operational = (ROOT / "docs/OPERATIONAL_ENVELOPE.md").read_text()
speculator_source = (ROOT / "overlays/speculator.py").read_text()
for token in ("O14_EXECUTE", "B12X_MLA_SPARSE", "nvfp4_ds_mla", "6,12,18,24"):
    assert token in serve, token
assert "VLLM_BUILDA_BMM" in serve
assert not any(line.startswith("export VLLM_BUILDA_VER=") for line in serve.splitlines())
assert not any(line.startswith("VLLM_BUILDA_VER=") for line in env.splitlines())
for name, value in {
    "O14_PROFILE_NAME": "o14-fast",
    "O14_PROFILE_STATUS": "READY",
    "MAX_MODEL_LEN": "249000",
    "KV_CACHE_MEMORY_BYTES": "7995534848",
    "MAX_NUM_SEQS": "4",
    "MAX_NUM_BATCHED_TOKENS": "2048",
}.items():
    assert re.search(rf"(?m)^{re.escape(name)}={re.escape(value)}$", env), name
for token in (
    'O14_PROFILE_NAME="${O14_PROFILE_NAME:-o14-fast}"',
    'O14_PROFILE_STATUS="${O14_PROFILE_STATUS:-READY}"',
    '--max-model-len "${MAX_MODEL_LEN:-249000}"',
    '--kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES:-7995534848}"',
    '"${O14_PROFILE_NAME}" != "o14-fast"',
    '"${O14_PROFILE_STATUS}" != "READY"',
    "only O14 Fast — 250K total KV, READY may render or serve",
):
    assert token in serve, token
assert "12792889000" not in serve and "MAX_MODEL_LEN:-399000" not in serve
assert "12792889000" not in env and "399000" not in env
assert "builda_bmm_v0 import" in mla and "builda_bmm_v1 import" not in mla
for token in (
    "overlays/mla_attention.py",
    "overlays/logits_processor.py",
    "kernels/builda_bmm_v0.py",
    "kernels/builda_bmm_v1.py",
    "runtime/r17_shim.py",
):
    assert token in dockerfile, token

runtime_defaults = {
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_USE_FLASHINFER_SAMPLER": "1",
    "VLLM_MARLIN_USE_ATOMIC_ADD": "1",
    "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": "256",
    "SAFETENSORS_FAST_GPU": "1",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_MODULE_LOADING": "LAZY",
    "CUTE_DSL_ARCH": "sm_121a",
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "1800",
    "NCCL_CUMEM_ENABLE": "0",
    "NCCL_MIN_NCHANNELS": "4",
    "NCCL_MAX_NCHANNELS": "4",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
}
for name, value in runtime_defaults.items():
    assert f'export {name}="${{{name}:-{value}}}"' in serve, name
    assert re.search(rf"(?m)^{re.escape(name)}={re.escape(value)}$", env), name

for token in (
    "VLLM_BUILDA_BMM",
    "VLLM_MOE_MARLIN_ATOMIC_ADD",
    "VLLM_USE_B12X_SPARSE_INDEXER",
    "KV_FP8_ROPE",
    "--kv-cache-dtype nvfp4_ds_mla",
    "--attention-backend B12X_MLA_SPARSE",
    '"rejection_sample_method": "block"',
    '--download-dir "${MODEL_PATH}" --load-format auto',
):
    assert token in serve, token

index_topk_pattern = "FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"
assert len(index_topk_pattern) == 78
hf_override = json.dumps(
    {"use_index_cache": True, "index_topk_pattern": index_topk_pattern},
    separators=(",", ":"),
)
assert f"--hf-overrides '{hf_override}'" in serve
assert "78-layer full/sparse attention schedule" in operational

assert 'export CUDA_DEVICE_MAX_CONNECTIONS="${O14_DRIVER_CUDA_DEVICE_MAX_CONNECTIONS:-32}"' in serve
assert re.search(r"(?m)^CUDA_DEVICE_MAX_CONNECTIONS=4$", env)
assert re.search(r"(?m)^O14_DRIVER_CUDA_DEVICE_MAX_CONNECTIONS=32$", env)
assert "`CUDA_DEVICE_MAX_CONNECTIONS=4`" in operational
assert "`O14_DRIVER_CUDA_DEVICE_MAX_CONNECTIONS`" in operational
assert "default is `32`" in operational
assert "inference from source, not an observation" in operational
assert "direct per-worker process-environment capture" in operational

for token in (
    'o14_cache_root="${O14_JIT_CACHE_ROOT:-',
    'export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${o14_cache_root}/cuda}"',
    'export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${o14_cache_root}/xdg}"',
    'export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${o14_cache_root}/vllm}"',
    'export B12X_CUTE_COMPILE_CACHE_DIR="${B12X_CUTE_COMPILE_CACHE_DIR:-${o14_cache_root}/b12x-cute}"',
    'export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${o14_cache_root}/triton}"',
    'export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${o14_cache_root}/torchinductor}"',
    'export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${o14_cache_root}/torch-extensions}"',
):
    assert token in serve, token
assert re.search(r"(?m)^O14_JIT_CACHE_ROOT=/var/cache/o14$", env)
for name, suffix in {
    "CUDA_CACHE_PATH": "cuda",
    "XDG_CACHE_HOME": "xdg",
    "VLLM_CACHE_ROOT": "vllm",
    "B12X_CUTE_COMPILE_CACHE_DIR": "b12x-cute",
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "TORCH_EXTENSIONS_DIR": "torch-extensions",
}.items():
    assert re.search(
        rf"(?m)^{re.escape(name)}=/var/cache/o14/{re.escape(suffix)}$",
        env,
    ), name

for token in (
    "TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas",
    "TORCH_CUDA_ARCH_LIST=12.1a",
    "FLASHINFER_CUDA_ARCH_LIST=12.1a",
):
    assert token in dockerfile_repro, token
assert "CUDA_DEVICE_MAX_CONNECTIONS" not in dockerfile_repro
assert '_R17_DRAFT_TEMP_SCALE = float(os.environ.get("R17_DRAFT_TEMP_SCALE", "1.0"))' in speculator_source

unsupported_unconsumed_overrides = (
    "R17_DRAFT_TEMP_SCALE",
    "VLLM_NVFP4_GEMM_BACKEND",
    "VLLM_NVFP4_ALLOW_SLOW_FALLBACK",
    "VLLM_QUANTIZATION_DISABLE_FUSED_MOE",
)
for name in unsupported_unconsumed_overrides:
    assert name not in serve, name
    assert name not in env, name
assert "intentionally omits" in operational
assert "source-complete O14 runtime published here contains no consumer" in operational

topology_variables = (
    "RAY_ADDRESS",
    "VLLM_HOST_IP",
    "NCCL_SOCKET_IFNAME",
    "GLOO_SOCKET_IFNAME",
    "NCCL_IB_HCA",
    "NCCL_IB_GID_INDEX",
    "NCCL_IB_TC",
    "UCX_NET_DEVICES",
    "UCX_TLS",
)
for name in topology_variables:
    assert f"`{name}`" in operational, name
    assignment = rf"(?m)^\s*(?:export\s+)?{re.escape(name)}\s*="
    assert re.search(assignment, operational) is None, name
    assert re.search(assignment, serve) is None, name
    assert re.search(assignment, env) is None, name
for token in (
    "GPU access",
    "host networking and host IPC",
    "16 GiB of shared memory",
    "unlimited memlock",
    "operator-selected RDMA device mappings",
    "writable persistent cache mount",
    "rather than a socket fallback",
):
    assert token in operational, token
assert re.search(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", operational) is None
assert re.search(r"/(?:Users|home)/", operational) is None
private_ipv4 = r"\b(?:10\.|192\.168\.|169\.254\.|172\.(?:1[6-9]|2[0-9]|3[01])\.)"
for public_launch_text in (serve, env):
    assert re.search(private_ipv4, public_launch_text) is None
    assert re.search(r"/(?:Users|home)/", public_launch_text) is None
for token in (
    "operator-prepared local checkpoint directory",
    "does not authorize an agent",
    "XDG cache itself is not relocated",
    "No checked-in O14 overlay consumes these names",
):
    assert token in operational, token
for relative, link in {
    "recipe/README.md": "../docs/OPERATIONAL_ENVELOPE.md",
    "docs/PUBLIC_BUILD.md": "OPERATIONAL_ENVELOPE.md",
    "reproducibility/README.md": "../docs/OPERATIONAL_ENVELOPE.md",
}.items():
    assert link in (ROOT / relative).read_text(), relative
for token in (
    "Continue to build or serve only when `status` is exactly `READY`",
    "unknown, `CANDIDATE`, `TESTING`, rejected, or superseded",
    "There is no published O14 Fast OCI image",
    "Do not build, pull, or launch Balanced",
    "Do not derive a Balanced tag from Fast",
    "verified immutable `sha256:` image digest",
    "100000 Mb/s per active RDMA link",
):
    assert token in agents, token
assert "docker pull" not in agents
assert "200GbE" not in agents

readme = (ROOT / "README.md").read_text()
for token in (
    "O14 Fast",
    "READY",
    "O14 Fast — 250K total KV, READY",
    "allocator `250023`",
    "KV_CACHE_MEMORY_BYTES=7995534848",
    "O14 Balanced",
    "TESTING / DO NOT DEPLOY",
    "500237 allocator expectation",
    "100000 Mb/s per active RDMA link",
    "Historical 399K O14 campaign results",
    "No public O14 Fast or O14 Balanced OCI image is claimed",
    "25.40 / 25.61",
    "54.68 / 53.62",
    "36.63",
    "80.56",
    "661.1",
    "0.27.2.dev0",
    "Build A v0 is the kernel active",
    "FIRST RESCORE FIRED",
    "71/71",
    "docs/VLLM_UPSTREAMING.md",
):
    assert token in readme, token
assert "production profile" not in readme
benchmarks_doc = (ROOT / "docs/BENCHMARKS.md").read_text()
port_lineage = (ROOT / "docs/PORT_LINEAGE.md").read_text()
assert "not revalidated" not in readme.lower()
assert "not revalidated" not in benchmarks_doc.lower()
assert "not revalidated" not in (ROOT / "recipe/README.md").read_text().lower()
assert "historical 2026-08-13 campaign" in port_lineage
assert "O14 reserves 12,792,889,000 bytes per rank" not in port_lineage
assert "O14 Fast — 250K total KV, READY" in port_lineage

upstreaming = (ROOT / "docs/VLLM_UPSTREAMING.md").read_text()
upstreaming_lines = upstreaming.splitlines()
catalogue_header = (
    "| Feature family | Ownership / provenance | Public O14 source / evidence "
    "| Active in measured O14? | Frozen vLLM-main status | Upstream action "
    "and public thread | Missing proof before action |"
)
catalogue_start = upstreaming_lines.index(catalogue_header) + 2
catalogue: dict[str, list[str]] = {}
for line in upstreaming_lines[catalogue_start:]:
    if not line.startswith("|"):
        break
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    assert len(cells) == 7, cells[0]
    assert "](" in cells[2], cells[0]
    catalogue[cells[0]] = cells
assert len(catalogue) == 18

probabilistic = " ".join(catalogue["**O14 probabilistic draft sampling**"])
for token in (
    "Native/vLLM-derived",
    "draft_sample_method=probabilistic",
    "draft_sample_method={greedy,probabilistic}",
    "**No** contribution",
    "shaped draft distribution `q`",
    "DraftModelSpeculator.sample_draft",
):
    assert token in probabilistic, token

full_cg = " ".join(catalogue["**Full-CUDA-graph adaptive verification shapes**"])
for token in (
    "6/12/18/24",
    "DCP1 does **not** validate data parallelism (DP)",
    "dynamic speculative decoding nevertheless remained disabled under DP",
    "93e2ab71119ff08805adc93be75196450382b088",
    "actual DP>1 multi-rank evidence",
    "source is public",
):
    assert token in full_cg, token

for feature in (
    "**Adaptive MTP depths 2/4/5 and telemetry**",
    "**B12X sparse MLA, sparse indexer, and int64/direct-K repair**",
    "**Compact NVFP4 MLA KV (`nvfp4_ds_mla`)**",
    "**Quantized MTP packed-module loading**",
):
    assert "../reproducibility/" in catalogue[feature][2], feature

print("VALIDATION_OK")
