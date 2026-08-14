#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
REQUIRED = {
    ".gitignore",
    "README.md",
    "LICENSE",
    "NOTICE",
    "recipe/o14.env.example",
    "recipe/serve-o14.sh",
    "recipe/README.md",
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

serve = (ROOT / "recipe/serve-o14.sh").read_text()
env = (ROOT / "recipe/o14.env.example").read_text()
mla = (ROOT / "overlays/mla_attention.py").read_text()
dockerfile = (ROOT / "docker/Dockerfile.o14-overlay").read_text()
for token in ("O14_EXECUTE", "B12X_MLA_SPARSE", "nvfp4_ds_mla", "6,12,18,24"):
    assert token in serve, token
assert "VLLM_BUILDA_BMM" in serve
assert not any(line.startswith("export VLLM_BUILDA_VER=") for line in serve.splitlines())
assert not any(line.startswith("VLLM_BUILDA_VER=") for line in env.splitlines())
assert "builda_bmm_v0 import" in mla and "builda_bmm_v1 import" not in mla
for token in (
    "overlays/mla_attention.py",
    "overlays/logits_processor.py",
    "kernels/builda_bmm_v0.py",
    "kernels/builda_bmm_v1.py",
    "runtime/r17_shim.py",
):
    assert token in dockerfile, token

readme = (ROOT / "README.md").read_text()
for token in (
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
