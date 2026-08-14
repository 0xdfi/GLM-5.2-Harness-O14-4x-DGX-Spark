#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class O14EvidenceTest(unittest.TestCase):
    def test_active_builda_version_matches_deployed_wrapper(self) -> None:
        data = json.loads((ROOT / "evidence/o14-results.json").read_text())
        mla = (ROOT / "overlays/mla_attention.py").read_text()
        serve = (ROOT / "recipe/serve-o14.sh").read_text()
        self.assertEqual(data["runtime"]["builda_active_version"], "v0")
        self.assertIn("builda_bmm_v0 import", mla)
        self.assertNotIn("builda_bmm_v1 import", mla)
        self.assertFalse(
            any(line.startswith("export VLLM_BUILDA_VER=") for line in serve.splitlines())
        )

    def test_final_battery_exact_values(self) -> None:
        records = [
            json.loads(line)
            for line in (ROOT / "evidence/final-o14-battery.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(records), 7)
        self.assertEqual([records[0]["tok_s"], records[1]["tok_s"]], [25.4, 25.61])
        self.assertEqual(
            [records[2]["aggregate_tok_s"], records[3]["aggregate_tok_s"]],
            [54.68, 53.62],
        )
        self.assertEqual(records[4]["tok_s"], 36.63)
        self.assertEqual(records[5]["aggregate_tok_s"], 80.56)
        self.assertEqual(records[6]["prompt_tokens"], 187022)
        self.assertEqual(records[6]["elapsed_seconds"], 282.9)
        self.assertEqual(records[6]["prefill_tok_s"], 661.1)

    def test_exact_rescore_fire_proof_is_four_rank(self) -> None:
        data = json.loads((ROOT / "evidence/o14-results.json").read_text())
        head = data["runtime"]["lmhead_exact_rescore"]
        self.assertTrue(head["first_rescore_fired_all_four_ranks"])
        self.assertEqual(len(head["changed_entries_by_tp_rank"]), 4)
        self.assertTrue(all(value > 0 for value in head["changed_entries_by_tp_rank"]))

    def test_upstream_catalogue_closes_review_gaps(self) -> None:
        text = (ROOT / "docs/VLLM_UPSTREAMING.md").read_text()
        lines = text.splitlines()
        header = (
            "| Feature family | Ownership / provenance | Public O14 source / evidence "
            "| Active in measured O14? | Frozen vLLM-main status | Upstream action "
            "and public thread | Missing proof before action |"
        )
        start = lines.index(header) + 2
        rows: dict[str, list[str]] = {}
        for line in lines[start:]:
            if not line.startswith("|"):
                break
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            self.assertEqual(len(cells), 7)
            rows[cells[0]] = cells

        self.assertEqual(len(rows), 18)
        self.assertTrue(all("](" in cells[2] for cells in rows.values()))

        probabilistic = rows["**O14 probabilistic draft sampling**"]
        probabilistic_text = " ".join(probabilistic)
        for token in (
            "Native/vLLM-derived",
            "draft_sample_method=probabilistic",
            "draft_sample_method={greedy,probabilistic}",
            "**No** contribution",
            "shaped draft distribution `q`",
            "DraftModelSpeculator.sample_draft",
        ):
            self.assertIn(token, probabilistic_text)

        full_cg = rows["**Full-CUDA-graph adaptive verification shapes**"]
        full_cg_text = " ".join(full_cg)
        for token in (
            "6/12/18/24",
            "DCP1 does **not** validate data parallelism (DP)",
            "dynamic speculative decoding nevertheless remained disabled under DP",
            "93e2ab71119ff08805adc93be75196450382b088",
            "actual DP>1 multi-rank evidence",
            "source is public",
        ):
            self.assertIn(token, full_cg_text)

        for feature in (
            "**Adaptive MTP depths 2/4/5 and telemetry**",
            "**B12X sparse MLA, sparse indexer, and int64/direct-K repair**",
            "**Compact NVFP4 MLA KV (`nvfp4_ds_mla`)**",
            "**Quantized MTP packed-module loading**",
        ):
            self.assertIn("../reproducibility/", rows[feature][2])

        normalized_text = " ".join(text.split())
        for token in (
            "AGENTS.md",
            "pure code-agent PRs are **not allowed**",
            "understand and defend the change end-to-end",
            "review every changed line",
            "run the relevant tests",
            "does not duplicate an existing PR",
            "list test commands and results",
            "include model-evaluation results when output, accuracy, or serving is",
            "clearly disclose the AI assistance",
            "`Co-authored-by:`",
            "`Assisted-by:`",
            "alongside the DCO `Signed-off-by:`",
            "architectural changes **>500 LOC**",
            "excluding kernel, data, config, and test code",
            "change modifies user-facing behavior",
        ):
            self.assertIn(token, normalized_text)
        self.assertNotIn("discourages “pure agent” PRs", text)

        builda_action = rows["**Build A v0 / v1 tiny MLA BMM**"][5]
        w8_action = rows["**W8 `lm_head` top-64 selected-row rescore**"][5]
        marlin_action = rows["**Marlin fused-MoE atomic add**"][5]
        adaptive_action = rows["**Adaptive MTP depths 2/4/5 and telemetry**"][5]
        self.assertIn("**Comment now**", builda_action)
        self.assertIn("vLLM#36297", builda_action)
        self.assertIn("**Watch / no comment now**", w8_action)
        self.assertIn("different fused tied-head mechanism", w8_action)
        self.assertIn("**No contact without new evidence**", marlin_action)
        self.assertIn("stronger GB10 atomic-off/on and paired-layer evidence", marlin_action)
        self.assertIn("**Watch / no comment now**", adaptive_action)
        self.assertIn("needs a rebase and is inactive", adaptive_action)
        self.assertLess(text.index("2. **Build A"), text.index("3. **Marlin"))
        self.assertEqual(text.count("**COMMENT NOW —"), 1)
        for token in (
            "**WATCH / NO COMMENT NOW — [`vLLM#47111`]",
            "**WATCH / NO COMMENT NOW — [`vLLM#48870`]",
            "**NO CONTACT WITHOUT NEW EVIDENCE — [`vLLM#48569`]",
            "**NO UMBRELLA COMMENT NOW — [`vLLM#46654`]",
        ):
            self.assertIn(token, text)

        self.assertNotIn("reconciled 27-file patch tree", text)
        self.assertNotIn("assisted that donor/campaign lineage", text)
        for path in ("docs/PORT_LINEAGE.md", "docs/CUSTOM_RUNTIME.md"):
            count_text = (ROOT / path).read_text()
            self.assertIn("74 vLLM files", count_text)
            self.assertIn("3 modified B12X files", count_text)
            self.assertIn("14 native", count_text)
            self.assertNotIn("scope reconciliation is unpublished", count_text)

        provenance = (ROOT / "docs/PROVENANCE.md").read_text()
        self.assertIn("co-authored `Claude Fable 5", provenance)
        self.assertIn("final Fable session transcript", provenance)
        self.assertIn("complete public Fable attribution boundary", provenance)
        self.assertIn("does not evidence Fable assistance", provenance)

    def test_public_probe_urls_are_local_or_environment_driven(self) -> None:
        for name in ("prose_probe.py", "peak_probe.py", "deep_prefill_probe.py"):
            text = (ROOT / "benchmarks" / name).read_text()
            self.assertIn("O14_BASE_URL", text)
            self.assertIn("127.0.0.1:8211", text)
            self.assertNotIn("192." + "168.", text)


if __name__ == "__main__":
    unittest.main()
