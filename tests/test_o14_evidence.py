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

    def test_public_probe_urls_are_local_or_environment_driven(self) -> None:
        for name in ("prose_probe.py", "peak_probe.py", "deep_prefill_probe.py"):
            text = (ROOT / "benchmarks" / name).read_text()
            self.assertIn("O14_BASE_URL", text)
            self.assertIn("127.0.0.1:8211", text)
            self.assertNotIn("192." + "168.", text)


if __name__ == "__main__":
    unittest.main()
