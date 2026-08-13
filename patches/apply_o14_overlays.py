#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Apply the exact O14 top-layer edits to a compatible vLLM 0.27 R17/O13 tree.

Every anchor and replacement count is checked. A source mismatch aborts rather
than producing an image that merely looks like O14.
"""
from __future__ import annotations

import argparse
import pathlib


def replace_exact(text: str, old: str, new: str, expected: int, label: str) -> str:
    actual = text.count(old)
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected} anchor(s), found {actual}")
    return text.replace(old, new)


def patch_marlin(path: pathlib.Path) -> None:
    text = path.read_text()
    if "_VLLM_MOE_MARLIN_ATOMIC_ADD" in text:
        raise RuntimeError("marlin overlay already present")
    text = replace_exact(text, "import math\n", "import math\nimport os\n", 1, "marlin import")
    anchor = (
        "from vllm.model_executor.layers.fused_moe.config import (\n"
        "    FusedMoEConfig,\n"
        "    FusedMoEParallelConfig,\n"
        "    FusedMoEQuantConfig,\n"
        ")\n"
    )
    flag = (
        anchor
        + "\n# M2 (R17 final-completeness audit): marlin MoE atomic-add literal flip.\n"
        "# Read once at import; default off (byte-identical to prior behavior).\n"
        "# Set VLLM_MOE_MARLIN_ATOMIC_ADD=1 to enable the second global-reduce pass\n"
        "# skip on tiny-M expert GEMMs (32.1 ms/step Marlin MoE pool).\n"
        "_VLLM_MOE_MARLIN_ATOMIC_ADD = os.environ.get(\"VLLM_MOE_MARLIN_ATOMIC_ADD\", \"0\") == \"1\"\n"
    )
    text = replace_exact(text, anchor, flag, 1, "marlin flag")
    text = replace_exact(
        text,
        "use_atomic_add=False,",
        "use_atomic_add=_VLLM_MOE_MARLIN_ATOMIC_ADD,",
        2,
        "marlin call sites",
    )
    path.write_text(text)


def patch_speculator(path: pathlib.Path) -> None:
    text = path.read_text()
    if "_R17_DRAFT_TEMP_SCALE" in text:
        raise RuntimeError("speculator overlay already present")
    text = replace_exact(text, "from abc import ABC, abstractmethod\n", "import os\nfrom abc import ABC, abstractmethod\n", 1, "speculator import")
    anchor = "import torch\nimport torch.nn as nn\n"
    flag = (
        anchor
        + "\n# M4 (R17 final-completeness audit, phaseF H4): draft-temperature calibrated-q.\n"
        "# Read once at import; default 1.0 is a no-op (byte-identical to prior\n"
        "# behavior). Scales the draft sampling temperature only; exact-preserving by\n"
        "# construction because rejection sampling still uses the true target/draft\n"
        "# probability ratio (see _copy_request_inputs below), so this only shifts\n"
        "# acceptance economics, never the output distribution.\n"
        "_R17_DRAFT_TEMP_SCALE = float(os.environ.get(\"R17_DRAFT_TEMP_SCALE\", \"1.0\"))\n"
    )
    text = replace_exact(text, anchor, flag, 1, "speculator flag")
    text = replace_exact(
        text,
        "self.temperature.copy_(temperature)",
        "self.temperature.copy_(temperature * _R17_DRAFT_TEMP_SCALE)",
        1,
        "speculator copy",
    )
    path.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-packages", type=pathlib.Path, required=True)
    args = parser.parse_args()
    base = args.site_packages
    marlin = base / "vllm/model_executor/layers/fused_moe/experts/marlin_moe.py"
    speculator = base / "vllm/v1/worker/gpu/spec_decode/speculator.py"
    for path in (marlin, speculator):
        if not path.is_file():
            raise SystemExit(f"missing compatible base file: {path}")
    patch_marlin(marlin)
    patch_speculator(speculator)
    print("O14_OVERLAYS_APPLIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
