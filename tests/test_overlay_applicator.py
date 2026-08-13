#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        site = pathlib.Path(tmp)
        marlin = site / "vllm/model_executor/layers/fused_moe/experts/marlin_moe.py"
        speculator = site / "vllm/v1/worker/gpu/spec_decode/speculator.py"
        marlin.parent.mkdir(parents=True)
        speculator.parent.mkdir(parents=True)
        marlin.write_text(
            "import math\n"
            "from collections.abc import Callable\n"
            "from vllm.model_executor.layers.fused_moe.config import (\n"
            "    FusedMoEConfig,\n"
            "    FusedMoEParallelConfig,\n"
            "    FusedMoEQuantConfig,\n"
            ")\n"
            "a = dict(use_atomic_add=False,)\n"
            "b = dict(use_atomic_add=False,)\n"
        )
        speculator.write_text(
            "from abc import ABC, abstractmethod\n"
            "import torch\n"
            "import torch.nn as nn\n"
            "from vllm.config import VllmConfig, get_layers_from_vllm_config\n"
            "self.temperature.copy_(temperature)\n"
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "patches/apply_o14_overlays.py"),
                "--site-packages",
                str(site),
            ],
            check=True,
        )
        mtext = marlin.read_text()
        stext = speculator.read_text()
        assert mtext.count("use_atomic_add=_VLLM_MOE_MARLIN_ATOMIC_ADD") == 2
        assert "VLLM_MOE_MARLIN_ATOMIC_ADD" in mtext
        assert "temperature * _R17_DRAFT_TEMP_SCALE" in stext
        assert "R17_DRAFT_TEMP_SCALE" in stext

        second = subprocess.run(
            [
                sys.executable,
                str(ROOT / "patches/apply_o14_overlays.py"),
                "--site-packages",
                str(site),
            ],
            text=True,
            capture_output=True,
        )
        assert second.returncode != 0

    # Byte-for-byte proof against the two exact sources captured from live O14:
    # reverse only the documented delta, re-apply it, and require identical output.
    with tempfile.TemporaryDirectory() as tmp:
        site = pathlib.Path(tmp)
        marlin = site / "vllm/model_executor/layers/fused_moe/experts/marlin_moe.py"
        speculator = site / "vllm/v1/worker/gpu/spec_decode/speculator.py"
        marlin.parent.mkdir(parents=True)
        speculator.parent.mkdir(parents=True)
        live_marlin = (ROOT / "overlays/marlin_moe.py").read_text()
        live_speculator = (ROOT / "overlays/speculator.py").read_text()
        flag = (
            "\n# M2 (R17 final-completeness audit): marlin MoE atomic-add literal flip.\n"
            "# Read once at import; default off (byte-identical to prior behavior).\n"
            "# Set VLLM_MOE_MARLIN_ATOMIC_ADD=1 to enable the second global-reduce pass\n"
            "# skip on tiny-M expert GEMMs (32.1 ms/step Marlin MoE pool).\n"
            "_VLLM_MOE_MARLIN_ATOMIC_ADD = os.environ.get(\"VLLM_MOE_MARLIN_ATOMIC_ADD\", \"0\") == \"1\"\n"
        )
        base_marlin = live_marlin.replace("import math\nimport os\n", "import math\n", 1)
        base_marlin = base_marlin.replace(flag, "", 1)
        base_marlin = base_marlin.replace(
            "use_atomic_add=_VLLM_MOE_MARLIN_ATOMIC_ADD,",
            "use_atomic_add=False,",
        )
        flag = (
            "\n# M4 (R17 final-completeness audit, phaseF H4): draft-temperature calibrated-q.\n"
            "# Read once at import; default 1.0 is a no-op (byte-identical to prior\n"
            "# behavior). Scales the draft sampling temperature only; exact-preserving by\n"
            "# construction because rejection sampling still uses the true target/draft\n"
            "# probability ratio (see _copy_request_inputs below), so this only shifts\n"
            "# acceptance economics, never the output distribution.\n"
            "_R17_DRAFT_TEMP_SCALE = float(os.environ.get(\"R17_DRAFT_TEMP_SCALE\", \"1.0\"))\n"
        )
        base_speculator = live_speculator.replace(
            "import os\nfrom abc import ABC, abstractmethod\n",
            "from abc import ABC, abstractmethod\n",
            1,
        )
        base_speculator = base_speculator.replace(flag, "", 1)
        base_speculator = base_speculator.replace(
            "self.temperature.copy_(temperature * _R17_DRAFT_TEMP_SCALE)",
            "self.temperature.copy_(temperature)",
            1,
        )
        assert base_marlin != live_marlin and base_speculator != live_speculator
        marlin.write_text(base_marlin)
        speculator.write_text(base_speculator)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "patches/apply_o14_overlays.py"),
                "--site-packages",
                str(site),
            ],
            check=True,
        )
        assert marlin.read_text() == live_marlin
        assert speculator.read_text() == live_speculator

    print("OVERLAY_APPLICATOR_TEST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())