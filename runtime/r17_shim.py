"""O14 compatibility shim for older FlashInfer autotuner packages.

Appends a no-op implementation only if `set_autotune_process_group` is absent.
Consequence: per-rank tactic selection without cross-rank timing averaging.
"""
import importlib
import pathlib


autotuner = importlib.import_module("flashinfer.autotuner")
module_file = getattr(autotuner, "__file__", None)
if not module_file:
    raise RuntimeError("flashinfer.autotuner has no filesystem module path")
path = pathlib.Path(module_file)
path = path if path.name == "__init__.py" else path.parent / "__init__.py"
source = path.read_text()
if "set_autotune_process_group" not in source:
    path.write_text(
        source
        + "\n\ndef set_autotune_process_group(group=None):\n"
        + "    return None  # O14 compatibility shim: per-rank tactics\n"
    )
    print("SHIM_ADDED")
else:
    print("SHIM_PRESENT")
