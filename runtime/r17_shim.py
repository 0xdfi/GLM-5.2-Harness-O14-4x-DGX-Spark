#!/usr/bin/env python3
"""Apply the exact R17 FlashInfer 0.6.15 compatibility patch.

The operation is hash-gated and idempotent. It refuses unknown preimages.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile

PREIMAGE_SHA256 = "0bc4ed866a3f679bb756bec21f2cb36c042ad35f9380a2239ff9717e808da2fc"
RESULT_SHA256 = "f1138881c26138cf7d1ace1e0c054c095657b411369e5e37c0c1ff019e1e6fa0"
PATCH_BYTES = (
    b"\n\ndef set_autotune_process_group(group=None):\n"
    b"    return None  # R17 shim: no-op, per-rank tactics\n"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def discover_target() -> Path:
    candidates: list[Path] = []
    for entry in sys.path:
        if not entry:
            continue
        candidate = Path(entry) / "flashinfer" / "autotuner" / "__init__.py"
        if candidate.is_file():
            resolved = candidate.resolve()
            if resolved not in candidates:
                candidates.append(resolved)
    if len(candidates) != 1:
        rendered = ", ".join(str(path) for path in candidates) or "none"
        raise RuntimeError(f"expected one FlashInfer autotuner target, found: {rendered}")
    return candidates[0]


def apply(path: Path) -> str:
    before = path.read_bytes()
    before_sha = sha256(before)
    if before_sha == RESULT_SHA256:
        return "SHIM_PRESENT"
    if before_sha != PREIMAGE_SHA256:
        raise RuntimeError(
            f"FlashInfer autotuner preimage mismatch: {before_sha}; "
            f"expected {PREIMAGE_SHA256}"
        )

    after = before + PATCH_BYTES
    after_sha = sha256(after)
    if after_sha != RESULT_SHA256:
        raise RuntimeError(
            f"FlashInfer autotuner result mismatch: {after_sha}; expected {RESULT_SHA256}"
        )

    mode = path.stat().st_mode
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".r17-shim-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(after)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

    if sha256(path.read_bytes()) != RESULT_SHA256:
        raise RuntimeError("FlashInfer autotuner post-write verification failed")
    return "SHIM_ADDED"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, help="explicit autotuner __init__.py path")
    args = parser.parse_args()
    path = args.path.resolve() if args.path else discover_target()
    print(f"{apply(path)} {path} sha256={RESULT_SHA256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
