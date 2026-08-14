#!/usr/bin/env python3
"""Fail-closed validation for the complete O14 public recipe pack."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import py_compile
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PRIVATE = re.compile(
    b"/" + b"home/" + b"dfi"
    + b"|/" + b"Users/" + b"daffi"
    + rb"|192\.168\."
    + rb"|BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"
    + rb"|hf_[A-Za-z0-9]{20,}"
    + rb"|gh[pousr]_[A-Za-z0-9]{20,}"
)


def die(message: str) -> None:
    raise SystemExit(f"RECIPE_VERIFY_FAILED: {message}")


def load(path: Path) -> dict:
    value: object = None
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        die(f"cannot parse {path.relative_to(ROOT)}: {exc}")
    if type(value) is not dict:
        die(f"{path.relative_to(ROOT)} must contain an object")
    return value


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_rel(text: object, context: str) -> Path:
    if not isinstance(text, str) or not text or "\\" in text:
        die(f"unsafe {context}")
    path = Path(text)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        die(f"unsafe {context}: {text!r}")
    return path


def verify_overlay(root: Path, manifest: dict, kind: str) -> int:
    files_value = manifest.get("files")
    if not isinstance(files_value, list):
        die(f"{kind} manifest files must be an array")
    files: list[object] = files_value
    expected: set[str] = set()
    for index, entry in enumerate(files):
        if type(entry) is not dict:
            die(f"{kind} entry {index} must be an object")
        rel = safe_rel(entry.get("path"), f"{kind} path").as_posix()
        if rel in expected:
            die(f"duplicate {kind} path: {rel}")
        expected.add(rel)
        target = root / rel
        if not target.is_file() or target.is_symlink():
            die(f"missing regular {kind} overlay: {rel}")
        expected_hash = entry.get("publication_sha256", entry.get("target_sha256"))
        if type(expected_hash) is not str or HEX64.fullmatch(expected_hash) is None:
            die(f"invalid {kind} target hash: {rel}")
        if sha(target) != expected_hash:
            die(f"{kind} target hash mismatch: {rel}")
        if target.suffix == ".py":
            py_compile.compile(str(target), doraise=True, cfile=str(Path("/tmp") / f"o14-{kind}-{index}.pyc"))
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        die(f"{kind} exact member mismatch: missing={sorted(expected-actual)} extras={sorted(actual-expected)}")
    return len(expected)


def verify_installed(prefix: Path, vllm_manifest: dict, b12x_manifest: dict, lock: dict) -> None:
    for package, manifest in (("vllm", vllm_manifest), ("b12x", b12x_manifest)):
        package_root = prefix / package
        for entry in manifest["files"]:
            path = package_root / safe_rel(entry["path"], f"installed {package} path")
            expected = entry.get("publication_sha256", entry.get("target_sha256"))
            if not path.is_file() or sha(path) != expected:
                die(f"installed {package} mismatch: {entry['path']}")
    flash = lock["flashinfer_compatibility_patch"]
    flash_target = prefix / flash["target_member"]
    if not flash_target.is_file() or sha(flash_target) != flash["result_sha256"]:
        die("installed FlashInfer compatibility target mismatch")
    native = prefix / "vllm" / "_C_stable_libtorch.abi3.so"
    if not native.is_file() or sha(native) != lock["native_nvfp4_cache_op"]["expected_native_sha256"]:
        die("installed native NVFP4 cache op mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--installed-prefix", type=Path, help="site-packages root to verify")
    args = parser.parse_args()

    vllm = load(REPRO / "manifests/vllm-overlay.json")
    b12x = load(REPRO / "manifests/b12x-overlay.json")
    native = load(REPRO / "manifests/native-patch.json")
    lock = load(REPRO / "build-lock.json")
    release = load(REPRO / "runtime-input-release.json")

    if vllm.get("schema") != "o14-stage0-wheel-runtime-overlay/1" or vllm.get("file_count") != 74:
        die("vLLM closure must be 74 files")
    if vllm.get("modified_count") != 67 or vllm.get("added_count") != 7:
        die("vLLM modified/added counts do not close")
    if b12x.get("schema") != "o14-b12x-git-overlay/1" or b12x.get("modified_count") != 3:
        die("B12X closure must be 3 files")
    if native.get("schema") != "o14-native-nvfp4-source-patch/1" or native.get("file_count") != 14:
        die("native closure must be 14 paths")

    vcount = verify_overlay(REPRO / "overlays/vllm/vllm", vllm, "vllm")
    bcount = verify_overlay(REPRO / "overlays/b12x/b12x", b12x, "b12x")
    patch = REPRO / "patches/native/exp1-r4-native.patch"
    if sha(patch) != native.get("patch_sha256"):
        die("native patch hash mismatch")

    shim = ROOT / "runtime/r17_shim.py"
    py_compile.compile(str(shim), doraise=True, cfile="/tmp/o14-r17-shim.pyc")
    flash = lock.get("flashinfer_compatibility_patch", {})
    shim_text = shim.read_text()
    for value in (flash.get("preimage_sha256"), flash.get("result_sha256"), flash.get("required_symbol")):
        if type(value) is not str or value not in shim_text:
            die("FlashInfer shim is not bound to the build lock")

    dockerfile = ROOT / "docker/Dockerfile.repro"
    if not dockerfile.is_file():
        die("docker/Dockerfile.repro missing")
    docker_text = dockerfile.read_text()
    for required in (
        lock["platform"]["cuda_base"]["arm64_digest"],
        lock["vllm_runtime_source"]["base_wheel_sha256"],
        lock["native_nvfp4_cache_op"]["expected_native_sha256"],
        "reproducibility/overlays/vllm/vllm",
        "reproducibility/overlays/b12x/b12x",
        "runtime/r17_shim.py",
        "recipe/serve-o14.sh",
    ):
        if required not in docker_text:
            die(f"Dockerfile missing locked input: {required}")

    recipe = (ROOT / "recipe/serve-o14.sh").read_text()
    for required in ("--tensor-parallel-size 4", "--kv-cache-dtype nvfp4_ds_mla", "--attention-backend B12X_MLA_SPARSE", "--speculative-config", "O14_EXECUTE"):
        if required not in recipe:
            die(f"serve recipe missing {required}")

    tracked = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and path.resolve() != Path(__file__).resolve()
    ]
    for path in tracked:
        if PRIVATE.search(path.read_bytes()):
            die(f"private identifier or credential pattern in {path.relative_to(ROOT)}")

    for asset in release.get("assets", []):
        if type(asset) is not dict or HEX64.fullmatch(str(asset.get("sha256", ""))) is None:
            die("release asset lacks a valid SHA-256")

    if args.installed_prefix:
        verify_installed(args.installed_prefix.resolve(), vllm, b12x, lock)

    print(f"O14_FULL_RECIPE_OK vllm={vcount} b12x={bcount} native={native['file_count']} flashinfer=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
