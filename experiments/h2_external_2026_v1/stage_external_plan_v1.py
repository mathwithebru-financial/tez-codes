"""Stage the external H2 preparation plan; no training or new-period access.

All archived arrays, including the old test arrays, are treated as opaque bytes.
This command does not create the final analysis lock. The separate STL training
and analysis-lock gates must finish before any new-period prices are acquired.
"""
from __future__ import annotations

import argparse
import datetime as dt
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Callable


PROJECT_ROOT = Path("/content/drive/MyDrive/tez_transformer_h2_external_2026_v1")
SOURCE_ROOT = Path("/content/drive/MyDrive/tez_transformer_v4_repro")
OLD_H2_ROOT = Path("/content/drive/MyDrive/tez_transformer_h2_t4_v1")
RUNTIME_PYTHON = Path("/content/h2_t4_runtime_v1/venv/bin/python")
PROTECTED_ROOTS = (
    SOURCE_ROOT,
    Path("/content/drive/MyDrive/tez_transformer_h2_v1"),
    OLD_H2_ROOT,
)
OLD_MANIFEST_SHA256 = "517c06f3b1a5d71de9739f56ef6a2af7cf9a918ed2cf303a52d30bfc7b9f393a"
OLD_FROZEN_SHA256 = "12cf2481978546689f9087d7d97cb4d8325756a15f393a9c936a31e2a47876f4"
STL_SOURCE_SHA256 = "df4fa2b29a6b522fc410f693a30520e045c144a15f7c7368af5eb6a1f0f12566"
OLD_FREEZE_SHA256 = "152451100bb0291e5ce32dbd48a75820ced33bcbf723b187e89330a69db0026c"
STL_SOURCE_RELATIVE = "scripts/08B_learned_baselines_test_v4.py"
BUNDLE_FILES = frozenset({
    "protocol_plan_v1.json", "stage_external_plan_v1.py",
    "run_stl_controls_v1.py", "prepare_external_sequences_v1.py", "README.md",
})
BUNDLE_MANIFEST = "bundle_manifest_v1.json"
CORE_PACKAGES = {
    "torch": "2.11.0+cu128", "numpy": "2.3.5", "pandas": "2.2.3",
    "scipy": "1.17.0", "scikit-learn": "1.6.1",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def identity(data: bytes) -> dict:
    return {"sha256": sha256(data), "size_bytes": len(data)}


def safe_root(value: str | Path, *, must_exist: bool = True) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if ".." in path.parts:
        raise ValueError(f"Parent traversal is not allowed: {path}")
    for component in reversed((path, *path.parents)):
        if component.is_symlink():
            raise ValueError(f"Symlink paths are not allowed: {component}")
        if component.exists() and not component.is_dir():
            raise ValueError(f"Expected a directory: {component}")
    path = Path(os.path.abspath(path))
    if must_exist and not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")
    return path


def safe_relative(value: str) -> str:
    relative = PurePosixPath(value)
    if (not value or "\\" in value or "\x00" in value or relative.is_absolute()
            or any(p in (".", "..") for p in relative.parts)
            or relative.as_posix() != value):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return value


def safe_file(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    path = root / safe_relative(relative)
    for component in reversed((path, *path.parents)):
        if component == root.parent:
            continue
        if component.is_symlink():
            raise ValueError(f"Symlink paths are not allowed: {component}")
        if component != path and component.exists() and not component.is_dir():
            raise ValueError(f"A parent path is not a directory: {component}")
    if path.exists() and not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")
    if must_exist and not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def read_verified(root: Path, relative: str, expected: dict) -> bytes:
    data = safe_file(root, relative).read_bytes()
    if identity(data) != {"sha256": expected["sha256"],
                          "size_bytes": expected["size_bytes"]}:
        raise ValueError(f"File identity mismatch: {root / relative}")
    return data


def clean_environment() -> dict:
    environment = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        environment.pop(key, None)
    environment.update({"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    return environment


def collect_runtime() -> dict:
    """Query the restored interpreter. The CUDA query does not fit a model."""
    if not RUNTIME_PYTHON.is_file():
        raise FileNotFoundError(f"Restore the H2 runtime first: {RUNTIME_PYTHON}")
    code = r'''
import hashlib, importlib.metadata as metadata, json, platform, re, subprocess
from pathlib import Path
import torch
packages = {name: metadata.version(name) for name in
            ["torch", "numpy", "pandas", "scipy", "scikit-learn"]}
def normalize(name): return re.sub(r"[-_.]+", "-", name).lower()
inventory = {}
for distribution in metadata.distributions():
    name = normalize(distribution.metadata["Name"])
    if name in inventory and inventory[name] != distribution.version:
        raise RuntimeError("Duplicate installed distribution: " + name)
    inventory[name] = distribution.version
canonical_freeze = "".join(name + "==" + inventory[name] + "\n"
                           for name in sorted(inventory)).encode()
freeze_file = Path("/content/h2_t4_runtime_v1/installed_packages_freeze.txt")
freeze = canonical_freeze
freeze_method = "normalized_importlib_metadata_inventory"
if freeze_file.is_file():
    archived = freeze_file.read_bytes()
    recorded = {}
    simple = True
    for line in archived.decode().splitlines():
        if not line.strip() or line.lstrip().startswith("#"): continue
        if line.count("==") != 1:
            simple = False
            break
        name, version = line.split("==", 1)
        recorded[normalize(name)] = version
    if simple and recorded == inventory:
        freeze, freeze_method = archived, "bootstrap_uv_freeze_verified_against_installed_inventory"
gpu_available = torch.cuda.is_available()
gpu_count = torch.cuda.device_count() if gpu_available else 0
driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                        check=True, capture_output=True, text=True, timeout=15).stdout.strip()
print(json.dumps({"python_version": platform.python_version(), "packages": packages,
    "cuda_available": gpu_available, "cuda_runtime": torch.version.cuda,
    "gpu_count": gpu_count,
    "gpu_name": torch.cuda.get_device_name(0) if gpu_count else None,
    "device_capability": list(torch.cuda.get_device_capability(0)) if gpu_count else None,
    "cudnn_version": torch.backends.cudnn.version(), "driver_version": driver,
    "installed_packages": inventory,
    "installed_packages_freeze_sha256": hashlib.sha256(freeze).hexdigest(),
    "freeze_method": freeze_method}, sort_keys=True))
'''
    completed = subprocess.run(
        [str(RUNTIME_PYTHON), "-B", "-c", code], check=True,
        capture_output=True, text=True, timeout=60, env=clean_environment(),
    )
    runtime = json.loads(completed.stdout)
    validate_runtime(runtime)
    runtime["freeze_matches_old_runtime"] = (
        runtime["installed_packages_freeze_sha256"] == OLD_FREEZE_SHA256
    )
    runtime["old_installed_packages_freeze_sha256"] = OLD_FREEZE_SHA256
    return runtime


def validate_runtime(runtime: dict) -> None:
    expected = {"python_version": "3.12.13", "packages": CORE_PACKAGES,
                "cuda_available": True, "cuda_runtime": "12.8", "gpu_count": 1,
                "device_capability": [7, 5], "cudnn_version": 91900}
    for key, value in expected.items():
        if runtime.get(key) != value:
            raise RuntimeError(f"H2 runtime mismatch for {key}: {runtime.get(key)!r}")
    if "T4" not in str(runtime.get("gpu_name", "")).upper():
        raise RuntimeError("A single Tesla T4 is required.")


def assert_destination_available(project_root: Path, relative: str, data: bytes) -> None:
    destination = safe_file(project_root, relative, must_exist=False)
    if destination.exists() and destination.read_bytes() != data:
        raise ValueError(f"Existing destination differs; nothing overwritten: {destination}")


def write_exclusive_or_identical(project_root: Path, relative: str, data: bytes) -> None:
    assert_destination_available(project_root, relative, data)
    destination = safe_file(project_root, relative, must_exist=False)
    if destination.exists():
        return
    safe_root(destination.parent, must_exist=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(destination, flags, 0o600), "wb") as handle:
        handle.write(data)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError as error:
            if error.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
                raise
    if destination.read_bytes() != data:
        raise OSError(f"Newly staged bytes failed readback verification: {destination}")


def stage_project(project_root: Path, bundle_root: Path, source_root: Path,
                  old_h2_root: Path, *, runtime_collector: Callable[[], dict] | None = None) -> dict:
    project_root = safe_root(project_root, must_exist=False)
    bundle_root, source_root, old_h2_root = (
        safe_root(bundle_root), safe_root(source_root), safe_root(old_h2_root)
    )
    for protected in (*PROTECTED_ROOTS, source_root, old_h2_root):
        protected = safe_root(protected, must_exist=False)
        if project_root.is_relative_to(protected) or protected.is_relative_to(project_root):
            raise ValueError(f"New project collides with a protected source: {protected}")

    actual_bundle_names = {path.name for path in bundle_root.iterdir()}
    if actual_bundle_names != BUNDLE_FILES | {BUNDLE_MANIFEST}:
        raise ValueError(f"Bundle must contain exactly the six allowlisted files: {sorted(actual_bundle_names)}")
    manifest_bytes = safe_file(bundle_root, BUNDLE_MANIFEST).read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest.get("files"), dict) or set(manifest["files"]) != BUNDLE_FILES:
        raise ValueError("Bundle manifest must identify exactly five files and exclude itself.")
    bundle = {name: read_verified(bundle_root, name, manifest["files"][name])
              for name in sorted(BUNDLE_FILES)}
    bundle[BUNDLE_MANIFEST] = manifest_bytes
    plan = json.loads(bundle["protocol_plan_v1.json"])
    if plan.get("status") != "STAGED_PLAN_NO_NEW_DATA":
        raise ValueError("Only the preparation plan may be staged; no final lock is created here.")
    if plan.get("project_root") != str(project_root):
        raise ValueError("Project root differs from the declared preparation plan.")
    if plan.get("gates", {}).get("new_data_access_allowed_now") is not False:
        raise ValueError("The preparation plan must keep new-data access disabled.")
    if plan.get("design", {}).get("new_control", {}).get("new_fits") != 6:
        raise ValueError("The preparation plan must declare six independent STL fits.")

    old_run = safe_root(old_h2_root / "runs/t4_full_run_v1")
    old_records = {}
    for name, expected_sha in (
        ("locked_training_manifest.json", OLD_MANIFEST_SHA256),
        ("selected_checkpoints_frozen.json", OLD_FROZEN_SHA256),
    ):
        data = safe_file(old_run, name).read_bytes()
        if sha256(data) != expected_sha:
            raise ValueError(f"Archived T4 provenance mismatch: {name}")
        old_records[name] = data
    old_manifest = json.loads(old_records["locked_training_manifest.json"])
    old_frozen = json.loads(old_records["selected_checkpoints_frozen.json"])
    if (old_frozen.get("locked_training_manifest_sha256") != OLD_MANIFEST_SHA256
            or len(old_frozen.get("completed_runs", [])) != 9):
        raise ValueError("Frozen checkpoint record does not describe the nine archived fits.")
    input_identities = old_manifest.get("input_identities")
    if not isinstance(input_identities, dict) or len(input_identities) != 21:
        raise ValueError("Expected exactly 21 archived input identities.")
    input_root = safe_root(old_h2_root / "inputs")
    inputs = {safe_relative(name): read_verified(input_root, name, expected)
              for name, expected in sorted(input_identities.items())}
    stl_bytes = safe_file(source_root, STL_SOURCE_RELATIVE).read_bytes()
    if sha256(stl_bytes) != STL_SOURCE_SHA256:
        raise ValueError("Archived independent-task model source identity mismatch.")
    if STL_SOURCE_RELATIVE in inputs:
        raise ValueError("Unexpected duplicate independent-task source input.")
    inputs[STL_SOURCE_RELATIVE] = stl_bytes

    runtime = (runtime_collector or collect_runtime)()
    validate_runtime(runtime)
    immutable = {
        "project_root": str(project_root), "source_root": str(source_root),
        "old_h2_root": str(old_h2_root), "protocol_plan": identity(bundle["protocol_plan_v1.json"]),
        "input_identities": {name: identity(data) for name, data in sorted(inputs.items())},
        "source_records": {name: identity(data) for name, data in sorted(old_records.items())},
        "bundle_identities": {name: identity(data) for name, data in sorted(bundle.items())},
    }
    receipt_path = safe_file(project_root, "stage_receipt_v1.json", must_exist=False)
    existing_receipt = None
    if receipt_path.exists():
        existing_receipt = json.loads(receipt_path.read_bytes())
        if existing_receipt.get("identities") != immutable:
            raise ValueError("Existing stage receipt differs; no restaging or relocking performed.")

    jobs = [(f"inputs/{name}", data) for name, data in sorted(inputs.items())]
    jobs += [(f"source_records/{name}", data) for name, data in sorted(old_records.items())]
    jobs += [(f"implementation/{name}", data) for name, data in sorted(bundle.items())]
    jobs += [("protocol_plan_v1.json", bundle["protocol_plan_v1.json"])]
    # Preflight every existing destination before creating even the first new file.
    for relative, data in jobs:
        assert_destination_available(project_root, relative, data)
    for relative, data in jobs:
        write_exclusive_or_identical(project_root, relative, data)

    if existing_receipt is None:
        receipt = {
            "status": "H2_EXTERNAL_PLAN_STAGED_NO_NEW_DATA", "receipt_schema": 1,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "identities": immutable, "runtime_environment_at_first_stage": runtime,
            "training_started": False, "new_period_downloaded": False,
            "test_arrays_loaded": False, "final_analysis_lock_created": False,
        }
        receipt_bytes = (json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
        write_exclusive_or_identical(project_root, "stage_receipt_v1.json", receipt_bytes)
    return {
        "status": "H2_EXTERNAL_PLAN_STAGED_NO_NEW_DATA", "project_root": str(project_root),
        "n_input": len(inputs), "n_stl_fits_planned": 6,
        "protocol_plan_sha256": immutable["protocol_plan"]["sha256"],
        "stage_receipt_reused": existing_receipt is not None,
        "runtime_changed_since_first_stage": bool(existing_receipt and
            existing_receipt.get("runtime_environment_at_first_stage") != runtime),
        "runtime_environment": runtime,
        "training_started": False, "new_period_downloaded": False,
        "test_arrays_loaded": False, "final_analysis_lock_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--bundle-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--old-h2-root", type=Path, default=OLD_H2_ROOT)
    args = parser.parse_args()
    result = stage_project(args.project_root, args.bundle_root, args.source_root, args.old_h2_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
