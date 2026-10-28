#!/usr/bin/env python3
"""Rerun the recovered Stage 09 source on verified inputs in a new directory.

This adapter was added when the complete research notebook was recovered.
The extracted scientific source is unchanged. Only its ROOT assignment is
adapted in memory; its input-hash checks and no-overwrite guards still run.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil


SOURCE_SHA256 = "b29519a53efa68600978bc3efec962b939eea2a664d25eadf3160b9a102a9612"
PROTOCOL_SHA256 = "703574062b981e5a4c5187b476c5ce5f7b1e01688818c58ec6d28b436ffbcec1"
RESULTS_SHA256 = "5b18432c1f8e3e5e0faae36564deff1d62ccec45d8b4b053ed43aed82d207d77"
PROTOCOL_RELATIVE = Path("config/09_dm_harvey_holm_protocol_lock_v4.json")
RESULTS_RELATIVE = Path("results/statistical_comparison/09_dm_harvey_holm_results_v4.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_source_inputs(project: Path, repository: Path):
    protocol_path = project / PROTOCOL_RELATIVE
    if not protocol_path.is_file():
        # The public repository publishes frozen locks under protocols/.
        protocol_path = repository / "protocols" / PROTOCOL_RELATIVE.name
    if not protocol_path.is_file() or sha256(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("The frozen Stage 09 protocol is missing or has the wrong SHA-256.")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    identities = protocol["locked_input_identities"]
    if len(identities) != 17:
        raise ValueError("The frozen Stage 09 protocol must contain exactly 17 input identities.")
    inputs = []
    for identity in identities:
        relative = Path(identity["path"])
        source = (project / relative).resolve()
        if relative.is_absolute() or not source.is_relative_to(project):
            raise ValueError("A locked input path leaves the project directory.")
        if not source.is_file():
            raise FileNotFoundError(f"Required frozen input missing: {relative}")
        if source.stat().st_size != int(identity["size_bytes"]) or sha256(source) != identity["sha256"]:
            raise ValueError(f"Frozen input identity mismatch: {relative}")
        inputs.append((source, relative))
    return protocol_path, inputs


def compare_results(reference_path: Path, reproduced_path: Path) -> dict:
    import numpy as np
    import pandas as pd

    if sha256(reference_path) != RESULTS_SHA256:
        raise ValueError("Reference result CSV does not match the frozen Stage 09 identity.")
    reference = pd.read_csv(reference_path).set_index("comparison_id").sort_index()
    reproduced = pd.read_csv(reproduced_path).set_index("comparison_id").sort_index()
    if reference.index.has_duplicates or reproduced.index.has_duplicates:
        raise AssertionError("Duplicate comparison identifiers.")
    if not reference.index.equals(reproduced.index) or not reference.columns.equals(reproduced.columns):
        raise AssertionError("Reproduced identifiers or result columns differ from the reference.")
    numeric = reference.select_dtypes(include=np.number).columns
    left, right = reference[numeric].to_numpy(dtype=float), reproduced[numeric].to_numpy(dtype=float)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise AssertionError("Non-finite numeric comparison result.")
    maximum = float(np.max(np.abs(left - right)))
    per_column = {name: float(np.max(np.abs(reference[name] - reproduced[name]))) for name in numeric}
    other = reference.columns.difference(numeric)
    equal = reference[other].astype(str).equals(reproduced[other].astype(str))
    counts = {str(key): int(value) for key, value in reproduced["interpretation"].value_counts().items()}
    expected = {"SIGNIFICANT_FINAL_BETTER": 5, "SIGNIFICANT_BASELINE_BETTER": 34, "NOT_SIGNIFICANT": 3}
    if len(reproduced) != 42 or counts != expected or maximum >= 1e-10 or not equal:
        raise AssertionError("Stage 09 reproduction failed the frozen comparison checks.")
    return {
        "status": "PASS", "comparison_count": 42, "n_per_comparison": 584,
        "interpretation_counts": counts, "all_nonnumeric_fields_equal": equal,
        "max_abs_numeric_difference_from_reference": maximum,
        "max_abs_difference_by_column": per_column, "comparison_tolerance": 1e-10,
    }


def run_stage09(project: Path, output: Path, repository: Path) -> None:
    project, output = project.expanduser().resolve(), output.expanduser().resolve()
    if not project.is_dir():
        raise FileNotFoundError(f"Frozen project directory not found: {project}")
    if output.exists() or output.is_relative_to(project):
        raise ValueError("Choose a new output directory outside the frozen input project.")
    source_path = repository / "scripts/09_dm_harvey_holm_v4.py"
    if sha256(source_path) != SOURCE_SHA256:
        raise ValueError("Recovered Stage 09 source identity mismatch.")
    protocol_path, inputs = verified_source_inputs(project, repository)
    # Check all identities before creating the separate reproduction directory.
    output.mkdir(parents=True, exist_ok=False)
    for source, relative in inputs:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    destination_protocol = output / PROTOCOL_RELATIVE
    destination_protocol.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(protocol_path, destination_protocol)
    (output / RESULTS_RELATIVE.parent).mkdir(parents=True, exist_ok=True)

    syntax = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    roots = [node for node in syntax.body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == "ROOT" for target in node.targets)]
    if len(roots) != 1:
        raise ValueError("Expected one canonical ROOT assignment in the recovered source.")
    roots[0].value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()), args=[ast.Constant(str(output))], keywords=[])
    ast.fix_missing_locations(syntax)
    namespace = {"__name__": "__main__", "__file__": str(source_path)}
    exec(compile(syntax, str(source_path), "exec"), namespace)

    check = compare_results(repository / RESULTS_RELATIVE, output / RESULTS_RELATIVE)
    check.update({
        "input_hashes_verified": 17, "source_sha256": SOURCE_SHA256,
        "protocol_sha256": PROTOCOL_SHA256, "source_adaptation": "ROOT assignment only, in memory",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy")},
    })
    check_path = output / "stage09_reproduction_check_v4.json"
    check_path.write_text(json.dumps(check, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Stage 09 isolated reproduction: PASS")
    print("Reproduction record:", check_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="analysis", required=True)
    stage09 = subparsers.add_parser("stage09", help="Recompute all 42 tests from 17 verified frozen inputs.")
    stage09.add_argument("--project-root", type=Path, required=True, help="Private frozen project containing all locked inputs.")
    stage09.add_argument("--output-dir", type=Path, required=True, help="New directory outside the input project; must not exist.")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    run_stage09(args.project_root, args.output_dir, repository)


if __name__ == "__main__":
    main()
