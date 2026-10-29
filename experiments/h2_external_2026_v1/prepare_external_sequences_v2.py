#!/usr/bin/env python3
"""Versioned NumPy date-scalar compatibility amendment to external preparation.

The original plan, before-data analysis lock, implementations, model selections,
prices, transforms and output schema remain intact. Only the failing metadata
conversion Timestamp(numpy.str_) becomes Timestamp(str(numpy.str_)). The changed
orchestration is recorded after price access and before external model metrics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import types

DEFAULT_ROOT = Path("/content/drive/MyDrive/tez_transformer_h2_external_2026_v1")
ANALYSIS_LOCK_SHA256 = "a155ad428996a6c1d40ca6a755212206ca9092a7b0fdc76200c3a1f063bdb274"
ORIGINAL_SOURCE_SHA256 = "8648220271dbed9d537e86f42f944b52ce69755a77a846d84a4fadecab501179"
RAW_SNAPSHOT_SHA256 = "a9c581a851d69498cd48cc5ee8bda9422c927b25c9a81ee1f154d9a74d95f889"
ORIGINAL_SOURCE = "implementation/prepare_external_sequences_v1.py"
CORRECTED_SOURCE = "implementation/prepare_external_sequences_v2.py"
EVALUATION_WRAPPER_SOURCE = "implementation/evaluate_external_h2_numpy_dates_v1.py"
AMENDMENT_PATH = "technical_amendment_numpy_dates_v1.json"
AMENDMENT_STATUS = "TECHNICAL_AMENDMENT_RECORDED_AFTER_PRICE_ACCESS_BEFORE_EXTERNAL_MODEL_EVALUATION"
RAW_PATH = "new_period/acquisition_v1/raw_prices.csv"
ACQUISITION_PATH = "new_period/acquisition_v1/acquisition_record.json"
PROTECTED_ROOTS = tuple(Path("/content/drive/MyDrive") / p for p in
    ("tez_transformer_v4_repro", "tez_transformer_h2_v1", "tez_transformer_h2_t4_v1"))


class AmendmentError(RuntimeError):
    """The original scientific identities or amendment chronology do not match."""


def reject_symlinks(path: Path) -> None:
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise AmendmentError("A symbolic-link path is forbidden.")


def checked_root(root: Path) -> Path:
    root = Path(root).expanduser()
    reject_symlinks(root)
    root = root.resolve()
    if not root.is_dir() or any(root.is_relative_to(p) or p.is_relative_to(root) for p in PROTECTED_ROOTS):
        raise AmendmentError("A separate existing external-project directory is required.")
    return root


def relative_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise AmendmentError("Only a safe project-relative artifact path is accepted.")
    path = root / relative
    reject_symlinks(path)
    if path.resolve() == root or not path.resolve().is_relative_to(root):
        raise AmendmentError("Artifact path escapes the external project.")
    return path


def file_identity(root: Path, relative: str, expected_sha: str | None = None) -> dict:
    path = relative_path(root, relative)
    if not path.is_file():
        raise AmendmentError("Required artifact is missing: " + relative)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    if expected_sha is not None and value != expected_sha:
        raise AmendmentError("Artifact identity differs: " + relative)
    return {"path": relative, "sha256": value, "size_bytes": path.stat().st_size}


def read_object(path: Path) -> dict:
    reject_symlinks(path)
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise AmendmentError("Expected a JSON object: " + path.name)
    return value


def checked_original_module(root: Path):
    """Import the unchanged original helper only after checking its real bytes."""
    root = checked_root(root)
    file_identity(root, "analysis_locked_v1.json", ANALYSIS_LOCK_SHA256)
    file_identity(root, ORIGINAL_SOURCE, ORIGINAL_SOURCE_SHA256)
    path = relative_path(root, ORIGINAL_SOURCE)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != ORIGINAL_SOURCE_SHA256:
        raise AmendmentError("Original helper changed during loading.")
    module = types.ModuleType("h2_original_preparation_for_numpy_date_amendment")
    module.__file__ = str(path)
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def amendment_core(root: Path, base=None) -> dict:
    """Read original metadata and opaque file identities; deserialize no models."""
    root = checked_root(root)
    base = checked_original_module(root) if base is None else base
    parent = file_identity(root, "analysis_locked_v1.json", ANALYSIS_LOCK_SHA256)
    lock, evidence = base.verify_analysis_lock(root)
    base.verify_frozen_inputs(root)
    original = file_identity(root, ORIGINAL_SOURCE, ORIGINAL_SOURCE_SHA256)
    corrected = file_identity(root, CORRECTED_SOURCE)
    evaluation_wrapper = file_identity(root, EVALUATION_WRAPPER_SOURCE)
    if Path(__file__).resolve() != relative_path(root, CORRECTED_SOURCE).resolve():
        raise AmendmentError("Execute or import the corrected implementation from this exact project.")
    raw = file_identity(root, RAW_PATH, RAW_SNAPSHOT_SHA256)
    acquisition = file_identity(root, ACQUISITION_PATH)
    acquired = read_object(relative_path(root, ACQUISITION_PATH))
    if (acquired.get("status") != "ACQUISITION_COMPLETE"
            or acquired.get("analysis_lock_sha256") != ANALYSIS_LOCK_SHA256
            or acquired.get("protocol_plan_sha256") != lock["protocol_plan_sha256"]
            or acquired.get("implementation_sha256") != lock["implementation_sha256"]
            or acquired.get("raw_file_sha256") != RAW_SNAPSHOT_SHA256
            or acquired.get("models_evaluated") is not False
            or acquired.get("old_test_arrays_loaded") is not False
            or acquired.get("old_project_writes") is not False):
        raise AmendmentError("Acquisition does not belong to the unchanged before-data analysis lock.")
    return {
        "status": AMENDMENT_STATUS, "amendment_schema": 1,
        "amendment_id": "H2_EXTERNAL_2026_NUMPY_DATE_SCALAR_COMPATIBILITY_V1",
        "project_root": str(root), "parent_analysis_lock": parent,
        "protocol_plan_sha256": lock["protocol_plan_sha256"],
        "original_implementation": original, "corrected_implementation": corrected,
        "evaluation_wrapper_implementation": evaluation_wrapper,
        "raw_snapshot": raw, "acquisition_record": acquisition,
        "existing_h2_checkpoints_manifest_sha256": evidence["existing_h2_checkpoints_manifest_sha256"],
        "selected_stl_checkpoints_manifest_sha256": evidence["selected_stl_checkpoints_manifest_sha256"],
        "change": {
            "function": "prepare", "purpose": "Python date-text compatibility for metadata lookup",
            "old_expression": 'pd.Timestamp(arrays["anchor_dates_external.npy"][0])',
            "new_expression": 'pd.Timestamp(str(arrays["anchor_dates_external.npy"][0]))',
            "failure": "TypeError: Expected unicode, got numpy.str_",
            "failure_stage": "First-anchor metadata conversion, before commit_outputs and any external model evaluation",
        },
        "new_data_already_accessed": True,
        "external_model_metrics_evaluated_before_amendment": False,
        "chronology_scope": "This dedicated external project; no claim about inaccessible or unsaved earlier work.",
        "amendment_recorded_before_corrected_preparation": True,
        "analytical_decisions_changed": False,
        "unchanged": ["hypothesis", "selected models and checkpoints", "raw snapshot", "asset identities",
            "target dates", "lookback", "feature and target arithmetic", "array dtypes and values",
            "train-fitted scalers", "missingness policy", "coverage thresholds", "checkpoint selection",
            "loss definitions", "DM/HLN/Bartlett settings", "Holm families"],
        "original_plan_and_analysis_lock_modified": False,
        "training_started_by_this_code": False, "old_test_arrays_loaded": False,
    }


def verify_technical_amendment(root: Path) -> tuple[dict, dict]:
    """Verify immutable amendment, current v2 bytes and all original gates."""
    root = checked_root(root)
    identity = file_identity(root, AMENDMENT_PATH)
    record = read_object(relative_path(root, AMENDMENT_PATH))
    core = {k: v for k, v in record.items() if k not in {"created_at_utc", "writer_python"}}
    if (core != amendment_core(root) or not isinstance(record.get("created_at_utc"), str)
            or not isinstance(record.get("writer_python"), str)):
        raise AmendmentError("Existing immutable technical amendment differs from current identities.")
    if file_identity(root, AMENDMENT_PATH) != identity:
        raise AmendmentError("Technical amendment changed during verification.")
    return record, identity


def ensure_technical_amendment(root: Path, base) -> tuple[dict, dict]:
    root = checked_root(root)
    path = relative_path(root, AMENDMENT_PATH)
    if path.exists():
        return verify_technical_amendment(root)
    new_period = relative_path(root, "new_period")
    if relative_path(root, "new_period/prepared_v1").exists():
        raise AmendmentError("Prepared output already exists; no retroactive amendment was created.")
    if new_period.exists():
        names = [p.name for p in new_period.iterdir()]
        if any(name.startswith("evaluation") or name.startswith(".evaluate")
               or name.startswith(".prepared_v1-stage-") or name == ".prepare_external_sequences_v1.lock"
               for name in names):
            raise AmendmentError("An evaluation or partial preparation artifact already exists; retain it for review.")
    core = amendment_core(root, base)
    # Recheck output chronology immediately before the one exclusive record write.
    if path.exists():
        return verify_technical_amendment(root)
    if relative_path(root, "new_period/prepared_v1").exists():
        raise AmendmentError("Prepared output appeared during amendment creation.")
    if any(p.name.startswith("evaluation") or p.name.startswith(".evaluate") for p in new_period.iterdir()):
        raise AmendmentError("Evaluation output appeared during amendment creation.")
    record = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "writer_python": platform.python_version(), **core}
    data = (json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    try:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError as exc:
                if exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
                    raise
    except FileExistsError:
        return verify_technical_amendment(root)
    if path.read_bytes() != data:
        raise AmendmentError("New amendment failed exact byte readback; retain it for review.")
    return verify_technical_amendment(root)


def prepare(root: Path, raw_file: Path, acquisition_record: Path) -> tuple[dict, bool]:
    """Original preparation orchestration with the one recorded date-text cast."""
    root = checked_root(root)
    raw_file, acquisition_record = Path(raw_file), Path(acquisition_record)
    reject_symlinks(raw_file)
    reject_symlinks(acquisition_record)
    if (raw_file.resolve() != relative_path(root, RAW_PATH).resolve()
            or acquisition_record.resolve() != relative_path(root, ACQUISITION_PATH).resolve()):
        raise AmendmentError("Use the already acquired immutable snapshot and its exact record.")
    base = checked_original_module(root)
    _, amendment = ensure_technical_amendment(root, base)
    # All original helpers execute in their unchanged original module namespace.
    root = base.project_root_checked(root)
    lock, identities = base.verify_analysis_lock(root)
    plan = base.read_json(root / "protocol_plan_v1.json")
    design = plan.get("design", {})
    if design.get("assets") != list(base.ASSETS) or design.get("lookback") != base.LOOKBACK or design.get("new_test_start_inclusive") != base.START or design.get("new_test_end_inclusive") != base.END:
        raise base.PreparationError("Locked plan differs from this processor's supported design.")
    minimum_rows = design.get("minimum_new_target_rows", 150)
    if type(minimum_rows) is not int or minimum_rows < 150:
        raise base.PreparationError("The external target-row minimum must be at least 150.")
    minimum_observed = design.get("minimum_observed_price_rows_per_asset_in_new_period", 150)
    if type(minimum_observed) is not int or minimum_observed < 150:
        raise base.PreparationError("The actually observed price-row minimum must be at least 150 per asset.")
    _, frozen = base.verify_frozen_inputs(root)
    record, acquired_identities = base.verify_acquisition(raw_file, acquisition_record)
    if file_identity(root, RAW_PATH, RAW_SNAPSHOT_SHA256)["sha256"] != record["raw_file_sha256"]:
        raise AmendmentError("Corrected preparation must use the already acquired immutable snapshot and record.")
    identities.update(acquired_identities)
    identities.update({"original_training_manifest_sha256": base.ORIGINAL_MANIFEST_SHA256,
        "old_raw_sha256": frozen["data/raw/raw_prices.csv"]["sha256"],
        "original_scalers_sha256": frozen["data/sequences/baseline/scalers.pkl"]["sha256"],
        "target_period": {"start": base.START, "end_inclusive": base.END},
        "technical_amendment_sha256": amendment["sha256"]})
    output = root / "new_period/prepared_v1"
    if output.exists():
        manifest = base.verify_existing_output(output, identities)
        if manifest.get("technical_amendment") != amendment:
            raise AmendmentError("Existing prepared output has another technical amendment.")
        verify_technical_amendment(root)
        return manifest, True
    old = base.read_prices(root / "inputs/data/raw/raw_prices.csv", "frozen old prices",
                          frozen["data/raw/raw_prices.csv"], allow_all_missing_rows=True)
    acquired = base.read_prices(raw_file, "acquired prices", {"sha256": record["raw_file_sha256"]})
    combined, overlap = base.combine_without_revisions(old, acquired,
        lock.get("overlap_max_relative_error_tolerance"))
    bundle, before = base.load_original_scalers(root / "inputs/data/sequences/baseline/scalers.pkl",
        frozen["data/sequences/baseline/scalers.pkl"])
    reproduction = base.historical_reproduction(old, bundle, root / "inputs")
    tables = base.baseline_tables(combined)
    arrays = base.select_external_sequences(tables, bundle)
    observed_per_asset = base.enforce_coverage(arrays, minimum_rows, minimum_observed)
    if base.scaler_snapshot(bundle) != before:
        raise base.PreparationError("Frozen scaler state changed during transform-only preparation.")
    # The sole numerical-path code change: convert NumPy scalar text to Python text.
    first_anchor = combined.index.get_loc(base.pd.Timestamp(str(arrays["anchor_dates_external.npy"][0])))
    earliest_price = combined.index[first_anchor - (base.LOOKBACK - 1) - 20]
    if earliest_price <= base.pd.Timestamp(base.OLD_END):
        raise base.PreparationError("Selected external windows still touch the historical price seam.")
    target_dates = base.pd.DatetimeIndex(arrays["target_realization_dates_external.npy"])
    details = {"assessment_classification": "Pre-specified external-period assessment within audited exposure scope; no absolute pristine-status claim.",
        "period": {"requested_start": base.START, "requested_end_inclusive": base.END,
            "n_sequences": len(target_dates), "first_target": target_dates.min().strftime("%Y-%m-%d"),
            "minimum_target_rows_required": minimum_rows,
            "minimum_observed_price_rows_per_asset_required": minimum_observed,
            "coverage_gate_scope": "Data coverage requirement; not a statistical power guarantee.",
            "last_target": target_dates.max().strftime("%Y-%m-%d"),
            "first_anchor": arrays["anchor_dates_external.npy"][0], "last_anchor": arrays["anchor_dates_external.npy"][-1],
            "earliest_raw_price_for_selected_windows": earliest_price.strftime("%Y-%m-%d")},
        "features": {"columns": base.FEATURE_COLUMNS, "lookback": base.LOOKBACK,
            "return": "log(P[t]/P[t-1])", "volatility": "rolling 20 log returns, ddof=1, multiplied by sqrt(252)"},
        "targets": {"columns": base.TARGET_COLUMNS, "alignment": "anchor t -> next union-calendar date t+1",
            "date_array_format": "YYYY-MM-DD", "raw_and_scaled_array_dtype": "float32"},
        "calendar": {"rule": "Observed four-asset union calendar; forward fill only; no backfill or resampling.",
            "missing_flags_asset_order": list(base.ASSETS),
            "raw_context_flag_window": "31 price rows: 30 rows before each target plus its realization row; includes Vol20 warmup dependencies.",
            "target_observed_counts": dict(zip(base.ASSETS, observed_per_asset.astype(int).tolist())),
            "target_missing_counts": dict(zip(base.ASSETS, arrays["target_missing_flags_external.npy"].sum(axis=0).astype(int).tolist())),
            "target_forward_fill_counts": dict(zip(base.ASSETS, arrays["target_ffill_flags_external.npy"].sum(axis=0).astype(int).tolist()))},
        "price_basis_overlap_audit": overlap, "historical_reproduction": reproduction,
        "scalers": {"original_file_sha256": identities["original_scalers_sha256"], "state": before,
            "refitted": False, "state_unchanged_after_transform": True},
        "acquisition_contract": {k: record[k] for k in ("provider", "price_field", "adjusted", "tickers", "download_parameters", "data_start", "data_end_inclusive")},
        "technical_amendment": amendment,
        "old_files_modified": False, "old_test_arrays_loaded": False,
        "model_checkpoints_deserialized": False, "models_evaluated": False, "predictions_produced": False}
    verify_technical_amendment(root)
    manifest, reused = base.commit_outputs(output, identities, arrays, details)
    for relative, identity in frozen.items():
        base.verify_identity(base.checked_relative(root / "inputs", relative), identity)
    verify_technical_amendment(root)
    return manifest, reused


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--raw-file", type=Path, required=True)
    parser.add_argument("--acquisition-record", type=Path, required=True)
    args = parser.parse_args()
    result, reused = prepare(args.project_root, args.raw_file, args.acquisition_record)
    print(json.dumps({"status": result["status"], "reused_identical_preparation": reused,
        "output": str(args.project_root / "new_period/prepared_v1"), "n_sequences": result["period"]["n_sequences"],
        "technical_amendment": result["technical_amendment"],
        "models_evaluated": False, "predictions_produced": False}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
