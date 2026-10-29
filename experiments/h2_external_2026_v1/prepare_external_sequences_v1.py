#!/usr/bin/env python3
"""Prepare frozen-model external inputs; acquire no data and evaluate no models.

The CLI refuses to parse any new prices until the analysis, implementations and
six independently trained STL checkpoints have been frozen. All old inputs are
read-only. The pure table/sequence functions and --self-test require no real data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import pickle
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ASSETS = ("BIST100", "USDTRY", "EURTRY", "GOLD")
TICKERS = dict(zip(ASSETS, ("XU100.IS", "USDTRY=X", "EURTRY=X", "GC=F")))
FEATURE_COLUMNS = [f"{a}_{f}" for a in ASSETS for f in ("LogRet", "Vol20")]
TARGET_COLUMNS = [f"{a}_NextRet" for a in ASSETS] + [f"{a}_NextVol" for a in ASSETS]
LOOKBACK = 10
START = "2026-01-01"
END = "2026-10-28"
OLD_END = "2024-12-31"
DOWNLOAD_START = "2024-11-01"
ORIGINAL_MANIFEST_SHA256 = "517c06f3b1a5d71de9739f56ef6a2af7cf9a918ed2cf303a52d30bfc7b9f393a"
OLD_FROZEN_SHA256 = "12cf2481978546689f9087d7d97cb4d8325756a15f393a9c936a31e2a47876f4"
OLD_RUN_ROOT = Path("/content/drive/MyDrive/tez_transformer_h2_t4_v1/runs/t4_full_run_v1")
REQUIRED_IMPLEMENTATIONS = frozenset({"run_stl_controls_v1.py", "prepare_external_sequences_v1.py",
    "acquire_external_prices_v1.py", "evaluate_external_h2_v1.py", "lock_external_analysis_v1.py"})
DEFAULT_ROOT = "/content/drive/MyDrive/tez_transformer_h2_external_2026_v1"
STL_MANIFEST_PATH = "runs/stl_run_v1/selected_stl_checkpoints_frozen.json"
PROTECTED_ROOTS = tuple(Path(p) for p in (
    "/content/drive/MyDrive/tez_transformer_v4_repro",
    "/content/drive/MyDrive/tez_transformer_h2_v1",
    "/content/drive/MyDrive/tez_transformer_h2_t4_v1"))


class PreparationError(RuntimeError):
    """An identity, chronology or preparation gate failed."""


def reject_symlink_components(path: Path) -> None:
    path = Path(path).absolute()
    if any(component.is_symlink() for component in (path, *path.parents)):
        raise PreparationError(f"Symlink paths are not allowed: {path}")


def project_root_checked(path: Path) -> Path:
    reject_symlink_components(path)
    root = Path(path).resolve()
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in PROTECTED_ROOTS):
        raise PreparationError("The output project collides with a protected historical root.")
    return root


def sha256_file(path: Path) -> str:
    reject_symlink_components(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    reject_symlink_components(path)
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise PreparationError(f"Expected JSON object: {path}")
    return value


def checked_relative(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise PreparationError("An artifact path must be project-relative.")
    reject_symlink_components(root / relative)
    candidate = (root / relative).resolve()
    if candidate == root.resolve() or root.resolve() not in candidate.parents:
        raise PreparationError("Artifact path escapes its project root.")
    return candidate


def verify_identity(path: Path, record: dict) -> None:
    if not path.is_file():
        raise PreparationError(f"Missing required file: {path}")
    expected = record.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64 or sha256_file(path) != expected:
        raise PreparationError(f"SHA-256 mismatch: {path}")
    if "size_bytes" in record and path.stat().st_size != record["size_bytes"]:
        raise PreparationError(f"Size mismatch: {path}")


def verify_analysis_lock(root: Path) -> tuple[dict, dict]:
    """No new-price path is opened here, including for hashing."""
    lock_path = root / "analysis_locked_v1.json"
    if not lock_path.is_file():
        raise PreparationError("Analysis is not locked; new prices will not be opened.")
    lock = read_json(lock_path)
    if lock.get("status") != "ANALYSIS_LOCKED_BEFORE_NEW_DATA":
        raise PreparationError("Analysis lock has not authorized new data access.")
    plan = root / "protocol_plan_v1.json"
    verify_identity(plan, {"sha256": lock.get("protocol_plan_sha256")})
    implementations = lock.get("implementation_sha256")
    if not isinstance(implementations, dict) or not REQUIRED_IMPLEMENTATIONS.issubset(implementations):
        raise PreparationError("The analysis lock must freeze all five required analysis implementations.")
    for name, expected in implementations.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise PreparationError("Implementation identities must use basenames.")
        verify_identity(checked_relative(root, f"implementation/{name}"), {"sha256": expected})
    verify_identity(Path(__file__).resolve(), {"sha256": implementations[Path(__file__).name]})
    old_record = lock.get("existing_h2_checkpoints_manifest")
    if not isinstance(old_record, dict) or old_record.get("path") != "source_records/selected_checkpoints_frozen.json" or old_record.get("sha256") != OLD_FROZEN_SHA256:
        raise PreparationError("The nine existing H2 checkpoints are not identified by their original frozen manifest.")
    old_path = checked_relative(root, old_record["path"])
    verify_identity(old_path, old_record)
    old_frozen = read_json(old_path)
    training_path = checked_relative(root, "source_records/locked_training_manifest.json")
    verify_identity(training_path, {"sha256": ORIGINAL_MANIFEST_SHA256})
    old_training = read_json(training_path)
    if old_training.get("run_root") != str(OLD_RUN_ROOT):
        raise PreparationError("The trusted original H2 checkpoint root differs.")
    if old_frozen.get("status") != "ALL_NINE_CHECKPOINTS_FROZEN_NO_TEST_ACCESS" or old_frozen.get("locked_training_manifest_sha256") != ORIGINAL_MANIFEST_SHA256:
        raise PreparationError("The original H2 checkpoint provenance differs.")
    completed = old_frozen.get("completed_runs")
    expected_old = {(a,s) for a in ("NS_base","FS_same_width","FS_budget_control") for s in (123,777,2026)}
    if not isinstance(completed,list) or len(completed)!=9 or {(j.get("arm",{}).get("id"),j.get("seed")) for j in completed} != expected_old:
        raise PreparationError("Exactly nine unique existing H2 arm/seed checkpoints are required.")
    old_checkpoint_identities = []
    for job in completed:
        checkpoint = job.get("best_checkpoint")
        if not isinstance(checkpoint,dict):
            raise PreparationError("An existing H2 selected checkpoint is missing.")
        verify_identity(checked_relative(OLD_RUN_ROOT, checkpoint.get("path")), checkpoint)
        old_checkpoint_identities.append({"arm":job["arm"]["id"],"seed":job["seed"],**checkpoint})
    stl_record = lock.get("selected_stl_checkpoints_manifest")
    if not isinstance(stl_record, dict) or stl_record.get("path") != STL_MANIFEST_PATH:
        raise PreparationError("The required frozen STL manifest is not identified.")
    stl_path = checked_relative(root, stl_record["path"])
    verify_identity(stl_path, stl_record)
    stl = read_json(stl_path)
    if stl.get("status") != "STL_CHECKPOINTS_FROZEN_BEFORE_NEW_DATA":
        raise PreparationError("STL checkpoints are not frozen before new data.")
    if stl.get("plan_sha256") != lock["protocol_plan_sha256"] or stl.get("training_implementation_sha256") != implementations["run_stl_controls_v1.py"]:
        raise PreparationError("The STL checkpoints were frozen under another plan or training implementation.")
    checkpoints = stl.get("checkpoints")
    expected_jobs = {(t, s) for t in ("return", "volatility") for s in (123, 777, 2026)}
    if not isinstance(checkpoints, list) or len(checkpoints) != 6:
        raise PreparationError("Exactly six frozen STL checkpoints are required.")
    if {(c.get("task"), c.get("seed")) for c in checkpoints} != expected_jobs:
        raise PreparationError("Frozen STL task/seed identities differ.")
    for checkpoint in checkpoints:
        verify_identity(checked_relative(root, checkpoint.get("path")), checkpoint)
    return lock, {"analysis_lock_sha256": sha256_file(lock_path),
                  "protocol_plan_sha256": lock["protocol_plan_sha256"],
                  "implementation_sha256": implementations,
                  "existing_h2_checkpoints_manifest_sha256": OLD_FROZEN_SHA256,
                  "existing_h2_selected_checkpoints": old_checkpoint_identities,
                  "selected_stl_checkpoints_manifest_sha256": sha256_file(stl_path)}


def verify_frozen_inputs(root: Path) -> tuple[dict, dict]:
    manifest_path = root / "source_records/locked_training_manifest.json"
    verify_identity(manifest_path, {"sha256": ORIGINAL_MANIFEST_SHA256})
    manifest = read_json(manifest_path)
    identities = manifest.get("input_identities")
    if manifest.get("status") != "LOCKED_BEFORE_TRAINING" or not isinstance(identities, dict) or len(identities) != 21:
        raise PreparationError("Unexpected original training manifest structure.")
    inputs = root / "inputs"
    # Old test arrays are hashed as opaque bytes and are never np.load'ed.
    for relative, identity in identities.items():
        verify_identity(checked_relative(inputs, relative), identity)
    return manifest, identities


def verify_acquisition(raw_file: Path, record_file: Path) -> tuple[dict, dict]:
    record = read_json(record_file)
    expected = {"status": "ACQUISITION_COMPLETE", "provider": "Yahoo Finance",
                "price_field": "Close", "adjusted": False,
                "data_start": DOWNLOAD_START, "data_end_inclusive": END}
    for key, value in expected.items():
        if record.get(key) != value or (key == "adjusted" and record.get(key) is not False):
            raise PreparationError(f"Acquisition contract differs: {key}")
    if record.get("assets") != list(ASSETS) or record.get("tickers") != TICKERS:
        raise PreparationError("Acquired assets/tickers differ from the frozen series.")
    parameters = record.get("download_parameters")
    if not isinstance(parameters, dict) or any(parameters.get(k) is not False for k in ("auto_adjust", "back_adjust", "repair")):
        raise PreparationError("Yahoo Close adjustment/repair parameters are not frozen.")
    if parameters.get("interval") != "1d":
        raise PreparationError("Only the pre-specified daily download interval is allowed.")
    verify_identity(raw_file, {"sha256": record.get("raw_file_sha256")})
    return record, {"acquisition_record_sha256": sha256_file(record_file),
                    "new_raw_sha256": record["raw_file_sha256"]}


def validate_prices(frame: pd.DataFrame, label: str, *, allow_all_missing_rows: bool = False) -> pd.DataFrame:
    if list(frame.columns) != list(ASSETS):
        raise PreparationError(f"{label}: exact four-column asset order required.")
    index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="raise"))
    if index.tz is not None or index.hasnans or not index.equals(index.normalize()):
        raise PreparationError(f"{label}: date-only, timezone-naive daily labels required.")
    if not index.is_monotonic_increasing or index.has_duplicates or len(index) == 0:
        raise PreparationError(f"{label}: nonempty, sorted, unique daily dates required.")
    out = frame.copy()
    out.index = index
    for col in ASSETS:
        out[col] = pd.to_numeric(out[col], errors="raise").astype(np.float64)
        observed = out[col].dropna().to_numpy()
        if not len(observed) or not np.isfinite(observed).all() or (observed <= 0).any():
            raise PreparationError(f"{label}: prices must be observed, finite and positive: {col}")
    if not allow_all_missing_rows and out.isna().all(axis=1).any():
        raise PreparationError(f"{label}: a union-calendar row cannot have no observed assets.")
    return out


def verified_bytes(path: Path, identity: dict) -> bytes:
    reject_symlink_components(path)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != identity.get("sha256"):
        raise PreparationError(f"SHA-256 mismatch in bytes being parsed: {path}")
    if "size_bytes" in identity and len(payload) != identity["size_bytes"]:
        raise PreparationError(f"Size mismatch in bytes being parsed: {path}")
    return payload


def read_prices(path: Path, label: str, identity: dict, *, allow_all_missing_rows: bool = False) -> pd.DataFrame:
    # Parse exactly the checked payload, rather than reopening a mutable path.
    return validate_prices(pd.read_csv(BytesIO(verified_bytes(path, identity)), index_col=0),
                           label, allow_all_missing_rows=allow_all_missing_rows)


def combine_without_revisions(old: pd.DataFrame, acquired: pd.DataFrame,
                              tolerance: float | None = None) -> tuple[pd.DataFrame, dict]:
    if old.index.max() > pd.Timestamp(OLD_END):
        raise PreparationError("Frozen raw history extends beyond its declared boundary.")
    if acquired.index.min() < pd.Timestamp(DOWNLOAD_START) or acquired.index.max() > pd.Timestamp(END):
        raise PreparationError("Acquisition dates exceed the fixed download bounds.")
    overlap = old.index.intersection(acquired.index)
    audit: dict[str, Any] = {"old_values_replaced": False, "assets": {}}
    if tolerance is not None and (not np.isfinite(tolerance) or tolerance < 0):
        raise PreparationError("Invalid pre-locked overlap error tolerance.")
    for asset in ASSETS:
        a, b = old.loc[overlap, asset], acquired.loc[overlap, asset]
        valid = a.notna() & b.notna()
        if int(valid.sum()) < 20:
            raise PreparationError(f"Insufficient observed overlap to audit price basis: {asset}")
        absolute = (a[valid] - b[valid]).abs()
        relative = absolute / a[valid].abs()
        maximum = float(relative.max())
        audit["assets"][asset] = {"n_common_observed": int(valid.sum()),
            "max_absolute_error": float(absolute.max()), "max_relative_error": maximum,
            "n_different_values": int((absolute > 0).sum())}
        if tolerance is not None and maximum > tolerance:
            raise PreparationError(f"Price-basis overlap quality gate failed: {asset}")
    audit["pre_locked_max_relative_error_tolerance"] = tolerance
    audit["gate"] = "PASS" if tolerance is not None else "RECORDED_ONLY_NO_NUMERIC_TOLERANCE"
    newer = acquired.loc[acquired.index > pd.Timestamp(OLD_END)]
    if newer.empty:
        raise PreparationError("No acquired extension rows beyond the frozen history.")
    combined = pd.concat([old, newer], axis=0)
    if not combined.loc[old.index].equals(old):
        raise PreparationError("Frozen historical values changed at concatenation.")
    return validate_prices(combined, "combined prices", allow_all_missing_rows=True), audit


def baseline_tables(raw: pd.DataFrame) -> dict:
    """Exact baseline arithmetic from archived rebuild; no fitting or lookahead."""
    missing = raw.isna()
    clean = raw.ffill()
    if clean.isna().any().any():
        raise PreparationError("Leading missing prices cannot be backfilled.")
    returns = np.log(clean / clean.shift(1))
    volatility = returns.rolling(window=20, min_periods=20).std(ddof=1) * np.sqrt(252)
    features = pd.DataFrame(index=raw.index)
    for asset in ASSETS:
        features[f"{asset}_LogRet"] = returns[asset]
        features[f"{asset}_Vol20"] = volatility[asset]
    targets = pd.concat([returns.shift(-1).rename(columns={a: f"{a}_NextRet" for a in ASSETS}),
                         volatility.shift(-1).rename(columns={a: f"{a}_NextVol" for a in ASSETS})], axis=1)
    realized = pd.Series(raw.index.to_series().shift(-1).values, index=raw.index,
                         name="target_realization_date")
    return {"features": features[FEATURE_COLUMNS], "targets": targets[TARGET_COLUMNS],
            "realized": realized, "missing": missing, "ffill": missing & clean.notna()}


def scaler_snapshot(bundle: dict) -> dict:
    result = {"feature_columns": bundle["feature_columns"], "target_columns": bundle["target_columns"],
              "fit_split": bundle["fit_split"], "fit_rows": int(bundle["fit_rows"])}
    for key in ("x_scaler", "y_scaler"):
        scaler = bundle[key]
        result[key] = {"class": f"{type(scaler).__module__}.{type(scaler).__name__}",
            "n_features_in": int(scaler.n_features_in_),
            "n_samples_seen": np.asarray(scaler.n_samples_seen_).tolist(),
            "mean": np.asarray(scaler.mean_).tolist(), "var": np.asarray(scaler.var_).tolist(),
            "scale": np.asarray(scaler.scale_).tolist()}
    return result


def load_original_scalers(path: Path, identity: dict) -> tuple[dict, dict]:
    # This is the only pickle deserialization, after the immutable manifest and
    # this original scaler's exact SHA/size have both been verified.
    verify_identity(path, identity)
    bundle = pickle.loads(verified_bytes(path, identity))
    if not isinstance(bundle, dict) or bundle.get("feature_columns") != FEATURE_COLUMNS or bundle.get("target_columns") != TARGET_COLUMNS:
        raise PreparationError("Original scaler column metadata differs.")
    if bundle.get("fit_split") != "train_only" or not isinstance(bundle.get("fit_rows"), int):
        raise PreparationError("Original scaler fit metadata differs.")
    for name in ("x_scaler", "y_scaler"):
        scaler = bundle.get(name)
        if type(scaler) is not StandardScaler or scaler.n_features_in_ != 8:
            raise PreparationError("Only original eight-column StandardScalers are allowed.")
        if not (np.atleast_1d(scaler.n_samples_seen_) == bundle["fit_rows"]).all():
            raise PreparationError("Scaler fit-row identity differs.")
        for attr in ("mean_", "var_", "scale_"):
            if not np.isfinite(getattr(scaler, attr)).all():
                raise PreparationError("Non-finite frozen scaler state.")
        if (scaler.scale_ <= 0).any():
            raise PreparationError("Non-positive frozen scaler scale.")
    return bundle, scaler_snapshot(bundle)


def sequences_for_anchors(tables: dict, bundle: dict, anchors: pd.DatetimeIndex) -> dict:
    features, targets = tables["features"], tables["targets"]
    positions = features.index.get_indexer(anchors)
    if (positions < LOOKBACK - 1).any():
        raise PreparationError("A selected anchor lacks its ten-row input history.")
    x_scaled = bundle["x_scaler"].transform(features)
    y_scaled = bundle["y_scaler"].transform(targets.loc[anchors])
    windows = np.stack([x_scaled[i - LOOKBACK + 1:i + 1] for i in positions])
    raw_targets = targets.loc[anchors].to_numpy()
    dates = pd.DatetimeIndex(tables["realized"].loc[anchors])
    if dates.hasnans or not (dates > anchors).all() or dates.has_duplicates:
        raise PreparationError("Target realization date alignment is invalid.")
    if not np.isfinite(windows).all() or not np.isfinite(raw_targets).all() or not np.isfinite(y_scaled).all():
        raise PreparationError("Selected inputs/targets contain invalid or insufficient history.")
    missing = tables["missing"].to_numpy(dtype=bool)
    filled = tables["ffill"].to_numpy(dtype=bool)
    target_positions = features.index.get_indexer(dates)
    return {"X_external.npy": windows.astype(np.float32),
            "y_external_raw.npy": raw_targets.astype(np.float32),
            "y_external_scaled.npy": y_scaled.astype(np.float32),
            "anchor_dates_external.npy": anchors.strftime("%Y-%m-%d").to_numpy(dtype="U10"),
            "target_realization_dates_external.npy": dates.strftime("%Y-%m-%d").to_numpy(dtype="U10"),
            "X_missing_flags_external.npy": np.stack([missing[i-LOOKBACK+1:i+1] for i in positions]),
            "X_ffill_flags_external.npy": np.stack([filled[i-LOOKBACK+1:i+1] for i in positions]),
            "raw_price_context_missing_flags_external.npy": np.stack([missing[i-LOOKBACK-19:i+2] for i in positions]),
            "raw_price_context_ffill_flags_external.npy": np.stack([filled[i-LOOKBACK-19:i+2] for i in positions]),
            "target_missing_flags_external.npy": missing[target_positions],
            "target_ffill_flags_external.npy": filled[target_positions]}


def select_external_sequences(tables: dict, bundle: dict, start: str = START,
                              end: str = END) -> dict:
    dates = tables["realized"]
    chosen = dates.between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")
    anchors = pd.DatetimeIndex(dates.index[chosen])
    if anchors.empty:
        raise PreparationError("The pre-specified period has no realizable target rows.")
    return sequences_for_anchors(tables, bundle, anchors)


def enforce_coverage(arrays: dict, minimum_rows: int, minimum_observed: int) -> np.ndarray:
    if len(arrays["y_external_raw.npy"]) < minimum_rows:
        raise PreparationError("The external period has fewer target rows than pre-specified.")
    observed = (~arrays["target_missing_flags_external.npy"]).sum(axis=0)
    if (observed < minimum_observed).any():
        raise PreparationError("An asset has fewer actually observed new-period price rows than pre-specified.")
    return observed


def historical_reproduction(old_raw: pd.DataFrame, bundle: dict, inputs: Path) -> dict:
    """Check archived TRAIN/VALIDATION only; no historical test array is loaded."""
    tables = baseline_tables(old_raw)
    report = {}
    for split in ("train", "val"):
        directory = inputs / "data/sequences/baseline/lb10"
        anchors = pd.DatetimeIndex(pd.to_datetime(np.load(directory / f"anchor_dates_{split}.npy", allow_pickle=False)))
        expected_dates = pd.DatetimeIndex(pd.to_datetime(np.load(directory / f"target_realization_dates_{split}.npy", allow_pickle=False)))
        actual = sequences_for_anchors(tables, bundle, anchors)
        if not pd.DatetimeIndex(actual["target_realization_dates_external.npy"]).equals(expected_dates):
            raise PreparationError(f"Historical {split} target-date reproduction failed.")
        differences = {}
        for actual_key, archived_name in (("X_external.npy", f"X_{split}.npy"),
                                          ("y_external_raw.npy", f"y_{split}_raw.npy"),
                                          ("y_external_scaled.npy", f"y_{split}.npy")):
            expected = np.load(directory / archived_name, allow_pickle=False)
            got = actual[actual_key]
            if got.shape != expected.shape or not np.allclose(got, expected, rtol=1e-5, atol=1e-8):
                raise PreparationError(f"Historical {split} reproduction failed: {archived_name}")
            differences[archived_name] = float(np.max(np.abs(got - expected)))
        report[split] = {"n_sequences": len(anchors), "max_absolute_differences": differences}
    return {"status": "ORIGINAL_TRAIN_VALIDATION_BASELINE_REPRODUCED", "splits": report,
            "old_test_arrays_loaded": False, "rtol": 1e-5, "atol": 1e-8}


def verify_existing_output(output: Path, identities: dict) -> dict:
    reject_symlink_components(output)
    manifest = read_json(output / "prepared_manifest_v1.json")
    if manifest.get("status") != "EXTERNAL_SEQUENCES_PREPARED_NO_MODEL_EVALUATION" or manifest.get("input_identities") != identities:
        raise PreparationError("Existing preparation has different identities; it will not be overwritten.")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise PreparationError("Existing preparation has no verifiable artifact manifest.")
    for relative, identity in artifacts.items():
        verify_identity(checked_relative(output, relative), identity)
    if {p.name for p in output.iterdir()} != set(artifacts) | {"prepared_manifest_v1.json"}:
        raise PreparationError("Existing preparation contains unmanifested files.")
    return manifest


def commit_outputs(output: Path, identities: dict, arrays: dict, details: dict) -> tuple[dict, bool]:
    reject_symlink_components(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(output)
    guard = output.parent / ".prepare_external_sequences_v1.lock"
    try:
        fd = os.open(guard, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise PreparationError("Another preparation, or an interrupted preparation lock, exists.") from exc
    os.close(fd)
    stage: Path | None = None
    try:
        if output.exists():
            return verify_existing_output(output, identities), True
        stage = Path(tempfile.mkdtemp(prefix=".prepared_v1-stage-", dir=output.parent))
        artifacts = {}
        for name, array in arrays.items():
            if Path(name).name != name or not name.endswith(".npy") or array.dtype.hasobject:
                raise PreparationError("Only named, non-object NumPy output arrays are permitted.")
            path = stage / name
            np.save(path, array, allow_pickle=False)
            artifacts[name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size,
                               "shape": list(array.shape), "dtype": str(array.dtype)}
        manifest = {"status": "EXTERNAL_SEQUENCES_PREPARED_NO_MODEL_EVALUATION",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "input_identities": identities,
            "artifacts": artifacts, **details}
        (stage / "prepared_manifest_v1.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            raise PreparationError("Output appeared concurrently; refusing to replace it.")
        os.rename(stage, output)
        stage = None
        return manifest, False
    finally:
        if stage is not None:
            shutil.rmtree(stage)
        guard.unlink()


def prepare(root: Path, raw_file: Path, acquisition_record: Path) -> tuple[dict, bool]:
    root = project_root_checked(root)
    # Strict order: no byte of new prices is read until the full analysis gate.
    lock, identities = verify_analysis_lock(root)
    plan = read_json(root / "protocol_plan_v1.json")
    design = plan.get("design", {})
    if design.get("assets") != list(ASSETS) or design.get("lookback") != LOOKBACK or design.get("new_test_start_inclusive") != START or design.get("new_test_end_inclusive") != END:
        raise PreparationError("Locked plan differs from this processor's supported design.")
    minimum_rows = design.get("minimum_new_target_rows", 150)
    if type(minimum_rows) is not int or minimum_rows < 150:
        raise PreparationError("The external target-row minimum must be at least 150.")
    minimum_observed = design.get("minimum_observed_price_rows_per_asset_in_new_period", 150)
    if type(minimum_observed) is not int or minimum_observed < 150:
        raise PreparationError("The actually observed price-row minimum must be at least 150 per asset.")
    _, frozen = verify_frozen_inputs(root)
    record, acquired_identities = verify_acquisition(raw_file, acquisition_record)
    identities.update(acquired_identities)
    identities.update({"original_training_manifest_sha256": ORIGINAL_MANIFEST_SHA256,
        "old_raw_sha256": frozen["data/raw/raw_prices.csv"]["sha256"],
        "original_scalers_sha256": frozen["data/sequences/baseline/scalers.pkl"]["sha256"],
        "target_period": {"start": START, "end_inclusive": END}})
    output = root / "new_period/prepared_v1"
    if output.exists():
        return verify_existing_output(output, identities), True
    old = read_prices(root / "inputs/data/raw/raw_prices.csv", "frozen old prices",
                      frozen["data/raw/raw_prices.csv"], allow_all_missing_rows=True)
    acquired = read_prices(raw_file, "acquired prices", {"sha256": record["raw_file_sha256"]})
    combined, overlap = combine_without_revisions(old, acquired,
        lock.get("overlap_max_relative_error_tolerance"))
    bundle, before = load_original_scalers(root / "inputs/data/sequences/baseline/scalers.pkl",
        frozen["data/sequences/baseline/scalers.pkl"])
    reproduction = historical_reproduction(old, bundle, root / "inputs")
    tables = baseline_tables(combined)
    arrays = select_external_sequences(tables, bundle)
    observed_per_asset = enforce_coverage(arrays, minimum_rows, minimum_observed)
    if scaler_snapshot(bundle) != before:
        raise PreparationError("Frozen scaler state changed during transform-only preparation.")
    # Every raw observation affecting a 2026 input window is strictly newer
    # than the old/new seam, excluding mixed-basis seam effects from assessment.
    first_anchor = combined.index.get_loc(pd.Timestamp(arrays["anchor_dates_external.npy"][0]))
    earliest_price = combined.index[first_anchor - (LOOKBACK - 1) - 20]
    if earliest_price <= pd.Timestamp(OLD_END):
        raise PreparationError("Selected external windows still touch the historical price seam.")
    target_dates = pd.DatetimeIndex(arrays["target_realization_dates_external.npy"])
    details = {"assessment_classification": "Pre-specified external-period assessment within audited exposure scope; no absolute pristine-status claim.",
        "period": {"requested_start": START, "requested_end_inclusive": END,
            "n_sequences": len(target_dates), "first_target": target_dates.min().strftime("%Y-%m-%d"),
            "minimum_target_rows_required": minimum_rows,
            "minimum_observed_price_rows_per_asset_required": minimum_observed,
            "coverage_gate_scope": "Data coverage requirement; not a statistical power guarantee.",
            "last_target": target_dates.max().strftime("%Y-%m-%d"),
            "first_anchor": arrays["anchor_dates_external.npy"][0], "last_anchor": arrays["anchor_dates_external.npy"][-1],
            "earliest_raw_price_for_selected_windows": earliest_price.strftime("%Y-%m-%d")},
        "features": {"columns": FEATURE_COLUMNS, "lookback": LOOKBACK,
            "return": "log(P[t]/P[t-1])", "volatility": "rolling 20 log returns, ddof=1, multiplied by sqrt(252)"},
        "targets": {"columns": TARGET_COLUMNS, "alignment": "anchor t -> next union-calendar date t+1",
            "date_array_format": "YYYY-MM-DD", "raw_and_scaled_array_dtype": "float32"},
        "calendar": {"rule": "Observed four-asset union calendar; forward fill only; no backfill or resampling.",
            "missing_flags_asset_order": list(ASSETS),
            "raw_context_flag_window": "31 price rows: 30 rows before each target plus its realization row; includes Vol20 warmup dependencies.",
            "target_observed_counts": dict(zip(ASSETS, observed_per_asset.astype(int).tolist())),
            "target_missing_counts": dict(zip(ASSETS, arrays["target_missing_flags_external.npy"].sum(axis=0).astype(int).tolist())),
            "target_forward_fill_counts": dict(zip(ASSETS, arrays["target_ffill_flags_external.npy"].sum(axis=0).astype(int).tolist()))},
        "price_basis_overlap_audit": overlap, "historical_reproduction": reproduction,
        "scalers": {"original_file_sha256": identities["original_scalers_sha256"], "state": before,
            "refitted": False, "state_unchanged_after_transform": True},
        "acquisition_contract": {k: record[k] for k in ("provider", "price_field", "adjusted", "tickers", "download_parameters", "data_start", "data_end_inclusive")},
        "old_files_modified": False, "old_test_arrays_loaded": False,
        "model_checkpoints_deserialized": False, "models_evaluated": False, "predictions_produced": False}
    manifest, reused = commit_outputs(output, identities, arrays, details)
    # Read-only original inputs retain exact identities after preparation.
    for relative, identity in frozen.items():
        verify_identity(checked_relative(root / "inputs", relative), identity)
    return manifest, reused


def synthetic_self_test() -> dict:
    """Functional checks use only generated prices and disposable local files."""
    from unittest.mock import patch
    dates = pd.bdate_range("2025-09-01", periods=130)
    steps = np.arange(len(dates), dtype=float)
    raw = pd.DataFrame({a: (100 + j * 10) * np.exp((0.001+j*0.0002)*steps + 0.005*np.sin(steps/(2+j))) for j,a in enumerate(ASSETS)}, index=dates)
    raw.iloc[45, 1] = np.nan
    raw = validate_prices(raw, "synthetic")
    tables = baseline_tables(raw)
    clean = raw.ffill()
    i = 60
    expected_returns = np.log(clean.iloc[i] / clean.iloc[i-1])
    expected_vol = np.log(clean / clean.shift(1)).iloc[i-19:i+1].std(ddof=1) * np.sqrt(252)
    np.testing.assert_allclose(tables["features"].iloc[i, ::2], expected_returns, rtol=1e-12)
    np.testing.assert_allclose(tables["features"].iloc[i, 1::2], expected_vol, rtol=1e-12)
    assert tables["features"].loc[dates[45], "USDTRY_LogRet"] == 0
    assert tables["missing"].loc[dates[45], "USDTRY"] and tables["ffill"].loc[dates[45], "USDTRY"]
    xs = StandardScaler().fit(tables["features"].iloc[20:70])
    ys = StandardScaler().fit(tables["targets"].iloc[20:70])
    bundle = {"x_scaler": xs, "y_scaler": ys, "feature_columns": FEATURE_COLUMNS,
              "target_columns": TARGET_COLUMNS, "fit_split": "train_only", "fit_rows": 50}
    state = scaler_snapshot(bundle)
    selected = select_external_sequences(tables, bundle, str(dates[80].date()), str(dates[90].date()))
    assert selected["X_external.npy"].shape == (11,10,8)
    assert selected["raw_price_context_missing_flags_external.npy"].shape == (11,31,4)
    assert selected["anchor_dates_external.npy"][0] == str(dates[79].date())
    assert selected["target_realization_dates_external.npy"][-1] == str(dates[90].date())
    np.testing.assert_allclose(selected["y_external_raw.npy"][0,:4], np.log(clean.iloc[80]/clean.iloc[79]), rtol=1e-6)
    np.testing.assert_allclose(selected["y_external_raw.npy"][0,4:], np.log(clean/clean.shift(1)).iloc[61:81].std(ddof=1)*np.sqrt(252), rtol=1e-6)
    assert scaler_snapshot(bundle) == state
    np.testing.assert_array_equal(enforce_coverage(selected,11,11),np.full(4,11))
    # Price changes beyond an anchor cannot change its feature window.
    future_changed = raw.copy(); future_changed.iloc[91:] *= 7
    later = select_external_sequences(baseline_tables(future_changed), bundle, str(dates[80].date()), str(dates[90].date()))
    for key in selected: np.testing.assert_array_equal(selected[key], later[key])
    # A minimum of 30 preceding raw rows produces the first L10 target window.
    minimal = baseline_tables(raw.iloc[50:81])
    minimal_array = select_external_sequences(minimal,bundle,str(dates[80].date()),str(dates[80].date()))
    np.testing.assert_allclose(minimal_array["X_external.npy"], selected["X_external.npy"][:1], rtol=1e-6, atol=1e-7)
    rejected = 0
    try: enforce_coverage(selected,12,1)
    except PreparationError: rejected += 1
    sparse = dict(selected)
    sparse["target_missing_flags_external.npy"] = selected["target_missing_flags_external.npy"].copy()
    sparse["target_missing_flags_external.npy"][:,0] = True
    try: enforce_coverage(sparse,11,6)
    except PreparationError: rejected += 1
    with tempfile.TemporaryDirectory(prefix="h2_external_prepare_test_") as td:
        temp = Path(td)
        with patch.object(pd, "read_csv", side_effect=AssertionError("New CSV opened before lock")):
            try: prepare(temp, temp/"new.csv", temp/"record.json")
            except PreparationError: rejected += 1
        # Exercise the complete lock schema with opaque synthetic checkpoint
        # bytes. Hash constants are patched only inside this disposable test.
        def fixture_json(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value,sort_keys=True),encoding="utf-8")
            return sha256_file(path)
        plan_sha = fixture_json(temp/"protocol_plan_v1.json", {"synthetic":True})
        implementation_hashes = {}
        for name in sorted(REQUIRED_IMPLEMENTATIONS):
            path=temp/"implementation"/name;path.parent.mkdir(exist_ok=True)
            path.write_bytes(Path(__file__).read_bytes() if name==Path(__file__).name else b"# synthetic non-executed implementation\n")
            implementation_hashes[name]=sha256_file(path)
        synthetic_old_root=temp/"old_read_only_run"
        old_training_sha=fixture_json(temp/"source_records/locked_training_manifest.json",{"run_root":str(synthetic_old_root)})
        old_jobs=[]
        for arm in ("NS_base","FS_same_width","FS_budget_control"):
            for seed in (123,777,2026):
                relative=f"runs/{arm}_seed{seed}/best_checkpoint.pt"
                path=synthetic_old_root/relative;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(f"opaque synthetic {arm} {seed}".encode())
                old_jobs.append({"arm":{"id":arm},"seed":seed,"best_checkpoint":{"path":relative,"sha256":sha256_file(path)}})
        old_frozen_sha=fixture_json(temp/"source_records/selected_checkpoints_frozen.json",{
            "status":"ALL_NINE_CHECKPOINTS_FROZEN_NO_TEST_ACCESS",
            "locked_training_manifest_sha256":old_training_sha,"completed_runs":old_jobs})
        stl_jobs=[]
        for task in ("return","volatility"):
            for seed in (123,777,2026):
                relative=f"runs/stl_run_v1/{task}_seed{seed}.pt"
                path=temp/relative;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(f"opaque synthetic {task} {seed}".encode())
                stl_jobs.append({"task":task,"seed":seed,"path":relative,"sha256":sha256_file(path)})
        stl_sha=fixture_json(temp/STL_MANIFEST_PATH,{
            "status":"STL_CHECKPOINTS_FROZEN_BEFORE_NEW_DATA","plan_sha256":plan_sha,
            "training_implementation_sha256":implementation_hashes["run_stl_controls_v1.py"],"checkpoints":stl_jobs})
        lock={"status":"ANALYSIS_LOCKED_BEFORE_NEW_DATA","protocol_plan_sha256":plan_sha,
            "implementation_sha256":implementation_hashes,
            "existing_h2_checkpoints_manifest":{"path":"source_records/selected_checkpoints_frozen.json","sha256":old_frozen_sha},
            "selected_stl_checkpoints_manifest":{"path":STL_MANIFEST_PATH,"sha256":stl_sha}}
        incomplete={**lock,"implementation_sha256":{Path(__file__).name:implementation_hashes[Path(__file__).name]}}
        fixture_json(temp/"analysis_locked_v1.json",incomplete)
        try: verify_analysis_lock(temp)
        except PreparationError: rejected += 1
        fixture_json(temp/"analysis_locked_v1.json",lock)
        with patch.dict(globals(),{"ORIGINAL_MANIFEST_SHA256":old_training_sha,
                                  "OLD_FROZEN_SHA256":old_frozen_sha,"OLD_RUN_ROOT":synthetic_old_root}):
            _,checked=verify_analysis_lock(temp)
            assert len(checked["existing_h2_selected_checkpoints"])==9
            (synthetic_old_root/old_jobs[0]["best_checkpoint"]["path"]).unlink()
            try: verify_analysis_lock(temp)
            except PreparationError: rejected += 1
        output = temp/"new_period/prepared_v1"
        details = {"models_evaluated":False}
        first,reused = commit_outputs(output,{"synthetic":"same"},selected,details)
        assert not reused
        second,reused = commit_outputs(output,{"synthetic":"same"},selected,details)
        assert reused and first==second
        try: commit_outputs(output,{"synthetic":"different"},selected,details)
        except PreparationError: rejected += 1
        with (output/"X_external.npy").open("ab") as f:f.write(b"tamper")
        try: verify_existing_output(output,{"synthetic":"same"})
        except PreparationError: rejected += 1
        outside = temp / "outside"; outside.mkdir()
        unsafe = temp / "unsafe"; unsafe.symlink_to(outside, target_is_directory=True)
        try: commit_outputs(unsafe / "prepared_v1", {"synthetic":"same"}, selected, details)
        except PreparationError: rejected += 1
        assert not list(outside.iterdir())
    assert rejected == 8
    return {"status":"SYNTHETIC_EXTERNAL_PREPARATION_SELF_TEST_PASS",
        "checks":["log-return and ddof1 annualized volatility", "interleaved features and return-first targets",
            "forward-fill flags and zero holiday return", "L10 realization-aware windows", "30-row raw warmup",
            "union-row and per-asset observed-price coverage gates",
            "no future-price effect on earlier windows", "frozen scaler state unchanged", "pre-lock CSV read refused",
            "complete five-implementation and fifteen-checkpoint gate", "missing old model rejected",
            "identical-output reuse", "changed-identity overwrite refused", "output tampering rejected",
            "output symlink escape rejected"],
        "negative_cases_rejected":rejected,"new_data_read":False,"network_used":False,"models_evaluated":False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument("--raw-file", type=Path)
    parser.add_argument("--acquisition-record", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.raw_file or args.acquisition_record:
            parser.error("--self-test accepts no external price/record files")
        print(json.dumps(synthetic_self_test(), ensure_ascii=False))
        return
    if not args.raw_file or not args.acquisition_record:
        parser.error("--raw-file and --acquisition-record are required")
    result,reused = prepare(args.project_root,args.raw_file,args.acquisition_record)
    print(json.dumps({"status":result["status"],"reused_identical_preparation":reused,
        "output":str(args.project_root/"new_period/prepared_v1"),"n_sequences":result["period"]["n_sequences"],
        "models_evaluated":False,"predictions_produced":False},ensure_ascii=False))


if __name__ == "__main__":
    main()
