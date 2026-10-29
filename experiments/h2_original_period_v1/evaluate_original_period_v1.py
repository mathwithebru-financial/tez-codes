#!/usr/bin/env python3
"""Evaluate the frozen sharing and independent-task models on the original test.

The test target dates are 2022-10-06 through 2024-12-31 (584 examples).
The original test and the separate 2026 aggregate outcomes have already been
examined. This is a supplementary retrospective comparison, not a new holdout.
--lock fixes this computation before its current test inference, using metadata
and opaque hashes only. --evaluate replays train/validation, then opens the
original test arrays. No fitting or price acquisition occurs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.dont_write_bytecode = True
DEFAULT_ROOT = Path("/content/drive/MyDrive/tez_transformer_h2_original_period_v1")
SOURCE_ROOT = Path("/content/drive/MyDrive/tez_transformer_h2_external_2026_v1")
OLD_RUN = Path("/content/drive/MyDrive/tez_transformer_h2_t4_v1/runs/t4_full_run_v1")
PROTECTED = tuple(Path("/content/drive/MyDrive") / p for p in
    ("tez_transformer_v4_repro", "tez_transformer_h2_v1", "tez_transformer_h2_t4_v1",
     "tez_transformer_h2_external_2026_v1"))
PLAN = "protocol_original_period_v1.json"
LOCK = "analysis_original_period_locked_v1.json"
OUTPUT = "runs/original_test_v1"
LOCK_STATUS = "ORIGINAL_PERIOD_ANALYSIS_FIXED_BEFORE_CURRENT_TEST_INFERENCE"
COMPLETE = "H2_ORIGINAL_PERIOD_EVALUATION_COMPLETE"
HELPER_PATH = "implementation/evaluate_external_h2_v1.py"
HELPER_SHA = "773f01d1c9d5cd8b535d2c8640c86b0b039ab9391d6265fb734543861d953a0f"
SOURCE_LOCK_SHA = "a155ad428996a6c1d40ca6a755212206ca9092a7b0fdc76200c3a1f063bdb274"
STL_FROZEN_SHA = "c1c546a9233f0ce7e77dffe33850b58ed5028e4d695597871f3c216d8f3c5f44"
LEGACY_TEST_PATH = "evaluation_v1/h2_losses_predictions_v1.npz"
LEGACY_TEST_SHA = "b7bcbdd52cd5c30b2d0f4e8c5cfc4859fad1a31cb366968f628f64a689448b0a"
OLD_TRAINING_SHA = "517c06f3b1a5d71de9739f56ef6a2af7cf9a918ed2cf303a52d30bfc7b9f393a"
OLD_FROZEN_SHA = "12cf2481978546689f9087d7d97cb4d8325756a15f393a9c936a31e2a47876f4"
SEQ = "data/sequences/baseline/lb10/"
TEST_FILES = ("X_test.npy", "y_test.npy", "y_test_raw.npy",
              "anchor_dates_test.npy", "target_realization_dates_test.npy")
ASSETS = ("BIST100", "USDTRY", "EURTRY", "GOLD")
SEEDS = (123, 777, 2026)
TASKS = ("return", "volatility")
ARMS = ("NS_base", "FS_same_width", "FS_budget_control", "STL_pair")
N_TEST, START, END = 584, "2022-10-06", "2024-12-31"
REL_TOL, ABS_TOL = 1e-5, 1e-8


class OriginalPeriodError(RuntimeError):
    """An original-period identity, chronology or numerical check failed."""


def no_symlinks(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise OriginalPeriodError("Symbolic-link paths are not allowed.")


def child(root, relative, *, must_exist=True):
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise OriginalPeriodError("A safe project-relative artifact path is required.")
    result = Path(root) / relative
    no_symlinks(result)
    result = result.resolve()
    if result == Path(root).resolve() or not result.is_relative_to(Path(root).resolve()):
        raise OriginalPeriodError("Artifact path escapes its designated project.")
    if must_exist and not result.is_file():
        raise OriginalPeriodError("Required artifact is missing: " + relative)
    return result


def root_checked(root):
    no_symlinks(root)
    root = Path(root).resolve()
    if not root.is_dir() or any(root.is_relative_to(p) or p.is_relative_to(root) for p in PROTECTED):
        raise OriginalPeriodError("Use the separate original-period project directory.")
    return root


def digest(path):
    no_symlinks(path)
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checked_bytes(path, identity):
    no_symlinks(path)
    data = Path(path).read_bytes()
    if (hashlib.sha256(data).hexdigest() != identity.get("sha256")
            or "size_bytes" in identity and len(data) != identity["size_bytes"]):
        raise OriginalPeriodError("Artifact identity differs: " + str(path))
    return data


def read_json(path):
    no_symlinks(path)
    value = json.loads(Path(path).read_bytes())
    if not isinstance(value, dict):
        raise OriginalPeriodError("A JSON object is required.")
    return value


def identity(root, relative):
    path = child(root, relative)
    return {"path": relative, "sha256": digest(path), "size_bytes": path.stat().st_size}


def import_helper(source=SOURCE_ROOT):
    path = child(source, HELPER_PATH)
    data = checked_bytes(path, {"sha256": HELPER_SHA})
    module = types.ModuleType("h2_verified_original_period_shared_helpers")
    module.__file__ = str(path)
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def verify_plan(root):
    plan = read_json(child(root, PLAN))
    if (plan.get("status") != "FIXED_BEFORE_ORIGINAL_PERIOD_STL_EVALUATION"
            or plan.get("project_root") != str(root)
            or plan.get("source_project_root") != str(SOURCE_ROOT)):
        raise OriginalPeriodError("The original-period scientific plan does not match this project.")
    classification = plan.get("classification", {})
    if (classification.get("label") != "supplementary_retrospective_original_test_comparison"
            or classification.get("original_test_already_examined") is not True
            or classification.get("prior_2026_aggregate_outcomes_already_examined") is not True
            or classification.get("independent_confirmation_claimed") is not False):
        raise OriginalPeriodError("The plan must state the known prior outcome exposure.")
    design = plan.get("design", {})
    expected = {"assets": list(ASSETS), "lookback": 10, "seeds": list(SEEDS),
        "n_test_sequences": N_TEST, "test_start_inclusive": START, "test_end_inclusive": END,
        "n_selected_models": 15, "n_new_fits": 0}
    if any(design.get(k) != v for k, v in expected.items()):
        raise OriginalPeriodError("The original 584-example test and frozen model design are required.")
    stats = plan.get("outcomes", {}).get("statistics", {})
    expected_stats = {"lag": 19, "h": 1, "autocovariance_denominator": "T", "two_sided": True,
                      "reference_df": 583, "holm_alpha": 0.05, "sensitivity_lag": 0}
    if any(stats.get(k) != v for k, v in expected_stats.items()):
        raise OriginalPeriodError("The fixed original-period statistical specification differs.")
    if plan.get("gates", {}).get("analysis_lock_status") != LOCK_STATUS:
        raise OriginalPeriodError("The current-inference analysis-lock status differs.")
    legacy=plan.get("legacy_original_period_reference",{})
    if (legacy.get("root")!=str(OLD_RUN) or legacy.get("path")!=LEGACY_TEST_PATH
            or legacy.get("sha256")!=LEGACY_TEST_SHA or legacy.get("size_bytes")!=561947
            or legacy.get("compatibility_relative_tolerance")!=REL_TOL
            or legacy.get("compatibility_absolute_tolerance")!=ABS_TOL):
        raise OriginalPeriodError("The archived original-test compatibility reference differs.")
    return plan


def collect_evidence(root):
    """Hash original test files as opaque bytes; open no test array or 2026 data."""
    root = root_checked(root)
    plan = verify_plan(root)
    self_path = child(root, "implementation/evaluate_original_period_v1.py")
    if Path(__file__).resolve() != self_path:
        raise OriginalPeriodError("Execute the evaluator staged in this exact project.")
    checked_bytes(child(SOURCE_ROOT, "analysis_locked_v1.json"), {"sha256": SOURCE_LOCK_SHA})
    checked_bytes(child(SOURCE_ROOT, "source_records/locked_training_manifest.json"), {"sha256": OLD_TRAINING_SHA})
    checked_bytes(child(SOURCE_ROOT, "source_records/selected_checkpoints_frozen.json"), {"sha256": OLD_FROZEN_SHA})
    checked_bytes(child(SOURCE_ROOT, "runs/stl_run_v1/selected_stl_checkpoints_frozen.json"), {"sha256": STL_FROZEN_SHA})
    declared=plan.get("source_identities",{})
    allowed_source_paths={"source_records/locked_training_manifest.json",
        "source_records/selected_checkpoints_frozen.json",
        "runs/stl_run_v1/selected_stl_checkpoints_frozen.json","analysis_locked_v1.json",
        HELPER_PATH,"implementation/run_stl_controls_v1.py","implementation/prepare_external_sequences_v1.py"}
    if not isinstance(declared,dict) or {v.get("path") for v in declared.values()}!=allowed_source_paths or len(declared)!=7:
        raise OriginalPeriodError("The source metadata/helper allowlist differs.")
    for record in declared.values():
        checked_bytes(child(SOURCE_ROOT,record["path"]),record)
    base = import_helper()
    source, source_lock, source_evidence, prep, runner = base.bootstrap(SOURCE_ROOT)
    old, stl, completed, auxiliary = base.frozen_auxiliary(source, prep, source_evidence)
    runner_evidence = runner.read_evidence(source)
    if runner_evidence.get("test_arrays_loaded") is not False:
        raise OriginalPeriodError("Source provenance verification unexpectedly loaded a test array.")
    original_inputs = read_json(child(source, "source_records/locked_training_manifest.json"))["input_identities"]
    if any(SEQ + name not in original_inputs for name in TEST_FILES):
        raise OriginalPeriodError("The original test array identities are missing.")
    for name in TEST_FILES:
        checked_bytes(child(source / "inputs", SEQ + name), original_inputs[SEQ + name])
    legacy_path = child(OLD_RUN, LEGACY_TEST_PATH)
    checked_bytes(legacy_path, {"sha256":LEGACY_TEST_SHA})
    selected = [{"scope": "original_h2_run", "arm": j["arm"]["id"], "seed": j["seed"],
                 **j["best_checkpoint"]} for j in old["completed_runs"]]
    selected += [{"scope": "source_project", "task": c["task"], "seed": c["seed"],
                  "path": c["path"], "sha256": c["sha256"], "size_bytes": c["size_bytes"]}
                 for c in stl["checkpoints"]]
    if len(selected) != 15:
        raise OriginalPeriodError("Exactly fifteen frozen checkpoint identities are required.")
    core = {"status": LOCK_STATUS, "schema_version": 1, "project_root": str(root),
        "source_project_root": str(source), "protocol_plan": identity(root, PLAN),
        "evaluation_implementation": identity(root, "implementation/evaluate_original_period_v1.py"),
        "unchanged_helper": identity(source, HELPER_PATH),
        "source_analysis_lock": identity(source, "analysis_locked_v1.json"),
        "source_original_training_manifest_sha256": OLD_TRAINING_SHA,
        "source_original_selected_manifest_sha256": OLD_FROZEN_SHA,
        "source_stl_selected_manifest_sha256": STL_FROZEN_SHA,
        "source_scientific_evidence": source_evidence,
        "source_input_identities": runner_evidence["input_identities"],
        "source_auxiliary_artifacts": auxiliary,
        "selected_checkpoints": selected, "classification": plan["classification"],
        "legacy_original_test_reference": {"root":str(OLD_RUN), "path":LEGACY_TEST_PATH,
            "sha256":LEGACY_TEST_SHA,"size_bytes":legacy_path.stat().st_size},
        "n_selected_models": 15, "n_test_sequences": N_TEST,
        "target_period": {"start_inclusive": START, "end_inclusive": END},
        "original_test_arrays_loaded_by_lock": False,
        "original_test_already_examined": True,
        "prior_2026_aggregate_outcomes_already_examined": True,
        "external_2026_data_or_output_files_opened_by_this_code": False,
        "new_models_fitted": False, "old_results_written": False}
    return core, (base, source, source_lock, source_evidence, prep, runner, old, stl, completed, original_inputs)


def write_new_json(path, value):
    no_symlinks(path)
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError as error:
            import errno
            if error.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
                raise
    if path.read_bytes() != data:
        raise OriginalPeriodError("Artifact failed exact byte readback.")


def lock_analysis(root):
    root = root_checked(root)
    lock_path = child(root, LOCK, must_exist=False)
    existing = read_json(lock_path) if lock_path.exists() else None
    if existing is None and (root / "runs").exists():
        raise OriginalPeriodError("A run directory already exists; no current-inference lock was created retroactively.")
    core, _ = collect_evidence(root)
    if existing is not None:
        if not isinstance(existing.get("created_at_utc"), str) or {k:v for k,v in existing.items() if k != "created_at_utc"} != core:
            raise OriginalPeriodError("Existing original-period lock differs; no overwrite performed.")
        reused = True
    else:
        if (root / "runs").exists() or lock_path.exists():
            raise OriginalPeriodError("Project state changed before the exclusive analysis-lock write.")
        write_new_json(lock_path, {"created_at_utc": datetime.now(timezone.utc).isoformat(), **core})
        reused = False
    return {"status": LOCK_STATUS, "analysis_lock": str(lock_path),
        "analysis_lock_sha256": digest(lock_path), "lock_reused": reused,
        "n_selected_checkpoints": 15, "original_test_arrays_loaded_by_this_code": False,
        "training_started": False, "old_results_written": False,
        "external_2026_data_or_output_files_opened_by_this_code": False}


def verify_analysis(root):
    root = root_checked(root)
    path = child(root, LOCK)
    locked = read_json(path)
    core, context = collect_evidence(root)
    if not isinstance(locked.get("created_at_utc"), str) or {k:v for k,v in locked.items() if k != "created_at_utc"} != core:
        raise OriginalPeriodError("Original-period analysis identity changed after the current-inference lock.")
    return locked, context


def load_original_test(source, identities, scaler):
    """Deserialize only the five original frozen test arrays after the lock."""
    import numpy as np
    meta_path = SEQ + "sequence_meta.json"
    meta = json.loads(checked_bytes(child(source / "inputs", meta_path), identities[meta_path]))
    expected_features = [f"{a}_{f}" for a in ASSETS for f in ("LogRet", "Vol20")]
    expected_targets = [f"{a}_NextRet" for a in ASSETS] + [f"{a}_NextVol" for a in ASSETS]
    if (meta.get("project_version") != "v4_repro" or meta.get("feature_set") != "baseline"
            or meta.get("lookback") != 10 or meta.get("split_basis") != "target_realization_date"
            or meta.get("scaler_rule") != "fit_train_only"
            or meta.get("feature_columns") != expected_features or meta.get("target_columns") != expected_targets
            or meta.get("date_ranges", {}).get("test_anchor") != ["2022-10-05","2024-12-30"]
            or meta.get("date_ranges", {}).get("test_target_realization") != [START,END]):
        raise OriginalPeriodError("Original baseline/L10 metadata, columns or target alignment differs.")
    values = {}
    for name in TEST_FILES:
        key = SEQ + name
        data = checked_bytes(child(source / "inputs", key), identities[key])
        values[name] = np.load(BytesIO(data), allow_pickle=False)
    for name, shape in (("X_test.npy", (N_TEST,10,8)), ("y_test.npy", (N_TEST,8)),
                        ("y_test_raw.npy", (N_TEST,8))):
        a = values[name]
        if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
            raise OriginalPeriodError("Original test array shape, dtype or finiteness differs: " + name)
    anchors = np.asarray(values["anchor_dates_test.npy"], dtype="datetime64[D]")
    targets = np.asarray(values["target_realization_dates_test.npy"], dtype="datetime64[D]")
    for dates in (anchors, targets):
        if dates.shape != (N_TEST,) or np.isnat(dates).any() or not np.all(np.diff(dates) > np.timedelta64(0,"D")):
            raise OriginalPeriodError("Original test dates must be finite, sorted and unique.")
    if ((str(targets[0]), str(targets[-1])) != (START, END)
            or (str(anchors[0]), str(anchors[-1])) != ("2022-10-05","2024-12-30")
            or not np.all(anchors < targets)):
        raise OriginalPeriodError("Original test realization dates differ from 2022-10-06 through 2024-12-31.")
    restored = scaler.inverse_transform(values["y_test.npy"].copy())
    if not np.allclose(restored, values["y_test_raw.npy"], rtol=REL_TOL, atol=ABS_TOL):
        raise OriginalPeriodError("Original target-scaler round trip differs.")
    return values


def original_period_loss_arrays(truth, prediction):
    """Match the archived 584-test evaluator: cast to float64 before subtraction."""
    import numpy as np
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if (truth.dtype != np.float32 or prediction.dtype != np.float32
            or truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != 8
            or not np.isfinite(truth).all() or not np.isfinite(prediction).all()):
        raise OriginalPeriodError("Finite matching raw float32 prediction/target arrays are required.")
    error = truth.astype(np.float64) - prediction.astype(np.float64)
    return {"return":np.abs(error[:,:4]),
            "volatility":np.maximum(0.5*error[:,4:],-0.5*error[:,4:])}


def verify_legacy_test_predictions(arrays):
    """Compare regenerated old-nine predictions/losses with the immutable archive."""
    import numpy as np
    payload = checked_bytes(child(OLD_RUN, LEGACY_TEST_PATH), {"sha256":LEGACY_TEST_SHA})
    proofs = []
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        for key,expected_key in (("target_realization_dates","target_realization_dates_test"),
                                 ("anchor_dates","anchor_dates_test")):
            actual_dates=np.asarray(arrays[key],dtype="datetime64[D]")
            expected_dates=np.asarray(archive[expected_key],dtype="datetime64[D]")
            if expected_dates.shape!=(N_TEST,) or not np.array_equal(actual_dates,expected_dates):
                raise OriginalPeriodError("Archived original-test date alignment differs: "+expected_key)
            proofs.append({"artifact":expected_key,"kind":"original_test_metadata","passed":True,
                           "max_absolute_difference":0.0})
        original_truth=archive["truth_raw_test"]
        if original_truth.shape!=(N_TEST,8) or original_truth.dtype!=np.float32 or not np.array_equal(original_truth,arrays["raw_truth"]):
            raise OriginalPeriodError("Archived original-test raw targets differ.")
        proofs.append({"artifact":"truth_raw_test","kind":"original_test_metadata","passed":True,
                       "max_absolute_difference":0.0})
        for arm in ARMS[:3]:
            predictions = arrays[f"{arm}__raw_predictions"]
            for i,seed in enumerate(SEEDS):
                prediction_key=f"prediction_raw__{arm}__seed{seed}__test"
                expected=archive[prediction_key]
                if expected.shape!=(N_TEST,8) or expected.dtype!=np.float32 or not np.allclose(predictions[i],expected,rtol=REL_TOL,atol=ABS_TOL):
                    raise OriginalPeriodError("Original nine-model raw prediction reproduction failed: "+prediction_key)
                proofs.append({"artifact":prediction_key,"kind":"raw_prediction","passed":True,
                    "max_absolute_difference":float(np.max(np.abs(predictions[i].astype(np.float64)-expected.astype(np.float64))))})
                for task in TASKS:
                    loss_key=f"loss__{arm}__seed{seed}__test__{task}"
                    actual=arrays[f"{arm}__{task}__per_asset_losses"][i].mean(axis=1)
                    expected_loss=archive[loss_key]
                    if expected_loss.shape!=(N_TEST,) or expected_loss.dtype!=np.float64 or not np.allclose(actual,expected_loss,rtol=REL_TOL,atol=ABS_TOL):
                        raise OriginalPeriodError("Original nine-model daily loss reproduction failed: "+loss_key)
                    proofs.append({"artifact":loss_key,"kind":"daily_primary_loss","passed":True,
                        "max_absolute_difference":float(np.max(np.abs(actual-expected_loss)))})
    return proofs


def original_period_columns(row):
    """Rename display columns only; retain every unchanged numerical value."""
    return {key.replace("external_primary_loss", "test_primary_loss")
                .replace("external_minus_train_gap", "test_minus_train_gap")
                .replace("external_minus_validation_gap", "test_minus_validation_gap"): value
            for key, value in row.items()}


def train_validation_primary_means(models, data, scaler, base, torch):
    """Calculate gap components in the same float64 loss arithmetic as test."""
    import numpy as np
    metrics = {}
    for arm in ARMS:
        for seed in SEEDS:
            for split,label in (("train","train"),("val","validation")):
                truth=data["y_raw_"+split]
                if arm=="STL_pair":
                    raw=np.empty_like(truth)
                    for task,columns in (("return",slice(0,4)),("volatility",slice(4,8))):
                        model=models[arm,seed,task].cuda().eval()
                        raw[:,columns]=base.predict(model,data["X_"+split],scaler,torch,task)
                        model.cpu()
                else:
                    model=models[arm,seed,None].cuda().eval()
                    raw=base.predict(model,data["X_"+split],scaler,torch)
                    model.cpu()
                losses=original_period_loss_arrays(truth,raw)
                for task in TASKS:
                    metrics[arm,seed,task,label]=float(losses[task].mean())
    if len(metrics)!=48:
        raise OriginalPeriodError("Every train/validation gap component must be calculated.")
    return metrics


def csv_write(path, rows):
    if not rows:
        raise OriginalPeriodError("An output table is empty.")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def existing_output(output, identities):
    manifest = read_json(child(output, "evaluation_manifest_v1.json"))
    if manifest.get("status") != COMPLETE or manifest.get("input_identities") != identities:
        raise OriginalPeriodError("Existing output belongs to another analysis; retain it without overwrite.")
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict) or not artifacts:
        raise OriginalPeriodError("Existing output lacks artifact identities.")
    for name, record in artifacts.items():
        if Path(name).name != name:
            raise OriginalPeriodError("Unsafe output artifact name.")
        checked_bytes(child(output, name), record)
    if {p.name for p in output.iterdir()} != set(artifacts) | {"evaluation_manifest_v1.json"}:
        raise OriginalPeriodError("Existing output contains unmanifested artifacts.")
    return manifest


def publish(output, identities, arrays, tables, report):
    import numpy as np
    no_symlinks(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    guard = output.parent / ".original_test_v1.lock"
    try:
        fd = os.open(guard, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise OriginalPeriodError("An active or interrupted publication guard exists; no overwrite.") from error
    os.close(fd)
    stage = None
    try:
        if output.exists():
            return existing_output(output, identities), True
        stage = Path(tempfile.mkdtemp(prefix=".original_test_v1-stage-", dir=output.parent))
        for name, rows in tables.items():
            if Path(name).name != name:
                raise OriginalPeriodError("Unsafe table name.")
            csv_write(stage / name, rows)
        if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
            raise OriginalPeriodError("Object arrays are forbidden in result archives.")
        np.savez_compressed(stage / "predictions_and_losses_v1.npz", **arrays)
        (stage / "analysis_report_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        artifacts = {p.name: {"sha256": digest(p), "size_bytes": p.stat().st_size} for p in stage.iterdir()}
        manifest = {"status": COMPLETE, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_identities": identities, "artifacts": artifacts,
            "period": {"start": START, "end_inclusive": END, "n_target_dates": N_TEST},
            "n_selected_models": 15, "n_arm_seed_predictions": 12,
            "n_planned_contrasts": 8, "n_dm_tests_including_sensitivity": 16,
            "original_test_arrays_loaded": True, "old_test_arrays_loaded": True,
            "training_started": False, "old_project_writes": False,
            "external_2026_data_or_output_files_opened": False}
        write_new_json(stage / "evaluation_manifest_v1.json", manifest)
        if output.exists():
            raise OriginalPeriodError("Output appeared concurrently; no overwrite performed.")
        os.rename(stage, output)
        stage = None
        return manifest, False
    finally:
        # Retain an interrupted partial stage and its guard for review. Only a
        # completed publication, or a read-only existing-output check, releases
        # this invocation's guard automatically.
        if stage is None:
            guard.unlink()


def evaluate(root):
    import numpy as np
    root = root_checked(root)
    locked, context = verify_analysis(root)
    base, source, _, source_evidence, prep, runner, old, stl, completed, old_inputs = context
    identities = {"analysis_lock_sha256": digest(child(root, LOCK)),
        "protocol_plan_sha256": locked["protocol_plan"]["sha256"],
        "evaluation_implementation_sha256": locked["evaluation_implementation"]["sha256"],
        "source_analysis_lock_sha256": SOURCE_LOCK_SHA,
        "source_scientific_evidence": source_evidence,
        "original_test_input_identities": {name:old_inputs[SEQ+name] for name in TEST_FILES}}
    output = child(root, OUTPUT, must_exist=False)
    if output.exists():
        return existing_output(output, identities), True
    if output.parent.exists() and any(p.name==".original_test_v1.lock" or p.name.startswith(".original_test_v1-stage-") for p in output.parent.iterdir()):
        raise OriginalPeriodError("An interrupted original-test publication is retained; no model inference was restarted.")
    print("Özgün dönem: 15 modelin eğitim/doğrulama sonuçları tekrar kontrol ediliyor; test dizileri henüz açılmadı.", flush=True)
    torch, runtime = runner.configure_runtime()
    namespace = runner.extract_definitions(base.child(source / "inputs", base.SOURCE_06), base.child(source / "inputs", base.SOURCE_08), torch)
    base.extract_joint(namespace, base.child(source / "inputs", base.SOURCE_06))
    data, scaler = runner.load_train_validation(source)
    curves = base.learning_curves(source, old, stl, completed)
    models, _, replays = base.replay_training(source, old, stl, completed, runner, namespace, data, scaler, torch)
    metrics=train_validation_primary_means(models,data,scaler,base,torch)
    test = load_original_test(source, old_inputs, scaler)
    truth, features = test["y_test_raw.npy"], test["X_test.npy"]
    arrays = {"raw_truth":truth, "scaled_truth":test["y_test.npy"],
        "anchor_dates":test["anchor_dates_test.npy"],
        "target_realization_dates":test["target_realization_dates_test.npy"],
        "seeds":np.asarray(SEEDS,dtype=np.int64), "assets":np.asarray(ASSETS), "arms":np.asarray(ARMS)}
    losses = {}
    for arm in ARMS:
        predictions = []
        for seed in SEEDS:
            if arm == "STL_pair":
                raw = np.empty_like(truth)
                for task, columns in (("return",slice(0,4)),("volatility",slice(4,8))):
                    model = models[arm,seed,task].cuda().eval()
                    raw[:,columns] = base.predict(model, features, scaler, torch, task)
                    model.cpu()
            else:
                model = models[arm,seed,None].cuda().eval()
                raw = base.predict(model, features, scaler, torch)
                model.cpu()
            predictions.append(raw)
            print(f"original test evaluated {arm} seed={seed}; 584 targets", flush=True)
        stack = np.stack(predictions)
        arrays[f"{arm}__raw_predictions"] = stack
        task_losses = [original_period_loss_arrays(truth, prediction) for prediction in stack]
        for task in TASKS:
            values = np.stack([item[task] for item in task_losses])
            losses[arm,task] = values
            arrays[f"{arm}__{task}__per_asset_losses"] = values
    legacy_replays = verify_legacy_test_predictions(arrays)
    comparisons, daily = base.comparisons(losses)
    for row in comparisons:
        if row["analysis"] == "prespecified":
            row["analysis"] = "fixed_original_period_supplement"
    seed_rows, summary_rows, gaps = base.descriptive_rows(losses, metrics)
    seed_rows = [original_period_columns(row) for row in seed_rows]
    summary_rows = [original_period_columns(row) for row in summary_rows]
    gaps = [original_period_columns(row) for row in gaps]
    gap_summary = []
    for arm in ARMS:
        for task in TASKS:
            selected = [row for row in gaps if row["arm"] == arm and row["task"] == task]
            row = {"arm":arm, "task":task, "n_seeds":3, "inference":"descriptive_only_three_seeds"}
            for key in ("train_primary_loss", "validation_primary_loss", "test_primary_loss",
                        "validation_minus_train_gap", "test_minus_train_gap", "test_minus_validation_gap"):
                values = np.asarray([item[key] for item in selected], dtype=np.float64)
                row[key+"_seed_mean"] = float(values.mean())
                row[key+"_seed_sample_sd"] = float(values.std(ddof=1))
            gap_summary.append(row)
    daily_rows = []
    for i, date in enumerate(arrays["target_realization_dates"]):
        daily_rows.append({"target_realization_date":str(date),
            **{f"{arm}__{task}":float(daily[arm,task][i]) for arm in ARMS for task in TASKS}})
    plan = read_json(child(root, PLAN))
    report = {"status":COMPLETE, "classification":plan["classification"],
        "period":{"main_raw_history":"2010-01-04 through 2024-12-31",
                  "test_first_target":START,"test_last_target":END,"n_target_dates":N_TEST},
        "runtime_environment":runtime, "statistics":plan["outcomes"]["statistics"],
        "aggregation":"Compute each seed's loss first; average four asset losses and then three seeds per date. Dates times seeds are not independent observations.",
        "loss_direction":"Control loss minus full-sharing loss; positive favors sharing.",
        "precision":"Store raw truth/forecasts as float32; cast both to float64 before pointwise subtraction, MAE/pinball and averaging, matching the archived original-584 evaluation.",
        "legacy_original_test_replay":{"all_passed":True,"n_raw_prediction_arrays":9,
            "n_daily_primary_loss_arrays":18,"n_original_metadata_arrays":3,
            "relative_tolerance":REL_TOL,"absolute_tolerance":ABS_TOL,
            "archive_sha256":LEGACY_TEST_SHA,
            "max_absolute_difference":max(row["max_absolute_difference"] for row in legacy_replays)},
        "train_validation_replay":{"all_passed":True,"n_metric_rows":48,"n_selected_models":15,
            "relative_tolerance":REL_TOL,"absolute_tolerance":ABS_TOL},
        "generalization_gap_components":{"n_train_validation_task_means":48,
            "precision":"raw truth and forecasts cast to float64 before pointwise loss and averaging",
            "checkpoint_selection_unchanged":True,"training_started":False},
        "comparisons":comparisons,
        "limitations":["The original test and prior 2026 aggregate outcomes were already examined; this is a supplementary retrospective comparison.",
            "Independent STL differs in optimizer, gradient clipping and task-specific checkpoint selection as well as sharing.",
            "Three-seed generalization gaps are descriptive; loss-series DM tests do not test a causal overfitting mechanism.",
            "All planned arms and tasks are reported; aggregate performance does not imply individual-asset significance."],
        "original_test_arrays_loaded":True,"old_test_arrays_loaded":True,
        "training_started":False,"old_project_writes":False,
        "external_2026_data_or_output_files_opened":False}
    fresh, _ = verify_analysis(root)
    if fresh != locked:
        raise OriginalPeriodError("Original-period lock changed during evaluation.")
    return publish(output, identities, arrays, {
        "dm_holm_comparisons_v1.csv":comparisons,
        "primary_losses_by_seed_and_asset_v1.csv":seed_rows,
        "primary_losses_seed_summary_v1.csv":summary_rows,
        "generalization_gaps_descriptive_v1.csv":gaps,
        "generalization_gaps_seed_summary_v1.csv":gap_summary,
        "daily_seed_mean_primary_losses_v1.csv":daily_rows,
        "train_validation_replay_v1.csv":replays,"learning_curves_v1.csv":curves,
        "original_nine_test_replay_v1.csv":legacy_replays}, report)


def self_test():
    """Generated-array integration checks only; no user data or model fitting."""
    import numpy as np
    from unittest.mock import patch
    checks = []
    with tempfile.TemporaryDirectory(prefix="h2-original-test-fixture-") as directory:
        root = Path(directory)
        source = root / "source"
        inputs = source / "inputs" / SEQ
        inputs.mkdir(parents=True)
        # Use all calendar dates and select a deterministic 584-date subsequence.
        calendar = np.arange(np.datetime64(START), np.datetime64(END)+1, dtype="datetime64[D]")
        positions = np.linspace(0,len(calendar)-1,N_TEST,dtype=int)
        dates = calendar[positions]
        truth = np.arange(N_TEST*8,dtype=np.float32).reshape(N_TEST,8)/1000
        values = {"X_test.npy":np.zeros((N_TEST,10,8),dtype=np.float32),
            "y_test.npy":truth.copy(),"y_test_raw.npy":truth.copy(),
            "anchor_dates_test.npy":(dates-np.timedelta64(1,"D")).astype("U10"),
            "target_realization_dates_test.npy":dates.astype("U10")}
        identities = {}
        for name, value in values.items():
            np.save(inputs/name,value,allow_pickle=False)
            identities[SEQ+name] = {"sha256":digest(inputs/name),"size_bytes":(inputs/name).stat().st_size}
        meta={"project_version":"v4_repro","feature_set":"baseline","lookback":10,
            "split_basis":"target_realization_date","scaler_rule":"fit_train_only",
            "feature_columns":[f"{a}_{f}" for a in ASSETS for f in ("LogRet","Vol20")],
            "target_columns":[f"{a}_NextRet" for a in ASSETS]+[f"{a}_NextVol" for a in ASSETS],
            "date_ranges":{"test_anchor":["2022-10-05","2024-12-30"],"test_target_realization":[START,END]}}
        (inputs/"sequence_meta.json").write_text(json.dumps(meta))
        identities[SEQ+"sequence_meta.json"]={"sha256":digest(inputs/"sequence_meta.json")}
        class IdentityScaler:
            def inverse_transform(self, value):
                return value
        actual = load_original_test(source,identities,IdentityScaler())
        assert np.array_equal(actual["y_test_raw.npy"],truth)
        checks.append("all_five_original_test_arrays_verified_and_loaded")
        for wrong_name, wrong in (("X_test.npy",np.zeros((583,10,8),dtype=np.float32)),
                                  ("y_test.npy",truth.astype(np.float64)),
                                  ("y_test_raw.npy",np.full((N_TEST,8),np.nan,dtype=np.float32))):
            saved = (inputs/wrong_name).read_bytes()
            np.save(inputs/wrong_name,wrong,allow_pickle=False)
            changed = dict(identities)
            changed[SEQ+wrong_name] = {"sha256":digest(inputs/wrong_name)}
            try:
                load_original_test(source,changed,IdentityScaler())
                raise AssertionError("Invalid test array was accepted.")
            except OriginalPeriodError:
                pass
            (inputs/wrong_name).write_bytes(saved)
        checks.append("wrong_shape_dtype_and_nonfinite_targets_rejected")
        saved = (inputs/"target_realization_dates_test.npy").read_bytes()
        np.save(inputs/"target_realization_dates_test.npy",dates[::-1].astype("U10"),allow_pickle=False)
        changed = dict(identities)
        changed[SEQ+"target_realization_dates_test.npy"]={"sha256":digest(inputs/"target_realization_dates_test.npy")}
        try:
            load_original_test(source,changed,IdentityScaler())
            raise AssertionError("Unsorted dates accepted.")
        except OriginalPeriodError:
            pass
        (inputs/"target_realization_dates_test.npy").write_bytes(saved)
        checks.append("date_chronology_rejected")
        bad = dict(identities)
        bad[SEQ+"X_test.npy"]={"sha256":"0"*64}
        try:
            load_original_test(source,bad,IdentityScaler())
            raise AssertionError("Changed test identity accepted.")
        except OriginalPeriodError:
            pass
        checks.append("tampered_test_identity_rejected")
        out = root / OUTPUT
        arrays = {"truth":truth, "dates":dates.astype("U10")}
        manifest,reused=publish(out,{"fixture":"only"},arrays,{"table.csv":[{"x":1}]},{"fixture":True})
        assert manifest["old_test_arrays_loaded"] is True and not reused
        before={p.name:p.read_bytes() for p in out.iterdir()}
        again,reused=publish(out,{"fixture":"only"},arrays,{"table.csv":[{"x":1}]},{"fixture":True})
        assert reused and again==manifest and before=={p.name:p.read_bytes() for p in out.iterdir()}
        checks.append("immutable_publication_and_retry")
        try:
            publish(out,{"fixture":"different"},arrays,{"table.csv":[{"x":1}]},{"fixture":True})
            raise AssertionError("Output identity change accepted.")
        except OriginalPeriodError:
            pass
        checks.append("changed_output_identity_rejected")
        partial=root/"partial"/"original_test_v1"
        with patch(__name__+".csv_write",side_effect=RuntimeError("synthetic write failure")):
            try:
                publish(partial,{"fixture":"partial"},arrays,{"table.csv":[{"x":1}]},{"fixture":True})
                raise AssertionError("Synthetic publication failure was hidden.")
            except RuntimeError as error:
                assert str(error)=="synthetic write failure"
        assert (partial.parent/".original_test_v1.lock").is_file()
        assert any(p.name.startswith(".original_test_v1-stage-") for p in partial.parent.iterdir())
        try:
            publish(partial,{"fixture":"partial"},arrays,{"table.csv":[{"x":1}]},{"fixture":True})
            raise AssertionError("Interrupted publication was restarted.")
        except OriginalPeriodError:
            pass
        checks.append("partial_publication_and_guard_retained_without_retry")
        renamed=original_period_columns({"external_primary_loss":1,"external_minus_train_gap":2,
            "external_minus_validation_gap_seed_mean":3})
        assert renamed=={"test_primary_loss":1,"test_minus_train_gap":2,"test_minus_validation_gap_seed_mean":3}
        checks.append("test_labels_preserve_numeric_values")
        prediction=(truth+np.float32(0.0100001)).astype(np.float32)
        primary=original_period_loss_arrays(truth,prediction)
        expected_error=truth.astype(np.float64)-prediction.astype(np.float64)
        assert primary["return"].dtype==np.float64
        assert np.array_equal(primary["return"],np.abs(expected_error[:,:4]))
        assert np.array_equal(primary["volatility"],np.abs(expected_error[:,4:])*0.5)
        checks.append("archived_float64_before_subtraction_precision_preserved")
        legacy_dir=root/"legacy_original_run"/"evaluation_v1"
        legacy_dir.mkdir(parents=True)
        legacy_values={"truth_raw_test":truth,"anchor_dates_test":values["anchor_dates_test.npy"],
            "target_realization_dates_test":values["target_realization_dates_test.npy"]}
        check_arrays={"raw_truth":truth,"anchor_dates":values["anchor_dates_test.npy"],
            "target_realization_dates":values["target_realization_dates_test.npy"]}
        for arm in ARMS[:3]:
            check_arrays[f"{arm}__raw_predictions"]=np.stack([prediction]*3)
            for i,seed in enumerate(SEEDS):
                legacy_values[f"prediction_raw__{arm}__seed{seed}__test"]=prediction
                for task in TASKS:
                    legacy_values[f"loss__{arm}__seed{seed}__test__{task}"]=primary[task].mean(axis=1)
            for task in TASKS:
                check_arrays[f"{arm}__{task}__per_asset_losses"]=np.stack([primary[task]]*3)
        legacy_file=legacy_dir/"h2_losses_predictions_v1.npz"
        np.savez_compressed(legacy_file,**legacy_values)
        with patch(__name__+".OLD_RUN",root/"legacy_original_run"),patch(__name__+".LEGACY_TEST_SHA",digest(legacy_file)):
            proof=verify_legacy_test_predictions(check_arrays)
            assert len(proof)==30 and all(item["max_absolute_difference"]==0 for item in proof)
            check_arrays["raw_truth"]=truth.copy()
            check_arrays["raw_truth"][0,0]+=np.float32(1)
            try:
                verify_legacy_test_predictions(check_arrays)
                raise AssertionError("Changed original-test truth accepted.")
            except OriginalPeriodError:
                pass
        checks.append("all_thirty_legacy_compatibility_arrays_verified_and_truth_drift_rejected")
        # The lock handler's own chronology/immutability, with metadata collection
        # replaced only in this temporary fixture; no production override exists.
        lock_root=root/"lock_project"
        lock_root.mkdir()
        core={"status":LOCK_STATUS,"fixture":"opaque_metadata"}
        with patch(__name__+".collect_evidence",return_value=(core,None)):
            first=lock_analysis(lock_root)
            second=lock_analysis(lock_root)
            assert not first["lock_reused"] and second["lock_reused"]
            assert second["original_test_arrays_loaded_by_this_code"] is False
            before=(lock_root/LOCK).read_bytes()
            (lock_root/LOCK).write_text(json.dumps({"created_at_utc":"fixture","status":"changed"}))
            try:
                lock_analysis(lock_root)
                raise AssertionError("Changed analysis lock accepted.")
            except OriginalPeriodError:
                pass
            (lock_root/LOCK).write_bytes(before)
        checks.append("current_inference_lock_idempotence_and_drift_rejection")
        retro=root/"retroactive_lock_project"
        (retro/"runs").mkdir(parents=True)
        try:
            lock_analysis(retro)
            raise AssertionError("Retroactive lock accepted.")
        except OriginalPeriodError:
            pass
        checks.append("retroactive_lock_rejected")
    return {"status":"ORIGINAL_PERIOD_OFFLINE_SELF_TEST_PASS","n_checks":len(checks),"checks":checks,
        "real_original_test_arrays_loaded":False,"model_training_started":False,
        "external_2026_data_or_output_files_opened":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=DEFAULT_ROOT)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--lock",action="store_true")
    modes.add_argument("--evaluate",action="store_true")
    modes.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        result=self_test()
    elif args.lock:
        result=lock_analysis(args.project_root)
    else:
        manifest,reused=evaluate(args.project_root)
        result={"status":manifest["status"],"reused_identical_evaluation":reused,
            "output":str(args.project_root/OUTPUT),"evaluation_manifest_sha256":digest(args.project_root/OUTPUT/"evaluation_manifest_v1.json"),
            "n_target_dates":N_TEST,"first_target":START,"last_target":END,
            "n_selected_models":15,"n_planned_contrasts":8,"n_dm_tests_including_sensitivity":16,
            "original_test_arrays_loaded":True,"old_test_arrays_loaded":True,
            "training_started":False,"old_project_writes":False,
            "external_2026_data_or_output_files_opened":False}
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
