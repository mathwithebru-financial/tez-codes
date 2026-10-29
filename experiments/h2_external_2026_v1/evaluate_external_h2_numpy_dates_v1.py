#!/usr/bin/env python3
"""Verify the recorded date-type repair, then run the unchanged H2 evaluator.

The original analysis lock and scientific implementations remain unchanged.
Only the separately versioned preprocessing date conversion is amended. This
wrapper verifies that amendment before any inference and records its linkage to
the original evaluator's outputs outside the immutable evaluation directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import sys
import types

sys.dont_write_bytecode = True
DEFAULT_ROOT = Path("/content/drive/MyDrive/tez_transformer_h2_external_2026_v1")
PARENT_LOCK_SHA256 = "a155ad428996a6c1d40ca6a755212206ca9092a7b0fdc76200c3a1f063bdb274"
ORIGINAL_EVALUATOR_SHA256 = "773f01d1c9d5cd8b535d2c8640c86b0b039ab9391d6265fb734543861d953a0f"
AMENDMENT = "technical_amendment_numpy_dates_v1.json"
AMENDMENT_STATUS = "TECHNICAL_AMENDMENT_RECORDED_AFTER_PRICE_ACCESS_BEFORE_EXTERNAL_MODEL_EVALUATION"
PREPARED = "new_period/prepared_v1/prepared_manifest_v1.json"
EVALUATION = "new_period/evaluation_v1/evaluation_manifest_v1.json"
RECEIPT = "evaluation_technical_receipt_numpy_dates_v1.json"
PROTECTED = tuple(Path("/content/drive/MyDrive") / name for name in
                  ("tez_transformer_v4_repro", "tez_transformer_h2_v1", "tez_transformer_h2_t4_v1"))


class AmendmentEvaluationError(RuntimeError):
    """A technical-amendment or execution identity does not match."""


def no_symlinks(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise AmendmentEvaluationError("Symbolic-link paths are not allowed.")


def child(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise AmendmentEvaluationError("An artifact path must be safely project-relative.")
    path = Path(root) / relative
    no_symlinks(path)
    resolved = path.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise AmendmentEvaluationError("Artifact path escapes the separately staged project.")
    return resolved


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def checked_file(root, relative, identity):
    path = child(root, relative)
    if not isinstance(identity, dict) or identity.get("path", relative) != relative:
        raise AmendmentEvaluationError("An artifact identity names another path.")
    data = path.read_bytes()
    if (sha_bytes(data) != identity.get("sha256")
            or "size_bytes" in identity and len(data) != identity["size_bytes"]):
        raise AmendmentEvaluationError("Artifact identity differs: " + relative)
    return data


def object_bytes(path):
    no_symlinks(path)
    data = Path(path).read_bytes()
    value = json.loads(data)
    if not isinstance(value, dict):
        raise AmendmentEvaluationError("A JSON object is required.")
    return value, data


def module_from_bytes(path, data, name):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def before_evaluation(root):
    no_symlinks(root)
    root = Path(root).resolve()
    if not root.is_dir() or any(root.is_relative_to(p) or p.is_relative_to(root) for p in PROTECTED):
        raise AmendmentEvaluationError("Use the separately staged external H2 project.")
    parent = checked_file(root, "analysis_locked_v1.json", {"sha256": PARENT_LOCK_SHA256})
    amendment, amendment_bytes = object_bytes(child(root, AMENDMENT))
    if (amendment.get("status") != AMENDMENT_STATUS
            or amendment.get("analytical_decisions_changed") is not False
            or amendment.get("new_data_already_accessed") is not True
            or amendment.get("external_model_metrics_evaluated_before_amendment") is not False):
        raise AmendmentEvaluationError("The required before-metrics technical repair is not recorded.")
    checked_file(root, "analysis_locked_v1.json", amendment.get("parent_analysis_lock"))
    wrapper_relative = "implementation/evaluate_external_h2_numpy_dates_v1.py"
    wrapper = checked_file(root, wrapper_relative, amendment.get("evaluation_wrapper_implementation"))
    if Path(__file__).resolve() != child(root, wrapper_relative):
        raise AmendmentEvaluationError("Execute the staged amended evaluation wrapper.")
    preparation_relative = "implementation/prepare_external_sequences_v2.py"
    preparation_source = checked_file(root, preparation_relative, amendment.get("corrected_implementation"))
    preparation = module_from_bytes(child(root, preparation_relative), preparation_source,
                                    "h2_verified_date_amendment")
    verified_amendment, amendment_identity = preparation.verify_technical_amendment(root)
    expected_amendment_identity = {"path": AMENDMENT, "sha256": sha_bytes(amendment_bytes),
                                   "size_bytes": len(amendment_bytes)}
    if verified_amendment != amendment or amendment_identity != expected_amendment_identity:
        raise AmendmentEvaluationError("The amendment changed during verification.")
    prepared, prepared_bytes = object_bytes(child(root, PREPARED))
    if (prepared.get("status") != "EXTERNAL_SEQUENCES_PREPARED_NO_MODEL_EVALUATION"
            or prepared.get("input_identities", {}).get("technical_amendment_sha256") != amendment_identity["sha256"]
            or prepared.get("technical_amendment") != amendment_identity
            or any(prepared.get(k) is not False for k in
                   ("models_evaluated", "predictions_produced", "old_test_arrays_loaded", "old_files_modified"))):
        raise AmendmentEvaluationError("Prepared sequences do not identify this technical amendment.")
    original_relative = "implementation/evaluate_external_h2_v1.py"
    original_source = checked_file(root, original_relative, {"sha256": ORIGINAL_EVALUATOR_SHA256})
    evaluator = module_from_bytes(child(root, original_relative), original_source,
                                 "h2_unchanged_original_evaluation")
    inputs = {"parent_analysis_lock_sha256": sha_bytes(parent),
              "technical_amendment_sha256": amendment_identity["sha256"],
              "corrected_preparation_sha256": sha_bytes(preparation_source),
              "original_evaluation_sha256": sha_bytes(original_source),
              "evaluation_wrapper_sha256": sha_bytes(wrapper),
              "prepared_manifest_sha256": sha_bytes(prepared_bytes)}
    return root, preparation, evaluator, inputs


def write_or_verify_receipt(root, core):
    path = child(root, RECEIPT)
    if path.exists():
        record, _ = object_bytes(path)
        if not isinstance(record.get("created_at_utc"), str) or {k: v for k, v in record.items() if k != "created_at_utc"} != core:
            raise AmendmentEvaluationError("Existing execution receipt differs; no overwrite performed.")
        return sha_bytes(path.read_bytes())
    record = {"created_at_utc": datetime.now(timezone.utc).isoformat(), **core}
    data = (json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError as error:
            if error.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP):
                raise
    if path.read_bytes() != data:
        raise AmendmentEvaluationError("Execution receipt failed exact byte readback.")
    return sha_bytes(data)


def evaluate(root):
    root, preparation, evaluator, inputs = before_evaluation(root)
    receipt_path = child(root, RECEIPT)
    if receipt_path.exists():
        previous, _ = object_bytes(receipt_path)
        if previous.get("input_identities") != inputs:
            raise AmendmentEvaluationError("Existing technical receipt belongs to different inputs.")
        checked_file(root, EVALUATION, previous.get("evaluation_manifest"))
    # This is the original evaluator function in its own original module. Its
    # original code, all fifteen checkpoint gates, replay, and statistics execute
    # unchanged; its __file__ correctly identifies that original source file.
    result, reused = evaluator.evaluate(root)
    if result.get("status") != "H2_EXTERNAL_EVALUATION_COMPLETE":
        raise AmendmentEvaluationError("The unchanged evaluator did not complete successfully.")
    _, _, _, after = before_evaluation(root)
    if after != inputs:
        raise AmendmentEvaluationError("Technical or scientific identities changed during evaluation.")
    manifest, manifest_bytes = object_bytes(child(root, EVALUATION))
    if manifest != result or manifest.get("input_identities", {}).get("prepared_manifest_sha256") != inputs["prepared_manifest_sha256"]:
        raise AmendmentEvaluationError("Evaluation output does not identify these prepared sequences.")
    core = {"status": "H2_EXTERNAL_EVALUATION_COMPLETE_WITH_RECORDED_TECHNICAL_DATE_FIX",
            "input_identities": inputs,
            "evaluation_manifest": {"path": EVALUATION, "sha256": sha_bytes(manifest_bytes),
                                    "size_bytes": len(manifest_bytes)},
            "scientific_evaluation_implementation_changed": False,
            "analytical_decisions_changed": False, "training_started": False,
            "old_test_arrays_loaded": False, "old_project_writes": False}
    receipt_sha = write_or_verify_receipt(root, core)
    return {"status": result["status"], "reused_identical_evaluation": reused,
            "technical_amendment_sha256": inputs["technical_amendment_sha256"],
            "technical_execution_receipt": str(child(root, RECEIPT)),
            "technical_execution_receipt_sha256": receipt_sha,
            "output": str(root / "new_period/evaluation_v1"),
            "n_selected_models": 15, "n_planned_contrasts": 8,
            "n_dm_tests_including_sensitivity": 16,
            "training_started": False, "old_test_arrays_loaded": False,
            "old_project_writes": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--evaluate", action="store_true", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.project_root), ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
