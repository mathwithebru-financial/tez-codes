#!/usr/bin/env python3
"""Evaluate every frozen H2 model on the prespecified external period.

No fitting, hyperparameter selection, scaler fitting or old-test loading is
performed. All fifteen checkpoint identities, the final analysis lock, frozen
inputs, and prepared external artifacts are verified before deserialization.
The old training/validation metrics are replayed before external inference.
The --self-test path uses generated numbers only and needs no PyTorch.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.dont_write_bytecode = True

DEFAULT_ROOT = Path("/content/drive/MyDrive/tez_transformer_h2_external_2026_v1")
PLAN_SHA = "0073b3bcc80a342d25f1034fef66d5eb353a9e4e5c14ae3719f57e98de81a659"
SOURCE_06_SHA = "35de2ee398699003dfef6be36b70c112fb2c0d1b1e9577cbf64bef58877e16d8"
SOURCE_08_SHA = "df4fa2b29a6b522fc410f693a30520e045c144a15f7c7368af5eb6a1f0f12566"
OLD_TRAINING_SHA = "517c06f3b1a5d71de9739f56ef6a2af7cf9a918ed2cf303a52d30bfc7b9f393a"
SOURCE_06 = "scripts/06_best_model_multiseed_v4.py"
SOURCE_08 = "scripts/08B_learned_baselines_test_v4.py"
ASSETS = ("BIST100", "USDTRY", "EURTRY", "GOLD")
SEEDS = (123, 777, 2026)
TASKS = ("return", "volatility")
ARMS = ("NS_base", "FS_same_width", "FS_budget_control", "STL_pair")
FS_ARMS = ("FS_same_width", "FS_budget_control")
OLD_RUN = Path("/content/drive/MyDrive/tez_transformer_h2_t4_v1/runs/t4_full_run_v1")
PROTECTED = tuple(Path(p) for p in (
    "/content/drive/MyDrive/tez_transformer_v4_repro",
    "/content/drive/MyDrive/tez_transformer_h2_v1",
    "/content/drive/MyDrive/tez_transformer_h2_t4_v1"))
PREPARED = "new_period/prepared_v1"
ACQUISITION = "new_period/acquisition_v1"
OUTPUT = "new_period/evaluation_v1"
REL_TOL, ABS_TOL = 1e-5, 1e-8


class EvaluationError(RuntimeError):
    """An identity, numerical-replay or analysis guard failed."""


def no_symlinks(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise EvaluationError(f"Symlink paths are not allowed: {path}")


def child(root, relative, *, must_exist=True):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise EvaluationError("A safe relative artifact path is required.")
    candidate = Path(root) / relative
    no_symlinks(candidate)
    result = candidate.resolve()
    if result == Path(root).resolve() or not result.is_relative_to(Path(root).resolve()):
        raise EvaluationError("Artifact path escapes its project root.")
    if must_exist and not result.is_file():
        raise EvaluationError(f"Missing artifact: {result}")
    return result


def digest(path):
    no_symlinks(path)
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_read(path):
    no_symlinks(path)
    value = json.loads(Path(path).read_bytes())
    if not isinstance(value, dict):
        raise EvaluationError("A JSON object is required.")
    return value


def checked_bytes(path, identity):
    no_symlinks(path)
    payload = Path(path).read_bytes()
    if (not isinstance(identity, dict) or hashlib.sha256(payload).hexdigest() != identity.get("sha256")
            or ("size_bytes" in identity and len(payload) != identity["size_bytes"])):
        raise EvaluationError(f"Artifact identity differs: {path}")
    return payload


def checked_json(path, identity):
    value = json.loads(checked_bytes(path, identity))
    if not isinstance(value, dict):
        raise EvaluationError("A verified JSON object is required.")
    return value


def checked_module(path, expected_sha, name):
    """Execute only an analysis implementation frozen by the final lock."""
    payload = checked_bytes(path, {"sha256": expected_sha})
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(payload, str(path), "exec"), module.__dict__)
    return module


def bootstrap(root):
    """No external arrays, prices, checkpoint payloads or historical modules open."""
    no_symlinks(root)
    root = Path(root).resolve()
    if not root.is_dir() or any(root.is_relative_to(p) or p.is_relative_to(root) for p in PROTECTED):
        raise EvaluationError("A separately staged, unprotected project root is required.")
    lock = json_read(child(root, "analysis_locked_v1.json"))
    if lock.get("status") != "ANALYSIS_LOCKED_BEFORE_NEW_DATA" or lock.get("protocol_plan_sha256") != PLAN_SHA:
        raise EvaluationError("The exact reviewed analysis plan is not locked.")
    implementation = lock.get("implementation_sha256", {})
    required = (Path(__file__).name, "prepare_external_sequences_v1.py", "run_stl_controls_v1.py")
    if not isinstance(implementation, dict) or any(n not in implementation for n in required):
        raise EvaluationError("The executing evaluator and its helpers must be frozen.")
    checked_bytes(Path(__file__).resolve(), {"sha256": implementation[Path(__file__).name]})
    prep = checked_module(child(root, "implementation/prepare_external_sequences_v1.py"),
                          implementation["prepare_external_sequences_v1.py"], "h2_verified_preparation")
    verified_lock, evidence = prep.verify_analysis_lock(root)
    if verified_lock != lock:
        raise EvaluationError("Analysis lock changed during verification.")
    checked_bytes(child(root,"stage_receipt_v1.json"),{"sha256":lock.get("stage_receipt_sha256")})
    auxiliary=lock.get("auxiliary_artifact_identities")
    if not isinstance(auxiliary,list) or not auxiliary:
        raise EvaluationError("The final lock must identify auxiliary training artifacts.")
    for item in auxiliary:
        scope=item.get("scope")
        if scope not in ("project","old_h2_run"):
            raise EvaluationError("Unknown auxiliary-artifact root scope.")
        checked_bytes(child(root if scope=="project" else OLD_RUN,item.get("path")),item)
    prep.verify_frozen_inputs(root)
    runner = checked_module(child(root, "implementation/run_stl_controls_v1.py"),
                            implementation["run_stl_controls_v1.py"], "h2_verified_stl_training")
    runner_evidence = runner.read_evidence(root)
    if runner_evidence["plan_sha256"] != PLAN_SHA:
        raise EvaluationError("Original input/model sources do not belong to the frozen plan.")
    return root, lock, evidence, prep, runner


def dm_hln_bartlett(differences, lag):
    """Two-sided DM; biased gamma_k/T; Bartlett weights; HLN h=1; t_(T-1)."""
    import numpy as np
    from scipy.stats import t as student_t
    d = np.asarray(differences, dtype=np.float64)
    if d.ndim != 1 or len(d) < 3 or not np.isfinite(d).all():
        raise EvaluationError("DM requires a finite, one-dimensional aligned loss series.")
    if type(lag) is not int or lag < 0 or lag >= len(d):
        raise EvaluationError("DM lag must be an integer in [0,T).")
    n = len(d)
    mean = float(d.mean())
    centered = d - mean
    gamma = [float(np.dot(centered[k:], centered[:n-k]) / n) for k in range(lag + 1)]
    lrv = gamma[0] + 2 * sum((1 - k / (lag + 1)) * gamma[k] for k in range(1, lag + 1))
    tolerance = np.finfo(np.float64).eps * max(gamma[0], 1e-300) * max(64, 8 * (lag + 1))
    if lrv < -tolerance:
        raise EvaluationError("Bartlett long-run variance is numerically negative.")
    lrv = max(0.0, lrv)
    factor = math.sqrt((n - 1) / n)
    result = {"n_dates": n, "lag": lag, "h": 1, "autocovariance_denominator": "T",
              "mean_control_minus_sharing_loss": mean, "long_run_variance": lrv,
              "hln_factor": factor, "reference_df": n - 1, "two_sided": True,
              "dm_hln_t": None, "p_raw": None, "ci95_lower": None, "ci95_upper": None}
    if lrv == 0:
        if np.all(d == 0):
            result.update(status="IDENTICAL_LOSS_SERIES", dm_hln_t=0.0, p_raw=1.0,
                          ci95_lower=0.0, ci95_upper=0.0)
        else:
            # Undefined standard error must not become a spurious p=0. The
            # planned family still contains this comparison, conservatively p=1.
            result.update(status="UNDEFINED_ZERO_LONG_RUN_VARIANCE")
        return result
    standard_error = math.sqrt(lrv / n) / factor
    statistic = mean / standard_error
    critical = float(student_t.ppf(0.975, n - 1))
    result.update(status="OK", dm_hln_t=statistic,
                  p_raw=float(2 * student_t.sf(abs(statistic), n - 1)),
                  ci95_lower=mean - critical * standard_error,
                  ci95_upper=mean + critical * standard_error)
    return result


def holm_adjust(p_values):
    """Return original-order Holm adjusted p; None retains a conservative slot."""
    values = [1.0 if p is None else float(p) for p in p_values]
    if not values or any(not math.isfinite(p) or p < 0 or p > 1 for p in values):
        raise EvaluationError("Holm requires finite probabilities in [0,1].")
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    adjusted = [None] * len(values)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = previous
    return adjusted


def loss_arrays(truth, prediction):
    """Pointwise raw-scale float32 errors; averaging/inference in float64."""
    import numpy as np
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != 8:
        raise EvaluationError("Raw prediction and truth require matching [date,8] arrays.")
    if truth.dtype != np.float32 or prediction.dtype != np.float32:
        raise EvaluationError("Frozen target and inverse-prediction arithmetic must be float32.")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise EvaluationError("Nonfinite target/prediction.")
    error = truth - prediction
    return {"return": np.abs(error[:, :4]).astype(np.float64),
            "volatility": np.maximum(0.5 * error[:, 4:], -0.5 * error[:, 4:]).astype(np.float64)}


def comparisons(losses):
    """Each input arm/task has shape [three_seeds,dates,four_assets]."""
    import numpy as np
    expected = {(a,t) for a in ARMS for t in TASKS}
    if set(losses) != expected:
        raise EvaluationError("All four arms and both tasks must be present.")
    shapes = {np.asarray(v).shape for v in losses.values()}
    if len(shapes) != 1:
        raise EvaluationError("Loss series must align on identical seeds, dates and assets.")
    shape = next(iter(shapes))
    if len(shape) != 3 or shape[0] != 3 or shape[1] < 150 or shape[2] != 4:
        raise EvaluationError("Loss arrays require [3,at_least_150,4].")
    if any(not np.isfinite(v).all() or (np.asarray(v) < 0).any() for v in losses.values()):
        raise EvaluationError("Loss arrays must be finite and nonnegative.")
    # Seed is averaged after each seed's pointwise loss is calculated. Do not
    # stack dates*seeds or evaluate a mean forecast as if it were this estimand.
    daily = {key: np.asarray(value, dtype=np.float64).mean(axis=2).mean(axis=0)
             for key, value in losses.items()}
    rows = []
    for lag in (19, 0):
        for family, control in (("primary", "STL_pair"), ("secondary", "NS_base")):
            family_rows = []
            for arm in FS_ARMS:
                for task in TASKS:
                    difference = daily[control, task] - daily[arm, task]
                    row = {"analysis": "prespecified" if lag == 19 else "lag0_sensitivity",
                           "family": family, "control": control, "sharing_arm": arm, "task": task,
                           "control_mean_loss": float(daily[control,task].mean()),
                           "sharing_mean_loss": float(daily[arm,task].mean()),
                           **dm_hln_bartlett(difference, lag)}
                    row["relative_loss_reduction"] = (float(difference.mean()) / row["control_mean_loss"]
                                                       if row["control_mean_loss"] > 0 else None)
                    family_rows.append(row)
            adjusted = holm_adjust([r["p_raw"] for r in family_rows])
            for row, p in zip(family_rows, adjusted):
                row["family_size"] = 4
                row["holm_p_input"] = 1.0 if row["p_raw"] is None else row["p_raw"]
                row["holm_adjusted_p"] = p
                row["holm_alpha"] = 0.05
                significant = row["p_raw"] is not None and p <= 0.05
                row["reject_equal_predictive_loss"] = significant
                row["decision"] = ("undefined_test_no_rejection" if row["p_raw"] is None else
                    "sharing_lower_loss" if significant and row["mean_control_minus_sharing_loss"] > 0 else
                    "sharing_higher_loss" if significant and row["mean_control_minus_sharing_loss"] < 0 else
                    "no_multiplicity_adjusted_difference")
            rows.extend(family_rows)
    return rows, daily


def frozen_auxiliary(root, prep, evidence):
    """Verify history and completed-job bytes before parsing any of them."""
    old = checked_json(root / "source_records/selected_checkpoints_frozen.json",
        {"sha256":evidence["existing_h2_checkpoints_manifest_sha256"]})
    stl = checked_json(root / "runs/stl_run_v1/selected_stl_checkpoints_frozen.json",
        {"sha256":evidence["selected_stl_checkpoints_manifest_sha256"]})
    identities, completed = [], {}
    for job in old["completed_runs"]:
        item = job["history"]
        prep.verify_identity(child(OLD_RUN, item["path"]), item)
        identities.append({"root": "original_h2_run", **item})
    for selected in stl["checkpoints"]:
        record_path = child(root, selected["completed_job_path"])
        record = checked_json(record_path, {"sha256": selected["completed_job_sha256"]})
        if (record.get("status") != "STL_TASK_TRAINING_COMPLETE_NO_NEW_DATA"
                or record.get("task") != selected["task"] or record.get("seed") != selected["seed"]
                or record.get("selected_epoch") != selected["selected_epoch"]):
            raise EvaluationError("A frozen STL completed-job identity differs.")
        for item in record.get("artifacts", {}).values():
            prep.verify_identity(child(root, item["path"]), item)
            identities.append({"root": "external_project", **item})
        completed[selected["task"], selected["seed"]] = record
    return old, stl, completed, identities


def verify_prepared(root, prep, evidence, plan):
    """This follows every source/checkpoint gate and precedes array parsing."""
    output = root / PREPARED
    no_symlinks(output)
    manifest = prep.read_json(child(output, "prepared_manifest_v1.json"))
    if manifest.get("status") != "EXTERNAL_SEQUENCES_PREPARED_NO_MODEL_EVALUATION":
        raise EvaluationError("External sequences have not passed the preparation gate.")
    identities = manifest.get("input_identities", {})
    for key, value in evidence.items():
        if identities.get(key) != value:
            raise EvaluationError(f"Prepared inputs belong to another analysis: {key}")
    if (manifest.get("historical_reproduction", {}).get("status") != "ORIGINAL_TRAIN_VALIDATION_BASELINE_REPRODUCED"
            or manifest.get("scalers", {}).get("refitted") is not False
            or manifest.get("scalers", {}).get("state_unchanged_after_transform") is not True
            or manifest.get("old_test_arrays_loaded") is not False):
        raise EvaluationError("Historical reconstruction/scaler provenance is incomplete.")
    period = manifest.get("period", {})
    design = plan["design"]
    if (period.get("requested_start") != "2026-01-01" or period.get("requested_end_inclusive") != "2026-10-28"
            or period.get("n_sequences", 0) < max(150, design["minimum_new_target_rows"])):
        raise EvaluationError("Prepared target period or coverage differs.")
    observed = manifest.get("calendar", {}).get("target_observed_counts", {})
    if set(observed) != set(ASSETS) or any(observed[a] < max(150, design["minimum_observed_price_rows_per_asset_in_new_period"]) for a in ASSETS):
        raise EvaluationError("The observed-price coverage gate has not passed.")
    acquisition = child(root, f"{ACQUISITION}/acquisition_record.json")
    raw = child(root, f"{ACQUISITION}/raw_prices.csv")
    record, acquired = prep.verify_acquisition(raw, acquisition)
    if (record.get("analysis_lock_sha256") != evidence["analysis_lock_sha256"]
            or record.get("protocol_plan_sha256")!=PLAN_SHA
            or record.get("implementation_sha256")!=evidence["implementation_sha256"]):
        raise EvaluationError("Price acquisition did not record this pre-existing analysis lock.")
    for key, expected in acquired.items():
        if identities.get(key) != expected:
            raise EvaluationError("Prepared raw acquisition identity differs.")
    artifacts = manifest.get("artifacts", {})
    required = {"X_external.npy", "y_external_raw.npy", "y_external_scaled.npy",
                "anchor_dates_external.npy", "target_realization_dates_external.npy",
                "target_missing_flags_external.npy", "target_ffill_flags_external.npy"}
    if not isinstance(artifacts, dict) or not required.issubset(artifacts):
        raise EvaluationError("Prepared artifact identities are incomplete.")
    for name, identity in artifacts.items():
        if Path(name).name != name:
            raise EvaluationError("Prepared artifact names must be basenames.")
        prep.verify_identity(child(output, name), identity)
    if {p.name for p in output.iterdir()} != set(artifacts) | {"prepared_manifest_v1.json"}:
        raise EvaluationError("Prepared output contains unmanifested artifacts.")
    return manifest


def load_prepared(root, manifest):
    import numpy as np
    result = {}
    for name, identity in manifest["artifacts"].items():
        value = np.load(BytesIO(checked_bytes(child(root / PREPARED, name), identity)), allow_pickle=False)
        if list(value.shape) != identity["shape"] or str(value.dtype) != identity["dtype"]:
            raise EvaluationError("Prepared array shape/dtype differs from its manifest.")
        result[name] = value
    n = manifest["period"]["n_sequences"]
    for name, shape in (("X_external.npy", (n,10,8)), ("y_external_raw.npy", (n,8)),
                        ("y_external_scaled.npy", (n,8))):
        v = result[name]
        if v.shape != shape or v.dtype != np.float32 or not np.isfinite(v).all():
            raise EvaluationError(f"Invalid finite float32 external array: {name}")
    dates = np.asarray(result["target_realization_dates_external.npy"], dtype="datetime64[D]")
    anchors = np.asarray(result["anchor_dates_external.npy"], dtype="datetime64[D]")
    if (dates.shape != (n,) or anchors.shape != (n,) or np.isnat(dates).any() or np.isnat(anchors).any()
            or not np.all(np.diff(dates) > np.timedelta64(0,"D")) or not np.all(anchors < dates)
            or str(dates[0]) < "2026-01-01" or str(dates[-1]) > "2026-10-28"):
        raise EvaluationError("Prepared external chronology differs.")
    return result


def extract_joint(namespace, source):
    payload=checked_bytes(source, {"sha256": SOURCE_06_SHA})
    tree = ast.parse(payload.decode("utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FullSharingMTL"]
    if len(nodes) != 1:
        raise EvaluationError("The archived full-sharing class is not uniquely defined.")
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), "exec"), namespace)


def build_model(record, namespace, torch, root, *, task=None):
    import numpy as np
    if task is None:
        arm = record["arm"]
        expected = {"NS_base": ("NoSharing",128,54536), "FS_same_width": ("FullSharingMTL",128,28456),
                    "FS_budget_control": ("FullSharingMTL",329,54586)}[arm["id"]]
        if (arm["architecture"],arm["d_ff"],arm["parameters"]) != expected:
            raise EvaluationError("An original sharing architecture changed.")
        model = namespace[arm["architecture"]](n_features=8,lookback=10,d_model=32,n_head=4,
                    n_layers=2,d_ff=arm["d_ff"],dropout=0.1)
        checkpoint = child(OLD_RUN, record["best_checkpoint"]["path"])
        identity = record["best_checkpoint"]
        metadata = {"arm": arm, "seed": record["seed"], "epoch": record["best_epoch"],
                    "source_sha256": SOURCE_06_SHA, "input_manifest_sha256":OLD_TRAINING_SHA,
                    "validation_score":record["best_validation_score"]}
        parameters = expected[2]
    else:
        model = namespace["SingleTaskTransformer"](n_features=8,lookback=10)
        checkpoint = child(root, record["path"])
        identity = record
        metadata = {"task": task,"seed":record["seed"],"selected_epoch":record["selected_epoch"],
                    "architecture":"SingleTaskTransformer","parameters":27268,"plan_sha256":PLAN_SHA,
                    "source_06_sha256":SOURCE_06_SHA,"source_08B_sha256":SOURCE_08_SHA,
                    "selected_validation_task_score":record["selected_validation_task_score"]}
        parameters = 27268
    payload = torch.load(BytesIO(checked_bytes(checkpoint,identity)),map_location="cpu",weights_only=True)
    if not isinstance(payload,dict) or any(payload.get(k) != v for k,v in metadata.items()):
        raise EvaluationError("A selected checkpoint's internal metadata differs.")
    if sum(p.numel() for p in model.parameters()) != parameters:
        raise EvaluationError("Model parameter-count check failed.")
    model.load_state_dict(payload["model_state_dict"],strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise EvaluationError("Nonfinite selected model state.")
    model.to(device="cuda",dtype=torch.float32).eval()
    return model


def predict(model, features, scaler, torch, task=None):
    import numpy as np
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0,len(features),64):
            batch = torch.as_tensor(np.asarray(features[start:start+64]),dtype=torch.float32,device="cuda")
            value = model(batch)
            if value.ndim != 2 or value.shape[1] != (8 if task is None else 4) or not torch.isfinite(value).all():
                raise EvaluationError("Nonfinite or malformed selected-model prediction.")
            predictions.append(value.cpu().numpy())
    scaled = np.concatenate(predictions)
    if task is None:
        raw = scaler.inverse_transform(scaled)
    else:
        columns = slice(0,4) if task == "return" else slice(4,8)
        padded = np.zeros((len(scaled),8),dtype=scaled.dtype)
        padded[:,columns] = scaled
        raw = scaler.inverse_transform(padded)[:,columns]
    if raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise EvaluationError("Inverse-scaled prediction must be finite float32.")
    return raw


def replay_training(root, old, stl, completed, runner, namespace, data, scaler, torch):
    """Replay all15selected checkpoints on old train/val before external inference."""
    import numpy as np
    models, metrics, rows = {}, {}, []
    for record in old["completed_runs"]:
        arm, seed = record["arm"]["id"], record["seed"]
        model = build_model(record,namespace,torch,root)
        for split,label in (("train","train"),("val","validation")):
            raw = predict(model,data["X_"+split],scaler,torch)
            archived = record["best_train_validation_metrics"][label]
            for task in TASKS:
                sl = slice(0,4) if task == "return" else slice(4,8)
                actual = runner.task_metrics(data["y_raw_"+split][:,sl],raw[:,sl],
                        data[task+"_denominators"],task,namespace)
                key = "return_mae" if task == "return" else "volatility_pinball"
                ratio = "avg_return_ratio" if task == "return" else "avg_vol_ratio"
                expected = {"primary_loss":archived[key],"task_score":archived[ratio]}
                for metric in expected:
                    if not np.isclose(actual[metric],expected[metric],rtol=REL_TOL,atol=ABS_TOL):
                        raise EvaluationError(f"Original train/val replay failed: {arm}/{seed}/{label}/{task}/{metric}")
                metrics[arm,seed,task,label] = actual["primary_loss"]
                rows.append({"arm":arm,"seed":seed,"task":task,"split":label,
                    "expected_primary_loss":expected["primary_loss"],"replayed_primary_loss":actual["primary_loss"],
                    "expected_task_score":expected["task_score"],"replayed_task_score":actual["task_score"],"passed":True})
        models[arm,seed,None] = model.cpu()
        print(f"train/validation reproduced {arm} seed={seed}; external predictions unopened",flush=True)
    for selected in stl["checkpoints"]:
        task,seed = selected["task"],selected["seed"]
        record = completed[task,seed]
        expected = json.loads(checked_bytes(child(root,record["artifacts"]["selected_metrics"]["path"]),
                                            record["artifacts"]["selected_metrics"]))
        model = build_model(selected,namespace,torch,root,task=task)
        for split,label in (("train","train"),("val","validation")):
            actual = runner.evaluate_task(model,data["X_"+split],data["y_raw_"+split],scaler,
                                            data[task+"_denominators"],task,torch,namespace)
            for metric in ("primary_loss","task_score","per_asset_primary_losses"):
                if not np.allclose(actual[metric],expected[label][metric],rtol=REL_TOL,atol=ABS_TOL):
                    raise EvaluationError(f"STL train/val replay failed: {task}/{seed}/{label}/{metric}")
            metrics["STL_pair",seed,task,label] = actual["primary_loss"]
            rows.append({"arm":"STL_pair","seed":seed,"task":task,"split":label,
                "expected_primary_loss":expected[label]["primary_loss"],"replayed_primary_loss":actual["primary_loss"],
                "expected_task_score":expected[label]["task_score"],"replayed_task_score":actual["task_score"],"passed":True})
        models["STL_pair",seed,task] = model.cpu()
        print(f"train/validation reproduced STL {task} seed={seed}; external predictions unopened",flush=True)
    if len(models)!=15 or len(rows)!=48 or len(metrics)!=48:
        raise EvaluationError("Every selected model's two-task training/validation replay is required.")
    return models,metrics,rows


def learning_curves(root, old, stl, completed):
    rows=[]
    for job in old["completed_runs"]:
        content=checked_bytes(child(OLD_RUN,job["history"]["path"]),job["history"]).decode("utf-8")
        history=list(csv.DictReader(content.splitlines()))
        if len(history)!=100 or [int(r["epoch"]) for r in history]!=list(range(1,101)):
            raise EvaluationError("Original 100-epoch history is incomplete.")
        if min(history,key=lambda r:float(r["validation_score"]))["epoch"] != str(job["best_epoch"]):
            raise EvaluationError("Original checkpoint is not the first minimum recorded validation score.")
        for source in history:
            for task in TASKS:
                train_key="train_eval_return_mae" if task=="return" else "train_eval_volatility_pinball"
                val_key="val_return_mae" if task=="return" else "val_volatility_pinball"
                rows.append({"arm":job["arm"]["id"],"seed":job["seed"],"task":task,
                    "epoch":int(source["epoch"]),"selected_epoch":job["best_epoch"],
                    "train_primary_loss":float(source[train_key]),"validation_primary_loss":float(source[val_key]),
                    "checkpoint_selection_score":float(source["validation_score"]),
                    "selection_score_scope":"joint normalized task score"})
    for selected in stl["checkpoints"]:
        record=completed[selected["task"],selected["seed"]]
        item=record["artifacts"]["history"]
        history=list(csv.DictReader(checked_bytes(child(root,item["path"]),item).decode("utf-8").splitlines()))
        if len(history)!=100 or [int(r["epoch"]) for r in history]!=list(range(1,101)):
            raise EvaluationError("STL 100-epoch history is incomplete.")
        if min(history,key=lambda r:float(r["validation_task_score"]))["epoch"]!=str(selected["selected_epoch"]):
            raise EvaluationError("STL checkpoint is not the first task-validation-score minimum.")
        for source in history:
            rows.append({"arm":"STL_pair","seed":selected["seed"],"task":selected["task"],
                "epoch":int(source["epoch"]),"selected_epoch":selected["selected_epoch"],
                "train_primary_loss":float(source["train_primary_loss"]),
                "validation_primary_loss":float(source["validation_primary_loss"]),
                "checkpoint_selection_score":float(source["validation_task_score"]),
                "selection_score_scope":"individual normalized task score"})
    if len(rows)!=2400 or any(not math.isfinite(r[k]) for r in rows for k in (
            "train_primary_loss","validation_primary_loss","checkpoint_selection_score")):
        raise EvaluationError("Learning curves are incomplete or nonfinite.")
    return rows


def descriptive_rows(losses, metrics):
    import numpy as np
    per_seed,aggregate,gaps=[],[],[]
    for arm in ARMS:
        for task in TASKS:
            values=losses[arm,task]
            for i,seed in enumerate(SEEDS):
                for j,asset in enumerate((*ASSETS,"four_asset_mean")):
                    daily=values[i,:,j] if j<4 else values[i].mean(axis=1)
                    per_seed.append({"arm":arm,"task":task,"seed":seed,"asset":asset,
                        "n_dates":len(daily),"external_primary_loss":float(daily.mean())})
                test=float(values[i].mean())
                train=metrics[arm,seed,task,"train"]
                val=metrics[arm,seed,task,"validation"]
                gaps.append({"arm":arm,"task":task,"seed":seed,"train_primary_loss":train,
                    "validation_primary_loss":val,"external_primary_loss":test,
                    "validation_minus_train_gap":val-train,"external_minus_train_gap":test-train,
                    "external_minus_validation_gap":test-val,"inference":"descriptive_only_three_seeds"})
            for j,asset in enumerate((*ASSETS,"four_asset_mean")):
                seed_means=values[:,:,j].mean(axis=1) if j<4 else values.mean(axis=(1,2))
                aggregate.append({"arm":arm,"task":task,"asset":asset,"n_seeds":3,
                    "n_dates":values.shape[1],"seed_mean_primary_loss":float(seed_means.mean()),
                    "seed_sample_sd_primary_loss":float(seed_means.std(ddof=1)),"sd_denominator":"n_seeds-1"})
    return per_seed,aggregate,gaps


def csv_write(path, rows):
    if not rows:
        raise EvaluationError("Cannot publish an empty planned analysis table.")
    columns=list(rows[0])
    if any(list(row)!=columns for row in rows):
        raise EvaluationError("Analysis table rows have incompatible fields.")
    with Path(path).open("x",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=columns)
        writer.writeheader();writer.writerows(rows)


def existing_output(output, identities):
    manifest=json_read(child(output,"evaluation_manifest_v1.json"))
    if manifest.get("status")!="H2_EXTERNAL_EVALUATION_COMPLETE" or manifest.get("input_identities")!=identities:
        raise EvaluationError("An existing evaluation has different identities; retain it without overwrite.")
    artifacts=manifest.get("artifacts",{})
    if not isinstance(artifacts,dict) or not artifacts:
        raise EvaluationError("Existing evaluation has no verifiable artifacts.")
    for name,identity in artifacts.items():
        if Path(name).name!=name:
            raise EvaluationError("An evaluation artifact path is unsafe.")
        checked_bytes(child(output,name),identity)
    if {p.name for p in output.iterdir()}!=set(artifacts)|{"evaluation_manifest_v1.json"}:
        raise EvaluationError("Existing evaluation has unmanifested files.")
    return manifest


def publish(output, identities, arrays, tables, report):
    import numpy as np
    no_symlinks(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    guard=output.parent/".evaluate_external_h2_v1.lock"
    try:
        fd=os.open(guard,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError as exc:
        raise EvaluationError("Another or interrupted evaluation lock exists; no overwrite.") from exc
    os.close(fd)
    stage=None
    try:
        if output.exists():
            return existing_output(output,identities),True
        stage=Path(tempfile.mkdtemp(prefix=".evaluation_v1-stage-",dir=output.parent))
        for name,rows in tables.items():
            csv_write(stage/name,rows)
        if any(v.dtype.hasobject for v in arrays.values()):
            raise EvaluationError("Object arrays are forbidden in evaluation outputs.")
        np.savez_compressed(stage/"predictions_and_losses_v1.npz",**arrays)
        (stage/"analysis_report_v1.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        artifacts={p.name:{"sha256":digest(p),"size_bytes":p.stat().st_size} for p in stage.iterdir()}
        manifest={"status":"H2_EXTERNAL_EVALUATION_COMPLETE","created_at_utc":datetime.now(timezone.utc).isoformat(),
            "input_identities":identities,"artifacts":artifacts,"n_selected_models":15,
            "n_arm_seed_predictions":12,"n_planned_contrasts":8,"n_dm_tests_including_sensitivity":16,
            "training_started":False,"old_test_arrays_loaded":False,"old_project_writes":False}
        (stage/"evaluation_manifest_v1.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        if output.exists():
            raise EvaluationError("Evaluation output appeared concurrently; no overwrite.")
        os.rename(stage,output);stage=None
        return manifest,False
    finally:
        if stage is not None:
            shutil.rmtree(stage)
        guard.unlink()


def evaluate(root):
    import numpy as np
    root,lock,evidence,prep,runner=bootstrap(root)
    old,stl,completed,auxiliary=frozen_auxiliary(root,prep,evidence)
    plan=prep.read_json(root/"protocol_plan_v1.json")
    prepared=verify_prepared(root,prep,evidence,plan)
    identities={**evidence,"prepared_manifest_sha256":digest(root/PREPARED/"prepared_manifest_v1.json"),
        "auxiliary_training_artifacts":auxiliary,"evaluation_implementation_sha256":digest(Path(__file__).resolve()),
        "old_test_arrays_loaded":False}
    output=child(root,OUTPUT,must_exist=False)
    if output.exists():
        return existing_output(output,identities),True
    # Report quality and missingness before producing or printing model metrics.
    print(json.dumps({"status":"EXTERNAL_QUALITY_GATES_VERIFIED_BEFORE_MODEL_METRICS",
        "period":prepared["period"],"calendar":prepared["calendar"],
        "price_basis_overlap_audit":prepared["price_basis_overlap_audit"],
        "historical_reproduction":prepared["historical_reproduction"],
        "checkpoint_payloads_opened":False,"external_arrays_loaded":False},ensure_ascii=False,allow_nan=False),flush=True)
    torch,runtime=runner.configure_runtime()
    namespace=runner.extract_definitions(child(root/"inputs",SOURCE_06),child(root/"inputs",SOURCE_08),torch)
    extract_joint(namespace,child(root/"inputs",SOURCE_06))
    data,scaler=runner.load_train_validation(root)
    curves=learning_curves(root,old,stl,completed)
    models,metrics,replays=replay_training(root,old,stl,completed,runner,namespace,data,scaler,torch)
    external=load_prepared(root,prepared)
    restored=scaler.inverse_transform(external["y_external_scaled.npy"].copy())
    if not np.allclose(restored,external["y_external_raw.npy"],rtol=REL_TOL,atol=ABS_TOL):
        raise EvaluationError("External target scaler round trip differs.")
    truth=external["y_external_raw.npy"]
    arrays={"raw_truth":truth,"anchor_dates":external["anchor_dates_external.npy"],
        "target_realization_dates":external["target_realization_dates_external.npy"],
        "target_missing_flags":external["target_missing_flags_external.npy"],
        "target_ffill_flags":external["target_ffill_flags_external.npy"],
        "seeds":np.asarray(SEEDS,dtype=np.int64),"assets":np.asarray(ASSETS),"arms":np.asarray(ARMS)}
    losses={}
    for arm in ARMS:
        predictions=[]
        for seed in SEEDS:
            if arm=="STL_pair":
                raw=np.empty_like(truth)
                for task,sl in (("return",slice(0,4)),("volatility",slice(4,8))):
                    model=models[arm,seed,task].cuda().eval()
                    raw[:,sl]=predict(model,external["X_external.npy"],scaler,torch,task)
                    model.cpu()
            else:
                model=models[arm,seed,None].cuda().eval()
                raw=predict(model,external["X_external.npy"],scaler,torch)
                model.cpu()
            predictions.append(raw)
            print(f"external evaluated {arm} seed={seed}",flush=True)
        prediction_array=np.stack(predictions)
        arrays[f"{arm}__raw_predictions"]=prediction_array
        task_values=[loss_arrays(truth,v) for v in prediction_array]
        for task in TASKS:
            values=np.stack([v[task] for v in task_values])
            losses[arm,task]=values
            arrays[f"{arm}__{task}__per_asset_losses"]=values
    comparisons_rows,daily=comparisons(losses)
    per_seed,aggregate,gaps=descriptive_rows(losses,metrics)
    gap_summary=[]
    for arm in ARMS:
        for task in TASKS:
            selected=[row for row in gaps if row["arm"]==arm and row["task"]==task]
            summary={"arm":arm,"task":task,"n_seeds":3,"inference":"descriptive_only_three_seeds"}
            for key in ("train_primary_loss","validation_primary_loss","external_primary_loss",
                        "validation_minus_train_gap","external_minus_train_gap","external_minus_validation_gap"):
                values=np.array([row[key] for row in selected],dtype=np.float64)
                summary[key+"_seed_mean"]=float(values.mean())
                summary[key+"_seed_sample_sd"]=float(values.std(ddof=1))
            gap_summary.append(summary)
    daily_rows=[]
    dates=external["target_realization_dates_external.npy"]
    for i,date in enumerate(dates):
        row={"target_realization_date":str(date)}
        for arm in ARMS:
            for task in TASKS:
                row[f"{arm}__{task}"]=float(daily[arm,task][i])
        daily_rows.append(row)
    report={"assessment_classification":plan["classification"],"runtime_environment":runtime,
        "period":prepared["period"],"calendar":prepared["calendar"],
        "price_basis_overlap_audit":prepared["price_basis_overlap_audit"],
        "aggregation":"Compute each seed's pointwise loss first; average four assets per date and then three seeds; dates*seeds are not independent observations.",
        "precision":"Float32 raw truth/prediction and pointwise subtraction/pinball; float64 loss averaging and statistical inference.",
        "loss_direction":"Control loss minus full-sharing loss; positive favors sharing.",
        "statistics":{"primary_lag":19,"sensitivity_lag":0,"HAC":"Bartlett weights 1-k/(L+1); gamma_k uses denominator T",
            "HLN":"h=1 factor sqrt((T-1)/T)","reference":"two-sided Student t, df=T-1",
            "multiplicity":"Separate Holm family of four comparisons for each primary/secondary family and each lag; lag0 never replaces prespecified lag19 decisions.",
            "undefined_test":"Undefined zero-long-run-variance test is reported with null raw p and conservative Holm input 1, retaining the planned family size.",
            "confidence_intervals":"Pointwise 95% t intervals for mean loss difference; not multiplicity-adjusted."},
        "train_validation_replay":{"all_passed":True,"n_metric_rows":48,"n_selected_models":15,
            "relative_tolerance":REL_TOL,"absolute_tolerance":ABS_TOL,"old_test_arrays_loaded":False},
        "limitations":["2026 exposure was not found in inspected records; deleted, unsaved or inaccessible experiments cannot be excluded.",
            "The independent STL training control changes optimizer, gradient clipping and task-specific checkpoint selection as well as sharing.",
            "Training/validation/new-test gaps and three-seed variation are descriptive; distribution shift and underfitting can affect the gaps.",
            "The paired loss-series DM tests do not test the generalization gap or establish a causal overfitting mechanism.",
            "A gain in only one task does not support the claim that both tasks improve; report every arm and seed.",
            "The 150-row data coverage threshold is not a statistical power guarantee."],
        "comparisons":comparisons_rows,"training_started":False,"old_project_writes":False,"old_test_arrays_loaded":False}
    # Recheck immutable analysis, checkpoints and prepared bytes after inference.
    fresh_lock,fresh_evidence=prep.verify_analysis_lock(root)
    if fresh_lock!=lock or fresh_evidence!=evidence:
        raise EvaluationError("Analysis identities changed during evaluation.")
    runner.read_evidence(root)
    _,_,_,fresh_aux=frozen_auxiliary(root,prep,evidence)
    if fresh_aux!=auxiliary or digest(root/PREPARED/"prepared_manifest_v1.json")!=identities["prepared_manifest_sha256"]:
        raise EvaluationError("Training/prepared identities changed during evaluation.")
    verify_prepared(root,prep,evidence,plan)
    return publish(output,identities,arrays,{"dm_holm_comparisons_v1.csv":comparisons_rows,
        "primary_losses_by_seed_and_asset_v1.csv":per_seed,"primary_losses_seed_summary_v1.csv":aggregate,
        "generalization_gaps_descriptive_v1.csv":gaps,"learning_curves_v1.csv":curves,
        "generalization_gaps_seed_summary_v1.csv":gap_summary,
        "train_validation_replay_v1.csv":replays,"daily_seed_mean_primary_losses_v1.csv":daily_rows},report)


def self_test():
    """Independent analytic/synthetic checks; never open a research artifact."""
    import numpy as np
    from scipy.stats import t as student_t
    checks=[]
    d=np.array([1.,2.,4.,8.,16.])
    lag0=dm_hln_bartlett(d,0)
    ordinary=float(d.mean()/(d.std(ddof=1)/math.sqrt(len(d))))
    np.testing.assert_allclose(lag0["dm_hln_t"],ordinary,rtol=1e-14)
    np.testing.assert_allclose(lag0["p_raw"],2*student_t.sf(abs(ordinary),4),rtol=1e-14)
    checks.append("Lag0 h1 equals ordinary one-sample t statistic with sample variance")
    # Matrix Bartlett covariance is an independent expression of the gamma sum.
    n=200;lag=19;rng=np.random.default_rng(4567);d=rng.normal(.1,.8,n)
    weights=np.maximum(0,1-np.abs(np.arange(n)[:,None]-np.arange(n)[None,:])/(lag+1))
    centered=d-d.mean();quadratic=float(centered@weights@centered/n)
    result=dm_hln_bartlett(d,lag)
    np.testing.assert_allclose(result["long_run_variance"],quadratic,rtol=1e-14)
    np.testing.assert_allclose(dm_hln_bartlett(-d,lag)["dm_hln_t"],-result["dm_hln_t"],rtol=1e-14)
    np.testing.assert_allclose(dm_hln_bartlett(-d,lag)["p_raw"],result["p_raw"],rtol=1e-14)
    checks.append("Bartlett gamma sum equals full covariance quadratic form; direction symmetry")
    np.testing.assert_allclose(holm_adjust([.01,.04,.03,.2]),[.04,.09,.09,.2])
    np.testing.assert_allclose(holm_adjust([.01,None,.03,.2]),[.04,1.,.09,.4])
    checks.append("Hand-calculated four-test Holm family and conservative undefined slot")
    assert dm_hln_bartlett(np.zeros(200),19)["p_raw"]==1
    assert dm_hln_bartlett(np.ones(200),19)["p_raw"] is None
    checks.append("Identical and zero-variance nonzero differences do not create false significance")
    truth=np.zeros((150,8),dtype=np.float32)
    pred=np.ones_like(truth)
    ll=loss_arrays(truth,pred)
    assert np.array_equal(ll["return"],np.ones((150,4)))
    assert np.array_equal(ll["volatility"],np.full((150,4),.5))
    checks.append("Raw per-asset MAE and tau0.5 pinball scale")
    synthetic={(arm,task):np.full((3,150,4),2.,dtype=np.float64) for arm in ARMS for task in TASKS}
    synthetic["FS_same_width","return"][:]=1.
    synthetic["FS_budget_control","volatility"][:]=3.
    rows,daily=comparisons(synthetic)
    assert len(rows)==16
    for row in rows:
        if row["sharing_arm"]=="FS_same_width" and row["task"]=="return":
            assert row["mean_control_minus_sharing_loss"]==1.
        if row["sharing_arm"]=="FS_budget_control" and row["task"]=="volatility":
            assert row["mean_control_minus_sharing_loss"]==-1.
        assert row["family_size"]==4 and row["n_dates"]==150
    checks.append("All16 tests, four-test family retention, signs and dates-only sample count")
    # Opposing seed forecasts have positive mean seed loss despite zero ensemble
    # forecast error. This explicitly prevents the wrong estimand substitution.
    predictions=[np.full_like(truth,v) for v in (-1.,0.,1.)]
    seed_loss=np.mean([loss_arrays(truth,p)["return"] for p in predictions],axis=0)
    ensemble_loss=loss_arrays(truth,np.mean(predictions,axis=0))["return"]
    assert float(seed_loss.mean())>0 and float(ensemble_loss.mean())==0
    checks.append("Mean of seed losses is distinguished from mean-forecast loss")
    rejected=0
    for action in (lambda:dm_hln_bartlett(np.array([1.,np.nan,2.]),0),
                   lambda:dm_hln_bartlett(np.ones(150),150),
                   lambda:holm_adjust([-.1,.2]),
                   lambda:loss_arrays(truth.astype(np.float64),pred.astype(np.float64)),
                   lambda:comparisons({**synthetic,("NS_base","return"):np.zeros((3,149,4))})):
        try:action()
        except EvaluationError:rejected+=1
        else:raise AssertionError("An invalid scientific input was accepted")
    checks.append("Nonfinite, invalid lag/probability/dtype/alignment inputs rejected")
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        try:bootstrap(root)
        except EvaluationError:rejected+=1
        else:raise AssertionError("Missing final lock accepted")
        try:child(root,"../escape",must_exist=False)
        except EvaluationError:rejected+=1
        else:raise AssertionError("Path escape accepted")
        (root/"link").symlink_to(root,target_is_directory=True)
        try:child(root,"link/file",must_exist=False)
        except EvaluationError:rejected+=1
        else:raise AssertionError("Symlink accepted")
        output=root/"new_period/evaluation_v1"
        manifest,reused=publish(output,{"test":"synthetic"},{"a":np.arange(4)},
                               {"synthetic.csv":[{"x":1,"y":2}]},{"synthetic":True})
        assert not reused
        again,reused=publish(output,{"test":"synthetic"},{"a":np.arange(4)},
                               {"synthetic.csv":[{"x":1,"y":2}]},{"synthetic":True})
        assert reused and again==manifest
        try:existing_output(output,{"test":"changed"})
        except EvaluationError:rejected+=1
        else:raise AssertionError("Changed-identity overwrite accepted")
        with (output/"synthetic.csv").open("ab") as handle:handle.write(b"tamper")
        try:existing_output(output,{"test":"synthetic"})
        except EvaluationError:rejected+=1
        else:raise AssertionError("Changed-output accepted")
    checks.append("Missing lock/path escape/symlink rejected; identical output reuse and tamper rejection")
    return {"status":"SYNTHETIC_EXTERNAL_EVALUATION_SELF_TEST_PASS","checks":checks,
            "n_checks":len(checks),"n_invalid_cases_rejected":rejected,
            "real_prices_accessed":False,"models_trained":False,"torch_required":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(),ensure_ascii=False,allow_nan=False));return
    result,reused=evaluate(args.project_root)
    print(json.dumps({"status":result["status"],"reused_identical_evaluation":reused,
        "output":str(Path(args.project_root)/OUTPUT),"n_selected_models":15,
        "n_planned_contrasts":8,"n_dm_tests_including_sensitivity":16,
        "old_test_arrays_loaded":False,"training_started":False,"old_project_writes":False},
        ensure_ascii=False,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
