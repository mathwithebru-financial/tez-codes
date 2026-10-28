#!/usr/bin/env python3
"""Evaluate nine frozen H2 checkpoints in a new, isolated output directory.

This program never selects models, trains, or imports the archived pipeline.
Test arrays are opened only after verifying the full pretraining input lock,
all nine selected checkpoint hashes, and the frozen completion manifest.
The old test period has already been examined: this is an exploratory supplement.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys

sys.dont_write_bytecode = True

SOURCE_SHA = "35de2ee398699003dfef6be36b70c112fb2c0d1b1e9577cbf64bef58877e16d8"
SEEDS = (123, 777, 2026)
ASSETS = ("BIST100", "USDTRY", "EURTRY", "GOLD")
ARMS = (
    {"id": "NS_base", "architecture": "NoSharing", "d_ff": 128, "parameters": 54536},
    {"id": "FS_same_width", "architecture": "FullSharingMTL", "d_ff": 128, "parameters": 28456},
    {"id": "FS_budget_control", "architecture": "FullSharingMTL", "d_ff": 329, "parameters": 54586},
)
TASKS = ("return", "volatility")
MODEL_SYMBOLS = {"TransformerBlock", "make_head", "FullSharingMTL", "NoSharing"}
SEQ_PREFIX = "data/sequences/baseline/lb10/"
REQUIRED_INPUTS = {
    "scripts/06_best_model_multiseed_v4.py", "config/schema_v4.json",
    "data/raw/raw_prices.csv", "data/sequences/baseline/scalers.pkl",
    "data/processed/selection_baseline_denominators_v4.json",
} | {SEQ_PREFIX + name + "_" + split + ".npy"
     for split in ("train", "val", "test") for name in ("X", "y")} | {
    SEQ_PREFIX + name + "_" + split + ".npy"
    for split in ("train", "val", "test")
    for name in ("anchor_dates", "target_realization_dates")
} | {SEQ_PREFIX + "y_" + split + "_raw.npy" for split in ("train", "val", "test")}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing to write an empty result table.")
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_child(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root or not path.is_file():
        raise ValueError(f"Missing file or unsafe relative path: {relative}")
    return path


def verify_identities(root, identities):
    missing = REQUIRED_INPUTS - set(identities)
    if missing:
        raise ValueError(f"Pretraining input lock omitted required files: {sorted(missing)}")
    for relative, record in identities.items():
        path = safe_child(root, relative)
        if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Frozen input changed: {relative}")
    if identities["scripts/06_best_model_multiseed_v4.py"]["sha256"] != SOURCE_SHA:
        raise ValueError("Archived model source identity mismatch.")


def verify_frozen_gate(input_root, run_root):
    """Hash file bytes only. Never deserialize a scaler, checkpoint or test array."""
    lock_path = safe_child(run_root, "locked_training_manifest.json")
    frozen_path = safe_child(run_root, "selected_checkpoints_frozen.json")
    lock, frozen = read_json(lock_path), read_json(frozen_path)
    if lock.get("status") != "LOCKED_BEFORE_TRAINING":
        raise ValueError("Training manifest was not locked before training.")
    if lock.get("execution_mode") != "H2_EXPLORATORY_FIXED_DESIGN" or lock.get("effective_epochs") != 100:
        raise ValueError("Pilot/incomplete training cannot enter H2 evaluation.")
    if frozen.get("status") != "ALL_NINE_CHECKPOINTS_FROZEN_NO_TEST_ACCESS":
        raise ValueError("All nine selected checkpoints must be frozen first.")
    if frozen.get("test_arrays_loaded") is not False or lock.get("test_arrays_loaded") is not False:
        raise ValueError("Training provenance does not certify no test-array access.")
    if frozen.get("locked_training_manifest_sha256") != sha256(lock_path):
        raise ValueError("Frozen completion manifest references another input lock.")
    if frozen.get("paired_seed_batch_orders_equal") is not True:
        raise ValueError("Paired-seed minibatch order audit is missing.")
    protocol = lock["protocol_draft"]
    if protocol.get("protocol_id") != "H2_CONTROLLED_SHARING_V1":
        raise ValueError("Unexpected experiment protocol.")
    if protocol.get("analysis", {}).get("evaluation_implementation_sha256") != sha256(Path(__file__)):
        raise ValueError("This evaluation implementation was not frozen in the pretraining protocol lock.")
    if protocol["training"]["seeds"] != list(SEEDS) or protocol["training"]["epochs"] != 100:
        raise ValueError("Seed/epoch settings differ from the fixed analysis.")
    if protocol["data"]["old_test_already_examined"] is not True:
        raise ValueError("Exploratory status of the old test period must be explicit.")
    implemented = lock.get("implemented_training_settings", {})
    required_settings = {"epochs":100,"batch_size":64,"optimizer":"AdamW","learning_rate":0.001,
                         "weight_decay":0.0001,"gradient_clip_norm":1.0,"dropout":0.1,
                         "return_loss":"scaled MSE","return_weight":0.7,"volatility_loss":"scaled pinball",
                         "volatility_weight":0.3,"tau":0.5,"early_stopping":False}
    if any(implemented.get(k) != v for k,v in required_settings.items()):
        raise ValueError("Implemented training settings differ from the fixed experiment.")
    expected = {(arm["id"], seed) for arm in ARMS for seed in SEEDS}
    jobs = lock.get("jobs", [])
    if len(jobs) != 9 or {(j["arm"]["id"], j["seed"]) for j in jobs} != expected:
        raise ValueError("Pretraining job lock does not contain exactly the nine specified fits.")
    if any(j["arm"] != next(a for a in ARMS if a["id"] == j["arm"]["id"]) for j in jobs):
        raise ValueError("Locked training arm settings differ from the fixed analysis.")
    records = frozen.get("completed_runs", [])
    if len(records) != 9 or {(r["arm"]["id"], r["seed"]) for r in records} != expected:
        raise ValueError("Frozen checkpoint list does not contain exactly the nine specified fits.")
    by_arm = {a["id"]: a for a in ARMS}
    paired_orders = {}
    for record in records:
        arm = by_arm[record["arm"]["id"]]
        if record["arm"] != arm or record.get("status") != "SUCCESS_NO_TEST_ACCESS":
            raise ValueError("Unexpected arm identity or unsuccessful run.")
        if record.get("test_arrays_loaded") is not False or record.get("epochs_ran") != 100:
            raise ValueError("A training fit is incomplete or accessed test arrays.")
        if not 1 <= record["best_epoch"] <= 100:
            raise ValueError("Selected epoch outside fixed training budget.")
        if record["best_train_validation_metrics"]["epoch"] != record["best_epoch"]:
            raise ValueError("Selected-checkpoint metric epoch mismatch.")
        for key in ("best_checkpoint", "final_checkpoint", "history"):
            item = record[key]
            if sha256(safe_child(run_root, item["path"])) != item["sha256"]:
                raise ValueError(f"Frozen {key} identity mismatch: {arm['id']}, seed {record['seed']}")
        # Independently check earliest-minimum checkpoint selection from frozen history.
        with safe_child(run_root, record["history"]["path"]).open(newline="", encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        if [int(r["epoch"]) for r in history] != list(range(1, 101)):
            raise ValueError("Training history must contain every fixed-budget epoch.")
        orders = [r["batch_order_sha256"] for r in history]
        if record["seed"] in paired_orders and paired_orders[record["seed"]] != orders:
            raise ValueError("Paired-seed minibatch histories differ across arms.")
        paired_orders[record["seed"]] = orders
        scores = [float(r["validation_score"]) for r in history]
        if not all(math.isfinite(v) for v in scores):
            raise ValueError("Nonfinite historical validation score.")
        best = min(range(100), key=lambda i: scores[i])
        if best + 1 != record["best_epoch"] or scores[best] != record["best_validation_score"]:
            raise ValueError("Frozen checkpoint violates earliest minimum validation-score rule.")
    verify_identities(input_root, lock["input_identities"])
    return lock, frozen, {"locked_training_manifest_sha256": sha256(lock_path),
                          "selected_checkpoints_frozen_sha256": sha256(frozen_path)}


def hac_hln(differences, lag):
    """Independent biased-autocovariance Bartlett HAC, h=1 HLN, t(T-1)."""
    import numpy as np
    from scipy.stats import t
    d = np.asarray(differences, dtype=np.float64)
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all() or not 0 <= lag < len(d):
        raise ValueError("Invalid loss differential series or HAC lag.")
    n = len(d)
    mean = float(d.mean())
    centered = d - mean
    gamma = np.array([np.dot(centered[k:], centered[:n-k]) / n for k in range(lag + 1)])
    weights = np.array([1.0] + [1.0 - k / (lag + 1) for k in range(1, lag + 1)])
    long_run = float(gamma[0] + 2.0 * np.dot(weights[1:], gamma[1:]))
    if long_run < 0:
        raise ValueError("Negative Bartlett variance; numerical analysis cannot continue.")
    correction = math.sqrt((n - 1) / n)
    if long_run == 0:
        statistic, raw, standard_error, p = None, None, 0.0, 1.0
        policy = "ZERO_VARIANCE_UNTESTABLE_CONSERVATIVE_P1"
    else:
        standard_error = math.sqrt(long_run / n)
        raw = mean / standard_error
        statistic = raw * correction
        p = float(2.0 * t.sf(abs(statistic), df=n - 1))
        policy = "STANDARD_POSITIVE_VARIANCE"
    result = {"n": n, "forecast_horizon_h": 1, "bartlett_max_lag_L": lag,
              "mean_loss_differential": mean, "loss_differential_gamma0": float(gamma[0]),
              "bartlett_long_run_variance": long_run, "bartlett_variance_of_mean": long_run/n,
              "dm_standard_error": standard_error, "dm_raw_statistic": raw,
              "hln_correction_factor": correction, "dm_hln_statistic": statistic,
              "t_reference_df": n-1, "dm_hln_p_t_two_sided": p, "variance_policy_case": policy}
    autocov = [{"lag": k, "autocovariance_denominator_T": float(gamma[k]),
                "bartlett_weight": float(weights[k])} for k in range(lag+1)]
    return result, autocov


def holm(rows, alpha=0.05):
    """One four-comparison family per prespecified lag analysis."""
    ordered = sorted(range(len(rows)), key=lambda i: rows[i]["dm_hln_p_t_two_sided"])
    previous = 0.0
    for position, i in enumerate(ordered):
        row = rows[i]
        adjusted = min(1.0, max(previous, (len(rows)-position) * row["dm_hln_p_t_two_sided"]))
        previous = adjusted
        rejected = adjusted <= alpha and row["variance_policy_case"] == "STANDARD_POSITIVE_VARIANCE"
        direction = "FS_BETTER" if row["mean_loss_differential"] > 0 else "NS_BETTER"
        row.update({"holm_family_size": len(rows), "holm_rank": position+1,
                    "holm_adjusted_p": adjusted, "null_rejected_holm": rejected,
                    "decision": direction if rejected else "NO_SIGNIFICANT_DIFFERENCE"})
    return rows


def statistics_self_test(npz_path, csv_path):
    """Check independent HAC against all old 22 frozen numeric loss series."""
    import numpy as np
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reference = list(csv.DictReader(handle))
    columns = ("mean_loss_differential", "loss_differential_gamma0", "bartlett_long_run_variance",
               "bartlett_variance_of_mean", "dm_raw_statistic", "dm_hln_statistic", "dm_hln_p_t_two_sided")
    maximum = {key: 0.0 for key in columns}
    with np.load(npz_path, allow_pickle=False) as archive:
        for row in reference:
            # Object-valued date metadata is deliberately never deserialized.
            d = archive[row["comparison_id"]]
            actual, _ = hac_hln(d, 19)
            if actual["n"] != 584:
                raise AssertionError("Unexpected reference sample size.")
            for key in columns:
                expected = float(row[key])
                delta = abs(actual[key] - expected)
                maximum[key] = max(maximum[key], delta)
                if not math.isclose(actual[key], expected, rel_tol=1e-11, abs_tol=1e-14):
                    raise AssertionError(f"HAC self-test mismatch: {row['comparison_id']}, {key}")
    if len(reference) != 22:
        raise AssertionError("Expected all 22 frozen reference comparisons.")
    probe = [{"dm_hln_p_t_two_sided": p, "variance_policy_case": "STANDARD_POSITIVE_VARIANCE",
              "mean_loss_differential": 1.0} for p in (0.04, 0.001, 0.02, 0.015)]
    got = [r["holm_adjusted_p"] for r in holm(probe)]
    if not np.allclose(got, [0.045, 0.004, 0.045, 0.045], rtol=0, atol=1e-15):
        raise AssertionError("Holm monotonicity self-test failed.")
    # Test metric arithmetic independently of model/scaler or current test data.
    truth = np.zeros((2, 8)); forecast = np.array([[1.0]*4 + [2.0]*4, [-3.0]*4 + [-4.0]*4])
    loss = task_loss_arrays(truth, forecast)
    if not np.array_equal(loss["return"], [1., 3.]) or not np.array_equal(loss["volatility"], [1., 2.]):
        raise AssertionError("Raw task-loss arithmetic self-test failed.")
    return {"status": "STATISTICS_SELF_TEST_PASS", "reference_comparisons": len(reference),
            "reference_npz_sha256": sha256(npz_path), "reference_csv_sha256": sha256(csv_path),
            "maximum_absolute_delta": maximum, "test_arrays_loaded": False, "file_writes": False}


def source_namespace(root, torch):
    source = safe_child(root, "scripts/06_best_model_multiseed_v4.py")
    if sha256(source) != SOURCE_SHA:
        raise ValueError("Model source changed immediately before AST extraction.")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in MODEL_SYMBOLS]
    if {n.name for n in nodes} != MODEL_SYMBOLS:
        raise ValueError("Archived model definitions missing.")
    namespace = {"torch": torch, "nn": torch.nn}
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


def load_arrays_after_gate(root):
    import numpy as np
    schema = read_json(root / "config/schema_v4.json")
    if schema["data"]["assets"] != list(ASSETS) or schema["features"]["baseline_dim"] != 8:
        raise ValueError("Frozen schema asset/feature order mismatch.")
    folder, data = root / SEQ_PREFIX, {}
    boundaries = {"train": (2714, "2010-02-15", "2020-07-09"),
                  "val": (584, "2020-07-10", "2022-10-05"),
                  "test": (584, "2022-10-06", "2024-12-31")}
    for split, (n, start, end) in boundaries.items():
        for name, shape in [("X_"+split, (n,10,8)), ("y_"+split, (n,8)), ("y_"+split+"_raw", (n,8))]:
            a = np.load(folder / (name+".npy"), allow_pickle=False)
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError(f"Array shape/finite-value check failed: {name}")
            data[name] = a
        for name in ("anchor_dates", "target_realization_dates"):
            dates = np.load(folder / f"{name}_{split}.npy", allow_pickle=False).astype("datetime64[D]")
            if dates.shape != (n,) or np.isnat(dates).any() or not np.all(np.diff(dates) > np.timedelta64(0,"D")):
                raise ValueError(f"Invalid or unordered dates: {name}_{split}")
            data[name+"_"+split] = dates
        target = data["target_realization_dates_"+split]
        if (str(target[0]), str(target[-1])) != (start,end):
            raise ValueError(f"Frozen target-date boundaries differ: {split}")
        if not np.all(target > data["anchor_dates_"+split]):
            raise ValueError(f"Targets must occur after input anchors: {split}")
    # Trusted original pickle, verified against the pretraining SHA lock first.
    with (root / "data/sequences/baseline/scalers.pkl").open("rb") as handle:
        scaler = pickle.load(handle)["y_scaler"]
    for split in boundaries:
        restored = scaler.inverse_transform(data["y_"+split].copy())
        if float(np.max(np.abs(restored-data["y_"+split+"_raw"]))) > 1e-5:
            raise ValueError(f"Frozen scaler inverse-transform check failed: {split}")
    return data, scaler


def task_loss_arrays(truth, prediction):
    import numpy as np
    error = np.asarray(truth, dtype=np.float64)-np.asarray(prediction, dtype=np.float64)
    if error.ndim != 2 or error.shape[1] != 8 or not np.isfinite(error).all():
        raise ValueError("Invalid raw truth or prediction array.")
    return {"return": np.mean(np.abs(error[:,:4]), axis=1),
            "volatility": np.mean(np.maximum(0.5*error[:,4:], -0.5*error[:,4:]), axis=1)}


def predict(model, features, scaler, torch):
    import numpy as np
    model.eval()
    output = []
    with torch.no_grad():
        for begin in range(0, len(features), 64):
            x = torch.as_tensor(features[begin:begin+64], dtype=torch.float32, device="cpu")
            output.append(model(x).cpu().numpy())
    scaled = np.concatenate(output, axis=0)
    raw = scaler.inverse_transform(scaled)
    if raw.shape != (len(features),8) or not np.isfinite(raw).all():
        raise ValueError("Invalid raw prediction.")
    return raw


def evaluate_all(input_root, run_root, output, lock, frozen, gate, self_test):
    import numpy as np
    import torch
    if not torch.__version__.startswith("2.11.") or "+cpu" not in torch.__version__:
        raise ValueError("This locked local evaluation requires PyTorch 2.11 CPU.")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    namespace = source_namespace(input_root, torch)
    data, scaler = load_arrays_after_gate(input_root)
    records = {(r["arm"]["id"],r["seed"]):r for r in frozen["completed_runs"]}
    seed_rows, asset_rows, final_rows, saved, per_arm_losses = [], [], [], {}, {}
    for arm in ARMS:
        per_arm_losses[arm["id"]] = {task:[] for task in TASKS}
        for seed in SEEDS:
            record = records[(arm["id"],seed)]
            with safe_child(run_root,record["history"]["path"]).open(newline="",encoding="utf-8") as handle:
                last = list(csv.DictReader(handle))[-1]
            final_rows.append({"arm":arm["id"],"seed":seed,"epoch":100,
                               "diagnostic_only_not_used_for_selection":True,
                               **{k:float(last[k]) for k in ("train_eval_return_mae","val_return_mae",
                                  "validation_return_gap","train_eval_volatility_pinball",
                                  "val_volatility_pinball","validation_volatility_gap","validation_score")}})
            checkpoint = safe_child(run_root, record["best_checkpoint"]["path"])
            if sha256(checkpoint) != record["best_checkpoint"]["sha256"]:
                raise ValueError("Selected checkpoint changed after gate.")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if payload["arm"] != arm or payload["seed"] != seed or payload["epoch"] != record["best_epoch"]:
                raise ValueError("Selected checkpoint metadata differs from frozen record.")
            if payload["source_sha256"] != SOURCE_SHA or payload["input_manifest_sha256"] != gate["locked_training_manifest_sha256"]:
                raise ValueError("Checkpoint source/input lineage mismatch.")
            if payload["validation_score"] != record["best_validation_score"]:
                raise ValueError("Checkpoint validation-score lineage mismatch.")
            model = namespace[arm["architecture"]](n_features=8,lookback=10,d_model=32,n_head=4,n_layers=2,
                                                   d_ff=arm["d_ff"],dropout=0.1).cpu()
            if sum(p.numel() for p in model.parameters() if p.requires_grad) != arm["parameters"]:
                raise ValueError("Model parameter count mismatch.")
            shared = sum(p.numel() for p in model.pcgrad_groups()["shared"])
            if (arm["id"]=="NS_base" and shared!=0) or (arm["id"]!="NS_base" and shared<=0):
                raise ValueError("Shared-parameter audit failed.")
            model.load_state_dict(payload["model_state_dict"],strict=True)
            losses, raw_predictions = {}, {}
            for split in ("train","val","test"):
                raw = predict(model,data["X_"+split],scaler,torch)
                raw_predictions[split] = raw
                losses[split] = task_loss_arrays(data["y_"+split+"_raw"],raw)
            for task in TASKS:
                metric = "return_mae" if task=="return" else "volatility_pinball"
                train, val, test = [float(losses[s][task].mean()) for s in ("train","val","test")]
                locked_metrics = record["best_train_validation_metrics"]
                deltas = [abs(value-float(locked_metrics[s][metric])) for value,s in [(train,"train"),(val,"validation")]]
                # Independent float64 loss arithmetic differs slightly from archived float32 means.
                if any(not math.isclose(value,float(locked_metrics[s][metric]),rel_tol=1e-5,abs_tol=1e-8)
                       for value,s in [(train,"train"),(val,"validation")]):
                    raise ValueError("Selected train/validation losses do not reproduce frozen metrics.")
                seed_rows.append({"arm":arm["id"],"seed":seed,"task":task,"selected_epoch":record["best_epoch"],
                                  "parameters":arm["parameters"],"train_eval_loss":train,"validation_loss":val,
                                  "test_loss":test,"validation_gap":val-train,"test_gap":test-train,
                                  "max_train_val_reproduction_abs_delta":max(deltas)})
                per_arm_losses[arm["id"]][task].append(losses["test"][task])
                for split in ("train","val","test"):
                    saved[f"loss__{arm['id']}__seed{seed}__{split}__{task}"] = losses[split][task]
            saved[f"prediction_raw__{arm['id']}__seed{seed}__test"] = raw_predictions["test"]
            error = np.asarray(data["y_test_raw"],dtype=np.float64)-raw_predictions["test"].astype(np.float64)
            for j,asset in enumerate(ASSETS):
                asset_rows.append({"arm":arm["id"],"seed":seed,"asset":asset,
                                   "return_mae":float(np.abs(error[:,j]).mean()),
                                   "volatility_pinball":float((0.5*np.abs(error[:,4+j])).mean())})
            print(f"evaluated {arm['id']} seed={seed} selected_epoch={record['best_epoch']}",flush=True)
    summary_rows=[]
    for arm in ARMS:
        for task in TASKS:
            subset=[r for r in seed_rows if r["arm"]==arm["id"] and r["task"]==task]
            row={"arm":arm["id"],"task":task,"n_seeds":3}
            for name in ("train_eval_loss","validation_loss","test_loss","validation_gap","test_gap"):
                values=np.array([r[name] for r in subset])
                row[name+"_mean"]=float(values.mean()); row[name+"_sample_sd"]=float(values.std(ddof=1))
            summary_rows.append(row)
            saved[f"seed_mean_loss__{arm['id']}__test__{task}"]=np.mean(per_arm_losses[arm["id"]][task],axis=0)
    comparisons, autocov_rows=[],[]
    for lag in (19,0):
        family=[]
        for arm in ARMS[1:]:
            for task in TASKS:
                ns=saved[f"seed_mean_loss__NS_base__test__{task}"]
                fs=saved[f"seed_mean_loss__{arm['id']}__test__{task}"]
                difference=ns-fs
                result,autocov=hac_hln(difference,lag)
                row={"comparison_id":f"{arm['id']}__vs__NS_base__{task}","fs_arm":arm["id"],"task":task,
                     "analysis":"PRIMARY_BARTLETT_L19" if lag==19 else "SENSITIVITY_LAG0",
                     "ns_mean_loss":float(ns.mean()),"fs_mean_loss":float(fs.mean()),**result}
                family.append(row)
                for item in autocov:
                    autocov_rows.append({"comparison_id":row["comparison_id"],"analysis":row["analysis"],**item})
                if lag==19:
                    saved[f"difference_NS_minus_FS__{arm['id']}__{task}"]=difference
        comparisons.extend(holm(family))
    contrast_rows=[]
    for arm in ARMS[1:]:
        for seed in SEEDS:
            for task in TASKS:
                ns=next(r for r in seed_rows if r["arm"]=="NS_base" and r["seed"]==seed and r["task"]==task)
                fs=next(r for r in seed_rows if r["arm"]==arm["id"] and r["seed"]==seed and r["task"]==task)
                contrast_rows.append({"fs_arm":arm["id"],"seed":seed,"task":task,
                                      **{f"ns_minus_fs_{name}":ns[name]-fs[name] for name in
                                         ("train_eval_loss","validation_loss","test_loss","validation_gap","test_gap")}})
    contrast_summary=[]
    for arm in ARMS[1:]:
        for task in TASKS:
            rows=[r for r in contrast_rows if r["fs_arm"]==arm["id"] and r["task"]==task]
            row={"fs_arm":arm["id"],"task":task,"n_seeds":3,"inference":"DESCRIPTIVE_ONLY"}
            for name in ("train_eval_loss","validation_loss","test_loss","validation_gap","test_gap"):
                key="ns_minus_fs_"+name
                values=np.array([r[key] for r in rows])
                row[key+"_mean"]=float(values.mean());row[key+"_sample_sd"]=float(values.std(ddof=1))
            contrast_summary.append(row)
    # Recheck before committing numerical artifacts; old inputs are read-only throughout.
    verify_identities(input_root,lock["input_identities"])
    if sha256(run_root/"locked_training_manifest.json")!=gate["locked_training_manifest_sha256"] or \
       sha256(run_root/"selected_checkpoints_frozen.json")!=gate["selected_checkpoints_frozen_sha256"]:
        raise ValueError("A frozen manifest changed during evaluation.")
    for record in frozen["completed_runs"]:
        if sha256(safe_child(run_root,record["best_checkpoint"]["path"]))!=record["best_checkpoint"]["sha256"]:
            raise ValueError("A selected checkpoint changed during evaluation.")
    saved["target_realization_dates_test"]=data["target_realization_dates_test"].astype("U10")
    saved["anchor_dates_test"]=data["anchor_dates_test"].astype("U10")
    saved["truth_raw_test"]=data["y_test_raw"]
    np.savez_compressed(output/"h2_losses_predictions_v1.npz",**saved)
    write_csv(output/"h2_seed_metrics_v1.csv",seed_rows)
    write_csv(output/"h2_arm_summary_v1.csv",summary_rows)
    write_csv(output/"h2_test_asset_metrics_v1.csv",asset_rows)
    write_csv(output/"h2_gap_contrasts_v1.csv",contrast_rows)
    write_csv(output/"h2_gap_contrast_summary_v1.csv",contrast_summary)
    write_csv(output/"h2_final_epoch_train_validation_diagnostics_v1.csv",final_rows)
    write_csv(output/"h2_dm_hln_holm_v1.csv",comparisons)
    write_csv(output/"h2_autocovariances_v1.csv",autocov_rows)
    write_json(output/"statistics_self_test_v1.json",self_test)
    primary=[r for r in comparisons if r["bartlett_max_lag_L"]==19]
    note = ["H2 supplementary exploratory evaluation / H2 tamamlayıcı keşifsel değerlendirme", "",
            "The 584-date old test period had already been examined. These results are supplementary, not a new independent confirmation.",
            "584 tarihli mevcut test dönemi daha önce incelendi. Sonuçlar tamamlayıcıdır; yeni bağımsız doğrulama değildir.", "",
            "Primary date-level losses average each seed's loss across three seeds and four assets. They are not losses of ensemble forecasts.",
            "Birincil tarih bazlı kayıplar, dört varlık ve üç tohumun ayrı kayıplarını ortalar; topluluk tahmini kaybı değildir.",
            "Positive NS-minus-FS differences favor sharing. Two-sided Bartlett-HAC L=19, h=1 HLN t(583), Holm family of four; lag 0 is sensitivity only.", ""]
    for row in primary:
        note.append(f"{row['comparison_id']}: NS-FS={row['mean_loss_differential']:.9g}; Holm p={row['holm_adjusted_p']:.6g}; {row['decision']}.")
    note += ["", "Generalization gaps show held-out loss minus selected-checkpoint train-evaluation loss. A smaller gap alone can reflect underfit or distribution shift; it does not establish pure overfitting reduction or causal mediation.",
             "Genelleme farkı, ayrılmış veri kaybı eksi seçilen kontrol noktasının eğitim-değerlendirme kaybıdır. Küçük fark tek başına aşırı öğrenmenin azaldığını veya nedensel mekanizmayı kanıtlamaz.",
             "Three seeds and approximate capacity matching limit interpretation. All components and seed-level results must be reported. No old scientific result or thesis file was changed."]
    with (output/"H2_analysis_note_v1.txt").open("x",encoding="utf-8") as handle:
        handle.write("\n".join(note)+"\n")
    artifacts={p.name:{"sha256":sha256(p),"size_bytes":p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()}
    summary={"status":"H2_EVALUATION_COMPLETE_EXPLORATORY", "completed_at_utc":datetime.now(timezone.utc).isoformat(),
             "evaluation_implementation_sha256":sha256(Path(__file__)),"provenance":gate,
             "software":{"python":sys.version,"numpy":np.__version__,"torch":torch.__version__},
             "test_arrays_loaded_only_after_gate":True,"n_test_dates":584,"seeds":list(SEEDS),
             "loss_aggregation":"mean of per-seed raw losses, not ensemble-forecast loss",
             "difference_direction":"NS loss minus FS loss; positive favors FS",
             "primary_lag":19,"sensitivity_lag":0,"hln_h":1,"holm_family_size":4,
             "arm_summary":summary_rows,"comparisons":comparisons,"artifacts":artifacts,
             "old_project_writes":False,"test_period_previously_examined":True}
    write_json(output/"h2_evaluation_summary_v1.json",summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root",type=Path)
    parser.add_argument("--run-root",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--verify-only",action="store_true",help="Hash-only gate check; never open test arrays.")
    parser.add_argument("--statistics-self-test",action="store_true",help="Read old numeric reference losses only; write nothing.")
    parser.add_argument("--reference-npz",type=Path,required=True)
    parser.add_argument("--reference-csv",type=Path,required=True)
    args=parser.parse_args()
    checked=statistics_self_test(args.reference_npz,args.reference_csv)
    if args.statistics_self_test:
        print(json.dumps(checked,ensure_ascii=False,allow_nan=False));return
    if args.input_root is None or args.run_root is None:
        parser.error("--input-root and --run-root are required for gate/evaluation.")
    root,run=args.input_root.expanduser().resolve(),args.run_root.expanduser().resolve()
    h2_root=Path(__file__).resolve().parents[1]
    if run==h2_root or not run.is_relative_to(h2_root):
        raise ValueError("Evaluation run root must be a dedicated directory within this H2 experiment root.")
    if run==root or run.is_relative_to(root) or root.is_relative_to(run):
        raise ValueError("H2 run directory must be separate from original input project.")
    lock,frozen,gate=verify_frozen_gate(root,run)
    if args.verify_only:
        print(json.dumps({"status":"ALL_NINE_HASHES_AND_INPUT_LOCK_VERIFIED","test_arrays_loaded":False,**gate}));return
    if args.output is None:
        parser.error("--output is required for evaluation.")
    output=args.output.expanduser().resolve()
    if output==run or not output.is_relative_to(run) or output.exists():
        raise ValueError("Evaluation output must be a nonexistent new directory inside this H2 run root.")
    output.mkdir(parents=False,exist_ok=False)
    result=evaluate_all(root,run,output,lock,frozen,gate,checked)
    print(json.dumps({"status":result["status"],"output":str(output),"comparisons":len(result["comparisons"])}))


if __name__=="__main__":
    main()
