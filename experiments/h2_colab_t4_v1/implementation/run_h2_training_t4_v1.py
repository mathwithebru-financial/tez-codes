#!/usr/bin/env python3
"""Isolated T4 H2 rerun on frozen train/validation inputs; never parse test arrays.

The existing model definitions and ValidationScore functions are extracted from
the hash-verified Stage 06 source without executing its top-level code. All
writes are confined to a newly created run directory outside the frozen input
project. Internal best-checkpoint updates are allowed only within that new run.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import pickle
import platform
import random
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

SOURCE_SHA = "35de2ee398699003dfef6be36b70c112fb2c0d1b1e9577cbf64bef58877e16d8"
RAW_SHA = "ab5f275d38dc98057b1cedcf58019adb26be7402c7ed5ae6ee3d6877b2444893"
ASSETS = ["BIST100", "USDTRY", "EURTRY", "GOLD"]
SEEDS = [123, 777, 2026]
ARMS = [
    {"id": "NS_base", "architecture": "NoSharing", "d_ff": 128, "parameters": 54536},
    {"id": "FS_same_width", "architecture": "FullSharingMTL", "d_ff": 128, "parameters": 28456},
    {"id": "FS_budget_control", "architecture": "FullSharingMTL", "d_ff": 329, "parameters": 54586},
]
SYMBOLS = {"TransformerBlock", "make_head", "FullSharingMTL", "NoSharing",
           "mae_np", "rmse_np", "r2_np", "pinball_np", "compute_raw_metrics",
           "compute_validation_score"}
USED_RELATIVE_INPUTS = [
    "scripts/06_best_model_multiseed_v4.py", "config/schema_v4.json",
    "data/raw/raw_prices.csv", "data/sequences/baseline/scalers.pkl",
    "data/processed/selection_baseline_denominators_v4.json",
    "data/sequences/baseline/lb10/sequence_meta.json",
] + ["data/sequences/baseline/lb10/" + name for name in [
    "X_train.npy", "y_train.npy", "y_train_raw.npy", "X_val.npy", "y_val.npy", "y_val_raw.npy",
    "anchor_dates_train.npy", "target_realization_dates_train.npy",
    "anchor_dates_val.npy", "target_realization_dates_val.npy",
]]
# Freeze the held-out inputs as opaque bytes before any fitting. Their arrays,
# dates and values are deliberately never parsed by this training program.
HELD_OUT_TEST_INPUTS = ["data/sequences/baseline/lb10/" + name for name in [
    "X_test.npy", "y_test.npy", "y_test_raw.npy", "anchor_dates_test.npy",
    "target_realization_dates_test.npy",
]]
_TORCH_CONFIGURED = False
PROTOCOL_ID = "H2_CONTROLLED_SHARING_T4_V1"
EXECUTION_MODE = "H2_T4_EXPLORATORY_FIXED_DESIGN"
RUNTIME_PINS = {
    "python_version": "3.12.13",
    "packages": {"torch": "2.11.0+cu128", "numpy": "2.3.5", "pandas": "2.2.3",
                 "scipy": "1.17.0", "scikit-learn": "1.6.1"},
    "cuda_runtime": "12.8", "gpu_count": 1, "gpu_name": "Tesla T4", "device_capability": [7, 5],
}
DETERMINISM = {
    "cublas_workspace_config": ":4096:8", "deterministic_algorithms": True,
    "cudnn_deterministic": True, "cudnn_benchmark": False,
    "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
    "float32_matmul_precision": "highest", "amp_enabled": False,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def json_clean(value):
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    if exclusive:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(json_clean(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    else:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(json_clean(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)


def input_identities(root):
    records = {}
    for relative in USED_RELATIVE_INPUTS + HELD_OUT_TEST_INPUTS:
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(f"Required frozen input missing or outside project: {relative}")
        records[relative] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    if records["scripts/06_best_model_multiseed_v4.py"]["sha256"] != SOURCE_SHA:
        raise ValueError("Original Stage 06 source identity mismatch.")
    if records["data/raw/raw_prices.csv"]["sha256"] != RAW_SHA:
        raise ValueError("Frozen raw-data identity mismatch.")
    return records


def verify_input_identities(root, records):
    for relative, identity in records.items():
        path = root / relative
        if path.stat().st_size != identity["size_bytes"] or sha256(path) != identity["sha256"]:
            raise ValueError(f"Frozen input changed: {relative}")


def configure_torch(threads):
    global _TORCH_CONFIGURED
    import torch
    torch.set_num_threads(threads)
    # A spawned pool worker may fit several arm/seed pairs. PyTorch permits
    # configuring its inter-op pool only once, before that process does work.
    if not _TORCH_CONFIGURED:
        torch.set_num_interop_threads(1)
        _TORCH_CONFIGURED = True
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return torch


def collect_runtime_environment(threads=1):
    """Collect the actual process environment, without fitting or writing files."""
    torch = configure_torch(threads)
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    count = torch.cuda.device_count()
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    versions = [line.strip() for line in driver.stdout.splitlines() if line.strip()]
    if len(versions) != 1:
        raise ValueError("Cannot record one unambiguous NVIDIA driver version.")
    return {
        "python_version": platform.python_version(),
        "packages": {"torch": str(torch.__version__), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "scikit-learn": sklearn.__version__},
        "cuda_runtime": torch.version.cuda,
        "gpu_count": count,
        "gpu_name": torch.cuda.get_device_name(0) if count else None,
        "device_capability": list(torch.cuda.get_device_capability(0)) if count else None,
        "cudnn_version": torch.backends.cudnn.version(),
        "driver_version": versions[0],
        "determinism": {
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "amp_enabled": torch.is_autocast_enabled("cuda"),
        },
    }


def validate_expected_runtime(expected):
    """Pure validation: a protocol cannot silently substitute another runtime."""
    if set(expected) != set(RUNTIME_PINS) | {"cudnn_version", "driver_version", "determinism"}:
        raise ValueError("Runtime lock fields are incomplete or unexpected.")
    for key, value in RUNTIME_PINS.items():
        if expected[key] != value:
            raise ValueError(f"T4 runtime pin differs: {key}")
    if expected["determinism"] != DETERMINISM:
        raise ValueError("T4 deterministic/FP32 settings differ.")
    if not isinstance(expected["cudnn_version"], int) or isinstance(expected["cudnn_version"], bool) or expected["cudnn_version"] <= 0:
        raise ValueError("Freeze the actual cuDNN version from the environment collector.")
    if not isinstance(expected["driver_version"], str) or not expected["driver_version"].strip():
        raise ValueError("Freeze the actual NVIDIA driver version from the environment collector.")


def validate_runtime_environment(actual, expected):
    validate_expected_runtime(expected)
    if actual != expected:
        different = sorted(key for key in expected if actual.get(key) != expected[key])
        raise ValueError(f"Actual environment differs from the frozen T4 runtime: {different}")


def validate_execution_paths(root, output, protocol):
    project = Path(__file__).resolve().parents[1]
    if project != Path(protocol["project_root"]).expanduser().resolve():
        raise ValueError("This implementation must run inside the locked new T4 project root.")
    if root != (project / "inputs").resolve():
        raise ValueError("Use only the copied frozen inputs inside the new T4 project.")
    if (project / "inputs").is_symlink() or (project / "runs").is_symlink():
        raise ValueError("The copied input and new run parent directories cannot be symlinks.")
    run_parent = (project / "runs").resolve()
    if not run_parent.is_relative_to(project) or not root.is_relative_to(project):
        raise ValueError("Copied inputs and run parent must remain inside the new T4 project.")
    if output.exists() or output == run_parent or not output.is_relative_to(run_parent):
        raise ValueError("Choose a nonexistent dedicated run directory inside the new T4 project's runs/.")
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("The output and frozen input directories must be separate.")
    protected = protocol.get("protected_roots")
    if not isinstance(protected, list) or len(protected) < 2:
        raise ValueError("Explicitly protect the original v4 and CPU H2 project roots.")
    if Path(protocol["source_root"]).expanduser().resolve() not in [Path(p).expanduser().resolve() for p in protected]:
        raise ValueError("The original v4 source root is missing from protected roots.")
    for value in protected:
        old = Path(value).expanduser().resolve()
        if project == old or project.is_relative_to(old) or old.is_relative_to(project):
            raise ValueError("The new T4 project must be separate from every protected old project.")
        if output == old or output.is_relative_to(old) or old.is_relative_to(output):
            raise ValueError("The original v4 and CPU H2 project roots are protected.")


def verify_implementation_identities(identities):
    names = {"run_h2_training_t4_v1.py", "evaluate_h2_t4_v1.py"}
    if set(identities) != names:
        raise ValueError("Freeze exactly the T4 training and evaluation implementations.")
    for name, expected in identities.items():
        path = Path(__file__).resolve().parent / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Frozen T4 implementation changed: {name}")


def validate_completed_history(rows, record, epochs):
    if [int(row["epoch"]) for row in rows] != list(range(1, epochs + 1)):
        raise ValueError("Every planned epoch must appear exactly once before checkpoint freeze.")
    scores = [float(row["validation_score"]) for row in rows]
    if not all(math.isfinite(value) for value in scores):
        raise ValueError("A historical validation score is non-finite.")
    selected = min(range(epochs), key=lambda i: scores[i])
    if selected + 1 != record["best_epoch"] or scores[selected] != record["best_validation_score"]:
        raise ValueError("Selected checkpoint does not match the earliest minimum validation score.")
    return [row["batch_order_sha256"] for row in rows]


def source_namespace(root, denominators):
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    source_path = root / "scripts/06_best_model_multiseed_v4.py"
    if sha256(source_path) != SOURCE_SHA:
        raise ValueError("Source identity mismatch immediately before extraction.")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in SYMBOLS]
    if {node.name for node in nodes} != SYMBOLS:
        raise ValueError("Required archived definitions are missing.")
    namespace = {"torch": torch, "nn": nn, "np": np, "pd": pd, "ASSET_ORDER": ASSETS,
                 "TAU": 0.5, "denominators": denominators}
    tree = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(source_path), "exec"), namespace)
    return namespace


def load_train_validation(root):
    import numpy as np
    schema = json.loads((root / "config/schema_v4.json").read_text(encoding="utf-8"))
    if schema["data"]["assets"] != ASSETS or schema["features"]["baseline_dim"] != 8:
        raise ValueError("Frozen schema asset/feature order mismatch.")
    folder = root / "data/sequences/baseline/lb10"
    expected = {"X_train": (2714, 10, 8), "y_train": (2714, 8), "y_train_raw": (2714, 8),
                "X_val": (584, 10, 8), "y_val": (584, 8), "y_val_raw": (584, 8)}
    data = {}
    for name, shape in expected.items():
        array = np.load(folder / (name + ".npy"), allow_pickle=False)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"Shape/finite-value check failed: {name}")
        data[name] = array
    date_sets = {}
    for split, n in [("train", 2714), ("val", 584)]:
        dates = {}
        for prefix in ["anchor_dates", "target_realization_dates"]:
            a = np.load(folder / f"{prefix}_{split}.npy", allow_pickle=False)
            if a.shape != (n,):
                raise ValueError("Date array shape mismatch.")
            a = a.astype("datetime64[D]")
            if np.isnat(a).any() or not np.all(np.diff(a) > np.timedelta64(0, "D")):
                raise ValueError("Date arrays must be finite, unique and increasing.")
            dates[prefix] = a
        if not np.all(dates["target_realization_dates"] > dates["anchor_dates"]):
            raise ValueError("Target realization must follow its input anchor.")
        date_sets[split] = dates
    train_end = str(date_sets["train"]["target_realization_dates"][-1])
    val_begin = str(date_sets["val"]["target_realization_dates"][0])
    val_end = str(date_sets["val"]["target_realization_dates"][-1])
    if (train_end, val_begin, val_end) != ("2020-07-09", "2020-07-10", "2022-10-05"):
        raise ValueError("Frozen target date boundaries differ.")
    # This is the user's trusted original scaler, fingerprinted before loading.
    with (root / "data/sequences/baseline/scalers.pkl").open("rb") as handle:
        scaler = pickle.load(handle)["y_scaler"]
    for split in ["train", "val"]:
        restored = scaler.inverse_transform(data["y_" + split].copy())
        maximum = float(np.max(np.abs(restored - data["y_" + split + "_raw"])))
        if maximum > 1e-5:
            raise ValueError(f"Frozen scaler inverse-transform check failed: {split}, {maximum}")
    denominators = json.loads((root / "data/processed/selection_baseline_denominators_v4.json").read_text())
    for family in ["return_denominator", "volatility_denominator"]:
        if any(not math.isfinite(float(denominators[family][asset]["value"]))
               or float(denominators[family][asset]["value"]) <= 0 for asset in ASSETS):
            raise ValueError("Frozen selection denominator must be finite and positive.")
    return data, scaler, denominators


def build_model(namespace, arm):
    model = namespace[arm["architecture"]](n_features=8, lookback=10, d_model=32,
        n_head=4, n_layers=2, d_ff=arm["d_ff"], dropout=0.1)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    shared = sum(p.numel() for p in model.pcgrad_groups()["shared"])
    if count != arm["parameters"]:
        raise ValueError(f"Unexpected parameter count for {arm['id']}: {count}")
    if (arm["architecture"] == "NoSharing" and shared != 0) or (arm["architecture"] == "FullSharingMTL" and shared <= 0):
        raise ValueError("Shared-parameter audit failed.")
    return model


def evaluate(model, loader, raw_truth, scaler, namespace, device):
    import numpy as np
    import torch
    model.eval()
    predictions = []
    with torch.no_grad():
        for features, _ in loader:
            output = model(features.to(device))
            if not torch.isfinite(output).all():
                raise ValueError("Non-finite evaluation prediction.")
            predictions.append(output.cpu().numpy())
    scaled = np.concatenate(predictions, axis=0)
    raw = scaler.inverse_transform(scaled)
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite inverse-transformed prediction.")
    frame = namespace["compute_raw_metrics"](raw_truth, raw)
    score = namespace["compute_validation_score"](frame)
    returns = frame.loc[frame["task"] == "return"]
    volatility = frame.loc[frame["task"] == "volatility"]
    metrics = {"n": int(len(raw)), "return_mae": float(returns["MAE"].mean()),
               "return_rmse": float(returns["RMSE"].mean()),
               "volatility_pinball": float(volatility["PinballLoss_tau_0.5"].mean()),
               "volatility_mae": float(volatility["MAE"].mean()),
               "validation_score": float(score["validation_score"]),
               "avg_return_ratio": float(score["avg_return_ratio"]),
               "avg_vol_ratio": float(score["avg_vol_ratio"]),
               "per_asset": frame.to_dict(orient="records")}
    if not all(math.isfinite(metrics[k]) for k in metrics if isinstance(metrics[k], (int, float))):
        raise ValueError("Non-finite primary evaluation metric.")
    return json_clean(metrics)


def save_checkpoint(path, model, metadata):
    import torch
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    payload = dict(metadata)
    payload["model_state_dict"] = {name: value.detach().cpu().clone()
                                   for name, value in model.state_dict().items()}
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run_job(job):
    import numpy as np
    verify_implementation_identities(job["implementation_identities"])
    torch = configure_torch(job["threads"])
    runtime = collect_runtime_environment(job["threads"])
    validate_runtime_environment(runtime, job["runtime_environment"])
    from torch.utils.data import DataLoader, TensorDataset
    root, output = Path(job["input_root"]), Path(job["run_root"])
    arm, seed, epochs = job["arm"], int(job["seed"]), int(job["epochs"])
    verify_input_identities(root, job["input_identities"])
    data, scaler, denominators = load_train_validation(root)
    namespace = source_namespace(root, denominators)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(job["device"])
    model = build_model(namespace, arm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    features = torch.tensor(data["X_train"], dtype=torch.float32)
    targets = torch.tensor(data["y_train"], dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    training = DataLoader(TensorDataset(features, targets, torch.arange(len(features))),
                          batch_size=64, shuffle=True, drop_last=False, generator=generator,
                          num_workers=0)
    train_eval = DataLoader(TensorDataset(features, targets), batch_size=64, shuffle=False, num_workers=0)
    val_eval = DataLoader(TensorDataset(torch.tensor(data["X_val"], dtype=torch.float32),
                                      torch.tensor(data["y_val"], dtype=torch.float32)),
                         batch_size=64, shuffle=False, num_workers=0)
    name = f"{arm['id']}_seed{seed}"
    folder = output / "runs" / name
    folder.mkdir(parents=True, exist_ok=False)
    history_path = folder / "history.csv"
    best_score = math.inf; best_epoch = None; best_metrics = None
    total_steps = 0; began = time.monotonic(); epoch_seconds = []
    fields = ["epoch", "train_mode_optimization_loss", "train_eval_return_mae", "val_return_mae",
              "validation_return_gap", "train_eval_volatility_pinball", "val_volatility_pinball",
              "validation_volatility_gap", "validation_score", "avg_return_ratio", "avg_vol_ratio",
              "batch_order_sha256", "optimizer_steps", "epoch_seconds"]
    with history_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for epoch in range(1, epochs + 1):
            epoch_start = time.monotonic(); model.train(); objective_sum = 0.0; seen = 0
            order = hashlib.sha256()
            for x, y, indices in training:
                order.update(indices.numpy().tobytes())
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                predicted = model(x)
                return_loss = torch.nn.functional.mse_loss(predicted[:, :4], y[:, :4])
                error = y[:, 4:] - predicted[:, 4:]
                volatility_loss = torch.maximum(0.5 * error, -0.5 * error).mean()
                objective = 0.7 * return_loss + 0.3 * volatility_loss
                if not torch.isfinite(objective):
                    raise ValueError(f"Non-finite optimization loss in {name}, epoch {epoch}.")
                objective.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise ValueError("Non-finite gradient norm.")
                optimizer.step(); total_steps += 1
                objective_sum += float(objective.detach()) * len(x); seen += len(x)
            train_metrics = evaluate(model, train_eval, data["y_train_raw"], scaler, namespace, device)
            val_metrics = evaluate(model, val_eval, data["y_val_raw"], scaler, namespace, device)
            current = val_metrics["validation_score"]
            elapsed = time.monotonic() - epoch_start; epoch_seconds.append(elapsed)
            row = {"epoch": epoch, "train_mode_optimization_loss": objective_sum / seen,
                   "train_eval_return_mae": train_metrics["return_mae"],
                   "val_return_mae": val_metrics["return_mae"],
                   "validation_return_gap": val_metrics["return_mae"] - train_metrics["return_mae"],
                   "train_eval_volatility_pinball": train_metrics["volatility_pinball"],
                   "val_volatility_pinball": val_metrics["volatility_pinball"],
                   "validation_volatility_gap": val_metrics["volatility_pinball"] - train_metrics["volatility_pinball"],
                   "validation_score": current, "avg_return_ratio": val_metrics["avg_return_ratio"],
                   "avg_vol_ratio": val_metrics["avg_vol_ratio"], "batch_order_sha256": order.hexdigest(),
                   "optimizer_steps": total_steps, "epoch_seconds": elapsed}
            writer.writerow(row); handle.flush()
            # Strict improvement preserves the earliest epoch when scores tie.
            if current < best_score:
                best_score, best_epoch = current, epoch
                best_metrics = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
                save_checkpoint(folder / "best_checkpoint.pt", model,
                                {"arm": arm, "seed": seed, "epoch": epoch, "validation_score": current,
                                 "source_sha256": SOURCE_SHA, "input_manifest_sha256": job["input_manifest_sha256"]})
                write_json(folder / "best_train_validation_metrics.json", best_metrics)
            print(f"[{name}] epoch={epoch:03d}/{epochs} score={current:.6f} seconds={elapsed:.3f}", flush=True)
    save_checkpoint(folder / "final_checkpoint.pt", model,
                    {"arm": arm, "seed": seed, "epoch": epochs,
                     "validation_score": val_metrics["validation_score"], "source_sha256": SOURCE_SHA,
                     "input_manifest_sha256": job["input_manifest_sha256"]})
    write_json(folder / "final_train_validation_metrics.json",
               {"epoch": epochs, "train": train_metrics, "validation": val_metrics}, exclusive=True)
    verify_input_identities(root, job["input_identities"])
    verify_implementation_identities(job["implementation_identities"])
    result = {"status": "SUCCESS_NO_TEST_ACCESS", "arm": arm, "seed": seed,
              "epochs_ran": epochs, "optimizer_steps": total_steps, "best_epoch": best_epoch,
              "best_validation_score": best_score, "runtime_seconds": time.monotonic() - began,
              "median_epoch_seconds": float(np.median(epoch_seconds)), "test_arrays_loaded": False,
              "runtime_environment": runtime,
              "best_train_validation_metrics": best_metrics,
              "best_checkpoint": {"path": str((folder / "best_checkpoint.pt").relative_to(output)),
                                  "sha256": sha256(folder / "best_checkpoint.pt")},
              "final_checkpoint": {"path": str((folder / "final_checkpoint.pt").relative_to(output)),
                                   "sha256": sha256(folder / "final_checkpoint.pt")},
              "history": {"path": str(history_path.relative_to(output)), "sha256": sha256(history_path)}}
    write_json(folder / "run_record.json", result, exclusive=True)
    return result


def check_protocol(protocol):
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Use the separately locked T4 H2 protocol, never the CPU protocol.")
    validate_expected_runtime(protocol["runtime_environment"])
    expected = {"epochs": 100, "early_stopping": False, "batch_size": 64, "optimizer": "AdamW",
                "learning_rate": 0.001, "weight_decay": 0.0001, "gradient_clip_norm": 1,
                "dropout": 0.1, "seeds": SEEDS,
                "loss": "0.7 * scaled-return MSE + 0.3 * scaled-volatility pinball(tau=0.5)",
                "checkpoint_rule": "Lowest original raw-scale combined validation score; earlier epoch breaks ties. Also report fixed final-epoch diagnostics."}
    for key, value in expected.items():
        if protocol["training"][key] != value:
            raise ValueError(f"Protocol differs from implemented fixed design: training.{key}")
    if protocol["data"]["features"] != "baseline" or protocol["data"]["lookback"] != 10:
        raise ValueError("H2 v1 requires baseline features and lookback 10.")
    if (protocol["data"]["assets"] != ASSETS or protocol["data"]["train_sequences"] != 2714
            or protocol["data"]["validation_sequences"] != 584 or protocol["data"]["test_sequences"] != 584):
        raise ValueError("Protocol frozen split counts or asset order differ.")
    if not protocol["data"]["old_test_already_examined"]:
        raise ValueError("The existing test set must be labelled previously examined/exploratory.")
    for actual, expected_arm in zip(protocol["arms"], ARMS):
        for key in ["id", "architecture", "d_ff"]:
            if actual[key] != expected_arm[key]:
                raise ValueError(f"Protocol arm differs: {key}")
        if (actual["d_model"], actual["n_head"], actual["n_layers"]) != (32, 4, 2):
            raise ValueError("Unexpected backbone dimensions.")
        if actual["expected_parameters_analytical"] != expected_arm["parameters"]:
            raise ValueError("Protocol expected parameter count differs.")
    if len(protocol["arms"]) != 3:
        raise ValueError("H2 v1 requires exactly three arms.")
    if set(protocol.get("input_identities", {})) != set(USED_RELATIVE_INPUTS + HELD_OUT_TEST_INPUTS):
        raise ValueError("The T4 protocol must preserve all 21 original CPU input identities.")


def self_test(root):
    import numpy as np
    torch = configure_torch(1)
    configure_torch(1)  # Reused spawned workers must allow another arm/seed fit.
    denominator = {family: {asset: {"value": 1.0} for asset in ASSETS}
                   for family in ["return_denominator", "volatility_denominator"]}
    namespace = source_namespace(root, denominator)
    expected = np.tile(np.array([1, 2, 3, 4, 2, 4, 6, 8], dtype=np.float32), (2, 1))
    frame = namespace["compute_raw_metrics"](np.zeros_like(expected), expected)
    score = namespace["compute_validation_score"](frame)
    if score["validation_score"] != 2.5 or score["avg_return_ratio"] != 2.5 or score["avg_vol_ratio"] != 2.5:
        raise AssertionError("Archived raw-loss/ValidationScore semantic self-test failed.")
    for arm in ARMS:
        model = build_model(namespace, arm)
        model.eval(); x = torch.zeros(2, 10, 8)
        with torch.no_grad():
            first, second = model(x), model(x)
        if first.shape != (2, 8) or not torch.equal(first, second) or not torch.isfinite(first).all():
            raise AssertionError("Model output/evaluation determinism self-test failed.")
    # These are deliberately synthetic runtime records. No CUDA runtime, GPU,
    # model fit or actual T4 compatibility is claimed by this CPU self-test.
    expected = json.loads(json.dumps(RUNTIME_PINS))
    expected.update({"cudnn_version": 90000, "driver_version": "SYNTHETIC_DRIVER_NOT_A_RUNTIME",
                     "determinism": dict(DETERMINISM)})
    validate_runtime_environment(expected, expected)
    mismatch_count = 0
    for key in expected:
        changed = json.loads(json.dumps(expected))
        if key == "packages":
            changed[key]["torch"] = "2.11.0+cu130"
        elif key == "determinism":
            changed[key]["amp_enabled"] = True
        elif isinstance(changed[key], int):
            changed[key] += 1
        elif isinstance(changed[key], list):
            changed[key] = [0, 0]
        else:
            changed[key] = "SYNTHETIC_MISMATCH"
        try:
            validate_runtime_environment(changed, expected)
        except ValueError:
            mismatch_count += 1
        else:
            raise AssertionError(f"Runtime mismatch did not fail closed: {key}")
    altered_protocol = json.loads(json.dumps(expected))
    altered_protocol["packages"]["numpy"] = "2.1.3"
    try:
        validate_runtime_environment(altered_protocol, altered_protocol)
    except ValueError:
        mismatch_count += 1
    else:
        raise AssertionError("A matching but unapproved runtime/protocol pair was accepted.")
    rows = [{"epoch": str(i), "validation_score": str(score), "batch_order_sha256": str(i) * 64}
            for i, score in enumerate([2.0, 1.0, 1.0], start=1)]
    record = {"best_epoch": 2, "best_validation_score": 1.0}
    validate_completed_history(rows, record, 3)
    history_guard_count = 0
    for bad_rows, bad_record in [(rows[:-1], record), (rows, {**record, "best_epoch": 3}),
                                 (rows, {**record, "best_validation_score": 0.5})]:
        try:
            validate_completed_history(bad_rows, bad_record, 3)
        except ValueError:
            history_guard_count += 1
        else:
            raise AssertionError("Incomplete or wrongly selected history was accepted.")
    print(json.dumps({"status": "CPU_MODEL_AND_RUNTIME_GUARD_SELF_TEST_PASS",
                      "parameter_counts": [a["parameters"] for a in ARMS],
                      "raw_metric_validation_score_check": 2.5, "runtime_mismatches_rejected": mismatch_count,
                      "history_guard_cases_rejected": history_guard_count,
                      "file_writes": False, "cuda_tested": False, "models_trained": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--threads", type=int, choices=[1], default=1)
    parser.add_argument("--workers", type=int, choices=[1], default=1)
    parser.add_argument("--pilot", action="store_true", help="One NS/seed123 epoch in a new directory; never an H2 result.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--self-test", action="store_true", help="No-write CPU-only model/metric and synthetic runtime-guard checks; never tests a GPU.")
    modes.add_argument("--runtime-report", action="store_true", help="Print the actual runtime JSON to freeze in a new protocol; no inputs, files or fitting.")
    args = parser.parse_args()
    if args.runtime_report:
        print(json.dumps(collect_runtime_environment(), ensure_ascii=False, allow_nan=False)); return
    if args.input_root is None:
        parser.error("--input-root is required for self-test or T4 training.")
    root = args.input_root.expanduser().resolve()
    if args.self_test:
        self_test(root); return
    if args.run_root is None or args.protocol is None:
        parser.error("--run-root and --protocol are required unless --self-test is used.")
    output = args.run_root.expanduser().resolve()
    protocol_path = args.protocol.expanduser().resolve()
    protocol_bytes = protocol_path.read_bytes()
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    protocol = json.loads(protocol_bytes)
    check_protocol(protocol)
    validate_execution_paths(root, output, protocol)
    implementation_identities = {
        "run_h2_training_t4_v1.py": protocol["analysis"]["training_implementation_sha256"],
        "evaluate_h2_t4_v1.py": protocol["analysis"]["evaluation_implementation_sha256"],
    }
    verify_implementation_identities(implementation_identities)
    runtime = collect_runtime_environment()
    validate_runtime_environment(runtime, protocol["runtime_environment"])
    identities = input_identities(root)
    if identities != protocol["input_identities"]:
        raise ValueError("Copied inputs differ from the original CPU experiment's 21 frozen identities.")
    data, _, _ = load_train_validation(root)
    del data
    import numpy as np
    import pandas as pd
    import torch
    output.mkdir(parents=True, exist_ok=False)
    (output / "runs").mkdir()
    jobs = [(ARMS[0], SEEDS[0])] if args.pilot else [(arm, seed) for arm in ARMS for seed in SEEDS]
    epochs = 1 if args.pilot else 100
    mode = "T4_PILOT_TIMING_AND_CODE_CHECK_NOT_AN_H2_RESULT" if args.pilot else EXECUTION_MODE
    manifest = {"status": "LOCKED_BEFORE_TRAINING", "created_at_utc": now(), "execution_mode": mode,
                "input_root": str(root), "run_root": str(output),
                "protocol_draft_sha256": protocol_sha, "protocol_draft": protocol,
                "training_implementation_sha256": sha256(Path(__file__)), "input_identities": identities,
                "effective_epochs": epochs, "jobs": [{"arm": arm, "seed": seed} for arm, seed in jobs],
                "device": args.device, "threads_per_worker": args.threads, "workers": args.workers,
                "runtime_environment": runtime,
                "software": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                             "torch": torch.__version__, "scikit-learn": importlib.metadata.version("scikit-learn")},
                "test_arrays_loaded": False, "old_project_writes": False}
    manifest["test_files_hashed_as_opaque_bytes_only"] = HELD_OUT_TEST_INPUTS
    manifest["implemented_training_settings"] = {
        "epochs": epochs, "batch_size": 64, "optimizer": "AdamW", "learning_rate": 0.001,
        "weight_decay": 0.0001, "gradient_clip_norm": 1.0, "dropout": 0.1,
        "return_loss": "scaled MSE", "return_weight": 0.7,
        "volatility_loss": "scaled pinball", "volatility_weight": 0.3, "tau": 0.5,
        "early_stopping": False, "checkpoint_selection": "strict minimum archived raw ValidationScore; first epoch wins ties",
        "training_evaluation_mode": "model.eval() with no_grad", "batch_order": "independent seeded DataLoader generator; hash each epoch",
    }
    manifest_path = output / "locked_training_manifest.json"
    write_json(manifest_path, manifest, exclusive=True)
    manifest_sha = sha256(manifest_path)
    write_json(output / "status.json", {"status": "TRAINING", "execution_mode": mode, "completed": 0}, exclusive=True)
    payloads = [{"input_root": str(root), "run_root": str(output), "arm": arm, "seed": seed,
                 "epochs": epochs, "threads": args.threads, "device": args.device,
                 "runtime_environment": runtime,
                 "implementation_identities": implementation_identities,
                 "input_identities": identities, "input_manifest_sha256": manifest_sha} for arm, seed in jobs]
    completed, failures = [], []
    try:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
            futures = {executor.submit(run_job, payload): payload for payload in payloads}
            for future in concurrent.futures.as_completed(futures):
                payload = futures[future]
                try:
                    completed.append(future.result())
                except Exception as exc:
                    failures.append({"arm": payload["arm"]["id"], "seed": payload["seed"],
                                     "error_type": type(exc).__name__, "error": str(exc)})
                write_json(output / "status.json", {"status": "TRAINING", "execution_mode": mode,
                           "completed": len(completed), "failures": failures})
        if failures:
            raise RuntimeError(f"Training jobs failed: {failures}")
        verify_input_identities(root, identities)
        verify_implementation_identities(implementation_identities)
        if sha256(protocol_path) != protocol_sha or sha256(manifest_path) != manifest_sha:
            raise ValueError("A protocol or pretraining manifest changed during the T4 run.")
        if len(completed) != len(jobs):
            raise AssertionError("Missing completed training jobs.")
        # Verify paired-seed minibatch permutations, independently of model RNG.
        orders = {}
        for record in completed:
            if record.get("epochs_ran") != epochs or record.get("runtime_environment") != runtime:
                raise ValueError("A completed fit differs from its fixed epoch/runtime lock.")
            for key in ("history", "best_checkpoint", "final_checkpoint"):
                item = record[key]
                artifact = (output / item["path"]).resolve()
                if not artifact.is_relative_to(output) or not artifact.is_file() or sha256(artifact) != item["sha256"]:
                    raise ValueError(f"Completed {key} changed before the frozen manifest was published.")
            history = output / record["history"]["path"]
            with history.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            order = validate_completed_history(rows, record, epochs)
            seed = record["seed"]
            if seed in orders and orders[seed] != order:
                raise AssertionError(f"Paired-seed batch order differs: {seed}")
            orders[seed] = order
        completed.sort(key=lambda r: (next(i for i,a in enumerate(ARMS) if a["id"] == r["arm"]["id"]), r["seed"]))
        status = "T4_PILOT_COMPLETE_NOT_AN_H2_RESULT" if args.pilot else "ALL_NINE_CHECKPOINTS_FROZEN_NO_TEST_ACCESS"
        record = {"status": status, "execution_mode": mode, "completed_at_utc": now(),
                  "locked_training_manifest_sha256": manifest_sha, "completed_runs": completed,
                  "runtime_environment": runtime,
                  "paired_seed_batch_orders_equal": True, "test_arrays_loaded": False,
                  "next_step": "Separate evaluation may load test arrays only after verifying this manifest and all nine checkpoint identities."}
        filename = "pilot_complete.json" if args.pilot else "selected_checkpoints_frozen.json"
        write_json(output / filename, record, exclusive=True)
        write_json(output / "status.json", {"status": status, "completed": len(completed), "test_arrays_loaded": False})
        print(json.dumps({"status": status, "runs": len(completed), "record": str(output / filename)}, ensure_ascii=False))
    except BaseException as exc:
        write_json(output / "status.json", {"status": "FAILED_NO_TEST_ACCESS", "completed": len(completed),
                   "error_type": type(exc).__name__, "error": str(exc), "failures": failures,
                   "test_arrays_loaded": False})
        raise


if __name__ == "__main__":
    main()
