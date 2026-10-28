#!/usr/bin/env python3
"""Read-only H2 v1 input and model preflight; no training and no file writes.

This file is a new preparation utility. It extracts only the four named class/
function definitions from the hash-verified original source. It never imports or
executes the archived pipeline module's top-level code.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

SOURCE_SHA = "35de2ee398699003dfef6be36b70c112fb2c0d1b1e9577cbf64bef58877e16d8"
RAW_SHA = "ab5f275d38dc98057b1cedcf58019adb26be7402c7ed5ae6ee3d6877b2444893"
CLASS_NAMES = {"TransformerBlock", "make_head", "FullSharingMTL", "NoSharing"}

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def inspect(input_root, planned_output_root):
    import numpy as np
    import torch
    import torch.nn as nn

    root = Path(input_root).expanduser().resolve()
    output = Path(planned_output_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input project not found: {root}")
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("The H2 output root must be separate from the original project.")
    source_path = root / "scripts/06_best_model_multiseed_v4.py"
    raw_path = root / "data/raw/raw_prices.csv"
    if sha256(source_path) != SOURCE_SHA:
        raise ValueError("Original model-source SHA-256 mismatch.")
    if sha256(raw_path) != RAW_SHA:
        raise ValueError("Frozen raw-data SHA-256 mismatch.")
    schema_path = root / "config/schema_v4.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema["data"]["assets"] != ["BIST100", "USDTRY", "EURTRY", "GOLD"]:
        raise ValueError("Asset order differs from the frozen design.")
    if schema["features"]["baseline_dim"] != 8:
        raise ValueError("Expected eight baseline inputs.")

    seq = root / "data/sequences/baseline/lb10"
    shapes = {
        "X_train.npy": (2714, 10, 8), "y_train.npy": (2714, 8),
        "y_train_raw.npy": (2714, 8), "X_val.npy": (584, 10, 8),
        "y_val.npy": (584, 8), "y_val_raw.npy": (584, 8),
    }
    identities = {}
    for name, expected in shapes.items():
        path = seq / name
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != expected or not np.isfinite(array).all():
            raise ValueError(f"Shape or finite-value check failed: {name}")
        identities[str(path.relative_to(root))] = {
            "shape": list(array.shape), "dtype": str(array.dtype),
            "sha256_recorded_for_future_lock": sha256(path),
        }
        del array

    dates = {}
    for split, count in [("train", 2714), ("val", 584)]:
        loaded = {}
        for kind in ["anchor_dates", "target_realization_dates"]:
            path = seq / f"{kind}_{split}.npy"
            array = np.load(path, allow_pickle=False)
            if array.shape != (count,):
                raise ValueError(f"Date-array shape mismatch: {path.name}")
            value = array.astype("datetime64[D]")
            if np.isnat(value).any() or not np.all(np.diff(value) > np.timedelta64(0, "D")):
                raise ValueError(f"Dates must be finite, unique and increasing: {path.name}")
            identities[str(path.relative_to(root))] = {
                "shape": list(array.shape), "sha256_recorded_for_future_lock": sha256(path),
            }
            loaded[kind] = value
        if not np.all(loaded["target_realization_dates"] > loaded["anchor_dates"]):
            raise ValueError(f"Target date must follow input anchor: {split}")
        dates[split] = loaded
    if not dates["train"]["target_realization_dates"][-1] < dates["val"]["target_realization_dates"][0]:
        raise ValueError("Training and validation target periods overlap.")
    if str(dates["train"]["target_realization_dates"][-1]) != "2020-07-09":
        raise ValueError("Training target-end date differs from the frozen split.")
    if str(dates["val"]["target_realization_dates"][0]) != "2020-07-10":
        raise ValueError("Validation target-start date differs from the frozen split.")
    if str(dates["val"]["target_realization_dates"][-1]) != "2022-10-05":
        raise ValueError("Validation target-end date differs from the frozen split.")

    # Record scaler/denominator identities without deserializing the pickle.
    for relative in ["data/sequences/baseline/scalers.pkl",
                     "data/processed/selection_baseline_denominators_v4.json"]:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen input missing: {relative}")
        identities[relative] = {"sha256_recorded_for_future_lock": sha256(path)}

    syntax = ast.parse(source_path.read_text(encoding="utf-8"))
    nodes = [node for node in syntax.body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in CLASS_NAMES]
    if {node.name for node in nodes} != CLASS_NAMES:
        raise ValueError("Expected archived model definitions not found.")
    namespace = {"torch": torch, "nn": nn}
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(source_path), "exec"), namespace)
    arms = [
        ("NS_base", "NoSharing", 128, 54536),
        ("FS_same_width", "FullSharingMTL", 128, 28456),
        ("FS_budget_control", "FullSharingMTL", 329, 54586),
    ]
    models = []
    for name, class_name, d_ff, expected in arms:
        torch.manual_seed(123)
        model = namespace[class_name](n_features=8, lookback=10, d_model=32,
                                      n_head=4, n_layers=2, d_ff=d_ff, dropout=0.1)
        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if parameters != expected:
            raise ValueError(f"Unexpected parameter count in {name}: {parameters}")
        groups = model.pcgrad_groups()
        shared = sum(p.numel() for p in groups["shared"])
        if (class_name == "NoSharing" and shared != 0) or (class_name == "FullSharingMTL" and shared <= 0):
            raise ValueError(f"Shared-parameter check failed: {name}")
        model.eval()
        with torch.no_grad():
            result = model(torch.zeros(2, 10, 8))
        if tuple(result.shape) != (2, 8) or not torch.isfinite(result).all():
            raise ValueError(f"Forward-only model smoke check failed: {name}")
        models.append({"arm": name, "trainable_parameters": parameters,
                       "shared_parameters": shared, "d_ff": d_ff})
        del model
    budget_difference = abs(models[2]["trainable_parameters"] - models[0]["trainable_parameters"])
    relative_difference = budget_difference / models[0]["trainable_parameters"]
    if relative_difference > 0.01:
        raise ValueError("Capacity-control tolerance exceeded.")

    return {
        "status": "PREFLIGHT_PASS_NOT_A_TRAINING_RESULT",
        "original_project_writes": False, "file_writes_by_this_script": False,
        "test_arrays_loaded": False, "models_trained": False,
        "input_root": str(root), "planned_separate_output_root": str(output),
        "output_directory_created": False, "output_already_exists": output.exists(),
        "known_frozen_identities_verified": {"model_source": SOURCE_SHA, "raw_data": RAW_SHA},
        "input_identities_recorded_not_previously_locked_here": identities,
        "scaler_pickle_loaded": False, "models": models,
        "capacity_relative_difference": relative_difference,
        "software": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "torch": torch.__version__, "cuda_available": torch.cuda.is_available()},
        "next_step": "Review this report and freeze H2 input identities before implementing/running training.",
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="/content/drive/MyDrive/tez_transformer_v4_repro")
    parser.add_argument("--planned-output-root", default="/content/drive/MyDrive/tez_transformer_h2_v1")
    args = parser.parse_args()
    print(json.dumps(inspect(args.input_root, args.planned_output_root),
                     ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
