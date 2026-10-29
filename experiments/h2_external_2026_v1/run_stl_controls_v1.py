#!/usr/bin/env python3
"""Train six isolated task-specific controls before accessing any new period.

Only frozen train/validation arrays are deserialized. The five old test files
are verified as opaque bytes. No historical module is imported, no historical
checkpoint is opened, and no external-period price file is read. ``--prepare-only``
performs synthetic, zero-optimizer-step checks and writes nothing. ``--train``
writes exclusively below this new project's runs/stl_run_v1 directory.

These independent-task controls change the optimizer, clipping and checkpoint
selection protocol as well as removing joint training. They are not a pure
causal intervention on parameter sharing. The original H1/H2 are unchanged.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pickle
import platform
import random
import subprocess
import sys
import time

# This is set before the first torch import, including when loaded for auditing.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.dont_write_bytecode = True

DEFAULT_PROJECT = Path("/content/drive/MyDrive/tez_transformer_h2_external_2026_v1")
PROTECTED_ROOTS = (
    Path("/content/drive/MyDrive/tez_transformer_v4_repro"),
    Path("/content/drive/MyDrive/tez_transformer_h2_v1"),
    Path("/content/drive/MyDrive/tez_transformer_h2_t4_v1"),
)
OLD_LOCK_SHA = "517c06f3b1a5d71de9739f56ef6a2af7cf9a918ed2cf303a52d30bfc7b9f393a"
OLD_FROZEN_SHA = "12cf2481978546689f9087d7d97cb4d8325756a15f393a9c936a31e2a47876f4"
SOURCE_06 = "scripts/06_best_model_multiseed_v4.py"
SOURCE_08 = "scripts/08B_learned_baselines_test_v4.py"
SOURCE_06_SHA = "35de2ee398699003dfef6be36b70c112fb2c0d1b1e9577cbf64bef58877e16d8"
SOURCE_08_SHA = "df4fa2b29a6b522fc410f693a30520e045c144a15f7c7368af5eb6a1f0f12566"
SEQ = "data/sequences/baseline/lb10/"
ASSETS = ("BIST100", "USDTRY", "EURTRY", "GOLD")
SEEDS = (123, 777, 2026)
TASKS = ("return", "volatility")
PACKAGES = {
    "torch": "2.11.0+cu128", "numpy": "2.3.5", "pandas": "2.2.3",
    "scipy": "1.17.0", "scikit-learn": "1.6.1",
}
SPLIT_SIZES = {"train": 2714, "val": 584}
OLD_INPUTS = {
    SOURCE_06, "config/schema_v4.json", "data/raw/raw_prices.csv",
    "data/sequences/baseline/scalers.pkl",
    "data/processed/selection_baseline_denominators_v4.json", SEQ + "sequence_meta.json",
} | {SEQ + name + "_" + split + ".npy"
     for split in ("train", "val", "test") for name in ("X", "y")} | {
    SEQ + name + "_" + split + ".npy"
    for split in ("train", "val", "test")
    for name in ("anchor_dates", "target_realization_dates")
} | {SEQ + "y_" + split + "_raw.npy" for split in ("train", "val", "test")}
OPAQUE_TEST_INPUTS = sorted(p for p in OLD_INPUTS if p.endswith("_test.npy") or p.endswith("_test_raw.npy"))
REL_TOL, ABS_TOL = 1e-5, 1e-8
SETTINGS = {
    "seeds": list(SEEDS), "tasks": list(TASKS), "actual_model_fits": 6,
    "epochs": 100, "early_stopping": False, "batch_size": 64,
    "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
    "gradient_clip_norm_per_task_model": 1.0, "dropout": 0.1,
    "return_training_loss": "unweighted scaled MSE",
    "volatility_training_loss": "unweighted scaled pinball tau=0.5",
    "selection": "strict minimum frozen-denominator normalized task validation score; earliest tie",
    "return_validation_score": "mean over four assets of raw MAE / frozen return denominator",
    "volatility_validation_score": "mean over four assets of raw pinball(tau=0.5) / frozen volatility denominator",
    "primary_losses": "raw MAE for return; raw pinball(tau=0.5) for volatility; equal asset weights",
    "initialization": "copy corresponding branch of seeded, frozen-source NoSharing template",
    "minibatch_order": "independently seeded DataLoader generator, equal between tasks for paired seed",
    "dropout_streams": "no claim of equality with the historical jointly evaluated NoSharing model",
    "parameters_per_model": 27268, "parameters_per_task_pair": 54536,
    "n_features": 8, "lookback": 10, "d_model": 32, "n_head": 4,
    "n_layers": 2, "d_ff": 128, "threads": 1, "amp": False,
    "interpretation": "independent-task training control; optimizer/clipping/selection bundle differs from joint training",
}
HISTORY_FIELDS = (
    "epoch", "scaled_optimization_loss", "train_primary_loss", "validation_primary_loss",
    "validation_gap", "train_task_score", "validation_task_score",
    "batch_order_sha256", "optimizer_steps", "epoch_seconds",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value, *, exclusive=True):
    path = Path(path)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(data)
    else:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(data)
        os.replace(temporary, path)


def reject_symlinks(path):
    path = Path(path).absolute()
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f"Symbolic links are not permitted: {component}")


def child(root, relative, *, must_exist=True):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Unsafe relative path.")
    candidate = Path(root) / relative
    reject_symlinks(candidate)
    resolved = candidate.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("Path escapes its dedicated root.")
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def validate_project(project):
    reject_symlinks(project)
    project = Path(project).expanduser().resolve()
    if not project.is_dir():
        raise ValueError("The separately staged project directory must already exist.")
    for protected in PROTECTED_ROOTS:
        protected = protected.resolve()
        if project == protected or project.is_relative_to(protected) or protected.is_relative_to(project):
            raise ValueError("The historical v4, CPU and T4 projects are protected.")
    return project


def read_evidence(project):
    plan_path = child(project, "protocol_plan_v1.json")
    plan_sha = sha256(plan_path)
    plan = read_json(plan_path)
    if plan.get("status") != "STAGED_PLAN_NO_NEW_DATA":
        raise ValueError("The scientific plan must be fixed before any new-period access.")
    design = plan.get("design", {})
    for key, expected in (("seeds", list(SEEDS)), ("epochs", 100), ("stl_tasks", list(TASKS))):
        if key in design and design[key] != expected:
            raise ValueError(f"Scientific plan conflicts with fixed STL design: {key}")
    if "project_root" in plan and Path(plan["project_root"]).expanduser().resolve() != project:
        raise ValueError("Plan project root differs from the staged project.")
    old_lock_path = child(project, "source_records/locked_training_manifest.json")
    old_frozen_path = child(project, "source_records/selected_checkpoints_frozen.json")
    if sha256(old_lock_path) != OLD_LOCK_SHA or sha256(old_frozen_path) != OLD_FROZEN_SHA:
        raise ValueError("Historical source manifest identities differ.")
    old_lock, old_frozen = read_json(old_lock_path), read_json(old_frozen_path)
    if (old_frozen.get("status") != "ALL_NINE_CHECKPOINTS_FROZEN_NO_TEST_ACCESS"
            or old_frozen.get("locked_training_manifest_sha256") != OLD_LOCK_SHA
            or len(old_frozen.get("completed_runs", [])) != 9):
        raise ValueError("Historical completion evidence is incomplete.")
    identities = old_lock.get("input_identities", {})
    if set(identities) != OLD_INPUTS or len(identities) != 21:
        raise ValueError("Expected exactly the 21 original frozen input identities.")
    copied = {}
    inputs = project / "inputs"
    for relative in sorted(identities):
        path = child(inputs, relative)
        item = identities[relative]
        if path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Frozen input differs: {relative}")
        copied[relative] = dict(item)
    extra = child(inputs, SOURCE_08)
    if copied[SOURCE_06]["sha256"] != SOURCE_06_SHA or sha256(extra) != SOURCE_08_SHA:
        raise ValueError("Frozen model source identity differs.")
    copied[SOURCE_08] = {"sha256": SOURCE_08_SHA, "size_bytes": extra.stat().st_size}
    return {"plan_sha256": plan_sha, "input_identities": copied,
            "old_locked_training_manifest_sha256": OLD_LOCK_SHA,
            "old_selected_checkpoints_frozen_sha256": OLD_FROZEN_SHA,
            "training_implementation_sha256": sha256(Path(__file__).resolve()),
            "test_files_hashed_as_opaque_bytes_only": OPAQUE_TEST_INPUTS,
            "test_arrays_loaded": False, "new_period_accessed": False}


def configure_runtime():
    actual_packages = {name: importlib.metadata.version(name) for name in PACKAGES}
    if platform.system() != "Linux" or platform.python_version() != "3.12.13" or actual_packages != PACKAGES:
        raise ValueError("Require the isolated Python 3.12.13 and all five frozen direct package pins.")
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Exactly one CUDA Tesla T4 is required.")
    name = torch.cuda.get_device_name(0)
    if (name != "Tesla T4" or torch.cuda.get_device_capability(0) != (7, 5)
            or torch.version.cuda != "12.8" or torch.backends.cudnn.version() != 91900):
        raise ValueError("Tesla T4 / CUDA 12.8 / cuDNN 91900 runtime differs.")
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=20,
    ).stdout.strip().splitlines()
    if len(driver) != 1 or not driver[0]:
        raise ValueError("Cannot identify the active NVIDIA driver.")
    return torch, {
        "python_version": platform.python_version(), "python_executable": sys.executable,
        "packages": actual_packages, "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(), "gpu_name": name,
        "gpu_count": 1, "device_capability": [7, 5], "driver_version": driver[0],
        "determinism": {"cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                        "deterministic_algorithms": True, "cudnn_deterministic": True,
                        "cudnn_benchmark": False, "cuda_matmul_allow_tf32": False,
                        "cudnn_allow_tf32": False, "float32_matmul_precision": "highest",
                        "amp_enabled": False},
        "threads": 1,
    }


def extract_definitions(source_06, source_08, torch):
    """Compile class/function definitions only; never execute archived top levels."""
    import numpy as np
    if sha256(source_06) != SOURCE_06_SHA or sha256(source_08) != SOURCE_08_SHA:
        raise ValueError("Source changed immediately before AST extraction.")
    namespace = {"torch": torch, "nn": torch.nn, "np": np, "D_MODEL": 32, "N_HEAD": 4,
                 "D_FF": 128, "N_LAYERS": 2, "DROPOUT": 0.1}
    for path, symbols in ((source_06, {"TransformerBlock", "make_head", "NoSharing", "mae_np", "pinball_np"}),
                          (source_08, {"SingleTaskTransformer"})):
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        nodes = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in symbols]
        if {node.name for node in nodes} != symbols or len(nodes) != len(symbols):
            raise ValueError("Required unique archived definitions are missing.")
        isolated = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(isolated, str(path), "exec"), namespace)
    return namespace


def state_identity(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def seeded_initial_pair(namespace, seed, torch):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    template = namespace["NoSharing"](n_features=8, lookback=10, d_model=32,
                                      n_head=4, n_layers=2, d_ff=128, dropout=0.1).cpu()
    if sum(p.numel() for p in template.parameters()) != 54536 or template.pcgrad_groups()["shared"]:
        raise ValueError("The archived NoSharing template does not match its frozen architecture.")
    prefixes = {
        "return": (("ret_projection.", "input_projection."), ("ret_positional", "positional_embedding"),
                   ("ret_encoder.", "encoder."), ("ret_norm.", "norm."), ("return_head.", "head.")),
        "volatility": (("vol_projection.", "input_projection."), ("vol_positional", "positional_embedding"),
                       ("vol_encoder.", "encoder."), ("vol_norm.", "norm."), ("vol_head.", "head.")),
    }
    template_state, models, proofs = template.state_dict(), {}, {}
    for task in TASKS:
        mapped, original = {}, {}
        for source_name, value in template_state.items():
            for old, new in prefixes[task]:
                if source_name == old or source_name.startswith(old):
                    target = new + source_name[len(old):]
                    mapped[target] = value.detach().clone()
                    original[target] = source_name
                    break
        model = namespace["SingleTaskTransformer"](n_features=8, lookback=10).cpu()
        model.load_state_dict(mapped, strict=True)
        if sum(p.numel() for p in model.parameters()) != 27268:
            raise ValueError("A single-task control must have exactly 27,268 parameters.")
        for name, value in model.state_dict().items():
            if not torch.equal(value, template_state[original[name]]):
                raise ValueError("Initial branch-copy verification failed.")
        models[task] = model
        proofs[task] = {"seed": seed, "task": task, "parameters": 27268,
                        "initial_state_sha256": state_identity(model),
                        "prefix_map": [{"NoSharing": old, "SingleTaskTransformer": new}
                                       for old, new in prefixes[task]],
                        "all_initial_tensors_equal_to_corresponding_template_branch": True}
    if {id(p) for p in models["return"].parameters()} & {id(p) for p in models["volatility"].parameters()}:
        raise ValueError("Independent task models share parameter objects.")
    if {p.data_ptr() for p in models["return"].parameters()} & {p.data_ptr() for p in models["volatility"].parameters()}:
        raise ValueError("Independent task models share parameter storage.")
    return models, proofs, template


def synthetic_smoke(namespace, torch, device="cuda"):
    results = []
    for seed in SEEDS:
        models, proofs, template = seeded_initial_pair(namespace, seed, torch)
        template = template.to(device).eval()
        zeros = torch.zeros((2, 10, 8), device=device, dtype=torch.float32)
        with torch.no_grad():
            expected = template(zeros)
        for task in TASKS:
            model = models[task].to(device).eval()
            before = state_identity(model)
            with torch.no_grad():
                evaluation_prediction = model(zeros)
            prediction = model(zeros)
            columns = slice(0, 4) if task == "return" else slice(4, 8)
            if prediction.shape != (2, 4) or not torch.isfinite(prediction).all():
                raise ValueError("Synthetic single-task forward check failed.")
            if not torch.equal(evaluation_prediction, expected[:, columns]):
                raise ValueError("STL and corresponding seeded NoSharing branch outputs differ.")
            prediction.sum().backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Synthetic backward check failed.")
            if state_identity(model) != before:
                raise ValueError("Synthetic smoke check changed model weights.")
            results.append({**proofs[task], "forward_backward_pass": True,
                            "branch_forward_equal_in_eval_mode": True, "optimizer_steps": 0})
        del models, template
    return {"status": "SYNTHETIC_MODELS_PASS_NO_FITTING", "device": device,
            "models_checked": 6, "optimizer_steps": 0, "real_data_predictions": False,
            "test_arrays_loaded": False, "new_period_accessed": False, "checks": results}


def load_train_validation(project):
    import numpy as np
    inputs = project / "inputs"
    schema = read_json(child(inputs, "config/schema_v4.json"))
    if schema["data"]["assets"] != list(ASSETS) or schema["features"]["baseline_dim"] != 8:
        raise ValueError("Frozen feature/asset order differs.")
    data, dates = {}, {}
    for split, n in SPLIT_SIZES.items():
        for name, shape in (("X", (n, 10, 8)), ("y", (n, 8)), ("y_raw", (n, 8))):
            file_name = f"y_{split}_raw.npy" if name == "y_raw" else f"{name}_{split}.npy"
            value = np.load(child(inputs, SEQ + file_name), allow_pickle=False)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid frozen train/validation array: {file_name}")
            data[name + "_" + split] = value
        dates[split] = {}
        for prefix in ("anchor_dates", "target_realization_dates"):
            value = np.load(child(inputs, SEQ + f"{prefix}_{split}.npy"), allow_pickle=False).astype("datetime64[D]")
            if value.shape != (n,) or np.isnat(value).any() or not np.all(np.diff(value) > np.timedelta64(0, "D")):
                raise ValueError("Invalid train/validation date metadata.")
            dates[split][prefix] = value
        if not np.all(dates[split]["target_realization_dates"] > dates[split]["anchor_dates"]):
            raise ValueError("Training/validation targets do not follow their input anchors.")
    if (str(dates["train"]["target_realization_dates"][-1]),
        str(dates["val"]["target_realization_dates"][0]),
        str(dates["val"]["target_realization_dates"][-1])) != ("2020-07-09", "2020-07-10", "2022-10-05"):
        raise ValueError("Frozen train/validation boundaries differ.")
    with child(inputs, "data/sequences/baseline/scalers.pkl").open("rb") as handle:
        scaler = pickle.load(handle)["y_scaler"]
    if (len(scaler.mean_) != 8 or len(scaler.scale_) != 8
            or not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all()
            or not np.all(scaler.scale_ > 0)):
        raise ValueError("Invalid frozen target scaler.")
    for split in SPLIT_SIZES:
        restored = scaler.inverse_transform(data["y_" + split].copy())
        if float(np.max(np.abs(restored - data["y_raw_" + split]))) > 1e-5:
            raise ValueError("Frozen target-scaler round trip failed.")
    denominators = read_json(child(inputs, "data/processed/selection_baseline_denominators_v4.json"))
    for task, family in (("return", "return_denominator"), ("volatility", "volatility_denominator")):
        values = np.array([float(denominators[family][asset]["value"]) for asset in ASSETS])
        if not np.isfinite(values).all() or not np.all(values > 0):
            raise ValueError("Frozen validation denominators must be finite and positive.")
        data[task + "_denominators"] = values
    return data, scaler


def task_slice(task):
    if task == "return":
        return slice(0, 4)
    if task == "volatility":
        return slice(4, 8)
    raise ValueError("Unknown task.")


def task_metrics(raw_truth, raw_prediction, denominators, task, namespace):
    """Preserve 06's per-asset dtype/arithmetic before float64 ratio averaging."""
    import numpy as np
    if task not in TASKS:
        raise ValueError("Unknown task.")
    truth, prediction = np.asarray(raw_truth), np.asarray(raw_prediction)
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != 4:
        raise ValueError("Task metrics require aligned four-output arrays.")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Nonfinite raw task truth or prediction.")
    if len(denominators) != 4 or not all(math.isfinite(float(x)) and float(x) > 0 for x in denominators):
        raise ValueError("Invalid frozen task denominators.")
    # Do not cast the arrays to float64 before these archived helpers. In 06,
    # float32 arrays undergo float32 subtraction/reduction before float(...).
    metric = namespace["mae_np"] if task == "return" else namespace["pinball_np"]
    per_asset = [float(metric(truth[:, i], prediction[:, i])) for i in range(4)]
    ratios = [loss / float(denominator) for loss, denominator in zip(per_asset, denominators)]
    result = {"primary_loss": float(np.mean(per_asset)),
              "task_score": float(np.mean(ratios)), "per_asset_primary_losses": per_asset}
    if not all(math.isfinite(x) for x in (result["primary_loss"], result["task_score"], *per_asset)):
        raise ValueError("Nonfinite raw task metric.")
    return result


def evaluate_task(model, X, raw_truth, scaler, denominators, task, torch, namespace):
    import numpy as np
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(X), 64):
            features = torch.as_tensor(np.asarray(X[start:start + 64]), dtype=torch.float32, device="cuda")
            value = model(features)
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite train/validation prediction.")
            values.append(value.cpu().numpy())
    scaled = np.concatenate(values, axis=0)
    # Use the same eight-dimensional inverse_transform arithmetic/dtype as 06.
    padded = np.zeros((len(scaled), 8), dtype=scaled.dtype)
    padded[:, task_slice(task)] = scaled
    raw = scaler.inverse_transform(padded)[:, task_slice(task)]
    return task_metrics(raw_truth[:, task_slice(task)], raw, denominators, task, namespace)


def save_checkpoint(path, model, metadata, torch):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise RuntimeError("A partial checkpoint already exists; retain it for review.")
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    with temporary.open("xb") as handle:
        torch.save({"model_state_dict": state, **metadata}, handle)
    os.replace(temporary, path)


def validate_history(rows, record):
    if len(rows) != 100 or [int(r["epoch"]) for r in rows] != list(range(1, 101)):
        raise ValueError("Each completed task must have exactly 100 ordered epochs.")
    for row in rows:
        for key in HISTORY_FIELDS:
            if key != "batch_order_sha256" and not math.isfinite(float(row[key])):
                raise ValueError("Nonfinite or missing history value.")
        if int(row["optimizer_steps"]) != int(row["epoch"]) * math.ceil(2714 / 64):
            raise ValueError("History optimizer-step count differs from the fixed budget.")
        digest = row["batch_order_sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid batch-order identity.")
    scores = [float(row["validation_task_score"]) for row in rows]
    best = min(range(100), key=lambda i: scores[i])
    if best + 1 != record["selected_epoch"] or scores[best] != record["selected_validation_task_score"]:
        raise ValueError("Selected task checkpoint violates the earliest normalized-validation minimum.")
    return [row["batch_order_sha256"] for row in rows]


def compare_metrics(actual, expected):
    checks = []
    for split in ("train", "validation"):
        if (len(actual[split]["per_asset_primary_losses"]) != 4
                or len(expected[split]["per_asset_primary_losses"]) != 4):
            raise ValueError("All four asset-level replay checks are required.")
        for key in ("primary_loss", "task_score"):
            checks.append((f"{split}/{key}", actual[split][key], expected[split][key]))
        for i, (a, b) in enumerate(zip(actual[split]["per_asset_primary_losses"], expected[split]["per_asset_primary_losses"])):
            checks.append((f"{split}/asset{i}", a, b))
    if any(not math.isfinite(float(a)) or not math.isfinite(float(b))
           or not math.isclose(float(a), float(b), rel_tol=REL_TOL, abs_tol=ABS_TOL)
           for _, a, b in checks):
        raise ValueError("Selected task checkpoint train/validation numerical replay failed.")
    return {"passed": True, "n_checks": len(checks),
            "maximum_absolute_delta": max(abs(a - b) for _, a, b in checks),
            "relative_tolerance": REL_TOL, "absolute_tolerance": ABS_TOL,
            "test_arrays_loaded": False, "new_period_accessed": False}


def replay_selected(project, job_folder, record, namespace, data, scaler, torch):
    task, seed = record["task"], record["seed"]
    checkpoint = child(project, record["artifacts"]["best_checkpoint"]["path"])
    if sha256(checkpoint) != record["artifacts"]["best_checkpoint"]["sha256"]:
        raise ValueError("Checkpoint changed before numerical replay.")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for key, expected in (("task", task), ("seed", seed), ("selected_epoch", record["selected_epoch"]),
                          ("selected_validation_task_score", record["selected_validation_task_score"]),
                          ("run_lock_sha256", record["run_lock_sha256"]), ("plan_sha256", record["plan_sha256"])):
        if payload.get(key) != expected:
            raise ValueError(f"Checkpoint metadata mismatch: {key}")
    model = namespace["SingleTaskTransformer"](n_features=8, lookback=10)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device="cuda", dtype=torch.float32)
    actual = {"train": evaluate_task(model, data["X_train"], data["y_raw_train"], scaler,
                                    data[task + "_denominators"], task, torch, namespace),
              "validation": evaluate_task(model, data["X_val"], data["y_raw_val"], scaler,
                                         data[task + "_denominators"], task, torch, namespace)}
    expected = read_json(child(project, record["artifacts"]["selected_metrics"]["path"]))
    if expected["validation"]["task_score"] != record["selected_validation_task_score"]:
        raise ValueError("Selected metrics disagree with the frozen selected task score.")
    result = compare_metrics(actual, expected)
    del model
    return result


def validate_completed_job(project, folder, core, lock_sha, namespace, data, scaler, torch,
                           *, numerical_replay=True):
    record_path = child(folder, "completed_job.json")
    record = read_json(record_path)
    if read_json(child(folder, "job_status.json")).get("status") != "STL_TASK_TRAINING_COMPLETE_NO_NEW_DATA":
        raise ValueError("Existing task has not completed all recording steps.")
    if (record.get("status") != "STL_TASK_TRAINING_COMPLETE_NO_NEW_DATA"
            or record.get("plan_sha256") != core["evidence"]["plan_sha256"]
            or record.get("run_lock_sha256") != lock_sha
            or record.get("epochs_ran") != 100 or record.get("parameters") != 27268
            or record.get("test_arrays_loaded") is not False or record.get("new_period_accessed") is not False):
        raise ValueError("Existing task record differs from this frozen run.")
    if folder.name != f"{record['task']}_seed{record['seed']}" or record["task"] not in TASKS or record["seed"] not in SEEDS:
        raise ValueError("Task/seed record is in the wrong output directory.")
    for item in record["artifacts"].values():
        path = child(project, item["path"])
        if not path.is_relative_to(folder) or path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
            raise ValueError("A completed task artifact changed.")
    with child(project, record["artifacts"]["history"]["path"]).open(newline="", encoding="utf-8") as handle:
        order = validate_history(list(csv.DictReader(handle)), record)
    replay = replay_selected(project, folder, record, namespace, data, scaler, torch) if numerical_replay else None
    return record, order, replay


def train_task(project, folder, task, seed, core, lock_sha, namespace, data, scaler, torch):
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset
    models, initialization, template = seeded_initial_pair(namespace, seed, torch)
    model = models[task].to(device="cuda", dtype=torch.float32)
    initialization = initialization[task]
    del models, template
    folder.mkdir(exist_ok=False)
    write_json(folder / "initialization.json", initialization)
    write_json(folder / "job_status.json", {"status": "TRAINING_NO_NEW_DATA", "task": task, "seed": seed})
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    features = torch.tensor(data["X_train"], dtype=torch.float32)
    targets = torch.tensor(data["y_train"][:, task_slice(task)], dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(features, targets, torch.arange(len(features))),
                        batch_size=64, shuffle=True, drop_last=False, generator=generator, num_workers=0)
    history = folder / "history.csv"
    best_score, best_epoch, best_metrics = math.inf, None, None
    steps = 0
    began = time.monotonic()
    metadata = {"architecture": "SingleTaskTransformer", "task": task, "seed": seed,
                "parameters": 27268, "plan_sha256": core["evidence"]["plan_sha256"],
                "run_lock_sha256": lock_sha, "source_06_sha256": SOURCE_06_SHA,
                "source_08B_sha256": SOURCE_08_SHA, "settings": SETTINGS,
                "initialization": initialization, "test_arrays_loaded": False, "new_period_accessed": False}
    try:
        with history.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            for epoch in range(1, 101):
                epoch_start = time.monotonic()
                model.train()
                total, seen, batch_digest = 0.0, 0, hashlib.sha256()
                for x, y, indices in loader:
                    batch_digest.update(indices.numpy().tobytes())
                    x, y = x.to("cuda"), y.to("cuda")
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(x)
                    if task == "return":
                        loss = torch.nn.functional.mse_loss(prediction, y)
                    else:
                        error = y - prediction
                        loss = torch.maximum(0.5 * error, -0.5 * error).mean()
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite independent-task optimization loss.")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if not torch.isfinite(norm):
                        raise ValueError("Nonfinite independent-task gradient norm.")
                    optimizer.step()
                    steps += 1
                    total += float(loss.detach()) * len(x)
                    seen += len(x)
                metrics = {
                    "train": evaluate_task(model, data["X_train"], data["y_raw_train"], scaler,
                                           data[task + "_denominators"], task, torch, namespace),
                    "validation": evaluate_task(model, data["X_val"], data["y_raw_val"], scaler,
                                                data[task + "_denominators"], task, torch, namespace),
                }
                score = metrics["validation"]["task_score"]
                row = {"epoch": epoch, "scaled_optimization_loss": total / seen,
                       "train_primary_loss": metrics["train"]["primary_loss"],
                       "validation_primary_loss": metrics["validation"]["primary_loss"],
                       "validation_gap": metrics["validation"]["primary_loss"] - metrics["train"]["primary_loss"],
                       "train_task_score": metrics["train"]["task_score"], "validation_task_score": score,
                       "batch_order_sha256": batch_digest.hexdigest(), "optimizer_steps": steps,
                       "epoch_seconds": time.monotonic() - epoch_start}
                writer.writerow(row)
                handle.flush()
                if score < best_score:
                    best_score, best_epoch, best_metrics = score, epoch, metrics
                    save_checkpoint(folder / "best_checkpoint.pt", model,
                                    {**metadata, "selected_epoch": epoch, "selected_validation_task_score": score}, torch)
                print(f"[STL {task} seed={seed}] epoch={epoch:03d}/100 task_score={score:.8f}", flush=True)
        save_checkpoint(folder / "final_checkpoint.pt", model,
                        {**metadata, "selected_epoch": 100, "selected_validation_task_score": score}, torch)
        write_json(folder / "selected_train_validation_metrics.json", best_metrics)
        write_json(folder / "final_train_validation_metrics.json", metrics)
        record = {"status": "STL_TASK_TRAINING_COMPLETE_NO_NEW_DATA", "task": task, "seed": seed,
                  "epochs_ran": 100, "optimizer_steps": steps, "parameters": 27268,
                  "selected_epoch": best_epoch, "selected_validation_task_score": best_score,
                  "plan_sha256": core["evidence"]["plan_sha256"], "run_lock_sha256": lock_sha,
                  "initialization": initialization, "runtime_seconds": time.monotonic() - began,
                  "test_arrays_loaded": False, "new_period_accessed": False,
                  "artifacts": {}}
        names = {"best_checkpoint": "best_checkpoint.pt", "final_checkpoint": "final_checkpoint.pt",
                 "history": "history.csv", "selected_metrics": "selected_train_validation_metrics.json",
                 "final_metrics": "final_train_validation_metrics.json", "initialization": "initialization.json"}
        for key, name in names.items():
            path = folder / name
            record["artifacts"][key] = {"path": str(path.relative_to(project)), "sha256": sha256(path),
                                         "size_bytes": path.stat().st_size}
        with history.open(newline="", encoding="utf-8") as handle:
            validate_history(list(csv.DictReader(handle)), record)
        replay = replay_selected(project, folder, record, namespace, data, scaler, torch)
        write_json(folder / "train_validation_replay.json", replay)
        path = folder / "train_validation_replay.json"
        record["artifacts"]["replay"] = {"path": str(path.relative_to(project)), "sha256": sha256(path),
                                          "size_bytes": path.stat().st_size}
        write_json(folder / "completed_job.json", record)
        write_json(folder / "job_status.json", {"status": record["status"], "task": task, "seed": seed}, exclusive=False)
        return record
    except BaseException as exc:
        write_json(folder / "job_status.json", {"status": "PARTIAL_TASK_RETAINED_NO_NEW_DATA",
                   "task": task, "seed": seed, "error_type": type(exc).__name__, "error": str(exc)}, exclusive=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-only", action="store_true", help="Read-only frozen-input and synthetic model checks; no fitting or writes.")
    modes.add_argument("--train", action="store_true", help="Explicitly run six independent-task fits in the dedicated new project.")
    args = parser.parse_args()
    project = validate_project(args.project_root)
    evidence = read_evidence(project)
    torch, runtime = configure_runtime()
    namespace = extract_definitions(child(project / "inputs", SOURCE_06), child(project / "inputs", SOURCE_08), torch)
    data, scaler = load_train_validation(project)
    smoke = synthetic_smoke(namespace, torch)
    if args.prepare_only:
        if read_evidence(project) != evidence:
            raise ValueError("A staged scientific plan or source changed during preparation.")
        print(json.dumps({"status": "STL_PREPARATION_PASS_NO_TRAINING_NO_NEW_DATA", "evidence": evidence,
                          "runtime_environment": runtime, "synthetic_checks": smoke,
                          "training_started": False, "file_writes": False, "test_arrays_loaded": False,
                          "new_period_accessed": False}, ensure_ascii=False, allow_nan=False))
        return
    output = child(project, "runs/stl_run_v1", must_exist=False)
    if output.exists() and not output.is_dir():
        raise ValueError("Dedicated STL output is not a directory.")
    core = {"schema_version": 1, "purpose": "STL_CONTROLS_FROZEN_BEFORE_NEW_DATA",
            "project_root": str(project), "output_root": str(output), "evidence": evidence,
            "runtime_environment": runtime, "settings": SETTINGS,
            "initialization_smoke_proofs": smoke, "test_arrays_loaded": False, "new_period_accessed": False}
    lock_path = output / "run_lock.json"
    if output.exists():
        if not lock_path.is_file():
            raise RuntimeError("A partial output directory lacks its run lock; retain it for review.")
        stored_lock = read_json(lock_path)
        if stored_lock.get("core") != core:
            raise ValueError("Existing STL run uses another plan, implementation, input or runtime.")
        if any(p.name not in {"run_lock.json", "jobs", "selected_stl_checkpoints_frozen.json"} for p in output.iterdir()):
            raise ValueError("Unrecognized file in the dedicated STL run directory.")
    else:
        output.parent.mkdir(exist_ok=True)
        output.mkdir(exist_ok=False)
        write_json(lock_path, {"created_at_utc": now(), "core": core})
    lock_sha = sha256(lock_path)
    jobs = output / "jobs"
    reject_symlinks(jobs)
    jobs.mkdir(exist_ok=True)
    names = {f"{task}_seed{seed}" for seed in SEEDS for task in TASKS}
    if any(p.name not in names or not p.is_dir() or p.is_symlink() for p in jobs.iterdir()):
        raise ValueError("Unrecognized or unsafe task output directory.")
    # A partial task is never silently resumed, deleted or replaced.
    for folder in jobs.iterdir():
        if not (folder / "completed_job.json").is_file():
            raise RuntimeError(f"Partial task retained without overwrite: {folder.name}")
    frozen_path = output / "selected_stl_checkpoints_frozen.json"
    # Validate/replay every pre-existing complete task before starting a new fit.
    prevalidated = {
        folder.name: validate_completed_job(project, folder, core, lock_sha, namespace, data, scaler, torch)
        for folder in jobs.iterdir()
    }
    records, orders, replays = [], {}, []
    for seed in SEEDS:
        for task in TASKS:
            folder = jobs / f"{task}_seed{seed}"
            if not folder.exists():
                if frozen_path.exists():
                    raise ValueError("A frozen run is missing a completed task; no new training was started.")
                train_task(project, folder, task, seed, core, lock_sha, namespace, data, scaler, torch)
            if folder.name in prevalidated:
                record, order, replay = prevalidated[folder.name]
            else:
                record, order, replay = validate_completed_job(project, folder, core, lock_sha, namespace, data, scaler, torch)
            if seed in orders and orders[seed] != order:
                raise ValueError("Paired STL tasks have different minibatch permutations.")
            orders[seed] = order
            records.append(record)
            replays.append({"task": task, "seed": seed, **replay})
            print(f"[STL VERIFIED] {task} seed={seed} selected_epoch={record['selected_epoch']}", flush=True)
    if len(records) != 6 or read_evidence(project) != evidence or sha256(lock_path) != lock_sha:
        raise ValueError("The six completed controls or their frozen evidence changed.")
    _, final_runtime = configure_runtime()
    if final_runtime != runtime:
        raise ValueError("Runtime changed before the six-task freeze.")
    # Recheck all six metadata/history/checkpoint identities together after the
    # last fit. In particular an early job may not drift during later fits.
    for earlier in records:
        folder = jobs / f"{earlier['task']}_seed{earlier['seed']}"
        current, _, _ = validate_completed_job(project, folder, core, lock_sha, namespace,
                                               data, scaler, torch, numerical_replay=False)
        if current != earlier:
            raise ValueError("A completed task record changed before publication.")
    checkpoints = []
    for record in records:
        item = record["artifacts"]["best_checkpoint"]
        metrics = read_json(child(project, record["artifacts"]["selected_metrics"]["path"]))
        checkpoints.append({"task": record["task"], "seed": record["seed"],
                            "path": item["path"], "sha256": item["sha256"], "size_bytes": item["size_bytes"],
                            "selected_epoch": record["selected_epoch"], "parameters": 27268,
                            "selected_validation_task_score": record["selected_validation_task_score"],
                            "train_primary_loss": metrics["train"]["primary_loss"],
                            "validation_primary_loss": metrics["validation"]["primary_loss"],
                            "completed_job_path": str((jobs / f"{record['task']}_seed{record['seed']}" / "completed_job.json").relative_to(project)),
                            "completed_job_sha256": sha256(jobs / f"{record['task']}_seed{record['seed']}" / "completed_job.json")})
    frozen = {"status": "STL_CHECKPOINTS_FROZEN_BEFORE_NEW_DATA", "plan_sha256": evidence["plan_sha256"],
              "run_lock_path": str(lock_path.relative_to(project)), "run_lock_sha256": lock_sha,
              "training_implementation_sha256": evidence["training_implementation_sha256"],
              "runtime_environment": runtime, "settings": SETTINGS, "checkpoints": checkpoints,
              "paired_seed_task_minibatch_orders_equal": True, "train_validation_replays": replays,
              "n_actual_fits": 6, "test_arrays_loaded": False, "new_period_accessed": False,
              "old_project_writes": False}
    if frozen_path.exists():
        existing = read_json(frozen_path)
        previous_replays = existing.get("train_validation_replays", [])
        if (len(previous_replays) != 6
                or {(item.get("task"), item.get("seed")) for item in previous_replays}
                != {(task, seed) for task in TASKS for seed in SEEDS}
                or any(item.get("passed") is not True or item.get("n_checks") != 12
                       or item.get("relative_tolerance") != REL_TOL
                       or item.get("absolute_tolerance") != ABS_TOL
                       or item.get("test_arrays_loaded") is not False
                       or item.get("new_period_accessed") is not False for item in previous_replays)):
            raise ValueError("Existing frozen replay evidence is incomplete.")
        # Fresh replay deltas need not be byte-identical; all tolerance checks
        # have already passed. The frozen scientific identities must be exact.
        for key in frozen:
            if key != "train_validation_replays" and existing.get(key) != frozen[key]:
                raise ValueError(f"Existing frozen manifest differs: {key}")
    else:
        write_json(frozen_path, {"completed_at_utc": now(), **frozen})
    print(json.dumps({"status": frozen["status"], "actual_fits": 6,
                      "record": str(frozen_path), "record_sha256": sha256(frozen_path),
                      "plan_sha256": evidence["plan_sha256"], "all_six_train_validation_replays_passed": True,
                      "test_arrays_loaded": False, "new_period_accessed": False}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
