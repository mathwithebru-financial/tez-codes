#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
08C — Locked expanding-window GARCH-family volatility baselines.

Official models
---------------
1) GARCH(1,1), Zero mean, standardized Student-t
2) GJR-GARCH(1,1), Zero mean, standardized Student-t

Core safeguards
---------------
- Verifies the locked protocol SHA-256 before any fit.
- Verifies all 10 locked input files by exact SHA-256 and size.
- Uses the exact 584 locked test anchors and NextVol targets.
- Re-fits independently at every anchor; no warm start.
- Captures all warnings and uses convergence_flag / optimizer success
  as the primary convergence decision.
- Allows one deterministic numerical retry only.
- Never uses persistence, normal-distribution, parameter, or prediction fallback.
- Checkpoints each model-asset pair and safely resumes after interruption.
- Does not compute final metrics unless all 4,672 observation-level runs resolve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
import scipy
from scipy.optimize import brentq

import arch
from arch import arch_model
from arch.univariate.distribution import StudentsT
from arch.utility.exceptions import ConvergenceWarning


PROJECT_VERSION = "v4_repro"
STAGE = "08C"

DEFAULT_ROOT = Path("/content/drive/MyDrive/tez_transformer_v4_repro")

PROTOCOL_RELATIVE_PATH = Path("config/08C_garch_protocol_lock_v4.json")
PROTOCOL_SHA_RELATIVE_PATH = Path("config/08C_garch_protocol_lock_v4.sha256")

EXPECTED_PROTOCOL_SHA256 = (
    "4238b4280021e8a265dcb6ba0aa6da379d1388db7d4396e29463d476283a6614"
)

OUTPUT_RELATIVE_DIR = Path("results/baselines/garch")
CHECKPOINT_RELATIVE_DIR = OUTPUT_RELATIVE_DIR / "checkpoints"

FINAL_PRED_RELATIVE_PATH = Path(
    "results/final_test/pred_final_ensemble_raw_v4.npy"
)
FINAL_TRUTH_RELATIVE_PATH = Path(
    "results/final_test/final_test_y_true_raw_v4.npy"
)
VOL_PERSISTENCE_PRED_RELATIVE_PATH = Path(
    "results/baselines/naive/pred_vol_persistence_raw_v4.npy"
)
NAIVE_LOSS_SERIES_RELATIVE_PATH = Path(
    "results/baselines/naive/naive_baseline_loss_series_v4.npz"
)

RUN_MANIFEST_START_FILENAME = "08C_run_manifest_start_v4.json"
SUMMARY_FILENAME = "garch_baseline_summary_v4.json"
OUTPUT_MANIFEST_FILENAME = "garch_output_manifest_v4.json"

DIAGNOSTICS_FILENAME = "garch_baseline_diagnostics_long_v4.csv"
ATTEMPTS_FILENAME = "garch_fit_attempts_v4.csv"
WARNINGS_FILENAME = "garch_warning_inventory_v4.csv"
METRICS_FILENAME = "garch_baseline_metrics_long_v4.csv"
TASK_AVERAGE_FILENAME = "garch_baseline_task_average_v4.csv"
FINAL_COMPARISON_FILENAME = "garch_final_comparison_v4.csv"
PERSISTENCE_COMPARISON_FILENAME = "garch_vol_persistence_comparison_v4.csv"
LOSS_SERIES_FILENAME = "garch_baseline_loss_series_v4.npz"
DATES_ANCHOR_FILENAME = "anchor_dates_08C_v4.npy"
DATES_REALIZATION_FILENAME = "target_realization_dates_08C_v4.npy"

NEGATIVE_TOLERANCE = 1e-14

WARNING_CHECKPOINT_COLUMNS = [
    "protocol_sha256",
    "script_sha256",
    "model_id",
    "asset",
    "anchor_index",
    "anchor_date",
    "attempt_number",
    "warning_index",
    "warning_category",
    "warning_message",
    "warning_filename",
    "warning_lineno",
    "is_convergence_warning",
]


# =============================================================================
# Generic utilities
# =============================================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"JSON serialization unsupported for {type(value).__name__}")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temp:
        json.dump(
            payload,
            temp,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=json_default,
        )
        temp.write("\n")
        temp_path = Path(temp.name)
    os.replace(temp_path, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temp:
        frame.to_csv(temp, index=False)
        temp_path = Path(temp.name)
    os.replace(temp_path, path)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temp:
        np.save(temp, array, allow_pickle=False)
        temp_path = Path(temp.name)
    os.replace(temp_path, path)


def atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temp:
        np.savez_compressed(temp, **arrays)
        temp_path = Path(temp.name)
    os.replace(temp_path, path)


def version_inventory() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "arch": arch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }


def normalize_dates(values: np.ndarray | Iterable[Any]) -> pd.DatetimeIndex:
    raw = np.asarray(values)
    return pd.DatetimeIndex(pd.to_datetime(raw.astype(str))).normalize()


def slugify_model_id(model_id: str) -> str:
    return (
        model_id.lower()
        .replace("students", "student")
        .replace("_zero_mean", "")
        .replace("__", "_")
    )


def safe_json_text(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        default=json_default,
    )


def read_csv_checkpoint(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        # Backward-safe handling for a headerless empty checkpoint created
        # before the fixed warning-checkpoint schema was introduced.
        return pd.DataFrame()


# =============================================================================
# Protocol and integrity validation
# =============================================================================

def load_and_verify_protocol(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = root / PROTOCOL_RELATIVE_PATH
    protocol_sha_path = root / PROTOCOL_SHA_RELATIVE_PATH

    require(protocol_path.exists(), f"Protocol missing: {protocol_path}")
    require(protocol_sha_path.exists(), f"Protocol SHA file missing: {protocol_sha_path}")

    actual_sha = sha256_file(protocol_path)
    sha_file_token = protocol_sha_path.read_text(
        encoding="utf-8",
        errors="strict",
    ).strip().split()[0]

    require(
        actual_sha == EXPECTED_PROTOCOL_SHA256,
        "Locked protocol SHA mismatch against script constant.\n"
        f"Expected: {EXPECTED_PROTOCOL_SHA256}\n"
        f"Actual  : {actual_sha}",
    )
    require(
        sha_file_token == EXPECTED_PROTOCOL_SHA256,
        "Companion SHA file does not match the locked protocol SHA.",
    )

    with protocol_path.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)

    require(protocol.get("project_version") == PROJECT_VERSION, "Project version mismatch.")
    require(protocol.get("stage") == STAGE, "Protocol stage mismatch.")
    require(
        protocol.get("status")
        == "LOCKED_BEFORE_08C_GARCH_FITS_PREDICTIONS_AND_METRICS",
        "Unexpected 08C protocol status.",
    )
    require(protocol.get("task") == "volatility_only", "Protocol task mismatch.")
    require(
        protocol["expected_inventory"]["primary_fits_without_retries"] == 4672,
        "Expected fit inventory is not 4,672.",
    )
    require(
        protocol["warm_start_policy"]["warm_start_allowed"] is False,
        "Warm start must be forbidden.",
    )
    require(
        protocol["warm_start_policy"]["starting_values"] is None,
        "starting_values must be null.",
    )
    require(
        float(protocol["metrics"]["primary"]["tau"]) == 0.5,
        "The locked primary PinballLoss tau must be 0.5.",
    )
    require(
        float(protocol["numerical_safety"]["negative_variance_tolerance"])
        == NEGATIVE_TOLERANCE,
        "Negative-value tolerance mismatch.",
    )

    target_definition = protocol["target_definition"]
    require(
        int(target_definition["vol20_window"]) == 20,
        "The locked Vol20 window must be 20.",
    )
    require(
        int(target_definition["vol20_min_periods"]) == 20,
        "The locked Vol20 min_periods must be 20.",
    )
    require(
        int(target_definition["vol20_ddof"]) == 1,
        "The locked Vol20 ddof must be 1.",
    )
    require(
        float(target_definition["annualization_factor"]) > 0.0,
        "Annualization factor must be positive.",
    )

    fit_scale = protocol["fit_scale"]
    for scale_key in (
        "input_log_return_multiplier",
        "sigma_to_decimal_divisor",
        "variance_to_decimal_divisor",
    ):
        require(
            float(fit_scale[scale_key]) > 0.0,
            f"Locked fit-scale value must be positive: {scale_key}",
        )

    environment = version_inventory()
    required_versions = dict(protocol["environment"]["required_package_versions"])
    required_versions["python"] = protocol["environment"]["python"]

    for package_name, required_version in required_versions.items():
        observed = environment[package_name]
        require(
            observed == required_version,
            f"Environment mismatch for {package_name}: "
            f"required={required_version}, observed={observed}",
        )

    protocol_info = {
        "protocol_path": str(protocol_path),
        "protocol_sha_path": str(protocol_sha_path),
        "protocol_sha256": actual_sha,
        "protocol_size_bytes": protocol_path.stat().st_size,
        "environment": environment,
    }
    return protocol, protocol_info


def verify_locked_inputs(
    root: Path,
    protocol: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    verified: dict[str, dict[str, Any]] = {}

    groups = (
        ("official_execution_inputs", protocol["data_lock"]["official_execution_inputs"]),
        (
            "independent_verification_inputs",
            protocol["data_lock"]["independent_verification_inputs"],
        ),
    )

    for group_name, entries in groups:
        for file_key, spec in entries.items():
            path = root / spec["relative_path"]
            require(path.exists(), f"Locked input missing: {path}")

            observed_size = path.stat().st_size
            observed_sha = sha256_file(path)

            require(
                observed_size == int(spec["size_bytes"]),
                f"Size mismatch for {file_key}: "
                f"expected={spec['size_bytes']}, observed={observed_size}",
            )
            require(
                observed_sha == spec["sha256"],
                f"SHA mismatch for {file_key}: "
                f"expected={spec['sha256']}, observed={observed_sha}",
            )

            verified[file_key] = {
                "group": group_name,
                "relative_path": spec["relative_path"],
                "size_bytes": observed_size,
                "sha256": observed_sha,
            }

    require(len(verified) == 10, f"Expected 10 locked inputs, found {len(verified)}.")
    return verified


def read_indexed_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).normalize()
    require(frame.index.is_unique, f"Index is not unique: {path}")
    require(frame.index.is_monotonic_increasing, f"Index is not chronological: {path}")
    return frame


def validate_data_alignment(
    root: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    official = protocol["data_lock"]["official_execution_inputs"]
    independent = protocol["data_lock"]["independent_verification_inputs"]

    features_path = root / official["features_baseline"]["relative_path"]
    anchor_path = root / official["anchor_dates_test"]["relative_path"]
    realization_path = root / official["target_realization_dates_test"]["relative_path"]
    y_test_path = root / official["y_test_raw"]["relative_path"]

    features = read_indexed_csv(features_path)
    anchor_raw = np.load(anchor_path, allow_pickle=False)
    realization_raw = np.load(realization_path, allow_pickle=False)
    y_test_raw = np.load(y_test_path, allow_pickle=False)

    assets = list(protocol["assets"])
    return_columns = list(protocol["data_lock"]["official_return_source"]["columns"])
    target_columns = list(protocol["target_columns"])
    target_indices = list(protocol["target_indices"])

    require(features.shape[0] == 3891, f"Unexpected features row count: {features.shape[0]}")
    require(
        features.index[0] == pd.Timestamp("2010-02-01"),
        f"Unexpected common-index start: {features.index[0]}",
    )
    require(
        features.index[-1] == pd.Timestamp("2024-12-30"),
        f"Unexpected common-index end: {features.index[-1]}",
    )
    require(
        all(column in features.columns for column in return_columns),
        "One or more locked LogRet columns are missing.",
    )

    returns = features[return_columns].to_numpy(dtype=np.float64)
    require(np.isfinite(returns).all(), "Official return matrix contains non-finite values.")

    anchor_dates = normalize_dates(anchor_raw)
    realization_dates = normalize_dates(realization_raw)

    require(len(anchor_dates) == 584, "Anchor count must be 584.")
    require(len(realization_dates) == 584, "Realization-date count must be 584.")
    require(anchor_dates.is_unique, "Anchor dates are not unique.")
    require(realization_dates.is_unique, "Realization dates are not unique.")
    require(anchor_dates.is_monotonic_increasing, "Anchor dates are not chronological.")
    require(realization_dates.is_monotonic_increasing, "Realization dates are not chronological.")

    require(anchor_dates[0] == pd.Timestamp("2022-10-05"), "First anchor mismatch.")
    require(anchor_dates[-1] == pd.Timestamp("2024-12-30"), "Last anchor mismatch.")
    require(
        realization_dates[0] == pd.Timestamp("2022-10-06"),
        "First realization date mismatch.",
    )
    require(
        realization_dates[-1] == pd.Timestamp("2024-12-31"),
        "Last realization date mismatch.",
    )

    anchor_positions = features.index.get_indexer(anchor_dates)
    require((anchor_positions >= 0).all(), "At least one anchor is absent from features.")
    expected_positions = np.arange(3307, 3891, dtype=np.int64)
    require(
        np.array_equal(anchor_positions, expected_positions),
        "Anchor positions are not exactly 3307..3890.",
    )

    require(y_test_raw.shape == (584, 8), f"Unexpected y_test shape: {y_test_raw.shape}")
    require(str(y_test_raw.dtype) == "float32", f"Unexpected y_test dtype: {y_test_raw.dtype}")

    targets_all = read_indexed_csv(root / independent["targets_all"]["relative_path"])
    target_from_csv = targets_all.loc[anchor_dates, target_columns].to_numpy()
    target_cast = target_from_csv.astype(y_test_raw.dtype, copy=False)
    official_vol_truth = y_test_raw[:, target_indices]

    require(
        np.array_equal(target_cast, official_vol_truth),
        "Official NPY targets are not exactly equal to dtype-aligned CSV targets.",
    )

    # Independent LogRet reconstruction from prices.
    prices = read_indexed_csv(root / independent["prices_clean"]["relative_path"])
    max_diff_by_asset: dict[str, float] = {}
    allowed = float(
        protocol["data_lock"]["logret_independent_reconstruction"][
            "maximum_allowed_diff"
        ]
    )

    for asset, return_column in zip(assets, return_columns):
        require(asset in prices.columns, f"Price column missing for {asset}.")
        reconstructed = np.log(
            prices[asset].astype(np.float64)
            / prices[asset].astype(np.float64).shift(1)
        ).reindex(features.index)
        observed = features[return_column].astype(np.float64)
        diff = np.abs(reconstructed.to_numpy() - observed.to_numpy())
        max_diff = float(np.nanmax(diff))
        max_diff_by_asset[asset] = max_diff
        require(
            max_diff <= allowed,
            f"Independent LogRet reconstruction failed for {asset}: {max_diff}",
        )

    # Comparison inputs from locked earlier stages.
    final_pred_path = root / FINAL_PRED_RELATIVE_PATH
    final_truth_path = root / FINAL_TRUTH_RELATIVE_PATH
    persistence_path = root / VOL_PERSISTENCE_PRED_RELATIVE_PATH
    naive_loss_path = root / NAIVE_LOSS_SERIES_RELATIVE_PATH

    for path in (final_pred_path, final_truth_path, persistence_path, naive_loss_path):
        require(path.exists(), f"Required earlier-stage comparison file missing: {path}")

    final_pred = np.load(final_pred_path, allow_pickle=False)
    final_truth = np.load(final_truth_path, allow_pickle=False)
    vol_persistence_pred = np.load(persistence_path, allow_pickle=False)

    require(final_pred.shape == (584, 8), "Final prediction shape mismatch.")
    require(final_truth.shape == (584, 8), "Final truth shape mismatch.")
    require(vol_persistence_pred.shape == (584, 4), "VolPersistence shape mismatch.")
    require(
        np.array_equal(final_truth, y_test_raw),
        "07 final_test_y_true_raw is not exactly equal to official y_test_raw.",
    )
    require(np.isfinite(final_pred).all(), "Final predictions contain non-finite values.")
    require(
        np.isfinite(vol_persistence_pred).all(),
        "VolPersistence predictions contain non-finite values.",
    )

    # Reconstruct and verify the stored 08A final-model and
    # VolPersistence volatility loss series. Exact equality is required
    # after aligning the reconstruction to the stored series dtype.
    tau = float(protocol["metrics"]["primary"]["tau"])
    with np.load(naive_loss_path, allow_pickle=False) as archive:
        for asset_index, asset in enumerate(assets):
            checks = (
                (
                    f"final_vol_pinball__{asset}",
                    final_pred[:, 4 + asset_index],
                    "Final",
                ),
                (
                    f"vol_persistence_pinball__{asset}",
                    vol_persistence_pred[:, asset_index],
                    "VolPersistence",
                ),
            )

            for key, prediction, label in checks:
                require(key in archive.files, f"08A loss key missing: {key}")
                stored = np.asarray(archive[key])
                reconstructed = pinball_loss_series(
                    official_vol_truth[:, asset_index].astype(np.float64),
                    np.asarray(prediction, dtype=np.float64),
                    tau=tau,
                )
                reconstructed_aligned = reconstructed.astype(
                    stored.dtype,
                    copy=False,
                )
                require(
                    np.array_equal(stored, reconstructed_aligned),
                    f"08A {label} loss series mismatch for {asset}.",
                )

    comparison_inputs = {}
    for label, path in {
        "final_prediction": final_pred_path,
        "final_truth": final_truth_path,
        "vol_persistence_prediction": persistence_path,
        "naive_loss_series": naive_loss_path,
    }.items():
        comparison_inputs[label] = {
            "relative_path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    return {
        "features": features,
        "returns": returns,
        "anchor_raw": anchor_raw,
        "realization_raw": realization_raw,
        "anchor_dates": anchor_dates,
        "realization_dates": realization_dates,
        "anchor_positions": anchor_positions,
        "y_test_raw": y_test_raw,
        "y_vol_truth": official_vol_truth.astype(np.float64),
        "final_pred": final_pred.astype(np.float64),
        "vol_persistence_pred": vol_persistence_pred.astype(np.float64),
        "max_logret_reconstruction_diff_by_asset": max_diff_by_asset,
        "comparison_inputs": comparison_inputs,
    }


# =============================================================================
# Numerical and statistical functions
# =============================================================================

def safe_nonnegative(
    value: float,
    *,
    name: str,
    tolerance: float = NEGATIVE_TOLERANCE,
) -> tuple[float, bool]:
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError(f"{name} is non-finite: {value}")
    if value < -tolerance:
        raise FloatingPointError(
            f"{name}={value} is below the allowed negative tolerance {-tolerance}."
        )
    if value < 0.0:
        return 0.0, True
    return value, False


def pinball_loss_series(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    tau: float,
) -> np.ndarray:
    residual = np.asarray(y_true, dtype=np.float64) - np.asarray(
        y_pred,
        dtype=np.float64,
    )
    return np.maximum(tau * residual, (tau - 1.0) * residual)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    tau: float,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    require(y_true.shape == y_pred.shape, "Metric arrays have different shapes.")
    require(np.isfinite(y_true).all(), "Metric truth contains non-finite values.")
    require(np.isfinite(y_pred).all(), "Metric prediction contains non-finite values.")

    error = y_true - y_pred
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))

    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = (
        float(1.0 - np.sum(error**2) / denominator)
        if denominator > 0.0
        else float("nan")
    )
    pinball = float(np.mean(pinball_loss_series(y_true, y_pred, tau=tau)))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "PinballLoss_tau_0.5": pinball,
    }


def standardized_t_cdf(
    distribution: StudentsT,
    x_decimal: float,
    *,
    sigma_decimal: float,
    nu: float,
) -> float:
    z = float(x_decimal) / float(sigma_decimal)
    cdf_value = distribution.cdf(
        np.asarray([z], dtype=np.float64),
        parameters=np.asarray([nu], dtype=np.float64),
    )
    return float(np.asarray(cdf_value, dtype=np.float64).reshape(-1)[0])


def solve_conditional_median_q(
    *,
    c: float,
    sigma_decimal: float,
    nu: float,
    root_settings: dict[str, Any],
    distribution: StudentsT,
) -> dict[str, float | int]:
    require(math.isfinite(c), "c is non-finite.")
    require(
        math.isfinite(sigma_decimal) and sigma_decimal > 0.0,
        "sigma_decimal must be finite and positive.",
    )
    require(math.isfinite(nu) and nu > 2.0, "nu must be finite and > 2.")

    lower = float(root_settings["lower_bound"])
    upper = max(sigma_decimal, abs(c), 1e-12)
    maximum_doublings = int(root_settings["maximum_upper_bound_doublings"])

    def equation(q: float) -> float:
        upper_cdf = standardized_t_cdf(
            distribution,
            c + q,
            sigma_decimal=sigma_decimal,
            nu=nu,
        )
        lower_cdf = standardized_t_cdf(
            distribution,
            c - q,
            sigma_decimal=sigma_decimal,
            nu=nu,
        )
        return (upper_cdf - lower_cdf) - 0.5

    f_lower = equation(lower)
    require(math.isfinite(f_lower), "Root function is non-finite at lower bound.")
    require(f_lower <= 0.0, f"Unexpected root sign at lower bound: {f_lower}")

    doublings = 0
    f_upper = equation(upper)
    while (not math.isfinite(f_upper) or f_upper < 0.0) and doublings < maximum_doublings:
        upper *= 2.0
        doublings += 1
        f_upper = equation(upper)

    require(math.isfinite(f_upper), "Root function stayed non-finite at upper bound.")
    require(
        f_upper >= 0.0,
        "CDF root could not be bracketed within the locked doubling limit.",
    )

    q = float(
        brentq(
            equation,
            lower,
            upper,
            xtol=float(root_settings["xtol"]),
            rtol=float(root_settings["rtol"]),
            maxiter=int(root_settings["maxiter"]),
        )
    )
    require(math.isfinite(q) and q >= 0.0, "Solved q is invalid.")

    central_mass = (
        standardized_t_cdf(
            distribution,
            c + q,
            sigma_decimal=sigma_decimal,
            nu=nu,
        )
        - standardized_t_cdf(
            distribution,
            c - q,
            sigma_decimal=sigma_decimal,
            nu=nu,
        )
    )
    mass_error = abs(float(central_mass) - 0.5)
    maximum_mass_error = float(root_settings["maximum_allowed_mass_error"])
    require(
        mass_error <= maximum_mass_error,
        f"Central probability mass error {mass_error} exceeds {maximum_mass_error}.",
    )

    return {
        "q": q,
        "central_probability_mass": float(central_mass),
        "central_mass_error": float(mass_error),
        "root_lower_bound": lower,
        "root_upper_bound": float(upper),
        "root_upper_bound_doublings": doublings,
    }


# =============================================================================
# Fit and forecast execution
# =============================================================================

def warning_records(
    caught: list[warnings.WarningMessage],
    *,
    protocol_sha: str,
    script_sha: str,
    model_id: str,
    asset: str,
    anchor_index: int,
    anchor_date: str,
    attempt_number: int,
) -> list[dict[str, Any]]:
    records = []
    for warning_index, item in enumerate(caught, start=1):
        category = item.category
        is_convergence = issubclass(category, ConvergenceWarning)
        records.append(
            {
                "protocol_sha256": protocol_sha,
                "script_sha256": script_sha,
                "model_id": model_id,
                "asset": asset,
                "anchor_index": anchor_index,
                "anchor_date": anchor_date,
                "attempt_number": attempt_number,
                "warning_index": warning_index,
                "warning_category": category.__name__,
                "warning_message": str(item.message),
                "warning_filename": str(item.filename),
                "warning_lineno": int(item.lineno),
                "is_convergence_warning": bool(is_convergence),
            }
        )
    return records


def extract_parameter(params: pd.Series, name: str) -> float:
    if name not in params.index:
        return float("nan")
    return float(params[name])


def fit_one_attempt(
    *,
    return_history_decimal: np.ndarray,
    model_spec: dict[str, Any],
    attempt_settings: dict[str, Any],
    fit_scale: dict[str, Any],
    protocol_sha: str,
    script_sha: str,
    model_id: str,
    asset: str,
    anchor_index: int,
    anchor_date: str,
    attempt_number: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    start = time.perf_counter()
    caught: list[warnings.WarningMessage] = []
    result = None
    failure_reasons: list[str] = []
    exception_type = ""
    exception_message = ""

    try:
        input_multiplier = float(fit_scale["input_log_return_multiplier"])
        variance_divisor = float(fit_scale["variance_to_decimal_divisor"])
        sigma_divisor = float(fit_scale["sigma_to_decimal_divisor"])

        require(input_multiplier > 0.0, "Input return multiplier must be positive.")
        require(variance_divisor > 0.0, "Variance divisor must be positive.")
        require(sigma_divisor > 0.0, "Sigma divisor must be positive.")

        fit_input_percent = (
            np.asarray(return_history_decimal, dtype=np.float64)
            * input_multiplier
        )
        require(
            np.isfinite(fit_input_percent).all(),
            "Fit input contains non-finite values.",
        )

        model = arch_model(
            fit_input_percent,
            mean=model_spec["mean"],
            vol=model_spec["vol"],
            p=int(model_spec["p"]),
            o=int(model_spec["o"]),
            q=int(model_spec["q"]),
            power=float(model_spec["power"]),
            dist=model_spec["dist"],
            rescale=bool(model_spec["rescale"]),
        )

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = model.fit(
                update_freq=int(attempt_settings["update_freq"]),
                disp=attempt_settings["disp"],
                show_warning=bool(attempt_settings["show_warning"]),
                starting_values=attempt_settings["starting_values"],
                cov_type=attempt_settings["cov_type"],
                tol=float(attempt_settings["tol"]),
                options=dict(attempt_settings["options"]),
                backcast=attempt_settings["backcast"],
            )
            caught = list(captured)

    except Exception as error:
        exception_type = type(error).__name__
        exception_message = str(error)
        failure_reasons.append(f"fit_exception:{exception_type}")

    warning_rows = warning_records(
        caught,
        protocol_sha=protocol_sha,
        script_sha=script_sha,
        model_id=model_id,
        asset=asset,
        anchor_index=anchor_index,
        anchor_date=anchor_date,
        attempt_number=attempt_number,
    )
    has_convergence_warning = any(
        row["is_convergence_warning"] for row in warning_rows
    )

    output: dict[str, Any] | None = None
    convergence_flag = None
    optimizer_success = None
    optimizer_status = None
    optimizer_message = ""
    params_finite = False
    nu = float("nan")
    h_percent_sq = float("nan")
    h_decimal = float("nan")
    sigma_decimal = float("nan")
    omega = float("nan")
    alpha1 = float("nan")
    gamma1 = float("nan")
    beta1 = float("nan")
    persistence = float("nan")
    variance_rounding_corrected = False

    if result is not None:
        try:
            convergence_flag = int(result.convergence_flag)
            optimizer_success = bool(result.optimization_result.success)
            optimizer_status = int(result.optimization_result.status)
            optimizer_message = str(result.optimization_result.message)

            if convergence_flag != 0:
                failure_reasons.append("convergence_flag_nonzero")
            if not optimizer_success:
                failure_reasons.append("optimizer_success_false")
            if has_convergence_warning:
                failure_reasons.append("captured_convergence_warning")

            params = result.params.astype(np.float64)
            params_finite = bool(np.isfinite(params.to_numpy()).all())
            if not params_finite:
                failure_reasons.append("nonfinite_parameters")

            nu = extract_parameter(params, "nu")
            if not (math.isfinite(nu) and nu > 2.0):
                failure_reasons.append("invalid_nu")

            omega = extract_parameter(params, "omega")
            alpha1 = extract_parameter(params, "alpha[1]")
            gamma1 = extract_parameter(params, "gamma[1]")
            beta1 = extract_parameter(params, "beta[1]")

            forecast = result.forecast(horizon=1, reindex=False)
            variance_array = np.asarray(forecast.variance, dtype=np.float64)
            if variance_array.ndim != 2 or variance_array.shape[1] < 1:
                failure_reasons.append("invalid_forecast_variance_shape")
            else:
                h_percent_sq_raw = float(variance_array[-1, 0])
                h_percent_sq, variance_rounding_corrected = safe_nonnegative(
                    h_percent_sq_raw,
                    name="one_step_variance_percent_squared",
                )
                if not (math.isfinite(h_percent_sq) and h_percent_sq > 0.0):
                    failure_reasons.append("nonpositive_one_step_variance")
                else:
                    h_decimal = h_percent_sq / variance_divisor
                    sigma_decimal = math.sqrt(h_percent_sq) / sigma_divisor
                    if not (
                        math.isfinite(sigma_decimal) and sigma_decimal > 0.0
                    ):
                        failure_reasons.append("invalid_sigma_decimal")

            if model_spec["o"] == 0:
                persistence = alpha1 + beta1
            else:
                persistence = alpha1 + 0.5 * gamma1 + beta1

            if not math.isfinite(persistence):
                failure_reasons.append("invalid_persistence")

            if not failure_reasons:
                output = {
                    "result": result,
                    "nu": nu,
                    "h_percent_sq": h_percent_sq,
                    "h_decimal": h_decimal,
                    "sigma_decimal": sigma_decimal,
                    "omega": omega,
                    "alpha1": alpha1,
                    "gamma1": gamma1,
                    "beta1": beta1,
                    "persistence": persistence,
                    "variance_rounding_corrected": variance_rounding_corrected,
                    "convergence_flag": convergence_flag,
                    "optimizer_success": optimizer_success,
                    "optimizer_status": optimizer_status,
                    "optimizer_message": optimizer_message,
                }

        except Exception as error:
            exception_type = type(error).__name__
            exception_message = str(error)
            failure_reasons.append(f"fitted_output_exception:{exception_type}")

    elapsed = time.perf_counter() - start
    attempt_valid = len(failure_reasons) == 0 and output is not None

    attempt_row = {
        "protocol_sha256": protocol_sha,
        "script_sha256": script_sha,
        "model_id": model_id,
        "asset": asset,
        "anchor_index": anchor_index,
        "anchor_date": anchor_date,
        "attempt_number": attempt_number,
        "fit_n": int(len(return_history_decimal)),
        "attempt_valid": bool(attempt_valid),
        "failure_reasons": "|".join(failure_reasons),
        "exception_type": exception_type,
        "exception_message": exception_message,
        "convergence_flag": convergence_flag,
        "optimizer_success": optimizer_success,
        "optimizer_status": optimizer_status,
        "optimizer_message": optimizer_message,
        "warning_count": len(warning_rows),
        "convergence_warning_count": sum(
            int(row["is_convergence_warning"]) for row in warning_rows
        ),
        "other_warning_count": sum(
            int(not row["is_convergence_warning"]) for row in warning_rows
        ),
        "all_params_finite": bool(params_finite),
        "nu": nu,
        "h_percent_sq": h_percent_sq,
        "h_decimal": h_decimal,
        "sigma_decimal": sigma_decimal,
        "omega": omega,
        "alpha1": alpha1,
        "gamma1": gamma1,
        "beta1": beta1,
        "persistence": persistence,
        "elapsed_seconds": float(elapsed),
        "settings_json": safe_json_text(attempt_settings),
    }
    return attempt_row, warning_rows, output


def execute_observation(
    *,
    protocol: dict[str, Any],
    protocol_sha: str,
    script_sha: str,
    model_spec: dict[str, Any],
    asset: str,
    asset_index: int,
    anchor_index: int,
    anchor_position: int,
    anchor_date: pd.Timestamp,
    realization_date: pd.Timestamp,
    return_series_decimal: np.ndarray,
    truth: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    model_id = model_spec["model_id"]
    anchor_date_str = anchor_date.strftime("%Y-%m-%d")
    realization_date_str = realization_date.strftime("%Y-%m-%d")
    history = np.asarray(
        return_series_decimal[: anchor_position + 1],
        dtype=np.float64,
    )

    attempt_policy = protocol["fit_and_retry_policy"]
    settings_sequence = [
        attempt_policy["primary_attempt"],
        attempt_policy["retry_attempt"],
    ]
    max_attempts = int(attempt_policy["maximum_attempts"])
    require(max_attempts == 2, "Locked maximum attempts must be 2.")

    attempt_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    accepted_output: dict[str, Any] | None = None
    accepted_attempt: int | None = None

    for attempt_number, settings in enumerate(settings_sequence, start=1):
        attempt_row, attempt_warning_rows, output = fit_one_attempt(
            return_history_decimal=history,
            model_spec=model_spec,
            attempt_settings=settings,
            fit_scale=protocol["fit_scale"],
            protocol_sha=protocol_sha,
            script_sha=script_sha,
            model_id=model_id,
            asset=asset,
            anchor_index=anchor_index,
            anchor_date=anchor_date_str,
            attempt_number=attempt_number,
        )
        attempt_rows.append(attempt_row)
        warning_rows.extend(attempt_warning_rows)

        if output is not None and attempt_row["attempt_valid"]:
            accepted_output = output
            accepted_attempt = attempt_number
            break

    base_row: dict[str, Any] = {
        "protocol_sha256": protocol_sha,
        "script_sha256": script_sha,
        "model_id": model_id,
        "asset": asset,
        "asset_index": asset_index,
        "anchor_index": anchor_index,
        "anchor_position_zero_based": anchor_position,
        "anchor_date": anchor_date_str,
        "target_realization_date": realization_date_str,
        "fit_n": int(len(history)),
        "truth_nextvol": float(truth),
        "attempts_used": len(attempt_rows),
        "retry_used": len(attempt_rows) > 1,
        "accepted_attempt": accepted_attempt,
        "warning_count_total": len(warning_rows),
        "convergence_warning_count": sum(
            int(row["is_convergence_warning"]) for row in warning_rows
        ),
        "other_warning_count": sum(
            int(not row["is_convergence_warning"]) for row in warning_rows
        ),
        "status": "",
        "failure_stage": "",
        "failure_type": "",
        "failure_message": "",
        "omega": float("nan"),
        "alpha1": float("nan"),
        "gamma1": float("nan"),
        "beta1": float("nan"),
        "nu": float("nan"),
        "persistence": float("nan"),
        "h_percent_sq": float("nan"),
        "h_decimal": float("nan"),
        "sigma_decimal": float("nan"),
        "S": float("nan"),
        "Q": float("nan"),
        "c": float("nan"),
        "C": float("nan"),
        "C_rounding_corrected": False,
        "q": float("nan"),
        "central_probability_mass": float("nan"),
        "central_mass_error": float("nan"),
        "root_lower_bound": float("nan"),
        "root_upper_bound": float("nan"),
        "root_upper_bound_doublings": None,
        "primary_prediction": float("nan"),
        "plugin_prediction": float("nan"),
        "direct_sigma_prediction": float("nan"),
        "primary_pinball_loss": float("nan"),
        "primary_inside_rounding_corrected": False,
        "plugin_inside_rounding_corrected": False,
        "variance_rounding_corrected": False,
        "optimizer_status": None,
        "optimizer_message": "",
    }

    if accepted_output is None:
        base_row["status"] = "UNRESOLVED_FIT"
        base_row["failure_stage"] = "fit_or_fitted_output"
        base_row["failure_type"] = "AllAttemptsFailed"
        base_row["failure_message"] = safe_json_text(
            [
                {
                    "attempt_number": row["attempt_number"],
                    "failure_reasons": row["failure_reasons"],
                    "exception_type": row["exception_type"],
                    "exception_message": row["exception_message"],
                }
                for row in attempt_rows
            ]
        )
        return base_row, attempt_rows, warning_rows

    base_row.update(
        {
            "omega": accepted_output["omega"],
            "alpha1": accepted_output["alpha1"],
            "gamma1": accepted_output["gamma1"],
            "beta1": accepted_output["beta1"],
            "nu": accepted_output["nu"],
            "persistence": accepted_output["persistence"],
            "h_percent_sq": accepted_output["h_percent_sq"],
            "h_decimal": accepted_output["h_decimal"],
            "sigma_decimal": accepted_output["sigma_decimal"],
            "variance_rounding_corrected": accepted_output[
                "variance_rounding_corrected"
            ],
            "optimizer_status": accepted_output["optimizer_status"],
            "optimizer_message": accepted_output["optimizer_message"],
        }
    )

    try:
        target_definition = protocol["target_definition"]
        vol_window = int(target_definition["vol20_window"])
        known_count = vol_window - 1
        annualization_factor = float(
            target_definition["annualization_factor"]
        )

        require(known_count == 19, "Locked known-return count must be 19.")
        require(len(history) >= known_count, "Insufficient known returns.")
        known_returns = history[-known_count:]
        require(
            np.isfinite(known_returns).all(),
            "Known return window is non-finite.",
        )

        S = float(np.sum(known_returns, dtype=np.float64))
        Q = float(np.sum(known_returns**2, dtype=np.float64))
        c = S / float(known_count)
        C_raw = (
            Q - (S**2) / float(known_count)
        ) / float(known_count)
        C, C_corrected = safe_nonnegative(C_raw, name="C")

        root = solve_conditional_median_q(
            c=c,
            sigma_decimal=accepted_output["sigma_decimal"],
            nu=accepted_output["nu"],
            root_settings=protocol["primary_forecast"]["root_settings"],
            distribution=StudentsT(),
        )

        q = float(root["q"])
        primary_inside_raw = C + (q**2) / float(vol_window)
        primary_inside, primary_corrected = safe_nonnegative(
            primary_inside_raw,
            name="primary_forecast_inside",
        )
        primary_prediction = math.sqrt(annualization_factor * primary_inside)

        plugin_inside_raw = C + (
            accepted_output["h_decimal"] + c**2
        ) / float(vol_window)
        plugin_inside, plugin_corrected = safe_nonnegative(
            plugin_inside_raw,
            name="plugin_forecast_inside",
        )
        plugin_prediction = math.sqrt(annualization_factor * plugin_inside)

        direct_inside, _ = safe_nonnegative(
            accepted_output["h_decimal"] * annualization_factor,
            name="direct_sigma_inside",
        )
        direct_prediction = math.sqrt(direct_inside)

        predictions = np.asarray(
            [primary_prediction, plugin_prediction, direct_prediction],
            dtype=np.float64,
        )
        require(np.isfinite(predictions).all(), "One or more predictions are non-finite.")

        loss = float(
            pinball_loss_series(
                np.asarray([truth], dtype=np.float64),
                np.asarray([primary_prediction], dtype=np.float64),
                tau=float(protocol["metrics"]["primary"]["tau"]),
            )[0]
        )
        require(math.isfinite(loss), "Primary pinball loss is non-finite.")

        base_row.update(
            {
                "status": "OK",
                "S": S,
                "Q": Q,
                "c": c,
                "C": C,
                "C_rounding_corrected": C_corrected,
                "q": q,
                "central_probability_mass": root[
                    "central_probability_mass"
                ],
                "central_mass_error": root["central_mass_error"],
                "root_lower_bound": root["root_lower_bound"],
                "root_upper_bound": root["root_upper_bound"],
                "root_upper_bound_doublings": root[
                    "root_upper_bound_doublings"
                ],
                "primary_prediction": primary_prediction,
                "plugin_prediction": plugin_prediction,
                "direct_sigma_prediction": direct_prediction,
                "primary_pinball_loss": loss,
                "primary_inside_rounding_corrected": primary_corrected,
                "plugin_inside_rounding_corrected": plugin_corrected,
            }
        )

    except Exception as error:
        # Per lock: a root/prediction failure after a valid fit does not trigger
        # model switching or a new fit attempt.
        base_row["status"] = "UNRESOLVED_ROOT_OR_PREDICTION"
        base_row["failure_stage"] = "cdf_root_or_prediction"
        base_row["failure_type"] = type(error).__name__
        base_row["failure_message"] = str(error)

    return base_row, attempt_rows, warning_rows


# =============================================================================
# Checkpointing
# =============================================================================

def checkpoint_paths(
    checkpoint_dir: Path,
    model_id: str,
    asset: str,
) -> tuple[Path, Path, Path]:
    slug = slugify_model_id(model_id)
    prefix = f"{slug}__{asset.lower()}"
    return (
        checkpoint_dir / f"{prefix}__observations.csv",
        checkpoint_dir / f"{prefix}__attempts.csv",
        checkpoint_dir / f"{prefix}__warnings.csv",
    )


def validate_checkpoint_identity(
    frame: pd.DataFrame,
    *,
    protocol_sha: str,
    script_sha: str,
    model_id: str,
    asset: str,
    frame_name: str,
) -> None:
    if frame.empty:
        return

    for column in ("protocol_sha256", "script_sha256", "model_id", "asset"):
        require(column in frame.columns, f"{frame_name} checkpoint lacks {column}.")

    require(
        set(frame["protocol_sha256"].astype(str)) == {protocol_sha},
        f"{frame_name} protocol SHA mismatch.",
    )
    require(
        set(frame["script_sha256"].astype(str)) == {script_sha},
        f"{frame_name} script SHA mismatch.",
    )
    require(
        set(frame["model_id"].astype(str)) == {model_id},
        f"{frame_name} model mismatch.",
    )
    require(
        set(frame["asset"].astype(str)) == {asset},
        f"{frame_name} asset mismatch.",
    )


def save_pair_checkpoints(
    paths: tuple[Path, Path, Path],
    observation_rows: list[dict[str, Any]],
    attempt_rows: list[dict[str, Any]],
    warning_rows: list[dict[str, Any]],
) -> None:
    observation_frame = pd.DataFrame(observation_rows)
    attempt_frame = pd.DataFrame(attempt_rows)
    warning_frame = pd.DataFrame(
        warning_rows,
        columns=WARNING_CHECKPOINT_COLUMNS,
    )

    atomic_write_csv(paths[0], observation_frame)
    atomic_write_csv(paths[1], attempt_frame)
    atomic_write_csv(paths[2], warning_frame)


def run_pair(
    *,
    root: Path,
    protocol: dict[str, Any],
    protocol_sha: str,
    script_sha: str,
    data: dict[str, Any],
    model_spec: dict[str, Any],
    asset: str,
    asset_index: int,
    checkpoint_every: int,
) -> None:
    checkpoint_dir = root / CHECKPOINT_RELATIVE_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths = checkpoint_paths(checkpoint_dir, model_spec["model_id"], asset)

    observation_frame = read_csv_checkpoint(paths[0])
    attempt_frame = read_csv_checkpoint(paths[1])
    warning_frame = read_csv_checkpoint(paths[2])

    validate_checkpoint_identity(
        observation_frame,
        protocol_sha=protocol_sha,
        script_sha=script_sha,
        model_id=model_spec["model_id"],
        asset=asset,
        frame_name="observation",
    )
    validate_checkpoint_identity(
        attempt_frame,
        protocol_sha=protocol_sha,
        script_sha=script_sha,
        model_id=model_spec["model_id"],
        asset=asset,
        frame_name="attempt",
    )
    validate_checkpoint_identity(
        warning_frame,
        protocol_sha=protocol_sha,
        script_sha=script_sha,
        model_id=model_spec["model_id"],
        asset=asset,
        frame_name="warning",
    )

    observation_rows = observation_frame.to_dict("records")
    attempt_rows = attempt_frame.to_dict("records")
    warning_rows = warning_frame.to_dict("records")

    completed = (
        set(observation_frame["anchor_index"].astype(int).tolist())
        if not observation_frame.empty
        else set()
    )
    require(len(completed) == len(observation_rows), "Duplicate observation checkpoints.")

    total = len(data["anchor_dates"])
    pair_label = f"{model_spec['model_id']} | {asset}"

    print("\n" + "=" * 110)
    print(f"PAIR START: {pair_label}")
    print(f"Already checkpointed: {len(completed)}/{total}")
    print("=" * 110)

    since_last_checkpoint = 0
    pair_start = time.perf_counter()

    for anchor_index in range(total):
        if anchor_index in completed:
            continue

        anchor_position = int(data["anchor_positions"][anchor_index])
        row, new_attempts, new_warnings = execute_observation(
            protocol=protocol,
            protocol_sha=protocol_sha,
            script_sha=script_sha,
            model_spec=model_spec,
            asset=asset,
            asset_index=asset_index,
            anchor_index=anchor_index,
            anchor_position=anchor_position,
            anchor_date=data["anchor_dates"][anchor_index],
            realization_date=data["realization_dates"][anchor_index],
            return_series_decimal=data["returns"][:, asset_index],
            truth=float(data["y_vol_truth"][anchor_index, asset_index]),
        )

        observation_rows.append(row)
        attempt_rows.extend(new_attempts)
        warning_rows.extend(new_warnings)
        completed.add(anchor_index)
        since_last_checkpoint += 1

        if (
            since_last_checkpoint >= checkpoint_every
            or anchor_index == total - 1
            or row["status"] != "OK"
        ):
            save_pair_checkpoints(
                paths,
                observation_rows,
                attempt_rows,
                warning_rows,
            )
            since_last_checkpoint = 0

        if (
            (anchor_index + 1) % 25 == 0
            or anchor_index == 0
            or anchor_index == total - 1
            or row["status"] != "OK"
        ):
            elapsed = time.perf_counter() - pair_start
            print(
                f"[{pair_label}] "
                f"{anchor_index + 1:3d}/{total} | "
                f"status={row['status']} | "
                f"attempts={row['attempts_used']} | "
                f"elapsed={elapsed:.1f}s"
            )

    save_pair_checkpoints(paths, observation_rows, attempt_rows, warning_rows)

    final_frame = pd.DataFrame(observation_rows)
    require(len(final_frame) == total, f"Pair row count is not {total}: {pair_label}")
    require(
        final_frame["anchor_index"].astype(int).nunique() == total,
        f"Pair anchor inventory is not unique and complete: {pair_label}",
    )

    status_counts = final_frame["status"].value_counts(dropna=False).to_dict()
    print(f"PAIR COMPLETE: {pair_label}")
    print(f"Status counts: {status_counts}")


# =============================================================================
# Finalization and outputs
# =============================================================================

def collect_all_checkpoints(
    *,
    root: Path,
    protocol: dict[str, Any],
    protocol_sha: str,
    script_sha: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    checkpoint_dir = root / CHECKPOINT_RELATIVE_DIR

    observation_frames = []
    attempt_frames = []
    warning_frames = []
    incomplete_pairs: list[str] = []

    for model_spec in protocol["official_models"]:
        for asset in protocol["assets"]:
            paths = checkpoint_paths(
                checkpoint_dir,
                model_spec["model_id"],
                asset,
            )
            if not paths[0].exists():
                incomplete_pairs.append(f"{model_spec['model_id']}|{asset}")
                continue

            obs = pd.read_csv(paths[0])
            attempts = pd.read_csv(paths[1]) if paths[1].exists() else pd.DataFrame()
            warning_inventory = (
                pd.read_csv(paths[2]) if paths[2].exists() else pd.DataFrame()
            )

            validate_checkpoint_identity(
                obs,
                protocol_sha=protocol_sha,
                script_sha=script_sha,
                model_id=model_spec["model_id"],
                asset=asset,
                frame_name="observation",
            )
            validate_checkpoint_identity(
                attempts,
                protocol_sha=protocol_sha,
                script_sha=script_sha,
                model_id=model_spec["model_id"],
                asset=asset,
                frame_name="attempt",
            )
            validate_checkpoint_identity(
                warning_inventory,
                protocol_sha=protocol_sha,
                script_sha=script_sha,
                model_id=model_spec["model_id"],
                asset=asset,
                frame_name="warning",
            )

            if len(obs) != 584 or obs["anchor_index"].astype(int).nunique() != 584:
                incomplete_pairs.append(f"{model_spec['model_id']}|{asset}")

            observation_frames.append(obs)
            if not attempts.empty:
                attempt_frames.append(attempts)
            if not warning_inventory.empty:
                warning_frames.append(warning_inventory)

    observations = (
        pd.concat(observation_frames, ignore_index=True)
        if observation_frames
        else pd.DataFrame()
    )
    attempts = (
        pd.concat(attempt_frames, ignore_index=True)
        if attempt_frames
        else pd.DataFrame()
    )
    warning_inventory = (
        pd.concat(warning_frames, ignore_index=True)
        if warning_frames
        else pd.DataFrame()
    )
    return observations, attempts, warning_inventory, incomplete_pairs


def matrix_from_observations(
    observations: pd.DataFrame,
    *,
    model_id: str,
    assets: list[str],
    value_column: str,
) -> np.ndarray:
    matrix = np.full((584, len(assets)), np.nan, dtype=np.float64)
    for asset_index, asset in enumerate(assets):
        subset = observations[
            (observations["model_id"] == model_id)
            & (observations["asset"] == asset)
        ].copy()
        subset["anchor_index"] = subset["anchor_index"].astype(int)
        subset = subset.sort_values("anchor_index")
        require(len(subset) == 584, f"Missing observations for {model_id}|{asset}.")
        require(
            np.array_equal(
                subset["anchor_index"].to_numpy(),
                np.arange(584),
            ),
            f"Anchor order mismatch for {model_id}|{asset}.",
        )
        matrix[:, asset_index] = subset[value_column].to_numpy(dtype=np.float64)

    require(
        np.isfinite(matrix).all(),
        f"Non-finite values in reconstructed matrix {model_id}|{value_column}.",
    )
    return matrix


def finalize_outputs(
    *,
    root: Path,
    protocol: dict[str, Any],
    protocol_info: dict[str, Any],
    locked_inputs: dict[str, dict[str, Any]],
    data: dict[str, Any],
    script_sha: str,
) -> dict[str, Any]:
    output_dir = root / OUTPUT_RELATIVE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    observations, attempts, warning_inventory, incomplete_pairs = (
        collect_all_checkpoints(
            root=root,
            protocol=protocol,
            protocol_sha=protocol_info["protocol_sha256"],
            script_sha=script_sha,
        )
    )

    if not observations.empty:
        observations = observations.sort_values(
            ["model_id", "asset_index", "anchor_index"]
        ).reset_index(drop=True)
        atomic_write_csv(output_dir / DIAGNOSTICS_FILENAME, observations)

    if not attempts.empty:
        attempts = attempts.sort_values(
            ["model_id", "asset", "anchor_index", "attempt_number"]
        ).reset_index(drop=True)
        atomic_write_csv(output_dir / ATTEMPTS_FILENAME, attempts)

    if not warning_inventory.empty:
        warning_inventory = warning_inventory.sort_values(
            [
                "model_id",
                "asset",
                "anchor_index",
                "attempt_number",
                "warning_index",
            ]
        ).reset_index(drop=True)
        atomic_write_csv(output_dir / WARNINGS_FILENAME, warning_inventory)
    else:
        atomic_write_csv(
            output_dir / WARNINGS_FILENAME,
            pd.DataFrame(columns=WARNING_CHECKPOINT_COLUMNS),
        )

    expected_observations = int(
        protocol["expected_inventory"]["primary_fits_without_retries"]
    )
    unresolved = (
        observations[observations["status"] != "OK"].copy()
        if not observations.empty and "status" in observations.columns
        else pd.DataFrame()
    )

    inventory_complete = (
        not incomplete_pairs
        and len(observations) == expected_observations
        and observations["anchor_index"].astype(int).nunique() == 584
    )
    clean_closure = inventory_complete and unresolved.empty

    summary: dict[str, Any] = {
        "project_version": PROJECT_VERSION,
        "stage": STAGE,
        "created_at_utc": utc_now_iso(),
        "script": Path(__file__).name if "__file__" in globals() else "",
        "script_sha256": script_sha,
        "protocol": protocol_info,
        "locked_inputs": locked_inputs,
        "comparison_inputs": data["comparison_inputs"],
        "inventory": {
            "expected_observations": expected_observations,
            "observed_observations": int(len(observations)),
            "expected_models": len(protocol["official_models"]),
            "expected_assets": len(protocol["assets"]),
            "expected_test_observations": 584,
            "attempt_rows": int(len(attempts)),
            "warning_rows": int(len(warning_inventory)),
            "incomplete_pairs": incomplete_pairs,
            "unresolved_observations": int(len(unresolved)),
            "inventory_complete": bool(inventory_complete),
        },
        "integrity": {
            "max_logret_reconstruction_diff_by_asset": data[
                "max_logret_reconstruction_diff_by_asset"
            ],
            "official_truth_exactly_matches_07_truth": True,
            "official_truth_exactly_matches_dtype_aligned_targets_csv": True,
            "08A_final_loss_series_reconstructed_exactly": True,
            "08A_vol_persistence_loss_series_reconstructed_exactly": True,
        },
        "policy": {
            "model_selection_inside_08C": False,
            "hyperparameter_selection_inside_08C": False,
            "test_tuning_inside_08C": False,
            "warm_start_used": False,
            "silent_fallback_used": False,
            "show_warning": protocol["fit_and_retry_policy"]["primary_attempt"][
                "show_warning"
            ],
            "convergence_decision": (
                "Primary: result.convergence_flag == 0 AND "
                "optimization_result.success is True. "
                "Captured ConvergenceWarning is an additional failure signal. "
                "Because the locked fit call uses show_warning=False, the arch "
                "library's own convergence warning display is suppressed; "
                "flag/status remain the authoritative convergence evidence."
            ),
        },
        "clean_closure": bool(clean_closure),
        "audit_is_model_success_claim": False,
        "statistical_significance_claim": False,
    }

    if not clean_closure:
        atomic_write_json(output_dir / SUMMARY_FILENAME, summary)
        print("\n08C did not reach clean closure.")
        print(f"Incomplete pairs      : {incomplete_pairs}")
        print(f"Unresolved observations: {len(unresolved)}")
        print(f"Summary written       : {output_dir / SUMMARY_FILENAME}")
        return summary

    assets = list(protocol["assets"])
    tau = float(protocol["metrics"]["primary"]["tau"])

    metrics_rows: list[dict[str, Any]] = []
    task_average_rows: list[dict[str, Any]] = []
    final_comparison_rows: list[dict[str, Any]] = []
    persistence_comparison_rows: list[dict[str, Any]] = []
    loss_arrays: dict[str, np.ndarray] = {}
    diagnostic_arrays: dict[str, np.ndarray] = {}

    final_vol_pred = data["final_pred"][:, 4:8]
    truth = data["y_vol_truth"]
    persistence_pred = data["vol_persistence_pred"]

    for model_spec in protocol["official_models"]:
        model_id = model_spec["model_id"]
        slug = slugify_model_id(model_id)

        primary = matrix_from_observations(
            observations,
            model_id=model_id,
            assets=assets,
            value_column="primary_prediction",
        )
        plugin = matrix_from_observations(
            observations,
            model_id=model_id,
            assets=assets,
            value_column="plugin_prediction",
        )
        direct = matrix_from_observations(
            observations,
            model_id=model_id,
            assets=assets,
            value_column="direct_sigma_prediction",
        )

        atomic_save_npy(
            output_dir / f"pred_primary_{slug}_raw_v4.npy",
            primary,
        )
        atomic_save_npy(
            output_dir / f"pred_plugin_{slug}_raw_v4.npy",
            plugin,
        )
        atomic_save_npy(
            output_dir / f"pred_direct_sigma_{slug}_raw_v4.npy",
            direct,
        )

        for diagnostic_name in (
            "h_percent_sq",
            "h_decimal",
            "sigma_decimal",
            "nu",
            "S",
            "Q",
            "c",
            "C",
            "q",
            "central_mass_error",
            "persistence",
        ):
            diagnostic_arrays[f"{model_id}__{diagnostic_name}"] = (
                matrix_from_observations(
                    observations,
                    model_id=model_id,
                    assets=assets,
                    value_column=diagnostic_name,
                )
            )

        per_asset_metrics = []

        for asset_index, asset in enumerate(assets):
            metric_values = regression_metrics(
                truth[:, asset_index],
                primary[:, asset_index],
                tau=tau,
            )
            metrics_rows.append(
                {
                    "model_id": model_id,
                    "prediction_type": "primary_conditional_median_nextvol20",
                    "task": "volatility",
                    "asset": asset,
                    "target_index": int(protocol["target_indices"][asset_index]),
                    **metric_values,
                }
            )
            per_asset_metrics.append(metric_values)

            loss = pinball_loss_series(
                truth[:, asset_index],
                primary[:, asset_index],
                tau=tau,
            )
            require(len(loss) == 584, "Loss series length is not 584.")
            loss_arrays[f"{model_id}__volatility__{asset}"] = loss

            final_metric = regression_metrics(
                truth[:, asset_index],
                final_vol_pred[:, asset_index],
                tau=tau,
            )
            persistence_metric = regression_metrics(
                truth[:, asset_index],
                persistence_pred[:, asset_index],
                tau=tau,
            )

            candidate_error = metric_values["PinballLoss_tau_0.5"]
            final_error = final_metric["PinballLoss_tau_0.5"]
            persistence_error = persistence_metric["PinballLoss_tau_0.5"]

            final_comparison_rows.append(
                {
                    "task": "volatility",
                    "asset": asset,
                    "primary_metric": "PinballLoss_tau_0.5",
                    "reference_model": "Final_NoSharing_FixedLambda_0.7_lb10_small_baseline_3seed_ensemble",
                    "reference_error": final_error,
                    "candidate_model": model_id,
                    "candidate_error": candidate_error,
                    "candidate_minus_reference": candidate_error - final_error,
                    "candidate_to_reference_ratio": candidate_error / final_error,
                    "candidate_beats_reference": candidate_error < final_error,
                }
            )

            persistence_comparison_rows.append(
                {
                    "task": "volatility",
                    "asset": asset,
                    "primary_metric": "PinballLoss_tau_0.5",
                    "reference_model": "VolPersistence",
                    "reference_error": persistence_error,
                    "candidate_model": model_id,
                    "candidate_error": candidate_error,
                    "candidate_minus_reference": candidate_error - persistence_error,
                    "candidate_to_reference_ratio": (
                        candidate_error / persistence_error
                    ),
                    "candidate_beats_reference": (
                        candidate_error < persistence_error
                    ),
                }
            )

        for metric_name in ("MAE", "RMSE", "R2", "PinballLoss_tau_0.5"):
            task_average_rows.append(
                {
                    "model_id": model_id,
                    "task": "volatility",
                    "metric": metric_name,
                    "task_average": float(
                        np.mean(
                            [
                                metric_values[metric_name]
                                for metric_values in per_asset_metrics
                            ]
                        )
                    ),
                    "n_assets": len(assets),
                }
            )

    metrics_frame = pd.DataFrame(metrics_rows)
    task_average_frame = pd.DataFrame(task_average_rows)
    final_comparison_frame = pd.DataFrame(final_comparison_rows)
    persistence_comparison_frame = pd.DataFrame(persistence_comparison_rows)

    atomic_write_csv(output_dir / METRICS_FILENAME, metrics_frame)
    atomic_write_csv(output_dir / TASK_AVERAGE_FILENAME, task_average_frame)
    atomic_write_csv(
        output_dir / FINAL_COMPARISON_FILENAME,
        final_comparison_frame,
    )
    atomic_write_csv(
        output_dir / PERSISTENCE_COMPARISON_FILENAME,
        persistence_comparison_frame,
    )
    atomic_save_npz(output_dir / LOSS_SERIES_FILENAME, loss_arrays)
    atomic_save_npz(
        output_dir / "garch_diagnostic_arrays_v4.npz",
        diagnostic_arrays,
    )
    atomic_save_npy(output_dir / DATES_ANCHOR_FILENAME, data["anchor_raw"])
    atomic_save_npy(
        output_dir / DATES_REALIZATION_FILENAME,
        data["realization_raw"],
    )

    summary["metrics"] = {
        "primary": metrics_frame.to_dict("records"),
        "task_average": task_average_frame.to_dict("records"),
    }
    summary["comparisons"] = {
        "final_vs_garch": final_comparison_frame.to_dict("records"),
        "vol_persistence_vs_garch": persistence_comparison_frame.to_dict("records"),
    }
    summary["required_output_checks"] = {
        "primary_predictions_per_model_asset": 584,
        "plugin_predictions_per_model_asset": 584,
        "direct_sigma_predictions_per_model_asset": 584,
        "loss_series_per_model_asset": 584,
        "diagnostic_observation_rows": int(len(observations)),
        "metrics_rows": int(len(metrics_frame)),
        "task_average_rows": int(len(task_average_frame)),
        "final_comparison_rows": int(len(final_comparison_frame)),
        "persistence_comparison_rows": int(len(persistence_comparison_frame)),
    }

    # Write summary before manifest, then hash every final output.
    atomic_write_json(output_dir / SUMMARY_FILENAME, summary)

    manifest_entries: dict[str, dict[str, Any]] = {}
    for path in sorted(output_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name == OUTPUT_MANIFEST_FILENAME:
            continue
        manifest_entries[path.name] = {
            "relative_path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    output_manifest = {
        "project_version": PROJECT_VERSION,
        "stage": STAGE,
        "created_at_utc": utc_now_iso(),
        "protocol_sha256": protocol_info["protocol_sha256"],
        "script_sha256": script_sha,
        "clean_closure": True,
        "outputs": manifest_entries,
    }
    atomic_write_json(output_dir / OUTPUT_MANIFEST_FILENAME, output_manifest)

    print("\n" + "=" * 110)
    print("08C OFFICIAL RUN — COMPLETE")
    print("=" * 110)
    print(f"Observation rows : {len(observations)}")
    print(f"Attempt rows     : {len(attempts)}")
    print(f"Warning rows     : {len(warning_inventory)}")
    print(f"Unresolved rows  : {len(unresolved)}")
    print(f"Clean closure    : {summary['clean_closure']}")
    print(f"Results directory: {output_dir}")
    print("Audit PASS/model success/significance are NOT claimed here.")
    print("=" * 110)

    return summary


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run locked 08C expanding-window GARCH-family baselines."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Project root.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Verify protocol, environment, inputs, dates, and prior-stage comparisons; do not fit.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Optional exact official model_id subset.",
    )
    parser.add_argument(
        "--asset",
        type=str,
        default=None,
        help="Optional exact asset subset.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Atomic checkpoint frequency within each model-asset pair.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    require(root.exists(), f"Project root does not exist: {root}")
    require(args.checkpoint_every >= 1, "--checkpoint-every must be >= 1.")

    script_path = Path(__file__).resolve()
    script_sha = sha256_file(script_path)

    print("=" * 110)
    print("08C LOCKED GARCH BASELINES")
    print("=" * 110)
    print(f"Project root : {root}")
    print(f"Script path  : {script_path}")
    print(f"Script SHA   : {script_sha}")

    protocol, protocol_info = load_and_verify_protocol(root)
    locked_inputs = verify_locked_inputs(root, protocol)
    data = validate_data_alignment(root, protocol)

    print("\nVALIDATION PASSED")
    print(f"Protocol SHA : {protocol_info['protocol_sha256']}")
    print(f"Locked inputs: {len(locked_inputs)}/10")
    print(f"Test anchors : {len(data['anchor_dates'])}")
    print(f"First fit n  : {int(data['anchor_positions'][0]) + 1}")
    print(f"Last fit n   : {int(data['anchor_positions'][-1]) + 1}")
    print("No model selection, hyperparameter selection, or test tuning.")

    if args.validate_only:
        print("\nVALIDATE-ONLY COMPLETE")
        print("Model fit performed : NO")
        print("Prediction computed : NO")
        print("08C metric computed : NO")
        print("Protocol modified   : NO")
        return

    output_dir = root / OUTPUT_RELATIVE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    start_manifest_path = output_dir / RUN_MANIFEST_START_FILENAME
    start_manifest = {
        "project_version": PROJECT_VERSION,
        "stage": STAGE,
        "started_at_utc": utc_now_iso(),
        "script_path": str(script_path),
        "script_sha256": script_sha,
        "protocol": protocol_info,
        "locked_inputs": locked_inputs,
        "comparison_inputs": data["comparison_inputs"],
        "environment": version_inventory(),
        "model_subset": args.model_id,
        "asset_subset": args.asset,
        "checkpoint_every": args.checkpoint_every,
        "official_fit_started": True,
        "test_tuning": False,
        "model_selection": False,
        "hyperparameter_selection": False,
    }

    if start_manifest_path.exists():
        existing = json.loads(start_manifest_path.read_text(encoding="utf-8"))
        require(
            existing["protocol"]["protocol_sha256"]
            == protocol_info["protocol_sha256"],
            "Existing run manifest protocol SHA mismatch.",
        )
        require(
            existing["script_sha256"] == script_sha,
            "Existing run manifest script SHA mismatch. "
            "Do not mix checkpoints from different script versions.",
        )
    else:
        atomic_write_json(start_manifest_path, start_manifest)

    models = list(protocol["official_models"])
    assets = list(protocol["assets"])

    if args.model_id is not None:
        models = [spec for spec in models if spec["model_id"] == args.model_id]
        require(models, f"Unknown official model_id: {args.model_id}")

    if args.asset is not None:
        require(args.asset in assets, f"Unknown official asset: {args.asset}")
        assets_to_run = [args.asset]
    else:
        assets_to_run = assets

    for model_spec in models:
        for asset in assets_to_run:
            asset_index = assets.index(asset)
            run_pair(
                root=root,
                protocol=protocol,
                protocol_sha=protocol_info["protocol_sha256"],
                script_sha=script_sha,
                data=data,
                model_spec=model_spec,
                asset=asset,
                asset_index=asset_index,
                checkpoint_every=args.checkpoint_every,
            )

    summary = finalize_outputs(
        root=root,
        protocol=protocol,
        protocol_info=protocol_info,
        locked_inputs=locked_inputs,
        data=data,
        script_sha=script_sha,
    )

    if not summary["clean_closure"]:
        print(
            "\nRun/checkpoint work completed, but the complete 08C inventory "
            "has not reached clean closure."
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
