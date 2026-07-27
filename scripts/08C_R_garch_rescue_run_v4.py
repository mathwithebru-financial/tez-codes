#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
08C-R — Locked numerical-convergence rescue for unresolved USDTRY
GARCH-family fits from the canonical 08C run.

This script:
- verifies the active 08C-R protocol and all referenced SHA-locked inputs;
- imports the canonical parent 08C script and reuses its locked data validation;
- targets exactly 574 parent rows marked UNRESOLVED_FIT (329 GARCH, 245 GJR-GARCH);
- uses exactly one deterministic projected warm-start attempt per target;
- rejects any fit with nonzero convergence_flag, optimizer success=False,
  captured ConvergenceWarning, or captured StartingValueWarning;
- writes checkpoints and rescue-only outputs under results/baselines/garch/08C_R;
- never modifies parent 08C outputs;
- does not merge parent and rescue rows and does not compute aggregate metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Validate-only must not create __pycache__ beside the canonical parent script.
sys.dont_write_bytecode = True

import arch
import numpy as np
import pandas as pd
import scipy
from arch import arch_model
from pandas.errors import EmptyDataError


PROJECT_VERSION = "v4_repro"
STAGE = "08C-R"
DEFAULT_ROOT = Path("/content/drive/MyDrive/tez_transformer_v4_repro")

ACTIVE_PROTOCOL_REL = Path("config/08C_R_garch_rescue_protocol_lock_v4.json")
ACTIVE_PROTOCOL_SHA_REL = Path("config/08C_R_garch_rescue_protocol_lock_v4.sha256")
PARENT_SCRIPT_REL = Path("scripts/08C_garch_baselines_test_v4.py")
SELF_SHA_REL = Path("scripts/08C_R_garch_rescue_run_v4.sha256")

EXPECTED_ACTIVE_PROTOCOL_SHA = (
    "542ec09f0e9782e0f6cf1d0d11e0b11481c0e4a9da61e28c9358d8d4dafcd4f5"
)
EXPECTED_PARENT_SCRIPT_SHA = (
    "c60c1a5174803275fa9f88d7bfeea1ea5116bf0160793b5534aa803f531b50da"
)
EXPECTED_TARGET_COUNTS = {
    "GARCH_1_1_StudentsT_ZeroMean": 329,
    "GJR_GARCH_1_1_StudentsT_ZeroMean": 245,
}
EXPECTED_TOTAL_TARGETS = 574
PERSISTENCE_TARGET = 0.999999
FLOAT_ATOL = 1e-12
FLOAT_RTOL = 1e-12

OUTPUT_DIR_REL = Path("results/baselines/garch/08C_R")
CHECKPOINT_DIRNAME = "checkpoints"

FINAL_ATTEMPTS = "08C_R_rescue_attempts_v4.csv"
FINAL_DIAGNOSTICS = "08C_R_rescue_diagnostics_v4.csv"
FINAL_WARNINGS = "08C_R_rescue_warning_inventory_v4.csv"
FINAL_SUMMARY = "08C_R_rescue_summary_v4.json"
FINAL_MANIFEST = "08C_R_rescue_manifest_v4.json"

CHECKPOINT_ATTEMPTS = "08C_R_rescue_attempts_checkpoint_v4.csv"
CHECKPOINT_DIAGNOSTICS = "08C_R_rescue_diagnostics_checkpoint_v4.csv"
CHECKPOINT_WARNINGS = "08C_R_rescue_warnings_checkpoint_v4.csv"
CHECKPOINT_STATE = "08C_R_rescue_checkpoint_state_v4.json"

WARNING_COLUMNS = [
    "protocol_stage",
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
    "is_starting_value_warning",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha_token(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="strict").strip().split()[0]


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
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(text)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=json_default,
    ) + "\n"
    atomic_write_text(path, text)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    text = frame.to_csv(index=False, float_format="%.17g", lineterminator="\n")
    atomic_write_text(path, text)


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def parse_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("true")


def version_inventory() -> dict[str, str]:
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "arch": arch.__version__,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def verify_identity(root: Path, record: dict[str, Any], label: str) -> dict[str, Any]:
    relative_path = record.get("relative_path")
    require(isinstance(relative_path, str) and relative_path, f"{label}: missing relative_path")
    path = root / relative_path
    require(path.exists(), f"{label}: missing file {path}")
    observed_size = int(path.stat().st_size)
    observed_sha = sha256_file(path)
    require(
        observed_size == int(record["size_bytes"]),
        f"{label}: size mismatch expected={record['size_bytes']} observed={observed_size}",
    )
    require(
        observed_sha == str(record["sha256"]),
        f"{label}: SHA mismatch expected={record['sha256']} observed={observed_sha}",
    )
    return {
        "relative_path": relative_path,
        "size_bytes": observed_size,
        "sha256": observed_sha,
    }


def load_parent_module(parent_script_path: Path) -> Any:
    module_spec = importlib.util.spec_from_file_location("garch08c_locked_parent", parent_script_path)
    require(module_spec is not None and module_spec.loader is not None, "Parent 08C module spec failed")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def load_and_validate_protocol(root: Path, script_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = root / ACTIVE_PROTOCOL_REL
    protocol_sha_path = root / ACTIVE_PROTOCOL_SHA_REL
    self_sha_path = root / SELF_SHA_REL

    require(protocol_path.exists(), f"Active protocol missing: {protocol_path}")
    require(protocol_sha_path.exists(), f"Active protocol SHA missing: {protocol_sha_path}")
    require(self_sha_path.exists(), f"Rescue script SHA file missing: {self_sha_path}")

    protocol_sha = sha256_file(protocol_path)
    require(protocol_sha == EXPECTED_ACTIVE_PROTOCOL_SHA, "Active 08C-R protocol SHA mismatch")
    require(sha_token(protocol_sha_path) == EXPECTED_ACTIVE_PROTOCOL_SHA, "Protocol companion SHA mismatch")

    script_sha = sha256_file(script_path)
    require(sha_token(self_sha_path) == script_sha, "Rescue script companion SHA mismatch")

    with protocol_path.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)

    require(protocol.get("project_version") == PROJECT_VERSION, "Project version mismatch")
    require(protocol.get("stage") == STAGE, "Protocol stage mismatch")
    require(
        protocol.get("protocol_status") == "LOCKED_BEFORE_ANY_08C_R_RESCUE_FIT",
        "Unexpected active protocol status",
    )
    require(protocol["scope"]["expected_target_count"] == EXPECTED_TOTAL_TARGETS, "Target count mismatch")
    require(protocol["scope"]["expected_by_model"] == EXPECTED_TARGET_COUNTS, "Model target counts mismatch")
    require(protocol["fit_policy"]["maximum_rescue_attempts_per_target"] == 1, "Rescue attempt count must be one")
    require(protocol["stopping_rule"]["one_rescue_attempt_only"] is True, "Single-attempt lock missing")
    require(protocol["warm_start_rule"]["rescued_fit_may_be_a_source"] is False, "Rescue chaining must be forbidden")
    require(float(protocol["projection_rule"]["persistence_target"]) == PERSISTENCE_TARGET, "Persistence target mismatch")
    require(protocol["output_policy"]["parent_overwrite_allowed"] is False, "Parent overwrite must be forbidden")
    require(
        protocol["output_policy"]["relative_output_directory"] == str(OUTPUT_DIR_REL),
        "Rescue output directory mismatch",
    )
    fit_call = protocol["fit_policy"]["fit_call"]
    require(int(fit_call["update_freq"]) == 0, "update_freq mismatch")
    require(fit_call["disp"] == "off", "disp mismatch")
    require(bool(fit_call["show_warning"]) is False, "show_warning mismatch")
    require(fit_call["cov_type"] == "robust", "cov_type mismatch")
    require(float(fit_call["tol"]) == 1e-08, "tol mismatch")
    require(int(fit_call["options"]["maxiter"]) == 5000, "maxiter mismatch")
    require(fit_call["backcast"] is None, "backcast must be null")
    for key in (
        "additional_starting_value_search_allowed",
        "additional_tolerance_search_allowed",
        "additional_maxiter_search_allowed",
        "optimizer_switching_allowed",
        "model_switching_allowed",
    ):
        require(protocol["stopping_rule"][key] is False, f"Stopping lock violated: {key}")
    require(protocol["scientific_role"]["igarch_is_outside_08C_R"] is True, "IGARCH must remain outside rescue")

    observed_runtime = version_inventory()
    creation_runtime = protocol["environment_policy"]["observed_at_protocol_creation"]
    for key in ("python", "arch", "scipy", "numpy", "pandas"):
        require(
            str(observed_runtime[key]) == str(creation_runtime[key]),
            f"Runtime mismatch for {key}: expected={creation_runtime[key]} observed={observed_runtime[key]}",
        )

    identities: dict[str, Any] = {}
    parent = protocol["parent_08C"]
    for key in ("protocol", "protocol_sha_file", "script", "attempts", "diagnostics", "summary"):
        identities[f"parent_08C.{key}"] = verify_identity(root, parent[key], f"parent_08C.{key}")

    evidence = protocol["prelock_feasibility_evidence"]
    for key in ("audit_csv", "audit_csv_sha_file", "audit_summary_json", "audit_summary_sha_file"):
        identities[f"feasibility.{key}"] = verify_identity(root, evidence[key], f"feasibility.{key}")

    historical = protocol["historical_protocol_record"]
    identities["historical.superseded_protocol"] = verify_identity(
        root, historical["superseded_protocol"], "historical.superseded_protocol"
    )
    identities["historical.superseded_note"] = verify_identity(
        root, historical["superseded_note"], "historical.superseded_note"
    )

    require(
        identities["parent_08C.script"]["sha256"] == EXPECTED_PARENT_SCRIPT_SHA,
        "Canonical parent script SHA mismatch",
    )

    return protocol, {
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha,
        "script_path": str(script_path),
        "script_sha256": script_sha,
        "runtime": observed_runtime,
        "verified_identities": identities,
    }


def exact_arch_feasibility(
    history_decimal: np.ndarray,
    model_specification: dict[str, Any],
    candidate_values: dict[str, float],
    input_multiplier: float,
) -> dict[str, Any]:
    fit_input = np.asarray(history_decimal, dtype=np.float64) * float(input_multiplier)
    require(np.isfinite(fit_input).all(), "Feasibility input contains non-finite values")

    model = arch_model(
        fit_input,
        mean=model_specification["mean"],
        vol=model_specification["vol"],
        p=int(model_specification["p"]),
        o=int(model_specification["o"]),
        q=int(model_specification["q"]),
        power=float(model_specification["power"]),
        dist=model_specification["dist"],
        rescale=bool(model_specification["rescale"]),
    )

    residuals = fit_input.copy()
    volatility = model.volatility
    distribution = model.distribution
    backcast = volatility.backcast(residuals)
    variance_bounds = volatility.variance_bounds(residuals)
    sigma2 = np.zeros(residuals.shape[0], dtype=np.float64)
    volatility_start = volatility.starting_values(residuals)
    volatility.compute_variance(volatility_start, residuals, sigma2, backcast, variance_bounds)
    standardized_residuals = residuals / np.sqrt(sigma2)

    component_constraints = (
        model.constraints(),
        volatility.constraints(),
        distribution.constraints(),
    )
    component_parameter_counts = np.array(
        (model.num_params, volatility.num_params, distribution.num_params), dtype=int
    )
    constraint_counts = np.array([item[0].shape[0] for item in component_constraints], dtype=int)
    total_parameters = int(component_parameter_counts.sum())
    total_constraints = int(constraint_counts.sum())
    matrix_a = np.zeros((total_constraints, total_parameters), dtype=np.float64)
    vector_b = np.zeros(total_constraints, dtype=np.float64)

    for index, (local_a, local_b) in enumerate(component_constraints):
        row_end = int(constraint_counts[: index + 1].sum())
        row_start = row_end - int(constraint_counts[index])
        column_end = int(component_parameter_counts[: index + 1].sum())
        column_start = column_end - int(component_parameter_counts[index])
        if row_end > row_start:
            matrix_a[row_start:row_end, column_start:column_end] = local_a
            vector_b[row_start:row_end] = local_b

    bounds = list(model.bounds())
    bounds.extend(volatility.bounds(residuals))
    bounds.extend(distribution.bounds(standardized_residuals))
    parameter_names = (
        list(model.parameter_names())
        + list(volatility.parameter_names())
        + list(distribution.parameter_names())
    )
    require(set(parameter_names) == set(candidate_values), "Candidate parameter names mismatch")
    candidate_vector = np.array([candidate_values[name] for name in parameter_names], dtype=np.float64)
    require(candidate_vector.shape[0] == total_parameters, "Candidate length mismatch")

    constraint_margins = (
        matrix_a @ candidate_vector - vector_b
        if total_constraints > 0
        else np.array([np.inf], dtype=np.float64)
    )
    lower_margins = np.array(
        [candidate_vector[index] - bound[0] for index, bound in enumerate(bounds)], dtype=np.float64
    )
    upper_margins = np.array(
        [bound[1] - candidate_vector[index] for index, bound in enumerate(bounds)], dtype=np.float64
    )

    finite = bool(np.isfinite(candidate_vector).all())
    constraints_valid = bool(np.all(constraint_margins >= 0.0))
    bounds_valid = bool(np.all(lower_margins >= 0.0) and np.all(upper_margins >= 0.0))
    return {
        "parameter_names": parameter_names,
        "candidate_vector": candidate_vector,
        "finite": finite,
        "constraints_valid": constraints_valid,
        "bounds_valid": bounds_valid,
        "fully_feasible": bool(finite and constraints_valid and bounds_valid),
        "minimum_constraint_margin": float(np.min(constraint_margins)),
        "minimum_lower_bound_margin": float(np.min(lower_margins)),
        "minimum_upper_bound_margin": float(np.min(upper_margins)),
    }


def append_failure_reason(attempt_row: dict[str, Any], reason: str) -> None:
    current = [item for item in str(attempt_row.get("failure_reasons", "")).split("|") if item]
    if reason not in current:
        current.append(reason)
    attempt_row["failure_reasons"] = "|".join(current)
    attempt_row["attempt_valid"] = False


def candidate_from_audit(row: pd.Series, model_spec: dict[str, Any]) -> tuple[dict[str, float], np.ndarray]:
    candidate = {
        "omega": float(row["omega_candidate"]),
        "alpha[1]": float(row["alpha_candidate"]),
        "beta[1]": float(row["beta_candidate"]),
        "nu": float(row["nu_candidate"]),
    }
    if int(model_spec["o"]) == 1:
        candidate["gamma[1]"] = float(row["gamma_candidate"])
        ordered = np.array(
            [candidate["omega"], candidate["alpha[1]"], candidate["gamma[1]"], candidate["beta[1]"], candidate["nu"]],
            dtype=np.float64,
        )
    else:
        ordered = np.array(
            [candidate["omega"], candidate["alpha[1]"], candidate["beta[1]"], candidate["nu"]],
            dtype=np.float64,
        )
    return candidate, ordered


def reconstruct_candidate(source: pd.Series, model_spec: dict[str, Any]) -> dict[str, float]:
    omega = float(source["omega"])
    alpha = float(source["alpha1"])
    beta = float(source["beta1"])
    nu = float(source["nu"])
    if int(model_spec["o"]) == 1:
        gamma = float(source["gamma1"])
        persistence = alpha + 0.5 * gamma + beta
    else:
        gamma = float("nan")
        persistence = alpha + beta
    require(math.isfinite(persistence) and persistence > 0.0, "Invalid source persistence")
    factor = PERSISTENCE_TARGET / persistence if persistence > PERSISTENCE_TARGET else 1.0
    candidate = {
        "omega": omega,
        "alpha_candidate": alpha * factor,
        "gamma_candidate": gamma * factor if int(model_spec["o"]) == 1 else float("nan"),
        "beta_candidate": beta * factor,
        "nu": nu,
        "persistence_before": persistence,
        "projection_factor": factor,
    }
    return candidate


def close_float(left: float, right: float, label: str) -> None:
    require(
        np.isclose(float(left), float(right), rtol=FLOAT_RTOL, atol=FLOAT_ATOL, equal_nan=True),
        f"Float mismatch for {label}: left={left!r} right={right!r}",
    )


def validate_target_inventory(
    root: Path,
    protocol: dict[str, Any],
    parent_protocol: dict[str, Any],
    data: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], pd.DataFrame]:
    attempts_path = root / protocol["parent_08C"]["attempts"]["relative_path"]
    diagnostics_path = root / protocol["parent_08C"]["diagnostics"]["relative_path"]
    audit_path = root / protocol["prelock_feasibility_evidence"]["audit_csv"]["relative_path"]

    attempts = pd.read_csv(attempts_path)
    diagnostics = pd.read_csv(diagnostics_path)
    audit = pd.read_csv(audit_path)
    attempts["attempt_valid_bool"] = parse_bool(attempts["attempt_valid"])

    unresolved = diagnostics[diagnostics["status"].astype(str).eq("UNRESOLVED_FIT")].copy()
    unresolved = unresolved.sort_values(["model_id", "anchor_index"]).reset_index(drop=True)
    require(len(unresolved) == EXPECTED_TOTAL_TARGETS, "Parent unresolved target count is not 574")
    require(set(unresolved["asset"].astype(str)) == {"USDTRY"}, "Target inventory contains another asset")
    require(
        unresolved.groupby("model_id").size().astype(int).to_dict() == EXPECTED_TARGET_COUNTS,
        "Parent unresolved model counts mismatch",
    )
    require(not unresolved.duplicated(["model_id", "asset", "anchor_index"]).any(), "Duplicate unresolved target")

    require(len(audit) == EXPECTED_TOTAL_TARGETS, "Audit row count is not 574")
    require(set(audit["asset"].astype(str)) == {"USDTRY"}, "Audit contains another asset")
    require(
        audit.groupby("model_id").size().astype(int).to_dict() == EXPECTED_TARGET_COUNTS,
        "Audit model counts mismatch",
    )
    for column in ("projection_applied", "finite", "constraints_valid", "bounds_valid", "fully_feasible"):
        require(parse_bool(audit[column]).all(), f"Audit column is not all True: {column}")

    unresolved_keys = set(zip(unresolved["model_id"].astype(str), unresolved["anchor_index"].astype(int)))
    audit_keys = set(zip(audit["model_id"].astype(str), audit["failed_anchor"].astype(int)))
    require(unresolved_keys == audit_keys, "Audit targets do not match unresolved targets")

    model_specs = {item["model_id"]: item for item in parent_protocol["official_models"]}
    require(set(EXPECTED_TARGET_COUNTS).issubset(model_specs), "Expected model spec missing")
    asset_index = list(parent_protocol["assets"]).index("USDTRY")
    returns = np.asarray(data["returns"][:, asset_index], dtype=np.float64)
    input_multiplier = float(parent_protocol["fit_scale"]["input_log_return_multiplier"])

    primary = attempts[
        attempts["asset"].astype(str).eq("USDTRY")
        & attempts["attempt_number"].astype(int).eq(1)
    ].copy()

    validated_rows: list[dict[str, Any]] = []
    for audit_row in audit.sort_values(["model_id", "failed_anchor"]).itertuples(index=False):
        row = pd.Series(audit_row._asdict())
        model_id = str(row["model_id"])
        failed_anchor = int(row["failed_anchor"])
        model_spec = model_specs[model_id]

        earlier = primary[
            primary["model_id"].astype(str).eq(model_id)
            & primary["attempt_valid_bool"]
            & primary["anchor_index"].astype(int).lt(failed_anchor)
        ].sort_values("anchor_index")
        require(not earlier.empty, f"No prior successful primary source: {model_id} anchor={failed_anchor}")
        source = earlier.iloc[-1]
        require(int(source["anchor_index"]) == int(row["source_anchor"]), "Nearest prior source mismatch")
        require(str(source["anchor_date"]) == str(row["source_date"]), "Source date mismatch")
        expected_anchor_date = pd.Timestamp(data["anchor_dates"][failed_anchor]).strftime("%Y-%m-%d")
        require(str(row["failed_date"]) == expected_anchor_date, "Failed-anchor date mismatch")
        require(int(row["source_attempt_number"]) == 1, "Audit source attempt is not one")

        reconstructed = reconstruct_candidate(source, model_spec)
        close_float(source["omega"], row["omega_source"], "omega_source")
        close_float(source["alpha1"], row["alpha_source"], "alpha_source")
        close_float(source["beta1"], row["beta_source"], "beta_source")
        close_float(source["nu"], row["nu_source"], "nu_source")
        if int(model_spec["o"]) == 1:
            close_float(source["gamma1"], row["gamma_source"], "gamma_source")
        close_float(reconstructed["persistence_before"], row["persistence_before"], "persistence_before")
        close_float(reconstructed["projection_factor"], row["projection_factor"], "projection_factor")
        close_float(reconstructed["alpha_candidate"], row["alpha_candidate"], "alpha_candidate")
        close_float(reconstructed["beta_candidate"], row["beta_candidate"], "beta_candidate")
        close_float(reconstructed["nu"], row["nu_candidate"], "nu_candidate")
        close_float(reconstructed["omega"], row["omega_candidate"], "omega_candidate")
        if int(model_spec["o"]) == 1:
            close_float(reconstructed["gamma_candidate"], row["gamma_candidate"], "gamma_candidate")
            reconstructed_after = (
                reconstructed["alpha_candidate"]
                + 0.5 * reconstructed["gamma_candidate"]
                + reconstructed["beta_candidate"]
            )
        else:
            reconstructed_after = reconstructed["alpha_candidate"] + reconstructed["beta_candidate"]
        close_float(reconstructed_after, row["persistence_after"], "persistence_after")
        close_float(row["persistence_after"], PERSISTENCE_TARGET, "locked persistence target")

        candidate_values, candidate_vector = candidate_from_audit(row, model_spec)
        anchor_position = int(data["anchor_positions"][failed_anchor])
        history = returns[: anchor_position + 1]
        feasibility = exact_arch_feasibility(history, model_spec, candidate_values, input_multiplier)
        require(feasibility["fully_feasible"], f"Candidate failed exact feasibility recheck: {model_id} {failed_anchor}")
        close_float(row["minimum_constraint_margin"], feasibility["minimum_constraint_margin"], "constraint margin")
        close_float(row["minimum_lower_bound_margin"], feasibility["minimum_lower_bound_margin"], "lower-bound margin")
        close_float(row["minimum_upper_bound_margin"], feasibility["minimum_upper_bound_margin"], "upper-bound margin")

        validated_rows.append(
            {
                "protocol_stage": STAGE,
                "model_id": model_id,
                "asset": "USDTRY",
                "anchor_index": failed_anchor,
                "anchor_date": str(row["failed_date"]),
                "source_anchor": int(row["source_anchor"]),
                "source_date": str(row["source_date"]),
                "source_attempt_number": 1,
                "candidate_values": candidate_values,
                "candidate_vector": candidate_vector,
                "minimum_constraint_margin": feasibility["minimum_constraint_margin"],
                "minimum_lower_bound_margin": feasibility["minimum_lower_bound_margin"],
                "minimum_upper_bound_margin": feasibility["minimum_upper_bound_margin"],
                "candidate_matches_locked_audit": True,
                "model_spec": model_spec,
            }
        )

    validated = pd.DataFrame(validated_rows).sort_values(["model_id", "anchor_index"]).reset_index(drop=True)
    require(len(validated) == EXPECTED_TOTAL_TARGETS, "Validated target count mismatch")
    return validated, model_specs, audit


def postfit_prediction(
    parent: Any,
    parent_protocol: dict[str, Any],
    output: dict[str, Any],
    history: np.ndarray,
    truth: float,
) -> dict[str, Any]:
    target_definition = parent_protocol["target_definition"]
    vol_window = int(target_definition["vol20_window"])
    known_count = vol_window - 1
    annualization_factor = float(target_definition["annualization_factor"])
    require(known_count == 19, "Locked known-return count must be 19")
    require(len(history) >= known_count, "Insufficient known returns")
    known_returns = np.asarray(history[-known_count:], dtype=np.float64)
    require(np.isfinite(known_returns).all(), "Known return window contains non-finite values")

    S = float(np.sum(known_returns, dtype=np.float64))
    Q = float(np.sum(known_returns**2, dtype=np.float64))
    c = S / float(known_count)
    C_raw = (Q - (S**2) / float(known_count)) / float(known_count)
    C, C_corrected = parent.safe_nonnegative(C_raw, name="C")

    root = parent.solve_conditional_median_q(
        c=c,
        sigma_decimal=output["sigma_decimal"],
        nu=output["nu"],
        root_settings=parent_protocol["primary_forecast"]["root_settings"],
        distribution=parent.StudentsT(),
    )
    q_value = float(root["q"])
    primary_inside, primary_corrected = parent.safe_nonnegative(
        C + (q_value**2) / float(vol_window), name="primary_forecast_inside"
    )
    primary_prediction = math.sqrt(annualization_factor * primary_inside)
    plugin_inside, plugin_corrected = parent.safe_nonnegative(
        C + (output["h_decimal"] + c**2) / float(vol_window), name="plugin_forecast_inside"
    )
    plugin_prediction = math.sqrt(annualization_factor * plugin_inside)
    direct_inside, _ = parent.safe_nonnegative(
        output["h_decimal"] * annualization_factor, name="direct_sigma_inside"
    )
    direct_prediction = math.sqrt(direct_inside)
    predictions = np.asarray([primary_prediction, plugin_prediction, direct_prediction], dtype=np.float64)
    require(np.isfinite(predictions).all(), "One or more rescue predictions are non-finite")
    loss = float(
        parent.pinball_loss_series(
            np.asarray([truth], dtype=np.float64),
            np.asarray([primary_prediction], dtype=np.float64),
            tau=float(parent_protocol["metrics"]["primary"]["tau"]),
        )[0]
    )
    require(math.isfinite(loss), "Rescue primary pinball loss is non-finite")
    return {
        "S": S,
        "Q": Q,
        "c": c,
        "C": C,
        "C_rounding_corrected": C_corrected,
        "q": q_value,
        "central_probability_mass": root["central_probability_mass"],
        "central_mass_error": root["central_mass_error"],
        "root_lower_bound": root["root_lower_bound"],
        "root_upper_bound": root["root_upper_bound"],
        "root_upper_bound_doublings": root["root_upper_bound_doublings"],
        "primary_prediction": primary_prediction,
        "plugin_prediction": plugin_prediction,
        "direct_sigma_prediction": direct_prediction,
        "primary_pinball_loss": loss,
        "primary_inside_rounding_corrected": primary_corrected,
        "plugin_inside_rounding_corrected": plugin_corrected,
    }


def execute_target(
    parent: Any,
    protocol: dict[str, Any],
    protocol_info: dict[str, Any],
    parent_protocol: dict[str, Any],
    data: dict[str, Any],
    target: pd.Series,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    model_id = str(target["model_id"])
    model_spec = dict(target["model_spec"])
    anchor_index = int(target["anchor_index"])
    asset = "USDTRY"
    asset_index = list(parent_protocol["assets"]).index(asset)
    anchor_position = int(data["anchor_positions"][anchor_index])
    anchor_date = pd.Timestamp(data["anchor_dates"][anchor_index]).strftime("%Y-%m-%d")
    realization_date = pd.Timestamp(data["realization_dates"][anchor_index]).strftime("%Y-%m-%d")
    return_series = np.asarray(data["returns"][:, asset_index], dtype=np.float64)
    history = return_series[: anchor_position + 1]
    truth = float(data["y_vol_truth"][anchor_index, asset_index])

    candidate_values = dict(target["candidate_values"])
    candidate_vector = np.asarray(target["candidate_vector"], dtype=np.float64)
    feasibility = exact_arch_feasibility(
        history,
        model_spec,
        candidate_values,
        float(parent_protocol["fit_scale"]["input_log_return_multiplier"]),
    )
    require(feasibility["fully_feasible"], "Pre-fit feasibility recheck failed")

    fit_call = protocol["fit_policy"]["fit_call"]
    attempt_settings = {
        "update_freq": int(fit_call["update_freq"]),
        "disp": fit_call["disp"],
        "show_warning": bool(fit_call["show_warning"]),
        "starting_values": candidate_vector,
        "cov_type": fit_call["cov_type"],
        "tol": float(fit_call["tol"]),
        "options": dict(fit_call["options"]),
        "backcast": fit_call["backcast"],
    }

    attempt_row, warning_rows, output = parent.fit_one_attempt(
        return_history_decimal=history,
        model_spec=model_spec,
        attempt_settings=attempt_settings,
        fit_scale=parent_protocol["fit_scale"],
        protocol_sha=protocol_info["protocol_sha256"],
        script_sha=protocol_info["script_sha256"],
        model_id=model_id,
        asset=asset,
        anchor_index=anchor_index,
        anchor_date=anchor_date,
        attempt_number=1,
    )

    for warning_row in warning_rows:
        warning_row["protocol_stage"] = STAGE
        warning_row["is_starting_value_warning"] = (
            str(warning_row.get("warning_category", "")) == "StartingValueWarning"
        )

    starting_value_warning_count = sum(int(row["is_starting_value_warning"]) for row in warning_rows)
    convergence_warning_count = sum(int(row["is_convergence_warning"]) for row in warning_rows)
    if starting_value_warning_count > 0:
        append_failure_reason(attempt_row, "captured_starting_value_warning")
        output = None
    if convergence_warning_count > 0:
        append_failure_reason(attempt_row, "captured_convergence_warning")
        output = None

    attempt_row.update(
        {
            "protocol_stage": STAGE,
            "rescue_attempt": True,
            "source_anchor": int(target["source_anchor"]),
            "source_date": str(target["source_date"]),
            "source_attempt_number": 1,
            "starting_values_submitted_to_arch": True,
            "candidate_matches_locked_audit": True,
            "candidate_fully_feasible_before_fit": True,
            "candidate_parameter_names_json": json.dumps(feasibility["parameter_names"]),
            "candidate_vector_json": json.dumps(candidate_vector.tolist()),
            "candidate_omega": float(candidate_values["omega"]),
            "candidate_alpha1": float(candidate_values["alpha[1]"]),
            "candidate_gamma1": float(candidate_values.get("gamma[1]", np.nan)),
            "candidate_beta1": float(candidate_values["beta[1]"]),
            "candidate_nu": float(candidate_values["nu"]),
            "candidate_minimum_constraint_margin": feasibility["minimum_constraint_margin"],
            "candidate_minimum_lower_bound_margin": feasibility["minimum_lower_bound_margin"],
            "candidate_minimum_upper_bound_margin": feasibility["minimum_upper_bound_margin"],
            "starting_value_warning_count": starting_value_warning_count,
            "convergence_warning_count": convergence_warning_count,
            "accepted_under_08C_R": bool(output is not None and attempt_row["attempt_valid"]),
        }
    )

    diagnostic: dict[str, Any] = {
        "protocol_stage": STAGE,
        "protocol_sha256": protocol_info["protocol_sha256"],
        "script_sha256": protocol_info["script_sha256"],
        "model_id": model_id,
        "asset": asset,
        "asset_index": asset_index,
        "anchor_index": anchor_index,
        "anchor_position_zero_based": anchor_position,
        "anchor_date": anchor_date,
        "target_realization_date": realization_date,
        "fit_n": int(len(history)),
        "truth_nextvol": truth,
        "attempts_used": 1,
        "retry_used": False,
        "accepted_attempt": 1 if output is not None and attempt_row["attempt_valid"] else None,
        "warning_count_total": len(warning_rows),
        "starting_value_warning_count": starting_value_warning_count,
        "convergence_warning_count": convergence_warning_count,
        "other_warning_count": len(warning_rows) - starting_value_warning_count - convergence_warning_count,
        "source_anchor": int(target["source_anchor"]),
        "source_date": str(target["source_date"]),
        "source_attempt_number": 1,
        "starting_values_submitted_to_arch": True,
        "candidate_matches_locked_audit": True,
        "candidate_fully_feasible_before_fit": True,
        "candidate_vector_json": json.dumps(candidate_vector.tolist()),
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
        "optimizer_status": attempt_row.get("optimizer_status"),
        "optimizer_message": attempt_row.get("optimizer_message", ""),
        "result_label": "",
    }

    if output is None or not bool(attempt_row["attempt_valid"]):
        diagnostic["status"] = "UNRESOLVED_FIT"
        diagnostic["failure_stage"] = "fit_or_fitted_output"
        diagnostic["failure_type"] = "RescueAttemptFailed"
        diagnostic["failure_message"] = json.dumps(
            {
                "attempt_number": 1,
                "failure_reasons": attempt_row.get("failure_reasons", ""),
                "exception_type": attempt_row.get("exception_type", ""),
                "exception_message": attempt_row.get("exception_message", ""),
            },
            sort_keys=True,
        )
        return diagnostic, attempt_row, warning_rows

    diagnostic.update(
        {
            "omega": output["omega"],
            "alpha1": output["alpha1"],
            "gamma1": output["gamma1"],
            "beta1": output["beta1"],
            "nu": output["nu"],
            "persistence": output["persistence"],
            "h_percent_sq": output["h_percent_sq"],
            "h_decimal": output["h_decimal"],
            "sigma_decimal": output["sigma_decimal"],
            "variance_rounding_corrected": output["variance_rounding_corrected"],
            "optimizer_status": output["optimizer_status"],
            "optimizer_message": output["optimizer_message"],
        }
    )

    try:
        prediction = postfit_prediction(parent, parent_protocol, output, history, truth)
        diagnostic.update(prediction)
        diagnostic["status"] = "OK"
        diagnostic["result_label"] = (
            "Obtained under the locked 08C-R numerical-convergence rescue protocol."
        )
    except Exception as error:
        diagnostic["status"] = "UNRESOLVED_ROOT_OR_PREDICTION"
        diagnostic["failure_stage"] = "cdf_root_or_prediction"
        diagnostic["failure_type"] = type(error).__name__
        diagnostic["failure_message"] = str(error)

    return diagnostic, attempt_row, warning_rows


def checkpoint_paths(output_dir: Path) -> dict[str, Path]:
    checkpoint_dir = output_dir / CHECKPOINT_DIRNAME
    return {
        "dir": checkpoint_dir,
        "attempts": checkpoint_dir / CHECKPOINT_ATTEMPTS,
        "diagnostics": checkpoint_dir / CHECKPOINT_DIAGNOSTICS,
        "warnings": checkpoint_dir / CHECKPOINT_WARNINGS,
        "state": checkpoint_dir / CHECKPOINT_STATE,
    }


def validate_checkpoint_frame(
    frame: pd.DataFrame,
    protocol_sha: str,
    script_sha: str,
    label: str,
) -> None:
    if frame.empty:
        return
    for column in ("protocol_stage", "protocol_sha256", "script_sha256", "model_id", "asset", "anchor_index"):
        require(column in frame.columns, f"{label} checkpoint lacks {column}")
    require(set(frame["protocol_stage"].astype(str)) == {STAGE}, f"{label} checkpoint stage mismatch")
    require(set(frame["protocol_sha256"].astype(str)) == {protocol_sha}, f"{label} checkpoint protocol mismatch")
    require(set(frame["script_sha256"].astype(str)) == {script_sha}, f"{label} checkpoint script mismatch")
    require(set(frame["asset"].astype(str)) == {"USDTRY"}, f"{label} checkpoint asset mismatch")


def save_checkpoints(
    paths: dict[str, Path],
    diagnostics: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    warnings_rows: list[dict[str, Any]],
    protocol_info: dict[str, Any],
) -> None:
    paths["dir"].mkdir(parents=True, exist_ok=True)
    diagnostics_frame = pd.DataFrame(diagnostics).sort_values(["model_id", "anchor_index"]).reset_index(drop=True)
    attempts_frame = pd.DataFrame(attempts).sort_values(["model_id", "anchor_index"]).reset_index(drop=True)
    warnings_frame = pd.DataFrame(warnings_rows, columns=WARNING_COLUMNS)
    if not warnings_frame.empty:
        warnings_frame = warnings_frame.sort_values(
            ["model_id", "anchor_index", "attempt_number", "warning_index"]
        ).reset_index(drop=True)
    atomic_write_csv(paths["diagnostics"], diagnostics_frame)
    atomic_write_csv(paths["attempts"], attempts_frame)
    atomic_write_csv(paths["warnings"], warnings_frame)
    state = {
        "project_version": PROJECT_VERSION,
        "stage": STAGE,
        "updated_at_utc": utc_now_iso(),
        "protocol_sha256": protocol_info["protocol_sha256"],
        "script_sha256": protocol_info["script_sha256"],
        "processed_target_count": int(len(diagnostics_frame)),
        "expected_target_count": EXPECTED_TOTAL_TARGETS,
        "fits_per_target": 1,
        "merge_performed": False,
        "aggregate_metrics_computed": False,
    }
    atomic_write_json(paths["state"], state)


def load_checkpoints(
    paths: dict[str, Path],
    protocol_info: dict[str, Any],
    valid_target_keys: set[tuple[str, int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoint_files = [paths["diagnostics"], paths["attempts"], paths["warnings"], paths["state"]]
    existing_checkpoint_files = [path for path in checkpoint_files if path.exists()]
    require(
        len(existing_checkpoint_files) in (0, len(checkpoint_files)),
        "Incomplete checkpoint file set detected; no file was modified",
    )
    diagnostics = read_csv_safe(paths["diagnostics"])
    attempts = read_csv_safe(paths["attempts"])
    warnings_frame = read_csv_safe(paths["warnings"])
    validate_checkpoint_frame(diagnostics, protocol_info["protocol_sha256"], protocol_info["script_sha256"], "diagnostics")
    validate_checkpoint_frame(attempts, protocol_info["protocol_sha256"], protocol_info["script_sha256"], "attempts")
    validate_checkpoint_frame(warnings_frame, protocol_info["protocol_sha256"], protocol_info["script_sha256"], "warnings")

    if existing_checkpoint_files:
        with paths["state"].open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        require(state.get("stage") == STAGE, "Checkpoint state stage mismatch")
        require(state.get("protocol_sha256") == protocol_info["protocol_sha256"], "Checkpoint state protocol mismatch")
        require(state.get("script_sha256") == protocol_info["script_sha256"], "Checkpoint state script mismatch")
        require(int(state.get("expected_target_count", -1)) == EXPECTED_TOTAL_TARGETS, "Checkpoint state target mismatch")
        require(int(state.get("fits_per_target", -1)) == 1, "Checkpoint state retry mismatch")
        require(state.get("merge_performed") is False, "Checkpoint state indicates merge")
        require(state.get("aggregate_metrics_computed") is False, "Checkpoint state indicates metrics")

    require(len(diagnostics) == len(attempts), "Checkpoint diagnostics/attempt counts differ")
    if not diagnostics.empty:
        require(not diagnostics.duplicated(["model_id", "anchor_index"]).any(), "Duplicate diagnostic checkpoint target")
        require(not attempts.duplicated(["model_id", "anchor_index"]).any(), "Duplicate attempt checkpoint target")
        diagnostic_keys = set(zip(diagnostics["model_id"].astype(str), diagnostics["anchor_index"].astype(int)))
        attempt_keys = set(zip(attempts["model_id"].astype(str), attempts["anchor_index"].astype(int)))
        require(diagnostic_keys == attempt_keys, "Checkpoint target sets differ")
        require(diagnostic_keys.issubset(valid_target_keys), "Checkpoint contains a non-target row")
        require(attempts["attempt_number"].astype(int).eq(1).all(), "Checkpoint contains a retry")
        require(
            int(state.get("processed_target_count", -1)) == len(diagnostics),
            "Checkpoint state processed count mismatch",
        )
    return diagnostics.to_dict("records"), attempts.to_dict("records"), warnings_frame.to_dict("records")


def finalize_outputs(
    root: Path,
    output_dir: Path,
    protocol: dict[str, Any],
    protocol_info: dict[str, Any],
    parent_info: dict[str, Any],
    diagnostics: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    warnings_rows: list[dict[str, Any]],
    started_at_utc: str,
) -> None:
    final_paths = {
        "attempts": output_dir / FINAL_ATTEMPTS,
        "diagnostics": output_dir / FINAL_DIAGNOSTICS,
        "warnings": output_dir / FINAL_WARNINGS,
        "summary": output_dir / FINAL_SUMMARY,
        "manifest": output_dir / FINAL_MANIFEST,
    }
    existing = [path for path in final_paths.values() if path.exists()]
    require(not existing, "Final rescue output already exists; nothing was overwritten:\n" + "\n".join(map(str, existing)))

    diagnostics_frame = pd.DataFrame(diagnostics).sort_values(["model_id", "anchor_index"]).reset_index(drop=True)
    attempts_frame = pd.DataFrame(attempts).sort_values(["model_id", "anchor_index"]).reset_index(drop=True)
    warnings_frame = pd.DataFrame(warnings_rows, columns=WARNING_COLUMNS)
    if not warnings_frame.empty:
        warnings_frame = warnings_frame.sort_values(
            ["model_id", "anchor_index", "attempt_number", "warning_index"]
        ).reset_index(drop=True)

    require(len(diagnostics_frame) == EXPECTED_TOTAL_TARGETS, "Final diagnostics count is not 574")
    require(len(attempts_frame) == EXPECTED_TOTAL_TARGETS, "Final attempt count is not 574")
    require(not diagnostics_frame.duplicated(["model_id", "anchor_index"]).any(), "Duplicate final diagnostic target")
    require(not attempts_frame.duplicated(["model_id", "anchor_index"]).any(), "Duplicate final attempt target")
    require(attempts_frame["attempt_number"].astype(int).eq(1).all(), "Final output contains a retry")
    require(parse_bool(attempts_frame["starting_values_submitted_to_arch"]).all(), "A fit lacks submitted starting values")
    require(parse_bool(attempts_frame["candidate_matches_locked_audit"]).all(), "A fit lacks audit match")
    require(parse_bool(attempts_frame["candidate_fully_feasible_before_fit"]).all(), "A fit lacks feasibility confirmation")

    status_counts = diagnostics_frame["status"].astype(str).value_counts(dropna=False).to_dict()
    model_status_counts: dict[str, dict[str, int]] = {}
    for model_id, group in diagnostics_frame.groupby("model_id", sort=True):
        model_status_counts[str(model_id)] = {
            str(key): int(value)
            for key, value in group["status"].astype(str).value_counts(dropna=False).to_dict().items()
        }

    summary = {
        "project_version": PROJECT_VERSION,
        "stage": STAGE,
        "protocol_sha256": protocol_info["protocol_sha256"],
        "script_sha256": protocol_info["script_sha256"],
        "completed_at_utc": utc_now_iso(),
        "target_count": EXPECTED_TOTAL_TARGETS,
        "attempt_count": int(len(attempts_frame)),
        "single_attempt_policy_respected": True,
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "status_counts_by_model": model_status_counts,
        "successful_observations": int((diagnostics_frame["status"].astype(str) == "OK").sum()),
        "remaining_unresolved_fit": int((diagnostics_frame["status"].astype(str) == "UNRESOLVED_FIT").sum()),
        "remaining_unresolved_root_or_prediction": int(
            (diagnostics_frame["status"].astype(str) == "UNRESOLVED_ROOT_OR_PREDICTION").sum()
        ),
        "rescue_clean_closure": bool((diagnostics_frame["status"].astype(str) == "OK").all()),
        "warning_count": int(len(warnings_frame)),
        "starting_value_warning_count": int(
            parse_bool(warnings_frame["is_starting_value_warning"]).sum() if not warnings_frame.empty else 0
        ),
        "convergence_warning_count": int(
            parse_bool(warnings_frame["is_convergence_warning"]).sum() if not warnings_frame.empty else 0
        ),
        "merge_performed": False,
        "aggregate_metrics_computed": False,
        "parent_08C_modified": False,
        "interpretation": (
            "This summary reports only the locked 08C-R rescue attempts. "
            "It does not merge parent and rescue rows and does not establish final 08C closure."
        ),
    }

    atomic_write_csv(final_paths["attempts"], attempts_frame)
    atomic_write_csv(final_paths["diagnostics"], diagnostics_frame)
    atomic_write_csv(final_paths["warnings"], warnings_frame)
    atomic_write_json(final_paths["summary"], summary)

    output_identities = {
        key: {
            "relative_path": str(path.relative_to(root)),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for key, path in final_paths.items()
        if key != "manifest"
    }
    manifest = {
        "project_version": PROJECT_VERSION,
        "stage": STAGE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": utc_now_iso(),
        "protocol": {
            "relative_path": str(ACTIVE_PROTOCOL_REL),
            "sha256": protocol_info["protocol_sha256"],
        },
        "script": {
            "relative_path": str(Path(protocol_info["script_path"]).relative_to(root)),
            "sha256": protocol_info["script_sha256"],
        },
        "runtime": protocol_info["runtime"],
        "verified_protocol_identities": protocol_info["verified_identities"],
        "parent_validation": parent_info,
        "target_inventory": {
            "total": EXPECTED_TOTAL_TARGETS,
            "by_model": EXPECTED_TARGET_COUNTS,
            "asset": "USDTRY",
        },
        "outputs": output_identities,
        "merge_performed": False,
        "aggregate_metrics_computed": False,
        "parent_08C_modified": False,
    }
    atomic_write_json(final_paths["manifest"], manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run locked 08C-R USDTRY GARCH rescue fits")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Project root")
    parser.add_argument("--validate-only", action="store_true", help="Validate all locks and targets; perform no fit and write no file")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Atomic checkpoint frequency")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    require(root.exists(), f"Project root does not exist: {root}")
    require(args.checkpoint_every >= 1, "--checkpoint-every must be >= 1")

    script_path = Path(__file__).resolve()
    started_at_utc = utc_now_iso()

    print("=" * 122)
    print("08C-R LOCKED USDTRY GARCH RESCUE")
    print("=" * 122)
    print(f"Project root : {root}")
    print(f"Script path  : {script_path}")

    protocol, protocol_info = load_and_validate_protocol(root, script_path)
    print(f"Script SHA   : {protocol_info['script_sha256']}")
    print(f"Protocol SHA : {protocol_info['protocol_sha256']}")

    parent_script_path = root / PARENT_SCRIPT_REL
    parent = load_parent_module(parent_script_path)
    parent_protocol, parent_protocol_info = parent.load_and_verify_protocol(root)
    parent_locked_inputs = parent.verify_locked_inputs(root, parent_protocol)
    data = parent.validate_data_alignment(root, parent_protocol)
    require(parent_protocol_info["protocol_sha256"] == protocol["parent_08C"]["protocol"]["sha256"], "Parent protocol SHA mismatch")
    require(sha256_file(parent_script_path) == EXPECTED_PARENT_SCRIPT_SHA, "Parent script changed")

    validated_targets, model_specs, _audit = validate_target_inventory(
        root, protocol, parent_protocol, data
    )
    require(len(model_specs) >= 2, "Parent model inventory is incomplete")

    required_outputs = set(protocol["output_policy"]["required_outputs"])
    require(
        required_outputs == {FINAL_ATTEMPTS, FINAL_DIAGNOSTICS, FINAL_WARNINGS, FINAL_SUMMARY, FINAL_MANIFEST},
        "Protocol required-output inventory differs from script output inventory",
    )

    output_dir = root / OUTPUT_DIR_REL
    final_paths = [output_dir / name for name in required_outputs]
    existing_final = [path for path in final_paths if path.exists()]
    require(not existing_final, "Final rescue output already exists:\n" + "\n".join(map(str, existing_final)))

    print("\nVALIDATION PASSED")
    print(f"Verified parent locked inputs : {len(parent_locked_inputs)}/10")
    print(f"Validated rescue targets      : {len(validated_targets)}/574")
    print("GARCH USDTRY                  : 329")
    print("GJR-GARCH USDTRY              : 245")
    print("Exact feasibility rechecked   : 574/574")
    print("Single rescue attempt locked  : YES")
    print("Merge and aggregate metrics   : DISABLED")

    if args.validate_only:
        print("\nVALIDATE-ONLY COMPLETE")
        print("Files written : NO")
        print("Fits performed: NO")
        print("Metrics merged: NO")
        print("Parent changed: NO")
        print("=" * 122)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = checkpoint_paths(output_dir)
    valid_keys = set(zip(validated_targets["model_id"].astype(str), validated_targets["anchor_index"].astype(int)))
    diagnostic_rows, attempt_rows, warning_rows = load_checkpoints(paths, protocol_info, valid_keys)
    processed_keys = set(
        (str(row["model_id"]), int(row["anchor_index"])) for row in diagnostic_rows
    )
    print(f"\nCheckpoint resume: {len(processed_keys)}/574 targets already processed")

    remaining = validated_targets[
        ~validated_targets.apply(
            lambda row: (str(row["model_id"]), int(row["anchor_index"])) in processed_keys,
            axis=1,
        )
    ].reset_index(drop=True)

    run_start = time.perf_counter()
    since_checkpoint = 0
    for index, target in remaining.iterrows():
        diagnostic, attempt, warnings_for_target = execute_target(
            parent,
            protocol,
            protocol_info,
            parent_protocol,
            data,
            target,
        )
        diagnostic_rows.append(diagnostic)
        attempt_rows.append(attempt)
        warning_rows.extend(warnings_for_target)
        since_checkpoint += 1
        processed_total = len(diagnostic_rows)
        print(
            f"[{processed_total:03d}/574] {diagnostic['model_id']} "
            f"anchor={diagnostic['anchor_index']} date={diagnostic['anchor_date']} "
            f"status={diagnostic['status']}"
        )
        if since_checkpoint >= args.checkpoint_every or processed_total == EXPECTED_TOTAL_TARGETS:
            save_checkpoints(paths, diagnostic_rows, attempt_rows, warning_rows, protocol_info)
            since_checkpoint = 0

    require(len(diagnostic_rows) == EXPECTED_TOTAL_TARGETS, "Rescue run did not process 574 targets")
    # Re-verify every protocol-referenced parent/evidence file after all fits
    # and before final rescue outputs are committed.
    protocol_after, protocol_info_after = load_and_validate_protocol(root, script_path)
    parent_protocol_after, parent_protocol_info_after = parent.load_and_verify_protocol(root)
    parent_locked_inputs_after = parent.verify_locked_inputs(root, parent_protocol_after)
    require(
        parent_protocol_info_after["protocol_sha256"] == parent_protocol_info["protocol_sha256"],
        "Parent protocol changed during rescue execution",
    )
    require(
        parent_locked_inputs_after == parent_locked_inputs,
        "One or more canonical parent locked inputs changed during rescue execution",
    )
    require(
        protocol_info_after["protocol_sha256"] == protocol_info["protocol_sha256"],
        "Active protocol changed during rescue execution",
    )
    require(
        protocol_info_after["script_sha256"] == protocol_info["script_sha256"],
        "Rescue script changed during execution",
    )

    finalize_outputs(
        root,
        output_dir,
        protocol_after,
        protocol_info_after,
        {
            "protocol": parent_protocol_info_after,
            "locked_inputs": parent_locked_inputs_after,
            "data_alignment_validated": True,
            "reverified_after_all_rescue_fits": True,
        },
        diagnostic_rows,
        attempt_rows,
        warning_rows,
        started_at_utc,
    )

    elapsed = time.perf_counter() - run_start
    final_diagnostics = pd.DataFrame(diagnostic_rows)
    status_counts = final_diagnostics["status"].astype(str).value_counts().to_dict()
    print("\n" + "=" * 122)
    print("08C-R RESCUE RUN COMPLETE")
    print("=" * 122)
    print(f"Targets processed: {len(final_diagnostics)}/574")
    for status, count in status_counts.items():
        print(f"{status:30s}: {int(count)}")
    print(f"Elapsed seconds: {elapsed:.3f}")
    print("Merge performed : NO")
    print("Metrics aggregated: NO")
    print("Parent changed  : NO")
    print("=" * 122)


if __name__ == "__main__":
    main()
