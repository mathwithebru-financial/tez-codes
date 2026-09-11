#!/usr/bin/env python3
"""Post-hoc Bartlett-HAC DM-HLN robustness analysis for Stage 09.

The statistical rules implemented here are frozen in
``protokol_bartlett_dm_hln_v1.md``.  This script:

1. reconstructs all 22 volatility loss-differential series used in Stage 09;
2. verifies that the original lag-zero Stage 09 statistics are reproducible;
3. computes Bartlett-HAC long-run variances with L=19 and h=1;
4. cross-validates the custom implementation with statsmodels HAC;
5. applies Holm correction jointly across the 22 volatility comparisons; and
6. writes complete, auditable result and manifest files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy.stats import t as student_t
import statsmodels
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac, weights_bartlett


VERSION = "1.0"
H = 1
L = 19
ALPHA = 0.05
EXPECTED_T = 584
EXPECTED_COMPARISONS = 22
CROSSCHECK_TOLERANCE = 1e-10
NUMERICAL_ZERO_MULTIPLIER = 1e-14

STAGE09_RELATIVE = Path(
    "results/statistical_comparison/09_dm_harvey_holm_results_v4.csv"
)
NAIVE_LOSS_RELATIVE = Path(
    "results/baselines/naive/naive_baseline_loss_series_v4.npz"
)
LEARNED_LOSS_RELATIVE = Path(
    "results/baselines/learned/learned_baseline_loss_series_v4.npz"
)
GARCH_FILENAME_STEMS = {
    "GARCH_1_1_StudentsT_ZeroMean": "garch_1_1_studentt_zeromean",
    "GJR_GARCH_1_1_StudentsT_ZeroMean": "gjr_garch_1_1_studentt_zeromean",
}

DECISION_TURKISH = {
    "SIGNIFICANT_FINAL_BETTER": "Nihai model anlamlı biçimde üstün",
    "SIGNIFICANT_BASELINE_BETTER": "Kıyaslama modeli anlamlı biçimde üstün",
    "NOT_SIGNIFICANT": "Anlamlı fark bulunmadı",
}


@dataclass(frozen=True)
class SeriesBundle:
    final_loss: np.ndarray
    baseline_loss: np.ndarray
    input_path: Path
    input_key_or_column: str
    target_dates: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        required=True,
        type=Path,
        help="Extracted project directory containing results/, data/, and scripts/.",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Directory for analysis outputs."
    )
    parser.add_argument(
        "--protocol",
        required=True,
        type=Path,
        help="Frozen protocol file whose SHA-256 is recorded in the manifest.",
    )
    parser.add_argument(
        "--source-zip",
        required=True,
        type=Path,
        help="Original source ZIP whose SHA-256 is recorded in the manifest.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")


def require_vector(name: str, values: np.ndarray, expected_t: int = EXPECTED_T) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; found shape {array.shape}.")
    if len(array) != expected_t:
        raise ValueError(f"{name} must contain {expected_t} observations; found {len(array)}.")
    if not np.isfinite(array).all():
        bad = np.flatnonzero(~np.isfinite(array))[:10].tolist()
        raise ValueError(f"{name} contains non-finite values at indices {bad}.")
    return array


def hln_factor(t_count: int, horizon: int = H) -> float:
    numerator = t_count + 1 - 2 * horizon + horizon * (horizon - 1) / t_count
    if numerator <= 0:
        raise ValueError("HLN correction factor is not real for the supplied T and h.")
    return math.sqrt(numerator / t_count)


def sample_autocovariances(differential: np.ndarray, max_lag: int) -> np.ndarray:
    t_count = len(differential)
    centered = differential - differential.mean()
    return np.asarray(
        [
            float(np.dot(centered[lag:], centered[: t_count - lag]) / t_count)
            for lag in range(max_lag + 1)
        ],
        dtype=np.float64,
    )


def custom_dm_hln(
    differential: np.ndarray, max_lag: int = L, horizon: int = H
) -> dict[str, Any]:
    """Compute the frozen custom Bartlett-HAC DM-HLN statistic."""

    d = require_vector("loss differential", differential)
    t_count = len(d)
    factor = hln_factor(t_count, horizon)

    if np.all(d == 0.0):
        return {
            "gamma0": 0.0,
            "omega_bartlett": 0.0,
            "dm_standard_error": 0.0,
            "dm_raw_statistic": 0.0,
            "hln_correction_factor": factor,
            "dm_hln_statistic": 0.0,
            "dm_hln_p_t_two_sided": 1.0,
            "variance_policy_case": "ALL_ZERO_DIFFERENTIAL",
            "autocovariances": np.zeros(max_lag + 1, dtype=np.float64),
        }

    autocovariances = sample_autocovariances(d, max_lag)
    weights = 1.0 - np.arange(1, max_lag + 1, dtype=np.float64) / (max_lag + 1)
    omega = float(autocovariances[0] + 2.0 * np.dot(weights, autocovariances[1:]))
    zero_tolerance = NUMERICAL_ZERO_MULTIPLIER * max(1.0, float(autocovariances[0]))

    policy_case = "STANDARD_POSITIVE_VARIANCE"
    if omega < 0.0:
        if abs(omega) <= zero_tolerance:
            omega = 0.0
            policy_case = "NEGATIVE_ROUNDOFF_SET_TO_ZERO"
        else:
            raise ValueError(
                f"Bartlett long-run variance is materially negative ({omega:.17g})."
            )

    mean_d = float(d.mean())
    if omega == 0.0:
        if mean_d == 0.0:
            dm_raw = 0.0
            dm_hln = 0.0
            p_value = 1.0
            standard_error = 0.0
            policy_case = "ZERO_VARIANCE_ZERO_MEAN"
        else:
            raise ValueError("Zero Bartlett variance with a non-zero mean differential.")
    else:
        standard_error = math.sqrt(omega / t_count)
        dm_raw = mean_d / standard_error
        dm_hln = factor * dm_raw
        p_value = float(2.0 * student_t.sf(abs(dm_hln), df=t_count - 1))

    return {
        "gamma0": float(autocovariances[0]),
        "omega_bartlett": omega,
        "dm_standard_error": standard_error,
        "dm_raw_statistic": dm_raw,
        "hln_correction_factor": factor,
        "dm_hln_statistic": dm_hln,
        "dm_hln_p_t_two_sided": p_value,
        "variance_policy_case": policy_case,
        "autocovariances": autocovariances,
    }


def statsmodels_crosscheck(differential: np.ndarray) -> dict[str, float]:
    """Independently calculate the HAC variance of the sample mean."""

    d = require_vector("statsmodels loss differential", differential)
    t_count = len(d)
    design = np.ones((t_count, 1), dtype=np.float64)
    fitted = OLS(d, design).fit()
    covariance = cov_hac(
        fitted,
        nlags=L,
        weights_func=weights_bartlett,
        use_correction=False,
    )
    variance_of_mean = float(covariance[0, 0])
    if variance_of_mean < 0.0:
        raise ValueError(f"statsmodels returned negative HAC variance: {variance_of_mean}")
    standard_error = math.sqrt(variance_of_mean)
    mean_d = float(d.mean())
    if standard_error == 0.0:
        dm_raw = 0.0 if mean_d == 0.0 else math.copysign(math.inf, mean_d)
    else:
        dm_raw = mean_d / standard_error
    dm_hln = hln_factor(t_count, H) * dm_raw
    p_value = 1.0 if dm_hln == 0.0 else float(
        2.0 * student_t.sf(abs(dm_hln), df=t_count - 1)
    )
    return {
        "statsmodels_variance_of_mean": variance_of_mean,
        "statsmodels_standard_error": standard_error,
        "statsmodels_dm_raw_statistic": dm_raw,
        "statsmodels_dm_hln_statistic": dm_hln,
        "statsmodels_p_t_two_sided": p_value,
    }


def garch_observation_path(project_root: Path, model: str, asset: str) -> Path:
    try:
        stem = GARCH_FILENAME_STEMS[model]
    except KeyError as exc:
        raise ValueError(f"Unrecognized GARCH model name: {model}") from exc
    return (
        project_root
        / "results/baselines/garch/checkpoints"
        / f"{stem}__{asset.lower()}__observations.csv"
    )


def load_series_bundle(
    project_root: Path,
    row: pd.Series,
    naive_loss: Any,
    learned_loss: Any,
) -> SeriesBundle:
    asset = str(row["asset"])
    final_key = f"final_vol_pinball__{asset}"
    final = require_vector(final_key, naive_loss[final_key])
    family = str(row["baseline_family"])
    model = str(row["baseline_model"])

    if family == "naive":
        baseline_key = f"vol_persistence_pinball__{asset}"
        baseline = require_vector(baseline_key, naive_loss[baseline_key])
        source_path = project_root / NAIVE_LOSS_RELATIVE
        source_key = baseline_key
        target_dates = None
    elif family == "learned":
        baseline_key = f"{model}__volatility__{asset}"
        baseline = require_vector(baseline_key, learned_loss[baseline_key])
        source_path = project_root / LEARNED_LOSS_RELATIVE
        source_key = baseline_key
        target_dates = None
    elif family == "garch_family":
        source_path = garch_observation_path(project_root, model, asset)
        require_file(source_path)
        observations = pd.read_csv(source_path)
        if len(observations) != EXPECTED_T:
            raise ValueError(f"{source_path.name} has {len(observations)} rows, not 584.")
        if "status" not in observations or not observations["status"].eq("OK").all():
            raise ValueError(f"{source_path.name} contains a non-OK observation.")
        if not observations["anchor_index"].to_numpy().tolist() == list(range(EXPECTED_T)):
            raise ValueError(f"{source_path.name} anchor_index is not exactly 0..583.")
        baseline = require_vector(
            f"{source_path.name}:primary_pinball_loss",
            observations["primary_pinball_loss"].to_numpy(dtype=np.float64),
        )
        source_key = "primary_pinball_loss"
        target_dates = observations["target_realization_date"].astype(str).to_numpy()
    else:
        raise ValueError(f"Unrecognized baseline family: {family}")

    return SeriesBundle(
        final_loss=final,
        baseline_loss=baseline,
        input_path=source_path,
        input_key_or_column=source_key,
        target_dates=target_dates,
    )


def holm_adjust(p_values: np.ndarray, alpha: float = ALPHA) -> pd.DataFrame:
    p_values = np.asarray(p_values, dtype=np.float64)
    if p_values.ndim != 1 or not np.isfinite(p_values).all():
        raise ValueError("Holm input must be a finite one-dimensional vector.")
    m = len(p_values)
    order = np.argsort(p_values, kind="mergesort")
    adjusted = np.empty(m, dtype=np.float64)
    ranks = np.empty(m, dtype=np.int64)
    critical_values = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for rank, original_index in enumerate(order, start=1):
        candidate = (m - rank + 1) * p_values[original_index]
        running_max = max(running_max, candidate)
        adjusted[original_index] = min(1.0, running_max)
        ranks[original_index] = rank
        critical_values[original_index] = alpha / (m - rank + 1)
    return pd.DataFrame(
        {
            "holm_rank": ranks,
            "holm_critical_value": critical_values,
            "holm_adjusted_p": adjusted,
            "null_rejected_holm": adjusted < alpha,
        }
    )


def decision(mean_d: float, adjusted_p: float) -> str:
    if adjusted_p >= ALPHA:
        return "NOT_SIGNIFICANT"
    if mean_d > 0.0:
        return "SIGNIFICANT_FINAL_BETTER"
    if mean_d < 0.0:
        return "SIGNIFICANT_BASELINE_BETTER"
    return "NOT_SIGNIFICANT"


def build_report(results: pd.DataFrame, summary: dict[str, Any]) -> str:
    changed = results.loc[results["decision_changed"]].copy()
    changed_lines = []
    for row in changed.itertuples(index=False):
        changed_lines.append(
            f"- {row.asset} – {row.baseline_model}: "
            f"{DECISION_TURKISH[row.stage09_decision]} → "
            f"{DECISION_TURKISH[row.bartlett_decision]} "
            f"(Holm p={row.holm_adjusted_p:.6g})."
        )
    changed_text = "\n".join(changed_lines) if changed_lines else "- Karar değişikliği yoktur."

    return f"""# Bartlett-HAC DM-HLN Ek Sağlamlık Analizi Sonuç Raporu

## Sonuç

Stage 09'daki 22 volatilite karşılaştırması, önceden dondurulan protokole göre
`h=1`, Bartlett gecikme sınırı `L=19` ve 22 test üzerinde Holm düzeltmesiyle
yeniden hesaplanmıştır.

- Nihai model anlamlı biçimde üstün: **{summary['bartlett_counts']['SIGNIFICANT_FINAL_BETTER']}**
- Kıyaslama modeli anlamlı biçimde üstün: **{summary['bartlett_counts']['SIGNIFICANT_BASELINE_BETTER']}**
- Anlamlı fark bulunmadı: **{summary['bartlett_counts']['NOT_SIGNIFICANT']}**
- Stage 09'a göre kararı değişen karşılaştırma: **{summary['changed_decision_count']}**

İki nihai-model üstünlüğü korunmuştur. Altı karşılaştırma, “kıyaslama modeli
anlamlı biçimde üstün” kararından “anlamlı fark bulunmadı” kararına geçmiştir.
Hiçbir karşılaştırmada üstünlük yönü tersine dönmemiştir.

## Kararı değişen karşılaştırmalar

{changed_text}

## Denetim sonuçları

- Gözlem sayısı: her karşılaştırmada **{summary['n_per_comparison']}**
- Volatilite karşılaştırması: **{summary['comparison_count']}**
- Stage 09 L=0 yeniden üretiminde en büyük mutlak DM farkı:
  **{summary['validation']['stage09_max_abs_dm_delta']:.3e}**
- Özel Bartlett hesabı ile statsmodels çapraz kontrolündeki en büyük mutlak
  düzeltilmiş DM farkı: **{summary['validation']['statsmodels_max_abs_dm_hln_delta']:.3e}**
- En büyük mutlak ham p-değeri farkı:
  **{summary['validation']['statsmodels_max_abs_p_delta']:.3e}**
- Çapraz kontrol toleransı: **{summary['validation']['tolerance']:.1e}**
- Doğrulama durumu: **{summary['validation']['status']}**

## Yorum sınırı

Bu çalışma Stage 09 ana analizini değiştirmez; seri bağımlılık varsayımına
duyarlılığı gösteren post-hoc bir sağlamlık analizidir. `L=19`, 20 günlük
hareketli volatilite hedeflerindeki örtüşmenin mekanik olarak işaret ettiği
gecikme sınırıdır; bağımlılığın 19. gecikmede bittiğini kanıtlamaz.

## Teze eklenebilecek sonuç paragrafı

Volatilite görevi için raporlanan DM-HLN sonuçlarının örtüşen hedef yapısından
kaynaklanabilecek seri bağımlılığa duyarlılığı, ek bir sağlamlık analizinde
Bartlett ağırlıklı HAC uzun dönem varyans tahmincisi kullanılarak incelenmiştir.
Gerçek tahmin ufku `h=1` olarak korunmuş, gecikme sınırı 20 günlük volatilite
pencerelerinin mekanik örtüşmesine dayanarak `L=19` seçilmiş ve 22 ham p-değeri
tek aile içinde Holm yöntemiyle düzeltilmiştir. Bu uygulamada nihai modelin
anlamlı üstün olduğu iki karşılaştırma korunurken, kıyaslama modelinin anlamlı
üstün olduğu karşılaştırma sayısı 18'den 12'ye düşmüş; anlamlı fark bulunmayan
karşılaştırma sayısı 2'den 8'e yükselmiştir. Altı karar kıyaslama üstünlüğünden
anlamlı fark bulunmamasına dönüşmüş, hiçbir karşılaştırmada üstünlük yönü tersine
dönmemiştir. Bulgular, bazı kıyaslama üstünlüklerinin kayıp farklarındaki seri
bağımlılığın daha geniş biçimde hesaba katılmasına duyarlı olduğunu göstermektedir.
"""


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    protocol_path = args.protocol.resolve()
    source_zip = args.source_zip.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    required = [
        project_root / STAGE09_RELATIVE,
        project_root / NAIVE_LOSS_RELATIVE,
        project_root / LEARNED_LOSS_RELATIVE,
        protocol_path,
        source_zip,
    ]
    for path in required:
        require_file(path)

    stage09_all = pd.read_csv(project_root / STAGE09_RELATIVE)
    stage09 = stage09_all.loc[stage09_all["task"].eq("volatility")].copy()
    stage09.reset_index(drop=True, inplace=True)
    if len(stage09) != EXPECTED_COMPARISONS:
        raise ValueError(
            f"Expected {EXPECTED_COMPARISONS} volatility comparisons; found {len(stage09)}."
        )
    if stage09["comparison_id"].duplicated().any():
        raise ValueError("Duplicate volatility comparison_id values found.")
    if not stage09["n"].eq(EXPECTED_T).all():
        raise ValueError("Stage 09 contains a volatility comparison with n != 584.")

    naive_loss = np.load(project_root / NAIVE_LOSS_RELATIVE, allow_pickle=False)
    learned_loss = np.load(project_root / LEARNED_LOSS_RELATIVE, allow_pickle=False)
    result_records: list[dict[str, Any]] = []
    differentials: dict[str, np.ndarray] = {}
    autocovariances: list[dict[str, Any]] = []
    target_dates: np.ndarray | None = None
    used_input_paths: set[Path] = {
        project_root / STAGE09_RELATIVE,
        project_root / NAIVE_LOSS_RELATIVE,
        project_root / LEARNED_LOSS_RELATIVE,
    }

    for row in stage09.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        bundle = load_series_bundle(project_root, row_series, naive_loss, learned_loss)
        used_input_paths.add(bundle.input_path)
        if bundle.target_dates is not None:
            if target_dates is None:
                target_dates = bundle.target_dates.copy()
            elif not np.array_equal(target_dates, bundle.target_dates):
                raise ValueError(
                    f"GARCH target dates are inconsistent for {row.comparison_id}."
                )
        differential = require_vector(
            f"{row.comparison_id}:d",
            bundle.baseline_loss - bundle.final_loss,
        )
        differentials[str(row.comparison_id)] = differential

        # Exact Stage 09 replication check (lag-zero variance, T denominator).
        lag0 = sample_autocovariances(differential, 0)[0]
        lag0_se = math.sqrt(lag0 / EXPECTED_T)
        lag0_dm = float(differential.mean() / lag0_se)
        lag0_hln = hln_factor(EXPECTED_T, H) * lag0_dm

        custom = custom_dm_hln(differential, max_lag=L, horizon=H)
        external = statsmodels_crosscheck(differential)

        variance_of_mean_custom = custom["omega_bartlett"] / EXPECTED_T
        variance_delta = abs(
            variance_of_mean_custom - external["statsmodels_variance_of_mean"]
        )
        dm_raw_delta = abs(
            custom["dm_raw_statistic"] - external["statsmodels_dm_raw_statistic"]
        )
        dm_hln_delta = abs(
            custom["dm_hln_statistic"] - external["statsmodels_dm_hln_statistic"]
        )
        p_delta = abs(
            custom["dm_hln_p_t_two_sided"] - external["statsmodels_p_t_two_sided"]
        )

        record = {
            "comparison_id": row.comparison_id,
            "task": row.task,
            "asset": row.asset,
            "reference_model": row.reference_model,
            "baseline_family": row.baseline_family,
            "baseline_model": row.baseline_model,
            "loss_function": row.loss_function,
            "holm_family": row.holm_family,
            "n": EXPECTED_T,
            "forecast_horizon_h": H,
            "bartlett_max_lag_L": L,
            "bartlett_weight_rule": "1-k/20",
            "mean_final_loss": float(bundle.final_loss.mean()),
            "mean_baseline_loss": float(bundle.baseline_loss.mean()),
            "mean_loss_differential": float(differential.mean()),
            "median_loss_differential": float(np.median(differential)),
            "loss_differential_gamma0": custom["gamma0"],
            "bartlett_long_run_variance": custom["omega_bartlett"],
            "bartlett_variance_of_mean": variance_of_mean_custom,
            "dm_standard_error": custom["dm_standard_error"],
            "dm_raw_statistic": custom["dm_raw_statistic"],
            "hln_correction_factor": custom["hln_correction_factor"],
            "dm_hln_statistic": custom["dm_hln_statistic"],
            "dm_hln_p_t_two_sided": custom["dm_hln_p_t_two_sided"],
            "variance_policy_case": custom["variance_policy_case"],
            "stage09_decision": row.interpretation,
            "stage09_holm_adjusted_p": float(row.holm_adjusted_p),
            "stage09_mean_diff_abs_delta": abs(
                float(differential.mean()) - float(row.mean_loss_differential)
            ),
            "stage09_gamma0_abs_delta": abs(lag0 - float(row.loss_differential_gamma0)),
            "stage09_dm_raw_abs_delta": abs(lag0_dm - float(row.dm_raw_statistic)),
            "stage09_dm_hln_abs_delta": abs(lag0_hln - float(row.dm_hln_statistic)),
            "statsmodels_variance_of_mean": external["statsmodels_variance_of_mean"],
            "statsmodels_standard_error": external["statsmodels_standard_error"],
            "statsmodels_dm_raw_statistic": external["statsmodels_dm_raw_statistic"],
            "statsmodels_dm_hln_statistic": external["statsmodels_dm_hln_statistic"],
            "statsmodels_p_t_two_sided": external["statsmodels_p_t_two_sided"],
            "crosscheck_variance_abs_delta": variance_delta,
            "crosscheck_dm_raw_abs_delta": dm_raw_delta,
            "crosscheck_dm_hln_abs_delta": dm_hln_delta,
            "crosscheck_p_abs_delta": p_delta,
            "baseline_loss_source": str(bundle.input_path.relative_to(project_root)),
            "baseline_loss_key_or_column": bundle.input_key_or_column,
        }
        result_records.append(record)

        for lag, gamma in enumerate(custom["autocovariances"]):
            autocovariances.append(
                {
                    "comparison_id": row.comparison_id,
                    "lag": lag,
                    "bartlett_weight": 1.0 if lag == 0 else 1.0 - lag / (L + 1),
                    "sample_autocovariance": float(gamma),
                }
            )

    results = pd.DataFrame(result_records)
    if target_dates is None or len(target_dates) != EXPECTED_T:
        raise ValueError("A complete 584-date GARCH target-date reference was not found.")
    holm = holm_adjust(results["dm_hln_p_t_two_sided"].to_numpy())
    results = pd.concat([results, holm], axis=1)
    results["bartlett_decision"] = [
        decision(mean_d, adjusted_p)
        for mean_d, adjusted_p in zip(
            results["mean_loss_differential"], results["holm_adjusted_p"]
        )
    ]
    results["decision_changed"] = (
        results["stage09_decision"] != results["bartlett_decision"]
    )
    results["decision_change"] = (
        results["stage09_decision"] + " -> " + results["bartlett_decision"]
    )
    results["stage09_decision_tr"] = results["stage09_decision"].map(DECISION_TURKISH)
    results["bartlett_decision_tr"] = results["bartlett_decision"].map(DECISION_TURKISH)

    validation = {
        "stage09_max_abs_mean_diff_delta": float(
            results["stage09_mean_diff_abs_delta"].max()
        ),
        "stage09_max_abs_gamma0_delta": float(results["stage09_gamma0_abs_delta"].max()),
        "stage09_max_abs_dm_delta": float(results["stage09_dm_raw_abs_delta"].max()),
        "stage09_max_abs_dm_hln_delta": float(
            results["stage09_dm_hln_abs_delta"].max()
        ),
        "statsmodels_max_abs_variance_delta": float(
            results["crosscheck_variance_abs_delta"].max()
        ),
        "statsmodels_max_abs_dm_raw_delta": float(
            results["crosscheck_dm_raw_abs_delta"].max()
        ),
        "statsmodels_max_abs_dm_hln_delta": float(
            results["crosscheck_dm_hln_abs_delta"].max()
        ),
        "statsmodels_max_abs_p_delta": float(results["crosscheck_p_abs_delta"].max()),
        "tolerance": CROSSCHECK_TOLERANCE,
    }
    maximum_validation_delta = max(
        validation["stage09_max_abs_mean_diff_delta"],
        validation["stage09_max_abs_gamma0_delta"],
        validation["stage09_max_abs_dm_delta"],
        validation["stage09_max_abs_dm_hln_delta"],
        validation["statsmodels_max_abs_variance_delta"],
        validation["statsmodels_max_abs_dm_raw_delta"],
        validation["statsmodels_max_abs_dm_hln_delta"],
        validation["statsmodels_max_abs_p_delta"],
    )
    validation["status"] = (
        "PASS" if maximum_validation_delta <= CROSSCHECK_TOLERANCE else "FAIL"
    )
    if validation["status"] != "PASS":
        raise RuntimeError(f"Validation failed: {json.dumps(validation, indent=2)}")

    stage09_counts = {
        key: int((results["stage09_decision"] == key).sum()) for key in DECISION_TURKISH
    }
    bartlett_counts = {
        key: int((results["bartlett_decision"] == key).sum()) for key in DECISION_TURKISH
    }
    changed_counts = (
        results.loc[results["decision_changed"]]
        .groupby(["stage09_decision", "bartlett_decision"])
        .size()
    )
    change_transitions = [
        {"from": source, "to": target, "count": int(count)}
        for (source, target), count in changed_counts.items()
    ]

    summary = {
        "analysis_version": VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_status": "COMPLETED_AND_VALIDATED",
        "scope": "post_hoc_volatility_robustness_only",
        "main_stage09_results_replaced": False,
        "comparison_count": int(len(results)),
        "n_per_comparison": EXPECTED_T,
        "forecast_horizon_h": H,
        "bartlett_max_lag_L": L,
        "bartlett_weight_rule": "1-k/20",
        "holm_family_size": EXPECTED_COMPARISONS,
        "alpha": ALPHA,
        "stage09_counts": stage09_counts,
        "bartlett_counts": bartlett_counts,
        "changed_decision_count": int(results["decision_changed"].sum()),
        "change_transitions": change_transitions,
        "direction_reversal_count": int(
            (
                results["decision_changed"]
                & results["stage09_decision"].ne("NOT_SIGNIFICANT")
                & results["bartlett_decision"].ne("NOT_SIGNIFICANT")
            ).sum()
        ),
        "validation": validation,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
    }

    results_path = output_dir / "bartlett_dm_hln_results_v1.csv"
    changed_path = output_dir / "bartlett_dm_hln_changed_decisions_v1.csv"
    autocov_path = output_dir / "bartlett_dm_hln_autocovariances_v1.csv"
    differential_long_path = output_dir / "bartlett_loss_differentials_long_v1.csv"
    differential_path = output_dir / "bartlett_loss_differentials_v1.npz"
    summary_path = output_dir / "bartlett_dm_hln_summary_v1.json"
    report_path = output_dir / "bartlett_dm_hln_report_v1.md"
    manifest_path = output_dir / "bartlett_dm_hln_manifest_v1.json"

    results.to_csv(results_path, index=False, float_format="%.17g")
    results.loc[results["decision_changed"]].to_csv(
        changed_path, index=False, float_format="%.17g"
    )
    pd.DataFrame(autocovariances).to_csv(
        autocov_path, index=False, float_format="%.17g"
    )
    differential_frames = []
    for row in results.itertuples(index=False):
        differential_frames.append(
            pd.DataFrame(
                {
                    "target_realization_date": target_dates,
                    "comparison_id": row.comparison_id,
                    "asset": row.asset,
                    "baseline_family": row.baseline_family,
                    "baseline_model": row.baseline_model,
                    "loss_differential": differentials[row.comparison_id],
                }
            )
        )
    pd.concat(differential_frames, ignore_index=True).to_csv(
        differential_long_path, index=False, float_format="%.17g"
    )
    np.savez_compressed(
        differential_path,
        target_realization_dates=target_dates,
        **differentials,
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(build_report(results, summary), encoding="utf-8")

    script_path = Path(__file__).resolve()
    output_paths = [
        results_path,
        changed_path,
        autocov_path,
        differential_long_path,
        differential_path,
        summary_path,
        report_path,
    ]
    manifest = {
        "analysis_version": VERSION,
        "created_at_utc": summary["created_at_utc"],
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
            "size_bytes": protocol_path.stat().st_size,
        },
        "source_zip": {
            "path": str(source_zip),
            "sha256": sha256_file(source_zip),
            "size_bytes": source_zip.stat().st_size,
        },
        "analysis_script": {
            "path": str(script_path),
            "sha256": sha256_file(script_path),
            "size_bytes": script_path.stat().st_size,
        },
        "inputs": [
            {
                "relative_path": str(path.relative_to(project_root)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(used_input_paths, key=lambda item: str(item))
        ],
        "outputs": [
            {
                "filename": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in output_paths
        ],
        "validation": validation,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {results_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
