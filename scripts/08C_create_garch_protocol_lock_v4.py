from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform

import arch
import numpy as np
import pandas as pd
import scipy
from arch.univariate import StudentsT
from scipy.optimize import brentq


# ==========================================================
# 1. PROJE YOLLARI
# ==========================================================

BASE = Path("/content/drive/MyDrive/tez_transformer_v4_repro")
PROCESSED = BASE / "data" / "processed"
SEQ = BASE / "data" / "sequences" / "baseline" / "lb10"
CONFIG = BASE / "config"

LOCK_PATH = CONFIG / "08C_garch_protocol_lock_v4.json"
SHA_PATH = CONFIG / "08C_garch_protocol_lock_v4.sha256"

if not BASE.exists():
    raise FileNotFoundError(
        "Proje klasörü bulunamadı. Google Drive bağlantısını kontrol et:\n"
        f"{BASE}"
    )

CONFIG.mkdir(parents=True, exist_ok=True)

if LOCK_PATH.exists() or SHA_PATH.exists():
    raise FileExistsError(
        "08C protokol kilidi veya SHA dosyası zaten mevcut. "
        "Üzerine yazma yasak:\n"
        f"{LOCK_PATH}\n{SHA_PATH}"
    )


# ==========================================================
# 2. RESMÎ VE DOĞRULAMA GİRDİLERİ
# ==========================================================

OFFICIAL_INPUTS = {
    "features_baseline": PROCESSED / "features_baseline.csv",
    "anchor_dates_test": SEQ / "anchor_dates_test.npy",
    "target_realization_dates_test": SEQ / "target_realization_dates_test.npy",
    "y_test_raw": SEQ / "y_test_raw.npy",
}

VERIFICATION_INPUTS = {
    "prices_clean": PROCESSED / "prices_clean.csv",
    "targets_all": PROCESSED / "targets_all.csv",
    "target_realization_dates": PROCESSED / "target_realization_dates.csv",
    "split_meta": PROCESSED / "split_meta_v4.json",
    "meta_v4": PROCESSED / "meta_v4.json",
    "sequence_meta": SEQ / "sequence_meta.json",
}

EXPECTED_PACKAGE_VERSIONS = {
    "arch": "8.0.0",
    "pandas": "2.2.2",
    "numpy": "2.0.2",
    "scipy": "1.16.3",
}

EXPECTED_SHA256 = {
    "features_baseline": "0ebfaae1f5891007507825f380a3512c7743d9c5d3d6176ee559b9ea34c14102",
    "anchor_dates_test": "40194047c9b7fe62adc0966f2c144e1be1e9c85f832d96ffe0a12624d34aab63",
    "target_realization_dates_test": "0100eea598141cf5990b4d7ba5a8071236846ec7a1d1196c078b6c3af1fcf441",
    "y_test_raw": "a37034c833365b89302dbf0e3d57c29dc2eaf6ada91bfd1fae16fd56f112adb8",
    "prices_clean": "5c417779e460994c5068ca1621b4292f388b958a3c93db0c9de67d6109adef95",
    "targets_all": "e4c3bbed7acd13ffe32628503a870824f9717ce08bb3c631f26578fdb4b28269",
    "target_realization_dates": "65a7fc7838f8e547c6640064fdd0d753be76f74ce363f74fe549654ace741ed5",
    "split_meta": "ab9fb042e3656400bdf5d8c1c920f57b2bbd1d25199dd20efe3eb7116861ece6",
    "meta_v4": "a80c21c16b1c57dc90fa734d271b0360f391bfce3b45370881771104a5028a27",
    "sequence_meta": "78fa849524fa0e352c6b3c5c560b5b8658b6a85678afb38ec9c8c34b49d3f7e9",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_file_record(name: str, path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Eksik dosya: {name}\n{path}")

    actual_sha = sha256_file(path)
    expected_sha = EXPECTED_SHA256[name]

    if actual_sha != expected_sha:
        raise RuntimeError(
            f"SHA-256 uyuşmazlığı: {name}\n"
            f"Beklenen: {expected_sha}\n"
            f"Gerçek  : {actual_sha}\n"
            f"Dosya   : {path}"
        )

    return {
        "relative_path": str(path.relative_to(BASE)),
        "size_bytes": int(path.stat().st_size),
        "sha256": actual_sha,
    }


official_file_records = {
    name: make_file_record(name, path)
    for name, path in OFFICIAL_INPUTS.items()
}

verification_file_records = {
    name: make_file_record(name, path)
    for name, path in VERIFICATION_INPUTS.items()
}


# ==========================================================
# 3. EXACT VERİ, TARİH VE HEDEF PREFLIGHT
# ==========================================================

features = pd.read_csv(
    OFFICIAL_INPUTS["features_baseline"],
    index_col=0,
    parse_dates=True,
)

prices = pd.read_csv(
    VERIFICATION_INPUTS["prices_clean"],
    index_col=0,
    parse_dates=True,
)

targets = pd.read_csv(
    VERIFICATION_INPUTS["targets_all"],
    index_col=0,
    parse_dates=True,
)

target_dates_csv = pd.read_csv(
    VERIFICATION_INPUTS["target_realization_dates"],
    index_col=0,
    parse_dates=True,
)

split_meta = json.loads(
    VERIFICATION_INPUTS["split_meta"].read_text(encoding="utf-8")
)

sequence_meta = json.loads(
    VERIFICATION_INPUTS["sequence_meta"].read_text(encoding="utf-8")
)

anchor_dates = pd.DatetimeIndex(
    pd.to_datetime(
        np.load(
            OFFICIAL_INPUTS["anchor_dates_test"],
            allow_pickle=False,
        ).astype(str)
    )
)

realization_dates_npy = pd.DatetimeIndex(
    pd.to_datetime(
        np.load(
            OFFICIAL_INPUTS["target_realization_dates_test"],
            allow_pickle=False,
        ).astype(str)
    )
)

y_test_raw = np.load(
    OFFICIAL_INPUTS["y_test_raw"],
    allow_pickle=False,
)

EXPECTED_FEATURE_COLUMNS = [
    "BIST100_LogRet",
    "BIST100_Vol20",
    "USDTRY_LogRet",
    "USDTRY_Vol20",
    "EURTRY_LogRet",
    "EURTRY_Vol20",
    "GOLD_LogRet",
    "GOLD_Vol20",
]

EXPECTED_TARGET_COLUMNS = [
    "BIST100_NextRet",
    "USDTRY_NextRet",
    "EURTRY_NextRet",
    "GOLD_NextRet",
    "BIST100_NextVol",
    "USDTRY_NextVol",
    "EURTRY_NextVol",
    "GOLD_NextVol",
]

LOGRET_COLUMNS = [
    "BIST100_LogRet",
    "USDTRY_LogRet",
    "EURTRY_LogRet",
    "GOLD_LogRet",
]

ASSETS = ["BIST100", "USDTRY", "EURTRY", "GOLD"]

if list(features.columns) != EXPECTED_FEATURE_COLUMNS:
    raise RuntimeError(
        "features_baseline sütun sırası beklenen yapıyla eşleşmiyor."
    )

if list(targets.columns) != EXPECTED_TARGET_COLUMNS:
    raise RuntimeError(
        "targets_all sütun sırası beklenen yapıyla eşleşmiyor."
    )

if features.shape != (3891, 8):
    raise RuntimeError(f"features_baseline shape yanlış: {features.shape}")

if targets.shape != (3891, 8):
    raise RuntimeError(f"targets_all shape yanlış: {targets.shape}")

if features.index.min() != pd.Timestamp("2010-02-01"):
    raise RuntimeError("Ortak geçerli indeks başlangıcı 2010-02-01 değil.")

if features.index.max() != pd.Timestamp("2024-12-30"):
    raise RuntimeError("Ortak geçerli indeks bitişi 2024-12-30 değil.")

if not features.index.equals(targets.index):
    raise RuntimeError("features_baseline ve targets_all indeksleri aynı değil.")

if len(anchor_dates) != 584 or not anchor_dates.is_unique:
    raise RuntimeError("Test anchor envanteri 584 unique tarihten oluşmuyor.")

if not anchor_dates.is_monotonic_increasing:
    raise RuntimeError("Test anchor tarihleri kronolojik değil.")

if anchor_dates[0] != pd.Timestamp("2022-10-05"):
    raise RuntimeError("İlk test anchor tarihi 2022-10-05 değil.")

if anchor_dates[-1] != pd.Timestamp("2024-12-30"):
    raise RuntimeError("Son test anchor tarihi 2024-12-30 değil.")

if len(realization_dates_npy) != 584:
    raise RuntimeError("Test realization tarih sayısı 584 değil.")

if realization_dates_npy[0] != pd.Timestamp("2022-10-06"):
    raise RuntimeError("İlk realization tarihi 2022-10-06 değil.")

if realization_dates_npy[-1] != pd.Timestamp("2024-12-31"):
    raise RuntimeError("Son realization tarihi 2024-12-31 değil.")

if not np.all(realization_dates_npy.values > anchor_dates.values):
    raise RuntimeError("Bazı realization tarihleri anchor sonrasında değil.")

if y_test_raw.shape != (584, 8):
    raise RuntimeError(f"y_test_raw shape yanlış: {y_test_raw.shape}")

if y_test_raw.dtype != np.float32:
    raise RuntimeError(f"y_test_raw dtype float32 değil: {y_test_raw.dtype}")

# Dtype-duyarlı hedef eşitliği
csv_targets_float64 = targets.loc[anchor_dates].to_numpy(dtype=np.float64)
csv_targets_official_dtype = csv_targets_float64.astype(y_test_raw.dtype)

if not np.array_equal(y_test_raw, csv_targets_official_dtype):
    raise RuntimeError(
        "y_test_raw, targets_all değerlerinin resmî NPY dtype'ına "
        "dönüştürülmüş hâliyle element bazında aynı değil."
    )

float64_target_max_abs_diff = float(
    np.max(
        np.abs(
            y_test_raw.astype(np.float64) - csv_targets_float64
        )
    )
)

# Realization CSV == NPY
if target_dates_csv.shape[1] != 1:
    raise RuntimeError("target_realization_dates.csv tek sütun değil.")

realization_dates_csv = pd.DatetimeIndex(
    pd.to_datetime(
        target_dates_csv.loc[anchor_dates].iloc[:, 0].to_numpy()
    )
)

if not np.array_equal(
    realization_dates_csv.values,
    realization_dates_npy.values,
):
    raise RuntimeError("CSV ve NPY realization tarihleri aynı değil.")

# LogRet bağımsız yeniden hesaplama
logret_max_abs_diff_by_asset = {}

for asset, feature_column in zip(ASSETS, LOGRET_COLUMNS):
    recomputed = np.log(prices[asset] / prices[asset].shift(1))
    common_index = features.index.intersection(recomputed.dropna().index)

    stored = features.loc[common_index, feature_column].to_numpy(dtype=np.float64)
    rebuilt = recomputed.loc[common_index].to_numpy(dtype=np.float64)

    max_diff = float(np.max(np.abs(stored - rebuilt)))
    logret_max_abs_diff_by_asset[asset] = max_diff

    if max_diff >= 1e-12:
        raise RuntimeError(
            f"{asset} LogRet yeniden hesaplama farkı yüksek: {max_diff}"
        )

first_anchor_position = int(features.index.get_loc(anchor_dates[0]))
last_anchor_position = int(features.index.get_loc(anchor_dates[-1]))

first_fit_observation_count = first_anchor_position + 1
last_fit_observation_count = last_anchor_position + 1

if first_anchor_position != 3307:
    raise RuntimeError(
        f"İlk test anchor pozisyonu 3307 değil: {first_anchor_position}"
    )

if first_fit_observation_count != 3308:
    raise RuntimeError(
        f"İlk expanding fit gözlem sayısı 3308 değil: "
        f"{first_fit_observation_count}"
    )

if last_fit_observation_count != 3891:
    raise RuntimeError(
        f"Son expanding fit gözlem sayısı 3891 değil: "
        f"{last_fit_observation_count}"
    )

# Split meta exact ana kontroller
if split_meta["test"]["start_idx"] != 3307:
    raise RuntimeError("split_meta test start_idx 3307 değil.")

if split_meta["test"]["n_rows"] != 584:
    raise RuntimeError("split_meta test n_rows 584 değil.")

if split_meta["target_sets_disjoint"] is not True:
    raise RuntimeError("split_meta target_sets_disjoint True değil.")


# ==========================================================
# 4. EXACT CDF-KÖK SENTETİK PREFLIGHT
#    Resmî bracket politikasıyla yeniden çalıştırılır.
# ==========================================================

dist = StudentsT()
nu_test = 8.0
sigma_test = 0.012
c_test = 0.0015


def standardized_t_cdf_scalar(z: float, nu_value: float) -> float:
    value = dist.cdf(
        np.asarray([z], dtype=np.float64),
        [float(nu_value)],
    )
    return float(np.asarray(value).reshape(-1)[0])


def predictive_cdf_test(x: float) -> float:
    return standardized_t_cdf_scalar(x / sigma_test, nu_test)


def central_mass_test(q_value: float) -> float:
    return (
        predictive_cdf_test(c_test + q_value)
        - predictive_cdf_test(c_test - q_value)
    )


def root_function_test(q_value: float) -> float:
    return central_mass_test(q_value) - 0.5


upper = max(sigma_test, abs(c_test), 1e-12)
upper_doublings = 0

while root_function_test(upper) < 0.0:
    upper *= 2.0
    upper_doublings += 1

    if upper_doublings > 100:
        raise RuntimeError("Sentetik CDF-kök bracket bulunamadı.")

q_test = brentq(
    root_function_test,
    0.0,
    upper,
    xtol=1e-14,
    rtol=1e-12,
    maxiter=200,
)

central_mass_value = central_mass_test(q_test)
mass_error = abs(central_mass_value - 0.5)

q0_upper = max(sigma_test, 1e-12)
q0_upper_doublings = 0

while (
    standardized_t_cdf_scalar(q0_upper / sigma_test, nu_test)
    - standardized_t_cdf_scalar(-q0_upper / sigma_test, nu_test)
    - 0.5
) < 0.0:
    q0_upper *= 2.0
    q0_upper_doublings += 1

    if q0_upper_doublings > 100:
        raise RuntimeError("Sentetik symmetry bracket bulunamadı.")

q0_root = brentq(
    lambda x: (
        standardized_t_cdf_scalar(x / sigma_test, nu_test)
        - standardized_t_cdf_scalar(-x / sigma_test, nu_test)
        - 0.5
    ),
    0.0,
    q0_upper,
    xtol=1e-14,
    rtol=1e-12,
    maxiter=200,
)

q0_ppf = sigma_test * float(
    np.asarray(dist.ppf(0.75, [nu_test])).reshape(-1)[0]
)

symmetry_difference = abs(q0_root - q0_ppf)
student_t_median = float(
    np.asarray(dist.ppf(0.5, [nu_test])).reshape(-1)[0]
)
student_t_second_moment = float(dist.moment(2, [nu_test]))

if mass_error > 1e-10:
    raise RuntimeError(f"Sentetik central-mass hatası yüksek: {mass_error}")

if symmetry_difference > 1e-10:
    raise RuntimeError(
        f"Sentetik symmetry kontrol farkı yüksek: {symmetry_difference}"
    )

if abs(student_t_median) > 1e-12:
    raise RuntimeError("Standardized Student-t medyanı sıfıra yakın değil.")

if abs(student_t_second_moment - 1.0) > 1e-12:
    raise RuntimeError("Standardized Student-t ikinci momenti 1 değil.")


# ==========================================================
# 5. RESMÎ 08C PROTOKOLÜ
# ==========================================================

protocol = {
    "project_version": "v4_repro",
    "stage": "08C",
    "protocol_version": 1,
    "status": "LOCKED_BEFORE_08C_GARCH_FITS_PREDICTIONS_AND_METRICS",
    "locked_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": (
        "GARCH-family econometric volatility baselines evaluated on "
        "the locked NextVol targets and test forecast origins."
    ),
    "environment": {
        "python": platform.python_version(),
        "arch": arch.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "required_package_versions": EXPECTED_PACKAGE_VERSIONS,
    },
    "data_lock": {
        "official_execution_inputs": official_file_records,
        "independent_verification_inputs": verification_file_records,
        "official_return_source": {
            "file_key": "features_baseline",
            "columns": LOGRET_COLUMNS,
            "unit": "decimal log return",
        },
        "official_test_target_source": {
            "file_key": "y_test_raw",
            "shape": [584, 8],
            "dtype": str(y_test_raw.dtype),
            "volatility_target_slice": [4, 8],
        },
        "target_verification_rule": {
            "rule": (
                "y_test_raw must be element-wise exactly equal to "
                "targets_all at test anchors after casting CSV values "
                "to the official NPY dtype."
            ),
            "exact_array_equality_after_dtype_alignment": True,
            "diagnostic_float64_max_abs_diff": float64_target_max_abs_diff,
            "tolerance_replacement_used": False,
        },
        "logret_independent_reconstruction": {
            "source": "prices_clean.csv",
            "formula": "log(price[t] / price[t-1])",
            "max_abs_diff_by_asset": logret_max_abs_diff_by_asset,
            "maximum_allowed_diff": 1e-12,
        },
        "test_dates": {
            "anchor_count": 584,
            "anchor_start": str(anchor_dates[0].date()),
            "anchor_end": str(anchor_dates[-1].date()),
            "realization_count": 584,
            "realization_start": str(realization_dates_npy[0].date()),
            "realization_end": str(realization_dates_npy[-1].date()),
            "csv_npy_realization_exact_equality": True,
        },
        "common_valid_index": {
            "start": str(features.index.min().date()),
            "end": str(features.index.max().date()),
            "n_rows": int(len(features)),
            "definition": (
                "Common valid index produced from baseline features, "
                "full features, targets and target-realization dates."
            ),
        },
    },
    "official_models": [
        {
            "model_id": "GARCH_1_1_StudentsT_ZeroMean",
            "mean": "Zero",
            "vol": "GARCH",
            "p": 1,
            "o": 0,
            "q": 1,
            "power": 2.0,
            "dist": "StudentsT",
            "rescale": False,
        },
        {
            "model_id": "GJR_GARCH_1_1_StudentsT_ZeroMean",
            "mean": "Zero",
            "vol": "GARCH",
            "p": 1,
            "o": 1,
            "q": 1,
            "power": 2.0,
            "dist": "StudentsT",
            "rescale": False,
        },
    ],
    "excluded_models": ["EGARCH"],
    "excluded_model_rule": (
        "EGARCH and any other econometric specification cannot be "
        "added after official 08C GARCH test results are observed."
    ),
    "task": "volatility_only",
    "assets": ASSETS,
    "target_columns": EXPECTED_TARGET_COLUMNS[4:8],
    "target_indices": [4, 5, 6, 7],
    "target_definition": {
        "nextvol_rule": "NextVol[t] = Vol20[t+1]",
        "vol20_window": 20,
        "vol20_min_periods": 20,
        "vol20_ddof": 1,
        "annualization_factor": 252,
        "return_type": "decimal log return",
    },
    "expanding_window": {
        "enabled": True,
        "source_start_date": str(features.index.min().date()),
        "first_test_anchor": str(anchor_dates[0].date()),
        "first_anchor_zero_based_position": first_anchor_position,
        "first_fit_observation_count_including_anchor": first_fit_observation_count,
        "last_test_anchor": str(anchor_dates[-1].date()),
        "last_fit_observation_count_including_anchor": last_fit_observation_count,
        "refit_frequency": "every test anchor",
        "forecast_horizon": 1,
        "past_test_information_rule": (
            "A realized test-period return may enter later fits only "
            "after it has become observable."
        ),
        "future_information_forbidden": True,
    },
    "fit_scale": {
        "input_log_return_multiplier": 100.0,
        "arch_input_unit": "percentage return",
        "forecast_variance_unit": "percentage-return squared",
        "variance_to_decimal_divisor": 10000.0,
        "sigma_to_decimal_divisor": 100.0,
    },
    "warm_start_policy": {
        "warm_start_allowed": False,
        "starting_values": None,
        "rule": (
            "Every asset-model-anchor fit must be independently "
            "initialized using the arch default starting-value "
            "construction. Parameters from earlier anchors are not reused."
        ),
        "reason": (
            "Prevent cross-anchor starting-parameter dependence and keep "
            "each fit independently reconstructable."
        ),
    },
    "predictive_distribution": {
        "mean_decimal": 0.0,
        "representation": (
            "R[t+1] = sigma_decimal[t+1|t] * Z, where Z follows "
            "the arch standardized Student's t distribution and Var(Z)=1."
        ),
        "sigma_decimal": "sqrt(one-step arch variance forecast) / 100",
        "nu_source": "nu estimated in the corresponding expanding-window fit",
        "required_nu_rule": "finite and greater than 2",
    },
    "primary_forecast": {
        "name": "Deterministic conditional-median NextVol20 forecast",
        "metric_alignment": "Conditional median for PinballLoss tau=0.5",
        "known_returns": "r[t-18], ..., r[t] in decimal log-return units",
        "definitions": {
            "S": "sum of the 19 known returns",
            "Q": "sum of squares of the 19 known returns",
            "c": "S / 19",
            "C": "(Q - S**2 / 19) / 19",
        },
        "variance_identity": "V(R) = C + (R-c)**2 / 20",
        "root_equation": "F_R(c+q) - F_R(c-q) = 0.5 for q >= 0",
        "predictive_cdf": (
            "F_R(x) = StudentsT.cdf(x / sigma_decimal, [nu]) "
            "under the Zero-mean specification."
        ),
        "forecast_formula": "sqrt(252 * (C + q**2 / 20))",
        "root_method": "scipy.optimize.brentq",
        "root_settings": {
            "lower_bound": 0.0,
            "initial_upper_bound": "max(sigma_decimal, abs(c), 1e-12)",
            "upper_bound_update": "multiply by 2 until bracketed",
            "maximum_upper_bound_doublings": 100,
            "xtol": 1e-14,
            "rtol": 1e-12,
            "maxiter": 200,
            "maximum_allowed_mass_error": 1e-10,
        },
        "random_simulation_used": False,
    },
    "secondary_forecasts": {
        "expected_sample_variance_plugin": {
            "role": "secondary diagnostic only",
            "formula_1": "sqrt(252 * (C + (h_decimal + c**2) / 20))",
            "formula_2_equivalent": (
                "sqrt(252 * (Q + h_decimal - "
                "(S**2 + h_decimal)/20) / 19)"
            ),
        },
        "direct_conditional_sigma": {
            "role": "secondary diagnostic only",
            "formula": "sqrt(h_decimal * 252)",
            "warning": "Not the official target-aligned primary prediction",
        },
    },
    "numerical_safety": {
        "negative_variance_tolerance": 1e-14,
        "negative_value_rule": (
            "A theoretically non-negative value below -1e-14 fails. "
            "Values in [-1e-14, 0) may be set to zero only as "
            "floating-point rounding correction."
        ),
        "blind_max_zero_for_large_negative_values": False,
    },
    "persistence_diagnostics": {
        "role": "diagnostic only; not a test-based exclusion rule",
        "GARCH_1_1": "alpha[1] + beta[1]",
        "GJR_GARCH_1_1": (
            "alpha[1] + 0.5 * gamma[1] + beta[1] under a symmetric "
            "standardized innovation distribution"
        ),
        "high_persistence_rule": (
            "High or near-integrated persistence is recorded and reported; "
            "the fit is not silently removed or replaced solely for this reason."
        ),
    },
    "fit_and_retry_policy": {
        "maximum_attempts": 2,
        "primary_attempt": {
            "starting_values": None,
            "backcast": None,
            "cov_type": "robust",
            "update_freq": 0,
            "disp": "off",
            "show_warning": False,
            "tol": 1e-8,
            "options": {"maxiter": 2000},
        },
        "retry_attempt": {
            "allowed": True,
            "reason": "numerical convergence only",
            "same_model": True,
            "same_distribution": True,
            "same_data": True,
            "same_scale": True,
            "starting_values": None,
            "backcast": None,
            "cov_type": "robust",
            "update_freq": 0,
            "disp": "off",
            "show_warning": False,
            "tol": 1e-8,
            "options": {"maxiter": 5000},
        },
        "retry_trigger": (
            "A second fit attempt is allowed only after a failed numerical fit "
            "or invalid fitted output, never because of forecast performance. "
            "A CDF-root failure after an otherwise valid fit is recorded as an "
            "unresolved observation-level failure and does not trigger model switching."
        ),
        "silent_fallbacks": {
            "persistence": False,
            "normal_distribution": False,
            "parameter_imputation": False,
            "prediction_imputation": False,
        },
        "unresolved_failure_rule": (
            "Any unresolved fit, parameter, variance, root or prediction failure "
            "must be stored explicitly and prevents a clean 08C closure until "
            "resolved or formally reported as unresolved."
        ),
    },
    "fit_success_criteria": {
        "all_required": True,
        "criteria": [
            "result.convergence_flag == 0",
            "bool(result.optimization_result.success) is True",
            "all estimated parameters are finite",
            "estimated nu is finite and greater than 2",
            "one-step conditional variance is finite and greater than 0",
            "conditional sigma is finite and greater than 0",
            "CDF root is successfully bracketed and solved",
            "central probability-mass error is <= 1e-10",
            "primary prediction is finite",
            "secondary predictions are finite",
            "no silent fallback or imputation is used",
        ],
        "warning_policy": {
            "all_warnings_captured": True,
            "convergence_warning_triggers_failed_attempt": True,
            "other_warnings_recorded_for_audit": True,
        },
    },
    "selection_policy": {
        "model_selection": False,
        "hyperparameter_selection": False,
        "test_tuning": False,
        "report_both_models_separately": True,
    },
    "metrics": {
        "primary": {"name": "PinballLoss", "tau": 0.5},
        "secondary": ["MAE", "RMSE", "R2"],
    },
    "expected_inventory": {
        "models": 2,
        "assets": 4,
        "test_observations": 584,
        "primary_fits_without_retries": 4672,
    },
    "required_outputs": [
        "584 primary predictions per model and asset",
        "584 expected-variance plug-in diagnostics per model and asset",
        "584 direct conditional-sigma diagnostics per model and asset",
        "per-model per-asset metrics",
        "task-average volatility metrics",
        "final-vs-GARCH comparison table",
        "GARCH-vs-VolPersistence comparison table",
        "584-length Pinball loss series for stage 09",
        "anchor and realization dates",
        "h, nu, S, Q, c, C, q and central-mass diagnostics",
        "fit, warning, convergence, retry and persistence diagnostics",
        "package versions and protocol SHA-256",
    ],
    "audit_requirements": [
        "Exact protocol SHA-256 verification",
        "Exact official-input SHA-256 verification",
        "Exact 584-date alignment for every model and asset",
        "No-look-ahead verification",
        "Independent reconstruction from stored h, nu, S, Q and dates",
        "Independent metric and loss-series reconstruction",
        "Convergence, warning and retry inventory reconstruction",
        "No silent fallback or imputation verification",
        "Comparison-table reconstruction",
    ],
    "interpretation_guardrails": [
        "08C audit PASS does not imply model success.",
        "Lower point-estimate error does not imply statistical significance.",
        "Statistical significance awaits stage 09 DM, Harvey and Holm-Bonferroni analyses.",
        "08C does not change or retrain the locked final model.",
        "08C is date-target aligned but not an input-representation-matched causal ablation.",
        "No econometric model may be added or removed after official 08C results are observed.",
        "GARCH-family performance may depend on asset and target definition.",
    ],
    "preflight_evidence": {
        "student_t": {
            "distribution_name": "Standardized Student's t",
            "test_nu": nu_test,
            "median": student_t_median,
            "second_moment": student_t_second_moment,
        },
        "cdf_root_synthetic": {
            "sigma": sigma_test,
            "c": c_test,
            "q": q_test,
            "official_initial_upper_bound": max(
                sigma_test, abs(c_test), 1e-12
            ),
            "upper_bound_doublings": upper_doublings,
            "central_probability_mass": central_mass_value,
            "mass_error": mass_error,
            "symmetry_root_q": q0_root,
            "symmetry_ppf_q": q0_ppf,
            "symmetry_difference": symmetry_difference,
            "symmetry_upper_bound_doublings": q0_upper_doublings,
            "passed": True,
        },
        "test_integrity": {
            "official_test_target_values_used_for_selection": False,
            "official_test_target_values_used_for_08C_metrics": False,
            "test_arrays_read_only_for_integrity_validation": True,
            "no_08C_model_fit": True,
            "no_08C_prediction_computed": True,
            "no_08C_metric_computed": True,
        },
    },
}


# ==========================================================
# 6. ORTAM SÜRÜM KİLİDİ
# ==========================================================

actual_package_versions = {
    "arch": arch.__version__,
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "scipy": scipy.__version__,
}

if actual_package_versions != EXPECTED_PACKAGE_VERSIONS:
    raise RuntimeError(
        "Paket sürümleri preflight ile kilitlenen ortamdan farklı.\n"
        f"Beklenen: {EXPECTED_PACKAGE_VERSIONS}\n"
        f"Gerçek  : {actual_package_versions}"
    )


# ==========================================================
# 7. DETERMINİSTİK JSON YAZIMI + TAM EŞİTLİK
# ==========================================================

json_text = json.dumps(
    protocol,
    ensure_ascii=False,
    indent=2,
    sort_keys=True,
)

LOCK_PATH.write_text(json_text + "\n", encoding="utf-8")

loaded = json.loads(LOCK_PATH.read_text(encoding="utf-8"))

if loaded != protocol:
    raise RuntimeError(
        "Yazılan JSON tekrar okunduğunda protocol sözlüğüyle tam aynı değil."
    )


# ==========================================================
# 8. SHA-256 YAZIMI + BAĞIMSIZ YENİDEN DOĞRULAMA
# ==========================================================

lock_sha256 = sha256_file(LOCK_PATH)
SHA_PATH.write_text(lock_sha256 + "\n", encoding="utf-8")

sha_from_file = SHA_PATH.read_text(encoding="utf-8").strip()
sha_recomputed = sha256_file(LOCK_PATH)

if sha_from_file != lock_sha256:
    raise RuntimeError("SHA dosyasındaki değer oluşturulan SHA ile aynı değil.")

if sha_recomputed != lock_sha256:
    raise RuntimeError("JSON SHA-256 yeniden hesaplamada aynı çıkmadı.")


# ==========================================================
# 9. SON KONTROL VE RAPOR
# ==========================================================

if loaded["expected_inventory"]["primary_fits_without_retries"] != 4672:
    raise RuntimeError("Beklenen fit sayısı 4672 değil.")

if loaded["expanding_window"]["first_fit_observation_count_including_anchor"] != 3308:
    raise RuntimeError("İlk fit gözlem sayısı 3308 değil.")

if loaded["data_lock"]["target_verification_rule"]["exact_array_equality_after_dtype_alignment"] is not True:
    raise RuntimeError("Dtype-duyarlı exact target eşitliği kilitte True değil.")

if loaded["warm_start_policy"]["warm_start_allowed"] is not False:
    raise RuntimeError("Warm-start yasağı kilitte değil.")

print("=" * 110)
print("08C GARCH PROTOCOL LOCK — CREATED AND SELF-VALIDATED")
print("=" * 110)
print("Protocol path :", LOCK_PATH)
print("SHA path      :", SHA_PATH)
print("SHA-256       :", lock_sha256)
print("Status        :", loaded["status"])
print("Models        :", [m["model_id"] for m in loaded["official_models"]])
print("First fit n   :", loaded["expanding_window"]["first_fit_observation_count_including_anchor"])
print("Last fit n    :", loaded["expanding_window"]["last_fit_observation_count_including_anchor"])
print("Target exact  :", loaded["data_lock"]["target_verification_rule"]["exact_array_equality_after_dtype_alignment"])
print("Warm start    :", loaded["warm_start_policy"]["warm_start_allowed"])
print("CDF preflight :", loaded["preflight_evidence"]["cdf_root_synthetic"]["passed"])
print("JSON equality :", loaded == protocol)
print("SHA verified  :", sha_from_file == sha_recomputed == lock_sha256)
print("\nActivity:")
print(" - 08C model fit       : NO")
print(" - 08C prediction      : NO")
print(" - 08C metric          : NO")
print(" - Input integrity read: YES")
print(" - Protocol locked     : YES")
print("=" * 110)
