from pathlib import Path
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
# 1. YOLLAR VE SABİT BEKLENTİLER
# ==========================================================

BASE = Path("/content/drive/MyDrive/tez_transformer_v4_repro")
PROCESSED = BASE / "data" / "processed"
SEQ = BASE / "data" / "sequences" / "baseline" / "lb10"
CONFIG = BASE / "config"

LOCK_PATH = CONFIG / "08C_garch_protocol_lock_v4.json"
SHA_PATH = CONFIG / "08C_garch_protocol_lock_v4.sha256"

EXPECTED_LOCK_SHA256 = (
    "4238b4280021e8a265dcb6ba0aa6da379d1388db7d4396e29463d476283a6614"
)

EXPECTED_PACKAGE_VERSIONS = {
    "arch": "8.0.0",
    "pandas": "2.2.2",
    "numpy": "2.0.2",
    "scipy": "1.16.3",
}

EXPECTED_INPUT_SHA256 = {
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

EXPECTED_RELATIVE_PATHS = {
    "features_baseline": "data/processed/features_baseline.csv",
    "anchor_dates_test": "data/sequences/baseline/lb10/anchor_dates_test.npy",
    "target_realization_dates_test": "data/sequences/baseline/lb10/target_realization_dates_test.npy",
    "y_test_raw": "data/sequences/baseline/lb10/y_test_raw.npy",
    "prices_clean": "data/processed/prices_clean.csv",
    "targets_all": "data/processed/targets_all.csv",
    "target_realization_dates": "data/processed/target_realization_dates.csv",
    "split_meta": "data/processed/split_meta_v4.json",
    "meta_v4": "data/processed/meta_v4.json",
    "sequence_meta": "data/sequences/baseline/lb10/sequence_meta.json",
}

CHECKS = []


def record(name: str, passed: bool, detail: str = ""):
    CHECKS.append((name, bool(passed), detail))
    if not passed:
        raise RuntimeError(f"CHECK FAILED: {name}\n{detail}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_from_record(record_dict: dict) -> Path:
    return BASE / record_dict["relative_path"]


# ==========================================================
# 2. DOSYA VARLIĞI, LOCK SHA VE JSON
# ==========================================================

record("LOCK_PATH exists", LOCK_PATH.exists(), str(LOCK_PATH))
record("SHA_PATH exists", SHA_PATH.exists(), str(SHA_PATH))

actual_lock_sha = sha256_file(LOCK_PATH)
sha_file_value = SHA_PATH.read_text(encoding="utf-8").strip()

record(
    "Lock SHA matches hard-coded expected SHA",
    actual_lock_sha == EXPECTED_LOCK_SHA256,
    f"actual={actual_lock_sha} expected={EXPECTED_LOCK_SHA256}",
)
record(
    "SHA file matches lock SHA",
    sha_file_value == actual_lock_sha,
    f"sha_file={sha_file_value} actual={actual_lock_sha}",
)

protocol = json.loads(LOCK_PATH.read_text(encoding="utf-8"))

record(
    "Protocol status exact",
    protocol["status"] == "LOCKED_BEFORE_08C_GARCH_FITS_PREDICTIONS_AND_METRICS",
    protocol["status"],
)
record("Protocol stage exact", protocol["stage"] == "08C", str(protocol["stage"]))
record("Protocol version exact", protocol["protocol_version"] == 1, str(protocol["protocol_version"]))


# ==========================================================
# 3. ORTAM SÜRÜMLERİ
# ==========================================================

actual_versions = {
    "arch": arch.__version__,
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "scipy": scipy.__version__,
}

record(
    "Runtime package versions exact",
    actual_versions == EXPECTED_PACKAGE_VERSIONS,
    f"actual={actual_versions}",
)
record(
    "JSON required package versions exact",
    protocol["environment"]["required_package_versions"] == EXPECTED_PACKAGE_VERSIONS,
    str(protocol["environment"]["required_package_versions"]),
)


# ==========================================================
# 4. INPUT DOSYALARI VE SHA KAYITLARI
# ==========================================================

all_records = {}
all_records.update(protocol["data_lock"]["official_execution_inputs"])
all_records.update(protocol["data_lock"]["independent_verification_inputs"])

record("All 10 input records present", set(all_records) == set(EXPECTED_INPUT_SHA256), str(sorted(all_records)))

for key in sorted(EXPECTED_INPUT_SHA256):
    rec = all_records[key]
    actual_path = path_from_record(rec)

    record(
        f"{key}: relative path exact",
        rec["relative_path"] == EXPECTED_RELATIVE_PATHS[key],
        rec["relative_path"],
    )
    record(f"{key}: file exists", actual_path.exists(), str(actual_path))

    actual_sha = sha256_file(actual_path)

    record(
        f"{key}: actual SHA exact",
        actual_sha == EXPECTED_INPUT_SHA256[key],
        f"actual={actual_sha}",
    )
    record(
        f"{key}: JSON SHA exact",
        rec["sha256"] == EXPECTED_INPUT_SHA256[key],
        f"json={rec['sha256']}",
    )
    record(
        f"{key}: JSON size exact",
        rec["size_bytes"] == actual_path.stat().st_size,
        f"json={rec['size_bytes']} actual={actual_path.stat().st_size}",
    )


# ==========================================================
# 5. VERİ, TARİH VE HEDEF BAĞIMSIZ YENİDEN DOĞRULAMA
# ==========================================================

features = pd.read_csv(
    BASE / EXPECTED_RELATIVE_PATHS["features_baseline"],
    index_col=0,
    parse_dates=True,
)
prices = pd.read_csv(
    BASE / EXPECTED_RELATIVE_PATHS["prices_clean"],
    index_col=0,
    parse_dates=True,
)
targets = pd.read_csv(
    BASE / EXPECTED_RELATIVE_PATHS["targets_all"],
    index_col=0,
    parse_dates=True,
)
target_dates_csv = pd.read_csv(
    BASE / EXPECTED_RELATIVE_PATHS["target_realization_dates"],
    index_col=0,
    parse_dates=True,
)

anchor_dates = pd.DatetimeIndex(
    pd.to_datetime(
        np.load(
            BASE / EXPECTED_RELATIVE_PATHS["anchor_dates_test"],
            allow_pickle=False,
        ).astype(str)
    )
)
realization_dates_npy = pd.DatetimeIndex(
    pd.to_datetime(
        np.load(
            BASE / EXPECTED_RELATIVE_PATHS["target_realization_dates_test"],
            allow_pickle=False,
        ).astype(str)
    )
)
y_test_raw = np.load(
    BASE / EXPECTED_RELATIVE_PATHS["y_test_raw"],
    allow_pickle=False,
)

record("features shape exact", features.shape == (3891, 8), str(features.shape))
record("targets shape exact", targets.shape == (3891, 8), str(targets.shape))
record("features index equals targets index", features.index.equals(targets.index))
record("common index start exact", features.index.min() == pd.Timestamp("2010-02-01"))
record("common index end exact", features.index.max() == pd.Timestamp("2024-12-30"))

record("anchor count exact", len(anchor_dates) == 584, str(len(anchor_dates)))
record("anchor unique", anchor_dates.is_unique)
record("anchor monotonic", anchor_dates.is_monotonic_increasing)
record("anchor start exact", anchor_dates[0] == pd.Timestamp("2022-10-05"))
record("anchor end exact", anchor_dates[-1] == pd.Timestamp("2024-12-30"))

record("realization count exact", len(realization_dates_npy) == 584)
record("realization start exact", realization_dates_npy[0] == pd.Timestamp("2022-10-06"))
record("realization end exact", realization_dates_npy[-1] == pd.Timestamp("2024-12-31"))
record(
    "all realization dates after anchors",
    np.all(realization_dates_npy.values > anchor_dates.values),
)

record("y_test_raw shape exact", y_test_raw.shape == (584, 8), str(y_test_raw.shape))
record("y_test_raw dtype exact", y_test_raw.dtype == np.float32, str(y_test_raw.dtype))

targets_csv = targets.loc[anchor_dates].to_numpy(dtype=np.float64)
targets_cast = targets_csv.astype(y_test_raw.dtype)

record(
    "dtype-aware target exact equality",
    np.array_equal(y_test_raw, targets_cast),
)
float64_max_diff = float(
    np.max(np.abs(y_test_raw.astype(np.float64) - targets_csv))
)
record(
    "diagnostic float64 max diff exact",
    abs(
        float64_max_diff
        - protocol["data_lock"]["target_verification_rule"]["diagnostic_float64_max_abs_diff"]
    ) < 1e-18,
    f"actual={float64_max_diff}",
)

realization_dates_csv = pd.DatetimeIndex(
    pd.to_datetime(
        target_dates_csv.loc[anchor_dates].iloc[:, 0].to_numpy()
    )
)
record(
    "CSV-NPY realization date exact equality",
    np.array_equal(realization_dates_csv.values, realization_dates_npy.values),
)

assets = ["BIST100", "USDTRY", "EURTRY", "GOLD"]
for asset in assets:
    feature_col = f"{asset}_LogRet"
    rebuilt = np.log(prices[asset] / prices[asset].shift(1))
    common = features.index.intersection(rebuilt.dropna().index)
    max_diff = float(
        np.max(
            np.abs(
                features.loc[common, feature_col].to_numpy(dtype=np.float64)
                - rebuilt.loc[common].to_numpy(dtype=np.float64)
            )
        )
    )
    record(
        f"{asset}: LogRet independent reconstruction < 1e-12",
        max_diff < 1e-12,
        f"max_diff={max_diff}",
    )

first_pos = int(features.index.get_loc(anchor_dates[0]))
last_pos = int(features.index.get_loc(anchor_dates[-1]))

record("first anchor zero-based position exact", first_pos == 3307, str(first_pos))
record("first fit observation count exact", first_pos + 1 == 3308, str(first_pos + 1))
record("last fit observation count exact", last_pos + 1 == 3891, str(last_pos + 1))


# ==========================================================
# 6. KRİTİK PROTOKOL KARARLARI
# ==========================================================

model_ids = [m["model_id"] for m in protocol["official_models"]]
record(
    "Official model IDs exact",
    model_ids == [
        "GARCH_1_1_StudentsT_ZeroMean",
        "GJR_GARCH_1_1_StudentsT_ZeroMean",
    ],
    str(model_ids),
)
record("EGARCH excluded", protocol["excluded_models"] == ["EGARCH"])
record("Warm start disabled", protocol["warm_start_policy"]["warm_start_allowed"] is False)
record("starting_values locked None", protocol["warm_start_policy"]["starting_values"] is None)
record("expected primary fits exact", protocol["expected_inventory"]["primary_fits_without_retries"] == 4672)

record(
    "Primary formula exact",
    protocol["primary_forecast"]["forecast_formula"]
    == "sqrt(252 * (C + q**2 / 20))",
)
record(
    "Primary bracket exact",
    protocol["primary_forecast"]["root_settings"]["initial_upper_bound"]
    == "max(sigma_decimal, abs(c), 1e-12)",
)
record(
    "Primary mass tolerance exact",
    protocol["primary_forecast"]["root_settings"]["maximum_allowed_mass_error"]
    == 1e-10,
)
record(
    "Secondary plugin formula exact",
    protocol["secondary_forecasts"]["expected_sample_variance_plugin"]["formula_1"]
    == "sqrt(252 * (C + (h_decimal + c**2) / 20))",
)
record(
    "Blind max-zero disabled",
    protocol["numerical_safety"]["blind_max_zero_for_large_negative_values"] is False,
)
record(
    "Maximum attempts exact",
    protocol["fit_and_retry_policy"]["maximum_attempts"] == 2,
)
record(
    "No silent fallback values",
    all(value is False for value in protocol["fit_and_retry_policy"]["silent_fallbacks"].values()),
)

criteria = set(protocol["fit_success_criteria"]["criteria"])
required_criteria = {
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
}
record("All fit-success criteria exact", criteria == required_criteria, str(criteria))

record("No model selection", protocol["selection_policy"]["model_selection"] is False)
record("No hyperparameter selection", protocol["selection_policy"]["hyperparameter_selection"] is False)
record("No test tuning", protocol["selection_policy"]["test_tuning"] is False)
record("Primary metric exact", protocol["metrics"]["primary"] == {"name": "PinballLoss", "tau": 0.5})


# ==========================================================
# 7. STUDENT-T VE CDF-KÖK BAĞIMSIZ PREFLIGHT
# ==========================================================

dist = StudentsT()
nu = 8.0
sigma = 0.012
c = 0.0015


def cdf_scalar(z: float) -> float:
    value = dist.cdf(np.asarray([z], dtype=np.float64), [nu])
    return float(np.asarray(value).reshape(-1)[0])


def predictive_cdf(x: float) -> float:
    return cdf_scalar(x / sigma)


def mass(q: float) -> float:
    return predictive_cdf(c + q) - predictive_cdf(c - q)


upper = max(sigma, abs(c), 1e-12)
doublings = 0
while mass(upper) < 0.5:
    upper *= 2.0
    doublings += 1
    if doublings > 100:
        raise RuntimeError("Independent CDF bracket could not be found.")

q = brentq(
    lambda x: mass(x) - 0.5,
    0.0,
    upper,
    xtol=1e-14,
    rtol=1e-12,
    maxiter=200,
)

mass_error = abs(mass(q) - 0.5)
record("Independent CDF-root mass error <= 1e-10", mass_error <= 1e-10, str(mass_error))

q0 = sigma * float(np.asarray(dist.ppf(0.75, [nu])).reshape(-1)[0])
q0_root = brentq(
    lambda x: cdf_scalar(x / sigma) - cdf_scalar(-x / sigma) - 0.5,
    0.0,
    max(sigma, 1e-12) * 4.0,
    xtol=1e-14,
    rtol=1e-12,
    maxiter=200,
)
record("Independent symmetry check <= 1e-10", abs(q0 - q0_root) <= 1e-10)

record(
    "Student-t median near zero",
    abs(float(np.asarray(dist.ppf(0.5, [nu])).reshape(-1)[0])) <= 1e-12,
)
record(
    "Student-t second moment exact",
    abs(float(dist.moment(2, [nu])) - 1.0) <= 1e-12,
)


# ==========================================================
# 8. SONUÇ
# ==========================================================

passed_count = sum(int(passed) for _, passed, _ in CHECKS)
total_count = len(CHECKS)

print("=" * 110)
print("08C GARCH PROTOCOL LOCK — INDEPENDENT VALIDATOR")
print("=" * 110)
print("Protocol path :", LOCK_PATH)
print("Lock SHA-256  :", actual_lock_sha)
print("Checks passed :", f"{passed_count}/{total_count}")
print("Validator PASS:", passed_count == total_count)
print("\nActivity:")
print(" - 08C model fit       : NO")
print(" - 08C prediction      : NO")
print(" - 08C metric          : NO")
print(" - Protocol modified   : NO")
print(" - Independent read/audit only: YES")
print("=" * 110)
