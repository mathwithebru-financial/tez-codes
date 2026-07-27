# ==========================================================
# 08A_naive_baselines_test_v4.py
#
# RESMÎ TEST NAİF BASELINE KARŞILAŞTIRMASI
#
# Amaç:
#   Kilitli final 3-seed ensemble tahminini test döneminde
#   üç naif baseline ile karşılaştırmak:
#
#   RETURN:
#     1. ReturnZero
#        y_hat_ret(t+1) = 0
#
#     2. ReturnPersistence
#        y_hat_ret(t+1) = LogRet(t)
#
#   VOLATILITY:
#     3. VolPersistence
#        y_hat_vol(t+1) = Vol20(t)
#
# Kritik metodolojik kural:
#   - Model seçimi YOKTUR.
#   - Hyperparameter değişikliği YOKTUR.
#   - Yeniden eğitim YOKTUR.
#   - 07 final model sonucu değiştirilmez.
#   - Test sonucu görülerek yeni finalist seçilmez.
#
# Persistence inşası:
#   NextRet[t] = LogRet[t+1]
#   NextVol[t] = Vol20[t+1]
#
# Bu nedenle test target dizisinin bir önceki gerçekleşmiş target değeri,
# bir sonraki test anchor günü için gözlenen güncel LogRet/Vol20 değeridir.
# İlk test satırında tarihsel başlangıç değeri olarak validation'ın son
# gerçekleşmiş target satırı kullanılır. Böylece:
#
#   prev_observed_raw[0]  = y_val_raw[-1]
#   prev_observed_raw[i]  = y_test_raw[i-1], i >= 1
#
# ve:
#   ReturnPersistence = prev_observed_raw[:, :4]
#   VolPersistence    = prev_observed_raw[:, 4:]
#
# Ana metrikler:
#   Return     : MAE
#   Volatility : PinballLoss(tau=0.5)
#
# Ek metrikler:
#   MAE, RMSE, R²
#
# Baseline-normalize oran:
#   ratio = FinalModelError / BaselineError
#
# Yorum:
#   ratio < 1  -> final model baseline'dan daha düşük hata
#   ratio = 1  -> eşit hata
#   ratio > 1  -> baseline daha düşük hata
#
# 1.30 ve 1.50:
#   Önceden kilitlenmiş, çalışmaya özgü tanısal raporlama eşikleridir.
#   Evrensel literatür standardı ve model seçim eşiği değildir.
#
# Çıktılar:
#   results/baselines/naive/
#     naive_baseline_metrics_long_v4.csv
#     naive_baseline_comparison_v4.csv
#     naive_baseline_comparison_summary_v4.csv
#     naive_baseline_summary_v4.json
#     pred_return_zero_raw_v4.npy
#     pred_return_persistence_raw_v4.npy
#     pred_vol_persistence_raw_v4.npy
#     naive_baseline_loss_series_v4.npz
#
# ==========================================================

import os
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ==========================================================
# 1. YOLLAR
# ==========================================================

BASE_DIR = "/content/drive/MyDrive/tez_transformer_v4_repro"

CONFIG_DIR = os.path.join(
    BASE_DIR,
    "config"
)

SCRIPTS_DIR = os.path.join(
    BASE_DIR,
    "scripts"
)

SEQUENCE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "sequences"
)

FINAL_TEST_DIR = os.path.join(
    BASE_DIR,
    "results",
    "final_test"
)

MULTISEED_RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "multiseed"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "baselines",
    "naive"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


SCRIPT_07_PATH = os.path.join(
    SCRIPTS_DIR,
    "07_final_test_evaluation_v4.py"
)

CODE_MANIFEST_PATH = os.path.join(
    CONFIG_DIR,
    "code_manifest_v4.csv"
)

WINNER_JSON = os.path.join(
    MULTISEED_RESULTS_DIR,
    "multiseed_winner_config_v4.json"
)

FINAL_SUMMARY_JSON = os.path.join(
    FINAL_TEST_DIR,
    "final_test_summary_v4.json"
)

FINAL_METRICS_CSV = os.path.join(
    FINAL_TEST_DIR,
    "final_test_metrics_long_v4.csv"
)

FINAL_Y_TRUE_RAW = os.path.join(
    FINAL_TEST_DIR,
    "final_test_y_true_raw_v4.npy"
)

FINAL_ENSEMBLE_PRED_RAW = os.path.join(
    FINAL_TEST_DIR,
    "pred_final_ensemble_raw_v4.npy"
)


METRICS_LONG_CSV = os.path.join(
    RESULTS_DIR,
    "naive_baseline_metrics_long_v4.csv"
)

COMPARISON_CSV = os.path.join(
    RESULTS_DIR,
    "naive_baseline_comparison_v4.csv"
)

COMPARISON_SUMMARY_CSV = os.path.join(
    RESULTS_DIR,
    "naive_baseline_comparison_summary_v4.csv"
)

SUMMARY_JSON = os.path.join(
    RESULTS_DIR,
    "naive_baseline_summary_v4.json"
)

RETURN_ZERO_PRED_PATH = os.path.join(
    RESULTS_DIR,
    "pred_return_zero_raw_v4.npy"
)

RETURN_PERSISTENCE_PRED_PATH = os.path.join(
    RESULTS_DIR,
    "pred_return_persistence_raw_v4.npy"
)

VOL_PERSISTENCE_PRED_PATH = os.path.join(
    RESULTS_DIR,
    "pred_vol_persistence_raw_v4.npy"
)

LOSS_SERIES_PATH = os.path.join(
    RESULTS_DIR,
    "naive_baseline_loss_series_v4.npz"
)


# ==========================================================
# 2. KİLİTLİ DEĞERLER
# ==========================================================

EXPECTED_07_SHA256 = (
    "8b0e3cf2edb9508b4fddd402ddcdbf8c"
    "4d2acd6080ffe6fe1876ad818306cd74"
)

EXPECTED_WINNER_CONFIG_ID = (
    "arch=NoSharing"
    "__loss=FixedLambda_0.7"
    "__lb=10"
    "__size=small"
    "__feat=baseline"
)

EXPECTED_PRIMARY_LABEL = (
    "FinalWinner_3SeedEnsemble"
)

EXPECTED_PRIMARY_POLICY = (
    "arithmetic_mean_of_three_locked_winner_seed_predictions_in_raw_scale"
)

EXPECTED_SEEDS = [
    123,
    777,
    2026
]

EXPECTED_TEST_SAMPLES = 584

EXPECTED_ASSETS = [
    "BIST100",
    "USDTRY",
    "EURTRY",
    "GOLD"
]

FEATURE_SET = "baseline"
LOOKBACK = 10

TAU = 0.5

DIAGNOSTIC_WARNING_THRESHOLD = 1.30
DIAGNOSTIC_SEVERE_THRESHOLD = 1.50

FLOAT_TOL = 1e-12


# ==========================================================
# 3. YARDIMCI FONKSİYONLAR
# ==========================================================

def sha256_file(
    path,
    chunk_size=1024 * 1024
):
    sha = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:
        while True:
            chunk = f.read(
                chunk_size
            )

            if not chunk:
                break

            sha.update(
                chunk
            )

    return sha.hexdigest()


def normalize_bool(
    value
):
    if isinstance(
        value,
        (bool, np.bool_)
    ):
        return bool(
            value
        )

    if value is None:
        return False

    if (
        isinstance(
            value,
            float
        )
        and np.isnan(
            value
        )
    ):
        return False

    text = str(
        value
    ).strip().lower()

    if text in {
        "true",
        "1",
        "yes"
    }:
        return True

    if text in {
        "false",
        "0",
        "no",
        ""
    }:
        return False

    raise ValueError(
        f"Boolean değere çevrilemeyen kayıt: {value!r}"
    )


def assert_close(
    actual,
    expected,
    name,
    tol=FLOAT_TOL
):
    actual = float(
        actual
    )

    expected = float(
        expected
    )

    if not np.isclose(
        actual,
        expected,
        rtol=0.0,
        atol=tol
    ):
        raise RuntimeError(
            f"{name} uyuşmuyor.\n"
            f"Beklenen: {expected}\n"
            f"Gerçek   : {actual}\n"
            f"Fark     : {abs(actual - expected)}"
        )


def mae_np(
    y_true,
    y_pred
):
    return float(
        np.mean(
            np.abs(
                y_true
                - y_pred
            )
        )
    )


def rmse_np(
    y_true,
    y_pred
):
    return float(
        np.sqrt(
            np.mean(
                (
                    y_true
                    - y_pred
                ) ** 2
            )
        )
    )


def r2_np(
    y_true,
    y_pred
):
    ss_res = np.sum(
        (
            y_true
            - y_pred
        ) ** 2
    )

    ss_tot = np.sum(
        (
            y_true
            - np.mean(
                y_true
            )
        ) ** 2
    )

    if ss_tot == 0:
        return float(
            "nan"
        )

    return float(
        1.0
        - ss_res
        / ss_tot
    )


def pinball_loss_series_np(
    y_true,
    y_pred,
    tau=0.5
):
    diff = (
        y_true
        - y_pred
    )

    return np.maximum(
        tau * diff,
        (
            tau
            - 1.0
        ) * diff
    )


def pinball_np(
    y_true,
    y_pred,
    tau=0.5
):
    return float(
        np.mean(
            pinball_loss_series_np(
                y_true,
                y_pred,
                tau=tau
            )
        )
    )


def build_return_metric_rows(
    model_label,
    y_true_return,
    y_pred_return
):
    rows = []

    for i, asset in enumerate(
        EXPECTED_ASSETS
    ):
        true = y_true_return[
            :,
            i
        ]

        pred = y_pred_return[
            :,
            i
        ]

        rows.append({
            "model_label":
                model_label,

            "task":
                "return",

            "asset":
                asset,

            "MAE":
                mae_np(
                    true,
                    pred
                ),

            "RMSE":
                rmse_np(
                    true,
                    pred
                ),

            "R2":
                r2_np(
                    true,
                    pred
                ),

            "PinballLoss_tau_0.5":
                np.nan,
        })

    return rows


def build_vol_metric_rows(
    model_label,
    y_true_vol,
    y_pred_vol
):
    rows = []

    for i, asset in enumerate(
        EXPECTED_ASSETS
    ):
        true = y_true_vol[
            :,
            i
        ]

        pred = y_pred_vol[
            :,
            i
        ]

        rows.append({
            "model_label":
                model_label,

            "task":
                "volatility",

            "asset":
                asset,

            "MAE":
                mae_np(
                    true,
                    pred
                ),

            "RMSE":
                rmse_np(
                    true,
                    pred
                ),

            "R2":
                r2_np(
                    true,
                    pred
                ),

            "PinballLoss_tau_0.5":
                pinball_np(
                    true,
                    pred,
                    tau=TAU
                ),
        })

    return rows


# ==========================================================
# 4. PREFLIGHT — GEREKLİ DOSYALAR
# ==========================================================

SEQ_DIR = os.path.join(
    SEQUENCE_DIR,
    FEATURE_SET,
    f"lb{LOOKBACK}"
)

Y_VAL_RAW_PATH = os.path.join(
    SEQ_DIR,
    "y_val_raw.npy"
)

Y_TEST_RAW_SEQUENCE_PATH = os.path.join(
    SEQ_DIR,
    "y_test_raw.npy"
)


required_paths = [
    SCRIPT_07_PATH,
    CODE_MANIFEST_PATH,
    WINNER_JSON,
    FINAL_SUMMARY_JSON,
    FINAL_METRICS_CSV,
    FINAL_Y_TRUE_RAW,
    FINAL_ENSEMBLE_PRED_RAW,
    Y_VAL_RAW_PATH,
    Y_TEST_RAW_SEQUENCE_PATH,
]


missing_paths = [
    path
    for path in required_paths
    if not os.path.exists(
        path
    )
]


if missing_paths:
    raise FileNotFoundError(
        "08A preflight için gerekli dosyalar eksik:\n"
        + "\n".join(
            missing_paths
        )
    )


print(
    "=" * 110
)

print(
    "08A — v4 RESMÎ TEST NAİF BASELINE KARŞILAŞTIRMASI"
)

print(
    "=" * 110
)

print(
    "[MODEL SELECTION INSIDE 08A] NONE"
)

print(
    "[RETRAINING INSIDE 08A] NONE"
)

print(
    "[FINAL MODEL CHANGE] NONE"
)

print(
    "[TEST SAMPLE EXPECTATION] 584"
)


# ==========================================================
# 5. 07 PROVENANCE + MANIFEST
# ==========================================================

actual_07_sha = sha256_file(
    SCRIPT_07_PATH
)


if (
    actual_07_sha
    != EXPECTED_07_SHA256
):
    raise RuntimeError(
        "07 script SHA-256 beklenen resmî sürümle eşleşmiyor.\n"
        f"Beklenen: {EXPECTED_07_SHA256}\n"
        f"Gerçek  : {actual_07_sha}"
    )


manifest_df = pd.read_csv(
    CODE_MANIFEST_PATH
)


manifest_match = manifest_df[
    (
        manifest_df[
            "script_name"
        ]
        == "07_final_test_evaluation_v4.py"
    )
    &
    (
        manifest_df[
            "sha256"
        ]
        == actual_07_sha
    )
]


if len(
    manifest_match
) < 1:
    raise RuntimeError(
        "07 script mevcut hash ile code manifest içinde bulunamadı."
    )


print(
    "\n[PROVENANCE]"
)

print(
    "07 SHA-256 manifest eşleşmesi: TRUE"
)

print(
    "07 SHA-256:",
    actual_07_sha
)


# ==========================================================
# 6. KİLİTLİ WINNER + 07 PRIMARY POLICY
# ==========================================================

with open(
    WINNER_JSON,
    "r",
    encoding="utf-8"
) as f:
    winner_payload = json.load(
        f
    )


winner = winner_payload[
    "winner"
]


if str(
    winner[
        "config_id"
    ]
) != EXPECTED_WINNER_CONFIG_ID:
    raise RuntimeError(
        "Kilitli winner config_id değişmiş."
    )


with open(
    FINAL_SUMMARY_JSON,
    "r",
    encoding="utf-8"
) as f:
    final_summary_json = json.load(
        f
    )


if str(
    final_summary_json[
        "winner_config_id"
    ]
) != EXPECTED_WINNER_CONFIG_ID:
    raise RuntimeError(
        "07 final summary winner_config_id yanlış."
    )


if str(
    final_summary_json[
        "primary_prediction"
    ]
) != EXPECTED_PRIMARY_LABEL:
    raise RuntimeError(
        "07 primary prediction ensemble değil."
    )


if str(
    final_summary_json[
        "primary_test_policy"
    ]
) != EXPECTED_PRIMARY_POLICY:
    raise RuntimeError(
        "07 primary test policy değişmiş."
    )


if sorted(
    final_summary_json[
        "expected_seeds"
    ]
) != EXPECTED_SEEDS:
    raise RuntimeError(
        "07 expected seed set yanlış."
    )


if not normalize_bool(
    final_summary_json[
        "test_access_started"
    ]
):
    raise RuntimeError(
        "07 test_access_started=False."
    )


if not normalize_bool(
    final_summary_json[
        "test_metrics_computed"
    ]
):
    raise RuntimeError(
        "07 test_metrics_computed=False."
    )


print(
    "\n[LOCKED FINAL MODEL]"
)

print(
    EXPECTED_WINNER_CONFIG_ID
)

print(
    "[PRIMARY TEST POLICY]"
)

print(
    EXPECTED_PRIMARY_POLICY
)


# ==========================================================
# 7. RAW VERİLERİ YÜKLE
# ==========================================================

y_true_raw = np.load(
    FINAL_Y_TRUE_RAW
)

final_pred_raw = np.load(
    FINAL_ENSEMBLE_PRED_RAW
)

y_val_raw = np.load(
    Y_VAL_RAW_PATH
)

y_test_raw_sequence = np.load(
    Y_TEST_RAW_SEQUENCE_PATH
)


expected_shape = (
    EXPECTED_TEST_SAMPLES,
    8
)


if (
    y_true_raw.shape
    != expected_shape
):
    raise RuntimeError(
        f"07 y_true_raw shape yanlış: {y_true_raw.shape}"
    )


if (
    final_pred_raw.shape
    != expected_shape
):
    raise RuntimeError(
        f"Final ensemble shape yanlış: {final_pred_raw.shape}"
    )


if (
    y_test_raw_sequence.shape
    != expected_shape
):
    raise RuntimeError(
        f"Sequence y_test_raw shape yanlış: {y_test_raw_sequence.shape}"
    )


if y_val_raw.ndim != 2 or y_val_raw.shape[1] != 8:
    raise RuntimeError(
        f"y_val_raw shape geçersiz: {y_val_raw.shape}"
    )


if not np.isfinite(
    y_true_raw
).all():
    raise RuntimeError(
        "y_true_raw içinde NaN/Inf var."
    )


if not np.isfinite(
    final_pred_raw
).all():
    raise RuntimeError(
        "Final prediction içinde NaN/Inf var."
    )


if not np.isfinite(
    y_val_raw
).all():
    raise RuntimeError(
        "y_val_raw içinde NaN/Inf var."
    )


if not np.isfinite(
    y_test_raw_sequence
).all():
    raise RuntimeError(
        "Sequence y_test_raw içinde NaN/Inf var."
    )


max_test_truth_diff = float(
    np.max(
        np.abs(
            y_true_raw
            - y_test_raw_sequence
        )
    )
)


if (
    max_test_truth_diff
    > FLOAT_TOL
):
    raise RuntimeError(
        "07 y_true_raw ile sequence y_test_raw uyuşmuyor.\n"
        f"Max fark: {max_test_truth_diff}"
    )


print(
    "\n[RAW DATA AUDIT]"
)

print(
    "y_true_raw shape        :",
    y_true_raw.shape
)

print(
    "final_pred_raw shape     :",
    final_pred_raw.shape
)

print(
    "y_val_raw shape          :",
    y_val_raw.shape
)

print(
    "Sequence truth max diff  :",
    f"{max_test_truth_diff:.16e}"
)


# ==========================================================
# 8. NAİF BASELINE TAHMİNLERİNİ OLUŞTUR
# ==========================================================

# Her test satırı için anchor gününde gözlenen son gerçekleşmiş
# return/volatility değerleri:
#
# İlk test satırı:
#   validation'ın son target realization satırı
#
# Sonraki test satırları:
#   bir önceki test target realization satırı

previous_observed_raw = np.vstack(
    [
        y_val_raw[
            -1,
            :
        ],
        y_true_raw[
            :-1,
            :
        ]
    ]
)


if (
    previous_observed_raw.shape
    != expected_shape
):
    raise RuntimeError(
        "previous_observed_raw shape yanlış."
    )


y_true_return = y_true_raw[
    :,
    :4
]

y_true_vol = y_true_raw[
    :,
    4:
]

final_return_pred = final_pred_raw[
    :,
    :4
]

final_vol_pred = final_pred_raw[
    :,
    4:
]


return_zero_pred = np.zeros_like(
    y_true_return
)

return_persistence_pred = previous_observed_raw[
    :,
    :4
].copy()

vol_persistence_pred = previous_observed_raw[
    :,
    4:
].copy()


for name, arr in {
    "return_zero_pred":
        return_zero_pred,

    "return_persistence_pred":
        return_persistence_pred,

    "vol_persistence_pred":
        vol_persistence_pred,
}.items():
    if arr.shape != (
        EXPECTED_TEST_SAMPLES,
        4
    ):
        raise RuntimeError(
            f"{name} shape yanlış: {arr.shape}"
        )

    if not np.isfinite(
        arr
    ).all():
        raise RuntimeError(
            f"{name} içinde NaN/Inf var."
        )


print(
    "\n[NAIVE PREDICTIONS]"
)

print(
    "ReturnZero shape        :",
    return_zero_pred.shape
)

print(
    "ReturnPersistence shape :",
    return_persistence_pred.shape
)

print(
    "VolPersistence shape    :",
    vol_persistence_pred.shape
)


# ==========================================================
# 9. TAHMİNLERİ KAYDET
# ==========================================================

np.save(
    RETURN_ZERO_PRED_PATH,
    return_zero_pred
)

np.save(
    RETURN_PERSISTENCE_PRED_PATH,
    return_persistence_pred
)

np.save(
    VOL_PERSISTENCE_PRED_PATH,
    vol_persistence_pred
)


# ==========================================================
# 10. METRİKLERİ HESAPLA
# ==========================================================

metric_rows = []


metric_rows.extend(
    build_return_metric_rows(
        model_label=
            "ReturnZero",

        y_true_return=
            y_true_return,

        y_pred_return=
            return_zero_pred
    )
)


metric_rows.extend(
    build_return_metric_rows(
        model_label=
            "ReturnPersistence",

        y_true_return=
            y_true_return,

        y_pred_return=
            return_persistence_pred
    )
)


metric_rows.extend(
    build_return_metric_rows(
        model_label=
            EXPECTED_PRIMARY_LABEL,

        y_true_return=
            y_true_return,

        y_pred_return=
            final_return_pred
    )
)


metric_rows.extend(
    build_vol_metric_rows(
        model_label=
            "VolPersistence",

        y_true_vol=
            y_true_vol,

        y_pred_vol=
            vol_persistence_pred
    )
)


metric_rows.extend(
    build_vol_metric_rows(
        model_label=
            EXPECTED_PRIMARY_LABEL,

        y_true_vol=
            y_true_vol,

        y_pred_vol=
            final_vol_pred
    )
)


metrics_df = pd.DataFrame(
    metric_rows
)


if len(
    metrics_df
) != 20:
    raise RuntimeError(
        f"Metrics satır sayısı 20 değil: {len(metrics_df)}"
    )


# ==========================================================
# 11. 07 FINAL METRİKLERİYLE BAĞIMSIZ EŞLEŞME
# ==========================================================

final_metrics_07 = pd.read_csv(
    FINAL_METRICS_CSV
)


final_primary_07 = final_metrics_07[
    final_metrics_07[
        "model_label"
    ]
    == EXPECTED_PRIMARY_LABEL
].copy()


if len(
    final_primary_07
) != 8:
    raise RuntimeError(
        "07 final primary metrics exact 8 satır değil."
    )


final_primary_08a = metrics_df[
    metrics_df[
        "model_label"
    ]
    == EXPECTED_PRIMARY_LABEL
].copy()


key_cols = [
    "task",
    "asset"
]


m07 = final_primary_07.set_index(
    key_cols
).sort_index()

m08 = final_primary_08a.set_index(
    key_cols
).sort_index()


if list(
    m07.index
) != list(
    m08.index
):
    raise RuntimeError(
        "07 ve 08A primary metric indexleri uyuşmuyor."
    )


metric_cols = [
    "MAE",
    "RMSE",
    "R2",
    "PinballLoss_tau_0.5"
]


max_final_metric_crosscheck_diff = 0.0


for idx in m08.index:
    for col in metric_cols:
        value_07 = m07.loc[
            idx,
            col
        ]

        value_08 = m08.loc[
            idx,
            col
        ]

        if (
            pd.isna(
                value_07
            )
            and pd.isna(
                value_08
            )
        ):
            continue

        diff = abs(
            float(
                value_07
            )
            - float(
                value_08
            )
        )

        max_final_metric_crosscheck_diff = max(
            max_final_metric_crosscheck_diff,
            diff
        )

        assert_close(
            value_08,
            value_07,
            f"07↔08A final metric {idx} {col}"
        )


# ==========================================================
# 12. FINAL VS BASELINE KARŞILAŞTIRMA TABLOSU
# ==========================================================

comparison_rows = []


for asset in EXPECTED_ASSETS:

    final_row = metrics_df[
        (
            metrics_df[
                "model_label"
            ]
            == EXPECTED_PRIMARY_LABEL
        )
        &
        (
            metrics_df[
                "task"
            ]
            == "return"
        )
        &
        (
            metrics_df[
                "asset"
            ]
            == asset
        )
    ].iloc[
        0
    ]


    for baseline_model in [
        "ReturnZero",
        "ReturnPersistence",
    ]:
        baseline_row = metrics_df[
            (
                metrics_df[
                    "model_label"
                ]
                == baseline_model
            )
            &
            (
                metrics_df[
                    "task"
                ]
                == "return"
            )
            &
            (
                metrics_df[
                    "asset"
                ]
                == asset
            )
        ].iloc[
            0
        ]


        final_error = float(
            final_row[
                "MAE"
            ]
        )

        baseline_error = float(
            baseline_row[
                "MAE"
            ]
        )

        ratio = (
            final_error
            / baseline_error
        )


        comparison_rows.append({
            "task":
                "return",

            "asset":
                asset,

            "primary_metric":
                "MAE",

            "final_model":
                EXPECTED_PRIMARY_LABEL,

            "baseline_model":
                baseline_model,

            "final_error":
                final_error,

            "baseline_error":
                baseline_error,

            "final_minus_baseline":
                final_error
                - baseline_error,

            "final_to_baseline_ratio":
                ratio,

            "final_beats_baseline":
                bool(
                    ratio < 1.0
                ),

            "warning_gt_1_30":
                bool(
                    ratio
                    > DIAGNOSTIC_WARNING_THRESHOLD
                ),

            "severe_gt_1_50":
                bool(
                    ratio
                    > DIAGNOSTIC_SEVERE_THRESHOLD
                ),
        })


for asset in EXPECTED_ASSETS:

    final_row = metrics_df[
        (
            metrics_df[
                "model_label"
            ]
            == EXPECTED_PRIMARY_LABEL
        )
        &
        (
            metrics_df[
                "task"
            ]
            == "volatility"
        )
        &
        (
            metrics_df[
                "asset"
            ]
            == asset
        )
    ].iloc[
        0
    ]


    baseline_row = metrics_df[
        (
            metrics_df[
                "model_label"
            ]
            == "VolPersistence"
        )
        &
        (
            metrics_df[
                "task"
            ]
            == "volatility"
        )
        &
        (
            metrics_df[
                "asset"
            ]
            == asset
        )
    ].iloc[
        0
    ]


    final_error = float(
        final_row[
            "PinballLoss_tau_0.5"
        ]
    )

    baseline_error = float(
        baseline_row[
            "PinballLoss_tau_0.5"
        ]
    )

    ratio = (
        final_error
        / baseline_error
    )


    comparison_rows.append({
        "task":
            "volatility",

        "asset":
            asset,

        "primary_metric":
            "PinballLoss_tau_0.5",

        "final_model":
            EXPECTED_PRIMARY_LABEL,

        "baseline_model":
            "VolPersistence",

        "final_error":
            final_error,

        "baseline_error":
            baseline_error,

        "final_minus_baseline":
            final_error
            - baseline_error,

        "final_to_baseline_ratio":
            ratio,

        "final_beats_baseline":
            bool(
                ratio < 1.0
            ),

        "warning_gt_1_30":
            bool(
                ratio
                > DIAGNOSTIC_WARNING_THRESHOLD
            ),

        "severe_gt_1_50":
            bool(
                ratio
                > DIAGNOSTIC_SEVERE_THRESHOLD
            ),
    })


comparison_df = pd.DataFrame(
    comparison_rows
)


if len(
    comparison_df
) != 12:
    raise RuntimeError(
        f"Comparison satır sayısı 12 değil: {len(comparison_df)}"
    )


# ==========================================================
# 13. KARŞILAŞTIRMA ÖZETİ
# ==========================================================

summary_rows = []


for (
    task,
    baseline_model
), group in comparison_df.groupby(
    [
        "task",
        "baseline_model"
    ],
    sort=False
):
    ratios = group[
        "final_to_baseline_ratio"
    ].astype(
        float
    )


    summary_rows.append({
        "task":
            task,

        "baseline_model":
            baseline_model,

        "n_assets":
            int(
                len(
                    group
                )
            ),

        "mean_final_to_baseline_ratio":
            float(
                ratios.mean()
            ),

        "median_final_to_baseline_ratio":
            float(
                ratios.median()
            ),

        "min_final_to_baseline_ratio":
            float(
                ratios.min()
            ),

        "max_final_to_baseline_ratio":
            float(
                ratios.max()
            ),

        "n_final_beats_baseline":
            int(
                group[
                    "final_beats_baseline"
                ].sum()
            ),

        "n_warning_gt_1_30":
            int(
                group[
                    "warning_gt_1_30"
                ].sum()
            ),

        "n_severe_gt_1_50":
            int(
                group[
                    "severe_gt_1_50"
                ].sum()
            ),
    })


comparison_summary_df = pd.DataFrame(
    summary_rows
)


# ==========================================================
# 14. KİLİTLİ STRONG-NAIVE ORANLARI
# ==========================================================

return_vs_zero = comparison_df[
    (
        comparison_df[
            "task"
        ]
        == "return"
    )
    &
    (
        comparison_df[
            "baseline_model"
        ]
        == "ReturnZero"
    )
].copy()


return_vs_persistence = comparison_df[
    (
        comparison_df[
            "task"
        ]
        == "return"
    )
    &
    (
        comparison_df[
            "baseline_model"
        ]
        == "ReturnPersistence"
    )
].copy()


vol_vs_persistence = comparison_df[
    (
        comparison_df[
            "task"
        ]
        == "volatility"
    )
    &
    (
        comparison_df[
            "baseline_model"
        ]
        == "VolPersistence"
    )
].copy()


avg_return_ratio_vs_zero = float(
    return_vs_zero[
        "final_to_baseline_ratio"
    ].mean()
)


avg_return_ratio_vs_persistence = float(
    return_vs_persistence[
        "final_to_baseline_ratio"
    ].mean()
)


avg_vol_ratio_vs_persistence = float(
    vol_vs_persistence[
        "final_to_baseline_ratio"
    ].mean()
)


# Bu skor model seçimi için kullanılmaz.
# Sadece 05/06'da kullanılan baseline-normalize ölçeğin
# test dönemindeki betimleyici karşılığıdır.

test_strong_naive_diagnostic_score = float(
    0.5
    * avg_return_ratio_vs_zero
    + 0.5
    * avg_vol_ratio_vs_persistence
)


# ==========================================================
# 15. DM İÇİN LOSS SERIES KAYDI
# ==========================================================

loss_series = {}


for i, asset in enumerate(
    EXPECTED_ASSETS
):
    loss_series[
        f"final_return_abs__{asset}"
    ] = np.abs(
        y_true_return[
            :,
            i
        ]
        - final_return_pred[
            :,
            i
        ]
    )

    loss_series[
        f"return_zero_abs__{asset}"
    ] = np.abs(
        y_true_return[
            :,
            i
        ]
        - return_zero_pred[
            :,
            i
        ]
    )

    loss_series[
        f"return_persistence_abs__{asset}"
    ] = np.abs(
        y_true_return[
            :,
            i
        ]
        - return_persistence_pred[
            :,
            i
        ]
    )

    loss_series[
        f"final_vol_pinball__{asset}"
    ] = pinball_loss_series_np(
        y_true_vol[
            :,
            i
        ],
        final_vol_pred[
            :,
            i
        ],
        tau=TAU
    )

    loss_series[
        f"vol_persistence_pinball__{asset}"
    ] = pinball_loss_series_np(
        y_true_vol[
            :,
            i
        ],
        vol_persistence_pred[
            :,
            i
        ],
        tau=TAU
    )


np.savez(
    LOSS_SERIES_PATH,
    **loss_series
)


# ==========================================================
# 16. ÇIKTILARI KAYDET
# ==========================================================

metrics_df.to_csv(
    METRICS_LONG_CSV,
    index=False
)

comparison_df.to_csv(
    COMPARISON_CSV,
    index=False
)

comparison_summary_df.to_csv(
    COMPARISON_SUMMARY_CSV,
    index=False
)


# ==========================================================
# 17. JSON PROVENANCE
# ==========================================================

summary_payload = {
    "project_version":
        "v4_repro",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "script":
        "08A_naive_baselines_test_v4.py",

    "purpose":
        (
            "Test-period naive baseline comparison for the locked "
            "final three-seed ensemble."
        ),

    "model_selection_inside_08A":
        False,

    "hyperparameter_change_inside_08A":
        False,

    "retraining_inside_08A":
        False,

    "final_model_changed":
        False,

    "winner_config_id":
        EXPECTED_WINNER_CONFIG_ID,

    "primary_final_prediction":
        EXPECTED_PRIMARY_LABEL,

    "primary_test_policy":
        EXPECTED_PRIMARY_POLICY,

    "expected_seeds":
        EXPECTED_SEEDS,

    "test_sample_count":
        EXPECTED_TEST_SAMPLES,

    "asset_order":
        EXPECTED_ASSETS,

    "feature_set":
        FEATURE_SET,

    "lookback":
        LOOKBACK,

    "baseline_definitions": {
        "ReturnZero":
            "y_hat_ret(t+1) = 0",

        "ReturnPersistence":
            "y_hat_ret(t+1) = LogRet(t)",

        "VolPersistence":
            "y_hat_vol(t+1) = Vol20(t)",
    },

    "persistence_construction": {
        "first_test_row":
            "previous observed state = last validation target-realization row",

        "subsequent_test_rows":
            "previous observed state = immediately preceding test target-realization row",

        "formula":
            "previous_observed_raw = vstack([y_val_raw[-1], y_test_raw[:-1]])",
    },

    "primary_metrics": {
        "return":
            "MAE",

        "volatility":
            "PinballLoss_tau_0.5",
    },

    "diagnostic_thresholds": {
        "warning_gt":
            DIAGNOSTIC_WARNING_THRESHOLD,

        "severe_gt":
            DIAGNOSTIC_SEVERE_THRESHOLD,

        "status":
            (
                "study-specific pre-locked diagnostic reporting thresholds; "
                "not universal standards and not model-selection thresholds"
            ),
    },

    "strong_naive_test_ratios": {
        "avg_return_ratio_vs_ReturnZero":
            avg_return_ratio_vs_zero,

        "avg_return_ratio_vs_ReturnPersistence":
            avg_return_ratio_vs_persistence,

        "avg_vol_ratio_vs_VolPersistence":
            avg_vol_ratio_vs_persistence,

        "test_strong_naive_diagnostic_score":
            test_strong_naive_diagnostic_score,

        "diagnostic_score_note":
            (
                "Descriptive only. It is not used for model selection, "
                "hyperparameter tuning, or any post-hoc decision."
            ),
    },

    "crosschecks": {
        "07_script_sha256_verified":
            True,

        "07_script_sha256":
            actual_07_sha,

        "07_final_metrics_max_crosscheck_diff":
            max_final_metric_crosscheck_diff,

        "07_y_true_vs_sequence_y_test_max_diff":
            max_test_truth_diff,
    },

    "outputs": {
        "metrics_long_csv":
            METRICS_LONG_CSV,

        "comparison_csv":
            COMPARISON_CSV,

        "comparison_summary_csv":
            COMPARISON_SUMMARY_CSV,

        "summary_json":
            SUMMARY_JSON,

        "return_zero_prediction_npy":
            RETURN_ZERO_PRED_PATH,

        "return_persistence_prediction_npy":
            RETURN_PERSISTENCE_PRED_PATH,

        "vol_persistence_prediction_npy":
            VOL_PERSISTENCE_PRED_PATH,

        "loss_series_npz":
            LOSS_SERIES_PATH,
    },

    "notes": [
        (
            "No model selection is performed in 08A."
        ),
        (
            "The final model and the primary ensemble policy are inherited unchanged from 07."
        ),
        (
            "Return ratios use MAE; volatility ratios use PinballLoss at tau=0.5."
        ),
        (
            "The 1.30 and 1.50 thresholds are descriptive study-specific flags only."
        ),
        (
            "Statistical significance is not inferred in 08A; DM + Harvey + Holm is reserved for stage 09."
        ),
    ],
}


with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary_payload,
        f,
        ensure_ascii=False,
        indent=2
    )


# ==========================================================
# 18. SON KONTROLLER
# ==========================================================

if (
    metrics_df[
        [
            "model_label",
            "task",
            "asset"
        ]
    ].duplicated().any()
):
    raise RuntimeError(
        "Metrics tablosunda duplicate model-task-asset var."
    )


if not (
    len(
        return_vs_zero
    )
    == 4
    and len(
        return_vs_persistence
    )
    == 4
    and len(
        vol_vs_persistence
    )
    == 4
):
    raise RuntimeError(
        "Comparison gruplarında 4 varlık bütünlüğü bozuk."
    )


# ==========================================================
# 19. FINAL ÇIKTI
# ==========================================================

print(
    "\n"
    + "=" * 110
)

print(
    "08A_naive_baselines_test_v4.py TAMAMLANDI"
)

print(
    "=" * 110
)


print(
    "\nFINAL MODEL:"
)

print(
    EXPECTED_WINNER_CONFIG_ID
)


print(
    "\nNAIVE BASELINE METRICS:"
)

print(
    metrics_df.to_string(
        index=False
    )
)


print(
    "\nFINAL VS NAIVE BASELINES:"
)

print(
    comparison_df[
        [
            "task",
            "asset",
            "primary_metric",
            "baseline_model",
            "final_error",
            "baseline_error",
            "final_to_baseline_ratio",
            "final_beats_baseline",
            "warning_gt_1_30",
            "severe_gt_1_50",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nCOMPARISON SUMMARY:"
)

print(
    comparison_summary_df.to_string(
        index=False
    )
)


print(
    "\nSTRONG-NAIVE TEST RATIOS:"
)

print(
    f"Avg Return Ratio vs ReturnZero        : "
    f"{avg_return_ratio_vs_zero:.16f}"
)

print(
    f"Avg Return Ratio vs ReturnPersistence : "
    f"{avg_return_ratio_vs_persistence:.16f}"
)

print(
    f"Avg Vol Ratio vs VolPersistence       : "
    f"{avg_vol_ratio_vs_persistence:.16f}"
)

print(
    f"Test Strong-Naive Diagnostic Score    : "
    f"{test_strong_naive_diagnostic_score:.16f}"
)


print(
    "\nRULE CHECK:"
)

print(
    "✅ 07 final model değiştirilmedi."
)

print(
    "✅ 07 primary 3-seed ensemble değiştirilmedi."
)

print(
    "✅ 08A içinde model seçimi yapılmadı."
)

print(
    "✅ 08A içinde hyperparameter değişmedi."
)

print(
    "✅ 08A içinde yeniden eğitim yapılmadı."
)

print(
    "✅ ReturnZero test tahmini üretildi."
)

print(
    "✅ ReturnPersistence test tahmini üretildi."
)

print(
    "✅ VolPersistence test tahmini üretildi."
)

print(
    "✅ 07 final metrikleri 08A'da bağımsız yeniden hesaplanıp eşleştirildi."
)

print(
    "✅ 09 için loss series kaydedildi."
)

print(
    "✅ İstatistiksel anlamlılık iddiası üretilmedi."
)


print(
    "\nNUMERIC CROSSCHECK:"
)

print(
    "07 final metric max diff:",
    f"{max_final_metric_crosscheck_diff:.16e}"
)

print(
    "07 truth vs sequence test max diff:",
    f"{max_test_truth_diff:.16e}"
)


print(
    "\nOUTPUTS:"
)

for path in [
    METRICS_LONG_CSV,
    COMPARISON_CSV,
    COMPARISON_SUMMARY_CSV,
    SUMMARY_JSON,
    RETURN_ZERO_PRED_PATH,
    RETURN_PERSISTENCE_PRED_PATH,
    VOL_PERSISTENCE_PRED_PATH,
    LOSS_SERIES_PATH,
]:
    print(
        " -",
        path
    )


print(
    "\n"
    + "=" * 110
)

print(
    "SON HÜKÜM: 08A NAİF BASELINE KARŞILAŞTIRMASI TAMAMLANDI."
)

print(
    "=" * 110
)
