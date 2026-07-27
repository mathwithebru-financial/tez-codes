
# ==========================================================
# 03_baseline_sanity_v4.py
#
# AMAÇ:
# - Validation setinde naive finansal baseline'ları hesaplamak
# - ReturnZero
# - ReturnPersistence
# - VolPersistence
# - ValidationScore için denominator'ları üretmek
# - Sequence yapısını sanity-check etmek
#
# ÖNEMLİ:
# - Test performansı HESAPLANMAZ.
# - Test seti model seçimi için KULLANILMAZ.
# ==========================================================

import os
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================================
# 1. YOLLAR
# ==========================================================

BASE_DIR = "/content/drive/MyDrive/tez_transformer_v4_repro"

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

SEQUENCE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "sequences"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "baselines"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


FEATURES_BASELINE_PATH = os.path.join(
    PROCESSED_DIR,
    "features_baseline.csv"
)

TARGETS_PATH = os.path.join(
    PROCESSED_DIR,
    "targets_all.csv"
)

TARGET_DATES_PATH = os.path.join(
    PROCESSED_DIR,
    "target_realization_dates.csv"
)

SPLIT_META_PATH = os.path.join(
    PROCESSED_DIR,
    "split_meta_v4.json"
)


required_files = [
    FEATURES_BASELINE_PATH,
    TARGETS_PATH,
    TARGET_DATES_PATH,
    SPLIT_META_PATH
]


for path in required_files:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Gerekli dosya bulunamadı:\n{path}"
        )


# ==========================================================
# 2. KİLİTLİ SABİTLER
# ==========================================================

ASSETS = [
    "BIST100",
    "USDTRY",
    "EURTRY",
    "GOLD"
]


RETURN_TARGETS = [
    "BIST100_NextRet",
    "USDTRY_NextRet",
    "EURTRY_NextRet",
    "GOLD_NextRet"
]


VOL_TARGETS = [
    "BIST100_NextVol",
    "USDTRY_NextVol",
    "EURTRY_NextVol",
    "GOLD_NextVol"
]


LOOKBACKS = [
    10,
    20,
    30,
    60
]


TAU = 0.5


# ==========================================================
# 3. METRİK FONKSİYONLARI
# ==========================================================

def rmse(
    y_true,
    y_pred
):

    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )
    )


def pinball_loss(
    y_true,
    y_pred,
    tau=0.5
):

    error = (
        y_true - y_pred
    )

    loss = np.maximum(
        tau * error,
        (tau - 1.0) * error
    )

    return float(
        np.mean(loss)
    )


def safe_r2(
    y_true,
    y_pred
):

    try:

        return float(
            r2_score(
                y_true,
                y_pred
            )
        )

    except Exception:

        return float("nan")


# ==========================================================
# 4. VERİLERİ OKU
# ==========================================================

features_baseline = pd.read_csv(
    FEATURES_BASELINE_PATH,
    index_col=0,
    parse_dates=True
)


targets_all = pd.read_csv(
    TARGETS_PATH,
    index_col=0,
    parse_dates=True
)


target_dates = pd.read_csv(
    TARGET_DATES_PATH,
    index_col=0,
    parse_dates=True
)


target_dates[
    "target_realization_date"
] = pd.to_datetime(
    target_dates[
        "target_realization_date"
    ]
)


with open(
    SPLIT_META_PATH,
    "r",
    encoding="utf-8"
) as f:

    split_meta = json.load(f)


print("=" * 80)
print("03 — VALIDATION BASELINE SANITY")
print("=" * 80)

print(
    "\nBaseline feature shape:",
    features_baseline.shape
)

print(
    "Target shape:",
    targets_all.shape
)


# ==========================================================
# 5. SPLIT SINIRLARINI META'DAN OKU
# ==========================================================

train_start = int(
    split_meta[
        "train"
    ][
        "start_idx"
    ]
)

train_end = int(
    split_meta[
        "train"
    ][
        "end_idx_exclusive"
    ]
)


val_start = int(
    split_meta[
        "validation"
    ][
        "start_idx"
    ]
)

val_end = int(
    split_meta[
        "validation"
    ][
        "end_idx_exclusive"
    ]
)


test_start = int(
    split_meta[
        "test"
    ][
        "start_idx"
    ]
)

test_end = int(
    split_meta[
        "test"
    ][
        "end_idx_exclusive"
    ]
)


if not (
    train_end == val_start
    and
    val_end == test_start
    and
    test_end == len(
        targets_all
    )
):

    raise RuntimeError(
        "Split index sınırlarında tutarsızlık var."
    )


# ==========================================================
# 6. SADECE VALIDATION VERİSİNİ AYIR
# ==========================================================

X_val = (
    features_baseline
    .iloc[
        val_start:val_end
    ]
    .copy()
)


y_val = (
    targets_all
    .iloc[
        val_start:val_end
    ]
    .copy()
)


dates_val = (
    target_dates
    .iloc[
        val_start:val_end
    ]
    .copy()
)


if not (
    X_val.index.equals(
        y_val.index
    )
    and
    y_val.index.equals(
        dates_val.index
    )
):

    raise RuntimeError(
        "Validation feature/target/date index hizası bozuk."
    )


print("\nVALIDATION")

print(
    "Örnek sayısı:",
    len(
        y_val
    )
)

print(
    "Anchor:",
    y_val.index.min().date(),
    "→",
    y_val.index.max().date()
)

print(
    "Target realization:",
    dates_val[
        "target_realization_date"
    ].min().date(),
    "→",
    dates_val[
        "target_realization_date"
    ].max().date()
)


# ==========================================================
# 7. NAIVE BASELINE TAHMİNLERİ
#
# ReturnZero:
#   Next return tahmini = 0
#
# ReturnPersistence:
#   Next return tahmini = mevcut LogRet[t]
#
# VolPersistence:
#   Next volatility tahmini = mevcut Vol20[t]
# ==========================================================

results = []


selection_denominators = {

    "project_version":
        "v4_repro",

    "created_at":
        datetime.now().isoformat(),

    "split":
        "validation_only",

    "validation_rows":
        int(
            len(
                y_val
            )
        ),

    "validation_anchor_start":
        str(
            y_val.index.min().date()
        ),

    "validation_anchor_end":
        str(
            y_val.index.max().date()
        ),

    "validation_target_realization_start":
        str(
            dates_val[
                "target_realization_date"
            ].min().date()
        ),

    "validation_target_realization_end":
        str(
            dates_val[
                "target_realization_date"
            ].max().date()
        ),

    "return_denominator":
        {},

    "volatility_denominator":
        {},

    "selection_score_rule":
        (
            "ValidationScore = "
            "0.5 * AvgReturnRatio + "
            "0.5 * AvgVolRatio"
        ),

    "return_ratio_rule":
        (
            "Model return MAE / "
            "ReturnZero validation MAE"
        ),

    "volatility_ratio_rule":
        (
            "Model volatility PinballLoss(tau=0.5) / "
            "VolPersistence validation PinballLoss(tau=0.5)"
        ),

    "test_used":
        False
}


for i, asset in enumerate(
    ASSETS
):

    # ------------------------------------------------------
    # GERÇEK TARGET'LAR
    # ------------------------------------------------------

    y_true_ret = (
        y_val[
            RETURN_TARGETS[i]
        ]
        .values
        .astype(float)
    )


    y_true_vol = (
        y_val[
            VOL_TARGETS[i]
        ]
        .values
        .astype(float)
    )


    # ------------------------------------------------------
    # CURRENT FEATURES
    # ------------------------------------------------------

    current_ret = (
        X_val[
            f"{asset}_LogRet"
        ]
        .values
        .astype(float)
    )


    current_vol = (
        X_val[
            f"{asset}_Vol20"
        ]
        .values
        .astype(float)
    )


    # ------------------------------------------------------
    # RETURN ZERO
    # ------------------------------------------------------

    pred_return_zero = np.zeros_like(
        y_true_ret
    )


    rz_mae = float(
        mean_absolute_error(
            y_true_ret,
            pred_return_zero
        )
    )


    rz_rmse = rmse(
        y_true_ret,
        pred_return_zero
    )


    rz_r2 = safe_r2(
        y_true_ret,
        pred_return_zero
    )


    results.append(
        {
            "asset":
                asset,

            "task":
                "return",

            "baseline":
                "ReturnZero",

            "MAE":
                rz_mae,

            "RMSE":
                rz_rmse,

            "R2":
                rz_r2,

            "PinballLoss_tau_0.5":
                np.nan
        }
    )


    # ------------------------------------------------------
    # RETURN PERSISTENCE
    # ------------------------------------------------------

    rp_mae = float(
        mean_absolute_error(
            y_true_ret,
            current_ret
        )
    )


    rp_rmse = rmse(
        y_true_ret,
        current_ret
    )


    rp_r2 = safe_r2(
        y_true_ret,
        current_ret
    )


    results.append(
        {
            "asset":
                asset,

            "task":
                "return",

            "baseline":
                "ReturnPersistence",

            "MAE":
                rp_mae,

            "RMSE":
                rp_rmse,

            "R2":
                rp_r2,

            "PinballLoss_tau_0.5":
                np.nan
        }
    )


    # ------------------------------------------------------
    # VOL PERSISTENCE
    # ------------------------------------------------------

    vp_mae = float(
        mean_absolute_error(
            y_true_vol,
            current_vol
        )
    )


    vp_rmse = rmse(
        y_true_vol,
        current_vol
    )


    vp_r2 = safe_r2(
        y_true_vol,
        current_vol
    )


    vp_pinball = pinball_loss(
        y_true_vol,
        current_vol,
        tau=TAU
    )


    results.append(
        {
            "asset":
                asset,

            "task":
                "volatility",

            "baseline":
                "VolPersistence",

            "MAE":
                vp_mae,

            "RMSE":
                vp_rmse,

            "R2":
                vp_r2,

            "PinballLoss_tau_0.5":
                vp_pinball
        }
    )


    # ------------------------------------------------------
    # SELECTION DENOMINATOR'LARI
    # ------------------------------------------------------

    selection_denominators[
        "return_denominator"
    ][asset] = {

        "baseline":
            "ReturnZero",

        "metric":
            "MAE",

        "value":
            rz_mae
    }


    selection_denominators[
        "volatility_denominator"
    ][asset] = {

        "baseline":
            "VolPersistence",

        "metric":
            "PinballLoss_tau_0.5",

        "tau":
            TAU,

        "value":
            vp_pinball
    }


# ==========================================================
# 8. SONUÇ DATAFRAME
# ==========================================================

results_df = pd.DataFrame(
    results
)


results_path = os.path.join(
    RESULTS_DIR,
    "validation_naive_baselines_v4.csv"
)


results_df.to_csv(
    results_path,
    index=False
)


print("\n" + "=" * 80)
print("VALIDATION NAIVE BASELINE SONUÇLARI")
print("=" * 80)

print(
    results_df.to_string(
        index=False
    )
)


# ==========================================================
# 9. DENOMINATOR JSON
# ==========================================================

denominator_path = os.path.join(
    PROCESSED_DIR,
    "selection_baseline_denominators_v4.json"
)


with open(
    denominator_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        selection_denominators,
        f,
        ensure_ascii=False,
        indent=2
    )


print("\nDenominator dosyası:")
print(
    denominator_path
)


# ==========================================================
# 10. SEQUENCE SANITY CHECKS
#
# Bu bölüm test performansı HESAPLAMAZ.
# Yalnızca sequence shape/date yapısını kontrol eder.
# ==========================================================

sequence_records = []


for feature_set in [
    "baseline",
    "full"
]:

    expected_dim = (
        8
        if feature_set == "baseline"
        else 28
    )


    for lookback in LOOKBACKS:

        lb_dir = os.path.join(
            SEQUENCE_DIR,
            feature_set,
            f"lb{lookback}"
        )


        required_sequence_files = [

            "X_train.npy",
            "y_train.npy",

            "X_val.npy",
            "y_val.npy",

            "X_test.npy",
            "y_test.npy",

            "anchor_dates_train.npy",
            "target_realization_dates_train.npy",

            "anchor_dates_val.npy",
            "target_realization_dates_val.npy",

            "anchor_dates_test.npy",
            "target_realization_dates_test.npy"
        ]


        for filename in required_sequence_files:

            path = os.path.join(
                lb_dir,
                filename
            )

            if not os.path.exists(path):

                raise FileNotFoundError(
                    f"Sequence dosyası eksik:\n{path}"
                )


        # Memory-map:
        # dosya yapısı okunur, test performansı hesaplanmaz.

        X_train_seq = np.load(
            os.path.join(
                lb_dir,
                "X_train.npy"
            ),
            mmap_mode="r"
        )


        y_train_seq = np.load(
            os.path.join(
                lb_dir,
                "y_train.npy"
            ),
            mmap_mode="r"
        )


        X_val_seq = np.load(
            os.path.join(
                lb_dir,
                "X_val.npy"
            ),
            mmap_mode="r"
        )


        y_val_seq = np.load(
            os.path.join(
                lb_dir,
                "y_val.npy"
            ),
            mmap_mode="r"
        )


        X_test_seq = np.load(
            os.path.join(
                lb_dir,
                "X_test.npy"
            ),
            mmap_mode="r"
        )


        y_test_seq = np.load(
            os.path.join(
                lb_dir,
                "y_test.npy"
            ),
            mmap_mode="r"
        )


        train_target_dates_seq = np.load(
            os.path.join(
                lb_dir,
                "target_realization_dates_train.npy"
            )
        )


        val_target_dates_seq = np.load(
            os.path.join(
                lb_dir,
                "target_realization_dates_val.npy"
            )
        )


        test_target_dates_seq = np.load(
            os.path.join(
                lb_dir,
                "target_realization_dates_test.npy"
            )
        )


        # --------------------------------------------------
        # SHAPE KONTROLÜ
        # --------------------------------------------------

        if X_train_seq.shape[1:] != (
            lookback,
            expected_dim
        ):

            raise RuntimeError(
                f"X_train shape yanlış: "
                f"{feature_set}, lb={lookback}, "
                f"{X_train_seq.shape}"
            )


        if X_val_seq.shape != (
            584,
            lookback,
            expected_dim
        ):

            raise RuntimeError(
                f"X_val shape yanlış: "
                f"{feature_set}, lb={lookback}, "
                f"{X_val_seq.shape}"
            )


        if X_test_seq.shape != (
            584,
            lookback,
            expected_dim
        ):

            raise RuntimeError(
                f"X_test shape yanlış: "
                f"{feature_set}, lb={lookback}, "
                f"{X_test_seq.shape}"
            )


        if y_val_seq.shape != (
            584,
            8
        ):

            raise RuntimeError(
                f"y_val shape yanlış: "
                f"{feature_set}, lb={lookback}, "
                f"{y_val_seq.shape}"
            )


        if y_test_seq.shape != (
            584,
            8
        ):

            raise RuntimeError(
                f"y_test shape yanlış: "
                f"{feature_set}, lb={lookback}, "
                f"{y_test_seq.shape}"
            )


        # --------------------------------------------------
        # TARGET TARİH AYRIŞMASI
        # --------------------------------------------------

        train_target_dt = pd.to_datetime(
            train_target_dates_seq
        )


        val_target_dt = pd.to_datetime(
            val_target_dates_seq
        )


        test_target_dt = pd.to_datetime(
            test_target_dates_seq
        )


        train_val_disjoint = bool(
            train_target_dt.max()
            <
            val_target_dt.min()
        )


        val_test_disjoint = bool(
            val_target_dt.max()
            <
            test_target_dt.min()
        )


        if not train_val_disjoint:

            raise RuntimeError(
                "Train/validation target dates ayrık değil."
            )


        if not val_test_disjoint:

            raise RuntimeError(
                "Validation/test target dates ayrık değil."
            )


        sequence_records.append(
            {
                "feature_set":
                    feature_set,

                "lookback":
                    lookback,

                "X_train_shape":
                    str(
                        tuple(
                            X_train_seq.shape
                        )
                    ),

                "y_train_shape":
                    str(
                        tuple(
                            y_train_seq.shape
                        )
                    ),

                "X_val_shape":
                    str(
                        tuple(
                            X_val_seq.shape
                        )
                    ),

                "y_val_shape":
                    str(
                        tuple(
                            y_val_seq.shape
                        )
                    ),

                "X_test_shape":
                    str(
                        tuple(
                            X_test_seq.shape
                        )
                    ),

                "y_test_shape":
                    str(
                        tuple(
                            y_test_seq.shape
                        )
                    ),

                "validation_window_loss":
                    int(
                        584 - len(
                            X_val_seq
                        )
                    ),

                "test_window_loss":
                    int(
                        584 - len(
                            X_test_seq
                        )
                    ),

                "train_val_target_dates_disjoint":
                    train_val_disjoint,

                "val_test_target_dates_disjoint":
                    val_test_disjoint,

                "test_metrics_computed":
                    False
            }
        )


sequence_sanity_df = pd.DataFrame(
    sequence_records
)


sequence_sanity_path = os.path.join(
    RESULTS_DIR,
    "sequence_sanity_checks_v4.csv"
)


sequence_sanity_df.to_csv(
    sequence_sanity_path,
    index=False
)


print("\n" + "=" * 80)
print("SEQUENCE SANITY CHECKS")
print("=" * 80)

print(
    sequence_sanity_df.to_string(
        index=False
    )
)


# ==========================================================
# 11. ÖZET JSON
# ==========================================================

summary = {

    "project_version":
        "v4_repro",

    "created_at":
        datetime.now().isoformat(),

    "script":
        "03_baseline_sanity_v4.py",

    "validation_only_metrics":
        True,

    "test_metrics_computed":
        False,

    "validation_rows":
        int(
            len(
                y_val
            )
        ),

    "baselines": [
        "ReturnZero",
        "ReturnPersistence",
        "VolPersistence"
    ],

    "selection_denominators": {

        "return":
            "ReturnZero MAE",

        "volatility":
            "VolPersistence PinballLoss tau=0.5"
    },

    "validation_target_period": {

        "start":
            str(
                dates_val[
                    "target_realization_date"
                ].min().date()
            ),

        "end":
            str(
                dates_val[
                    "target_realization_date"
                ].max().date()
            )
    },

    "sequence_configs_checked":
        int(
            len(
                sequence_sanity_df
            )
        ),

    "all_train_val_target_dates_disjoint":
        bool(
            sequence_sanity_df[
                "train_val_target_dates_disjoint"
            ].all()
        ),

    "all_val_test_target_dates_disjoint":
        bool(
            sequence_sanity_df[
                "val_test_target_dates_disjoint"
            ].all()
        ),

    "all_validation_window_loss_zero":
        bool(
            (
                sequence_sanity_df[
                    "validation_window_loss"
                ]
                == 0
            ).all()
        ),

    "all_test_window_loss_zero":
        bool(
            (
                sequence_sanity_df[
                    "test_window_loss"
                ]
                == 0
            ).all()
        )
}


summary_path = os.path.join(
    RESULTS_DIR,
    "baseline_sanity_summary_v4.json"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2
    )


# ==========================================================
# 12. KRİTİK SON KONTROLLER
# ==========================================================

if len(
    results_df
) != 12:

    raise RuntimeError(
        "Beklenen 12 baseline sonucu üretilemedi."
    )


if not (
    sequence_sanity_df[
        "validation_window_loss"
    ]
    == 0
).all():

    raise RuntimeError(
        "Validation pencere kaybı sıfır değil."
    )


if not (
    sequence_sanity_df[
        "test_window_loss"
    ]
    == 0
).all():

    raise RuntimeError(
        "Test pencere kaybı sıfır değil."
    )


if not sequence_sanity_df[
    "train_val_target_dates_disjoint"
].all():

    raise RuntimeError(
        "Bazı train/validation target tarihleri ayrık değil."
    )


if not sequence_sanity_df[
    "val_test_target_dates_disjoint"
].all():

    raise RuntimeError(
        "Bazı validation/test target tarihleri ayrık değil."
    )


# Denominator'lar pozitif olmalı.

for asset in ASSETS:

    return_value = (
        selection_denominators[
            "return_denominator"
        ][asset][
            "value"
        ]
    )


    vol_value = (
        selection_denominators[
            "volatility_denominator"
        ][asset][
            "value"
        ]
    )


    if return_value <= 0:

        raise RuntimeError(
            f"{asset} ReturnZero denominator pozitif değil."
        )


    if vol_value <= 0:

        raise RuntimeError(
            f"{asset} VolPersistence denominator pozitif değil."
        )


# ==========================================================
# 13. SON ÇIKTI
# ==========================================================

print("\n")
print("=" * 80)
print("03_baseline_sanity_v4.py BAŞARIYLA TAMAMLANDI")
print("=" * 80)

print("\nDosyalar:")

print(
    " -",
    results_path
)

print(
    " -",
    denominator_path
)

print(
    " -",
    sequence_sanity_path
)

print(
    " -",
    summary_path
)


print("\nKURAL KONTROLÜ:")

print(
    "✅ ReturnZero hesaplandı."
)

print(
    "✅ ReturnPersistence hesaplandı."
)

print(
    "✅ VolPersistence hesaplandı."
)

print(
    "✅ ValidationScore denominator'ları kaydedildi."
)

print(
    "✅ 8 sequence konfigürasyonu kontrol edildi."
)

print(
    "✅ Validation pencere kaybı = 0."
)

print(
    "✅ Test pencere kaybı = 0."
)

print(
    "✅ Train/validation target tarihleri ayrık."
)

print(
    "✅ Validation/test target tarihleri ayrık."
)

print(
    "✅ Test performansı hesaplanmadı."
)

print("=" * 80)
