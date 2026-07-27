
# ==========================================================
# 02_preprocessing_v4.py
#
# AMAÇ:
# - 01 çıktıları üzerinden target-realization-aware
#   kronolojik train / validation / test split oluşturmak
# - StandardScaler'ı SADECE train setine fit etmek
# - Validation ve test setlerine SADECE transform uygulamak
# - Lookback = 10, 20, 30, 60 için sequence üretmek
# - Validation/test için geçmiş input penceresini taşımak
# - Target değerlerini hiçbir zaman split dışına taşımamak
#
# BU DOSYADA YOK:
# - Model eğitimi yok
# - Validation skoru yok
# - Test değerlendirmesi yok
# - Model seçimi yok
# ==========================================================

import os
import json
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


# ==========================================================
# 1. ANA YOLLAR
# ==========================================================

BASE_DIR = "/content/drive/MyDrive/tez_transformer_v4_repro"

CONFIG_DIR = os.path.join(
    BASE_DIR,
    "config"
)

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

LOG_DIR = os.path.join(
    BASE_DIR,
    "logs"
)


for path in [
    CONFIG_DIR,
    PROCESSED_DIR,
    SEQUENCE_DIR,
    LOG_DIR
]:
    os.makedirs(
        path,
        exist_ok=True
    )


# ==========================================================
# 2. GEREKLİ DOSYA YOLLARI
# ==========================================================

schema_path = os.path.join(
    CONFIG_DIR,
    "schema_v4.json"
)

features_baseline_path = os.path.join(
    PROCESSED_DIR,
    "features_baseline.csv"
)

features_full_path = os.path.join(
    PROCESSED_DIR,
    "features_full.csv"
)

targets_all_path = os.path.join(
    PROCESSED_DIR,
    "targets_all.csv"
)

target_dates_path = os.path.join(
    PROCESSED_DIR,
    "target_realization_dates.csv"
)


required_files = [
    schema_path,
    features_baseline_path,
    features_full_path,
    targets_all_path,
    target_dates_path
]


for file_path in required_files:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Gerekli dosya bulunamadı:\n{file_path}"
        )


# ==========================================================
# 3. DOSYALARI OKU
# ==========================================================

with open(
    schema_path,
    "r",
    encoding="utf-8"
) as f:

    schema = json.load(f)


features_baseline = pd.read_csv(
    features_baseline_path,
    index_col=0,
    parse_dates=True
)


features_full = pd.read_csv(
    features_full_path,
    index_col=0,
    parse_dates=True
)


targets_all = pd.read_csv(
    targets_all_path,
    index_col=0,
    parse_dates=True
)


target_dates = pd.read_csv(
    target_dates_path,
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


print("=" * 80)
print("02 — GİRDİ DOSYALARI OKUNDU")
print("=" * 80)

print(
    "\nBaseline feature:",
    features_baseline.shape
)

print(
    "Full feature:",
    features_full.shape
)

print(
    "Targets:",
    targets_all.shape
)

print(
    "Target realization dates:",
    target_dates.shape
)


# ==========================================================
# 4. KİLİTLİ KARARLAR
# ==========================================================

LOOKBACKS = schema[
    "sequence"
][
    "lookbacks"
]


TARGET_NAMES = schema[
    "targets"
][
    "definition"
]


BASELINE_DIM = schema[
    "features"
][
    "baseline_dim"
]


FULL_DIM = schema[
    "features"
][
    "full_dim"
]


FEATURE_SETS = {

    "baseline":
        features_baseline,

    "full":
        features_full
}


TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


print("\n" + "=" * 80)
print("KİLİTLİ PREPROCESSING KARARLARI")
print("=" * 80)

print(
    "Lookback değerleri:",
    LOOKBACKS
)

print(
    "Feature setleri:",
    list(FEATURE_SETS.keys())
)

print(
    "Scaler:",
    "StandardScaler"
)

print(
    "Scaler fit:",
    "TRAIN ONLY"
)

print(
    "Split:",
    "Target-realization-aware chronological 70/15/15"
)

print(
    "Overlap:",
    "Geçmiş input taşınabilir; target taşınamaz."
)


# ==========================================================
# 5. TEMEL BÜTÜNLÜK KONTROLLERİ
# ==========================================================

if not features_baseline.index.equals(
    features_full.index
):

    raise ValueError(
        "Baseline ve full indexleri aynı değil."
    )


if not features_baseline.index.equals(
    targets_all.index
):

    raise ValueError(
        "Feature ve target indexleri aynı değil."
    )


if not targets_all.index.equals(
    target_dates.index
):

    raise ValueError(
        "Target ve target realization date indexleri aynı değil."
    )


if list(
    targets_all.columns
) != TARGET_NAMES:

    raise ValueError(
        "Target kolon sırası schema_v4 ile uyuşmuyor.\n"
        f"Beklenen: {TARGET_NAMES}\n"
        f"Gerçek   : {targets_all.columns.tolist()}"
    )


if features_baseline.shape[1] != BASELINE_DIM:

    raise ValueError(
        f"Baseline feature boyutu yanlış: "
        f"{features_baseline.shape[1]}"
    )


if features_full.shape[1] != FULL_DIM:

    raise ValueError(
        f"Full feature boyutu yanlış: "
        f"{features_full.shape[1]}"
    )


if targets_all.shape[1] != 8:

    raise ValueError(
        f"Target boyutu 8 değil: "
        f"{targets_all.shape[1]}"
    )


# Anchor date her zaman target realization date'ten önce olmalı.

anchor_dates = targets_all.index

realization_series = target_dates[
    "target_realization_date"
]


if not (
    realization_series.values >
    anchor_dates.values
).all():

    raise ValueError(
        "Bazı target realization date değerleri "
        "anchor date'ten ileri değil."
    )


# Target realization tarihleri kronolojik artmalı.

if not realization_series.is_monotonic_increasing:

    raise ValueError(
        "Target realization date kronolojik artan değil."
    )


if realization_series.duplicated().any():

    raise ValueError(
        "Duplicate target realization date bulundu."
    )


print("\n[OK] Temel bütünlük kontrolleri geçti.")

print(
    "Toplam ortak örnek sayısı:",
    len(targets_all)
)


# ==========================================================
# 6. TARGET-REALIZATION-AWARE SPLIT
#
# ÖNEMLİ:
#
# Split assignment TARGET REALIZATION DATE sırasına göre yapılır.
#
# Böylece:
# - Train target tarihleri sadece train dönemindedir.
# - Validation target tarihleri sadece validation dönemindedir.
# - Test target tarihleri sadece test dönemindedir.
#
# Geçmiş input penceresi önceki split'e uzanabilir.
# Ancak target ileri doğru başka split'e geçmez.
# ==========================================================

N = len(targets_all)


# Önce yaklaşık %70 train.
train_n = int(
    np.floor(
        N * TRAIN_RATIO
    )
)


# Kalan örnekleri validation ve test arasında
# mümkün olduğunca eşit böl.
remaining_n = (
    N - train_n
)


val_n = (
    remaining_n // 2
)


test_n = (
    remaining_n - val_n
)


train_start = 0
train_end = train_n

val_start = train_end
val_end = val_start + val_n

test_start = val_end
test_end = N


# Sayısal bütünlük kontrolü

if not (
    train_n +
    val_n +
    test_n
    ==
    N
):

    raise RuntimeError(
        "Split örnek sayıları toplamı N'e eşit değil."
    )


# ==========================================================
# 7. SPLIT DATAFRAME'LERİ
# ==========================================================

train_anchor_dates = (
    anchor_dates[
        train_start:train_end
    ]
)

val_anchor_dates = (
    anchor_dates[
        val_start:val_end
    ]
)

test_anchor_dates = (
    anchor_dates[
        test_start:test_end
    ]
)


train_target_dates = (
    realization_series.iloc[
        train_start:train_end
    ]
)

val_target_dates = (
    realization_series.iloc[
        val_start:val_end
    ]
)

test_target_dates = (
    realization_series.iloc[
        test_start:test_end
    ]
)


# ==========================================================
# 8. KRİTİK SPLIT SINIR KONTROLLERİ
# ==========================================================

# Hedef tarihleri splitler arasında kesin ayrık olmalı.

if not (
    train_target_dates.max()
    <
    val_target_dates.min()
):

    raise RuntimeError(
        "Train ve validation target realization "
        "tarihleri kronolojik olarak ayrık değil."
    )


if not (
    val_target_dates.max()
    <
    test_target_dates.min()
):

    raise RuntimeError(
        "Validation ve test target realization "
        "tarihleri kronolojik olarak ayrık değil."
    )


# Target tarih kümeleri kesişmemeli.

train_target_set = set(
    train_target_dates
)

val_target_set = set(
    val_target_dates
)

test_target_set = set(
    test_target_dates
)


if (
    train_target_set
    &
    val_target_set
):

    raise RuntimeError(
        "Train ve validation target tarihleri kesişiyor."
    )


if (
    val_target_set
    &
    test_target_set
):

    raise RuntimeError(
        "Validation ve test target tarihleri kesişiyor."
    )


if (
    train_target_set
    &
    test_target_set
):

    raise RuntimeError(
        "Train ve test target tarihleri kesişiyor."
    )


print("\n" + "=" * 80)
print("TARGET-REALIZATION-AWARE SPLIT")
print("=" * 80)

print(
    f"\nToplam örnek: {N}"
)

print(
    f"Train      : {train_n} "
    f"({train_n / N * 100:.4f}%)"
)

print(
    f"Validation : {val_n} "
    f"({val_n / N * 100:.4f}%)"
)

print(
    f"Test       : {test_n} "
    f"({test_n / N * 100:.4f}%)"
)


print("\nTRAIN")

print(
    "Anchor:",
    train_anchor_dates.min().date(),
    "→",
    train_anchor_dates.max().date()
)

print(
    "Target realization:",
    train_target_dates.min().date(),
    "→",
    train_target_dates.max().date()
)


print("\nVALIDATION")

print(
    "Anchor:",
    val_anchor_dates.min().date(),
    "→",
    val_anchor_dates.max().date()
)

print(
    "Target realization:",
    val_target_dates.min().date(),
    "→",
    val_target_dates.max().date()
)


print("\nTEST")

print(
    "Anchor:",
    test_anchor_dates.min().date(),
    "→",
    test_anchor_dates.max().date()
)

print(
    "Target realization:",
    test_target_dates.min().date(),
    "→",
    test_target_dates.max().date()
)


print("\n[OK] Target realization tarihleri splitler arasında ayrık.")


# ==========================================================
# 9. SPLIT META DOSYASI
# ==========================================================

split_meta = {

    "project_version":
        "v4_repro",

    "created_at":
        datetime.now().isoformat(),

    "split_method":
        "chronological_target_realization_aware",

    "split_basis":
        "target_realization_date",

    "ratios_requested": {

        "train":
            TRAIN_RATIO,

        "validation":
            VAL_RATIO,

        "test":
            TEST_RATIO
    },

    "total_rows":
        int(N),

    "train": {

        "start_idx":
            int(train_start),

        "end_idx_exclusive":
            int(train_end),

        "n_rows":
            int(train_n),

        "anchor_start":
            str(
                train_anchor_dates.min().date()
            ),

        "anchor_end":
            str(
                train_anchor_dates.max().date()
            ),

        "target_realization_start":
            str(
                train_target_dates.min().date()
            ),

        "target_realization_end":
            str(
                train_target_dates.max().date()
            )
    },

    "validation": {

        "start_idx":
            int(val_start),

        "end_idx_exclusive":
            int(val_end),

        "n_rows":
            int(val_n),

        "anchor_start":
            str(
                val_anchor_dates.min().date()
            ),

        "anchor_end":
            str(
                val_anchor_dates.max().date()
            ),

        "target_realization_start":
            str(
                val_target_dates.min().date()
            ),

        "target_realization_end":
            str(
                val_target_dates.max().date()
            )
    },

    "test": {

        "start_idx":
            int(test_start),

        "end_idx_exclusive":
            int(test_end),

        "n_rows":
            int(test_n),

        "anchor_start":
            str(
                test_anchor_dates.min().date()
            ),

        "anchor_end":
            str(
                test_anchor_dates.max().date()
            ),

        "target_realization_start":
            str(
                test_target_dates.min().date()
            ),

        "target_realization_end":
            str(
                test_target_dates.max().date()
            )
    },

    "critical_rule":
        (
            "Input history may cross backward into the previous split, "
            "but target realization may never cross forward into the next split."
        ),

    "target_sets_disjoint":
        True
}


split_meta_path = os.path.join(
    PROCESSED_DIR,
    "split_meta_v4.json"
)


with open(
    split_meta_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        split_meta,
        f,
        ensure_ascii=False,
        indent=2
    )


print("\nSplit meta kaydedildi:")
print(split_meta_path)


# ==========================================================
# 10. YARDIMCI FONKSİYONLAR
# ==========================================================

def split_dataframe(
    df: pd.DataFrame
):

    train = df.iloc[
        train_start:train_end
    ].copy()

    val = df.iloc[
        val_start:val_end
    ].copy()

    test = df.iloc[
        test_start:test_end
    ].copy()

    return (
        train,
        val,
        test
    )



def fit_transform_scalers(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test
):

    """
    StandardScaler yalnızca train setine fit edilir.

    Validation ve test:
        transform only.
    """

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()


    # ------------------------------------------------------
    # FIT: SADECE TRAIN
    # ------------------------------------------------------

    X_train_scaled = x_scaler.fit_transform(
        X_train
    )

    y_train_scaled = y_scaler.fit_transform(
        y_train
    )


    # ------------------------------------------------------
    # TRANSFORM: VALIDATION VE TEST
    # ------------------------------------------------------

    X_val_scaled = x_scaler.transform(
        X_val
    )

    y_val_scaled = y_scaler.transform(
        y_val
    )

    X_test_scaled = x_scaler.transform(
        X_test
    )

    y_test_scaled = y_scaler.transform(
        y_test
    )


    # ------------------------------------------------------
    # KRİTİK LEAKAGE KONTROLÜ
    #
    # Scaler'ın gördüğü örnek sayısı yalnızca
    # train satır sayısına eşit olmalı.
    # ------------------------------------------------------

    x_seen_raw = np.atleast_1d(
        x_scaler.n_samples_seen_
    )

    y_seen_raw = np.atleast_1d(
        y_scaler.n_samples_seen_
    )

    x_seen = int(
        x_seen_raw[0]
    )

    y_seen = int(
        y_seen_raw[0]
    )


    if x_seen != len(X_train):

        raise RuntimeError(
            "X scaler train dışında örnek görmüş olabilir."
        )


    if y_seen != len(y_train):

        raise RuntimeError(
            "Y scaler train dışında örnek görmüş olabilir."
        )


    return (
        X_train_scaled,
        y_train_scaled,

        X_val_scaled,
        y_val_scaled,

        X_test_scaled,
        y_test_scaled,

        x_scaler,
        y_scaler
    )


def make_train_sequences(
    X_scaled,
    y_scaled,
    y_raw,
    anchor_dates_array,
    target_dates_array,
    lookback
):

    """
    Train sequence üretimi.

    Her örnek için:
        son lookback kadar feature
        → mevcut anchor'daki target

    İlk lookback-1 train hedefi için yeterli
    geçmiş olmadığı için sequence üretilemez.
    """

    X_seq = []

    y_seq = []

    y_raw_seq = []

    anchor_seq = []

    target_date_seq = []


    n = len(
        X_scaled
    )


    for i in range(
        lookback - 1,
        n
    ):

        start = (
            i - lookback + 1
        )

        end = (
            i + 1
        )


        X_seq.append(
            X_scaled[
                start:end
            ]
        )


        y_seq.append(
            y_scaled[i]
        )


        y_raw_seq.append(
            y_raw[i]
        )


        anchor_seq.append(
            anchor_dates_array[i]
        )


        target_date_seq.append(
            target_dates_array[i]
        )


    return (

        np.asarray(
            X_seq,
            dtype=np.float32
        ),

        np.asarray(
            y_seq,
            dtype=np.float32
        ),

        np.asarray(
            y_raw_seq,
            dtype=np.float32
        ),

        np.asarray(
            anchor_seq
        ).astype(str),

        np.asarray(
            target_date_seq
        ).astype(str)
    )


def make_overlap_sequences(
    previous_X_scaled,
    current_X_scaled,
    current_y_scaled,
    current_y_raw,
    current_anchor_dates,
    current_target_dates,
    lookback
):

    """
    Validation ve test için overlap-aware sequence.

    Kural:
        Geçmiş INPUT penceresi önceki split'ten alınabilir.
        TARGET kesinlikle current split içinde kalır.
    """

    if lookback == 1:

        context_X = (
            current_X_scaled
        )

    else:

        needed_tail = (
            lookback - 1
        )


        if len(
            previous_X_scaled
        ) < needed_tail:

            raise ValueError(
                "Önceki split uzunluğu lookback için yetersiz.\n"
                f"needed_tail={needed_tail}\n"
                f"previous_len={len(previous_X_scaled)}"
            )


        tail_X = (
            previous_X_scaled[
                -needed_tail:
            ]
        )


        context_X = np.vstack(
            [
                tail_X,
                current_X_scaled
            ]
        )


    X_seq = []

    y_seq = []

    y_raw_seq = []

    anchor_seq = []

    target_date_seq = []


    n_current = len(
        current_X_scaled
    )


    for j in range(
        n_current
    ):

        start = j

        end = (
            j + lookback
        )


        window = context_X[
            start:end
        ]


        if len(
            window
        ) != lookback:

            raise RuntimeError(
                "Overlap-aware sequence uzunluğu yanlış."
            )


        X_seq.append(
            window
        )


        y_seq.append(
            current_y_scaled[j]
        )


        y_raw_seq.append(
            current_y_raw[j]
        )


        anchor_seq.append(
            current_anchor_dates[j]
        )


        target_date_seq.append(
            current_target_dates[j]
        )


    return (

        np.asarray(
            X_seq,
            dtype=np.float32
        ),

        np.asarray(
            y_seq,
            dtype=np.float32
        ),

        np.asarray(
            y_raw_seq,
            dtype=np.float32
        ),

        np.asarray(
            anchor_seq
        ).astype(str),

        np.asarray(
            target_date_seq
        ).astype(str)
    )


def save_numpy_array(
    path,
    arr
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    np.save(
        path,
        arr
    )


def save_json(
    path,
    obj
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2
        )


def save_pickle(
    path,
    obj
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(
        path,
        "wb"
    ) as f:

        pickle.dump(
            obj,
            f
        )


# ==========================================================
# 11. GLOBAL META
# ==========================================================

global_meta = {

    "project_version":
        "v4_repro",

    "created_at":
        datetime.now().isoformat(),

    "script":
        "02_preprocessing_v4.py",

    "base_dir":
        BASE_DIR,

    "split_method":
        "chronological_target_realization_aware",

    "split_meta_file":
        split_meta_path,

    "feature_sets":
        {},

    "important_notes": [

        "Scaler yalnızca train setine fit edilmiştir.",

        "Validation ve test setlerine yalnızca transform uygulanmıştır.",

        "Split assignment target realization date sırasına göre yapılmıştır.",

        "Train, validation ve test target realization tarihleri ayrıdır.",

        "Validation ve test sequence üretiminde geçmiş input penceresi taşınabilir.",

        "Target değerler kendi splitleri içinde kalır.",

        "Anchor date ve target realization date ayrı kaydedilmiştir.",

        "Bu dosyada model eğitimi yapılmamıştır.",

        "Bu dosyada validation veya test skoru hesaplanmamıştır."
    ]
}


# ==========================================================
# 12. HER FEATURE SET İÇİN PREPROCESSING
# ==========================================================

for (
    feature_set_name,
    X_df
) in FEATURE_SETS.items():


    print("\n")
    print("=" * 80)
    print(
        f"FEATURE SET İŞLENİYOR: "
        f"{feature_set_name.upper()}"
    )
    print("=" * 80)


    feature_dir = os.path.join(
        SEQUENCE_DIR,
        feature_set_name
    )


    os.makedirs(
        feature_dir,
        exist_ok=True
    )


    y_df = targets_all.copy()


    # ------------------------------------------------------
    # Split
    # ------------------------------------------------------

    (
        X_train,
        X_val,
        X_test
    ) = split_dataframe(
        X_df
    )


    (
        y_train,
        y_val,
        y_test
    ) = split_dataframe(
        y_df
    )


    (
        date_train,
        date_val,
        date_test
    ) = split_dataframe(
        target_dates
    )


    print(
        "\nX_train:",
        X_train.shape
    )

    print(
        "X_val:",
        X_val.shape
    )

    print(
        "X_test:",
        X_test.shape
    )


    print(
        "\ny_train:",
        y_train.shape
    )

    print(
        "y_val:",
        y_val.shape
    )

    print(
        "y_test:",
        y_test.shape
    )


    # ------------------------------------------------------
    # Scaler
    # ------------------------------------------------------

    (
        X_train_scaled,
        y_train_scaled,

        X_val_scaled,
        y_val_scaled,

        X_test_scaled,
        y_test_scaled,

        x_scaler,
        y_scaler

    ) = fit_transform_scalers(

        X_train,
        y_train,

        X_val,
        y_val,

        X_test,
        y_test
    )


    # Raw targetlar:
    # metrik ve inverse-scale kontrolü için saklanır.

    y_train_raw = (
        y_train
        .values
        .astype(
            np.float32
        )
    )


    y_val_raw = (
        y_val
        .values
        .astype(
            np.float32
        )
    )


    y_test_raw = (
        y_test
        .values
        .astype(
            np.float32
        )
    )


    # Anchor dates

    train_anchor_array = (
        X_train.index
        .astype(str)
        .values
    )


    val_anchor_array = (
        X_val.index
        .astype(str)
        .values
    )


    test_anchor_array = (
        X_test.index
        .astype(str)
        .values
    )


    # Target realization dates

    train_target_date_array = (
        date_train[
            "target_realization_date"
        ]
        .astype(str)
        .values
    )


    val_target_date_array = (
        date_val[
            "target_realization_date"
        ]
        .astype(str)
        .values
    )


    test_target_date_array = (
        date_test[
            "target_realization_date"
        ]
        .astype(str)
        .values
    )


    # ------------------------------------------------------
    # Scaler kaydet
    # ------------------------------------------------------

    scaler_path = os.path.join(
        feature_dir,
        "scalers.pkl"
    )


    save_pickle(

        scaler_path,

        {

            "x_scaler":
                x_scaler,

            "y_scaler":
                y_scaler,

            "feature_columns":
                list(
                    X_df.columns
                ),

            "target_columns":
                list(
                    y_df.columns
                ),

            "fit_split":
                "train_only",

            "fit_rows":
                int(
                    len(
                        X_train
                    )
                ),

            "train_anchor_start":
                str(
                    X_train.index.min().date()
                ),

            "train_anchor_end":
                str(
                    X_train.index.max().date()
                ),

            "train_target_realization_start":
                str(
                    date_train[
                        "target_realization_date"
                    ].min().date()
                ),

            "train_target_realization_end":
                str(
                    date_train[
                        "target_realization_date"
                    ].max().date()
                ),

            "fit_rule":
                (
                    "x_scaler ve y_scaler yalnızca "
                    "train setine fit edilmiştir."
                )
        }
    )


    print(
        "\n[KAYIT] Scaler:",
        scaler_path
    )


    feature_set_meta = {

        "feature_set":
            feature_set_name,

        "feature_columns":
            list(
                X_df.columns
            ),

        "target_columns":
            list(
                y_df.columns
            ),

        "scaler_file":
            scaler_path,

        "scaler_fit_rows":
            int(
                len(
                    X_train
                )
            ),

        "split_shapes_before_sequence": {

            "X_train":
                list(
                    X_train.shape
                ),

            "X_val":
                list(
                    X_val.shape
                ),

            "X_test":
                list(
                    X_test.shape
                ),

            "y_train":
                list(
                    y_train.shape
                ),

            "y_val":
                list(
                    y_val.shape
                ),

            "y_test":
                list(
                    y_test.shape
                )
        },

        "lookbacks":
            {}
    }


    # Test input geçmişi:
    # train + validation feature geçmişini kullanabilir.
    #
    # Bu target taşımaz.
    # Sadece geçmiş input context'idir.

    train_val_X_scaled = np.vstack(
        [
            X_train_scaled,
            X_val_scaled
        ]
    )


    # ------------------------------------------------------
    # Lookback döngüsü
    # ------------------------------------------------------

    for lookback in LOOKBACKS:


        print(
            f"\n[LOOKBACK {lookback}] "
            f"Sequence üretiliyor..."
        )


        lb_dir = os.path.join(
            feature_dir,
            f"lb{lookback}"
        )


        os.makedirs(
            lb_dir,
            exist_ok=True
        )


        # TRAIN

        (
            X_train_seq,
            y_train_seq,
            y_train_raw_seq,
            anchor_train_seq,
            target_date_train_seq

        ) = make_train_sequences(

            X_scaled=
                X_train_scaled,

            y_scaled=
                y_train_scaled,

            y_raw=
                y_train_raw,

            anchor_dates_array=
                train_anchor_array,

            target_dates_array=
                train_target_date_array,

            lookback=
                lookback
        )


        # VALIDATION

        (
            X_val_seq,
            y_val_seq,
            y_val_raw_seq,
            anchor_val_seq,
            target_date_val_seq

        ) = make_overlap_sequences(

            previous_X_scaled=
                X_train_scaled,

            current_X_scaled=
                X_val_scaled,

            current_y_scaled=
                y_val_scaled,

            current_y_raw=
                y_val_raw,

            current_anchor_dates=
                val_anchor_array,

            current_target_dates=
                val_target_date_array,

            lookback=
                lookback
        )


        # TEST

        (
            X_test_seq,
            y_test_seq,
            y_test_raw_seq,
            anchor_test_seq,
            target_date_test_seq

        ) = make_overlap_sequences(

            previous_X_scaled=
                train_val_X_scaled,

            current_X_scaled=
                X_test_scaled,

            current_y_scaled=
                y_test_scaled,

            current_y_raw=
                y_test_raw,

            current_anchor_dates=
                test_anchor_array,

            current_target_dates=
                test_target_date_array,

            lookback=
                lookback
        )


        # --------------------------------------------------
        # SHAPE KONTROLLERİ
        # --------------------------------------------------

        expected_n_features = (
            X_df.shape[1]
        )


        expected_n_targets = (
            y_df.shape[1]
        )


        for (
            name,
            arr
        ) in [

            (
                "X_train_seq",
                X_train_seq
            ),

            (
                "X_val_seq",
                X_val_seq
            ),

            (
                "X_test_seq",
                X_test_seq
            )
        ]:


            if arr.ndim != 3:

                raise ValueError(
                    f"{name} 3 boyutlu değil: "
                    f"{arr.shape}"
                )


            if arr.shape[1] != lookback:

                raise ValueError(
                    f"{name} lookback boyutu yanlış: "
                    f"{arr.shape}"
                )


            if (
                arr.shape[2]
                !=
                expected_n_features
            ):

                raise ValueError(
                    f"{name} feature boyutu yanlış: "
                    f"{arr.shape}"
                )


        for (
            name,
            arr
        ) in [

            (
                "y_train_seq",
                y_train_seq
            ),

            (
                "y_val_seq",
                y_val_seq
            ),

            (
                "y_test_seq",
                y_test_seq
            )
        ]:


            if arr.ndim != 2:

                raise ValueError(
                    f"{name} 2 boyutlu değil: "
                    f"{arr.shape}"
                )


            if (
                arr.shape[1]
                !=
                expected_n_targets
            ):

                raise ValueError(
                    f"{name} target boyutu yanlış: "
                    f"{arr.shape}"
                )


        # Validation ve test pencere kaybı 0 olmalı.

        if len(
            X_val_seq
        ) != val_n:

            raise RuntimeError(
                "Validation sequence sayısında "
                "pencere kaybı oluştu."
            )


        if len(
            X_test_seq
        ) != test_n:

            raise RuntimeError(
                "Test sequence sayısında "
                "pencere kaybı oluştu."
            )


        # Target realization dates kendi splitlerinde kalmalı.

        if not (
            pd.to_datetime(
                target_date_train_seq
            ).max()
            <
            pd.to_datetime(
                target_date_val_seq
            ).min()
        ):

            raise RuntimeError(
                "Train ve validation sequence target "
                "tarihleri ayrık değil."
            )


        if not (
            pd.to_datetime(
                target_date_val_seq
            ).max()
            <
            pd.to_datetime(
                target_date_test_seq
            ).min()
        ):

            raise RuntimeError(
                "Validation ve test sequence target "
                "tarihleri ayrık değil."
            )


        # --------------------------------------------------
        # DOSYALARI KAYDET
        # --------------------------------------------------

        # TRAIN

        save_numpy_array(
            os.path.join(
                lb_dir,
                "X_train.npy"
            ),
            X_train_seq
        )


        save_numpy_array(
            os.path.join(
                lb_dir,
                "y_train.npy"
            ),
            y_train_seq
        )


        save_numpy_array(
            os.path.join(
                lb_dir,
                "y_train_raw.npy"
            ),
            y_train_raw_seq
        )


        save_numpy_array(
            os.path.join(
                lb_dir,
                "anchor_dates_train.npy"
            ),
            anchor_train_seq
        )


        save_numpy_array(
            os.path.join(
                lb_dir,
                "target_realization_dates_train.npy"
            ),
            target_date_train_seq
        )


        # VALIDATION

        save_numpy_array(
            os.path.join(
                lb_dir,
                "X_val.npy"
            ),
            X_val_seq
        )


        save_numpy_array(
            os.path.join(
                lb_dir,
                "y_val.npy"
            ),
            y_val_seq
        )


        save_numpy_array(
            os.path.join(
                lb_dir,
                "y_val_raw.npy"
            ),
            y_val_raw_seq
        )


        save_numpy_array(
            os.path.join(
                lb_dir,
                "anchor_dates_val.npy"
            ),
            anchor_val_seq
        )


        save_numpy_array(
            os.path.join(
                lb_dir,
                "target_realization_dates_val.npy"
            ),
            target_date_val_seq
        )


        # TEST

        save_numpy_array(
            os.path.join(
                lb_dir,
                "X_test.npy"
            ),
            X_test_seq
        )


        save_numpy_array(
            os.path.join(
                lb_dir,
                "y_test.npy"
            ),
            y_test_seq
        )


        save_numpy_array(
            os.path.join(
                lb_dir,
                "y_test_raw.npy"
            ),
            y_test_raw_seq
        )


        save_numpy_array(
            os.path.join(
                lb_dir,
                "anchor_dates_test.npy"
            ),
            anchor_test_seq
        )


        save_numpy_array(
            os.path.join(
                lb_dir,
                "target_realization_dates_test.npy"
            ),
            target_date_test_seq
        )


        # --------------------------------------------------
        # SEQUENCE META
        # --------------------------------------------------

        sequence_meta = {

            "project_version":
                "v4_repro",

            "feature_set":
                feature_set_name,

            "lookback":
                int(
                    lookback
                ),

            "feature_columns":
                list(
                    X_df.columns
                ),

            "target_columns":
                list(
                    y_df.columns
                ),

            "split_basis":
                "target_realization_date",

            "scaler_rule":
                "fit_train_only",

            "overlap_rule":
                (
                    "Past input history may cross backward "
                    "into previous split; targets may not cross."
                ),

            "shapes": {

                "X_train":
                    list(
                        X_train_seq.shape
                    ),

                "y_train":
                    list(
                        y_train_seq.shape
                    ),

                "y_train_raw":
                    list(
                        y_train_raw_seq.shape
                    ),

                "X_val":
                    list(
                        X_val_seq.shape
                    ),

                "y_val":
                    list(
                        y_val_seq.shape
                    ),

                "y_val_raw":
                    list(
                        y_val_raw_seq.shape
                    ),

                "X_test":
                    list(
                        X_test_seq.shape
                    ),

                "y_test":
                    list(
                        y_test_seq.shape
                    ),

                "y_test_raw":
                    list(
                        y_test_raw_seq.shape
                    )
            },

            "date_ranges": {

                "train_anchor": [
                    str(
                        anchor_train_seq[0]
                    ),
                    str(
                        anchor_train_seq[-1]
                    )
                ],

                "train_target_realization": [
                    str(
                        target_date_train_seq[0]
                    ),
                    str(
                        target_date_train_seq[-1]
                    )
                ],

                "validation_anchor": [
                    str(
                        anchor_val_seq[0]
                    ),
                    str(
                        anchor_val_seq[-1]
                    )
                ],

                "validation_target_realization": [
                    str(
                        target_date_val_seq[0]
                    ),
                    str(
                        target_date_val_seq[-1]
                    )
                ],

                "test_anchor": [
                    str(
                        anchor_test_seq[0]
                    ),
                    str(
                        anchor_test_seq[-1]
                    )
                ],

                "test_target_realization": [
                    str(
                        target_date_test_seq[0]
                    ),
                    str(
                        target_date_test_seq[-1]
                    )
                ]
            },

            "important_note":
                (
                    "Test sequence dosyaları yalnızca "
                    "önceden hazırlanmıştır; model seçimi "
                    "ve validation sürecinde test skorları "
                    "kullanılmayacaktır."
                )
        }


        save_json(
            os.path.join(
                lb_dir,
                "sequence_meta.json"
            ),
            sequence_meta
        )


        feature_set_meta[
            "lookbacks"
        ][
            str(
                lookback
            )
        ] = sequence_meta


        print(
            "X_train:",
            X_train_seq.shape,
            "y_train:",
            y_train_seq.shape
        )


        print(
            "X_val:",
            X_val_seq.shape,
            "y_val:",
            y_val_seq.shape
        )


        print(
            "X_test:",
            X_test_seq.shape,
            "y_test:",
            y_test_seq.shape
        )


        print(
            "[KAYIT]",
            lb_dir
        )


    global_meta[
        "feature_sets"
    ][
        feature_set_name
    ] = feature_set_meta


# ==========================================================
# 13. GLOBAL PREPROCESSING META
# ==========================================================

preprocessing_meta_path = os.path.join(
    SEQUENCE_DIR,
    "preprocessing_meta_v4.json"
)


save_json(
    preprocessing_meta_path,
    global_meta
)


# ==========================================================
# 14. SON ÖZET
# ==========================================================

print("\n")
print("=" * 80)
print("02_preprocessing_v4.py BAŞARIYLA TAMAMLANDI")
print("=" * 80)


print("\nSplit meta:")
print(split_meta_path)


print("\nPreprocessing meta:")
print(preprocessing_meta_path)


print("\nÜretilen ana klasörler:")

print(
    " - data/sequences/baseline/"
    "lb10, lb20, lb30, lb60"
)

print(
    " - data/sequences/full/"
    "lb10, lb20, lb30, lb60"
)


print("\nKURAL KONTROLÜ:")

print(
    "✅ Split target realization date esaslı."
)

print(
    "✅ Train / validation / test target tarihleri ayrık."
)

print(
    "✅ StandardScaler yalnızca train setine fit edildi."
)

print(
    "✅ Validation/test için geçmiş input penceresi taşındı."
)

print(
    "✅ Validation/test pencere kaybı = 0."
)

print(
    "✅ Anchor date ve target realization date ayrı kaydedildi."
)

print(
    "✅ Model eğitimi yapılmadı."
)

print(
    "✅ Test değerlendirmesi yapılmadı."
)

print("=" * 80)
