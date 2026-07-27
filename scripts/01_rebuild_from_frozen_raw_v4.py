
# ==========================================================
# 01_rebuild_from_frozen_raw_v4.py
#
# AMAÇ:
# - İnternete bağlanmadan frozen raw_prices.csv dosyasını kullanmak
# - Fiyatları leakage-free biçimde temizlemek
# - Baseline ve full feature setlerini üretmek
# - RSI14 zero-loss durumunu doğru ele almak
# - NextRet ve NextVol hedeflerini üretmek
# - Anchor date ile target realization date'i ayrı kaydetmek
# ==========================================================

import os
import json
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime


# ==========================================================
# 1. PROJE YOLLARI
# ==========================================================

BASE_DIR = "/content/drive/MyDrive/tez_transformer_v4_repro"

RAW_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "raw_prices.csv"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

CONFIG_DIR = os.path.join(
    BASE_DIR,
    "config"
)

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)


EXPECTED_RAW_SHA256 = (
    "ab5f275d38dc98057b1cedcf58019adb26be7402c7ed5ae6ee3d6877b2444893"
)


ASSETS = [
    "BIST100",
    "USDTRY",
    "EURTRY",
    "GOLD"
]


BASELINE_FEATURE_NAMES = [
    "LogRet",
    "Vol20"
]


FULL_FEATURE_NAMES = [
    "LogRet",
    "Vol20",
    "MA5_Ratio",
    "MA20_Ratio",
    "RSI14",
    "MACD",
    "MACDSignal"
]


TARGET_ORDER = [
    "BIST100_NextRet",
    "USDTRY_NextRet",
    "EURTRY_NextRet",
    "GOLD_NextRet",
    "BIST100_NextVol",
    "USDTRY_NextVol",
    "EURTRY_NextVol",
    "GOLD_NextVol"
]


# ==========================================================
# 2. SHA-256 FONKSİYONU
# ==========================================================

def sha256_file(path, chunk_size=1024 * 1024):

    sha256 = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(chunk_size)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ==========================================================
# 3. FROZEN RAW VERİYİ OKU VE DOĞRULA
# ==========================================================

if not os.path.exists(RAW_PATH):

    raise FileNotFoundError(
        f"Frozen raw_prices.csv bulunamadı:\n{RAW_PATH}"
    )


raw_hash = sha256_file(RAW_PATH)


if raw_hash != EXPECTED_RAW_SHA256:

    raise RuntimeError(
        "Frozen raw data SHA-256 uyuşmuyor.\n"
        f"Beklenen: {EXPECTED_RAW_SHA256}\n"
        f"Gerçek   : {raw_hash}"
    )


raw = pd.read_csv(
    RAW_PATH,
    index_col=0,
    parse_dates=True
)


# ==========================================================
# 4. RAW YAPI KONTROLLERİ
# ==========================================================

if list(raw.columns) != ASSETS:

    raise ValueError(
        "Raw kolon sırası beklenen yapıyla uyuşmuyor.\n"
        f"Beklenen: {ASSETS}\n"
        f"Gerçek   : {raw.columns.tolist()}"
    )


if not raw.index.is_monotonic_increasing:

    raise ValueError(
        "Raw tarih index'i kronolojik artan değil."
    )


duplicate_count = int(
    raw.index.duplicated().sum()
)


if duplicate_count != 0:

    raise ValueError(
        f"Duplicate tarih bulundu: {duplicate_count}"
    )


print("=" * 80)
print("01 — FROZEN RAW VERİ DOĞRULANDI")
print("=" * 80)

print("\nRaw shape:")
print(raw.shape)

print("\nTarih aralığı:")
print(raw.index.min(), "→", raw.index.max())

print("\nRaw NaN sayıları:")
print(raw.isna().sum())

print("\nSHA-256:")
print(raw_hash)


# ==========================================================
# 5. FİYAT TEMİZLEME
#
# Union calendar korunur.
# Sadece geçmişte bilinen son değer ileri taşınır.
# bfill YOK.
# ==========================================================

prices_clean = raw.ffill()


remaining_nan = prices_clean.isna().sum()


print("\n" + "=" * 80)
print("FFILL SONRASI NaN KONTROLÜ")
print("=" * 80)

print(remaining_nan)


if int(remaining_nan.sum()) != 0:

    raise ValueError(
        "ffill sonrası NaN kaldı. "
        "Leading missing durumları ayrıca incelenmeli; "
        "otomatik bfill yapılmayacak."
    )


# ==========================================================
# 6. RSI14 FONKSİYONU — DÜZELTİLMİŞ
#
# Kural:
# avg_loss == 0 ve avg_gain > 0  → RSI = 100
# avg_gain == 0 ve avg_loss > 0  → RSI = 0
# avg_gain == 0 ve avg_loss == 0 → RSI = 50
# diğer durum                     → standart RSI formülü
# ==========================================================

def compute_rsi14(close: pd.Series):

    delta = close.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(
        window=14,
        min_periods=14
    ).mean()

    avg_loss = loss.rolling(
        window=14,
        min_periods=14
    ).mean()

    rsi = pd.Series(
        np.nan,
        index=close.index,
        dtype=float
    )

    normal_mask = (
        (avg_gain > 0) &
        (avg_loss > 0)
    )

    rs = (
        avg_gain[normal_mask] /
        avg_loss[normal_mask]
    )

    rsi.loc[normal_mask] = (
        100.0 -
        (100.0 / (1.0 + rs))
    )

    only_gain_mask = (
        (avg_gain > 0) &
        (avg_loss == 0)
    )

    rsi.loc[only_gain_mask] = 100.0

    only_loss_mask = (
        (avg_gain == 0) &
        (avg_loss > 0)
    )

    rsi.loc[only_loss_mask] = 0.0

    flat_mask = (
        (avg_gain == 0) &
        (avg_loss == 0)
    )

    rsi.loc[flat_mask] = 50.0

    return rsi


# ==========================================================
# 7. FEATURE VE TARGET ÜRETİMİ
# ==========================================================

baseline_parts = []
full_parts = []

next_ret_parts = []
next_vol_parts = []

rsi_audit_records = []


for asset in ASSETS:

    close = prices_clean[asset].astype(float)


    # ------------------------------------------------------
    # Log Return
    # ------------------------------------------------------

    logret = np.log(
        close / close.shift(1)
    )


    # ------------------------------------------------------
    # 20 günlük annualized historical volatility
    # ------------------------------------------------------

    vol20 = (
        logret
        .rolling(
            window=20,
            min_periods=20
        )
        .std()
        * np.sqrt(252)
    )


    # ------------------------------------------------------
    # Moving Average Ratios
    # ------------------------------------------------------

    ma5 = close.rolling(
        window=5,
        min_periods=5
    ).mean()

    ma20 = close.rolling(
        window=20,
        min_periods=20
    ).mean()

    ma5_ratio = (
        close / ma5
    ) - 1.0

    ma20_ratio = (
        close / ma20
    ) - 1.0


    # ------------------------------------------------------
    # RSI14 — corrected zero-loss handling
    # ------------------------------------------------------

    delta = close.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(
        window=14,
        min_periods=14
    ).mean()

    avg_loss = loss.rolling(
        window=14,
        min_periods=14
    ).mean()

    rsi14 = compute_rsi14(close)

    zero_loss_positive_gain_count = int(
        (
            (avg_loss == 0) &
            (avg_gain > 0)
        ).sum()
    )

    zero_gain_positive_loss_count = int(
        (
            (avg_gain == 0) &
            (avg_loss > 0)
        ).sum()
    )

    flat_count = int(
        (
            (avg_gain == 0) &
            (avg_loss == 0)
        ).sum()
    )

    rsi_audit_records.append(
        {
            "asset":
                asset,

            "zero_loss_positive_gain_count":
                zero_loss_positive_gain_count,

            "zero_gain_positive_loss_count":
                zero_gain_positive_loss_count,

            "flat_count":
                flat_count
        }
    )


    # ------------------------------------------------------
    # MACD
    # ------------------------------------------------------

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    macd = ema12 - ema26

    macd_signal = macd.ewm(
        span=9,
        adjust=False
    ).mean()


    # ------------------------------------------------------
    # Baseline feature set
    # ------------------------------------------------------

    baseline_asset = pd.DataFrame(
        {
            f"{asset}_LogRet":
                logret,

            f"{asset}_Vol20":
                vol20,
        },
        index=prices_clean.index
    )

    baseline_parts.append(
        baseline_asset
    )


    # ------------------------------------------------------
    # Full feature set
    # ------------------------------------------------------

    full_asset = pd.DataFrame(
        {
            f"{asset}_LogRet":
                logret,

            f"{asset}_Vol20":
                vol20,

            f"{asset}_MA5_Ratio":
                ma5_ratio,

            f"{asset}_MA20_Ratio":
                ma20_ratio,

            f"{asset}_RSI14":
                rsi14,

            f"{asset}_MACD":
                macd,

            f"{asset}_MACDSignal":
                macd_signal,
        },
        index=prices_clean.index
    )

    full_parts.append(
        full_asset
    )


    # ------------------------------------------------------
    # Targets
    #
    # Anchor date = t
    # NextRet[t]  = LogRet[t+1]
    # NextVol[t]  = Vol20[t+1]
    # ------------------------------------------------------

    next_ret_parts.append(
        logret.shift(-1).rename(
            f"{asset}_NextRet"
        )
    )

    next_vol_parts.append(
        vol20.shift(-1).rename(
            f"{asset}_NextVol"
        )
    )


# ==========================================================
# 8. BİRLEŞTİR
# ==========================================================

features_baseline_raw = pd.concat(
    baseline_parts,
    axis=1
)

features_full_raw = pd.concat(
    full_parts,
    axis=1
)


# Target order bilinçli olarak:
# önce 4 return, sonra 4 volatility

targets_raw = pd.concat(
    next_ret_parts + next_vol_parts,
    axis=1
)


if list(targets_raw.columns) != TARGET_ORDER:

    raise RuntimeError(
        "Target sırası kilitli sırayla uyuşmuyor.\n"
        f"Beklenen: {TARGET_ORDER}\n"
        f"Gerçek   : {targets_raw.columns.tolist()}"
    )


# ==========================================================
# 9. TARGET REALIZATION DATE ÜRET
#
# Her anchor date t için target, bir sonraki union-calendar
# tarihinde gerçekleşir.
# ==========================================================

target_realization_dates = pd.Series(
    data=prices_clean.index.to_series().shift(-1).values,
    index=prices_clean.index,
    name="target_realization_date"
)


# ==========================================================
# 10. ORTAK GEÇERLİ INDEX
#
# Baseline ve full feature setleri aynı örneklemde
# karşılaştırılsın diye ortak index kullanılır.
# ==========================================================

combined_for_index = pd.concat(
    [
        features_baseline_raw,
        features_full_raw,
        targets_raw,
        target_realization_dates
    ],
    axis=1
)


valid_index = (
    combined_for_index
    .dropna()
    .index
)


features_baseline = (
    features_baseline_raw
    .loc[valid_index]
    .copy()
)

features_full = (
    features_full_raw
    .loc[valid_index]
    .copy()
)

targets_all = (
    targets_raw
    .loc[valid_index]
    .copy()
)

target_dates = (
    target_realization_dates
    .loc[valid_index]
    .to_frame()
)


# ==========================================================
# 11. TEMEL BÜTÜNLÜK KONTROLLERİ
# ==========================================================

if not (
    features_baseline.index.equals(
        features_full.index
    )
    and
    features_full.index.equals(
        targets_all.index
    )
    and
    targets_all.index.equals(
        target_dates.index
    )
):

    raise RuntimeError(
        "Feature/target/target-date index hizası bozuk."
    )


if features_baseline.shape[1] != 8:

    raise RuntimeError(
        f"Baseline feature dim 8 değil: "
        f"{features_baseline.shape[1]}"
    )


if features_full.shape[1] != 28:

    raise RuntimeError(
        f"Full feature dim 28 değil: "
        f"{features_full.shape[1]}"
    )


if targets_all.shape[1] != 8:

    raise RuntimeError(
        f"Target dim 8 değil: "
        f"{targets_all.shape[1]}"
    )


if not (
    pd.to_datetime(
        target_dates[
            "target_realization_date"
        ]
    )
    >
    target_dates.index
).all():

    raise RuntimeError(
        "Bazı target realization date değerleri "
        "anchor date'ten ileri değil."
    )


# ==========================================================
# 12. ÇIKTI DOSYALARI
# ==========================================================

prices_clean_path = os.path.join(
    PROCESSED_DIR,
    "prices_clean.csv"
)

baseline_path = os.path.join(
    PROCESSED_DIR,
    "features_baseline.csv"
)

full_path = os.path.join(
    PROCESSED_DIR,
    "features_full.csv"
)

targets_path = os.path.join(
    PROCESSED_DIR,
    "targets_all.csv"
)

target_dates_path = os.path.join(
    PROCESSED_DIR,
    "target_realization_dates.csv"
)

rsi_audit_path = os.path.join(
    PROCESSED_DIR,
    "rsi14_audit_v4.csv"
)


prices_clean.to_csv(
    prices_clean_path
)

features_baseline.to_csv(
    baseline_path
)

features_full.to_csv(
    full_path
)

targets_all.to_csv(
    targets_path
)

target_dates.to_csv(
    target_dates_path
)

pd.DataFrame(
    rsi_audit_records
).to_csv(
    rsi_audit_path,
    index=False
)


# ==========================================================
# 13. HASH KAYITLARI
# ==========================================================

derived_files = [
    prices_clean_path,
    baseline_path,
    full_path,
    targets_path,
    target_dates_path,
    rsi_audit_path
]


hash_records = []


for path in derived_files:

    hash_records.append(
        {
            "file":
                os.path.relpath(
                    path,
                    BASE_DIR
                ),

            "sha256":
                sha256_file(path)
        }
    )


hash_df = pd.DataFrame(
    hash_records
)


hash_output_path = os.path.join(
    CONFIG_DIR,
    "derived_data_sha256_v4.csv"
)


hash_df.to_csv(
    hash_output_path,
    index=False
)


# ==========================================================
# 14. META DOSYASI
# ==========================================================

meta = {

    "project_version":
        "v4_repro",

    "created_at":
        datetime.now().isoformat(),

    "raw_data": {

        "file":
            "data/raw/raw_prices.csv",

        "sha256":
            raw_hash,

        "shape":
            list(raw.shape),

        "date_start":
            str(raw.index.min().date()),

        "date_end":
            str(raw.index.max().date())
    },

    "clean_prices": {

        "shape":
            list(prices_clean.shape),

        "date_start":
            str(prices_clean.index.min().date()),

        "date_end":
            str(prices_clean.index.max().date()),

        "fill_method":
            "ffill_only",

        "bfill_used":
            False
    },

    "features": {

        "baseline_shape":
            list(features_baseline.shape),

        "full_shape":
            list(features_full.shape),

        "common_index":
            True,

        "rsi14_rule":
            {
                "avg_loss_0_avg_gain_positive":
                    100.0,

                "avg_gain_0_avg_loss_positive":
                    0.0,

                "avg_gain_0_avg_loss_0":
                    50.0
            }
    },

    "targets": {

        "shape":
            list(targets_all.shape),

        "order":
            TARGET_ORDER,

        "return_rule":
            "NextRet[t] = LogRet[t+1]",

        "volatility_rule":
            "NextVol[t] = Vol20[t+1]",

        "target_realization_dates_saved":
            True
    },

    "final_common_data": {

        "rows":
            len(valid_index),

        "anchor_date_start":
            str(valid_index.min().date()),

        "anchor_date_end":
            str(valid_index.max().date()),

        "first_target_realization_date":
            str(
                pd.to_datetime(
                    target_dates.iloc[0, 0]
                ).date()
            ),

        "last_target_realization_date":
            str(
                pd.to_datetime(
                    target_dates.iloc[-1, 0]
                ).date()
            )
    },

    "scientific_principle":
        "Kararlar kilitli. Sonuçlar kilitli değil. Veri karar verir."
}


meta_path = os.path.join(
    PROCESSED_DIR,
    "meta_v4.json"
)


with open(
    meta_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        meta,
        f,
        ensure_ascii=False,
        indent=2
    )


# ==========================================================
# 15. SONUÇLARI YAZDIR
# ==========================================================

print("\n" + "=" * 80)
print("01_rebuild_from_frozen_raw_v4.py TAMAMLANDI")
print("=" * 80)

print("\nRAW:")
print("shape =", raw.shape)
print(
    "date  =",
    raw.index.min().date(),
    "→",
    raw.index.max().date()
)

print("\nCLEAN PRICES:")
print("shape =", prices_clean.shape)
print(
    "NaN   =",
    int(prices_clean.isna().sum().sum())
)

print("\nFEATURES:")
print(
    "baseline =",
    features_baseline.shape
)

print(
    "full     =",
    features_full.shape
)

print("\nTARGETS:")
print(
    "targets  =",
    targets_all.shape
)

print("\nTARGET ORDER:")
for i, col in enumerate(
    targets_all.columns
):
    print(
        f"[{i}] {col}"
    )

print("\nFINAL COMMON INDEX:")
print(
    len(valid_index),
    "satır"
)

print(
    "Anchor:",
    valid_index.min().date(),
    "→",
    valid_index.max().date()
)

print(
    "Target realization:",
    pd.to_datetime(
        target_dates.iloc[0, 0]
    ).date(),
    "→",
    pd.to_datetime(
        target_dates.iloc[-1, 0]
    ).date()
)

print("\nRSI14 AUDIT:")
print(
    pd.DataFrame(
        rsi_audit_records
    ).to_string(
        index=False
    )
)

print("\nDOSYALAR:")
print(" -", prices_clean_path)
print(" -", baseline_path)
print(" -", full_path)
print(" -", targets_path)
print(" -", target_dates_path)
print(" -", rsi_audit_path)
print(" -", hash_output_path)
print(" -", meta_path)

print("\n✅ Frozen raw veri değiştirilmedi.")
print("✅ İnternet/yfinance kullanılmadı.")
print("✅ bfill kullanılmadı.")
print("✅ RSI14 zero-loss handling düzeltildi.")
print("✅ Target realization date ayrı kaydedildi.")
