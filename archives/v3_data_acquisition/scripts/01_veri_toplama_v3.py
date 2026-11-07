
# ==========================================================
# 01_veri_toplama_v3.py
# Çoklu Görevli Transformer ile Finansal Risk ve Getiri Tahmini
#
# Görev:
#   - Yahoo Finance üzerinden verileri indirmek
#   - Fiyatları hizalamak
#   - Baseline ve Full feature setlerini üretmek
#   - NextRet ve NextVol target dosyasını oluşturmak
#
# Bu dosyada YOK:
#   - scaler yok
#   - sequence yok
#   - model yok
#   - train/validation/test dosyası üretimi yok
# ==========================================================

import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ==========================================================
# 0. Paket kontrolü
# ==========================================================

try:
    import yfinance as yf
except ImportError as e:
    raise ImportError(
        "yfinance bulunamadı. Önce Colab'da şu komutu çalıştır:\n"
        "!pip install yfinance -q"
    ) from e


# ==========================================================
# 1. Ana yollar
# ==========================================================

BASE_DIR = "/content/drive/MyDrive/tez_transformer"

CONFIG_DIR = os.path.join(BASE_DIR, "config")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
LOG_DIR = os.path.join(BASE_DIR, "logs")

for path in [CONFIG_DIR, RAW_DIR, PROCESSED_DIR, LOG_DIR]:
    os.makedirs(path, exist_ok=True)


# ==========================================================
# 2. Schema oku
# ==========================================================

schema_path = os.path.join(CONFIG_DIR, "schema_v3.json")

if not os.path.exists(schema_path):
    raise FileNotFoundError(
        "schema_v3.json bulunamadı. Önce 00_setup_v3.py çalıştırılmalı."
    )

with open(schema_path, "r", encoding="utf-8") as f:
    schema = json.load(f)

print("[OK] Schema okundu:", schema_path)


# ==========================================================
# 3. Kilitli kararları schema'dan al
# ==========================================================

ASSETS = schema["data"]["assets"]
ASSET_ORDER = schema["data"]["asset_order"]

START_DATE = schema["data"]["start_date"]
END_DATE = schema["data"]["end_date"]

BASELINE_FEATURES_PER_ASSET = schema["features"]["baseline"]["features_per_asset"]
FULL_FEATURES_PER_ASSET = schema["features"]["full"]["features_per_asset"]

TARGET_NAMES = schema["targets"]["target_order"]

print("\n[KİLİTLİ VERİ KARARLARI]")
print("Varlık sırası:", ASSET_ORDER)
print("Tickerlar:", ASSETS)
print("Başlangıç:", START_DATE)
print("Bitiş:", END_DATE)
print("Baseline feature:", BASELINE_FEATURES_PER_ASSET)
print("Full feature:", FULL_FEATURES_PER_ASSET)
print("Target sırası:", TARGET_NAMES)


# ==========================================================
# 4. Yardımcı fonksiyonlar
# ==========================================================

def get_yfinance_end_date_inclusive(end_date: str) -> str:
    """
    yfinance end parametresi pratikte bitiş tarihini dışlayabilir.
    Bu nedenle 2024-12-31'i dahil etmek için end tarihine 1 gün ekliyoruz.
    """
    return (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def extract_price_column(df: pd.DataFrame, ticker: str) -> pd.Series:
    """
    yfinance çıktısında Adj Close veya Close kolonunu güvenli biçimde seçer.
    Yeni yfinance sürümlerinde bazen MultiIndex kolon dönebildiği için
    bu fonksiyon iki durumu da yakalar.
    """
    preferred_cols = ["Adj Close", "Close"]

    if df.empty:
        raise ValueError(f"{ticker} için boş veri döndü.")

    # Normal kolon yapısı
    for col in preferred_cols:
        if col in df.columns:
            s = df[col].copy()
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            return s

    # MultiIndex kolon yapısı
    if isinstance(df.columns, pd.MultiIndex):
        for col in preferred_cols:
            for level in range(df.columns.nlevels):
                if col in df.columns.get_level_values(level):
                    sub = df.xs(col, axis=1, level=level)
                    if isinstance(sub, pd.DataFrame):
                        s = sub.iloc[:, 0].copy()
                    else:
                        s = sub.copy()
                    return s

    raise ValueError(
        f"{ticker} için Adj Close veya Close kolonu bulunamadı. "
        f"Mevcut kolonlar: {df.columns}"
    )


def download_close_price(ticker: str, start_date: str, end_date: str) -> pd.Series:
    """
    Yahoo Finance üzerinden tek ticker için kapanış fiyatı indirir.
    """
    end_date_inclusive = get_yfinance_end_date_inclusive(end_date)

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date_inclusive,
        auto_adjust=False,
        progress=False
    )

    s = extract_price_column(df, ticker)
    s.name = ticker
    s = s.dropna()

    return s


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    RSI14 hesaplar.

    RSI, fiyat artışlarının ve azalışlarının göreli gücünü ölçen teknik göstergedir.
    Bu tezde RSI, full feature set içinde ablasyon amacıyla kullanılmaktadır.
    """
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_features_for_asset(close: pd.Series, asset_name: str) -> pd.DataFrame:
    """
    Tek bir varlık için feature set üretir.

    Üretilen kolonlar:
        LogRet
        Vol20
        MA5_Ratio
        MA20_Ratio
        RSI14
        MACD
        MACDSignal
    """
    out = pd.DataFrame(index=close.index)

    logret = np.log(close / close.shift(1))

    vol20 = logret.rolling(
        window=20,
        min_periods=20
    ).std() * np.sqrt(252)

    ma5 = close.rolling(window=5, min_periods=5).mean()
    ma20 = close.rolling(window=20, min_periods=20).mean()

    ma5_ratio = close / ma5 - 1
    ma20_ratio = close / ma20 - 1

    rsi14 = compute_rsi(close, window=14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    out[f"{asset_name}_LogRet"] = logret
    out[f"{asset_name}_Vol20"] = vol20
    out[f"{asset_name}_MA5_Ratio"] = ma5_ratio
    out[f"{asset_name}_MA20_Ratio"] = ma20_ratio
    out[f"{asset_name}_RSI14"] = rsi14
    out[f"{asset_name}_MACD"] = macd
    out[f"{asset_name}_MACDSignal"] = macd_signal

    return out


def compute_targets_for_asset(close: pd.Series, asset_name: str) -> pd.DataFrame:
    """
    Tek bir varlık için t+1 hedeflerini üretir.

    NextRet:
        Bir sonraki günün log getirisi.

    NextVol:
        Bir sonraki günün 20 günlük yıllıklandırılmış volatilitesi.
    """
    out = pd.DataFrame(index=close.index)

    logret = np.log(close / close.shift(1))

    vol20 = logret.rolling(
        window=20,
        min_periods=20
    ).std() * np.sqrt(252)

    out[f"{asset_name}_NextRet"] = logret.shift(-1)
    out[f"{asset_name}_NextVol"] = vol20.shift(-1)

    return out


def assert_no_nan(df: pd.DataFrame, name: str):
    """
    Nihai dosyada eksik değer kalmadığını kontrol eder.
    """
    n_nan = int(df.isna().sum().sum())

    if n_nan != 0:
        raise ValueError(f"{name} içinde {n_nan} adet NaN var.")

    print(f"[OK] {name}: NaN yok.")


# ==========================================================
# 5. Verileri indir
# ==========================================================

print("\n[1/8] Veriler Yahoo Finance üzerinden indiriliyor...")

price_series = []

for asset_name in ASSET_ORDER:
    ticker = ASSETS[asset_name]

    print(f"  - {asset_name} ({ticker}) indiriliyor...")

    s = download_close_price(
        ticker=ticker,
        start_date=START_DATE,
        end_date=END_DATE
    )

    s.name = asset_name
    price_series.append(s)

raw_prices = pd.concat(price_series, axis=1).sort_index()

raw_prices_path = os.path.join(RAW_DIR, "raw_prices.csv")
raw_prices.to_csv(raw_prices_path)

print("[KAYIT] Ham fiyat dosyası:", raw_prices_path)
print("Ham fiyat shape:", raw_prices.shape)
print("Ham tarih aralığı:", raw_prices.index.min(), "→", raw_prices.index.max())


# ==========================================================
# 6. Fiyatları hizala ve eksikleri doldur
# ==========================================================

print("\n[2/8] Fiyatlar ortak takvimde hizalanıyor...")

prices_clean = raw_prices.copy()

# Karar defterindeki union calendar + ffill/bfill yaklaşımı
prices_clean = prices_clean.ffill().bfill()

prices_clean_path = os.path.join(PROCESSED_DIR, "prices_clean.csv")
prices_clean.to_csv(prices_clean_path)

print("[KAYIT] Temiz fiyat dosyası:", prices_clean_path)
print("Temiz fiyat shape:", prices_clean.shape)

assert_no_nan(prices_clean, "prices_clean")


# ==========================================================
# 7. Feature setlerini üret
# ==========================================================

print("\n[3/8] Feature setleri üretiliyor...")

feature_dfs = []

for asset_name in ASSET_ORDER:
    asset_features = compute_features_for_asset(
        close=prices_clean[asset_name],
        asset_name=asset_name
    )

    feature_dfs.append(asset_features)

features_full = pd.concat(feature_dfs, axis=1)

# Full feature kolon sırası kilitlenir
full_columns = []

for asset_name in ASSET_ORDER:
    for feat in FULL_FEATURES_PER_ASSET:
        full_columns.append(f"{asset_name}_{feat}")

features_full = features_full[full_columns]

# Baseline kolonları full içinden seçilir
baseline_columns = []

for asset_name in ASSET_ORDER:
    for feat in BASELINE_FEATURES_PER_ASSET:
        baseline_columns.append(f"{asset_name}_{feat}")

features_baseline = features_full[baseline_columns]

print("Full feature shape ham:", features_full.shape)
print("Baseline feature shape ham:", features_baseline.shape)


# ==========================================================
# 8. Target üret
# ==========================================================

print("\n[4/8] Target dosyası üretiliyor...")

target_dfs = []

for asset_name in ASSET_ORDER:
    asset_targets = compute_targets_for_asset(
        close=prices_clean[asset_name],
        asset_name=asset_name
    )

    target_dfs.append(asset_targets)

targets_all = pd.concat(target_dfs, axis=1)

# Target sırası kesin olarak kilitlenir
targets_all = targets_all[TARGET_NAMES]

print("Target shape ham:", targets_all.shape)


# ==========================================================
# 9. Ortak index üzerinde NaN temizliği
# ==========================================================

print("\n[5/8] Feature ve target dosyaları ortak index üzerinde temizleniyor...")

combined = pd.concat(
    [
        features_full,
        features_baseline.add_prefix("__BASE__"),
        targets_all
    ],
    axis=1
)

combined_clean = combined.dropna().copy()

features_full_clean = combined_clean[features_full.columns].copy()

features_baseline_clean = combined_clean[
    [f"__BASE__{col}" for col in features_baseline.columns]
].copy()

features_baseline_clean.columns = features_baseline.columns

targets_all_clean = combined_clean[TARGET_NAMES].copy()

print("Full feature shape temiz:", features_full_clean.shape)
print("Baseline feature shape temiz:", features_baseline_clean.shape)
print("Target shape temiz:", targets_all_clean.shape)


# ==========================================================
# 10. Kilitli kontroller
# ==========================================================

print("\n[6/8] Kilitli kontroller yapılıyor...")

expected_baseline_dim = schema["features"]["baseline"]["n_features"]
expected_full_dim = schema["features"]["full"]["n_features"]
expected_target_dim = schema["targets"]["n_targets"]

if features_baseline_clean.shape[1] != expected_baseline_dim:
    raise ValueError(
        f"Baseline feature boyutu {expected_baseline_dim} olmalıydı, "
        f"gelen: {features_baseline_clean.shape[1]}"
    )

if features_full_clean.shape[1] != expected_full_dim:
    raise ValueError(
        f"Full feature boyutu {expected_full_dim} olmalıydı, "
        f"gelen: {features_full_clean.shape[1]}"
    )

if targets_all_clean.shape[1] != expected_target_dim:
    raise ValueError(
        f"Target boyutu {expected_target_dim} olmalıydı, "
        f"gelen: {targets_all_clean.shape[1]}"
    )

if list(targets_all_clean.columns) != TARGET_NAMES:
    raise ValueError("Target sırası schema ile uyuşmuyor.")

if not features_full_clean.index.equals(features_baseline_clean.index):
    raise ValueError("Full ve baseline feature indexleri aynı değil.")

if not features_full_clean.index.equals(targets_all_clean.index):
    raise ValueError("Feature ve target indexleri aynı değil.")

assert_no_nan(features_full_clean, "features_full_clean")
assert_no_nan(features_baseline_clean, "features_baseline_clean")
assert_no_nan(targets_all_clean, "targets_all_clean")

print("[OK] Baseline feature boyutu doğru:", features_baseline_clean.shape[1])
print("[OK] Full feature boyutu doğru:", features_full_clean.shape[1])
print("[OK] Target boyutu doğru:", targets_all_clean.shape[1])
print("[OK] Target sırası kilitli ve doğru.")


# ==========================================================
# 11. Dosyaları kaydet
# ==========================================================

print("\n[7/8] Processed dosyaları kaydediliyor...")

features_baseline_path = os.path.join(PROCESSED_DIR, "features_baseline.csv")
features_full_path = os.path.join(PROCESSED_DIR, "features_full.csv")
targets_all_path = os.path.join(PROCESSED_DIR, "targets_all.csv")

features_baseline_clean.to_csv(features_baseline_path)
features_full_clean.to_csv(features_full_path)
targets_all_clean.to_csv(targets_all_path)

print("[KAYIT] Baseline feature:", features_baseline_path)
print("[KAYIT] Full feature:", features_full_path)
print("[KAYIT] Target:", targets_all_path)


# ==========================================================
# 12. Split meta üret
# ==========================================================

n = len(targets_all_clean)

train_end = int(n * schema["split"]["train_ratio"])
val_end = int(n * (schema["split"]["train_ratio"] + schema["split"]["validation_ratio"]))

dates = targets_all_clean.index

split_meta = {
    "n_total": int(n),

    "train": {
        "start_idx": 0,
        "end_idx_exclusive": int(train_end),
        "start_date": str(dates[0].date()),
        "end_date": str(dates[train_end - 1].date())
    },

    "validation": {
        "start_idx": int(train_end),
        "end_idx_exclusive": int(val_end),
        "start_date": str(dates[train_end].date()),
        "end_date": str(dates[val_end - 1].date())
    },

    "test": {
        "start_idx": int(val_end),
        "end_idx_exclusive": int(n),
        "start_date": str(dates[val_end].date()),
        "end_date": str(dates[-1].date())
    },

    "split_rule": "chronological_70_15_15",
    "note": "Bu dosyada split meta sadece kayıt amaçlıdır. Scaler ve sequence 02_preprocessing_v3.py içinde yapılacaktır."
}

split_meta_path = os.path.join(PROCESSED_DIR, "split_meta.json")

with open(split_meta_path, "w", encoding="utf-8") as f:
    json.dump(split_meta, f, ensure_ascii=False, indent=2)

print("[KAYIT] Split meta:", split_meta_path)


# ==========================================================
# 13. Meta dosyası
# ==========================================================

meta_v3 = {
    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "script": "01_veri_toplama_v3.py",
    "base_dir": BASE_DIR,
    "schema_path": schema_path,

    "data": {
        "assets": ASSET_ORDER,
        "tickers": ASSETS,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "source": "Yahoo Finance via yfinance",
        "price_column_priority": schema["data"]["price_column_priority"],
        "calendar_rule": schema["data"]["calendar_rule"]
    },

    "features": {
        "baseline_columns": list(features_baseline_clean.columns),
        "full_columns": list(features_full_clean.columns)
    },

    "targets": {
        "target_names": list(targets_all_clean.columns),
        "target_order_rule": schema["targets"]["rule"]
    },

    "shapes": {
        "raw_prices": list(raw_prices.shape),
        "prices_clean": list(prices_clean.shape),
        "features_baseline": list(features_baseline_clean.shape),
        "features_full": list(features_full_clean.shape),
        "targets_all": list(targets_all_clean.shape)
    },

    "split_meta_file": "data/processed/split_meta.json",

    "important_notes": [
        "Bu dosyada scaler uygulanmamıştır.",
        "Bu dosyada sequence üretilmemiştir.",
        "Bu dosyada model eğitilmemiştir.",
        "Scaler sadece 02_preprocessing_v3.py içinde train setine fit edilecektir.",
        "Target sırası kilitlidir: ilk 4 kolon NextRet, son 4 kolon NextVol."
    ]
}

meta_v3_path = os.path.join(PROCESSED_DIR, "meta_v3.json")

with open(meta_v3_path, "w", encoding="utf-8") as f:
    json.dump(meta_v3, f, ensure_ascii=False, indent=2)

print("[KAYIT] Meta dosyası:", meta_v3_path)


# ==========================================================
# 14. Final özet
# ==========================================================

print("\n==========================================================")
print("01_veri_toplama_v3.py BAŞARIYLA TAMAMLANDI")
print("==========================================================")
print("Ham fiyat shape:", raw_prices.shape)
print("Temiz fiyat shape:", prices_clean.shape)
print("Baseline feature shape:", features_baseline_clean.shape)
print("Full feature shape:", features_full_clean.shape)
print("Target shape:", targets_all_clean.shape)
print()
print("Temiz veri tarih aralığı:")
print(str(features_full_clean.index[0].date()), "→", str(features_full_clean.index[-1].date()))
print()
print("Train tarih aralığı:")
print(split_meta["train"]["start_date"], "→", split_meta["train"]["end_date"])
print()
print("Validation tarih aralığı:")
print(split_meta["validation"]["start_date"], "→", split_meta["validation"]["end_date"])
print()
print("Test tarih aralığı:")
print(split_meta["test"]["start_date"], "→", split_meta["test"]["end_date"])
print()
print("Üretilen dosyalar:")
print(" - data/raw/raw_prices.csv")
print(" - data/processed/prices_clean.csv")
print(" - data/processed/features_baseline.csv")
print(" - data/processed/features_full.csv")
print(" - data/processed/targets_all.csv")
print(" - data/processed/split_meta.json")
print(" - data/processed/meta_v3.json")
print("==========================================================")
