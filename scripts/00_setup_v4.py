
# ==========================================================
# 00_setup_v4.py
# TEZ v4 — TEMİZ VE YENİDEN ÜRETİLEBİLİR PROJE KURULUMU
# ==========================================================

import os
import json
import hashlib
from datetime import datetime


# ==========================================================
# 1. ANA PROJE YOLU
# ==========================================================

BASE_DIR = "/content/drive/MyDrive/tez_transformer_v4_repro"


# ==========================================================
# 2. KLASÖR YAPISI
# ==========================================================

FOLDERS = [
    "config",
    "scripts",

    "data",
    "data/raw",
    "data/processed",

    "data/sequences",
    "data/sequences/baseline",
    "data/sequences/full",

    "models",

    "results",
    "results/baselines",
    "results/small_model_test",
    "results/grid_search",
    "results/multiseed",
    "results/final_test",
    "results/dm_tests",
    "results/shap",
    "results/robustness",

    "logs",
]


for rel_path in FOLDERS:
    full_path = os.path.join(BASE_DIR, rel_path)
    os.makedirs(full_path, exist_ok=True)


# ==========================================================
# 3. FROZEN RAW VERİ DOĞRULAMASI
# ==========================================================

RAW_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "raw_prices.csv"
)

EXPECTED_RAW_SHA256 = (
    "ab5f275d38dc98057b1cedcf58019adb26be7402c7ed5ae6ee3d6877b2444893"
)


def sha256_file(path, chunk_size=1024 * 1024):

    sha256 = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(chunk_size)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


if not os.path.exists(RAW_PATH):

    raise FileNotFoundError(
        f"Frozen raw veri bulunamadı:\n{RAW_PATH}"
    )


actual_raw_hash = sha256_file(RAW_PATH)


if actual_raw_hash != EXPECTED_RAW_SHA256:

    raise RuntimeError(
        "Frozen raw veri hash'i beklenen değerle eşleşmiyor.\n"
        f"Beklenen: {EXPECTED_RAW_SHA256}\n"
        f"Gerçek   : {actual_raw_hash}"
    )


# ==========================================================
# 4. RESMÎ v4 ŞEMASI
# ==========================================================

schema = {

    "project_version": "v4_repro",

    "project_title":
        "Çoklu Görevli Transformer ile Finansal Risk ve Getiri Tahmini",

    "created_at": datetime.now().isoformat(),

    "data": {

        "source":
            "Frozen raw_prices.csv inherited from audited v3 source",

        "internet_redownload":
            False,

        "raw_file":
            "data/raw/raw_prices.csv",

        "raw_sha256":
            actual_raw_hash,

        "assets": [
            "BIST100",
            "USDTRY",
            "EURTRY",
            "GOLD"
        ],

        "tickers": {
            "BIST100": "XU100.IS",
            "USDTRY": "USDTRY=X",
            "EURTRY": "EURTRY=X",
            "GOLD": "GC=F"
        },

        "original_period": {
            "start": "2010-01-01",
            "end": "2024-12-31"
        }
    },


    "features": {

        "baseline": [
            "LogRet",
            "Vol20"
        ],

        "full": [
            "LogRet",
            "Vol20",
            "MA5_Ratio",
            "MA20_Ratio",
            "RSI14",
            "MACD",
            "MACDSignal"
        ],

        "baseline_dim": 8,

        "full_dim": 28,

        "rsi14_zero_loss_rule":
            "If avg_loss == 0 and avg_gain > 0, RSI14 = 100"
    },


    "targets": {

        "definition": [
            "BIST100_NextRet",
            "USDTRY_NextRet",
            "EURTRY_NextRet",
            "GOLD_NextRet",
            "BIST100_NextVol",
            "USDTRY_NextVol",
            "EURTRY_NextVol",
            "GOLD_NextVol"
        ],

        "return_rule":
            "NextRet[t] = LogRet[t+1]",

        "volatility_rule":
            "NextVol[t] = Vol20[t+1]"
    },


    "split": {

        "type":
            "chronological_target_realization_aware",

        "ratios": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15
        },

        "rule":
            "Input history may cross backward into the previous split, "
            "but target realization may never cross forward into the next split.",

        "random_split":
            False
    },


    "scaler": {

        "type":
            "StandardScaler",

        "fit_on":
            "train_only",

        "validation":
            "transform_only",

        "test":
            "transform_only"
    },


    "sequence": {

        "lookbacks": [
            10,
            20,
            30,
            60
        ],

        "overlap_aware":
            True,

        "principle":
            "Past input window may be carried; target may not be carried."
    },


    "models": [

        "FullSharingMTL",
        "PartialSharingMTL",
        "HierarchicalMTL",
        "NoSharing"
    ],


    "loss_strategies": [

        "FixedLambda_0.3",
        "FixedLambda_0.5",
        "FixedLambda_0.7",
        "UncertaintyWeighting",
        "PCGrad"
    ],


    "model_sizes": {

        "small": {
            "d_model": 32,
            "n_head": 4,
            "n_layers": 2,
            "d_ff": 128
        },

        "medium": {
            "d_model": 64,
            "n_head": 4,
            "n_layers": 2,
            "d_ff": 256
        },

        "large": {
            "d_model": 128,
            "n_head": 8,
            "n_layers": 4,
            "d_ff": 512
        }
    },


    "grid": {

        "architectures": 4,
        "loss_strategies": 5,
        "lookbacks": 4,
        "sizes": 3,
        "feature_sets": 2,

        "total_configs": 480,

        "official_grid_max_epochs": 50
    },


    "test_policy": {

        "used_for_model_selection":
            False,

        "first_model_evaluation_stage":
            "07_final_test_evaluation_v4.py",

        "post_test_model_changes_allowed":
            False
    },


    "scientific_principle":
        "Kararlar kilitli. Sonuçlar kilitli değil. Veri karar verir."
}


schema_path = os.path.join(
    BASE_DIR,
    "config",
    "schema_v4.json"
)


with open(
    schema_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        schema,
        f,
        ensure_ascii=False,
        indent=2
    )


# ==========================================================
# 5. README_v4.md
# ==========================================================

readme_path = os.path.join(
    BASE_DIR,
    "README_v4.md"
)


readme_text = f"""# TEZ TRANSFORMER v4 REPRO

## Amaç

Bu proje, dört finansal varlık için getiri ve volatilitenin
Transformer tabanlı çok görevli mimariler ile tahmin edilmesini amaçlar.

## Varlıklar

- BIST100
- USDTRY
- EURTRY
- GOLD

## Bilimsel İlke

**Kararlar kilitli. Sonuçlar kilitli değil. Veri karar verir.**

## Frozen Raw Data

Dosya:

`data/raw/raw_prices.csv`

SHA-256:

`{actual_raw_hash}`

Bu veri dosyası yeniden internetten indirilmemiştir.
Audit edilmiş v3 kaynağındaki frozen raw veri birebir korunmuştur.

## v4 Temel Düzeltmeleri

1. RSI14 için zero-loss handling açıkça tanımlanmıştır.
2. Split, target-realization-aware olarak uygulanacaktır.
3. Input history taşınabilir; target split sınırını geçemez.
4. StandardScaler yalnızca train setine fit edilir.
5. Test model seçiminde kullanılmaz.

## Resmî Pipeline

- 00_setup_v4.py
- 01_rebuild_from_frozen_raw_v4.py
- 02_preprocessing_v4.py
- 03_baseline_sanity_v4.py
- 04_small_model_test_v4.py
- 05_grid_search_v4.py
- 06_best_model_multiseed_v4.py
- 07_final_test_evaluation_v4.py
- 08_baseline_full_v4.py
- 09_diebold_mariano_v4.py

"""


with open(
    readme_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(readme_text)


# ==========================================================
# 6. SONUÇ
# ==========================================================

print("=" * 70)
print("00_setup_v4.py TAMAMLANDI")
print("=" * 70)

print("\nProje klasörü:")
print(BASE_DIR)

print("\nFrozen raw veri:")
print(RAW_PATH)

print("\nFrozen raw SHA-256:")
print(actual_raw_hash)

print("\nHash doğrulandı:")
print(actual_raw_hash == EXPECTED_RAW_SHA256)

print("\nSchema:")
print(schema_path)

print("\nREADME:")
print(readme_path)

print("\nKlasör sayısı:")
print(len(FOLDERS))

print("\n✅ v4 proje iskeleti hazır.")
