"""Arşivde kayıtlı test tarihlerini, Vol20 hedeflerini ve tahminleri doğrular."""
import hashlib
import json

import numpy as np
import pandas as pd


VARLIKLAR = ["BIST100", "USDTRY", "EURTRY", "GOLD"]
HAM_SHA256 = "ab5f275d38dc98057b1cedcf58019adb26be7402c7ed5ae6ee3d6877b2444893"


def veri_yukle(proje):
    ham = proje / "data/raw/raw_prices.csv"
    if hashlib.sha256(ham.read_bytes()).hexdigest() != HAM_SHA256:
        raise ValueError("Ham fiyat dosyasının SHA-256 özeti arşiv kaydıyla uyuşmuyor.")
    diziler = proje / "data/sequences/baseline/lb10"
    son = proje / "results/final_test"
    meta = json.loads((diziler / "sequence_meta.json").read_text(encoding="utf-8"))
    hedef_sirasi = [f"{v}_{gorev}" for gorev in ("NextRet", "NextVol") for v in VARLIKLAR]
    if meta["target_columns"] != hedef_sirasi:
        raise ValueError("Volatilite sütunlarının sırası beklenen sıradan farklı.")
    fiyat = pd.read_csv(ham, parse_dates=["Date"], index_col="Date")[VARLIKLAR].ffill()
    ankraj = pd.DatetimeIndex(np.load(diziler / "anchor_dates_test.npy", allow_pickle=False))
    tarih = pd.DatetimeIndex(np.load(diziler / "target_realization_dates_test.npy", allow_pickle=False))
    if not fiyat.index.is_unique or not fiyat.index.is_monotonic_increasing:
        raise ValueError("Ortak takvim sıralı ve benzersiz olmalı.")
    indis = fiyat.index.get_indexer(tarih)
    if (len(indis) != 584 or tarih[0] != pd.Timestamp("2022-10-06")
            or tarih[-1] != pd.Timestamp("2024-12-31") or np.any(indis <= 0)
            or np.any(np.diff(indis) != 1) or not ankraj.equals(fiyat.index[indis - 1])):
        raise ValueError("Test hedef tarihleri ve önceki gözlem tarihleri uyuşmuyor.")
    getiri = np.log(fiyat / fiyat.shift(1))
    vol = getiri.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252)
    hedef = np.load(son / "final_test_y_true_raw_v4.npy", allow_pickle=False)
    tahmin = np.load(son / "pred_final_ensemble_raw_v4.npy", allow_pickle=False)
    sureklilik = np.load(proje / "results/baselines/naive/pred_vol_persistence_raw_v4.npy",
                         allow_pickle=False)
    if hedef.shape != (584, 8) or tahmin.shape != hedef.shape or sureklilik.shape != (584, 4):
        raise ValueError("Hedef veya tahmin dizilerinin boyutları beklenen boyutlardan farklı.")
    tohumlar = [np.load(son / f"pred_final_seed{s}_raw_v4.npy", allow_pickle=False)
                for s in (123, 777, 2026)]
    np.testing.assert_allclose(tahmin, np.mean(tohumlar, axis=0), atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(hedef[:, :4], getiri.loc[tarih], rtol=0, atol=1e-6)
    np.testing.assert_allclose(hedef[:, 4:], vol.loc[tarih], rtol=0, atol=1e-6)
    np.testing.assert_allclose(sureklilik, vol.loc[ankraj], rtol=0, atol=1e-6)
    for seri in [hedef[:, 4:], tahmin[:, 4:], sureklilik]:
        if not np.isfinite(seri).all() or (seri <= 0).any():
            raise ValueError("Volatilite serilerinin tüm değerleri sonlu ve pozitif olmalı.")
    return tarih, hedef[:, 4:], tahmin[:, 4:], sureklilik
