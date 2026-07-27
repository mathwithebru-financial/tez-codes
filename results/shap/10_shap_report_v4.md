# Stage 10 — Nihai SHAP Açıklanabilirlik Raporu

## Durum

`SHAP_COMPUTED_PENDING_INDEPENDENT_AUDIT`

Bu aşamada model yeniden eğitilmemiş, yeniden uyarlanmamış veya yeniden seçilmemiştir.

## Yöntem

- Referans: `FinalWinner_3SeedEnsemble`
- Mimari: `NoSharing`
- Seed'ler: `123`, `777`, `2026`
- Açıklayıcı: `GradientExplainer / Expected Gradients`
- Eğitim verisinden arka plan pencere sayısı: `64`
- Açıklanan test gözlemi: `584`
- Çıktı sayısı: `8`
- `nsamples = 256`
- Rastgelelik tohumu: `2026`
- Hesaplama ölçeği: standartlaştırılmış hedef ölçeği
- Raporlama ölçeği: ham hedef ölçeği

Ham ölçekli SHAP katkıları, v4a yöntem ekinde kilitlendiği biçimde her hedefin `StandardScaler.scale_` katsayısıyla elde edilmiştir.

## Veri Bütünlüğü

- SHAP tensörü: `(584, 10, 8, 8)`
- Varyans tensörü: `(584, 10, 8, 8)`
- Chunk sayısı: `37`
- Kapsanan test indeksleri: `0–583`
- NaN/Inf: bulunmadı

## Additivity Tanısı

Expected Gradients Monte Carlo yaklaşımı nedeniyle additivity farkları tanısal olarak raporlanmıştır. Sonuç görüldükten sonra bir kabul eşiği tanımlanmamıştır.

- Genel ortalama mutlak artık: `0.00125155026806`
- Genel medyan mutlak artık: `0.000570342264973`
- Genel maksimum mutlak artık: `0.0121249987516`
- Genel artık RMSE: `0.00197723281361`

## Çıktı Bazında En Önemli Üç Özellik

### BIST100_NextRet

1. `BIST100_LogRet` — ortalama |SHAP| = `0.000205219687401`, çıktı içi pay = `20.4429%`
2. `USDTRY_Vol20` — ortalama |SHAP| = `0.000191307723657`, çıktı içi pay = `19.0570%`
3. `BIST100_Vol20` — ortalama |SHAP| = `0.000171661225623`, çıktı içi pay = `17.1000%`

### USDTRY_NextRet

1. `BIST100_LogRet` — ortalama |SHAP| = `0.00035748652775`, çıktı içi pay = `36.7669%`
2. `BIST100_Vol20` — ortalama |SHAP| = `0.000128552911869`, çıktı içi pay = `13.2215%`
3. `GOLD_LogRet` — ortalama |SHAP| = `9.73399965365e-05`, çıktı içi pay = `10.0113%`

### EURTRY_NextRet

1. `BIST100_LogRet` — ortalama |SHAP| = `0.000301101591368`, çıktı içi pay = `34.5425%`
2. `BIST100_Vol20` — ortalama |SHAP| = `0.000127842483306`, çıktı içi pay = `14.6662%`
3. `USDTRY_Vol20` — ortalama |SHAP| = `8.84925294016e-05`, çıktı içi pay = `10.1519%`

### GOLD_NextRet

1. `USDTRY_Vol20` — ortalama |SHAP| = `0.000171135658482`, çıktı içi pay = `19.6203%`
2. `BIST100_LogRet` — ortalama |SHAP| = `0.000165705341714`, çıktı içi pay = `18.9977%`
3. `BIST100_Vol20` — ortalama |SHAP| = `0.000161308125665`, çıktı içi pay = `18.4936%`

### BIST100_NextVol

1. `BIST100_Vol20` — ortalama |SHAP| = `0.00879549400606`, çıktı içi pay = `83.4809%`
2. `USDTRY_Vol20` — ortalama |SHAP| = `0.000559076146371`, çıktı içi pay = `5.3064%`
3. `BIST100_LogRet` — ortalama |SHAP| = `0.000385106961199`, çıktı içi pay = `3.6552%`

### USDTRY_NextVol

1. `USDTRY_Vol20` — ortalama |SHAP| = `0.00723171257532`, çıktı içi pay = `80.9701%`
2. `EURTRY_Vol20` — ortalama |SHAP| = `0.000673320357267`, çıktı içi pay = `7.5389%`
3. `BIST100_Vol20` — ortalama |SHAP| = `0.000456886080983`, çıktı içi pay = `5.1155%`

### EURTRY_NextVol

1. `EURTRY_Vol20` — ortalama |SHAP| = `0.0037246212368`, çıktı içi pay = `64.4155%`
2. `USDTRY_Vol20` — ortalama |SHAP| = `0.00125904601719`, çıktı içi pay = `21.7746%`
3. `BIST100_Vol20` — ortalama |SHAP| = `0.000280971971302`, çıktı içi pay = `4.8593%`

### GOLD_NextVol

1. `GOLD_Vol20` — ortalama |SHAP| = `0.00297585919377`, çıktı içi pay = `70.1383%`
2. `BIST100_Vol20` — ortalama |SHAP| = `0.00037120767222`, çıktı içi pay = `8.7490%`
3. `USDTRY_Vol20` — ortalama |SHAP| = `0.00035109831989`, çıktı içi pay = `8.2751%`

## Çıktı Bazında En Önemli Üç Gecikme

### BIST100_NextRet

1. `t` — ortalama |SHAP| = `0.000502324618258`, çıktı içi pay = `40.0311%`
2. `t-1` — ortalama |SHAP| = `9.42283199577e-05`, çıktı içi pay = `7.5092%`
3. `t-4` — ortalama |SHAP| = `9.33969364611e-05`, çıktı içi pay = `7.4430%`

### USDTRY_NextRet

1. `t` — ortalama |SHAP| = `0.000618689494731`, çıktı içi pay = `50.9050%`
2. `t-1` — ortalama |SHAP| = `8.07955439696e-05`, çıktı içi pay = `6.6478%`
3. `t-7` — ortalama |SHAP| = `7.98971824101e-05`, çıktı içi pay = `6.5738%`

### EURTRY_NextRet

1. `t` — ortalama |SHAP| = `0.000516086903219`, çıktı içi pay = `47.3646%`
2. `t-1` — ortalama |SHAP| = `7.7755987106e-05`, çıktı içi pay = `7.1362%`
3. `t-9` — ortalama |SHAP| = `7.39675109403e-05`, çıktı içi pay = `6.7885%`

### GOLD_NextRet

1. `t` — ortalama |SHAP| = `0.000386059808048`, çıktı içi pay = `35.4086%`
2. `t-1` — ortalama |SHAP| = `9.85678351515e-05`, çıktı içi pay = `9.0404%`
3. `t-7` — ortalama |SHAP| = `8.83581560729e-05`, çıktı içi pay = `8.1040%`

### BIST100_NextVol

1. `t` — ortalama |SHAP| = `0.0119034780495`, çıktı içi pay = `90.3838%`
2. `t-1` — ortalama |SHAP| = `0.000168337000323`, çıktı içi pay = `1.2782%`
3. `t-2` — ortalama |SHAP| = `0.000161183436136`, çıktı içi pay = `1.2239%`

### USDTRY_NextVol

1. `t` — ortalama |SHAP| = `0.00951693599105`, çıktı içi pay = `85.2453%`
2. `t-1` — ortalama |SHAP| = `0.00020849468747`, çıktı içi pay = `1.8675%`
3. `t-3` — ortalama |SHAP| = `0.000200130061388`, çıktı içi pay = `1.7926%`

### EURTRY_NextVol

1. `t` — ortalama |SHAP| = `0.00587730659763`, çıktı içi pay = `81.3161%`
2. `t-1` — ortalama |SHAP| = `0.000181416605012`, çıktı içi pay = `2.5100%`
3. `t-3` — ortalama |SHAP| = `0.000168094589578`, çıktı içi pay = `2.3257%`

### GOLD_NextVol

1. `t` — ortalama |SHAP| = `0.00478041728011`, çıktı içi pay = `90.1361%`
2. `t-8` — ortalama |SHAP| = `7.04644206101e-05`, çıktı içi pay = `1.3286%`
3. `t-2` — ortalama |SHAP| = `6.88812956323e-05`, çıktı içi pay = `1.2988%`

## Yerel Açıklamalar

Protokolde sonuç öncesi belirlenen yerel test pozisyonları: `0`, `194`, `389`, `583`.

Her yerel örnek ve çıktı için imzalı özellik-gecikme katkıları `10_shap_figures_v4` klasöründe verilmiştir.

## Yorumlama Sınırları

- SHAP değerleri model davranışını açıklar; nedensellik göstermez.
- Ham SHAP büyüklükleri farklı hedef birimleri arasında doğrudan karşılaştırılmamalıdır.
- Bu sonuçlar özellik seçimi, model yeniden seçimi veya hiperparametre değişikliği için kullanılmamıştır.
- Negatif transfer yalnızca SHAP sonuçlarına dayanılarak ileri sürülmemelidir.

## Üretilen Çıktılar

- `10_shap_values_ensemble_raw_v4.npz`
- `10_shap_global_importance_v4.csv`
- `10_shap_temporal_importance_v4.csv`
- `10_shap_quality_diagnostics_v4.json`
- `10_shap_summary_v4.json`
- `10_shap_report_v4.md`
- `10_shap_figures_v4/` — 48 görsel
