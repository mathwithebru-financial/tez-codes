# Stage 10 — Final Post-Hoc SHAP Protokol Kilidi

## Durum

Bu protokol SHAP katkı değerleri, özellik önem sıralamaları ve
görseller üretilmeden önce oluşturulmuş ve kilitlenmiştir.

Durum:

`LOCKED_BEFORE_SHAP_COMPUTATION`

## Açıklanan Model

- Nihai tahmin: `FinalWinner_3SeedEnsemble`
- Mimari: `NoSharing`
- Seed'ler: `123, 777, 2026`
- Ensemble: üç seed tahmininin ham hedef ölçeğindeki aritmetik ortalaması
- Girdi: 10 zaman adımı × 8 özellik
- Çıktı: 4 getiri + 4 volatilite tahmini

## SHAP Yöntemi

Birincil yöntem:

`SHAP GradientExplainer / Expected Gradients`

- Arka plan kümesi yalnızca eğitim verisinden seçilecektir.
- Arka plan büyüklüğü: 64
- Seçim: eğitim dönemine kronolojik olarak eşit aralıklı gerçek pencereler
- Açıklanan test gözlemi: 584
- Monte Carlo örnek sayısı: 256
- Rastgelelik tohumu: 2026
- Bütün sekiz çıktı açıklanacaktır.

## Çıktı Ölçeği

SHAP katkıları ham hedef ölçeğinde hesaplanacaktır.

`y_scaler` ters dönüşümü PyTorch model sarmalayıcısına kesin bir
afin dönüşüm olarak eklenecek ve SHAP hesaplanmadan önce kayıtlı
ham tahminlerle sıfır farkla doğrulanacaktır.

## Toplulaştırma

- Hedef bazlı özellik önemi:
  örnekler ve gecikmeler üzerinden ortalama mutlak SHAP
- Hedef bazlı zaman önemi:
  örnekler ve özellikler üzerinden ortalama mutlak SHAP
- Özellik-gecikme ısı haritası:
  test örnekleri üzerinden ortalama mutlak SHAP
- Getiri görev özeti:
  dört getiri çıktısının ortalaması
- Volatilite görev özeti:
  dört volatilite çıktısının ortalaması

## Yerel Açıklamalar

Sonuç görülmeden önce sabitlenen test konumları:

`[0, 194, 389, 583]`

Tarihler:

`['2022-10-05', '2023-07-04', '2024-04-02', '2024-12-30']`

## Yorum Sınırları

- SHAP katkısı nedensel etki değildir.
- Sonuçlar modelin davranışını açıklar.
- SHAP ile özellik seçimi yapılmayacaktır.
- SHAP ile mimari veya hiperparametre değiştirilmeyecektir.
- Nihai model yeniden seçilmeyecektir.
- SHAP tek başına negatif transfer kanıtı olarak kullanılmayacaktır.

## Bu Kilit Aşamasında Yapılmayanlar

- Model eğitilmedi.
- Model yeniden uyarlanmadı.
- Yeni tahmin üretilmedi.
- SHAP değeri hesaplanmadı.
- Özellik sıralaması oluşturulmadı.
