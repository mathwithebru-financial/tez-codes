# Stage 10 v4a — SHAP Çıktı Ölçeği Yöntem Eki

## Durum

`LOCKED_BEFORE_SHAP_COMPUTATION`

Bu belge, orijinal Stage 10 SHAP protokolünü değiştirmez veya
üzerine yazmaz. Yalnızca ham hedef ölçeğine dönüşümün sayısal
uygulamasını sonuçlar hesaplanmadan önce açıklığa kavuşturur.

## Tanı Sonucu

Resmî tahmin üretim yolu:

`PyTorch ölçekli tahmin → sklearn StandardScaler.inverse_transform`

Bu yolla üç seed için kayıtlı ham tahminler tam olarak yeniden
üretilmiştir:

- Seed 123 maksimum fark: 0.0
- Seed 777 maksimum fark: 0.0
- Seed 2026 maksimum fark: 0.0
- Üç kayıtlı seed ortalaması ile kayıtlı ensemble farkı:
  0.0

PyTorch grafiği içinde float32 çarpma ve toplama kullanılması,
işlem sırasına bağlı olarak en fazla
`5.960464477539063e-08` fark üretmiştir. Bu fark model,
checkpoint, veri veya scaler kimliği hatası değildir.

## Kilitlenen Uygulama

1. GradientExplainer, üç seed'in standartlaştırılmış çıktılarının
   türevlenebilir aritmetik ortalamasını açıklar.
2. Her hedef için ölçekli SHAP katkısı ilgili
   `y_scaler.scale_` katsayısıyla çarpılır.
3. Ham beklenen çıktı:

   `E_raw = E_scaled × scale_ + mean_`

4. Ham SHAP katkısı:

   `phi_raw = phi_scaled × scale_`

5. SHAP varyansı:

   `variance_raw = variance_scaled × scale_^2`

6. Kayıtlı ham tahmin dosyaları nihai tahmin referansı olarak
   korunur.

## Değişmeyen Kararlar

- FinalWinner_3SeedEnsemble
- NoSharing mimarisi
- Seed 123, 777 ve 2026
- GradientExplainer / Expected Gradients
- Eğitim verisinden 64 arka plan penceresi
- 584 test gözlemi
- Sekiz çıktı
- nsamples = 256
- rastgelelik tohumu = 2026
- sonuç öncesi sabitlenen yerel örnekler
- yorum sınırları

## Bilimsel Sınırlar

- Sayısal tolerans eklenmemiştir.
- Model eğitilmemiş veya yeniden uyarlanmamıştır.
- Kazanan model değiştirilmemiştir.
- Özellik seçimi yapılmamıştır.
- SHAP sonucu görülmeden bu yöntem eki oluşturulmuştur.
