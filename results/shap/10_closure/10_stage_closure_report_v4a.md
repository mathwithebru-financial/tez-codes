# Stage 10 v4a — SHAP Aşama Kapanış Raporu

## Durum

`CLOSED`

Stage 10, nihai SHAP sonuçlarının salt okunur bağımsız
denetimden geçmesinin ardından kapatılmıştır.

## Kapsam

- Referans model: `FinalWinner_3SeedEnsemble`
- Mimari: `NoSharing`
- Seed'ler: `123`, `777`, `2026`
- Test gözlemleri: `584`
- Geriye bakış uzunluğu: `10`
- Girdi özelliği: `8`
- Açıklanan çıktı: `8`
- Kanonik SHAP tensörü: `(584, 10, 8, 8)`
- SHAP chunk sayısı: `37`
- Görsel sayısı: `48`
- Test indeksleri: `0–583`

## Yöntem

- `GradientExplainer`
- `Expected Gradients`
- Eğitim verisinden `64` arka plan penceresi
- `nsamples = 256`
- Rastgelelik tohumu: `2026`
- Hesaplama: standartlaştırılmış hedef ölçeği
- Raporlama: ham hedef ölçeği

## Bağımsız Denetim

Aşağıdaki kontroller PASS sonucu vermiştir:

- Altı nihai dosyanın SHA kimlikleri
- Kanonik SHAP tensör şekli
- Tam test indeks kapsamı
- 37 chunk dosyasının SHA kimlikleri
- Chunk ve nihai NPZ tam eşitliği
- Additivity istatistiklerinin yeniden hesaplanması
- Global önem sıralamalarının yeniden hesaplanması
- Zamansal önem sıralamalarının yeniden hesaplanması
- Çıktı bazında ilk üç sıralamasının doğrulanması
- 48 görselin dosya bütünlüğü
- Model ve yöntem sınırları

## Additivity Tanısı

Bu değerler Expected Gradients Monte Carlo yaklaşımı için
tanısal olarak raporlanmıştır. Sonuç görüldükten sonra bir kabul
eşiği oluşturulmamıştır.

- Ortalama mutlak artık:
  `0.001251550268057652`
- Medyan mutlak artık:
  `0.0005703422649728448`
- Maksimum mutlak artık:
  `0.0121249987516453`
- Artık RMSE:
  `0.001977232813614278`

## Yorumlama Sınırları

- SHAP nedensellik göstermez.
- Farklı hedef birimleri arasındaki ham SHAP büyüklükleri
  doğrudan karşılaştırılmamalıdır.
- SHAP sonuçları model, özellik veya hiperparametre yeniden
  seçimi için kullanılmamıştır.
- Negatif transfer yalnızca SHAP sonuçlarına dayanılarak
  ileri sürülmeyecektir.

## Sonraki İş Akışı

`THESIS_RESULTS_INTERPRETATION_AND_WRITING`
