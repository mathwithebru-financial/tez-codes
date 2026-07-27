# Stage 09 — DM, Harvey ve Holm Protokol Kilidi

## Durum

Bu protokol, istatistiksel test sonuçları görülmeden önce
oluşturulmuş ve kilitlenmiştir.

- Referans model: `FinalWinner_3SeedEnsemble`
- Kazanan konfigürasyon: `arch=NoSharing__loss=FixedLambda_0.7__lb=10__size=small__feat=baseline`
- Test gözlemi: `584`
- Tahmin ufku: `h = 1`
- Toplam karşılaştırma: `42`

## Birincil Kayıp Serileri

- Getiri: mutlak hata; toplulaştırılmış karşılığı MAE.
- Volatilite: Pinball loss, tau = 0.5.

Kayıp farkı şu yönde tanımlanmıştır:

`d_t = baseline_loss_t - final_model_loss_t`

Pozitif ortalama kayıp farkı, nihai Transformer modelinin daha düşük
kayıp ürettiğini gösterir.

## Diebold–Mariano Testi

Birincil hipotez testi iki yönlüdür.

- H0: Modellerin beklenen tahmin doğruluğu eşittir.
- H1: Beklenen tahmin doğrulukları farklıdır.
- Tahmin ufku bir gün olduğu için DM uzun dönem varyansında
  maksimum gecikme `h - 1 = 0` olarak kilitlenmiştir.
- Ham DM p-değeri standart normal dağılımdan raporlanacaktır.

## Harvey–Leybourne–Newbold Düzeltmesi

Küçük örneklem düzeltmesi uygulanacaktır.

- Düzeltilmiş istatistik:
  `DM_HLN = correction_factor × DM`
- Düzeltilmiş p-değeri:
  `t(T - 1)` dağılımından iki yönlü olarak hesaplanacaktır.
- Holm düzeltmesinin girdisi Harvey-düzeltilmiş p-değerleridir.

## Holm–Bonferroni Aileleri

İki ayrı birincil hipotez ailesi kilitlenmiştir:

1. Getiri ailesi: 20 karşılaştırma.
2. Volatilite ailesi: 22 karşılaştırma.

Aileler varlık veya baseline türüne göre daha küçük alt gruplara
bölünmeyecektir.

Anlamlılık düzeyi `alpha = 0.05` olarak kilitlenmiştir.

## GARCH Kapsamı

BIST100, EURTRY ve GOLD için GARCH ve GJR-GARCH karşılaştırmaları
dahil edilmiştir.

USDTRY için iki GARCH-family serisi tam 584 gözlem sağlamadığı için
istatistiksel karşılaştırmaya dahil edilmemiştir. USDTRY tezden
çıkarılmamıştır ve diğer baseline, Transformer ve SHAP analizlerinde
kalmaktadır.

## Yorumlama Kuralları

- Holm-düzeltilmiş p < 0.05 ve ortalama kayıp farkı pozitif:
  nihai Transformer anlamlı derecede daha iyi.
- Holm-düzeltilmiş p < 0.05 ve ortalama kayıp farkı negatif:
  baseline anlamlı derecede daha iyi.
- Holm-düzeltilmiş p >= 0.05:
  istatistiksel olarak anlamlı fark saptanmamıştır.

Anlamlı olmayan sonuç, modellerin eşdeğer olduğunu kanıtlamaz.

## Çalıştırma Sınırları

Bu kilit aşamasında:

- model eğitilmemiştir,
- tahmin üretilmemiştir,
- DM testi çalıştırılmamıştır,
- p-değeri hesaplanmamıştır,
- mevcut sonuç dosyaları değiştirilmemiştir.
