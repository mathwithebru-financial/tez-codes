# Bir ortak gözlem gecikmeli deney (P1 V2)

Bu klasör, ana model seçildikten sonra yapılan 2010–2024 sağlamlık incelemesini içerir. Ana deneyde `t+1` hedefi için son girdi `t` gözlemidir. Burada son girdi `t−1` olur. Her iki durumda da getiri hedefi `t` ile `t+1` arasındaki log getiridir. Volatilite hedefi, `t+1` ile biten 20 getirinin yıllıklandırılmış örnek standart sapmasıdır. Tarihler ortak günlük gözlemleri gösterir. Gün içi yayın ve revizyon saatleri eldeki seriden çıkarılamaz. [Varlık ve zaman bilgisi](protocol/price_and_clock_contract.csv)

## Ham veriden eğitime hazır dizilere

Bu komutları depo kökünden ve Python 3.13 ortamından başlayarak sırasıyla çalıştırın. Veri hazırlığında doğrulanan sürümler NumPy 2.3.3, pandas 3.0.1 ve scikit-learn 1.7.2'dir. Aşağıdaki akış CPU'da çalışır. `../p1_replay` henüz bulunmayan, depo dışında bir dizin olmalıdır.

```shell
python experiments/p1_v2/prepare_workspace.py ../p1_replay
cd ../p1_replay
python pipeline/independent_data_checks.py
python scoring/build_scoring_masks.py
python benchmarks/validation/build_validation_denominators.py
python pipeline/build_p1_datasets.py
python training/train_p1.py
```

Son komut varsayılan olarak ön kontrol yapar. Eğitim ayrı bir `--execute-train` seçeneği gerektirir. `prepare_workspace.py` ilk üç dosyayı ana depodan kopyalar (ham CSV ve iki P0 kaynağı), P1 çalışma yapısını kurar ve yeniden hesapladığı tarih eşlemesinin SHA-256 değerini özgün kayıtla karşılaştırır. Çalıştırma sırasındaki denetimlerde 3.912 fiyat tarihi, eğitimde 2.721 ölçekleme satırı ve eğitim/doğrulama/testte 2.712/583/584 örnek bulunmalıdır. Son dosyalar `datasets/p1_v2/` içindedir. Yayıma hazırlıkta temiz bir dizinde üretilen **67 NPY dizisinin** şekli, veri türü ve tüm değerleri saklanmış P1 dizileriyle birebir karşılaştırıldı.

## Hangi kod hangi soruyu yanıtlıyor?

| Soru | Ana kaynak | Beklenen ürün |
|---|---|---|
| Tarihler nasıl eşleşir? Hangi üç satır dışarıda kalır? | [Taşınabilir eşleme](prepare_workspace.py) · [Sınır denetimi](protocol/strict_information_boundary_validation.json) | Hedef tarihi, son girdi tarihi ve elenme nedeni |
| Hangi hedef veya girdi ileri doldurulmuş fiyat içerir? | [Puanlama maskeleri](scoring/build_scoring_masks.py) · [Kapsam tanımları](scoring/scoring_policy_v2.json) | Hedef ve girdi desteğine göre değerlendirme maskeleri |
| Doğrulama kıyaslamasının paydası nasıl hesaplanır? | [Referans kayıpları](benchmarks/validation/build_validation_denominators.py) · [Doğrulama skoru](benchmarks/validation/validation_score_contract.py) | 583 doğrulama hedefinde dört getiri, dört volatilite paydası |
| Ölçekleyici ve pencere nasıl oluşturulur? | [P1 veri hazırlığı](pipeline/build_p1_datasets.py) | X ve y için ayrı eğitim ölçekleri, tarihler ve tensörler |
| İki eğitim kolu hangi açıdan farklıdır? | [Yapılandırma](training/config.py) · [NoSharing](training/models.py) · [Kayıplar](training/losses.py) · [Eğitim](training/train_p1.py) | Getiride MSE veya MAE; volatilitede iki kolda da τ=0,5 pinball. 0,7/0,3 ağırlıkları ve üç tohum ortak |
| Tahminler nasıl karşılaştırılır? | [Sabit modellerle değerlendirme](training/comparison/evaluate_fixed_runs.py) · [Hata hesapları](training/comparison/numeric_metrics.py) | Üç ham tahminin ortalaması ve aynı bilgi sınırındaki basit referanslar |
| Alt dönem farkı nereden gelir? | [Dönem tanımları](analysis/subperiods_20261113/subperiod_protocol.json) · [Hesap](analysis/subperiods_20261113/analyze_subperiods.py) | 2022'nin testteki kısmı, 2023 ve 2024; gerçekleşme tarihine göre sırasıyla 62, 260 ve 262 hedef |
| SHAP sıralaması ne kadar değişir? | [Örneklem seçimi](analysis/shap_stability_20261113/sample_selection.py) · [Hesap](analysis/shap_stability_20261113/run_shap_stability_vectorized.py) · [Ölçütler](analysis/shap_stability_20261113/stability_metrics.py) · [Özetleme](analysis/shap_stability_20261113/analyze_saved_shap_stability.py) | 128 test penceresi; dört eğitim arka planı, üç model tohumu ve iki Monte Carlo tohumu |
| Çekiliş sayısı yerel açıklama artığını nasıl etkiler? | [Çekiliş kontrolü](analysis/shap_convergence_20261113/run_convergence.py) · [Özetleme](analysis/shap_convergence_20261113/summarize_convergence.py) | Seçilmiş sekiz pencere-koşulda 256, 1.024 ve 4.096 çekiliş |
| GARCH'ın daha uzun bilgi ufkuna uyarlaması nedir? | [Bilgi sınırı](benchmarks/garch/p1_information_adapter.py) · [İki getiri için dönüşüm](benchmarks/garch/two_increment_vol20_bridge.py) | Sentetik sayısal kontrol; piyasaya uyarlanmış bir P1 GARCH sonuç serisi üretilmedi |

## Kaydedilmiş sonuçlar

Tam testte getirinin dört varlık üzerinden ortalama MAE’si, MSE eğitiminde 0,007448, MAE eğitiminde 0,007098 ve sıfır getiri kıyaslamasında 0,006894’tür. Volatilitede aynı sırayla pinball kayıpları 0,007438, 0,007301 ve 0,004934'tür. Son değer, son erişilebilir volatiliteyi sürdüren referansa aittir. Bu P1 farklarına yeni bir anlamlılık testi uygulanmadı. [Dönem ve görev bazlı sayılar](results/primary_metrics.csv) · [Altı koşunun kayıtları ve ağırlık kimlikleri](results/six_runs.json)

SHAP kararlılık karşılaştırmaları ve sekiz örnek üzerindeki çekiliş deneyi farklı kapsamlara sahiptir. [Kararlılık özeti](results/SHAP_Kararlilik_Ozeti.txt) · [Yerel çekiliş duyarlılığı](results/SHAP_Cekilis_Duyarliligi_Notu.txt)

## Yeniden üretimin kapsamı

[Gecikmeli deney arşivi ve taşınabilir çalıştırma komutları](../../docs/TEZ_ARSIVLERI.md) bölümünde Ek C'de adı geçen ZIP'in güncel dosya kimliği ve temiz bir klasörde izlenecek adımlar yer alır. Buradaki tarihsel betikler ve çalıştırma kayıtları yapılan deneyin kaynaklarıdır.

Klonlanan depoyla yukarıdaki veri ve ön kontrol zinciri çalışır. Saklanmış **aynı** modelden yeniden çıkarım yapmak için altı özgün `best_checkpoint.pt` dosyası, çalışma kayıtları ve eşleşen girdiler gerekir. SHAP çalıştırıcısı ayrıca örneklem, ortam, pilot eşdeğerliği ve giriş kimliği kayıtlarını doğrular. Güvenlik denetimleri eksik kayıtla devam etmez. Tüm ağırlıklar ve büyük diziler bu açık kod deposunda yer almıyor. 12 Kasım ana deney arşivi ayrı kapsamdadır. Kaydedilmiş P1 özetleri bu dizindeki küçük dosyalardan okunabilir. [Kaynak ve seçilmiş çıktı kimlikleri](archived_source_hashes.json)
