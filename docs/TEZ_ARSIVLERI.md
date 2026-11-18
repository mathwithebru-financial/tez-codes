# Tez arşivleriyle sonuçları doğrulama

Özgün dönem 6 Ekim 2022–31 Aralık 2024 arasındaki 584 test hedefini kapsar. Ek C’de yer alan iki ZIP farklı deneylere aittir. Özgün düzende (P0) son girdi hedefin bir önceki ortak gözlemidir. Gecikmeli düzende (P1) aynı hedef için girdi penceresi bir ortak gözlem daha erken biter.

| Arşiv | İndirme ve kapsam | İlk komut |
|---|---|---|
| `Tez_Yeniden_Uretim_Arsivi_20261112.zip` | [Ana deney ve özgün dönem H2 arşivi](https://drive.google.com/file/d/1sLdEDADQtwbntvfOVI2IS2bqsxVyprE7/view). Ana istatistiksel karşılaştırmalar, H2, getiri ve volatilite HAC kontrolleri, kayıtlı SHAP tanıları. | Açılan ZIP kökünde `python tools/verify_archive.py --archive-root .` |
| `Tez_Tamamlayici_Analiz_Arsivi_20261117.zip` | **P1, 19 Kasım 2026 dağıtımı.** Üçer tohumla MSE ve MAE eğitiminin altı seçilmiş ağırlığı, girdileri, tahminleri, alt dönem sonuçları, SHAP kayıtları ve taşınabilir çalıştırıcı. | Açılan ZIP kökünde `python dogrula.py` |

P1 dosyasının adındaki 17 Kasım, Ek C’de anılan özgün arşivi tanımlar. 19 Kasım dağıtımı pilot kanıtlarını ve taşınabilir çalıştırıcıyı içerir. Bu ZIP, tezle birlikte paylaşılan danışman inceleme klasöründedir. Erişim için tez yazarından yukarıdaki dosya adıyla istenebilir. Paket düzeyindeki SHA-256 değerini açmadan önce dosyayla karşılaştırın. Windows PowerShell’de `Get-FileHash -Algorithm SHA256 -LiteralPath "dosya.zip"`, Linux’ta `sha256sum dosya.zip`, macOS’ta `shasum -a 256 dosya.zip` kullanılabilir.

| Dağıtım | ZIP SHA-256 |
|---|---|
| P0/H2, 12 Kasım | `042b10a6f7f48d3342d59c52a397bc5ae1110230ad82f7fa59cc8a327f2d2a0b` |
| P1, 19 Kasım | `ab9ede7bb8516853baa3ebaa581a7b9cfa4d5a4bed3ba01858839798ce94d83e` |

## Ana deney ve H2

12 Kasım ZIP’ini boş bir dizine açın. Aşağıdaki komutları `PACKAGE_MANIFEST.json` ile aynı dizinde çalıştırın. İlk komut 699 dosya kimliğini doğrular. Manifest dâhil ZIP’te 700 üye vardır. Yeniden hesaplama için ayrı bir Python ortamı ve henüz kullanılmamış çıktı dizinleri seçin.

```shell
python tools/verify_archive.py --archive-root .
python -m pip install -r sources/main/requirements-bartlett.txt
python sources/main/scripts/reproduce_archived_analysis_v4.py stage09 --project-root main_project --output-dir ../p0_stage09
python tools/replay_h2_frozen.py --archive-root . --output-dir ../h2_original
```

Stage 09, 17 girdiyi kontrol edip günlük kayıp farklarından 42 karşılaştırmayı yeniden hesaplar. Nihai model lehine 5, kıyaslamalar lehine 34 anlamlı sonuç ve 3 anlamsız fark beklenir. H2 komutu, kayıtlı tahmin ve hedeflerden 16 istatistiksel kararı doğrular. Volatilite HAC için tam komut arşivdeki `ONCE_OKU.txt` içinde, getiri HAC ve raporlama adımları `tools/POST_REVIEW_REPLAY.txt` içindedir. 12 Kasım notunda o tarihteki Word dosyasının adı geçer. Güncel tez kopyası `TezmEBRUALTUNBAG.docx` ve `TezmEBRUALTUNBAG.pdf` adlarını taşır.

## Gecikmeli deney

P1 ZIP’ini ayrı bir dizine açın. Windows’ta yol uzunluğu sınırına takılmamak için `C:\tez\p1` gibi kısa bir kök seçin. Pakette manifest dâhil 577 dosya bulunur. Kök dizinde `MANIFEST.json`, `ONCE_OKU.txt` ve `dogrula.py` yer alır. `python dogrula.py` yalnızca Python standart kütüphanesini kullanır. Manifestteki 576 dosyanın kimliğini denetler ve arşivlenmiş günlük tahminlerden Çizelge 4.12’nin tam test kayıplarını üretir.

| 584 hedefte dört varlık ortalaması | MSE kolu | MAE kolu | Naif |
|---|---:|---:|---:|
| Getiri MAE | 0,007448 | 0,007098 | 0,006894 |
| Volatilite pinball (τ = 0,5) | 0,007438 | 0,007301 | 0,004934 |

Tezde alıntılanan [P1 kaynak sürümü](https://github.com/mathwithebru-financial/tez-codes/tree/40da5a1b8887a96117cf4ededbee674d63a7a980/experiments/p1_v2) tarihsel deney kodunu gösterir. [Veri hazırlığı rehberinde](../experiments/p1_v2/README.md) açık ham veriden 10 gözlemlik pencerelerin ve ölçeklerin yeniden oluşturulma sırası yer alır.

### Ağırlıklardan tahmin ve SHAP hesabı

19 Kasım dağıtımındaki ZIP kökünde aşağıdaki komutları çalıştırın. Her komuta paket dışında yeni bir çıktı klasörü verin. Aynı [çalıştırıcı](../experiments/p1_v2/replay_archive.py) bu depoda da bulunur. Kod, tarihsel yollara bağlı girdileri açılan paket içinde bulur ve dosya kimliklerini denetler. Her işlem sonucu kendi çıktı dizinindeki `replay_receipt.json` dosyasına yazılır.

```shell
python -B replay_archive.py check --archive-root . --output-dir ../p1_kontrol
python -B replay_archive.py predict --archive-root . --output-dir ../p1_tahmin --device cuda
python -B replay_archive.py subperiods --archive-root . --output-dir ../p1_alt_donem
python -B replay_archive.py stability --archive-root . --output-dir ../p1_shap_ozet
python -B replay_archive.py shap-one --archive-root . --output-dir ../p1_shap_tek
```

| İşlem | Kontrol edilen sonuç |
|---|---|
| `check` | Altı seçilmiş ağırlığın kimliği, kaynaklar ve tarihsel pilot kararına dayanak oluşturan kayıtlar. V3 pilot sonucunun SHA-256 değeri `df772bf124ef4f6980ca1381d790b1c8c8db81edb8baebc847f8e80bfeff8c1f`. |
| `predict` | 584 hedef için altı modelin ve iki topluluğun tahminleri. CPU için `--device cpu` seçilebilir. Doğrulama yapılan GPU’da sekiz tahmin dizisi kayıtlarla birebir eşleşti. CPU’da dört topluluk ölçütü tezdeki altı ondalık basamakta eşleşti. |
| `subperiods` | Kayıtlı tahminlerden 2022’nin testteki kısmı, 2023 ve 2024 için beş sonuç CSV’si. |
| `stability` | Saklanmış SHAP dizilerinden sekiz karşılaştırma CSV’si. Kaynak açıklamalar 48 tek-model koşulunu ve bunlardan türetilen 16 topluluk koşulunu kapsar. |
| `shap-one` | Bir koşul için girdi gradyanlarından 128 pencerenin açıklamaları. MSE kolu, model tohumu 123, referans arka planı ve ilk çekiliş tohumu kullanılır. |

Tüm kararlılık deneyini yeniden hesaplamak için aynı çağrıda mod olarak `shap-all` ve kullanılmamış bir çıktı dizini seçilir. Bu işlem 48 tek-model açıklamasını hesaplar ve 16 topluluk koşulunu oluşturur. 19 Kasım’daki temiz klasör testinde yeni gradyan hesabı bir koşulda yürütüldü ve o koşulun on dizisi birebir eşleşti. Kayıtlı koşulların tamamının kararlılık özetleri ayrıca hesaplandı. Sekiz seçili örneğin 256/1.024/4.096 çekiliş duyarlılığı ayrı bir deneydir.

Python ve paket sürümlerini P1 ZIP’indeki `ONCE_OKU.txt` ve `thesis_rebuild_20261113/analysis/shap_stability_20261113/environment/` altında bulabilirsiniz. Kaynak ve pilot denetimi için kayıtlı SHAP uygulaması gerekir. Gradyan testinde kullanılan ortam Python 3.13, NumPy 2.3.3, PyTorch 2.11.0+cu128, SHAP 0.52.0 ve NVIDIA RTX 4060 Laptop GPU içerir. Başka bir donanımda oluşan sayısal farklar çıktı kaydında veya açık bir doğrulama hatasıyla görünür.
