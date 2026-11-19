# Tez şekillerini yeniden çizme

Bu klasörde tezdeki 11 şeklin her biri için ayrı bir çizim betiği bulunur. `sekilleri_ciz.py` hepsini tezdeki sırayla PNG olarak kaydeder. Betikler özgün deneyin kayıtlı verilerini ve sonuçlarını kullanır.

## Çalıştırma

[Tez arşivleri rehberindeki](../docs/TEZ_ARSIVLERI.md) P0/H2 arşivini (`Tez_Yeniden_Uretim_Arsivi_20261112.zip`) indirip açın. Komutları bu deponun kök dizininde, Python 3.12 ortamında çalıştırın. Baskıdaki yazı biçimi için Times New Roman ailesinin normal, kalın, italik ve kalın italik biçimleri sistemde kurulu olmalıdır.

11 şekli birden üretmek için aşağıdaki komutlarda `../tez_arsivi` yerine `main_project` klasörünün veya onu içeren dizinin yolunu yazın.

```shell
python -m pip install -r figures/requirements.txt
python figures/sekilleri_ciz.py --archive "../tez_arsivi" --output "../tez_sekilleri"
```

Betik, `--output` ile verdiğiniz klasörü oluşturur ve her şekli kendi numarasıyla kaydeder. Örneğin yalnız Şekil 5.1’i üretmek için şu komutu kullanın.

```shell
python figures/sekil_5_1.py --archive "../tez_arsivi" --output "../tez_sekilleri"
```

Şekil 4.1 ve 4.2’nin önem tabloları depodaki [results/shap/](../results/shap/) klasöründe de bulunur. Bu iki şekli yalnızca depo dosyalarıyla çizmek için bağımlılıkları kurduktan sonra depo kökünde şu komutları çalıştırın.

```shell
python figures/sekil_4_1.py --archive . --output "../tez_sekilleri"
python figures/sekil_4_2.py --archive . --output "../tez_sekilleri"
```

## Baskı ölçüleri

Şekil 4.1 ve 4.2, özgün baskı görselleriyle aynı biçimde 2.400 × 1.740 piksel ve 400 dpi olarak kaydedilir. Diğer dokuz PNG’nin çözünürlük kaydı 600 dpi’dir. Word'de en-boy oranını kilitleyerek aşağıdaki baskı genişliklerini kullanın.

| Şekil | Açıklama | Genişlik |
|---|---|---:|
| 2.1 | Girdi penceresi, hedefler ve iki Vol20 penceresindeki ortak getiriler | 14,50 cm |
| 2.2 | Özgün Transformer mimarisinde kodlayıcı ve kod çözücü (Vaswani vd., 2017’den uyarlanmıştır) | 13,50 cm |
| 2.3 | Tek dalın kodlayıcı yapısı | 10,50 cm |
| 2.4 | Görevler arasında paylaşım seçenekleri | 14,50 cm |
| 3.1 | Ana deneyin veri hazırlığı, model seçimi ve test adımları | 14,50 cm |
| 3.2 | Fiyat, getiri ve volatilite | 14,00 cm |
| 3.3 | Hedef tarihlerine göre eğitim, doğrulama ve test dönemleri | 14,50 cm |
| 4.1 | Her çıktı için sekiz girdi özelliğinin SHAP önem payları | 14,50 cm |
| 4.2 | Her çıktı için girdi penceresindeki on gözlemin SHAP önem payları | 14,50 cm |
| 4.3 | Kayıp farkları ve Holm kararları | 14,50 cm |
| 5.1 | Gerçekleşen Vol20 ve bir sonraki gözlem için tahminler | 14,50 cm |

## Verilerin kaynağı

Aşağıdaki yollar `main_project` klasörüne göredir.

- Şekil 2.1–2.4’ün şemaları betiklerde tanımlıdır. Şekil 2.1, ardışık iki Vol20 penceresindeki 19 ortak getiriyi gösterir. Şekil 2.2 özgün Transformer mimarisini, 2.3 ve 2.4 ise tezde kullanılan mimarileri gösterir.
- Şekil 3.1, ham veri ve örnek sayılarını `data/processed/meta_v4.json`, `data/processed/split_meta_v4.json` ve `data/sequences/baseline/lb10/sequence_meta.json` dosyalarından alır. Adaylar ve tohumlar için `results/multiseed/multiseed_winner_config_v4.json` kullanılır.
- Şekil 3.2, `data/raw/raw_prices.csv` içindeki ortak takvim fiyatlarını kullanır. BIST100 fiyat ekseni logaritmiktir. Döviz ve altın fiyat eksenleri doğrusaldır.
- Şekil 3.3’te dönem sınırları ve pencereleme öncesi hedef sayıları `data/processed/split_meta_v4.json` dosyasından okunur.
- Şekil 4.1 ve 4.2, `results/shap_analysis/` klasöründeki `10_shap_global_importance_v4.csv` ve `10_shap_temporal_importance_v4.csv` dosyalarını kullanır. Şekil 4.1’de her sütun, Şekil 4.2’de her satır bir çıktının önem paylarını verir.
- Şekil 4.3, `results/statistical_comparison/09_dm_harvey_holm_results_v4.csv` içindeki 42 temel karşılaştırmayı, kayıtlı ortalama kayıp farkları ve Holm sonrası kararlarla gösterir. Pozitif fark nihai modelin kaybının daha düşük olduğunu belirtir.
- Şekil 5.1 için `data/sequences/baseline/lb10/` içindeki `sequence_meta.json`, `anchor_dates_test.npy` ve `target_realization_dates_test.npy` hedef sırasını ve tarihleri belirler. Çizilen değerler `results/final_test/` içindeki `final_test_y_true_raw_v4.npy` ile `pred_final_ensemble_raw_v4.npy` ve `results/baselines/naive/pred_vol_persistence_raw_v4.npy` dosyalarından gelir.
- `veri_5_1_hazirla.py` ham fiyatların SHA-256 özetini, 584 test tarihinin sırasını, gerçek hedef ve süreklilik değerlerini, ayrıca tohumları 123, 777 ve 2026 olan üç modelin ham ölçekteki tahmin ortalamasını kontrol eder. Dört volatilite sütununun sırası BIST100, USD/TRY, EUR/TRY ve Altın'dır. Her nokta hedefin gerçekleşme tarihinde gösterilir.

Vol20, ardışık 20 ortak takvim gözleminin logaritmik getirileri üzerinden hesaplanan örnek standart sapmanın √252 ile çarpımıdır. Düşey eksen yıllıklandırılmış ondalık ölçektedir. Örneğin 0,20 yıllıklandırılmış %20 volatiliteyi ifade eder. Süreklilik tahmini, hedef tarihinden bir önceki ortak takvim gözleminin Vol20 değeridir.
