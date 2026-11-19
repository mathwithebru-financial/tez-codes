# Tez şekillerini yeniden çizme

Bu paket, tezin baskı sürümündeki Şekil 2.3, 2.4, 3.2, 4.3 ve 5.1'i PNG biçiminde üretir. Şekil 5.1 kayıtlı test tahminlerini kullanır. Çizimler model eğitiminden bağımsız çalışır. Proje verileri tez arşivinden okunur.

## Çalıştırma

[Tez arşivleri rehberindeki](../docs/TEZ_ARSIVLERI.md) P0/H2 arşivini (`Tez_Yeniden_Uretim_Arsivi_20261112.zip`) indirip açın. Komutları bu deponun kök dizininde, Python 3.12 ortamında çalıştırın. Baskıdaki yazı biçimi için Times New Roman ailesinin normal, kalın, italik ve kalın italik biçimleri sistemde kurulu olmalıdır.

Beş şekli birden üretmek için aşağıdaki komutlarda `../tez_arsivi` yerine `main_project` klasörünün veya onu içeren dizinin yolunu yazın.

```shell
python -m pip install -r figures/requirements.txt
python figures/sekilleri_ciz.py --archive "../tez_arsivi" --output "../tez_sekilleri"
```

Betik, `--output` ile verdiğiniz klasörü oluşturur ve her şekli kendi numarasıyla kaydeder. Örneğin yalnız Şekil 5.1’i üretmek için şu komutu kullanın.

```shell
python figures/sekil_5_1.py --archive "../tez_arsivi" --output "../tez_sekilleri"
```

## Baskı ölçüleri

Çıktılar 600 dpi ve temel metin boyutu 10 puntodur. Word'de en-boy oranını kilitleyerek aşağıdaki genişlikleri kullanın. Matematiksel alt ve üst simgeler oransal olarak daha küçük çizilir.

| Şekil | Açıklama | Genişlik |
|---|---|---:|
| 2.3 | Tek dalın kodlayıcı yapısı | 10,49 cm |
| 2.4 | Görevler arasında paylaşım seçenekleri | 14,48 cm |
| 3.2 | Fiyat, getiri ve volatilite | 14,00 cm |
| 4.3 | Kayıp farkları ve Holm kararları | 14,50 cm |
| 5.1 | Gerçekleşen Vol20 ve bir sonraki gözlem için tahminler | 14,50 cm |

## Verilerin kaynağı

Aşağıdaki yollar `main_project` klasörüne göredir.

- Şekil 2.3 ve 2.4 mimariyi doğrudan betiklerde tanımlar. 2.4'ün (d) panelindeki başlık baskı nüshasındaki “NoSharing (toplam 54.536 parametre)” ifadesini taşır.
- Şekil 3.2, `data/raw/raw_prices.csv` içindeki ortak takvim fiyatlarını kullanır. BIST100 fiyat ekseni logaritmiktir. Döviz ve altın fiyat eksenleri doğrusaldır.
- Şekil 4.3, `results/statistical_comparison/09_dm_harvey_holm_results_v4.csv` içindeki 42 temel karşılaştırmayı, kayıtlı ortalama kayıp farkları ve Holm sonrası kararlarla gösterir. Pozitif fark nihai modelin kaybının daha düşük olduğunu belirtir.
- Şekil 5.1 için `data/sequences/baseline/lb10/` içindeki `sequence_meta.json`, `anchor_dates_test.npy` ve `target_realization_dates_test.npy` hedef sırasını ve tarihleri belirler. Çizilen değerler `results/final_test/` içindeki `final_test_y_true_raw_v4.npy` ile `pred_final_ensemble_raw_v4.npy` ve `results/baselines/naive/pred_vol_persistence_raw_v4.npy` dosyalarından gelir.
- `veri_5_1_hazirla.py` ham fiyatların SHA-256 özetini, 584 test tarihinin sırasını, gerçek hedef ve süreklilik değerlerini, ayrıca tohumları 123, 777 ve 2026 olan üç modelin ham ölçekteki tahmin ortalamasını kontrol eder. Dört volatilite sütununun sırası BIST100, USD/TRY, EUR/TRY ve Altın'dır. Her nokta hedefin gerçekleşme tarihinde gösterilir.

Vol20, ardışık 20 ortak takvim gözleminin logaritmik getirileri üzerinden hesaplanan örnek standart sapmanın √252 ile çarpımıdır. Düşey eksen yıllıklandırılmış ondalık ölçektedir. Örneğin 0,20 yıllıklandırılmış %20 volatiliteyi ifade eder. Süreklilik tahmini, hedef tarihinden bir önceki ortak takvim gözleminin Vol20 değeridir.
