# Tez kodlarını hangi sırayla okumalı?

Konuları ilk kez çalışıyorsanız, [uygulamalı ders defterinde](../notebooks/Finansal_Zaman_Serileri_Uygulamali_Dersler.ipynb) tek bir fiyat hareketinden başlayın. Örnek hesapları elle deneyebilir, ardından kodu çalıştırıp tablolar ve grafiklerle karşılaştırabilirsiniz. [Colab’da aç](https://colab.research.google.com/github/mathwithebru-financial/tez-codes/blob/main/notebooks/Finansal_Zaman_Serileri_Uygulamali_Dersler.ipynb)

Bir örnek üzerinden başlayalım. Ana deneyin ilk test girdisi 22 Eylül–5 Ekim 2022 arasındaki **10 ortak takvim gözleminden** oluşur. Son satırdaki dört getiri ve dört tarihsel volatilite, geçmişin modele verilen özetidir. Bu pencerenin hedefi 6 Ekim’de biten tek gözlemlik getiri ile 6 Ekim’deki 20 gözlemlik tarihsel volatilitedir. Tamamlayıcı deney aynı hedefi korur ve son girdi tarihini 4 Ekim’e çeker.

Tablodaki sıra, her aşamanın önceki hangi ürüne ihtiyaç duyduğunu gösterir. Betiklerin bulunması, her tarihsel çalıştırmanın kaydı anlamına gelmez. `123`, `777`, `2026` model başlangıçlarını belirleyen üç rastgelelik tohumudur.

## Ana deney (P0)

| Sıra | Girdi ve işlem | Dosya | Elde edilen ürün |
|---|---|---|---|
| 1 | Yahoo Finance'tan dört serinin günlük fiyatlarını edinme | [Özgün v3 betiği](../archives/v3_data_acquisition/scripts/01_veri_toplama_v3.py) · [Veri kökeni](../archives/v3_data_acquisition/README.md) | [Kullanılan ham fiyat dosyası](../data/raw/raw_prices.csv), 3.912 tarih × 4 varlık |
| 2 | Sütunlar, varlık sırası ve aday modelleri tanımlama | [00 · Kurulum](../scripts/00_setup_v4.py) · [Şema](../config/schema_v4.json) | Ortak ayarlar |
| 3 | Son bilinen fiyatla doldurma, log getiri, yıllıklandırılmış tarihsel volatilite, teknik göstergeler ve hedefler | [01 · Veri hazırlama](../scripts/01_rebuild_from_frozen_raw_v4.py) | Aynı 3.891 tarihte 8 veya 28 özellik ve sekiz hedef |
| 4 | Gerçekleşme tarihine göre kronolojik bölme, eğitimde ölçekleme, geçmiş pencere oluşturma | [02 · Ön işleme](../scripts/02_preprocessing_v4.py) | Temel küme ve L=10 için eğitim/doğrulama/testte 2.714/584/584 pencere |
| 5 | Basit tahminler ve küçük model ile hesap akışını sınama | [03 · Basit referans](../scripts/03_baseline_sanity_v4.py) · [04 · Küçük model](../scripts/04_small_model_test_v4.py) · [05a · Küçük arama](../scripts/05a_mini_grid_v4.py) | Doğrulama referansları ve deneme koşuları |
| 6 | Dört paylaşım mimarisi, beş kayıp stratejisi, dört pencere, üç kapasite ve iki özellik kümesi | [05 · Aday taraması](../scripts/05_grid_search_v4.py) | 480 yapılandırmalık arama alanı |
| 7 | On farklı adayı üç tohumla değerlendirme | [06 · Çoklu tohum eğitimi](../scripts/06_best_model_multiseed_v4.py) | En düşük ortalama doğrulama skoru ile NoSharing, temel sekiz özellik, L=10, küçük model, λ=0,7 |
| 8 | Her tohumun sekiz tahminini gerçek ölçeğe çevirme, üç ham tahminin ortalamasını alma | [07 · Son test](../scripts/07_final_test_evaluation_v4.py) | Hedef tarihi × sekiz çıktı |
| 9 | Sıfır getiri, süreklilik, tek görevli Transformer, LSTM ve XGBoost ile karşılaştırma | [08A · Basit modeller](../scripts/08A_naive_baselines_test_v4.py) · [08B · Öğrenen modeller](../scripts/08B_learned_baselines_test_v4.py) | Getiride MAE, volatilitede τ=0,5 pinball |
| 10 | GARCH/GJR koşullu dağılımını tahmin edilen NextVol20 ölçeğine taşıma | [08C · GARCH](../scripts/08C_garch_baselines_test_v4.py) · [Eksik koşu işlemi](../scripts/08C_R_garch_rescue_run_v4.py) | Tam kapsanan varlık-model çiftleri için kıyaslama |
| 11 | Günlük kayıp farklarından DM–HLN ve aile bazında Holm testi; HAC duyarlılığı | [09 · Ana test](../scripts/09_dm_harvey_holm_v4.py) · [Volatilite HAC](../scripts/09D_bartlett_dm_hln_robustness_v1.py) · [Getiri HAC](../scripts/post_review/replay_return_hac.py) · [Getiri karar sayıları](../results/robustness/return_hac/decision_counts.csv) | Gecikme seçimine göre çıkarımlar; tahminler aynı kalır |
| 12 | Seçilmiş modelin tahminlerini girdiler ve zaman konumları üzerinden açıklama | [Defterde Stage 10](../notebooks/tez_transformer_v4_repro.ipynb) · [SHAP sonuçları](../results/shap/) · [Çıktı düzeyindeki tanılar ve raporlama](../scripts/post_review/replay_reporting_diagnostics.py) | Özellik katkıları ve yerel açıklama hataları |

Seçilen modelde getiri ve volatilite dallarının öğrenilebilir parametreleri ayrıdır. Her dal bütün varlıkların sekiz girdisini görür. Eğitimde standartlaştırılmış getirinin MSE’si ile volatilitenin τ=0,5 pinball kaybı 0,7 ve 0,3 katsayılarıyla birleşir. Doğrulamadaki seçim kuralı, her görevin basit referansına bölünen hatasına eşit ağırlık verir.

## Özgün dönem H2 karşılaştırması

6 Ekim 2022–31 Aralık 2024 arasındaki 584 hedefte iki tam paylaşım düzeni ile bağımsız tek görevli Transformer çifti karşılaştırıldı. Aynı genişlik ve yaklaşık aynı parametre bütçesi ayrı koşullardır. H2 hesabında önce her tohumun kaybı bulunur, sonra varlıklar ve tohumlar üzerinde ortalama alınır. Dört birincil karşılaştırmada bağımsız çift daha düşük kayıp verdi. [Sabit sürümde yöntem, kod ve sonuçlar](https://github.com/mathwithebru-financial/tez-codes/tree/31f94d42ba2d95c82ac838a9df27dba4d0065b76/experiments/h2_original_period_v1)

## Tamamlayıcı deney (P1 V2)

Bir ortak gözlem geciktirilmiş girişlerle seçilmiş NoSharing mimarisi üçer tohumla MSE ve MAE kullanılarak eğitildi. Getiri MAE'si iki kol arasında yaklaşık %4,70 azaldı; bu iyileşme 2023 alt döneminden geldi. Tam testte MAE ansamblının sekiz varlık-görev hatası da eşleştirilmiş basit referansın üzerindeydi. Sağlayıcının geçmişteki ilk yayın saatleri belgelenemedi. [Veri hazırlama, kayıplar, alt dönem ve SHAP kodları](../experiments/p1_v2/) · [Kaydedilmiş ölçütler](../experiments/p1_v2/results/primary_metrics.csv)

## Çalıştırmak için

[Tez arşivleri ve doğrulama komutları](TEZ_ARSIVLERI.md) rehberinden ilgili paketi seçin. Ana deney ile özgün dönem H2 için 12 Kasım paketini, bir gözlem gecikmeli deney için 19 Kasım’da tamamlanan P1 paketini kullanın. Her paketin SHA-256 değeri ve ilk kontrol komutu rehberde yer alıyor.

P0'ın özgün Colab ortamı ve kilitli arşiv girdileri için [yeniden üretim rehberini](REPRODUCIBILITY.md) kullanın. Getiri HAC ve raporlama betikleri, 12 Kasım arşivinin açıldığı klasörü `--archive-root` ve yeni bir çalışma klasörünü `--output-dir` ile alır. P1'in aşağıdaki rehberinde, açık depodaki ham dosyadan model eğitmeden pencere ve doğrulama paydası üretme yolu vardır. [P1 çalıştırma sırası](../experiments/p1_v2/README.md)
