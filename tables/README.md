# Tez çizelgelerinin kaynakları ve yeniden üretimi

Baskı tezindeki 48 çizelgenin her biri aşağıda numarasıyla yer alır. 44 çizelgenin hücreleri arşivlenmiş deney sonuçları, protokol kayıtları veya ham fiyatlardan üretilebilir. Çizelge 2.2, 3.3, 3.8 ve C.1 yazarın kaynak incelemesi ve kod envanteridir. Bu dört satırda derlemenin dayanağı gösterilmiştir.

## Çalıştırma

[Tez arşivleri rehberinden](../docs/TEZ_ARSIVLERI.md) ilgili ZIP’i açın. `--p0` için `main_project/` ve `h2_project/` klasörlerinin bulunduğu P0/H2 arşiv kökünü, `--p1` için `dogrula.py` dosyasının bulunduğu P1 arşiv kökünü verin. Çizelge 4.12 ve E.14–E.17 P1 arşivini kullanır. Diğer 39 aktarım P0/H2 arşivini kullanır.

Komutları deponun kökünde Python 3.12 ile çalıştırın. Örneğin Çizelge 4.2’yi tek başına şu komutla üretebilirsiniz.

```shell
python tables/export_tables.py --table 4.2 --p0 "../p0" --output "../tez_cizelgeleri"
```

İki arşivdeki 44 çizelgeyi birlikte üretmek için aşağıdaki komutları kullanın. NumPy, E.12’de kayıtlı günlük tahmin dizilerini okumak için gereklidir. Diğer 43 aktarım Python’un standart kütüphanesiyle çalışır.

```shell
python -m pip install numpy
python tables/export_tables.py --p0 "../p0" --p1 "../p1" --output "../tez_cizelgeleri"
```

Tek arşivle çalışmak için yalnız ilgili arşiv seçeneğini verin. Çıktı adları `Cizelge_4_2.csv` biçimindedir. CSV’ler Türkçe ondalık virgülleriyle, noktalı virgül sütun ayracıyla ve UTF-8 BOM kodlamasıyla yazılır. Panel başlıkları gerektiğinde ayrı bir sütunda, devam sayfaları aynı dosyada yer alır. Çıktılar çizelge hücrelerini içerir. Sayfa düzeni ve açıklama paragrafları Word belgesindedir.

## Çizelge dizini

**Özgün hesaplama veya kaynak** sütunu deney sonucunun nereden geldiğini, **CSV betiği** sütunu arşivden tezin satırlarının nasıl oluşturulduğunu gösterir. Bağlantısız dosya yolları ilgili arşivin içindedir. H2 kod bağlantıları tezin atıf yaptığı sabit sürüme gider.

### Yöntem ve veri

| Çizelge | Özgün hesaplama veya kaynak | Tezde kullanılan kayıt ve seçim | CSV betiği |
|---|---|---|---|
| 2.1 · Volatilite sürekliliğinin R² ve pinball kaybı | [08A naif kıyaslamalar](../scripts/08A_naive_baselines_test_v4.py) · `main_project/results/baselines/naive/naive_baseline_metrics_long_v4.csv` | Dört `VolPersistence` satırı ve varlıkların aritmetik ortalaması. | [early.py](early.py) |
| 2.2 · Literatür çizgileri ile deneyin sentezi | Tez §2.6.1 ve çizelgede gösterilen yayınlar | Yayınlardaki bulgular ile tezdeki deney tasarımı arasındaki ilişkiyi yazar özetler. Çizelgenin iki baskı parçası aynı numara altındadır. | Yazar derlemesi |
| 3.1 · Varlıklar ve ham veride eksik gözlemler | [Özgün veri edinimi](../archives/v3_data_acquisition/scripts/01_veri_toplama_v3.py), [ham fiyat](../data/raw/raw_prices.csv) ve [sembol şeması](../config/schema_v4.json) | Dört fiyat sütununda doldurma öncesi boş hücreleri sayar. | [early.py](early.py) |
| 3.2 · Kronolojik bölmeler, örnek sayıları ve tarihler | [02 ön işleme](../scripts/02_preprocessing_v4.py) · `main_project/data/processed/split_meta_v4.json` | Eğitim, doğrulama ve test için pencereleme öncesi sayı, girdi tarihi ve hedef tarihi. | [early.py](early.py) |
| 3.3 · Dört görev paylaşımı mimarisinin bileşenleri | [05 model sınıfları](../scripts/05_grid_search_v4.py) içinde `FullSharingMTL`, `PartialSharingMTL`, `HierarchicalMTL` ve `NoSharing` | Model kodunun yazar tarafından sözel karşılaştırması. | Yazar derlemesi |
| 3.4 · Küçük, orta ve büyük Transformer boyutları | [00 şema tanımı](../scripts/00_setup_v4.py) ve [kaydedilen şema](../config/schema_v4.json) · `model_sizes` | Üç boyut için model genişliği, başlık, katman ve ileri besleme genişliği. | [early.py](early.py) |
| 3.5 · Tek görevli modellerde seçilen yapılandırma ve parametre sayıları | [08B öğrenen kıyaslamalar](../scripts/08B_learned_baselines_test_v4.py) · `main_project/results/baselines/learned/` altındaki `learned_baseline_selection_v4.csv`, `learned_baseline_parameter_report_v4.csv` ve `main_project/data/sequences/baseline/lb10/sequence_meta.json` | Üç model ailesinin getiri ve volatilite seçimlerini model, görev ve yapılandırma kimliğiyle parametre raporuna bağlar. | [early.py](early.py) |
| 3.6 · Hesaplama ortamı ve yazılım sürümleri | [Ortam kayıtlarının kaynakları](../docs/REPRODUCIBILITY.md), [8C protokolü](../protocols/08C_garch_protocol_lock_v4.json), [SHAP kilidi](../protocols/10_shap_protocol_lock_v4.json) ve [SHAP yöntem eki](../protocols/10_shap_protocol_addendum_v4a.json) | Bu kayıtları 08B'nin `learned_baseline_summary_v4.json` dosyası ve özgün ana `main_project/requirements.txt` ile birleştirir. scikit-learn 1.6.1 için kullanılan kayıt yöntem ekindeki `runtime_at_addendum_lock.sklearn` alanıdır. | [early.py](early.py) |
| 3.7 · Tamamlayıcı H2 mimarileri ve eğitilebilir parametreler | [H2 özgün dönem kaydı](https://github.com/mathwithebru-financial/tez-codes/tree/31f94d42ba2d95c82ac838a9df27dba4d0065b76/experiments/h2_original_period_v1) · `h2_project/t4/protocol_locked_t4_v1.json` ve `h2_project/stl_original_period/protocol_plan_v1.json` | Üç mimarinin kilitli genişliği ve kapasitesi ile iki bağımsız dalın toplam kapasitesi. | [early.py](early.py) |
| 3.8 · Günlük serilerde örnek saatler ve tarihsel erişilebilirlik | Çizelgedeki Borsa İstanbul, Yahoo Finance, COMEX ve Chicago Board of Trade kayıtları; [veri ediniminin kapsamı](../archives/v3_data_acquisition/README.md) | Seans, zaman etiketi ve uzlaşı bilgilerinin kaynaklara dayalı yazar derlemesi. Ham fiyat dosyasından tarihsel yayımlanma saati hesaplanamaz. | Yazar derlemesi |

### Bulgular ve tartışma

| Çizelge | Özgün hesaplama veya kaynak | Tezde kullanılan kayıt ve seçim | CSV betiği |
|---|---|---|---|
| 4.1 · Tek görevli modeller | [Ana DM–HLN ve Holm hesabı][ana09] | `09_dm_harvey_holm_results_v4.csv` içindeki 24 öğrenen model karşılaştırması, getiri ve volatilite panelleri. | [results.py](results.py) |
| 4.2 · Naif modeller | [Ana DM–HLN ve Holm hesabı][ana09] | Aynı CSV’nin 12 naif karşılaştırması, getiri ve volatilite panelleri. | [results.py](results.py) |
| 4.3 · GARCH ailesi | [Ana DM–HLN ve Holm hesabı][ana09] | Aynı CSV’nin GARCH ve GJR-GARCH için altı volatilite karşılaştırması. | [results.py](results.py) |
| 4.4 · Çıktı bazında SHAP payları | [Özgün araştırma defteri, Aşama 10][ana10] | `10_shap_global_importance_v4.csv` ve `10_shap_temporal_importance_v4.csv`. Sekiz çıktı için baskın özellik, getiri ve Vol20 girdilerinin toplam payları ve son adımın payı. | [results.py](results.py) |
| 4.5 · Üç tohumda doğrulama | [Çok tohumlu model seçimi][tohum06] | `multiseed_winner_config_v4.json` ile seçilen yapılandırmanın `multiseed_runs_v4.csv` satırları. Son iki satır üç tohumun ortalamasını ve örneklem standart sapmasını verir. | [results.py](results.py) |
| 4.6 · Görev bazında karar dağılımı | [Ana DM–HLN ve Holm hesabı][ana09] | 42 satırın göreve ve Holm sonrası karara göre sayımı. | [results.py](results.py) |
| 4.7 · Ana analiz ve Bartlett-HAC | [Ana hesap][ana09] ve [Bartlett-HAC hesabı][hac09] | `bartlett_dm_hln_results_v1.csv` içinde aynı 22 volatilite karşılaştırmasının özgün ve sağlamlık kararları. | [results.py](results.py) |
| 4.8 · Kararı değişen karşılaştırmalar | [Bartlett-HAC hesabı][hac09] | `bartlett_dm_hln_changed_decisions_v1.csv` içinde kararı değişen altı satır. | [results.py](results.py) |
| 4.9 · H2 test kayıpları | [Özgün dönem H2 değerlendirmesi, sabit sürüm][h2kod] | `primary_losses_seed_summary_v1.csv` içindeki `four_asset_mean` satırları. Üç tohumun ortalaması ve örneklem standart sapması. | [results.py](results.py) |
| 4.10 · H2 eğitim, doğrulama ve test kayıpları | [Özgün dönem H2 değerlendirmesi][h2kod] | `generalization_gaps_seed_summary_v1.csv` içindeki sekiz mimari–görev satırı. Arşivdeki `tools/replay_h2_frozen.py` yeniden üretimde eğitim ve doğrulama için kayıtlı skaler ölçümleri kullanır. | [results.py](results.py) |
| 4.11 · H2 birincil ve ikincil karşılaştırmaları | [Özgün dönem H2 değerlendirmesi][h2kod] | `dm_holm_comparisons_v1.csv` içinden `fixed_original_period_supplement` analizi. İki ailede sekiz karşılaştırma. | [results.py](results.py) |
| 4.12 · Gecikmeli protokol ve alt dönemler | [P1 alt dönem hesabı][p1alt], [taşınabilir çalıştırıcı][p1yeniden] (`subperiods`) | P1 `subperiods_20261113/metrics_equal_asset_macro.csv` kaydında `primary` gözlemler ve topluluk tahminleri. Fark (%) = 100 × (MAE kolu kaybı − MSE kolu kaybı) / MSE kolu kaybı. | [results.py](results.py) |
| 5.1 · Seçilmiş kayıp farklarının büyüklüğü | [Ana DM–HLN hesabı][ana09] ve [nihai test değerlendirmesi][test07] | Ana sonuç CSV’sinden tezde gösterilen 13 karşılaştırma ve `final_test_summary_v4.csv` içinden üç tohumlu topluluğun görev ortalaması. Son iki sütunda sırasıyla mutlak farkın görev kaybına oranı ve kıyaslama kaybına göre işaretli azalma kullanılır. | [results.py](results.py) |

### Ekler

| Çizelge | Özgün hesaplama veya kaynak | Tezde kullanılan kayıt ve seçim | CSV betiği |
|---|---|---|---|
| A.1 · Ham fiyat ve altı protokolün SHA-256 özetleri | [Ham fiyat](../data/raw/raw_prices.csv) ve [protokol dosyaları](../protocols/) · P0 arşivinde `main_project/data/raw/` ve `main_project/config/` | Yedi özgün dosyanın baytlarından SHA-256 değerlerini hesaplar. | [early.py](early.py) |
| B.1 · 20 getiri karşılaştırmasının tam sonuçları | [09 DM–HLN/Holm](../scripts/09_dm_harvey_holm_v4.py) · [42 karşılaştırmanın kayıtlı CSV’si](../results/statistical_comparison/09_dm_harvey_holm_results_v4.csv) | Getiri satırlarında kayıp farkı, test istatistiği, düzeltilmiş p ve kararı tez sırasıyla verir. | [early.py](early.py) |
| B.2 · 22 volatilite karşılaştırmasının tam sonuçları | [09 DM–HLN/Holm](../scripts/09_dm_harvey_holm_v4.py) · [42 karşılaştırmanın kayıtlı CSV’si](../results/statistical_comparison/09_dm_harvey_holm_results_v4.csv) | Volatilite satırlarında kayıp farkı, test istatistiği, düzeltilmiş p ve kararı tez sırasıyla verir. | [early.py](early.py) |
| B.3 · 22 volatilite karşılaştırmasında Bartlett-HAC kontrolü | [09D sağlamlık betiği](../scripts/09D_bartlett_dm_hln_robustness_v1.py), [Bartlett sonuçları](../results/robustness/bartlett_dm_hln_v1/bartlett_dm_hln_results_v1.csv) ve [09 sonuçları](../results/statistical_comparison/09_dm_harvey_holm_results_v4.csv) | Karşılaştırma kimliğiyle iki sonucu eşler. Uzun dönem varyansını sıfır gecikmeli varyansa bölerek ω/γ₀ sütununu hesaplar. | [early.py](early.py) |
| C.1 · İşlem hattındaki ana betikler | [Ana işlem hattının kaynakları](../scripts/) | 19 dosyanın adı ve işlevi yazarın kod envanteridir. Sayısal bir analiz çıktısı yoktur. | Yazar derlemesi |
| D.1 · Seçilen kontrol noktaları | [Özgün dönem H2 değerlendiricisi][ek-h2] | P0/H2’de `learning_curves_v1.csv` seçilen dönemi, `generalization_gaps_descriptive_v1.csv` eğitim/test kayıplarını ve farklarını verir. Dört mimari × üç tohum × iki görev. | [appendix.py](appendix.py) |
| D.2 · Sıfır gecikmeli H2 karşılaştırmaları | [Özgün dönem H2 değerlendiricisi][ek-h2] | `dm_holm_comparisons_v1.csv` içinde `analysis=lag0_sensitivity` ile işaretli sekiz satır; iki kontrol ailesinin kendi Holm p değerleri. | [appendix.py](appendix.py) |
| D.3 · H2 çalışma ortamı | [Özgün dönem H2 değerlendiricisi][ek-h2] | `analysis_report_v1.json` içindeki çalışma ortamı, çıkarım ve kayıp hassasiyeti, eğitim/doğrulama toleransları. Buradaki paket sürümleri H2 deneyine aittir. | [appendix.py](appendix.py) |
| E.1 · Dört varlıkta görev ortalaması | [Raporlama hesabı][ek-rapor] | `absolute_task_metrics.csv` içindeki makro MAE ve pinball. Getiri sıfır ve volatilite sürekliliği referansları aynı satırda gösterilir. | [appendix.py](appendix.py) |
| E.2 · Getiri hata ve uyum ölçütleri | [Raporlama hesabı][ek-rapor] | `absolute_asset_metrics.csv` içinden dört varlık × altı model için getiri MAE, RMSE ve R². | [appendix.py](appendix.py) |
| E.3 · Volatilite hata ve uyum ölçütleri | [Raporlama hesabı][ek-rapor] | Aynı dosyadan dört varlık × beş model için volatilite pinball, RMSE ve R². | [appendix.py](appendix.py) |
| E.4 · İki göreli kayıp azalması | [Raporlama hesabı][ek-rapor] | `reference_relative_reductions.csv` içindeki USD/TRY–XGBoost ve EUR/TRY–tek görevli Transformer karşılaştırmaları. | [appendix.py](appendix.py) |
| E.5 · SHAP toplamsallık tanıları | [Raporlama hesabı][ek-rapor] | `shap_output_diagnostics.csv` içindeki sekiz çıktı için eğitim standart sapması ve artık büyüklükleri. | [appendix.py](appendix.py) |
| E.6 · Yerel SHAP, BIST100 getirisi | [Raporlama hesabı][ek-rapor] | İlk test örneğinin `BIST100_NextRet` sütunu; `shap_local_top5.csv` içinden beş işaretli katkı, `shap_local_summary.csv` içinden diğer 75 katkının toplamı ve artık. | [appendix.py](appendix.py) |
| E.7 · Yerel SHAP, BIST100 volatilitesi | [Raporlama hesabı][ek-rapor] | E.6 ile aynı kaynaklarda ilk örneğin `BIST100_NextVol` sütunu. | [appendix.py](appendix.py) |
| E.8 · Dört test tarihinde yerel ayrıştırma | [Raporlama hesabı][ek-rapor] | `shap_local_summary.csv` içinden 0, 194, 389 ve 583 numaralı örneklerde BIST100’ün iki çıktısı. | [appendix.py](appendix.py) |
| E.9 · Getiri kararı sayıları | [Getiri Bartlett-HAC hesabı][ek-hac] | `decision_counts.csv` içinden L = 0, 3, 6, 10 ve 19 için karar sayıları ve sıfır gecikmeye göre değişimler. | [appendix.py](appendix.py) |
| E.10 · Getiri Holm p değerleri | [Getiri Bartlett-HAC hesabı][ek-hac] | `pvalue_matrix.csv` içinden 20 karşılaştırma için ortalama kayıp farkı ve beş bant genişliğinde düzeltilmiş p değerleri. | [appendix.py](appendix.py) |
| E.11 · Getiri bağımlılık tanıları | [Getiri Bartlett-HAC hesabı][ek-hac] | `dependence_diagnostics.csv` içinden 1. gecikmede ACF, ilk 20 gecikmede en büyük mutlak ACF ve 60 Ljung–Box testi üzerinden Holm ile düzeltilmiş p değerleri. | [appendix.py](appendix.py) |
| E.12 · Gözlenmiş ham fiyatlarla desteklenen hedefler | [Sabit tahmin ve maske hesabı](appendix.py) | P0 ham fiyatlarında hedef günü dâhil getiri için iki, Vol20 için 21 ardışık fiyatın gözlendiği tarihler seçilir. Kayıtlı test tarihleri, hedefler, nihai topluluk ve naif tahminlerden filtreli ve tüm test kayıp oranları hesaplanır. | [appendix.py](appendix.py) |
| E.13 · GARCH ailesinin mutlak kayıpları | [Aşama 09 kayıp karşılaştırmaları](../scripts/09_dm_harvey_holm_v4.py) | `09_dm_harvey_holm_results_v4.csv` içinden 584 hedefin tamamını kapsayan altı GARCH ve GJR-GARCH karşılaştırmasının iki ortalama kaybı. | [appendix.py](appendix.py) |
| E.14 · Eksik fiyat ve doldurma blokları | P1 arşivinde `advisor_followup_20261112/fill_audit/run_fill_audit.py` | Aynı dizindeki `asset_summary.csv` varlık toplamlarını verir. `fill_runs.csv` kayıtları varlık ve 1–4 ortak gözlemlik blok uzunluğuna göre sayılır. | [appendix.py](appendix.py) |
| E.15 · P1 test hedeflerinin ham fiyat desteği | [P1 tarih maskeleri](../experiments/p1_v2/scoring/build_scoring_masks.py) | `scoring_mask_counts.csv` içinden test dönemi, iki görev ve dört varlığın ana kapsam, ham hedef ve tüm model girdileriyle hedefi kapsayan sıkı maske sayıları. | [appendix.py](appendix.py) |
| E.16 · P1 test kayıpları | [Sabit P1 kontrol noktalarının karşılaştırması](../experiments/p1_v2/training/comparison/evaluate_fixed_runs.py) | `comparison/metrics_by_asset.csv` içinde `test/primary` satırları, MSE ve MAE toplulukları ve eşleştirilmiş naif tahmin. Getiri için MAE, volatilite için pinball. | [appendix.py](appendix.py) |
| E.17 · P1 SHAP sıralama kararlılığı | [Kayıtlı P1 SHAP kararlılık hesabı](../experiments/p1_v2/analysis/shap_stability_20261113/analyze_saved_shap_stability.py) | `stability/task_factor_summaries.csv` içinde `individual_seed` modellerinin `spearman_midrank` medyanları. Üç sütun arka plan, model tohumu ve Monte Carlo tohumunu gösterir. | [appendix.py](appendix.py) |

[ana09]: ../scripts/09_dm_harvey_holm_v4.py
[ana10]: ../notebooks/tez_transformer_v4_repro.ipynb
[tohum06]: ../scripts/06_best_model_multiseed_v4.py
[hac09]: ../scripts/09D_bartlett_dm_hln_robustness_v1.py
[test07]: ../scripts/07_final_test_evaluation_v4.py
[h2kod]: https://github.com/mathwithebru-financial/tez-codes/blob/31f94d42ba2d95c82ac838a9df27dba4d0065b76/experiments/h2_original_period_v1/evaluate_original_period_v1.py
[p1alt]: ../experiments/p1_v2/analysis/subperiods_20261113/analyze_subperiods.py
[p1yeniden]: ../experiments/p1_v2/replay_archive.py
[ek-h2]: https://github.com/mathwithebru-financial/tez-codes/blob/31f94d42ba2d95c82ac838a9df27dba4d0065b76/experiments/h2_original_period_v1/evaluate_original_period_v1.py
[ek-rapor]: ../scripts/post_review/replay_reporting_diagnostics.py
[ek-hac]: ../scripts/post_review/replay_return_hac.py
