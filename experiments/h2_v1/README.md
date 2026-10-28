# H2: kontrollü görev paylaşımı deneyi, v1

28 Ekim 2026: dokuz eğitim, eğitim-sonrası kimlik kapısı ve keşifsel test değerlendirmesi tamamlandı. H2 bu kurulumda desteklenmedi: iki FullSharing kolu da her iki görevde daha yüksek test kaybı üretti. Mevcut v4 tez dosyası, sonuçlar ve ana dal değiştirilmedi. `protocol_draft.json` hazırlık geçmişidir; yürütülen tasarım `protocol_locked_v1.json` içindedir.

| Kol | Yapı | FF boyutu | Doğrulanmış parametre |
|---|---|---:|---:|
| NS_base | NoSharing | 128 | 54.536 |
| FS_same_width | FullSharingMTL | 128 | 28.456 |
| FS_budget_control | FullSharingMTL | 329 | 54.586 |

Her kol 123/777/2026 tohumlarıyla 100 epoch eğitilir. Bütün kollarda aynı veri, ölçekleyici, eşleştirilmiş batch sırası, AdamW ve kayıp kullanılır. Erken durdurma yoktur; kontrol noktası özgün ham ölçekli ValidationScore ile seçilir. Bütçe kolunun parametre farkı %0,092'dir. FF genişliği değiştiğinden bu, paylaşımın saf nedensel etkisini tamamen ayırmaz.

## Yürütme ve yeniden üretim

- `run_h2_training_v1.py` yalnız eğitim/doğrulama dizilerini açar. Beş test dosyasının yalnız byte kimliği kilitlenir. Yeni ve girdilerden ayrı bir çıktı dizini gerekir; eski sonuçların üzerine yazılmaz.
- `evaluate_h2_v1.py`, dokuz eğitimin tamamını, 100 epoch geçmişini, en iyi/son kontrol noktalarını, kaynakları ve kilitli analiz kodunu doğrulamadan test dizilerini açmaz.
- `requirements_h2_v1.txt` bu çalıştırmanın doğrudan bağımlılık sürümleridir. Torch CPU resmi indeksinden kurulmuştur. Pilot eski sklearn sürümü uyarısını yakalamış, tam eğitimden önce scaler ile aynı sklearn 1.6.1 kurulmuştur.
- Arşivde `inputs/`, `implementation/`, `protocol_locked_v1.json`, `full_run_v2/` ve bağımsız sayısal doğrulama için eski Bartlett referansları birlikte saklanmıştır. Yerel mutlak yollar yürütme kaydının parçasıdır; tekrar çalıştırırken girdi yolu yeni konuma verilir ve dosya kimlikleri aynı tutulur.

Eğitim örneği (girdi kopyası ve yeni çıktı yolu verilerek):

```bash
python run_h2_training_v1.py --input-root /path/to/frozen_inputs --run-root /path/to/h2_v1/new_run --protocol protocol_locked_v1.json --device cpu --threads 1 --workers 3
```

Analiz kodunun güvenlik sınırı için çıktı dizini analiz dosyasının üst proje dizini içinde olmalıdır; özgün girdi projesiyle iç içe olamaz. `implementation/evaluate_h2_v1.py` için `h2_v1/new_run` uygundur. Analiz komutu:

```bash
python implementation/evaluate_h2_v1.py --input-root /path/to/frozen_inputs --run-root /path/to/h2_v1/new_run --reference-npz reference/bartlett_loss_differentials_v1.npz --reference-csv reference/bartlett_dm_hln_results_v1.csv --verify-only
```

Kimlik kontrolü başarılıysa `--verify-only` yerine `--output /path/to/h2_v1/new_run/evaluation_v1` verilir. Çıktı dizini yeni olmalıdır. İlk ön kontrol defteri yalnız ön kontrol içindir; tam eğitim hücreleri içermez.

## Yorum sınırı

Mevcut 584 tarihli test dönemi daha önce incelendiğinden bu deney tamamlayıcı ve keşifseldir. Getiri MAE ve volatilite pinball kayıpları dört varlık ve üç tohum üzerinde eşit ağırlıkla hesaplanır; birincil seri, tohumların kayıp ortalamasıdır. Dört karşılaştırmada Bartlett-HAC L=19, h=1 HLN ve Holm kullanılır; L=0 duyarlılık analizi ayrıdır. Eğitim-dış örneklem kayıp farkları üç tohumla betimseldir. Küçük fark tek başına aşırı uyum mekanizmasını kanıtlamaz. Bütün tohumlar ve kontroller raporlanır; olumlu sonuç zorunlu değildir.


## Sonuç ve teknik kayıt

Üç tohumun ortalama getiri MAE / volatilite pinball kayıpları: NS_base 0.007527158 / 0.005522560; FS_same_width 0.008344425 / 0.008160073; FS_budget_control 0.009146968 / 0.008945911. Dört birincil L=19/Holm karşılaştırması NoSharing lehinedir; L=0 duyarlılık kararları aynıdır. Tohum düzeyindeki 12 test kaybı karşılaştırmasında da NoSharing daha düşüktür. Ortalama genelleme farkları paylaşım kollarında daha büyüktür; bu, tek başına nedensel aşırı uyum testi değildir.

İlk dokuz eğitimlik deneme (`full_run_v1`), bir geçmiş dosyasının kimlik/satır uyuşmazlığı nedeniyle test açılmadan reddedilmiştir. Aynı sabit kod, ayar ve tohumlarla dokuz eğitimin tamamı `full_run_v2` içinde yeniden yapılmış; bütün kapılar geçtikten sonra değerlendirilmiştir. Reddedilen denemenin teknik kayıtları arşivde ayrı tutulur. Çıkarımda yalnız kabul edilen ikinci deneme kullanılır.

Sayısal tablolar `results/` altındadır; dört yeni karşılaştırma eski ana analizdeki 42 testle birleştirilmez. Tam girdiler, kontrol noktaları, geçmişler, dosya kimlikleri ve tekrar çalıştırma yönergeleri ayrı Drive arşivindedir: https://drive.google.com/drive/folders/1BzYnz5qPTI6fST9dwa1g55nehUtZDYTs . Yeni sonuçlar henüz tez metnine eklenmemiştir.

Drive arşivi yeniden indirilip kayıtlı dokuz modelden analiz ayrı bir dizinde tekrar çalıştırılmıştır. Dokuz sayısal çıktı dosyasının (tahmin/kayıp NPZ dahil) byte kimlikleri özgün çıktılarla aynıdır; kontrol kaydı `results/drive_replay_check_v1.json` içindedir. Bu kontrol sıfırdan bağımsız ortamda tam eğitimin tekrarlandığı iddiasını taşımaz. Tekrar çalıştırma metninin terminale kopyalanan Python bloğundaki girintiler de düzeltilip çalıştırılarak kontrol edilmiştir.
