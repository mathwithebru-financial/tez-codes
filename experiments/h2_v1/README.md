# H2: kontrollü görev paylaşımı deneyi, v1

28 Ekim 2026: ön kontrol ve bir epoch teknik pilot tamamlandı; dokuz tam eğitim ayrı dizinde başlatıldı. Bu aşamada test sonucu yoktur. Mevcut v4 tez dosyası, sonuçlar ve ana dal değiştirilmez. `protocol_draft.json` ve ön kontrol defteri hazırlık geçmişi olarak korunur; yürütülen tasarım `protocol_locked_v1.json` içindedir.

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
- Arşivde `inputs/`, `implementation/`, `protocol_locked_v1.json`, `full_run_v1/` ve bağımsız sayısal doğrulama için eski Bartlett referansları birlikte saklanacaktır. Yerel mutlak yollar yürütme kaydının parçasıdır; tekrar çalıştırırken girdi yolu yeni konuma verilir ve dosya kimlikleri aynı tutulur.

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
