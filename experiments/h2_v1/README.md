# H2 kontrollü paylaşım ek deneyi

[Salt okunur ön kontrolü Colab’da aç](https://colab.research.google.com/github/mathwithebru-financial/tez-codes/blob/h2-controlled-sharing-v1-20261028/experiments/h2_v1/H2_01_On_Kontrol_v1.ipynb)

Durum: **hazırlık ve salt okunur ön kontrol; eğitim uygulanmadı.** Bu çalışma ana v4 sonuçlarından ayrı bir ek deneydir. Mevcut 584 test gözlemi daha önce incelendiğinden elde edilecek sonuçlar tamamlayıcı/keşifsel olarak raporlanacaktır.

## Eski sonuçların korunması

- Girdiler: `/content/drive/MyDrive/tez_transformer_v4_repro`.
- Gelecekteki çıktılar: ayrı kardeş klasör `/content/drive/MyDrive/tez_transformer_h2_v1`.
- Ön kontrol hiçbir dosya yazmaz ve test dizilerini yüklemez.
- Eski pipeline modülü doğrudan içe aktarılmaz: üst düzey kodu eski çıktı yollarına yazabilir. Yalnız SHA-256 kimliği doğrulanan dört model tanımı AST üzerinden alınır.
- Ana tez, ana Git dalı, eski sonuçlar ve kontrol noktaları bu hazırlıkla değiştirilmez.

## Planlanan karşılaştırma

| Düzen | Omurga | FF boyutu | Beklenen parametre |
|---|---|---:|---:|
| NoSharing kontrol | d=32, 4 başlık, 2 katman | 128 | 54.536 |
| FullSharing aynı boyut | d=32, 4 başlık, 2 katman | 128 | 28.456 |
| FullSharing bütçe kontrolü | d=32, 4 başlık, 2 katman | 329 | 54.586 |

Sayılar kaynak yapısından hesaplanan beklentilerdir; ön kontrol gerçek model nesnelerinde doğrulamadan eğitim açılmaz. Bütçe kolundaki fark yaklaşık %0,092'dir. FF boyutu yalnız parametre sayısına göre seçilir; performansa bakılarak ayarlanmaz. Bu iki kontrol, paylaşımın saf nedensel etkisini tamamen ayırdığı iddiasını taşımaz.

Her kol için 123/777/2026 tohumları planlanmıştır: toplam **9 yapı–tohum eğitim koşusu**. NoSharing koşusu tek bir eğitim çağrısında iki bağımsız parametre dalını içerir; bu sayı dokuz tek görevli model anlamına gelmez. Eğitim uygulaması henüz bu hazırlık paketinde yoktur.

## İlk çalıştırılacak dosya

`H2_01_On_Kontrol_v1.ipynb` Colab defterini açın. Drive bağlandıktan sonra salt okunur kontrol; kaynak ve ham veri kimliklerini, eğitim/doğrulama dizilerini ve tarih ayrımını, üç modelin parametre sayılarını ve yalnız ileri geçişini denetler. GPU gerektirmez. Eğitim başlatmaz.

Beklenen son durum `PREFLIGHT_PASS_NOT_A_TRAINING_RESULT` olur. Bir hata çıkarsa özgün dosyalara dokunmadan durur. Çıktı, ileride H2'ye özel girdi kimliklerinin sabitlenmesi içindir; henüz eğitim sonucu veya hipotez desteği değildir.

## H2 nasıl yorumlanacak

Görev bazlı ham ölçekte dış örneklem kaybı, aynı ölçütle hesaplanan eğitim kaybı ve aralarındaki fark birlikte raporlanır. Küçük fark tek başına yeterli değildir; kötü eğitim uyumu veya dağılım kayması da farkı etkileyebilir. Bütün tohumlar ve iki paylaşım kontrolü gösterilir. Olumsuz veya karma sonuç da açıkça raporlanır.

Ayrıntılar: `protocol_draft.json`. Ön kontrol raporu incelendikten sonra ayrı eğitim kodu hazırlanıp doğrulanacaktır. Mevcut sonuçların üzerine çalışan bir eğitim hücresi bu pakete eklenmemiştir.
