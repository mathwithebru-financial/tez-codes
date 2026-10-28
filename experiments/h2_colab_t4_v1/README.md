# H2 Colab T4 tekrarı

Bu hazırlık, tamamlanmış CPU H2 deneyini aynı sabit bilimsel ayarlarla Colab/Tesla T4 üzerinde tekrar çalıştırmak içindir. Henüz T4 üzerinde eğitim sonucu yoktur. Ana v4 deneyine ve CPU H2 çıktısına yazılmaz.

## Ortam kurulumu

`bootstrap_t4_environment_v1.py`, yalnız `/content/h2_t4_runtime_v1` altında ayrı bir Python ortamı oluşturur. Colab defterinin Python3.13 çekirdeği ve paketleri değiştirilmez. H2 komutlarında `/content/h2_t4_runtime_v1/venv/bin/python` kullanılacaktır. Betik Drive bağlamaz, veri okumaz ve model eğitmez.

Hedefler: Python3.12.13, AstralCPython yapısı20260807, uv0.13.4, torch2.11.0+cu128, numpy2.3.5, pandas2.2.3, scipy1.17.0, scikit-learn1.6.1. Eski ana deneyin doğrulanmış Python/PyTorch sürümlerine yaklaşılır; eski ortamın bütün bileşenlerinin birebir yeniden kurulduğu iddia edilmez. Özellikle ana v4 için kesin bilinmeyen sürümler geriye dönük olarak atanmaz. Diğer sayısal paketler CPU H2 ortamıyla eşleştirilir.

Kurulum yalnız bir TeslaT4 GPU bulunduğunda ilerler. Kurulum sonundaki ayrı yorumlayıcıda bütün hedef sürümler, CUDA12.8 erişimi ve T4 aygıtı denetlenir. Gerçek cuDNN, sürücü ve paket dondurma kayıtları sonuçları görmeden ayrıca sabitlenecektir.

## Yapılan hazırlık kontrolleri

- Kurulum ve gömülü ortam kontrol kodunun söz dizimi doğrulandı.
- Ortam temizleyicisinin ana sürecin çevre değişkenlerini değiştirmediği doğrulandı.
- uv0.13.4 ile CPython3.12.13/build20260807 indirilip ayrı ortam gerçekten oluşturuldu.
- Resmî CUDA12.8 indeksinde torch2.11.0+cu128/cp312 wheel metaverisi doğrulandı.
- T4/GPU kurulumunun bütünü yerel hazırlık ortamında çalıştırılmadı; kullanıcı Colab çıktısıyla doğrulanacaktır.

Sonraki aşama, sürümleri doğrulanan ayrı ortamdan 21 donmuş girdiyi kontrol edip yeni T4 eğitim/analiz kodlarını ve protokolü sabitlemektir. T4 tekrarı önceden incelenmiş test dönemi üzerinde keşifsel bir ortam sağlamlık kontrolüdür. Üç model × üç tohum koşusunun tamamı raporlanacaktır.
