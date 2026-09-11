# Bartlett-HAC DM-HLN Ek Sağlamlık Analizi Protokolü (v1.0)

## 1. Statü ve amaç

Bu protokol, tezde raporlanan Stage 09 DM-HLN/Holm sonuçları bilindikten sonra, ancak aşağıda tanımlanan Bartlett-HAC sağlamlık sonuçları hesaplanmadan ve yüklenen ZIP arşivinin içeriği açılmadan önce dondurulmuştur. Bu nedenle çalışma bir ön-kayıt değildir; sonuçtan bağımsız kuralları önceden sabitlenmiş, sonradan yürütülen ek bir sağlamlık analizidir.

Amaç, ardışık 20 günlük volatilite hedeflerinin 19 getiriyi ortak kullanmasından kaynaklanabilecek seri bağımlılığa karşı, volatilite ailesindeki istatistiksel kararların duyarlılığını değerlendirmektir. Stage 09 ana sonuçları değiştirilmez veya silinmez.

## 2. Analiz kapsamı

- Yalnız volatilite görevi yeniden değerlendirilecektir.
- Hipotez ailesi, tezde önceden tanımlanan 22 volatilite karşılaştırmasının tamamından oluşacaktır:
  - 12 öğrenilmiş tek görevli model karşılaştırması: dört varlık × (tek görevli Transformer, tek görevli LSTM, XGBoost),
  - 4 volatilite sürekliliği karşılaştırması,
  - 6 GARCH ailesi karşılaştırması: BIST100, EUR/TRY ve altın × (GARCH(1,1), GJR-GARCH(1,1)).
- Eksik 584 gözlemlik kayıp dizisi nedeniyle Stage 09 dışında bırakılan USD/TRY-GARCH ve USD/TRY-GJR-GARCH çiftleri bu analizde de kapsam dışıdır.
- Her dâhil edilen karşılaştırmada aynı 584 hedef tarihi ve yalnızca eksiksiz, sonlu, ham ölçekli gerçekleşen/tahmin çiftleri kullanılacaktır.
- Getiri ailesindeki 20 karşılaştırma yeniden hesaplanmayacaktır; getiri hedefleri 20 günlük örtüşen hedef yapısına sahip değildir.

## 3. Tahminler ve kayıp dizileri

- Hiçbir model yeniden eğitilmeyecek; mevcut nihai tahmin dosyaları değiştirilmeyecektir.
- Tahminlere ölçekleme, kırpma, yuvarlama, yeniden örnekleme veya sonuca bağlı filtreleme uygulanmayacaktır.
- Volatilite için birincil kayıp, tezdeki tanımla aynı olan \(\tau=0{,}5\) pinball kaybıdır:

  \[
  L_{\tau}(y_t,\hat y_t)=\max\{\tau(y_t-\hat y_t),(\tau-1)(y_t-\hat y_t)\},\qquad \tau=0{,}5.
  \]

- Her karşılaştırmada kayıp farkı:

  \[
  d_t=L_{\text{kıyaslama},t}-L_{\text{nihai},t}
  \]

  olarak tanımlanacaktır. Dolayısıyla \(\bar d>0\) nihai modelin, \(\bar d<0\) kıyaslama modelinin daha düşük ortalama kayıp ürettiğini gösterir.

## 4. Bartlett-HAC uzun dönem varyansı

- Gerçek tahmin ufku bir gündür ve \(h=1\) olarak korunacaktır.
- Bartlett gecikme sınırı tahmin ufkundan bağımsız olarak \(L=19\) seçilecektir.
- \(L=19\), 20 günlük hareketli volatilite hedeflerindeki örtüşmenin doğrudan işaret ettiği en yüksek gecikmedir. Bu seçim, örtüşme kaynaklı seri bağımlılığa yönelik mekanik gerekçeli bir sağlamlık seçimidir; gerçek seri bağımlılığın 19. gecikmede sona erdiği veya 1-19 arasındaki her gecikmede sıfırdan farklı olduğu varsayılmaz.
- Örnek otokovaryanslar, \(T\) paydasıyla hesaplanacaktır:

  \[
  \hat\gamma_k=\frac{1}{T}\sum_{t=k+1}^{T}(d_t-\bar d)(d_{t-k}-\bar d),\qquad k=0,\ldots,19.
  \]

- Bartlett ağırlıkları:

  \[
  w_k=1-\frac{k}{L+1}=1-\frac{k}{20}
  \]

  olacaktır.
- Uzun dönem varyansı ve düzeltilmemiş DM istatistiği:

  \[
  \widehat\Omega_B=\hat\gamma_0+2\sum_{k=1}^{19}w_k\hat\gamma_k,
  \qquad
  DM_B=\frac{\bar d}{\sqrt{\widehat\Omega_B/T}}
  \]

  olarak hesaplanacaktır.

## 5. HLN küçük örneklem düzeltmesi ve p-değeri

- HLN düzeltmesindeki \(h\), Bartlett gecikme sınırı değil, gerçek tahmin ufkudur; bu nedenle \(h=1\) kullanılacaktır.
- Düzeltme çarpanı:

  \[
  c_{T,h}=\sqrt{\frac{T+1-2h+\{h(h-1)/T\}}{T}}
  \]

  ve düzeltilmiş istatistik:

  \[
  DM_{B,HLN}=c_{T,1}DM_B
  \]

  olacaktır. \(T=584\) için \(c_{584,1}=\sqrt{583/584}\approx0{,}9991434688\)'dir.
- İki yönlü ham p-değeri, \(T-1=583\) serbestlik dereceli Student-t dağılımından hesaplanacaktır:

  \[
  p=2\Pr\{t_{T-1}\geq |DM_{B,HLN}|\}.
  \]
- Bartlett gecikme sınırının \(h\)'den bağımsız belirlenmesi, standart `forecast::dm.test` uygulamasının doğrudan kullanılamadığı anlamına gelir. HLN çarpanı, ana analizle karşılaştırılabilirlik amacıyla gerçek \(h=1\) üzerinden korunacaktır. Sonuçlar, standart Stage 09 analizinin yerine geçen yeni bir ana test değil, ek sağlamlık analizi olarak yorumlanacaktır.

## 6. Sayısal istisnalar

- Kayıp farkı dizisinin bütün elemanları tam olarak sıfırsa \(DM_{B,HLN}=0\) ve \(p=1\) atanacaktır.
- \(\widehat\Omega_B\) yalnız kayan nokta yuvarlamasına bağlanabilecek ölçüde negatifse, sıfır toleransı \(10^{-14}\times\max(1,\hat\gamma_0)\) uygulanarak değer sıfıra eşitlenecektir.
- Bu toleransın altında olmayan negatif bir \(\widehat\Omega_B\) ya da sıfır olmayan \(\bar d\) ile sıfır varyans oluşursa karşılaştırma geçersiz olarak işaretlenecek; yapay p-değeri üretilmeyecektir.
- Eksik veya sonlu olmayan değer bulunduğunda satır bazında sessiz silme yapılmayacak; karşılaştırma veri bütünlüğü hatası olarak durdurulacaktır.

## 7. Çoklu test düzeltmesi ve karar kuralı

- İki yönlü 22 ham p-değeri tek volatilite ailesi içinde Holm yöntemiyle düzeltilecektir. Varlık veya model ailesine göre daha küçük alt aile oluşturulmayacaktır.
- Aile düzeyi anlamlılık seviyesi \(\alpha=0{,}05\)'tir.
- Karar kuralları:
  - Holm-düzeltilmiş \(p<0{,}05\) ve \(\bar d>0\): “Nihai model anlamlı biçimde üstün”,
  - Holm-düzeltilmiş \(p<0{,}05\) ve \(\bar d<0\): “Kıyaslama modeli anlamlı biçimde üstün”,
  - Holm-düzeltilmiş \(p\geq0{,}05\): “Anlamlı fark bulunmadı”.
- Eşik karşılaştırmaları ve kararlar yuvarlanmamış p-değerleriyle yapılacaktır. Yuvarlama yalnız sunum aşamasında uygulanacaktır.

## 8. Uygulama ve bağımsız doğrulama

- Yetkili hesap, yukarıdaki denklemleri doğrudan uygulayan özel bir fonksiyonla üretilecektir.
- Bağımsız çapraz kontrolde, kayıp farkının yalnız sabit terim üzerine OLS regresyonundan elde edilen katsayı varyansı `statsmodels` Bartlett-HAC yordamıyla `nlags=19` ve küçük örneklem kovaryans düzeltmesi kapalı olacak şekilde hesaplanacaktır.
- Özel hesap ile bağımsız hesapta \(\widehat\Omega_B/T\), \(DM_B\) ve iki yönlü ham p-değerleri karşılaştırılacaktır. Göreli veya mutlak farkın \(10^{-10}\)'u aşması doğrulama hatası sayılacak ve sonuç tablosu yayımlanmayacaktır.
- Kaynak ZIP, kullanılan girdi dosyaları, protokol, analiz betiği ve çıktı dosyalarının SHA-256 özetleri kaydedilecektir.

## 9. Çıktılar ve raporlama

Her karşılaştırma için en az şu alanlar kaydedilecektir: varlık, kıyaslama modeli, model ailesi, \(T\), \(\bar d\), \(\hat\gamma_0\), \(\widehat\Omega_B\), \(DM_B\), HLN çarpanı, \(DM_{B,HLN}\), ham p, Holm p, Stage 09 kararı, Bartlett sağlamlık kararı ve karar değişikliği.

Önceki ve yeni kararlar eksiksiz biçimde raporlanacaktır. Yalnız anlamlı kalan veya beklenen yönde değişen sonuçlar seçilmeyecektir. Tez metnindeki yorumlar ancak hesap ve doğrulama tamamlandıktan sonra, elde edilen sonuçların yönüne göre hazırlanacaktır.

## 10. Değişmezlik kuralı

Bu protokolün SHA-256 özeti alındıktan sonra analiz kuralları sonuçlara göre değiştirilmeyecektir. Zorunlu bir yazılım veya veri hatası bulunursa protokol v1.0 korunacak; gerekçesi açıkça yazılmış yeni bir sürüm oluşturulacak ve ayrı SHA-256 özeti alınacaktır.
