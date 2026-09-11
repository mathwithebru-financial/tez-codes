# Bartlett-HAC DM-HLN Ek Sağlamlık Analizi Sonuç Raporu

## Sonuç

Stage 09'daki 22 volatilite karşılaştırması, önceden dondurulan protokole göre
`h=1`, Bartlett gecikme sınırı `L=19` ve 22 test üzerinde Holm düzeltmesiyle
yeniden hesaplanmıştır.

- Nihai model anlamlı biçimde üstün: **2**
- Kıyaslama modeli anlamlı biçimde üstün: **12**
- Anlamlı fark bulunmadı: **8**
- Stage 09'a göre kararı değişen karşılaştırma: **6**

İki nihai-model üstünlüğü korunmuştur. Altı karşılaştırma, “kıyaslama modeli
anlamlı biçimde üstün” kararından “anlamlı fark bulunmadı” kararına geçmiştir.
Hiçbir karşılaştırmada üstünlük yönü tersine dönmemiştir.

## Kararı değişen karşılaştırmalar

- BIST100 – VolPersistence: Kıyaslama modeli anlamlı biçimde üstün → Anlamlı fark bulunmadı (Holm p=0.0553031).
- BIST100 – SingleTaskTransformer: Kıyaslama modeli anlamlı biçimde üstün → Anlamlı fark bulunmadı (Holm p=0.0712252).
- BIST100 – SingleTaskLSTM: Kıyaslama modeli anlamlı biçimde üstün → Anlamlı fark bulunmadı (Holm p=0.0642773).
- BIST100 – XGBoost: Kıyaslama modeli anlamlı biçimde üstün → Anlamlı fark bulunmadı (Holm p=0.222203).
- EURTRY – SingleTaskLSTM: Kıyaslama modeli anlamlı biçimde üstün → Anlamlı fark bulunmadı (Holm p=0.0548518).
- GOLD – SingleTaskTransformer: Kıyaslama modeli anlamlı biçimde üstün → Anlamlı fark bulunmadı (Holm p=0.222203).

## Denetim sonuçları

- Gözlem sayısı: her karşılaştırmada **584**
- Volatilite karşılaştırması: **22**
- Stage 09 L=0 yeniden üretiminde en büyük mutlak DM farkı:
  **3.553e-15**
- Özel Bartlett hesabı ile statsmodels çapraz kontrolündeki en büyük mutlak
  düzeltilmiş DM farkı: **1.776e-15**
- En büyük mutlak ham p-değeri farkı:
  **2.255e-17**
- Çapraz kontrol toleransı: **1.0e-10**
- Doğrulama durumu: **PASS**

## Yorum sınırı

Bu çalışma Stage 09 ana analizini değiştirmez; seri bağımlılık varsayımına
duyarlılığı gösteren post-hoc bir sağlamlık analizidir. `L=19`, 20 günlük
hareketli volatilite hedeflerindeki örtüşmenin mekanik olarak işaret ettiği
gecikme sınırıdır; bağımlılığın 19. gecikmede bittiğini kanıtlamaz.

## Teze eklenebilecek sonuç paragrafı

Volatilite görevi için raporlanan DM-HLN sonuçlarının örtüşen hedef yapısından
kaynaklanabilecek seri bağımlılığa duyarlılığı, ek bir sağlamlık analizinde
Bartlett ağırlıklı HAC uzun dönem varyans tahmincisi kullanılarak incelenmiştir.
Gerçek tahmin ufku `h=1` olarak korunmuş, gecikme sınırı 20 günlük volatilite
pencerelerinin mekanik örtüşmesine dayanarak `L=19` seçilmiş ve 22 ham p-değeri
tek aile içinde Holm yöntemiyle düzeltilmiştir. Bu uygulamada nihai modelin
anlamlı üstün olduğu iki karşılaştırma korunurken, kıyaslama modelinin anlamlı
üstün olduğu karşılaştırma sayısı 18'den 12'ye düşmüş; anlamlı fark bulunmayan
karşılaştırma sayısı 2'den 8'e yükselmiştir. Altı karar kıyaslama üstünlüğünden
anlamlı fark bulunmamasına dönüşmüş, hiçbir karşılaştırmada üstünlük yönü tersine
dönmemiştir. Bulgular, bazı kıyaslama üstünlüklerinin kayıp farklarındaki seri
bağımlılığın daha geniş biçimde hesaba katılmasına duyarlı olduğunu göstermektedir.
