# Stage 09 — DM, Harvey ve Holm Sonuç Raporu

## Çalıştırma Kimliği

- Protokol SHA256: `703574062b981e5a4c5187b476c5ce5f7b1e01688818c58ec6d28b436ffbcec1`
- Test gözlemi: `584`
- Tahmin ufku: `h = 1`
- Anlamlılık düzeyi: `0.05`
- Kayıp farkı: `baseline loss - final model loss`

## Karşılaştırma Sayıları

- Getiri: 20
- Volatilite: 22
- Toplam: 42

## Holm Sonuç Özeti

- Nihai model anlamlı derecede daha iyi: 5
- Baseline anlamlı derecede daha iyi: 34
- Anlamlı fark saptanmadı: 3

## Görev Bazında Sonuçlar

- Getiri — nihai model daha iyi: 3; baseline daha iyi: 16; anlamlı değil: 1.
- Volatilite — nihai model daha iyi: 2; baseline daha iyi: 18; anlamlı değil: 2.

## Holm Sonrası Anlamlı Karşılaştırmalar

- `return__BIST100__FinalWinner_vs_ReturnZero`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.0002807802234; HLN p=0.00434528370091; Holm p=0.0173811348037.
- `return__BIST100__FinalWinner_vs_ReturnPersistence`: SIGNIFICANT_FINAL_BETTER; ortalama kayıp farkı=0.0052837425926; HLN p=2.25848986744e-15; Holm p=3.61358378791e-14.
- `return__USDTRY__FinalWinner_vs_ReturnZero`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.00136766913127; HLN p=7.53646579862e-26; Holm p=1.35656384375e-24.
- `return__EURTRY__FinalWinner_vs_ReturnZero`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.000561815188074; HLN p=1.04410542768e-05; Holm p=9.39694884914e-05.
- `return__EURTRY__FinalWinner_vs_ReturnPersistence`: SIGNIFICANT_FINAL_BETTER; ortalama kayıp farkı=0.00132634038731; HLN p=4.42442864894e-07; Holm p=6.19420010852e-06.
- `return__GOLD__FinalWinner_vs_ReturnZero`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.000461559753871; HLN p=5.7002313534e-07; Holm p=6.84027762409e-06.
- `return__GOLD__FinalWinner_vs_ReturnPersistence`: SIGNIFICANT_FINAL_BETTER; ortalama kayıp farkı=0.00301707540101; HLN p=1.8407102592e-21; Holm p=3.12920744063e-20.
- `return__BIST100__FinalWinner_vs_SingleTaskTransformer`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.000281339422516; HLN p=0.00217643521414; Holm p=0.015235046499.
- `return__BIST100__FinalWinner_vs_SingleTaskLSTM`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.000308478004521; HLN p=0.00278555004682; Holm p=0.0167133002809.
- `return__BIST100__FinalWinner_vs_XGBoost`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.000265064446386; HLN p=0.00778047972549; Holm p=0.0233414391765.
- `return__USDTRY__FinalWinner_vs_SingleTaskTransformer`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.000261460296311; HLN p=0.0029640445514; Holm p=0.0167133002809.
- `return__USDTRY__FinalWinner_vs_SingleTaskLSTM`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.00129544020209; HLN p=3.37301768432e-26; Holm p=6.4087336002e-25.
- `return__USDTRY__FinalWinner_vs_XGBoost`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.0013110217609; HLN p=9.80539717969e-29; Holm p=1.96107943594e-27.
- `return__EURTRY__FinalWinner_vs_SingleTaskTransformer`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.000218724092962; HLN p=0.0100914875529; Holm p=0.0233414391765.
- `return__EURTRY__FinalWinner_vs_SingleTaskLSTM`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.000571835838894; HLN p=1.12875489588e-06; Holm p=1.24163038547e-05.
- `return__EURTRY__FinalWinner_vs_XGBoost`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.000421371751111; HLN p=1.65704932097e-05; Holm p=0.000132563945677.
- `return__GOLD__FinalWinner_vs_SingleTaskTransformer`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.000477618373516; HLN p=1.70195111807e-07; Holm p=2.55292667711e-06.
- `return__GOLD__FinalWinner_vs_SingleTaskLSTM`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.000468026315566; HLN p=5.24305856154e-07; Holm p=6.81597613001e-06.
- `return__GOLD__FinalWinner_vs_XGBoost`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.000426979678436; HLN p=7.50628611255e-06; Holm p=7.50628611255e-05.
- `volatility__BIST100__FinalWinner_vs_VolPersistence`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.00188784076743; HLN p=7.54203380409e-11; Holm p=6.03362704327e-10.
- `volatility__USDTRY__FinalWinner_vs_VolPersistence`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.00418030438516; HLN p=3.34548517007e-66; Holm p=7.36006737416e-65.
- `volatility__EURTRY__FinalWinner_vs_VolPersistence`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.000820223867179; HLN p=1.85448864537e-21; Holm p=2.0399375099e-20.
- `volatility__GOLD__FinalWinner_vs_VolPersistence`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.0015746318378; HLN p=6.05244153482e-27; Holm p=1.02891506092e-25.
- `volatility__BIST100__FinalWinner_vs_SingleTaskTransformer`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.000461409210331; HLN p=2.08765156844e-06; Holm p=1.04382578422e-05.
- `volatility__BIST100__FinalWinner_vs_SingleTaskLSTM`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.00161175906878; HLN p=1.50276620846e-10; Holm p=1.05193634592e-09.
- `volatility__BIST100__FinalWinner_vs_XGBoost`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.00091440825366; HLN p=1.76658751853e-05; Holm p=7.06635007413e-05.
- `volatility__USDTRY__FinalWinner_vs_SingleTaskLSTM`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.0028227746202; HLN p=1.05821608007e-46; Holm p=2.22225376815e-45.
- `volatility__USDTRY__FinalWinner_vs_XGBoost`: SIGNIFICANT_FINAL_BETTER; ortalama kayıp farkı=0.00209506591; HLN p=3.42516031382e-29; Holm p=6.16528856487e-28.
- `volatility__EURTRY__FinalWinner_vs_SingleTaskTransformer`: SIGNIFICANT_FINAL_BETTER; ortalama kayıp farkı=0.000616182709895; HLN p=5.97170263891e-18; Holm p=5.37453237502e-17.
- `volatility__EURTRY__FinalWinner_vs_SingleTaskLSTM`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.000396148645208; HLN p=1.43959802475e-06; Holm p=8.63758814853e-06.
- `volatility__GOLD__FinalWinner_vs_SingleTaskTransformer`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.000194503495611; HLN p=5.08960594935e-05; Holm p=0.000152688178481.
- `volatility__GOLD__FinalWinner_vs_SingleTaskLSTM`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.00136603157376; HLN p=3.73107529731e-22; Holm p=4.47729035677e-21.
- `volatility__GOLD__FinalWinner_vs_XGBoost`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.00116222347743; HLN p=1.01714409096e-19; Holm p=1.01714409096e-18.
- `volatility__BIST100__FinalWinner_vs_GARCH_1_1_StudentsT_ZeroMean`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.00377818863716; HLN p=2.33825050709e-23; Holm p=3.03972565921e-22.
- `volatility__EURTRY__FinalWinner_vs_GARCH_1_1_StudentsT_ZeroMean`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.00165570691891; HLN p=1.03842725784e-24; Holm p=1.66148361254e-23.
- `volatility__GOLD__FinalWinner_vs_GARCH_1_1_StudentsT_ZeroMean`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.00242208913923; HLN p=2.93793979539e-36; Holm p=5.58208561123e-35.
- `volatility__BIST100__FinalWinner_vs_GJR_GARCH_1_1_StudentsT_ZeroMean`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.00380137900665; HLN p=1.21176534409e-23; Holm p=1.69647148173e-22.
- `volatility__EURTRY__FinalWinner_vs_GJR_GARCH_1_1_StudentsT_ZeroMean`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.00165105354357; HLN p=1.42973254487e-24; Holm p=2.1445988173e-23.
- `volatility__GOLD__FinalWinner_vs_GJR_GARCH_1_1_StudentsT_ZeroMean`: SIGNIFICANT_BASELINE_BETTER; ortalama kayıp farkı=-0.00242206021018; HLN p=2.56822828447e-36; Holm p=5.13645656894e-35.

## Yorumlama Sınırı

Anlamlı olmayan sonuç, iki modelin eşdeğer olduğunu kanıtlamaz.
Yön yalnızca p-değerinden değil, ortalama kayıp farkının işaretinden belirlenmiştir.

## Çalıştırma Sınırları

- Model yeniden eğitilmedi.
- Yeni tahmin üretilmedi.
- Hiperparametre değiştirilmedi.
- Kayıp fonksiyonu değiştirilmedi.
- Eksik USDTRY–GARCH çiftleri test edilmedi.

