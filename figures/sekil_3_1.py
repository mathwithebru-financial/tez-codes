"""Şekil 3.1. Ana deneyin veri hazırlığı, model seçimi ve nihai test adımları."""
import json

from ortak import ayarlar
from sema_cizimi import BLUE, GREEN, GREY, ORANGE, Canvas


def binlik(sayi):
    return f"{sayi:,}".replace(",", ".")


def ciz(proje, cikti):
    islenmis = proje / "data/processed"
    veri = json.loads((islenmis / "meta_v4.json").read_text(encoding="utf-8"))
    bolme = json.loads((islenmis / "split_meta_v4.json").read_text(encoding="utf-8"))
    pencere = json.loads((proje / "data/sequences/baseline/lb10/sequence_meta.json")
                         .read_text(encoding="utf-8"))
    secim = json.loads((proje / "results/multiseed/multiseed_winner_config_v4.json")
                       .read_text(encoding="utf-8"))
    ham_satir, varlik = veri["raw_data"]["shape"]
    egitim = binlik(bolme["train"]["n_rows"])
    dogrulama = bolme["validation"]["n_rows"]
    test = bolme["test"]["n_rows"]
    diziler = pencere["shapes"]
    if diziler["X_val"][0] != diziler["X_test"][0]:
        raise ValueError("Baskıdaki ortak doğrulama ve test örnek sayısı arşivle uyuşmuyor.")
    tohumlar = ", ".join(map(str, secim["seeds"]))
    aday = secim["candidate_count"]
    egitim_kosusu = aday * len(secim["seeds"])

    sol = [
        (f"1. Günlük ham fiyatlar\n{binlik(ham_satir)} ortak takvim tarihi\n{varlik} varlık", GREY),
        ("2. SHA-256 ile\nveri kimliği ve bütünlük denetimi", "white"),
        ("3. Ortak takvim\nYalnız ileri doldurma\nGeriye doldurma yok", "white"),
        ("4. Getiri ve Vol20 üretimi\nNextRet / NextVol hedefleri\n"
         f"{binlik(bolme['total_rows'])} geçerli tarih satırı", "white"),
        (f"5. Hedef tarihine göre bölme\nEğitim: {egitim}\nDoğrulama: {dogrulama} • Test: {test}",
         BLUE),
    ]
    sag = [
        ("6. Eğitim verisiyle ölçekleme\n10, 20, 30, 60 gözlemlik pencereler\n"
         f"10 gözlemlik pencerede eğitim: {binlik(diziler['X_train'][0])}\n"
         f"Doğrulama ve testte {diziler['X_test'][0]}’er örnek", "white"),
        ("7. 480 yapılandırmanın taranması\nRastgelelik tohumu 42\nDoğrulama skoruna göre sıralama",
         "white"),
        (f"8. İlk {aday} adayın her biri için üç eğitim\nRastgelelik tohumları {tohumlar}\n"
         f"{egitim_kosusu} eğitim, doğrulama ile seçim", "white"),
        ("9. Nihai yapı ve 3 kontrol noktası\nTest sonuçları görülmeden sabitlenir\n"
         "Kimlikler SHA-256 ile kaydedilir", ORANGE),
        (f"10. Nihai test ve kıyaslamalar\n{test} test tarihinde sekiz çıktı\n"
         "Üç modelin tahmin ortalaması\nDM–HLN, Holm ve SHAP", GREEN),
    ]

    tuval = Canvas(1000, 930)
    genislik, yukseklik, aralik = 420, 135, 32
    ust = 65
    adim = yukseklik + aralik
    tuval.frame(540, 45, 450, 3 * adim + yukseklik + 35)
    tuval.text(765, 25, "Test kümesi model seçiminde kullanılmadı", 23, True)
    for x, sutun in ((25, sol), (555, sag)):
        for sira, (etiket, renk) in enumerate(sutun):
            y = ust + sira * adim
            tuval.box(x, y, genislik, yukseklik, etiket, renk, 23)
            if sira < len(sutun) - 1:
                orta = x + genislik / 2
                tuval.line([(orta, y + yukseklik), (orta, y + adim)], arrow=True)

    # Sol sütunun son kutusundan sağ sütunun ilk kutusuna geçiş.
    ilk_orta = ust + yukseklik / 2
    son_orta = ilk_orta + 4 * adim
    tuval.line([(445, son_orta), (500, son_orta), (500, ilk_orta), (555, ilk_orta)], arrow=True)
    tuval.save(cikti, "3_1")


if __name__ == "__main__":
    args = ayarlar(__doc__)
    ciz(args.archive, args.output)
