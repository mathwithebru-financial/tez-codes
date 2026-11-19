"""Şekil 3.3. Hedefin gerçekleşme tarihine göre kronolojik veri bölme."""
import json
from datetime import date

from ortak import ayarlar
from sema_cizimi import BLUE, GREEN, ORANGE, Canvas


def ciz(proje, cikti):
    kaynak = proje / "data/processed/split_meta_v4.json"
    bolme = json.loads(kaynak.read_text(encoding="utf-8"))
    if bolme["split_basis"] != "target_realization_date":
        raise ValueError("Şekil 3.3, hedefin gerçekleşme tarihine göre yapılan bölmeyi gösterir.")

    tuval = Canvas(1000, 390)
    tuval.text(500, 30, "Hedefin gerçekleşme tarihi", 26, True)
    tuval.text(500, 65, "Bir sonraki ortak takvim tarihi", 24)
    tuval.line([(75, 103), (925, 103)], arrow=True)

    # Bant genişliği pencereleme öncesindeki hedef tarihlerinin sayısıyla orantılıdır.
    x = 75
    sinirlar = []
    for anahtar, baslik, renk in (
        ("train", "Eğitim", BLUE),
        ("validation", "Doğrulama", GREEN),
        ("test", "Test", ORANGE),
    ):
        donem = bolme[anahtar]
        sayi = donem["n_rows"]
        sinirlar.append((x, donem["target_realization_start"]))
        genislik = 850 * sayi / bolme["total_rows"]
        adet = f"{sayi:,}".replace(",", ".")
        tuval.box(x, 143, genislik, 112, f"{baslik}\nn = {adet}", renk, 24, bold=True, r=0)
        x += genislik
    sinirlar.append((x, bolme["test"]["target_realization_end"]))
    for x, tarih in sinirlar:
        tuval.line([(x, 255), (x, 277)])
        tuval.text(x, 307, date.fromisoformat(tarih).strftime("%d.%m.%Y"), 22)
    tuval.text(500, 363, "n, pencereleme öncesindeki hedef tarihi sayısıdır.", 23)
    tuval.save(cikti, "3_3")


if __name__ == "__main__":
    args = ayarlar(__doc__)
    ciz(args.archive, args.output)
