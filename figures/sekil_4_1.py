"""Şekil 4.1. Çıktı bazında SHAP küresel önem payları."""

from ortak import ayarlar
from shap_isiharitasi import LEFT, WIDTH, SYMBOLS, draw_heatmap, read_shares


def main():
    args = ayarlar(__doc__)
    shares = read_shares(args.archive, "global", "feature_index", 8)
    matrix = [[shares[output, feature] for output in range(8)] for feature in range(8)]
    labels = [f"{symbol} {feature}" for symbol in SYMBOLS
              for feature in ("getirisi", "volatilitesi")]
    cw = 227
    titles = [
        (LEFT + 2 * cw, 53, "GETİRİ ÇIKTILARI", (LEFT, 0, LEFT + 4 * cw, 120)),
        (LEFT + 6 * cw, 53, "VOLATİLİTE ÇIKTILARI", (LEFT + 4 * cw, 0, WIDTH, 120)),
    ]
    notes = [
        "Her sütun, bir çıktı için mutlak SHAP öneminin sekiz girdi özelliğine yüzde",
        "olarak dağılımını gösterir. Paylar test gözlemleri ve gecikmeler üzerinden hesaplanmıştır.",
    ]
    draw_heatmap(args.output / "Sekil_4_1.png", SYMBOLS * 2, labels, matrix, titles, notes, cw)


if __name__ == "__main__":
    main()
