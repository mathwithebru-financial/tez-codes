"""Şekil 4.2. Girdi penceresindeki zaman adımlarının SHAP önem payları."""

from ortak import ayarlar
from shap_isiharitasi import LEFT, WIDTH, SYMBOLS, draw_heatmap, read_shares


def main():
    args = ayarlar(__doc__)
    shares = read_shares(args.archive, "temporal", "lag_index", 10)
    matrix = [[shares[output, lag] for lag in range(10)] for output in range(8)]
    labels = [f"{symbol} {target}" for target in ("getirisi", "volatilitesi")
              for symbol in SYMBOLS]
    headings = [f"t−{i}" for i in range(9, 0, -1)] + ["t"]
    cw = 181
    titles = [(LEFT + 5 * cw, 54, "GİRDİ PENCERESİNDEKİ KONUM", (0, 0, WIDTH, 119))]
    notes = [
        "Her satır, bir çıktı için mutlak SHAP öneminin on gözleme yüzde olarak dağılımını",
        "gösterir. t, ortak takvimdeki en güncel girdi gözlemidir.",
    ]
    draw_heatmap(args.output / "Sekil_4_2.png", headings, labels, matrix, titles, notes, cw)


if __name__ == "__main__":
    main()
