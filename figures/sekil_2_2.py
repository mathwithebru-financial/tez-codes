"""Şekil 2.2. Özgün Transformer mimarisinde kodlayıcı ve kod çözücü."""
from ortak import ayarlar
from sema_cizimi import BLUE, GREEN, GREY, ORANGE, Canvas


def residual(c, x, start_y, end_y, side_x):
    """Route a sublayer's input around the sublayer to Add & Norm."""
    c.dot(x, start_y)
    edge_x = x + 155 if side_x > x else x - 155
    c.line([(x, start_y), (side_x, start_y), (side_x, end_y), (edge_x, end_y)],
           arrow=True)


def draw(output):
    c = Canvas(1000, 1050, min_font_size=26.15)
    encoder, decoder = 245, 755
    width, height = 310, 64
    left, right = encoder - width / 2, decoder - width / 2
    c.text(encoder, 65, "KODLAYICI", 30, True)
    c.text(decoder, 65, "KOD ÇÖZÜCÜ", 30, True)
    c.text(encoder, 993, "Girdi dizisi", 25)
    c.text(decoder, 993, "Hedef dizisi\n(bir konum sağa kaydırılmış)")
    c.box(left, 870, width, 65, "Girdi gömme katmanı", GREY)
    c.box(right, 870, width, 65, "Çıktı gömme katmanı", GREY)
    c.box(left, 786, width, 64, "+ Konumsal kodlama", GREY)
    c.box(right, 786, width, 64, "+ Konumsal kodlama", GREY)
    for x in (encoder, decoder):
        c.line([(x, 961), (x, 935)], arrow=True)
        c.line([(x, 870), (x, 850)], arrow=True)

    c.frame(44, 352, 402, 416)
    c.text(68, 335, "N ×", 23, True, anchor="start")
    c.box(left, 684, width, height, "Çok başlı öz-dikkat", BLUE)
    c.box(left, 602, width, height, "Ekle ve normalleştir")
    c.box(left, 484, width, 88, "Konum bazlı\nileri besleme ağı", GREEN)
    c.box(left, 402, width, height, "Ekle ve normalleştir")
    for y1, y2 in [(786, 748), (684, 666), (602, 572), (484, 466)]:
        c.line([(encoder, y1), (encoder, y2)], arrow=True)
    residual(c, encoder, 762, 634, 62)
    residual(c, encoder, 588, 434, 62)

    c.frame(554, 202, 402, 566)
    c.text(554, 190, "N ×", 23, True, anchor="start")
    c.box(right, 684, width, height, "Maskeli çok başlı\nöz-dikkat", BLUE)
    c.box(right, 602, width, height, "Ekle ve normalleştir")
    c.box(right, 488, width, 84, "Çok başlı çapraz dikkat", ORANGE)
    c.box(right, 406, width, height, "Ekle ve normalleştir")
    c.box(right, 298, width, 78, "Konum bazlı\nileri besleme ağı", GREEN)
    c.box(right, 216, width, height, "Ekle ve normalleştir")
    for y1, y2 in [(786, 748), (684, 666), (602, 572), (488, 470), (406, 376), (298, 280)]:
        c.line([(decoder, y1), (decoder, y2)], arrow=True)
    residual(c, decoder, 762, 634, 938)
    residual(c, decoder, 588, 438, 938)
    residual(c, decoder, 392, 248, 938)
    c.box(right, 116, width, 70, "Doğrusal katman\nve softmax", GREY)
    c.line([(decoder, 216), (decoder, 186)], arrow=True)
    c.line([(decoder, 116), (decoder, 96)], arrow=True)

    # The decoder supplies the query; the encoder supplies the keys and values.
    c.line([(encoder, 402), (encoder, 306), (508, 306), (508, 530), (right, 530)],
           color="#a25d12", width=2.7, arrow=True)
    c.text(484, 285, "K, V", 23, True, color="#86500e")
    c.text(729, 590, "Q", 22, color="#86500e")
    c.line([(80, 163), (142, 163)], arrow=True)
    c.text(154, 163, "Yan yollar: artık bağlantılar", 23, anchor="start")
    c.save(output, "2_2")


if __name__ == "__main__":
    draw(ayarlar(__doc__).output)
