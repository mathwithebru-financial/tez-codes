"""Şekil 2.1. Girdi penceresi, hedef oluşumu ve ardışık volatilite pencereleri."""
from ortak import ayarlar
from sema_cizimi import Canvas, ORANGE


def draw(output):
    c = Canvas(1000, 600)
    c.text(370, 40, "Girdi: 10 ortak takvim gözlemi × 8 özellik", 25)
    c.text(875, 40, "Hedefler", 25, True)
    for i in range(10):
        c.box(70 + i * 62, 95, 50, 78, fill="#cfdfef", stroke="#7699bb", r=0)
    c.line([(705, 134), (800, 134)], arrow=True)
    c.box(820, 95, 110, 78, "4 getiri\n4 Vol20", ORANGE, 23)
    c.text(95, 204, "t − 9", 25)
    c.text(653, 204, "t", 25)
    c.text(875, 220, "r(t + 1)\nVol20(t + 1)", 23)
    c.text(500, 278, "(a) Girdi penceresi ve sonraki ortak takvim satırındaki hedef")

    # Each square is one return. The window shifts by one common observation.
    for i in range(20):
        c.box(170 + i * 35, 340, 30, 44, fill="#91bfa9", stroke="white", r=0)
        fill = "#b45f43" if i == 19 else "#91bfa9"
        c.box(205 + i * 35, 420, 30, 44, fill=fill, stroke="white", r=0)
    c.text(25, 362, "Vol20(t)", 25, anchor="start")
    c.text(25, 442, "Vol20(t + 1)", 25, anchor="start")
    c.line([(200, 329), (870, 329), (870, 475), (200, 475), (200, 329)],
           color="#a15339", width=2, dash=True)
    c.text(520, 510, "İki pencerede ortak olan 19 getiri", 25, color="#91462f")
    c.text(899, 510, "Yeni getiri", 23, color="#91462f")
    c.text(500, 563, "(b) Ardışık yıllıklandırılmış 20 gözlemlik volatilite pencereleri")
    c.save(output, "2_1")


if __name__ == "__main__":
    draw(ayarlar(__doc__).output)
