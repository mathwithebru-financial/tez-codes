"""Şekil 2.3. Transformer mimarisi."""
from ortak import *
W, H = 4.13, 5.98
fig, ax = tuval(W, H)
bw, bh, gap = 2.60, 0.40, 0.13
cx = 2.30
x_res = cx - bw / 2 - 0.22            # artık bağlantı hattı (Şekil 2.2'deki gibi solda)
satirlar = [
    ("Girdi dizisi\n10 gözlem × 8 özellik", F_GRI),
    ("Doğrusal izdüşüm\n$d_{\\mathrm{model}}$ = 32", F_BEYAZ),
    ("Öğrenilebilir konumsal gömme ($\\mathbf{E}$)\neklenir", F_BEYAZ),
    ("Çok başlı öz-dikkat\n$h$ = 4 · Dropout", F_MAVI),
    ("Ekle ve normalleştir", F_BEYAZ),
    ("İleri besleme ağı\n$d_{\\mathrm{ff}}$ = 128 · GELU · Dropout", F_YES),
    ("Ekle ve normalleştir", F_BEYAZ),
    ("Son konumun temsilini seç\n$\\mathbf{H}^{\\mathrm{(2)}}_{\\mathrm{10,:}}$", F_BEYAZ),
    ("Katman normalizasyonu", F_BEYAZ),
    ("Göreve özgü çıktı başlığı (4 çıktı)\nDoğrusal · GELU · Dropout · Doğrusal", F_TUR),
]
UST, ALT = 0.30, 0.24                  # çerçeve üst ve alt payı
toplam = len(satirlar) * bh + (len(satirlar) - 1) * gap + UST + ALT
y = H - (H - toplam) / 2 - bh / 2
ys = []
for i in range(len(satirlar)):
    if i == 3: y -= UST
    if i == 7: y -= ALT
    ys.append(y); y -= bh + gap
for (t, fc), yy in zip(satirlar, ys):
    kutu(ax, cx, yy, bw, bh, t, fc=fc)
for i in range(len(ys) - 1):
    ok(ax, cx, ys[i] - bh / 2, cx, ys[i + 1] + bh / 2)
top = ys[3] + bh / 2 + UST - 0.10; bot = ys[6] - bh / 2 - ALT + 0.10
x0 = x_res - 0.12; x1 = cx + bw / 2 + 0.12
kesikli(ax, x0, bot, x1 - x0, top - bot)
ax.text(x1 + 0.05, (top + bot) / 2, "Kodlayıcı katmanı · 2 ×", rotation=270, ha="left", va="center",
        fontsize=10, color=NAVY, fontweight="bold")
for a, b in [(3, 4), (5, 6)]:
    yb = ys[a] + bh / 2 + (0.10 if a == 3 else gap / 2)
    ax.plot([cx], [yb], "o", ms=2.2, color=NAVY, zorder=3)
    cizgi(ax, [cx, x_res], [yb, yb]); cizgi(ax, [x_res, x_res], [yb, ys[b]])
    ok(ax, x_res, ys[b], cx - bw / 2, ys[b])
ax.text(x0 - 0.05, (ys[3] + ys[6]) / 2, "Artık bağlantılar", rotation=90, ha="right", va="center",
        fontsize=10, color=NAVY)
kaydet(fig, ayarlar(__doc__).output, "2_3")
