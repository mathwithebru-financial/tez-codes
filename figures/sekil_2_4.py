"""Şekil 2.4. Transformer mimarisi."""
from ortak import *
from matplotlib.patches import FancyBboxPatch
W, H = 5.70, 6.62
fig, ax = tuval(W, H)
PW, G = 2.72, 0.12
PH1, PH2 = 3.02, 3.38
ox = (W - 2 * PW - G) / 2; oy = (H - PH1 - PH2 - G) / 2
def panel(satir, sutun, baslik, alt):
    x = ox + sutun * (PW + G)
    y, ph = (oy + PH2 + G, PH1) if satir == 0 else (oy, PH2)
    kesikli(ax, x, y, PW, ph)
    T = y + ph
    ax.text(x + PW / 2, T - 0.17, baslik, ha="center", va="center", fontsize=10, fontweight="bold", color=NAVY)
    ax.text(x + PW / 2, T - 0.36, alt, ha="center", va="center", fontsize=10, color=NAVY)
    return x, T
def bol(cx, ytop, xl, xr, ybot):
    ym = (ytop + ybot) / 2
    cizgi(ax, [cx, cx], [ytop, ym]); ax.plot([cx], [ym], "o", ms=2.0, color=NAVY, zorder=3)
    cizgi(ax, [xl, xr], [ym, ym]); ok(ax, xl, ym, xl, ybot); ok(ax, xr, ym, xr, ybot)
GH, SW, BH = 0.30, 1.18, 0.40
def girdi(cx, T):
    kutu(ax, cx, T - 0.70, 1.35, GH, "Girdi (10 × 8)", fc=F_GRI); return T - 0.70 - GH / 2
def iki_kutu(x, yc, tl, tr, fcl=F_TUR, fcr=F_YES):
    xl, xr = x + 0.12 + SW / 2, x + PW - 0.12 - SW / 2
    kutu(ax, xl, yc, SW, BH, tl, fc=fcl); kutu(ax, xr, yc, SW, BH, tr, fc=fcr); return xl, xr
# (a)
x, T = panel(0, 0, "(a) Tam paylaşım", "FullSharingMTL"); cx = x + PW / 2
gb = girdi(cx, T); ok(ax, cx, gb, cx, T - 1.00)
kutu(ax, cx, T - 1.33, 2.10, 0.66, "Ortak gömme ve konum\n2 kodlayıcı katmanı\nSon normalizasyon", fc=F_MAVI)
xl, xr = x + 0.12 + SW / 2, x + PW - 0.12 - SW / 2
bol(cx, T - 1.66, xl, xr, T - 2.00)
iki_kutu(x, T - 2.20, "Getiri başlığı\n4 çıktı", "Volatilite başlığı\n4 çıktı")
ax.text(cx, T - 2.74, "İki görev aynı temsili kullanır.", ha="center", va="center", fontsize=10, color=NAVY)
# (b)
x, T = panel(0, 1, "(b) Kısmi paylaşım", "PartialSharingMTL"); cx = x + PW / 2
xl, xr = x + 0.12 + SW / 2, x + PW - 0.12 - SW / 2
gb = girdi(cx, T); ok(ax, cx, gb, cx, T - 1.00)
kutu(ax, cx, T - 1.25, 2.10, 0.50, "Ortak gömme ve konum\n1 ortak kodlayıcı katmanı", fc=F_MAVI)
bol(cx, T - 1.50, xl, xr, T - 1.78)
iki_kutu(x, T - 1.98, "1 ayrık katman\n+ son norm", "1 ayrık katman\n+ son norm")
for xx in (xl, xr): ok(ax, xx, T - 2.18, xx, T - 2.42)
iki_kutu(x, T - 2.62, "Getiri başlığı\n4 çıktı", "Volatilite başlığı\n4 çıktı")
# (c)
x, T = panel(1, 0, "(c) Hiyerarşik paylaşım", "HierarchicalMTL"); cx = x + PW / 2
xl, xr = x + 0.12 + SW / 2, x + PW - 0.12 - SW / 2
gb = girdi(cx, T); ok(ax, cx, gb, cx, T - 1.00)
kutu(ax, cx, T - 1.33, 2.10, 0.66, "Ortak gömme ve kodlayıcı\nSon normalizasyon\n(temel temsil)", fc=F_MAVI)
bol(cx, T - 1.66, xl, xr, T - 2.02)
iki_kutu(x, T - 2.22, "Getiri\nara temsili", "Volatilite temsili\n[temel; getiri]")
ok(ax, xl + SW / 2, T - 2.22, xr - SW / 2, T - 2.22, renk=VURGU, lw=1.1)
for xx in (xl, xr): ok(ax, xx, T - 2.42, xx, T - 2.70)
iki_kutu(x, T - 2.90, "Getiri başlığı\n4 çıktı", "Volatilite başlığı\n4 çıktı")
# (d)
x, T = panel(1, 1, "(d) Paylaşım yok — nihai", "NoSharing (toplam 54.536 parametre)"); cx = x + PW / 2
gb = girdi(cx, T)
DW = 1.26; xl, xr = x + 0.08 + DW / 2, x + PW - 0.08 - DW / 2
fr_top = T - 1.16; fr_bot = T - PH2 + 0.06
bol(cx, gb, xl, xr, fr_top)
for xx, fc, ad in [(xl, F_TUR, "Getiri"), (xr, F_YES, "Volatilite")]:
    ax.add_patch(FancyBboxPatch((xx - DW / 2, fr_bot), DW, fr_top - fr_bot, boxstyle="round,pad=0,rounding_size=0.06",
                                fc="none", ec=GRI_K, lw=0.9, zorder=1))
    adim = [("Gömme + konum", 0.29), ("2 kodlayıcı\nkatmanı", 0.40), ("Son norm", 0.29), (f"{ad} başlığı\n4 çıktı", 0.40)]
    yy = fr_top - 0.07
    for k, (t, hh) in enumerate(adim):
        kutu(ax, xx, yy - hh / 2, DW - 0.10, hh, t, fc=fc)
        yy -= hh
        if k < len(adim) - 1: ok(ax, xx, yy, xx, yy - 0.08); yy -= 0.08
    ax.text(xx, (yy + fr_bot) / 2, f"{ad} dalı\n27.268 parametre", ha="center", va="center", fontsize=10,
            fontweight="bold", color=NAVY, linespacing=1.15)
kaydet(fig, ayarlar(__doc__).output, "2_4")
