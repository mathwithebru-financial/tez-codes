import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib import font_manager as fm

FONT = "Times New Roman"
plt.rcParams.update({
    "font.family": "serif", "font.serif": [FONT], "font.size": 10,
    "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": FONT + ":italic",
    "mathtext.bf": FONT + ":bold", "mathtext.default": "it", "mathtext.cal": FONT, "mathtext.sf": FONT, "mathtext.tt": FONT,
    "axes.unicode_minus": True, "svg.fonttype": "none", "pdf.fonttype": 42,
})
for weight, style in (("normal", "normal"), ("bold", "normal"), ("normal", "italic"), ("bold", "italic")):
    prop = fm.FontProperties(family=FONT, weight=weight, style=style)
    path = fm.findfont(prop, fallback_to_default=False)
    if fm.FontProperties(fname=path).get_name() != FONT:
        raise RuntimeError("Bu baskı için Times New Roman yazı tipini kurun.")

NAVY = "#24364B"; GRI_K = "#7C8A99"
F_GRI = "#F2F4F7"; F_MAVI = "#E3EEF9"; F_TUR = "#FCE8CE"; F_YES = "#E0F1E8"; F_BEYAZ = "#FFFFFF"
VURGU = "#A25D12"
DPI = 600

def tuval(w, h):
    fig = plt.figure(figsize=(w, h), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax

def kutu(ax, cx, cy, w, h, metin, fc=F_BEYAZ, fs=10, kalin=False, ec=NAVY, lw=0.9, renk=NAVY, ls=1.2):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.045",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(cx, cy, metin, ha="center", va="center", fontsize=fs, color=renk,
            fontweight="bold" if kalin else "normal", linespacing=ls, zorder=3)

def ok(ax, x0, y0, x1, y1, renk=NAVY, lw=0.9, bas=True):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                arrowprops=dict(arrowstyle="-|>" if bas else "-", color=renk, lw=lw,
                                mutation_scale=7, shrinkA=0, shrinkB=0))

def cizgi(ax, xs, ys, renk=NAVY, lw=0.9):
    ax.plot(xs, ys, color=renk, lw=lw, solid_capstyle="butt", zorder=1)

def kesikli(ax, x, y, w, h, renk=GRI_K, lw=0.8):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=renk, lw=lw, ls=(0, (4, 3)), zorder=0))

def ayarlar(aciklama):
    import argparse
    from pathlib import Path
    p = argparse.ArgumentParser(description=aciklama)
    p.add_argument("--archive", type=Path, required=True,
                   help="main_project klasörünü içeren tez arşivi veya main_project klasörü")
    p.add_argument("--output", type=Path, default=Path("sekiller"), help="PNG çıktı klasörü")
    args = p.parse_args()
    if (args.archive / "main_project").is_dir():
        args.archive = args.archive / "main_project"
    if not (args.archive / "data/raw/raw_prices.csv").is_file():
        p.error("Arşivde main_project/data/raw/raw_prices.csv bulunamadı.")
    return args

def kaydet(fig, output, numara):
    from PIL import Image
    from io import BytesIO
    output.mkdir(exist_ok=True, parents=True)
    hedef = output / f"Sekil_{numara}.png"
    with BytesIO() as data:
        fig.savefig(data, dpi=DPI, facecolor="white", format="png")
        plt.close(fig)
        data.seek(0)
        with Image.open(data) as im:
            im.convert("RGB").save(hedef, dpi=(DPI, DPI), optimize=True)
    print(hedef)
