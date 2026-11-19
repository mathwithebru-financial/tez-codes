"""Şekil 3.2. Ortak takvimdeki günlük fiyat, getiri ve volatilite serileri."""
from ortak import *
from pathlib import Path
args = ayarlar(__doc__)
import numpy as np, pandas as pd, hashlib
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator
RAW = args.archive / "data/raw/raw_prices.csv"
assert hashlib.sha256(open(RAW, "rb").read()).hexdigest() == "ab5f275d38dc98057b1cedcf58019adb26be7402c7ed5ae6ee3d6877b2444893"
raw = pd.read_csv(RAW, parse_dates=["Date"], index_col="Date")
P = raw.ffill(); R = np.log(P / P.shift(1)); V = R.rolling(20, min_periods=20).std() * np.sqrt(252)
VARLIK = [("BIST100", "BIST100", "#2B6CB0"), ("USDTRY", "USD/TRY", "#D9822B"), ("EURTRY", "EUR/TRY", "#3E9B6B"), ("GOLD", "Altın", "#9B4F96")]
D_VAL, D_TEST = pd.Timestamp("2020-07-10"), pd.Timestamp("2022-10-06")
def tr(v, nd):
    s = f"{v:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return s.replace("-", "−")
fig, axs = plt.subplots(4, 3, figsize=(5.51, 5.75), dpi=DPI, sharex=True)
fig.subplots_adjust(left=0.165, right=0.972, top=0.895, bottom=0.075, wspace=0.42, hspace=0.16)
for i, (k, ad, renk) in enumerate(VARLIK):
    seriler = [P[k].dropna(), R[k].dropna(), V[k].dropna()]
    for j, s in enumerate(seriler):
        a = axs[i, j]
        a.plot(s.index, s.values, color=renk, lw=0.45)
        a.axvline(D_VAL, color=GRI_K, lw=0.7, ls=(0, (3, 2)))
        a.axvline(D_TEST, color=NAVY, lw=0.7, ls=(0, (5, 1.5, 1, 1.5)))
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): a.spines[sp].set_color(NAVY); a.spines[sp].set_linewidth(0.6)
        a.tick_params(labelsize=10, colors=NAVY, width=0.6, length=2.5, pad=1.5)
        a.grid(axis="y", color="#E6E9EE", lw=0.5); a.set_axisbelow(True)
        a.set_xlim(pd.Timestamp("2010-01-01"), pd.Timestamp("2025-01-01"))
        TIK = {("BIST100", 0): [1000, 3000, 10000], ("USDTRY", 0): [0, 15, 30], ("EURTRY", 0): [0, 15, 30], ("GOLD", 0): [1200, 1800, 2400],
               ("BIST100", 1): [-0.10, 0, 0.10], ("USDTRY", 1): [-0.20, 0, 0.20], ("EURTRY", 1): [-0.20, 0, 0.20], ("GOLD", 1): [-0.05, 0, 0.05],
               ("BIST100", 2): [0, 0.25, 0.50], ("USDTRY", 2): [0, 0.50, 1.00], ("EURTRY", 2): [0, 0.50, 1.00], ("GOLD", 2): [0, 0.20, 0.40]}
        if j == 0 and k == "BIST100":
            a.set_yscale("log"); a.yaxis.set_minor_locator(NullLocator())
        a.yaxis.set_major_locator(FixedLocator(TIK[(k, j)]))
        nd = 0 if j == 0 else 2
        a.yaxis.set_major_formatter(FuncFormatter(lambda v, p, nd=nd: tr(v, nd)))
        if j == 2 or (j == 0 and k in ("USDTRY", "EURTRY")): a.set_ylim(bottom=0)
        if j == 1: m = max(abs(s.min()), abs(s.max())) * 1.05; a.set_ylim(-m, m)
    axs[i, 0].set_ylabel(ad, fontsize=10, color=NAVY, labelpad=4)
for j, t in enumerate(["Günlük fiyat", "Logaritmik getiri", "20 gözlemlik volatilite"]):
    axs[0, j].set_title(t, fontsize=10, color=NAVY, pad=5)
for a in axs[-1]:
    a.xaxis.set_major_locator(mdates.YearLocator(base=7, month=1, day=1))
    a.xaxis.set_major_locator(FixedLocator([mdates.date2num(pd.Timestamp(f"{y}-01-01")) for y in (2010, 2017, 2024)]))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.text(0.575, 0.012, "Yıl", ha="center", va="bottom", fontsize=10, color=NAVY)
from matplotlib.lines import Line2D
h = [Line2D([], [], color=GRI_K, lw=0.9, ls=(0, (3, 2))), Line2D([], [], color=NAVY, lw=0.9, ls=(0, (5, 1.5, 1, 1.5)))]
fig.legend(h, ["Doğrulama başlangıcı (10.07.2020)", "Test başlangıcı (06.10.2022)"], loc="upper center",
           bbox_to_anchor=(0.575, 0.995), ncol=2, frameon=False, fontsize=10, handlelength=2.2, columnspacing=1.2,
           labelcolor=NAVY)
fig.align_ylabels(axs[:, 0])
args.output.mkdir(parents=True, exist_ok=True)
hedef = args.output / "Sekil_3_2.png"
fig.savefig(hedef, dpi=600)
plt.close(fig)
print(hedef)
