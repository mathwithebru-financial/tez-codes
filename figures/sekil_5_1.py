"""Şekil 5.1. Test hedef tarihine göre volatilite tahminleri."""
from ortak import *
import numpy as np, pandas as pd
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator, FuncFormatter
from matplotlib.lines import Line2D
from veri_5_1_hazirla import veri_yukle
args = ayarlar(__doc__)
D, Y, F, Pp = veri_yukle(args.archive)
AD = ["BIST100", "USD/TRY", "EUR/TRY", "Altın"]
C_GER, C_MOD, C_SUR = "#B9C2CC", NAVY, "#E0904A"
AY = {1: "Oca", 4: "Nis", 7: "Tem", 10: "Eki"}
W, H = 5.71, 4.30
fig, axs = plt.subplots(2, 2, figsize=(W, H), dpi=DPI, sharex=True)
fig.subplots_adjust(left=0.085, right=0.985, top=0.865, bottom=0.075, wspace=0.20, hspace=0.30)
for i, a in enumerate(axs.flat):
    a.plot(D, Y[:, i], color=C_GER, lw=2.2, solid_capstyle="round", zorder=1)
    a.plot(D, Pp[:, i], color=C_SUR, lw=0.7, ls=(0, (3, 1.6)), zorder=2)
    a.plot(D, F[:, i], color=C_MOD, lw=0.8, zorder=3)
    a.set_title(AD[i], fontsize=10, color=NAVY, fontweight="bold", pad=3)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): a.spines[sp].set_color(NAVY); a.spines[sp].set_linewidth(0.6)
    a.tick_params(labelsize=10, colors=NAVY, width=0.6, length=2.5, pad=1.5)
    a.grid(axis="y", color="#E6E9EE", lw=0.5); a.set_axisbelow(True)
    a.set_ylim(bottom=0); a.yaxis.set_major_locator(plt.MaxNLocator(4))
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.2f}".replace(".", ",")))
    a.set_xlim(pd.Timestamp("2022-10-01"), pd.Timestamp("2025-01-10"))
ticks = [pd.Timestamp(s) for s in ("2023-01-01", "2024-01-01")]
minor = [pd.Timestamp(s) for s in ("2023-07-01", "2024-07-01", "2025-01-01")]
for a in axs[-1]:
    a.xaxis.set_major_locator(FixedLocator([mdates.date2num(t) for t in ticks]))
    a.xaxis.set_minor_locator(FixedLocator([mdates.date2num(t) for t in minor])); a.tick_params(axis="x", which="minor", length=1.8, width=0.5, colors=NAVY)
    a.xaxis.set_major_formatter(FuncFormatter(lambda v, p: (lambda d: f"{AY[d.month]} {d.year}")(mdates.num2date(v))))
h = [Line2D([], [], color=C_GER, lw=2.2), Line2D([], [], color=C_MOD, lw=0.9), Line2D([], [], color=C_SUR, lw=0.9, ls=(0, (3, 1.6)))]
fig.legend(h, ["Gerçekleşen Vol20", "Nihai model (NoSharing)", "Volatilite sürekliliği"], loc="upper center",
           bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False, fontsize=10, handlelength=2.0, columnspacing=1.2, labelcolor=NAVY)
kaydet(fig, args.output, "5_1")
