"""Şekil 4.3. Holm düzeltilmiş temel karşılaştırmalar."""
from ortak import *
from collections import Counter
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter
from matplotlib.transforms import blended_transform_factory
from PIL import Image

args = ayarlar(__doc__)
DATA = args.archive / "results/statistical_comparison/09_dm_harvey_holm_results_v4.csv"
ORANGE = "#E0904A"
GRAY = "#606F7F"
plt.rcParams.update({"font.family": FONT, "font.size": 10, "axes.unicode_minus": True})

with DATA.open(encoding="utf-8-sig", newline="") as file:
    rows = list(csv.DictReader(file))
observations = {(r["task"], r["baseline_model"], r["asset"]): r for r in rows}
assert len(rows) == len(observations) == 42

groups = {
    "return": [("ReturnZero", "Sıfır getiri"), ("ReturnPersistence", "Getiri sürekliliği"),
               ("SingleTaskTransformer", "Tek görevli Transformer"), ("SingleTaskLSTM", "Tek görevli LSTM"),
               ("XGBoost", "XGBoost")],
    "volatility": [("VolPersistence", "Volatilite sürekliliği"), ("GARCH_1_1_StudentsT_ZeroMean", "GARCH(1,1)"),
                   ("GJR_GARCH_1_1_StudentsT_ZeroMean", "GJR-GARCH(1,1)"),
                   ("SingleTaskTransformer", "Tek görevli Transformer"), ("SingleTaskLSTM", "Tek görevli LSTM"),
                   ("XGBoost", "XGBoost")],
}
assets = {"BIST100": "BIST100", "USDTRY": "USD/TRY", "EURTRY": "EUR/TRY", "GOLD": "Altın"}
styles = {"SIGNIFICANT_FINAL_BETTER": dict(color=NAVY),
          "SIGNIFICANT_BASELINE_BETTER": dict(color=ORANGE),
          "NOT_SIGNIFICANT": dict(facecolor="white", edgecolor=GRAY, hatch="//////", linewidth=.6)}

# All positions are in inches at the printed width used by Word (411.05 pt).
width, height = 5.71, 5.48
fig = plt.figure(figsize=(width, height), dpi=600)
panels = [
    ("return", .035, 2.79, "Getiri görevi", [-2, 0, 2, 4, 6], (-2.2, 6.2)),
    ("volatility", 2.918, 5.636, "Volatilite görevi", [-4, -2, 0, 2], (-4.7, 2.7)),
]
plotted = set()
for task, left, value_right, title, ticks, limits in panels:
    axis_left, axis_width = left + .66, 1.55
    ax = fig.add_axes([axis_left / width, .80 / height, axis_width / width, 28 * .157 / height])
    trans = blended_transform_factory(fig.dpi_scale_trans, ax.transData)
    ax.set_xlim(*limits)
    ax.set_ylim(27.5, -.5)
    line = 0
    for key, acronym in groups[task]:
        ax.text(left, line, acronym, transform=trans, ha="left", va="center", zorder=6,
                fontweight="bold", color=NAVY, bbox={"fc":"white", "ec":"none", "pad":.4})
        line += 1
        for asset, label in assets.items():
            index = (task, key, asset)
            if index not in observations:
                continue
            record = observations[index]
            plotted.add(index)
            val = float(record["mean_loss_differential"]) * 1000
            decision = record["interpretation"]
            is_ns = decision == "NOT_SIGNIFICANT"
            assert (record["null_rejected_holm"] == "False") == is_ns
            assert (float(record["holm_adjusted_p"]) >= .05) == is_ns
            if not is_ns:
                assert (val > 0) == (decision == "SIGNIFICANT_FINAL_BETTER")
            ax.barh(line, val, height=.62, zorder=2, **styles[decision])
            ax.text(axis_left - .05, line, label, transform=trans, ha="right", va="center", color=NAVY)
            value = f"{val:+.3f}".replace(".", ",").replace("-", "−")
            ax.text(value_right, line, value, transform=trans, ha="right", va="center",
                    color=GRAY if is_ns else NAVY, style="italic" if is_ns else "normal")
            line += 1
    ax.axvline(0, color=NAVY, lw=.8, zorder=3)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}".replace("-", "−")))
    ax.tick_params(axis="x", labelsize=10, colors=NAVY, width=.6, length=2.5, pad=2)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(NAVY)
    ax.spines["bottom"].set_linewidth(.6)
    ax.grid(axis="x", color="#E6E9EE", lw=.5)
    ax.set_axisbelow(True)
    fig.text((left + value_right) / (2 * width), (height - .12) / height, title,
             ha="center", va="center", fontweight="bold", color=NAVY)
assert plotted == set(observations)
assert Counter(r["interpretation"] for r in rows) == {
    "SIGNIFICANT_FINAL_BETTER": 5, "SIGNIFICANT_BASELINE_BETTER": 34, "NOT_SIGNIFICANT": 3}
fig.text(.5, .45 / height, "ΔL × 10³ = (kıyaslama kaybı − nihai model kaybı) × 10³",
         ha="center", va="center", color=NAVY)
handles = [Patch(**style) for style in styles.values()]
legend = fig.legend(handles, ["Nihai model anlamlı üstün", "Kıyaslama modeli anlamlı üstün", "Anlamlı fark yok"],
           loc="center", bbox_to_anchor=(.5, .17 / height), ncol=3, frameon=False, handlelength=1.4,
           handletextpad=.4, columnspacing=1.0, labelcolor=NAVY)
args.output.mkdir(parents=True, exist_ok=True)
hedef = args.output / "Sekil_4_3.png"
fig.savefig(hedef, dpi=600)
plt.close(fig)
print(hedef)
