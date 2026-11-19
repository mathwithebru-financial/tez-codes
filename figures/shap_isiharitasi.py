"""Tezdeki SHAP ısı haritalarını kayıtlı önem paylarından çizer."""

import csv
from functools import lru_cache
from math import isclose

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from ortak import FONT


SYMBOLS = ("BIST100", "USD/TRY", "EUR/TRY", "Altın")
WIDTH, HEIGHT = 2400, 1740
LEFT, TOP, ROW_HEIGHT = 553, 265, 153


@lru_cache(maxsize=None)
def font(size=59, bold=False):
    weight = "bold" if bold else "normal"
    properties = font_manager.FontProperties(family=FONT, weight=weight)
    path = font_manager.findfont(properties, fallback_to_default=False)
    return ImageFont.truetype(path, size)


def read_shares(project, importance, axis_name, axis_size):
    filename = f"10_shap_{importance}_importance_v4.csv"
    results = project / "results"
    # The archive and the public repository store the same Stage 10 tables.
    csv_path = results / "shap_analysis" / filename
    if not csv_path.is_file():
        csv_path = results / "shap" / filename

    shares = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            output = int(row["output_index"])
            axis = int(row[axis_name])
            value = float(row["normalized_share_within_output"])
            if not (0 <= output < 8 and 0 <= axis < axis_size and 0 <= value <= 1):
                raise ValueError(f"Geçersiz SHAP hücresi: {output}, {axis}, {value}")
            if (output, axis) in shares:
                raise ValueError(f"Tekrarlanan SHAP hücresi: {output}, {axis}")
            shares[output, axis] = value

    if len(shares) != 8 * axis_size:
        raise ValueError(f"Eksik SHAP hücreleri: {csv_path}")
    for output in range(8):
        total = sum(shares[output, axis] for axis in range(axis_size))
        if not isclose(total, 1.0, abs_tol=1e-7):
            raise ValueError(f"SHAP paylarının toplamı birden farklı: {output}, {total}")
    return shares


def draw_heatmap(output_path, headings, labels, matrix, group_titles, notes, column_width):
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    pen = ImageDraw.Draw(image)

    def put(x, y, text, text_font, area, fill="black", anchor="mm"):
        left, top, right, bottom = area
        x0, y0, x1, y1 = pen.textbbox((x, y), text, font=text_font, anchor=anchor)
        if not (left <= x0 and top <= y0 and x1 <= right and y1 <= bottom):
            raise ValueError(f"Şekil alanına sığmayan metin: {text}")
        pen.text((x, y), text, font=text_font, fill=fill, anchor=anchor)

    for x, y, title, area in group_titles:
        put(x, y, title, font(60, True), area)
    for col, title in enumerate(headings):
        x = LEFT + col * column_width
        center = x + column_width / 2
        if "/" in title:
            first, second = title.split("/")
            put(center, 165, first + "/", font(59, True), (x, 121, x + column_width, 200))
            put(center, 224, second, font(59, True), (x, 194, x + column_width, TOP - 2))
        else:
            put(center, 194, title, font(59, True), (x, 121, x + column_width, TOP - 5))

    for row, label in enumerate(labels):
        y = TOP + row * ROW_HEIGHT
        cy = y + ROW_HEIGHT / 2
        put(LEFT - 18, cy, label, font(), (5, y, LEFT - 2, y + ROW_HEIGHT), anchor="rm")
        for col, value in enumerate(matrix[row]):
            x = LEFT + col * column_width
            light, dark = (241, 246, 250), (20, 67, 103)
            shade = tuple(round(a + value * (b - a)) for a, b in zip(light, dark))
            pen.rectangle((x, y, x + column_width, y + ROW_HEIGHT), fill=shade,
                          outline="#cfd5db", width=3)
            text = f"{value * 100:.1f}".replace(".", ",")
            area = (x + 2, y + 2, x + column_width - 2, y + ROW_HEIGHT - 2)
            put(x + column_width / 2, cy, text, font(61, True), area,
                "white" if value > 0.48 else "black")

    for i, line in enumerate(notes):
        put(40, 1553 + i * 77, line, font(), (32, 1500, 2375, 1700), anchor="lm")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, dpi=(400, 400))
    print(output_path)
