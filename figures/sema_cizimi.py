"""Pillow drawing primitives shared by the four thesis schematics."""
from functools import lru_cache
import math

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont


INK = "#24364b"
BLUE = "#e3eef9"
ORANGE = "#fce8ce"
GREEN = "#e0f1e8"
GREY = "#f2f4f7"


@lru_cache(maxsize=None)
def _font(pixel_size, bold):
    style = font_manager.FontProperties(
        family="Times New Roman", weight="bold" if bold else "normal"
    )
    path = font_manager.findfont(style, fallback_to_default=False)
    return ImageFont.truetype(path, pixel_size)


class Canvas:
    """Draw in layout coordinates at the original print scale."""

    def __init__(self, width, height, min_font_size=24.35):
        self.scale = 3
        self.min_font_size = min_font_size
        self.image = Image.new("RGB", (width * self.scale, height * self.scale), "white")
        self.pen = ImageDraw.Draw(self.image)

    def _pixels(self, coords):
        return tuple(value * self.scale for value in coords)

    def box(self, x, y, w, h, label="", fill="white", fs=24, bold=False, stroke=INK, r=8):
        self.pen.rounded_rectangle(
            self._pixels((x, y, x + w, y + h)),
            radius=r * self.scale, fill=fill, outline=stroke, width=2 * self.scale,
        )
        if label:
            self.text(x + w / 2, y + h / 2, label, fs, bold)

    def text(self, x, y, label, fs=24, bold=False, color=INK, anchor="middle"):
        size = max(fs, self.min_font_size)
        font = _font(int(size * self.scale), bold)
        lines = label.split("\n")
        spacing = size * 1.15
        start = y - (len(lines) - 1) * spacing / 2
        alignment = {"middle": "mm", "start": "lm", "end": "rm"}[anchor]
        for i, line in enumerate(lines):
            point = self._pixels((x, start + i * spacing))
            self.pen.text(point, line, font=font, fill=color, anchor=alignment)

    def line(self, pts, color=INK, width=2.2, arrow=False, dash=False):
        stroke = round(width * self.scale)
        if dash:
            for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
                n = max(1, int(math.hypot(x2 - x1, y2 - y1) / 11))
                for i in range(n):
                    begin = (x1 + (x2 - x1) * i / n, y1 + (y2 - y1) * i / n)
                    end_fraction = (i + 0.55) / n
                    end = (x1 + (x2 - x1) * end_fraction, y1 + (y2 - y1) * end_fraction)
                    self.pen.line([self._pixels(begin), self._pixels(end)], fill=color, width=stroke)
        else:
            self.pen.line([self._pixels(p) for p in pts], fill=color, width=stroke, joint="curve")

        if arrow:
            (x1, y1), (x2, y2) = pts[-2:]
            angle = math.atan2(y2 - y1, x2 - x1)
            dx, dy = math.cos(angle), math.sin(angle)
            triangle = [(x2, y2), (x2 - 10 * dx + 5 * dy, y2 - 10 * dy - 5 * dx),
                        (x2 - 10 * dx - 5 * dy, y2 - 10 * dy + 5 * dx)]
            self.pen.polygon([self._pixels(p) for p in triangle], fill=color)

    def dot(self, x, y, r=3):
        self.pen.ellipse(self._pixels((x - r, y - r, x + r, y + r)), fill=INK)

    def frame(self, x, y, w, h):
        points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
        self.line(points, color="#7c8a99", width=1.7, dash=True)

    def save(self, output, number):
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"Sekil_{number}.png"
        self.image.save(path, dpi=(600, 600))
        print(path)
