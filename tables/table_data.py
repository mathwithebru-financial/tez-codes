"""Frozen CSV inputs and numeric formatting shared by the thesis tables."""
import csv


SUPERSCRIPTS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def records(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value, places=6):
    return f"{float(value):.{places}f}".replace(".", ",")


def scientific(value):
    mantissa, exponent = f"{float(value):.3e}".split("e")
    power = str(int(exponent)).translate(SUPERSCRIPTS)
    return f"{mantissa.replace('.', ',')} × 10{power}"
