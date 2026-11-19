"""Baskı tezindeki çizelgelerin hücrelerini özgün arşiv kayıtlarından CSV'ye aktarır."""
import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import appendix
import early
import results


TABLES = {**early.TABLES, **results.TABLES, **appendix.TABLES}
P1_TABLES = {"4.12", "E.14", "E.15", "E.16", "E.17"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0", type=Path, help="Açılan P0/H2 arşivinde main_project klasörünün üst dizini")
    parser.add_argument("--p1", type=Path, help="Açılan P1 arşivinde dogrula.py dosyasının bulunduğu dizin")
    parser.add_argument("--table", choices=list(TABLES), help="Yalnız bu numaralı çizelgeyi üret")
    parser.add_argument("--output", required=True, type=Path, help="CSV dosyalarının yazılacağı dizin")
    args = parser.parse_args()
    if args.p0 is None and args.p1 is None:
        parser.error("En az bir arşiv belirtin: --p0 veya --p1.")
    if args.table:
        required = "--p1" if args.table in P1_TABLES else "--p0"
        if getattr(args, required[2:]) is None:
            parser.error(f"Çizelge {args.table} için {required} gereklidir.")

    args.output.mkdir(parents=True, exist_ok=True)
    completed = []
    # Build the selected tables before replacing any previous CSV output.
    with TemporaryDirectory(dir=args.output) as temporary:
        for number, build in TABLES.items():
            archive = args.p1 if number in P1_TABLES else args.p0
            if archive is None or (args.table and number != args.table):
                continue
            try:
                rows = build(archive)
            except FileNotFoundError as exc:
                parser.exit(1, f"Çizelge {number} için arşiv girdisi bulunamadı: {exc.filename}\n")
            except ModuleNotFoundError as exc:
                if exc.name == "numpy":
                    parser.exit(1, "Çizelge E.12 için NumPy gerekir: python -m pip install numpy\n")
                raise
            if len(rows) < 2 or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
                raise ValueError(f"Çizelge {number} için satır veya sütun sayısı geçersiz.")
            path = Path(temporary) / f"Cizelge_{number.replace('.', '_')}.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                csv.writer(stream, delimiter=";").writerows(rows)
            completed.append(path)
        for path in completed:
            path.replace(args.output / path.name)
    print(f"{len(completed)} çizelge yazıldı: {args.output}")


if __name__ == "__main__":
    main()
