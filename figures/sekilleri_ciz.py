"""Tezin 11 baskı şeklini numara sırasıyla aynı çıktı klasöründe üretir."""
from pathlib import Path
import subprocess
import sys

from ortak import ayarlar


def main():
    args = ayarlar(__doc__)
    kaynak = Path(__file__).resolve().parent
    sekiller = (
        "2_1", "2_2", "2_3", "2_4",
        "3_1", "3_2", "3_3",
        "4_1", "4_2", "4_3",
        "5_1",
    )
    for numara in sekiller:
        subprocess.run(
            [sys.executable, str(kaynak / f"sekil_{numara}.py"),
             "--archive", str(args.archive), "--output", str(args.output)],
            check=True,
        )


if __name__ == "__main__":
    main()
