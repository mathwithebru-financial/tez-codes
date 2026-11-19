"""Tezin beş baskı şeklini aynı çıktı klasöründe üretir."""
from pathlib import Path
import subprocess
import sys

from ortak import ayarlar


def main():
    args = ayarlar(__doc__)
    kaynak = Path(__file__).resolve().parent
    for numara in ("2_3", "2_4", "3_2", "4_3", "5_1"):
        subprocess.run(
            [sys.executable, str(kaynak / f"sekil_{numara}.py"),
             "--archive", str(args.archive), "--output", str(args.output)],
            check=True,
        )


if __name__ == "__main__":
    main()
