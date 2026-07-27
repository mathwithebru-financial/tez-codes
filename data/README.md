# Frozen thesis dataset

This directory contains the exact raw price snapshot used by the locked v4 thesis pipeline.

## File

- Path: `data/raw/raw_prices.csv`
- Shape: 3,912 rows × 5 columns (date plus four price series)
- Observed date range: 2010-01-04 to 2024-12-31
- SHA-256: `ab5f275d38dc98057b1cedcf58019adb26be7402c7ed5ae6ee3d6877b2444893`

## Columns and source identifiers

| Column | Market series | Source ticker identifier |
|---|---|---|
| `BIST100` | BIST 100 index | `XU100.IS` |
| `USDTRY` | USD/TRY exchange rate | `USDTRY=X` |
| `EURTRY` | EUR/TRY exchange rate | `EURTRY=X` |
| `GOLD` | Gold futures | `GC=F` |

The CSV is a frozen research snapshot. It is supplied so the archived pipeline can be reproduced without silently replacing the thesis data with a later vendor revision.

## Integrity check

From the repository root:

```bash
python - <<'PY'
from pathlib import Path
import hashlib

path = Path("data/raw/raw_prices.csv")
print(hashlib.sha256(path.read_bytes()).hexdigest())
PY
```

The output must equal the SHA-256 value above. Script `scripts/01_rebuild_from_frozen_raw_v4.py` performs this check before processing.

## Use and limitations

- Missing observations are retained in this raw union-calendar snapshot.
- The pipeline applies forward filling only; it does not backfill.
- This file contains market observations, not investment advice.
- Users remain responsible for complying with the original data provider's terms when redistributing or using the market data.
- Cite the thesis/repository and the original market-data source where appropriate.
