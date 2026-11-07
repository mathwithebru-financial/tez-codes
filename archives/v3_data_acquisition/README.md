# Historical v3 data acquisition sources

Documentation added on 7 November 2026. This directory preserves selected original v3 files recovered from `tez_v3_code_audit_bundle.zip`. It documents the origin of the exact raw dataset later copied into the locked v4 thesis pipeline.

## Preserved original files

| File | Purpose |
|---|---|
| [`scripts/01_veri_toplama_v3.py`](scripts/01_veri_toplama_v3.py) | Original Yahoo Finance acquisition and v3 preparation source. |
| [`config/schema_v3.json`](config/schema_v3.json) | Original asset identifiers, dates, price-column priority, and historical v3 configuration. |
| [`config/frozen_data_sha256_v3.csv`](config/frozen_data_sha256_v3.csv) | Original raw and processed dataset checksum inventory. |
| [`data/processed/meta_v3.json`](data/processed/meta_v3.json) | Original v3 production metadata, including the 3,912-by-4 raw shape. |
| [`config/audit_manifest_v3.txt`](config/audit_manifest_v3.txt) | Original audit manifest covering 24 files in the source bundle. |

These five files are copied byte for byte. The original manifest covers the full source bundle, not only the selected files published here. All 24 listed entries in the recovered bundle matched their recorded SHA-256 values. The new [`verification.json`](verification.json) separately records the source ZIP identity, selected-file hashes, and the checks performed for this addition; it is not a historical execution artifact.

Source archive SHA-256:

```text
b2aa8674df37e45320288d84600324ed47488345a54ec9fc514823efe9614837
```

## Acquisition settings preserved in the source

The source calls:

```python
df = yf.download(
    ticker,
    start=start_date,
    end=end_date_inclusive,
    auto_adjust=False,
    progress=False
)
```

The schema requests 1 January 2010 through 31 December 2024 for `XU100.IS`, `USDTRY=X`, `EURTRY=X`, and `GC=F`. The code adds one day to the schema end date, so the provider receives `end="2025-01-01"`. It selects `Adj Close` first and falls back to `Close`; `auto_adjust=False` therefore does not mean that the selected series is necessarily the unadjusted `Close` column.

Daily frequency is recorded in the schema. This historical download call does not explicitly pass `interval` or `ignore_tz`, and these selected files do not establish the exact historical yfinance version. They also do not establish market-by-market clock-time availability at a forecasting origin.

## Connection to the v4 thesis dataset

The v3 frozen-data inventory and the existing v4 frozen-raw copy record identify the same raw file:

```text
ab5f275d38dc98057b1cedcf58019adb26be7402c7ed5ae6ee3d6877b2444893
```

The exact frozen CSV has 3,912 date rows, four price columns, and observations from 4 January 2010 through 31 December 2024. The v4 record documents copying `tez_transformer/data/raw/raw_prices.csv` to `tez_transformer_v4_repro/data/raw/raw_prices.csv`. Hash comparison confirms the same bytes.

The `created_at` value in the v3 production metadata is `2026-06-20 14:45:25`; its time zone is not recorded there. This is a preserved production record, not proof of the first-ever Yahoo download, the original chat identity, or use in a particular CMES submission.

## Historical-source boundary

The selected v3 script also contains historical preprocessing and writes to its original Colab root, `/content/drive/MyDrive/tez_transformer`. It is preserved for inspection and must not be treated as the current v4 entry point. If a fresh acquisition is needed, use an isolated working copy and a separate output directory; a fresh vendor download need not reproduce the frozen snapshot.

The historical v3 code uses `ffill().bfill()`, while the current v4 preparation uses forward filling only. A read-only check on the exact frozen raw CSV found that backfilling after forward filling would add zero values. Historical v3 feature counts, targets, splits, and results remain distinct from the final v4 experiment.

This documentation addition did not execute the historical script, download new market data, retrain a model, replace the frozen dataset, or alter v4 protocol locks or experimental outputs.
