# Multi-Asset Return and Volatility Forecasting

Reproducibility materials for the master's thesis:

> **Multi-Task Transformer-Based Architecture for Multi-Asset Financial Risk and Return Forecasting**

The study evaluates return and volatility forecasts for four financial assets under one chronological, leakage-aware experimental protocol.

## Research scope

| Component | Design |
|---|---|
| Assets | BIST 100, USD/TRY, EUR/TRY, Gold |
| Raw period | 2010-01-04 to 2024-12-31 |
| Return target | Next-period log return |
| Volatility target | Next-period 20-day annualized realized volatility |
| Primary return loss | Mean squared error |
| Primary volatility loss | Pinball loss, τ = 0.5 |
| Split | Chronological and target-realization-aware |
| Test observations | 584 |
| Final ensemble seeds | 123, 777, 2026 |

## Experimental design

The locked v4 experiment searched 480 configurations:

- four architecture families: `FullSharingMTL`, `PartialSharingMTL`, `HierarchicalMTL`, and `NoSharing`;
- five loss strategies: fixed λ values of 0.3, 0.5, and 0.7, uncertainty weighting, and PCGrad;
- lookback windows of 10, 20, 30, and 60;
- three model sizes;
- baseline and full feature sets.

The final three-seed configuration was:

```text
NoSharing + FixedLambda_0.7 + lookback 10 + small + baseline features
```

`NoSharing` has separate task-specific parameter paths. It is retained as a multi-output experimental condition, but it does not provide evidence of shared representation learning or cross-task knowledge transfer.

## What the results show

The final model was compared with naive, learned, and econometric baselines using loss differentials and Holm-adjusted DM-HLN tests.

| Task | Comparisons | Final model better | Comparator better | No significant difference |
|---|---:|---:|---:|---:|
| Return | 20 | 3 | 16 | 1 |
| Volatility | 22 | 2 | 18 | 2 |
| **Total** | **42** | **5** | **34** | **3** |

“No significant difference” is not interpreted as model equivalence. These findings do not support a general superiority claim for the final Transformer model. The second thesis hypothesis was not directly tested because the selected `NoSharing` architecture contains no shared task representation.

SHAP was used only as a post-hoc description of the locked ensemble. It was not used for feature selection, model reselection, or causal inference.

## Repository structure

```text
.
├── config/       # schema and checksum inventories
├── data/          # exact frozen raw dataset and data notes
├── docs/         # result and reproducibility notes
├── notebooks/    # output-cleared research notebook
├── protocols/    # locked Stage 8C, 9, and 10 protocol records
├── results/      # compact, publishable SHAP summaries
└── scripts/      # verified v4 pipeline scripts
```

## Reproducibility boundary

The Python scripts are preserved as executed in Google Colab and therefore use:

```text
/content/drive/MyDrive/tez_transformer_v4_repro
```

They have not been refactored after validation because that would change the audited source hashes. To run the archived pipeline as written:

1. Use Python 3.12 in Google Colab.
2. Place the project at the Drive path above.
3. Install the dependencies in `requirements.txt`.
4. Use the included frozen raw file at [`data/raw/raw_prices.csv`](data/raw/raw_prices.csv).
5. Verify its SHA-256 before executing the pipeline; see [`data/README.md`](data/README.md).
6. Run the scripts in the order documented in [`scripts/README.md`](scripts/README.md).

The exact frozen raw dataset is included for reproducibility. Derived datasets, scalers, model checkpoints, and large prediction arrays remain intentionally excluded; their expected identities are recorded through checksums and protocol locks.

## Verified environment records

The archived artifacts record:

- Python 3.12.13
- NVIDIA Tesla T4
- PyTorch 2.11.0+cu128
- NumPy 2.0.2
- pandas 2.2.2
- SciPy 1.16.3
- XGBoost 3.3.0
- `arch` 8.0.0
- SHAP 0.52.0

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the boundary between exact recorded versions and dependencies whose exact runtime version was not preserved.

## Integrity

The frozen raw file expected by the v4 pipeline has SHA-256:

```text
ab5f275d38dc98057b1cedcf58019adb26be7402c7ed5ae6ee3d6877b2444893
```

Protocol files and companion checksum files are included under `protocols/`. The code and protocol records are research artifacts; inclusion in this repository does not by itself guarantee bitwise reproduction on different hardware or library builds.

## Status

The thesis and repository are under final documentation review. Scientific claims may be clarified as the thesis text is finalized, but locked experimental outputs are not retroactively changed.
