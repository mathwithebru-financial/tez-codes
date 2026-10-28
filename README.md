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
| Volatility target | Next-period 20-day annualized historical volatility |
| Return training loss | Mean squared error (MSE) |
| Primary return evaluation and statistical comparison loss | Mean absolute error (MAE) |
| Volatility training, primary evaluation, and statistical comparison loss | Pinball loss, τ = 0.5 |
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

### Post-hoc volatility robustness check

The 22 volatility comparisons were also evaluated in a separate, post-hoc robustness check using a Bartlett-HAC long-run variance estimate with `h=1`, `L=19`, weights `1-k/20`, and Holm correction across all 22 tests.

| Analysis | Final model better | Comparator better | No significant difference |
|---|---:|---:|---:|
| Main Stage 09 | 2 | 18 | 2 |
| Bartlett-HAC robustness | 2 | 12 | 8 |

Six decisions changed from “comparator better” to “no significant difference”; no superiority direction reversed. The two significant final-model wins were preserved. This sensitivity check does not replace the locked Stage 09 analysis. See [`results/robustness/bartlett_dm_hln_v1/`](results/robustness/bartlett_dm_hln_v1/) for the validated compact outputs.

SHAP was used only as a post-hoc description of the locked ensemble. It was not used for feature selection, model reselection, or causal inference.

The final SHAP closure record is `CLOSED` with independent audit status `PASS`. The earlier SHAP report is retained unchanged, including its then-pending audit status; the later closure record documents completion. See [`results/shap/`](results/shap/).

## Repository structure

```text
.
├── config/       # schema and checksum inventories
├── data/          # exact frozen raw dataset and data notes
├── docs/         # result and reproducibility notes
├── notebooks/    # output-cleared research and robustness notebooks
├── protocols/    # locked protocols and the post-hoc robustness protocol
├── results/      # compact SHAP and robustness summaries
└── scripts/      # archived pipeline sources and isolated rerun adapters
```

## Reproducibility boundary

The original pipeline scripts and the recovered main Stage 09 source retain the Google Colab root used during execution:

```text
/content/drive/MyDrive/tez_transformer_v4_repro
```

Their original calculation code has not been silently refactored. The recovered main notebook now includes the actual Stage 09 DM-HLN/Holm and Stage 10 SHAP computation cells. Its outputs and account-specific execution metadata are cleared for publication.

Use the reproduction route appropriate to the available inputs:

- **Inspect the archive:** read the scripts, notebook, protocols, compact results, and checksum inventories. Do not use **Run all** on the historical main notebook: it includes protocol creation, prior diagnostics, and guards against overwriting locked outputs.
- **Recompute main Stage 09:** use `scripts/reproduce_archived_analysis_v4.py` with an isolated copy of the exact frozen Stage 09 inputs. It adapts execution paths around the recovered source; it does not refit models. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).
- **Recompute Bartlett-HAC robustness:** use the standalone Stage 09D script or the corrected Bartlett notebook, with `requirements-bartlett.txt` and the required archived inputs.
- **Repeat training or SHAP:** provide the required processed data, scalers, checkpoints, and other locked inputs in a separate working copy, following the recorded stage order and environment. The main notebook preserves the SHAP preparation, smoke-test, full computation, assembly, audit, and closure source.

The frozen raw dataset is included. Derived datasets, scalers, checkpoints, and large arrays remain intentionally excluded from the public repository. Their hashes identify the required frozen inputs; they do not supply those bytes. Fresh training is a separate reproduction level and may not recreate the exact checkpoint hashes required by later locked analyses.

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

The recovered analysis sources are inventoried separately in [`config/recovered_analysis_manifest_v4.json`](config/recovered_analysis_manifest_v4.json). The [28 October 2026 verification record](docs/validation/archive_recovery_checks_20261028.json) records isolated reruns of all 42 main Stage 09 tests and all 22 Bartlett tests, plus source and SHAP-output identity checks. It does not claim a new training or GPU SHAP run.

## Status

The thesis and repository are under final documentation review. Scientific claims may be clarified as the thesis text is finalized, but locked experimental outputs are not retroactively changed.
