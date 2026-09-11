# Reproducibility Notes

## Recorded execution environment

The following versions are explicitly recorded in locked protocol or diagnostic artifacts:

| Component | Recorded value | Evidence stage |
|---|---|---|
| Python | 3.12.13 | Stage 8C |
| GPU | NVIDIA Tesla T4 | training notebook / Stage 10 |
| CUDA availability | true | Stage 10 diagnostics |
| PyTorch | 2.11.0+cu128 | Stage 10 diagnostics |
| NumPy | 2.0.2 | Stage 8C and Stage 10 |
| pandas | 2.2.2 | Stage 8C |
| SciPy | 1.16.3 | Stage 8C |
| XGBoost | 3.3.0 | Stage 8B |
| `arch` | 8.0.0 | Stage 8C |
| SHAP | 0.52.0 | Stage 10 diagnostics |
| statsmodels | 0.14.6 | Stage 8C installation record |

The exact scikit-learn runtime version was not preserved in the verified artifacts. `requirements.txt` therefore records a compatible major-version range instead of inventing an exact pin.

### Stage 09D post-hoc rerun environment

The independent Colab reproduction record for the Bartlett-HAC robustness check contains:

| Component | Recorded value |
|---|---|
| Python | 3.13.15 |
| NumPy | 2.1.3 |
| pandas | 2.2.3 |
| SciPy | 1.16.3 |
| statsmodels | 0.14.5 |

These versions describe only the later Stage 09D robustness rerun; they do not revise the locked main-pipeline environment above. The Stage 09D notebook installs and records the statsmodels version used for that check.

## Data integrity

The pipeline does not re-download market data. It expects the frozen raw file:

```text
data/raw/raw_prices.csv
```

Expected SHA-256:

```text
ab5f275d38dc98057b1cedcf58019adb26be7402c7ed5ae6ee3d6877b2444893
```

The raw and derived datasets are not included in the public package. Checksums, schemas, and protocol locks are included to make the expected inputs auditable.

## Leakage controls

- Splits are chronological.
- Split membership is based on target realization dates.
- Input history may cross backward into an earlier split; targets may not cross forward.
- `StandardScaler` is fitted only on the training set.
- Validation data is used for configuration selection.
- Test data is not used for hyperparameter selection, checkpoint selection, or early stopping.

## Source preservation

The archived scripts contain the Google Colab Drive root used during execution:

```text
/content/drive/MyDrive/tez_transformer_v4_repro
```

Changing these paths would alter the source hashes recorded during the experiment. The public copy therefore preserves the executed source and documents the path requirement rather than presenting a modified file as the audited original.

The public notebook has its cell outputs and execution counters removed. This cleaning changes the notebook file hash, but not the source code contained in its cells. The separately stored `.py` scripts and protocol records remain the primary source artifacts.

The Stage 09D publication notebook was cleaned in the same way: source cells were retained, while outputs, execution counters, and account-specific Colab metadata were removed. Its standalone script and frozen protocol are included separately.

## Excluded artifacts

The following are intentionally excluded:

- raw and processed market data;
- fitted scalers;
- model checkpoints and `.pt` files;
- large `.npy` and `.npz` arrays;
- temporary SHAP chunks;
- runtime logs and notebook output.
- detailed Bartlett loss-differential series and numerical archives;
- the Bartlett spreadsheet workbook;
- source and audit ZIP packages, whose SHA-256 values are recorded in the robustness result notes.

## Reproduction levels

1. **Structural reproduction:** inspect scripts, schema, protocol rules, and checksums.
2. **Computational rerun:** provide the exact frozen raw file and execute under the recorded Colab path and environment.
3. **Bitwise verification:** compare produced artifact hashes with the locked inventory. Hardware and dependency build differences may prevent exact equality even when numerical conclusions agree.
