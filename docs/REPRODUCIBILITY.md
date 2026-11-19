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

`requirements.txt` specifies a compatible range for scikit-learn. The fitted preprocessing scalers record version `1.6.1` in `_sklearn_version`. The [SHAP protocol addendum of 22 July 2026](../protocols/10_shap_protocol_addendum_v4a.json) independently records `1.6.1` in `runtime_at_addendum_lock.sklearn`. Individual training sessions have no separate scikit-learn version record.

Matplotlib is required by the archived plotting and SHAP report source. Its exact historical runtime version was not preserved, so the dependency is not presented as an exact recorded pin.

### Stage 09D post-hoc rerun environment

The independent Colab reproduction record for the Bartlett-HAC robustness check contains:

| Component | Recorded value |
|---|---|
| Python | 3.13.15 |
| NumPy | 2.1.3 |
| pandas | 2.2.3 |
| SciPy | 1.16.3 |
| statsmodels | 0.14.5 |

`requirements-bartlett.txt` records the environment for the later Stage 09D rerun. That environment used statsmodels `0.14.5`. The main-pipeline installation record lists `0.14.6`.

## Data integrity

The pipeline does not re-download market data. It expects the frozen raw file:

```text
data/raw/raw_prices.csv
```

Expected SHA-256:

```text
ab5f275d38dc98057b1cedcf58019adb26be7402c7ed5ae6ee3d6877b2444893
```

The exact frozen raw dataset is included at `data/raw/raw_prices.csv` for reproducibility. Its SHA-256 matches the locked value above. Derived datasets are not included. Checksums, schemas, and protocol locks make the expected inputs auditable.

### Recovered acquisition source

The [historical v3 acquisition archive](../archives/v3_data_acquisition/README.md) contains the download code, production metadata and checksums. A later verification record links the v3 raw file to the v4 snapshot used for preparation. The time of the first download and the time each daily price first became available from the provider remain unknown.

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

The archived scripts retain this path because it forms part of the file content covered by the recorded checksums.

The recovered main notebook contains the previously unpublished main Stage 09 and Stage 10 computation cells. Its cell source is retained; outputs, execution counters, and account-specific execution metadata are removed. This cleaning changes the notebook file hash, not the source code in its original code cells.

The main Stage 09 calculation cell is additionally extracted verbatim to `scripts/09_dm_harvey_holm_v4.py`. Its separate adapter, `scripts/reproduce_archived_analysis_v4.py`, changes execution paths for an isolated rerun while leaving the archived calculation file unchanged. The adapter is new utility code, not an original executed artifact.

The Stage 09D publication notebook calls the archived Bartlett implementation. Its wrapper uses the current directory layout. The path correction and removal of saved outputs changed the notebook’s file hash.

The original `config/code_manifest_v4.csv` is retained unchanged. Its historical scope is the scripts recorded when it was created, not every subsequently recovered artifact. A separate recovery inventory records the added sources and wrapper changes.

## Running the recovered analyses

### Main Stage 09

The [12 November P0/H2 archive](TEZ_ARSIVLERI.md) contains all 17 inputs and the matching source tree. The following commands run **from the archive root**, the extracted directory containing `PACKAGE_MANIFEST.json`:

```bash
python tools/verify_archive.py --archive-root .
python -m pip install -r sources/main/requirements-bartlett.txt
python sources/main/scripts/reproduce_archived_analysis_v4.py stage09 --project-root main_project --output-dir ../new-stage09-run
```

The adapter checks the input hashes, including the loss arrays, GARCH diagnostics and test dates, and verifies them again after copying the inputs to the new output directory. It rebinds the historical `ROOT` assignment in memory for this run. The expected protocols and diagnostic records are included in the archive.

This combined replay route was checked with Python 3.13.3, NumPy 2.1.3, pandas 2.2.3, SciPy 1.16.3 and statsmodels 0.14.5. The recorded training environment is listed separately above.

For provenance, the September source ZIP held 16 of the 17 inputs. The recovered `data/sequences/baseline/lb10/anchor_dates_test.npy` (SHA-256 `40194047c9b7fe62adc0966f2c144e1be1e9c85f832d96ffe0a12624d34aab63`) is supplied inside `main_project/` in the November archive.

This replay uses existing predictions and losses to recompute 42 statistical comparisons. The reference values and decisions are in `results/statistical_comparison/09_dm_harvey_holm_results_v4.csv`. Timestamps and paths can differ between runs with the same numerical results.

### Bartlett-HAC robustness

The following commands install the Bartlett dependencies and show the arguments accepted by the archived calculation script:

```bash
python -m pip install -r requirements-bartlett.txt
python scripts/09D_bartlett_dm_hln_robustness_v1.py --help
```

The corrected Bartlett notebook runs this implementation in Colab. It requires the archived project inputs and the additional files listed in the notebook. The large arrays and ZIP packages are distributed separately from the public repository. The reproduction uses a fresh output directory. The independent check compared all 22 results with the archived values at a numerical tolerance of `1e-10`.

### Stage 10 SHAP

The recovered main notebook is the source for model preparation, protocol and addendum checks, readiness checks, smoke test, full chunk computation, assembly, independent final audit, and closure. Preparation and smoke-test cells establish globals needed by later cells. Running only the full-computation cell in an empty session is insufficient.

A full SHAP rerun starts in a separate working directory with the archived processed arrays, fitted scalers and three selected checkpoints. Protocol and output creation cells stop when they encounter an existing locked file. The audit cells can read completed artifacts in place. The archive recovery record covers source verification. It does not record a new training or GPU SHAP run.

The original SHAP report records the audit as pending at the time of report creation. The later closure JSON and report record `CLOSED` and audit `PASS`; both stages of that record are preserved without rewriting the original report.

## Excluded artifacts

The following are intentionally excluded:

- processed market data;
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
2. **Recompute frozen analyses:** provide the exact locked intermediate inputs and rerun Stage 09, Stage 09D, or SHAP in isolated working copies. Published hashes identify required inputs but do not provide excluded files.
3. **Repeat training:** start from the included raw dataset and follow the recorded preprocessing, selection, training, and evaluation workflow. Fresh training may produce different checkpoint bytes; those outputs cannot silently replace identities in the original protocol locks.
4. **Bitwise verification:** compare produced artifact hashes with the locked inventory. Hardware and dependency build differences may prevent exact equality even when numerical conclusions agree.
