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

These versions describe only the later Stage 09D robustness rerun; they do not revise the locked main-pipeline environment above. Use `requirements-bartlett.txt` for this route. In particular, the recorded Stage 09D statsmodels version is `0.14.5`, whereas the main-pipeline installation record contains `0.14.6`.

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

The recovered main notebook contains the previously unpublished main Stage 09 and Stage 10 computation cells. Its cell source is retained; outputs, execution counters, and account-specific execution metadata are removed. This cleaning changes the notebook file hash, not the source code in its original code cells.

The main Stage 09 calculation cell is additionally extracted verbatim to `scripts/09_dm_harvey_holm_v4.py`. Its separate adapter, `scripts/reproduce_archived_analysis_v4.py`, changes execution paths for an isolated rerun while leaving the archived calculation file unchanged. The adapter is new utility code, not an original executed artifact.

The Stage 09D publication notebook is a corrected operational wrapper. It replaces references to a removed historical folder with the current layout and calls the unchanged standalone Bartlett implementation. Its cleared outputs and corrected wrapper are not represented as byte-identical to the historical executed notebook.

The original `config/code_manifest_v4.csv` is retained unchanged. Its historical scope is the scripts recorded when it was created, not every subsequently recovered artifact. A separate recovery inventory records the added sources and wrapper changes.

## Running the recovered analyses

### Main Stage 09

Install the statistical dependencies and run the adapter from the repository root:

```bash
python -m pip install -r requirements-statistics.txt
python scripts/reproduce_archived_analysis_v4.py stage09 \
  --project-root /path/to/frozen-project \
  --output-dir /path/to/new-stage09-run
```

Provide the private frozen project containing the loss arrays, GARCH diagnostics, and other files identified by the Stage 09 protocol's 17-entry locked-input inventory. The adapter checks all 17 inputs before copying them into a new directory outside that project; it then changes only the archived source's `ROOT` assignment in memory. The frozen protocol is read from the private project's `config/` directory, or from this repository's `protocols/` directory. The public compact CSV is a reference output, not a replacement for the per-observation losses. The archived execution source verifies the copied input identities again and retains its no-overwrite guards.

The September source ZIP contains 16 of the 17 required inputs. It does not include `data/sequences/baseline/lb10/anchor_dates_test.npy`. The original file was recovered separately from Drive; its required SHA-256 is `40194047c9b7fe62adc0966f2c144e1be1e9c85f832d96ffe0a12624d34aab63`. Restore that exact file in the private input project before using the ZIP as the basis for a Stage 09 rerun. No replacement file was fabricated.

This route recomputes 42 statistical comparisons from existing predictions and losses. It does not retrain the forecasting models. Compare numerical results and decisions with `results/statistical_comparison/09_dm_harvey_holm_results_v4.csv`. Timestamps and execution paths in newly created metadata can differ even when the numerical results agree.

### Bartlett-HAC robustness

Use the separate dependencies and inspect the unchanged standalone implementation's interface:

```bash
python -m pip install -r requirements-bartlett.txt
python scripts/09D_bartlett_dm_hln_robustness_v1.py --help
```

The corrected Bartlett notebook provides the Colab wrapper around this implementation. It requires the frozen input project and the additional files named by the wrapper. Large input arrays and ZIP packages remain excluded from the public repository. Use a fresh output directory and compare all 22 results against the frozen compact outputs. The recorded independent reproduction used a numerical tolerance of `1e-10`.

### Stage 10 SHAP

The recovered main notebook is the source for model preparation, protocol and addendum checks, readiness checks, smoke test, full chunk computation, assembly, independent final audit, and closure. Preparation and smoke-test cells establish globals needed by later cells. Running only the full-computation cell in an empty session is insufficient.

Use an isolated copy with the exact frozen processed arrays, fitted scalers, and three selected checkpoints. Protocol and output creation cells deliberately stop when locked files already exist. Audit-only cells can inspect completed artifacts; full recomputation requires a separate working copy with its required inputs and appropriately empty output locations. The source-preserving archive update does not itself constitute a new training or GPU SHAP run.

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
