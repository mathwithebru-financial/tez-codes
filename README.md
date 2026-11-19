# Multi-Asset Return and Volatility Forecasting

**[Türkçe kod rehberi](docs/KOD_REHBERI_TR.md)** · [Tez şekilleri](figures/README.md) · [48 çizelgenin betik ve kaynak dizini](tables/README.md) · [Tez arşivleri ve doğrulama komutları](docs/TEZ_ARSIVLERI.md) · [Tamamlayıcı deney ve sonuçları (P1)](experiments/p1_v2/)

**Türkçe uygulamalı dersler.** Ders defterinde, bir fiyat hareketi üzerinden log getiriyi hesaplıyor, ardından modelin o getiriyi tahmin ederken hangi tarihleri görebildiğini gösteriyorum. Örnek hesaplar, kod ve grafikler aynı defterde yer alıyor. [Uygulamalı ders defteri](notebooks/Finansal_Zaman_Serileri_Uygulamali_Dersler.ipynb) · [Colab sürümü](https://colab.research.google.com/github/mathwithebru-financial/tez-codes/blob/main/notebooks/Finansal_Zaman_Serileri_Uygulamali_Dersler.ipynb)

Reproducibility materials for the master's thesis:

> **Multi-Task Transformer-Based Architecture for Multi-Asset Financial Risk and Return Forecasting**

The study evaluates return and volatility forecasts for four financial assets on chronological target-date splits. Historical intraday availability of the provider's daily quotes could not be established. The post-review P1 experiment shifts inputs back one common observation.

The [archive guide](docs/TEZ_ARSIVLERI.md) links each set of thesis tables to its P0/H2 or P1 archive. Each entry includes the ZIP checksum and the command for verifying the extracted files. Appendix C cites commit `40da5a1b8887` for the P1 research code. The portable replay utility and archive instructions were added after that snapshot.

## Research scope

| Component | Design |
|---|---|
| Assets | BIST 100, USD/TRY, EUR/TRY, COMEX gold futures (GC=F) |
| Raw period | 2010-01-04 to 2024-12-31 |
| Return target | Next-period log return |
| Volatility target | Annualized trailing 20-observation historical volatility at the next common observation |
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

## Locked Stage 09 results (L = 0)

The final model was compared with naive, learned, and econometric baselines using loss differentials and Holm-adjusted DM-HLN tests.

| Task | Comparisons | Final model better | Comparator better | No significant difference |
|---|---:|---:|---:|---:|
| Return | 20 | 3 | 16 | 1 |
| Volatility | 22 | 2 | 18 | 2 |
| **Total** | **42** | **5** | **34** | **3** |

“No significant difference” is not interpreted as model equivalence. These findings do not support a general superiority claim for the final Transformer model. The main model selection does not isolate a sharing effect. The [separate controlled H2 comparison](https://github.com/mathwithebru-financial/tez-codes/tree/31f94d42ba2d95c82ac838a9df27dba4d0065b76/experiments/h2_original_period_v1) evaluates full sharing against independent single-task models on the same 584 target dates. Both full-sharing variants had larger losses on both tasks in the four primary Holm-adjusted comparisons.

### Post-hoc volatility robustness check

The 22 volatility comparisons were also evaluated in a separate, post-hoc robustness check using a Bartlett-HAC long-run variance estimate with `h=1`, `L=19`, weights `1-k/20`, and Holm correction across all 22 tests.

| Analysis | Final model better | Comparator better | No significant difference |
|---|---:|---:|---:|
| Main Stage 09 | 2 | 18 | 2 |
| Bartlett-HAC robustness | 2 | 12 | 8 |

Six comparisons that previously favoured the comparator became inconclusive. The two significant final-model wins remained significant, and no comparison switched from favouring one model to favouring the other. The original Stage 09 results and the [Bartlett-HAC results](results/robustness/bartlett_dm_hln_v1/) are reported separately.

For returns, the [post-review HAC tables](results/robustness/return_hac/) report 3 model wins, 10 comparator wins and 7 inconclusive comparisons at L = 10 and L = 19. Each bandwidth uses a separate Holm adjustment across 20 tests.

SHAP was used only as a post-hoc description of the locked ensemble. It was not used for feature selection, model reselection, or causal inference.

The earlier SHAP report predates the independent audit. The [final SHAP record](results/shap/) gives the closure status as `CLOSED` and the audit result as `PASS`.

## Repository structure

```text
.
├── config/       # schema and checksum inventories
├── data/          # exact frozen raw dataset and data notes
├── experiments/p1_v2/  # supplementary lagged-input and loss-sensitivity code
├── docs/         # result and reproducibility notes
├── figures/      # print figures and plotting requirements
├── notebooks/    # applied lessons with outputs and archived research notebooks
├── protocols/    # locked protocols and the post-hoc robustness protocol
├── results/      # compact SHAP and robustness summaries
├── scripts/      # archived pipeline sources and isolated rerun adapters
└── tables/       # print-table exporters and table-by-table source index
```

## Reproducing the thesis figures

The [figure scripts and Turkish instructions](figures/README.md) reproduce all 11 print figures (2.1–2.4, 3.1–3.3, 4.1–4.3 and 5.1) as PNGs. To draw the complete set, extract the **P0/H2** archive (`Tez_Yeniden_Uretim_Arsivi_20261112.zip`) linked in the [archive guide](docs/TEZ_ARSIVLERI.md). It supplies the split records, prices, statistical results and saved predictions.

From the repository root, use a Python 3.12 environment with Times New Roman installed. Replace `../tez_arsivi` with the extracted directory containing `main_project/`, or with `main_project/` itself.

```shell
python -m pip install -r figures/requirements.txt
python figures/sekilleri_ciz.py --archive "../tez_arsivi" --output "../tez_sekilleri"
```

To draw one figure, replace `sekilleri_ciz.py` with its script name, such as `sekil_5_1.py`. For the SHAP heatmaps (4.1 and 4.2), you can use `--archive .` from the repository root. Their input CSVs are included in [results/shap/](results/shap/). The output directory is created automatically.

## Reproducing the thesis tables

The [table-by-table index](tables/README.md) identifies the original calculation, archive inputs and export method for all 48 numbered tables. The following modules assemble 44 tables from the frozen experiment records. Tables 2.2, 3.3, 3.8 and C.1 are author-compiled literature, architecture or source inventories. Their evidence is linked in the index.

| Printed tables | Export module |
|---|---|
| 2.1, 3.1–3.2, 3.4–3.7, A.1, B.1–B.3 | [Methods and statistical appendices](tables/early.py) |
| 4.1–4.12, 5.1 | [Main results](tables/results.py) |
| D.1–D.3, E.1–E.17 | [Supplementary results](tables/appendix.py) |

To export one table, run this from the repository root after extracting the [P0/H2 archive](docs/TEZ_ARSIVLERI.md). The [Turkish guide](tables/README.md) also shows how to export the complete set with both archives.

```shell
python tables/export_tables.py --table 4.2 --p0 "../p0" --output "../tez_cizelgeleri"
```

## Reproducibility boundary

The original pipeline scripts and the recovered main Stage 09 source retain the Google Colab root used during execution:

```text
/content/drive/MyDrive/tez_transformer_v4_repro
```

The recovered main notebook contains the original Stage 09 DM-HLN/Holm and Stage 10 SHAP calculation cells. The publication copy retains their source and omits saved outputs and account-specific execution metadata.

The materials support four uses:

- **Archive review.** The scripts, protocols, compact results and checksums document the original analysis. The historical notebook also contains one-time protocol creation and earlier diagnostics. **Run all** is unsuitable for an existing archive because some cells create files and others stop when those files already exist.
- **Main Stage 09.** With an isolated copy of the frozen Stage 09 inputs, `scripts/reproduce_archived_analysis_v4.py` recalculates the statistical comparisons from saved losses. The [reproduction notes](docs/REPRODUCIBILITY.md) list the inputs and commands.
- **Bartlett-HAC robustness.** The standalone Stage 09D script and the corrected Bartlett notebook use `requirements-bartlett.txt` and archived inputs.
- **Training and SHAP.** Separate working copies follow the recorded stage order and environment. Training requires the processed data and scalers. The archived SHAP computation also requires the selected checkpoints. Its notebook cells cover model preparation, the smoke test, full computation, assembly, audit and closure.

The frozen raw dataset is included. Derived datasets, scalers, checkpoints, and large arrays remain intentionally excluded from the public repository. Their hashes identify the required frozen inputs; they do not supply those bytes. Fresh training is a separate reproduction level and may not recreate the exact checkpoint hashes required by later locked analyses.

## Data acquisition provenance

The recovered [historical v3 acquisition sources](archives/v3_data_acquisition/README.md) preserve the Yahoo Finance download code, original schema and production metadata, and checksum evidence linking the v3 raw snapshot to the exact dataset used by v4. These sources are a historical archive; the current v4 workflow continues to use the frozen raw CSV.

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

The [environment notes](docs/REPRODUCIBILITY.md) distinguish versions recorded during the experiments from compatible dependencies used for reproduction.

## Integrity

The frozen raw file expected by the v4 pipeline has SHA-256:

```text
ab5f275d38dc98057b1cedcf58019adb26be7402c7ed5ae6ee3d6877b2444893
```

Protocol files and companion checksum files are included under `protocols/`. The code and protocol records are research artifacts; inclusion in this repository does not by itself guarantee bitwise reproduction on different hardware or library builds.

[`config/recovered_analysis_manifest_v4.json`](config/recovered_analysis_manifest_v4.json) lists the recovered analysis sources. The [28 October 2026 verification record](docs/validation/archive_recovery_checks_20261028.json) covers isolated reruns of all 42 main Stage 09 tests and all 22 Bartlett tests, together with checks on source files and saved SHAP outputs.
