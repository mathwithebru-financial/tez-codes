# Notebooks

## Applied lessons in Turkish

In the [applied time-series notebook](Finansal_Zaman_Serileri_Uygulamali_Dersler.ipynb), I follow one price observation from target construction to forecast evaluation. The worked examples include saved tables and plots. In Colab, **Runtime → Run all** recalculates the examples on CPU using files from a fixed repository revision. [Colab version](https://colab.research.google.com/github/mathwithebru-financial/tez-codes/blob/main/notebooks/Finansal_Zaman_Serileri_Uygulamali_Dersler.ipynb)

## Historical research notebooks

`tez_transformer_v4_repro.ipynb` is the recovered publication copy of the historical research notebook, including main Stage 09 DM-HLN/Holm and Stage 10 SHAP computation source.

- Code-cell outputs were removed.
- Execution counters were reset.
- Source cells were retained.
- The notebook still assumes the original Google Colab and Google Drive directory layout.

The historical notebook includes earlier diagnostics and cells that create protocols and outputs once. Running **Run all** against a completed archive can stop at the first protected file. The documented reruns use separate working directories.

The `.py` files in `scripts/` are the preferred source artifacts for Stages 00 through 08C-R. The recovered main Stage 09 calculation cell is also published verbatim as `scripts/09_dm_harvey_holm_v4.py`, with a separate isolated-run adapter.

The SHAP calculation starts with model preparation, protocol and method-addendum checks, and a readiness check. A two-sample smoke test sets up variables needed by the resumable computation and assembly cells. The independent audit and closure follow the final report. The computation requires the exact archived scalers, arrays and three checkpoints, which are distributed separately from the public notebook.

`09D_Bartlett_DM_HLN_Saglamlik_v1.ipynb` runs the independent Bartlett-HAC robustness check. Its paths reflect the current project layout. The underlying calculation is preserved in `scripts/09D_bartlett_dm_hln_robustness_v1.py`.

Both historical notebooks are published with cleared outputs, execution counters and account-specific Colab metadata. The path corrections changed the Bartlett notebook’s file hash. The [reproduction notes](../docs/REPRODUCIBILITY.md) document its inputs and the steps that can be rerun.
