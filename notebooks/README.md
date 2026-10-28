# Notebooks

`tez_transformer_v4_repro.ipynb` is the recovered publication copy of the historical research notebook, including main Stage 09 DM-HLN/Holm and Stage 10 SHAP computation source.

- Code-cell outputs were removed.
- Execution counters were reset.
- Source cells were retained.
- The notebook still assumes the original Google Colab and Google Drive directory layout.

Do not use **Run all** on a completed project archive. The notebook includes earlier diagnostics and controlled one-time protocol and output creation. Its no-overwrite guards are part of the recorded workflow; removing them to rerun on top of locked results is not the documented reproduction route.

The `.py` files in `scripts/` are the preferred source artifacts for Stages 00 through 08C-R. The recovered main Stage 09 calculation cell is also published verbatim as `scripts/09_dm_harvey_holm_v4.py`, with a separate isolated-run adapter.

For SHAP, follow the titled source cells in order: standalone model preparation; locked protocol and method-addendum verification; execution readiness; two-sample smoke test; resumable full computation; assembly and report generation; independent final audit; closure. The full computation and assembly depend on variables created by the preparation and smoke-test cells. They require the exact frozen scalers, arrays, and three checkpoints, which are not supplied by the public notebook.

`09D_Bartlett_DM_HLN_Saglamlik_v1.ipynb` is the corrected operational wrapper for the independent Bartlett-HAC robustness rerun. Its wrapper paths use the current project layout, rather than the removed historical folder name. This is a documented wrapper correction, not a change to the Bartlett calculation method. The original standalone calculation source remains `scripts/09D_bartlett_dm_hln_robustness_v1.py`.

Both publication notebooks have outputs, execution counters, and account-specific Colab metadata cleared. The corrected Bartlett wrapper is a derivative; its notebook hash is not represented as the historical executed notebook hash. See [`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) for inputs and execution boundaries.
