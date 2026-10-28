# Pipeline Order

The scripts are preserved with their original Google Colab paths and scientific guardrails.

| Stage | Script | Purpose |
|---|---|---|
| 00 | `00_setup_v4.py` | Create the v4 project structure and verify frozen raw data |
| 01 | `01_rebuild_from_frozen_raw_v4.py` | Rebuild cleaned prices and derived tables |
| 02 | `02_preprocessing_v4.py` | Create features, targets, chronological splits, scalers, and sequences |
| 03 | `03_baseline_sanity_v4.py` | Validate sequences and naive validation baselines |
| 04 | `04_small_model_test_v4.py` | Smoke-test the training pipeline |
| 05a | `05a_mini_grid_v4.py` | Run pre-grid checks |
| 05 | `05_grid_search_v4.py` | Evaluate the locked 480-configuration grid |
| 06 | `06_best_model_multiseed_v4.py` | Evaluate candidate configurations across seeds 123, 777, and 2026 |
| 07 | `07_final_test_evaluation_v4.py` | Evaluate the locked final ensemble on the held-out test set |
| 08A | `08A_naive_baselines_test_v4.py` | Evaluate naive test baselines |
| 08B | `08B_learned_baselines_test_v4.py` | Train and evaluate learned single-task baselines |
| 08B audit | `08B_FINAL_AUDIT_v4.py` | Audit Stage 08B artifacts |
| 08C lock | `08C_create_garch_protocol_lock_v4.py` | Create the pre-fit econometric protocol lock |
| 08C check | `08C_validate_garch_protocol_lock_v4.py` | Validate the locked GARCH protocol |
| 08C | `08C_garch_baselines_test_v4.py` | Evaluate GARCH and GJR-GARCH baselines |
| 08C-R | `08C_R_garch_rescue_run_v4.py` | Run the locked numerical-convergence rescue |
| 09 | `09_dm_harvey_holm_v4.py` | Exact recovered source cell for the 42 locked main DM-HLN/Holm comparisons |
| 09D | `09D_bartlett_dm_hln_robustness_v1.py` | Run the post-hoc Bartlett-HAC DM-HLN volatility robustness analysis |

`09_dm_harvey_holm_v4.py` was extracted from the recovered main notebook without changing its calculation source. It retains the original Colab root and output guards. `reproduce_archived_analysis_v4.py` is a separately documented adapter for executing that source against an isolated frozen-input copy and new outputs. Use the adapter's `--help` and the instructions in [`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md).

Stage 10 SHAP computation source is in [`notebooks/tez_transformer_v4_repro.ipynb`](../notebooks/tez_transformer_v4_repro.ipynb). Preparation and smoke-test cells must precede full SHAP computation: they create the loaded ensemble and other required runtime variables. The notebook also preserves assembly, independent audit, and closure code. No standalone Stage 10 implementation has been invented.

Stage 09D is a later volatility-only robustness check with its own protocol and standalone script. It does not replace main Stage 09. Its separate dependency file is `requirements-bartlett.txt`.

The numbered stages describe the historical workflow, not an instruction to rerun every cell over a completed archive. Preserve original locked outputs and use a separate working copy for reproduction.
