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
| 09D | `09D_bartlett_dm_hln_robustness_v1.py` | Run the post-hoc Bartlett-HAC DM-HLN volatility robustness analysis |

Stage 09 DM-HLN/Holm and Stage 10 SHAP audit material is preserved in the main notebook and in `protocols/`. Standalone scripts for the locked main Stage 09 and Stage 10 analyses were not present in the verified Drive `scripts/` folder and have not been invented for this release. Stage 09D is a later, post-hoc robustness check with its own frozen protocol and verified standalone script.
