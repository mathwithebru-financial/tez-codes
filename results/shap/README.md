# Stage 10 SHAP Summaries

This directory contains compact post-hoc SHAP summaries for the locked final ensemble.

The final closure record is `CLOSED`, with independent audit status `PASS`. The original `10_shap_report_v4.md` was generated before that audit and therefore retains its then-pending status. The later closure record and report in `10_closure/` document completion; the original report and result hashes have not been rewritten.

The recovered main notebook contains the actual SHAP preparation, full computation, assembly, audit, and closure source. Its saved execution records were matched to the six final result identities before publication cleaning. This archive update does not claim a new GPU SHAP computation. See [`notebooks/README.md`](../../notebooks/README.md) and [`docs/REPRODUCIBILITY.md`](../../docs/REPRODUCIBILITY.md).

The results:

- describe model behavior;
- do not establish causality;
- were not used for feature selection or model reselection;
- do not independently establish positive or negative task transfer.
