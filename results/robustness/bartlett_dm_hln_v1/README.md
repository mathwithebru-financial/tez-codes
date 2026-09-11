# Bartlett-HAC DM-HLN Robustness Check

## Status

Both the post-hoc analysis and its independent Google Colab reproduction passed the recorded validation checks.

| Item | Result |
|---|---:|
| Volatility comparisons | 22 |
| Test observations per comparison | 584 |
| Final model significantly better | 2 |
| Comparator significantly better | 12 |
| No significant difference | 8 |
| Decisions changed from main Stage 09 | 6 |
| Superiority direction reversals | 0 |

All six changed decisions moved from “comparator better” to “no significant difference.” The two significant final-model wins from the locked Stage 09 analysis were preserved.

## Frozen method

- Forecast horizon: `h=1`
- Bartlett truncation lag: `L=19`
- Bartlett weights: `1-k/20`
- HLN correction evaluated with `h=1`
- Student-`t` reference distribution: 583 degrees of freedom
- Multiple-testing correction: Holm across all 22 volatility comparisons
- Primary implementation: explicit Bartlett autocovariance sum
- Cross-check: equivalent statsmodels HAC calculation

The lag choice is a mechanically motivated sensitivity setting based on the overlap of adjacent 20-return realized-volatility targets. It is not presented as proof that dependence ends at lag 19.

## Published files

- `bartlett_dm_hln_results_v1.csv`: complete 22-row robustness result table
- `bartlett_dm_hln_changed_decisions_v1.csv`: six decisions that changed relative to main Stage 09
- `bartlett_dm_hln_autocovariances_v1.csv`: lag-specific autocovariance diagnostics
- `bartlett_dm_hln_summary_v1.json`: machine-readable result summary
- `colab_reproduction_check_v1.json`: independent rerun comparison
- `bartlett_dm_hln_report_v1.md`: human-readable analysis report

The detailed loss-differential series, numerical archive, spreadsheet workbook, and source/audit ZIP packages are intentionally excluded from the public repository.

## Integrity and independent reproduction

| Artifact | SHA-256 |
|---|---|
| Frozen protocol | `70a7aeb6bbfb0dfcfe7375e2d4e678837762af3b720e5b08c00fd8110f0074fc` |
| Source ZIP | `bc7dac4b88d8808fe556351a7045b524832ab8588148736243c230370ad56871` |
| Audit ZIP | `e7aba2784d2710ff27262c430e1b7614ac7d7f3700551f77fd097ec2cc338e9e` |

The maximum absolute numerical difference between the reference outputs and the independent Colab rerun was `2.6645352591003757e-15`, below the prespecified `1e-10` tolerance.

## Interpretation boundary

This is a separate post-hoc robustness check. It does not replace, overwrite, or retroactively modify the locked Stage 09 DM-HLN/Holm analysis.
