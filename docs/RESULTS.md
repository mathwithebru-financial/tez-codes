# Verified Result Summary

## Final model

The final locked ensemble contains three independently trained models with seeds `123`, `777`, and `2026`.

```text
Architecture: NoSharing
Loss: FixedLambda_0.7
Lookback: 10
Model size: small
Feature set: baseline
```

The single-seed grid winner used `FixedLambda_0.3`; the final three-seed selection used `FixedLambda_0.7`. These are distinct selection stages and should not be conflated.

## Statistical comparisons

Forty-two task-level comparisons were evaluated with Holm-adjusted DM-HLN tests.

| Task | Total | Final model better | Comparator better | No significant difference |
|---|---:|---:|---:|---:|
| Return | 20 | 3 | 16 | 1 |
| Volatility | 22 | 2 | 18 | 2 |
| **Total** | **42** | **5** | **34** | **3** |

The final model's statistically significant wins occurred:

- for return: BIST 100, EUR/TRY, and Gold against Return Persistence;
- for volatility: USD/TRY against XGBoost and EUR/TRY against the single-task Transformer.

These results do not establish universal model superiority. A non-significant DM-HLN result is not an equivalence result.

## Post-hoc Bartlett-HAC robustness check

The 22 volatility comparisons were rerun as a separate sensitivity analysis under the following frozen settings: one-step forecast horizon (`h=1`), Bartlett lag truncation `L=19`, weights `1-k/20`, Student-`t` reference distribution with 583 degrees of freedom, and Holm correction across all 22 volatility tests.

| Analysis | Final model better | Comparator better | No significant difference |
|---|---:|---:|---:|
| Main Stage 09 | 2 | 18 | 2 |
| Bartlett-HAC robustness | 2 | 12 | 8 |

Six decisions changed from comparator superiority to no significant difference. No comparison reversed superiority direction, and the two significant final-model wins were preserved. The independent Colab reproduction differed from the reference by at most `2.6645352591003757e-15`, below the prespecified `1e-10` tolerance.

This post-hoc check does not replace or retroactively modify the locked Stage 09 analysis. Compact validated outputs are available under [`results/robustness/bartlett_dm_hln_v1/`](../results/robustness/bartlett_dm_hln_v1/).

## Hypothesis boundary

- **H1:** not supported by the aggregate comparison evidence.
- **H2:** not directly tested. The selected `NoSharing` model does not share parameters or representations between return and volatility tasks, and the experiment did not separately measure overfitting reduction.

## SHAP boundary

The final SHAP tensor has shape `(584, 10, 8, 8)`:

```text
584 test observations
× 10 lookback positions
× 8 input features
× 8 outputs
```

SHAP was computed after model selection. It describes the locked ensemble's local behavior and does not establish causality, negative transfer, or feature-selection validity.
