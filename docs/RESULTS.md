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
