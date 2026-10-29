# H2 comparison within the original thesis period

This supplementary evaluation compares the same fifteen frozen models on the
thesis's original **584 test examples**, whose target dates run from
**6 October 2022 through 31 December 2024**. The full thesis price history remains
4 January 2010 through 31 December 2024.

The dedicated output project is
`/content/drive/MyDrive/tez_transformer_h2_original_period_v1`. It preserves the
original thesis, existing H2 results and separate 2026 supplementary assessment.

The source project `tez_transformer_h2_external_2026_v1` supplies already frozen
original inputs, scalers, model definitions and selected checkpoints. Its name
describes their storage location. This evaluation reads no price or result files
from that project's `new_period` subtree, downloads no prices and performs no
training, resplitting, scaler fitting or checkpoint reselection.

The original test period and earlier results were already examined; the 2026
aggregate outcomes were also known when this plan was written. This is therefore
a **retrospective supplementary comparison on the original test period**. The
new analysis lock fixes the present workflow before its original-test inference;
it does not restore unseen status or establish independent confirmation.

The four primary comparisons test the two full-sharing controls against the
independent return/volatility model pair. Four separate secondary comparisons use
the no-sharing control. Each family uses its own Holm correction. The fixed main
statistic is Bartlett-HAC lag19 DM with HLN h1 correction and two-sided
Student-t reference with 583 degrees of freedom. Lag0 is separately reported as
sensitivity analysis. All three seeds and four assets are retained. Primary loss
series average each seed's losses, rather than losses of averaged forecasts.

Raw forecasts retain float32 storage. Raw truth and predictions are cast to
float64 before pointwise loss calculation, matching the archived original-period
H2 evaluator. The same arithmetic applies to all fifteen models and to the
training/validation/test gap components. The immutable legacy archive supplies a
compatibility check for the nine existing H2 models; no archived result is
replaced.

Training/validation/test gaps and seed variation remain descriptive. The
independent-task control also changes optimization, gradient clipping and
task-specific checkpoint selection; results concern this bundled comparison.

`protocol_original_period_v1.json` contains the complete fixed plan and source
identity anchors. A period-specific immutable lock must be verified before
deserializing the original test arrays in this workflow. No existing source or
result file is replaced.
