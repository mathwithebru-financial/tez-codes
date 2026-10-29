# H2: new-period assessment with independent task controls

This version prepares a separate study from the frozen v4 and T4 H2 inputs. It does not replace the thesis results or rerun the original nine fits.

## What is supplied in this preparation version

- `protocol_plan_v1.json`: the immutable scientific preparation plan, including known prior exposure, the new-period boundary, control models and the intended comparison families.
- `stage_external_plan_v1.py`: verifies the original records and stages 22 allowlisted input files in a new Drive project. Old test arrays are handled only as opaque bytes.
- `run_stl_controls_v1.py`: prepares or trains six independent task models, using old training and validation data only.
- `prepare_external_sequences_v1.py`: a gated, transform-only processor. It refuses access to new prices until a later analysis lock and frozen STL checkpoints exist.

The initial project status is `STAGED_PLAN_NO_NEW_DATA`. It is **not** the final analysis lock, and it does not permit new-price acquisition or model evaluation on the new period. Acquisition and evaluation commands will be versioned and verified before that gate is opened.

## Existing results and new controls

The existing NoSharing and two FullSharing variants have three frozen seeds each. These nine selected checkpoints remain unchanged. The new control consists of one return-only and one volatility-only Transformer per seed, for **six additional fits**. Each task model receives the same eight input features and has 27,268 trainable parameters; the pair has 54,536.

All new fits run for 100 epochs with separate AdamW optimizers and task-specific checkpoint selection. The checkpoint rule uses the corresponding task component of the original normalized validation score, with frozen per-asset baseline denominators and the earliest minimum. It is distinct from the raw primary loss used to report external performance. The independent control changes optimization, clipping and checkpoint selection as a bundle; it must not be described as a pure intervention on parameter sharing.

## Data history and interpretation

The current v4 and existing H2 data end on 2024-12-31. Saved outputs from an earlier three-asset experiment demonstrate that 2025 data were already used for model testing. Therefore 2025 is excluded as a previously unexamined external test period.

No 2026 model evaluation was found in the inspected project records and finance notebooks. The user's recollection is uncertain, and two older project directories were not present in the current Drive mount. The audit cannot rule out deleted, unsaved or inaccessible prior work. Report the new study as a prespecified new-period external assessment with this scope; do not assert an absolutely pristine holdout.

The planned new target period is **2026-01-01 through 2026-10-28**. Its dates are fixed before new-price access. At least 150 prepared target dates and 150 actually observed price rows for each asset in that period are required. These are coverage requirements, not a statistical power guarantee. Failure of a data-quality gate must stop the evaluation; it must not lead to outcome-based instrument or date changes.

The instruments remain `XU100.IS`, `USDTRY=X`, `EURTRY=X` and `GC=F`. `GOLD` is the original USD gold-futures series; it is not the earlier `GOLD_TRY` series. The original scalers, splits and stored raw history are retained.

## Colab preparation

Use the already prepared separate interpreter:

```text
/content/h2_t4_runtime_v1/venv/bin/python
```

Stage the project from this complete verified bundle, then run the trainer's preparation mode:

```bash
python -B stage_external_plan_v1.py --bundle-root .
/content/h2_t4_runtime_v1/venv/bin/python -B \
  /content/drive/MyDrive/tez_transformer_h2_external_2026_v1/implementation/run_stl_controls_v1.py \
  --project-root /content/drive/MyDrive/tez_transformer_h2_external_2026_v1 \
  --prepare-only
```

Preparation must report that no training or new-data access occurred. Do not rerun the old notebook's training, staging or evaluation cells.

## Subsequent gates

1. Complete and verify all six new STL fits using the old training/validation splits.
2. Freeze their selected checkpoints and verify training/validation replay.
3. Finalize the acquisition and evaluation implementation; record all code, plan and checkpoint identities in `analysis_locked_v1.json` with status `ANALYSIS_LOCKED_BEFORE_NEW_DATA`.
4. Acquire and freeze the predetermined price period, validate coverage and instrument continuity, and prepare the new sequences without refitting scalers.
5. Evaluate every planned comparison and report all results, including unfavorable or inconclusive results.

Primary inference compares the two FullSharing controls with independent STL for both tasks in one four-test Holm family. Secondary inference compares FullSharing with the frozen NoSharing control in a separate four-test family. Both use Bartlett-HAC lag 19 and the stated DM-HLN convention; lag-zero results are explicitly separate sensitivity analyses. Seed averaging means averaging loss series, not averaging predictions before scoring.

Training-to-test gaps and learning curves are descriptive with three seeds. They do not establish a causal change in overfitting, and the DM test of loss differences is not a test of those gaps. The original hypothesis must not be relabeled as causally confirmed solely because a gap is smaller.

## Preservation and replay

All original project roots are protected. A rerun of staging verifies existing identical files; a different existing file is never silently overwritten. Completed STL jobs require identity and numerical replay checks before reuse. Partial jobs are retained for inspection rather than automatically deleted or restarted.
