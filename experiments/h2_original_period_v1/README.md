# H2: original-period evaluation and verified results

This completed supplementary comparison evaluates fifteen frozen models on the
original thesis test period: **584 target dates, 6 October 2022–31 December 2024**.
The full price history remains 4 January 2010–31 December 2024. The separate 2026
assessment is outside the results reported here.

## Findings

Three seeds (123, 777, 2026) and four assets (BIST100, USDTRY, EURTRY, GOLD) are
included. Each seed's losses are calculated first, then averaged across assets
and seeds; these are not losses of an ensemble-averaged forecast.

| Arm | Return MAE, mean ± sample SD across seeds | Volatility pinball loss, mean ± sample SD across seeds |
|---|---:|---:|
| No sharing (`NS_base`) | 0.00797135 ± 0.00035560 | 0.00502645 ± 0.00023121 |
| Full sharing, same width (`FS_same_width`) | 0.00930838 ± 0.00142422 | 0.00824244 ± 0.00060896 |
| Full sharing, parameter-count control (`FS_budget_control`) | 0.00960510 ± 0.00059047 | 0.00957530 ± 0.00116280 |
| Independent task models (`STL_pair`) | 0.00726778 ± 0.00002505 | 0.00499759 ± 0.00045702 |

The four primary comparisons use the independent task pair as control:

| Full-sharing arm | Task | Loss increase relative to STL | Holm-adjusted p |
|---|---|---:|---:|
| Same width | Return | 28.08% | 7.17272e-11 |
| Same width | Volatility | 64.93% | 1.81475e-9 |
| Parameter-count control | Return | 32.16% | 1.92934e-10 |
| Parameter-count control | Volatility | 91.60% | 1.81475e-9 |

Loss increase is `(sharing loss / STL loss - 1) × 100`. Higher loss means worse
performance. The main specification is DM with Bartlett-HAC lag 19, HLN h=1
correction and a two-sided Student-t reference (583 degrees of freedom). Holm
correction is applied within each four-comparison family. All four secondary
comparisons against `NS_base` also favor the control; the lag-0 sensitivity
analysis retains the same decisions. The complete sixteen rows are retained in
[the comparison table](published_results_v1/dm_holm_comparisons_v1.csv).

**H2 is not supported for the architectures and experimental conditions studied.**
This conclusion concerns the four assets and fixed evaluation period above.
The STL–NS pair is not a directly tested contrast in these comparison families.
Training–validation–test gaps are descriptive across three seeds and do not
establish a causal effect on overfitting. Independent task training also differs
in optimization, gradient clipping and task-specific checkpoint selection.

The original test period had already been examined, and the separate 2026
aggregate results were known before this analysis plan was written. This is a
**retrospective supplementary analysis**, not a new unseen confirmation. The
[original fixed plan](protocol_original_period_v1.json) and
[frozen method note](README_ORIGINAL_PERIOD_V1.md) retain that chronology.

## Verification and result files

The Colab run completed on a Tesla T4. It reproduced all 48 recorded
training/validation metric rows for the fifteen selected models. The nine
existing H2 models matched their legacy archive exactly: nine prediction arrays,
eighteen daily test-loss arrays and three truth/date arrays, thirty arrays total.

An independent host audit scored the saved forecasts, recomputed losses,
aggregation, sample SDs, gaps, sixteen DM–HLN tests and four Holm families, and
checked the linked records. All **6,801 numerical and consistency checks** passed;
the largest DM/Holm numerical discrepancy was 7.11e-15. This count denotes audit
assertions, not observations. The host audit did not rerun model forward passes.
Small float64 gap-component versus archived float32 replay differences were
checked against the documented tolerances.

[Published results](published_results_v1/) contain exact compact CSV/JSON copies
and audit reports. The complete original output, including the forecast/loss NPZ
and learning curves, remains in the access-controlled
[Drive output folder](https://drive.google.com/drive/folders/1o_FmAKTk468o99UCeyIRQ6evKvjjRgYo).
The original evaluation manifest lists all eleven scientific output artifacts;
the public folder contains a documented subset, not the complete private bundle.
`PUBLISHED_FILES_SHA256.json` identifies every file included in the public snapshot.

Original run identities:

- Evaluation manifest SHA256: `193dbc57d5f56b1ff45541d2030204054e197c79fd61506196f9c00313d0cfb0`
- Analysis lock SHA256: `dbd7167d8aecc2cb02696d82b07bcfce1f6e3260cc53467a50d948a9f0d5d908`
- Forecast/loss NPZ SHA256: `acd314e80afd12ba27fc967c96a43369175cefbb09863b27e4bae0d6ba495100`

## Reproduce in the existing Colab/Drive setup

The [launcher](colab_original_period_v1.py) requires the previously staged private
inputs, frozen scalers, fifteen checkpoints, source records and legacy archive.
The public snapshot alone is not sufficient to rerun model inference. Mount the
same Drive account and restore the recorded T4 environment before running it:

- Interpreter: `/content/h2_t4_runtime_v1/venv/bin/python`
- Python 3.12.13; torch 2.11.0+cu128; numpy 2.3.5; pandas 2.2.3;
  scipy 1.17.0; scikit-learn 1.6.1; CUDA 12.8; cuDNN 91900.
- Frozen source project: `/content/drive/MyDrive/tez_transformer_h2_external_2026_v1`.
- Separate output project: `/content/drive/MyDrive/tez_transformer_h2_original_period_v1`.

Run this cell after mounting Drive and checking that the separate interpreter
exists. It downloads the reviewed launcher from an immutable commit and checks
its bytes before execution. The launcher verifies the source identities and
analysis lock, replays training/validation, and evaluates the original test.
An already completed identical run is verified and reused; conflicting or
partially published outputs are retained and rejected rather than overwritten.

```python
from pathlib import Path
import hashlib
import subprocess
import sys
import urllib.request

url = (
    "https://raw.githubusercontent.com/mathwithebru-financial/tez-codes/"
    "ff60d092fd5714a3c7fe2f1f164ca5ce492d7b48/"
    "experiments/h2_original_period_v1/colab_original_period_v1.py"
)
payload = urllib.request.urlopen(url, timeout=45).read()
expected_sha256 = "57a4ed5269c93022b33619b8b36d05c9716015233f1f508607db2c86637e6669"
if hashlib.sha256(payload).hexdigest() != expected_sha256:
    raise RuntimeError("Launcher identity mismatch")
script = Path("/content/h2_original_period_57a4ed5269c9.py")
if script.exists() and script.read_bytes() != payload:
    raise RuntimeError("Existing different launcher preserved")
if not script.exists():
    with script.open("xb") as handle:
        handle.write(payload)
subprocess.run([sys.executable, "-u", "-B", str(script)], check=True)
```

This update adds documentation and compact results. The reviewed launcher,
evaluator, protocol and frozen method note retain their original identities.
