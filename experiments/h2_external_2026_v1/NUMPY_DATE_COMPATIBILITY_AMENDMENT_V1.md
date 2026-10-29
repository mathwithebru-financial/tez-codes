# NumPy date-scalar compatibility amendment

The original preparation failed at the first-anchor metadata lookup because
`pandas.Timestamp(numpy.str_)` raises `TypeError: Expected unicode, got numpy.str_`
with NumPy 2.3.5 and pandas 2.2.3. The failure occurred before output publication
and external model evaluation.

The corrected preparation converts that scalar with Python `str()` before
constructing the timestamp. Features, targets, date selection, missingness,
scalers, coverage requirements, selected models, losses and statistical tests
retain their original definitions.

`prepare_external_sequences_v2.py` uses the unchanged, hash-verified v1 helper
functions. It creates `technical_amendment_numpy_dates_v1.json` before corrected
processing. That immutable record identifies the original analysis lock, original
and corrected implementations, evaluation wrapper, acquired snapshot and model
manifests. It explicitly records that prices had already been accessed and
external model evaluation had yet to occur in this project. The original
`analysis_locked_v1.json` remains unchanged.

Prepared arrays retain the existing `new_period/prepared_v1` schema and identify
the technical amendment in their manifest. The entry point for subsequent model
evaluation is `evaluate_external_h2_numpy_dates_v1.py --evaluate`. This wrapper
verifies the amendment and delegates computation to the unchanged original
evaluator. It records execution provenance in a separate project-level receipt.

Offline regression checks reproduced the original failure and exercised the
complete corrected preparation, including publication and immutable reuse. All
11 numeric, date and missingness arrays matched the reference values exactly;
original input, checkpoint, source, raw-price and analysis-lock bytes remained
unchanged. Negative cases covered altered files and attempts to record the
amendment after preparation or evaluation outputs already existed.
