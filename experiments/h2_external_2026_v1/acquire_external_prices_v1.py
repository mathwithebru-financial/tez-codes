#!/usr/bin/env python3
"""Acquire the fixed H2 external-period prices once, after the final analysis lock.

This program never installs packages, trains a model, opens an old test array,
repairs prices, fills missing observations or overwrites historical files. Use a
separate virtual environment, NOT /content/h2_t4_runtime_v1/venv, with Python
3.12.13, numpy==2.3.5, pandas==2.2.3 and yfinance==0.2.66. The complete installed
package inventory is recorded with the acquired snapshot. --self-test uses only
synthetic provider tables and makes no network requests.

Provider API sources checked before any acquisition:
https://pypi.org/project/yfinance/0.2.66/
https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html
The fixed Close request explicitly disables auto_adjust, back_adjust and repair;
start is inclusive and end is exclusive. The source code hash freezes all other
kwargs. There is one application-level download call and no automatic rerun of
partial or failed acquisitions. Internal provider-library requests are outside
the application's retry control and are recorded as such.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import errno
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any, Callable

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

DEFAULT_ROOT = Path('/content/drive/MyDrive/tez_transformer_h2_external_2026_v1')
PROTECTED_ROOTS = tuple(Path(p) for p in (
    '/content/drive/MyDrive/tez_transformer_v4_repro',
    '/content/drive/MyDrive/tez_transformer_h2_v1',
    '/content/drive/MyDrive/tez_transformer_h2_t4_v1'))
TRAINING_VENV = Path('/content/h2_t4_runtime_v1/venv')
OLD_RUN_ROOT = PROTECTED_ROOTS[2] / 'runs/t4_full_run_v1'
ASSETS = ('BIST100', 'USDTRY', 'EURTRY', 'GOLD')
TICKERS = dict(zip(ASSETS, ('XU100.IS', 'USDTRY=X', 'EURTRY=X', 'GC=F')))
PLAN_SHA = '0073b3bcc80a342d25f1034fef66d5eb353a9e4e5c14ae3719f57e98de81a659'
OLD_MANIFEST_SHA = '517c06f3b1a5d71de9739f56ef6a2af7cf9a918ed2cf303a52d30bfc7b9f393a'
OLD_FROZEN_SHA = '12cf2481978546689f9087d7d97cb4d8325756a15f393a9c936a31e2a47876f4'
SOURCE_08 = 'scripts/08B_learned_baselines_test_v4.py'
SOURCE_08_SHA = 'df4fa2b29a6b522fc410f693a30520e045c144a15f7c7368af5eb6a1f0f12566'
MANDATORY_CODE = frozenset(('run_stl_controls_v1.py', 'prepare_external_sequences_v1.py',
    'acquire_external_prices_v1.py', 'evaluate_external_h2_v1.py', 'lock_external_analysis_v1.py'))
OUTPUT_REL = 'new_period/acquisition_v1'
START, END, TEST_START = '2024-11-01', '2026-10-29', '2026-01-01'
LAST_DAY = '2026-10-28'
BASE_PARAMETERS = {'auto_adjust': False, 'back_adjust': False, 'repair': False, 'interval': '1d'}
RUNTIME_CONTRACT = {'provider': 'Yahoo Finance', 'library': 'yfinance',
    'library_version': '0.2.66', 'isolated_from_training': True,
    'python_version': '3.12.13', 'numpy_version': '2.3.5', 'pandas_version': '2.2.3',
    'download_parameters': BASE_PARAMETERS, 'download_start_inclusive': START,
    'download_end_exclusive': END}
DOWNLOAD_KWARGS = {'start': START, 'end': END, **BASE_PARAMETERS,
    'group_by': 'ticker', 'threads': False, 'progress': False, 'ignore_tz': True,
    'keepna': True, 'actions': False, 'prepost': False, 'rounding': False,
    'timeout': 30, 'multi_level_index': True}


class AcquisitionError(RuntimeError):
    """An immutable identity, acquisition prerequisite or price gate failed."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def no_symlinks(path: Path) -> None:
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise AcquisitionError(f'Symlink paths are not allowed: {path}')


def child(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise AcquisitionError('Artifact paths must be nonempty and relative.')
    no_symlinks(root / relative)
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise AcquisitionError('Artifact path escapes its root.')
    return path


def checked_root(root: Path) -> Path:
    no_symlinks(root)
    root = Path(root).resolve()
    if not root.is_dir():
        raise AcquisitionError('The already-staged project root must exist.')
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in PROTECTED_ROOTS):
        raise AcquisitionError('Project collides with a protected historical root.')
    return root


def digest(path: Path) -> str:
    no_symlinks(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_json(path: Path, expected_sha: str | None = None) -> dict:
    no_symlinks(path)
    payload = path.read_bytes()
    if expected_sha is not None and hashlib.sha256(payload).hexdigest() != expected_sha:
        raise AcquisitionError(f'JSON identity differs: {path}')
    record = json.loads(payload)
    if not isinstance(record, dict):
        raise AcquisitionError(f'Expected a JSON object: {path}')
    return record


def verify_file(path: Path, identity: dict) -> None:
    if not isinstance(identity, dict) or not path.is_file():
        raise AcquisitionError(f'Required immutable file is missing: {path}')
    expected = identity.get('sha256')
    if not isinstance(expected, str) or len(expected) != 64 or digest(path) != expected:
        raise AcquisitionError(f'File identity differs: {path}')
    if 'size_bytes' in identity and path.stat().st_size != identity['size_bytes']:
        raise AcquisitionError(f'File size differs: {path}')


def gate_before_prices(root: Path) -> tuple[dict, dict]:
    """Only immutable metadata and opaque file hashes are read by this gate."""
    root = checked_root(root)
    lock_path = child(root, 'analysis_locked_v1.json')
    lock = checked_json(lock_path)
    if lock.get('status') != 'ANALYSIS_LOCKED_BEFORE_NEW_DATA' or lock.get('protocol_plan_sha256') != PLAN_SHA:
        raise AcquisitionError('The complete analysis must be locked before any price access.')
    plan = checked_json(child(root, 'protocol_plan_v1.json'), PLAN_SHA)
    if plan.get('project_root') != str(root):
        raise AcquisitionError('Project root differs from the original fixed plan.')
    implementations = lock.get('implementation_sha256')
    if not isinstance(implementations, dict) or not MANDATORY_CODE.issubset(implementations):
        raise AcquisitionError('The final lock must identify all five analysis implementations.')
    for name, sha in implementations.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise AcquisitionError('Locked implementation names must be basenames.')
        verify_file(child(root, f'implementation/{name}'), {'sha256': sha})
    verify_file(Path(__file__).resolve(), {'sha256': implementations[Path(__file__).name]})
    if lock.get('acquisition_runtime') != RUNTIME_CONTRACT:
        raise AcquisitionError('The acquisition library, runtime or request contract was not fixed exactly.')

    old_manifest = checked_json(child(root, 'source_records/locked_training_manifest.json'), OLD_MANIFEST_SHA)
    if old_manifest.get('status') != 'LOCKED_BEFORE_TRAINING' or old_manifest.get('run_root') != str(OLD_RUN_ROOT):
        raise AcquisitionError('Original training provenance differs.')
    inputs = old_manifest.get('input_identities')
    if not isinstance(inputs, dict) or len(inputs) != 21:
        raise AcquisitionError('Exactly 21 originally frozen input files are required.')
    for relative, identity in inputs.items():
        verify_file(child(root, 'inputs/' + relative), identity)
    verify_file(child(root, 'inputs/' + SOURCE_08), {'sha256': SOURCE_08_SHA})

    receipt_path = child(root, 'stage_receipt_v1.json')
    receipt = checked_json(receipt_path)
    # The original training manifest and 08B constant independently anchor every
    # input; the lock also anchors the complete stage receipt when provided.
    if 'stage_receipt_sha256' in lock:
        verify_file(receipt_path, {'sha256': lock['stage_receipt_sha256']})
    staged = receipt.get('identities', {})
    expected_inputs = dict(inputs)
    expected_inputs[SOURCE_08] = {'sha256': SOURCE_08_SHA,
        'size_bytes': child(root, 'inputs/' + SOURCE_08).stat().st_size}
    if (receipt.get('status') != 'H2_EXTERNAL_PLAN_STAGED_NO_NEW_DATA'
            or staged.get('project_root') != str(root)
            or staged.get('protocol_plan', {}).get('sha256') != PLAN_SHA
            or staged.get('input_identities') != expected_inputs
            or any(receipt.get(k) is not False for k in
                   ('training_started', 'new_period_downloaded', 'test_arrays_loaded', 'final_analysis_lock_created'))):
        raise AcquisitionError('The staged preparation receipt differs from the frozen input identities.')

    old_record = lock.get('existing_h2_checkpoints_manifest')
    if not isinstance(old_record, dict) or old_record.get('path') != 'source_records/selected_checkpoints_frozen.json' or old_record.get('sha256') != OLD_FROZEN_SHA:
        raise AcquisitionError('The nine old checkpoints are not anchored to the original frozen manifest.')
    old = checked_json(child(root, old_record['path']), OLD_FROZEN_SHA)
    expected_old = {(a, s) for a in ('NS_base', 'FS_same_width', 'FS_budget_control') for s in (123, 777, 2026)}
    jobs = old.get('completed_runs')
    if (old.get('status') != 'ALL_NINE_CHECKPOINTS_FROZEN_NO_TEST_ACCESS'
            or old.get('locked_training_manifest_sha256') != OLD_MANIFEST_SHA
            or not isinstance(jobs, list) or len(jobs) != 9
            or {(j.get('arm', {}).get('id'), j.get('seed')) for j in jobs} != expected_old):
        raise AcquisitionError('All nine existing H2 arm/seed selections must be frozen.')
    selected = []
    for job in jobs:
        cp = job.get('best_checkpoint')
        if not isinstance(cp, dict):
            raise AcquisitionError('An old selected checkpoint identity is missing.')
        verify_file(child(OLD_RUN_ROOT, cp.get('path')), cp)
        selected.append({'arm': job['arm']['id'], 'seed': job['seed'], **cp})

    stl_record = lock.get('selected_stl_checkpoints_manifest')
    stl_relative = 'runs/stl_run_v1/selected_stl_checkpoints_frozen.json'
    if not isinstance(stl_record, dict) or stl_record.get('path') != stl_relative:
        raise AcquisitionError('The final lock does not identify the six STL selections.')
    stl_path = child(root, stl_relative)
    verify_file(stl_path, stl_record)
    stl = checked_json(stl_path, stl_record['sha256'])
    controls = stl.get('checkpoints')
    if (stl.get('status') != 'STL_CHECKPOINTS_FROZEN_BEFORE_NEW_DATA'
            or stl.get('plan_sha256') != PLAN_SHA
            or stl.get('training_implementation_sha256') != implementations['run_stl_controls_v1.py']
            or stl.get('n_actual_fits') != 6
            or stl.get('test_arrays_loaded') is not False or stl.get('new_period_accessed') is not False
            or not isinstance(controls, list) or len(controls) != 6
            or {(c.get('task'), c.get('seed')) for c in controls}
                != {(t, s) for t in ('return', 'volatility') for s in (123, 777, 2026)}):
        raise AcquisitionError('Six independent task checkpoints must be frozen before new data.')
    for cp in controls:
        verify_file(child(root, cp.get('path')), cp)
    return lock, {'analysis_lock_sha256': digest(lock_path), 'protocol_plan_sha256': PLAN_SHA,
        'implementation_sha256': implementations, 'stage_receipt_sha256': digest(receipt_path),
        'existing_h2_checkpoints_manifest_sha256': OLD_FROZEN_SHA,
        'selected_stl_checkpoints_manifest_sha256': stl_record['sha256'],
        'n_verified_inputs': 22, 'n_verified_selected_checkpoints': 15,
        'test_arrays_loaded': False, 'old_project_writes': False}


def acquisition_environment() -> dict:
    # sys.executable.resolve() dereferences a venv Python symlink to the base
    # binary, so the actual environment identity is sys.prefix, not that target.
    prefix = Path(sys.prefix).absolute()
    if sys.prefix == sys.base_prefix or prefix == TRAINING_VENV or prefix.is_relative_to(TRAINING_VENV):
        raise AcquisitionError('Use a separate acquisition virtual environment; do not install into the H2 training environment.')
    pins = {'numpy': '2.3.5', 'pandas': '2.2.3', 'yfinance': '0.2.66'}
    actual = {name: importlib.metadata.version(name) for name in pins}
    if platform.python_version() != '3.12.13' or actual != pins:
        raise AcquisitionError('Acquisition requires Python 3.12.13, numpy2.3.5, pandas2.2.3 and yfinance0.2.66 exactly.')
    inventory = sorted(({'name': d.metadata.get('Name', ''), 'version': d.version}
        for d in importlib.metadata.distributions()), key=lambda d: (d['name'].lower(), d['version']))
    return {'python_version': platform.python_version(), 'interpreter': sys.executable,
        'prefix': sys.prefix, 'base_prefix': sys.base_prefix, 'platform': platform.platform(),
        'direct_packages': actual, 'installed_packages': inventory,
        'isolated_from_training': True}


def extract_prices(response: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate the exact ticker-level Close response; never impute or trim."""
    if not isinstance(response, pd.DataFrame) or response.empty:
        raise AcquisitionError('Provider returned no tabular prices.')
    if not isinstance(response.columns, pd.MultiIndex) or response.columns.nlevels != 2 or response.columns.has_duplicates:
        raise AcquisitionError('Expected the exact two-level ticker/price-column response.')
    if set(response.columns.get_level_values(0)) != set(TICKERS.values()):
        raise AcquisitionError('Provider ticker identities differ from the four fixed instruments.')
    if not isinstance(response.index, pd.DatetimeIndex):
        raise AcquisitionError('Provider daily index must be a DatetimeIndex.')
    index = response.index
    if (index.tz is not None or index.hasnans or not index.equals(index.normalize())
            or not index.is_monotonic_increasing or index.has_duplicates):
        raise AcquisitionError('Provider daily dates must be timezone-naive, sorted, unique date labels.')
    if index.min() < pd.Timestamp(START) or index.max() >= pd.Timestamp(END):
        raise AcquisitionError('Provider returned dates outside the prespecified acquisition window; no silent trimming is allowed.')
    frame = pd.DataFrame(index=index)
    for asset, ticker in TICKERS.items():
        if (ticker, 'Close') not in response.columns:
            raise AcquisitionError(f'The requested unadjusted Close column is missing: {ticker}')
        series = pd.to_numeric(response[(ticker, 'Close')], errors='raise').astype(np.float64)
        values = series.dropna().to_numpy()
        if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
            raise AcquisitionError(f'Observed Close prices must be finite and strictly positive: {ticker}')
        frame[asset] = series
    empty_rows = frame.isna().all(axis=1)
    # A provider table can include all-missing holiday rows when keepna=True.
    # Their removal creates the declared union of actually observed dates and is
    # separately counted; no asset observation is changed or manufactured.
    n_empty = int(empty_rows.sum())
    frame = frame.loc[~empty_rows].copy()
    frame.index.name = 'Date'
    in_test = (frame.index >= pd.Timestamp(TEST_START)) & (frame.index <= pd.Timestamp(LAST_DAY))
    coverage = {}
    for asset in ASSETS:
        observed = frame[asset].dropna()
        n_test = int(frame.loc[in_test, asset].notna().sum())
        if n_test < 150:
            raise AcquisitionError(f'Insufficient actually observed new-period Close rows for {asset}: {n_test} < 150; dates are not shifted.')
        coverage[asset] = {'ticker': TICKERS[asset], 'n_observed': int(len(observed)),
            'first_observed_date': observed.index.min().strftime('%Y-%m-%d'),
            'last_observed_date': observed.index.max().strftime('%Y-%m-%d'),
            'n_missing_on_observed_union_calendar': int(frame[asset].isna().sum()),
            'n_observed_in_new_test_period': n_test,
            'n_missing_in_new_test_period': int(frame.loc[in_test, asset].isna().sum())}
    if int(in_test.sum()) < 150:
        raise AcquisitionError('Fewer than150 actual union-calendar new-period rows; dates are not shifted.')
    return frame, {'assets': coverage, 'n_provider_rows': int(len(response)),
        'n_all_assets_missing_provider_rows_removed': n_empty, 'n_observed_union_rows': int(len(frame)),
        'n_new_period_union_rows': int(in_test.sum()), 'forward_fill_applied': False,
        'backward_fill_applied': False, 'price_repair_applied': False}


def json_bytes(record: dict) -> bytes:
    return (json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + '\n').encode('utf-8')


def write_new(path: Path, payload: bytes) -> dict:
    no_symlinks(path)
    with path.open('xb') as f:
        f.write(payload)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError as error:
            if error.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP):
                raise
    if path.read_bytes() != payload:
        raise AcquisitionError(f'Acquisition write/readback differs: {path}')
    return {'sha256': hashlib.sha256(payload).hexdigest(), 'size_bytes': len(payload)}


def verify_existing(output: Path, evidence: dict) -> dict:
    """A completed snapshot can be reused without any provider import or call."""
    record_file = child(output, 'acquisition_record.json')
    if not record_file.is_file():
        raise AcquisitionError('A partial/failed acquisition already exists. Its files are preserved; review it before any further provider request.')
    record = checked_json(record_file)
    if (record.get('status') != 'ACQUISITION_COMPLETE'
            or any(record.get(k) != v for k, v in evidence.items())
            or record.get('acquisition_runtime_contract') != RUNTIME_CONTRACT
            or record.get('assets') != list(ASSETS) or record.get('tickers') != TICKERS
            or record.get('download_parameters') != BASE_PARAMETERS
            or record.get('effective_download_kwargs') != DOWNLOAD_KWARGS
            or record.get('provider') != 'Yahoo Finance' or record.get('price_field') != 'Close'
            or record.get('adjusted') is not False
            or record.get('data_start') != START or record.get('data_end_inclusive') != LAST_DAY):
        raise AcquisitionError('Existing acquisition differs from the fixed analysis and request identities.')
    artifacts = record.get('artifacts')
    names = {'acquisition_started.json', 'provider_response.csv', 'raw_prices.csv'}
    if not isinstance(artifacts, dict) or set(artifacts) != names:
        raise AcquisitionError('Completed acquisition artifact inventory differs.')
    if {p.name for p in output.iterdir()} != names | {'acquisition_record.json'}:
        raise AcquisitionError('Unexpected files in the completed acquisition; review before reuse.')
    for name, identity in artifacts.items():
        verify_file(child(output, name), identity)
    if record.get('raw_file_sha256') != artifacts['raw_prices.csv']['sha256']:
        raise AcquisitionError('Canonical raw snapshot identity differs.')
    return record


def acquire(root: Path) -> tuple[dict, bool]:
    # No provider module import, new-period path read, CSV parse or request can
    # precede this complete immutable-metadata/checkpoint gate.
    root = checked_root(root)
    _, evidence = gate_before_prices(root)
    output = child(root, OUTPUT_REL)
    if output.exists():
        return verify_existing(output, evidence), True
    environment = acquisition_environment()
    import yfinance as yf
    if yf.__version__ != RUNTIME_CONTRACT['library_version']:
        raise AcquisitionError('Imported provider-library version differs from the distribution pin.')
    output.parent.mkdir(parents=True, exist_ok=True)
    no_symlinks(output.parent)
    try:
        output.mkdir()
    except FileExistsError as error:
        raise AcquisitionError('Another acquisition already created the output directory; no download was attempted.') from error
    started = {'status': 'ACQUISITION_STARTED', 'started_at_utc': now(), **evidence,
        'acquisition_runtime_contract': RUNTIME_CONTRACT, 'runtime_environment': environment,
        'assets': list(ASSETS), 'tickers': TICKERS, 'effective_download_kwargs': DOWNLOAD_KWARGS,
        'application_level_download_calls_planned': 1, 'application_level_retries': 0,
        'library_internal_requests': 'yfinance may perform internal requests/retries; no application retry or window/ticker substitution',
        'old_test_arrays_loaded': False, 'models_evaluated': False}
    artifacts = {'acquisition_started.json': write_new(output / 'acquisition_started.json', json_bytes(started))}
    try:
        response = yf.download(tickers=list(TICKERS.values()), **DOWNLOAD_KWARGS)
        # Preserve every returned provider column before scientific data gates.
        if isinstance(response, pd.DataFrame):
            payload = response.to_csv(index=True, lineterminator='\n', float_format='%.17g').encode('utf-8')
            artifacts['provider_response.csv'] = write_new(output / 'provider_response.csv', payload)
        prices, coverage = extract_prices(response)
        raw = prices.to_csv(index=True, date_format='%Y-%m-%d', lineterminator='\n', float_format='%.17g').encode('utf-8')
        artifacts['raw_prices.csv'] = write_new(output / 'raw_prices.csv', raw)
        record = {'status': 'ACQUISITION_COMPLETE', 'started_at_utc': started['started_at_utc'],
            'completed_at_utc': now(), **evidence, 'provider': 'Yahoo Finance',
            'price_field': 'Close', 'adjusted': False, 'assets': list(ASSETS), 'tickers': TICKERS,
            'download_parameters': BASE_PARAMETERS, 'effective_download_kwargs': DOWNLOAD_KWARGS,
            'data_start': START, 'data_end_inclusive': LAST_DAY, 'data_end_exclusive': END,
            'acquisition_runtime_contract': RUNTIME_CONTRACT, 'runtime_environment': environment,
            'raw_file_sha256': artifacts['raw_prices.csv']['sha256'], 'artifacts': artifacts,
            'coverage': coverage, 'application_level_download_calls': 1, 'application_level_retries': 0,
            'library_internal_requests': started['library_internal_requests'],
            'old_test_arrays_loaded': False, 'models_evaluated': False,
            'historical_2026_exposure_scope': 'No2026 experiment found in inspected records; deleted, unsaved or inaccessible work cannot be ruled out.'}
        write_new(output / 'acquisition_record.json', json_bytes(record))
        verify_existing(output, evidence)
        return record, False
    except BaseException as error:
        failure = {'status': 'ACQUISITION_FAILED_REVIEW_REQUIRED', 'failed_at_utc': now(),
            'exception_type': type(error).__name__, 'exception_message': str(error),
            'artifacts_preserved': artifacts, 'automatic_retry_allowed': False,
            'application_level_download_calls': 1, **evidence}
        failed = output / 'acquisition_failed.json'
        if not failed.exists():
            write_new(failed, json_bytes(failure))
        raise


def self_test() -> dict:
    """Numerical, identity and immutable-output tests with no network."""
    from unittest.mock import patch
    checks, rejections = [], []
    dates = pd.bdate_range('2025-12-01', '2026-10-28')
    cols = pd.MultiIndex.from_product([list(TICKERS.values()), ['Close', 'Open']], names=['Ticker', 'Price'])
    table = pd.DataFrame(10.0 + np.arange(len(dates))[:, None] / 100 + np.arange(8)[None, :], index=dates, columns=cols)
    table.loc[dates[3], (TICKERS['GOLD'], 'Close')] = np.nan
    prices, coverage = extract_prices(table)
    assert list(prices) == list(ASSETS) and prices.index.name == 'Date'
    assert np.isnan(prices.loc[dates[3], 'GOLD']) and coverage['forward_fill_applied'] is False
    assert prices.loc[dates[4], 'USDTRY'] == table.loc[dates[4], (TICKERS['USDTRY'], 'Close')]
    checks.append('exact ticker/Close mapping preserves missing observations')
    holiday = table.copy()
    for ticker in TICKERS.values():
        holiday.loc[dates[2], (ticker, 'Close')] = np.nan
    _, holiday_coverage = extract_prices(holiday)
    assert holiday_coverage['n_all_assets_missing_provider_rows_removed'] == 1
    checks.append('all-empty holidays removed explicitly to form observed union calendar')

    def rejects(label: str, fn: Callable[[], Any]) -> None:
        try:
            fn()
        except (AcquisitionError, ValueError, FileNotFoundError, KeyError):
            rejections.append(label)
        else:
            raise AssertionError(f'Invalid case was accepted: {label}')

    wrong = table.rename(columns={TICKERS['GOLD']: 'GOLD_TRY'}, level=0)
    rejects('wrong instrument', lambda: extract_prices(wrong))
    rejects('missing exact Close column', lambda: extract_prices(table.drop(columns=[(TICKERS['GOLD'], 'Close')])))
    rejects('flat unverified ticker columns', lambda: extract_prices(prices))
    rejects('duplicate daily dates', lambda: extract_prices(pd.concat([table, table.tail(1)])))
    rejects('unsorted daily dates', lambda: extract_prices(table.iloc[::-1]))
    rejects('timezone-bearing daily dates', lambda: extract_prices(table.set_axis(table.index.tz_localize('UTC'))))
    bad = table.copy(); bad.iloc[0, 0] = 0
    rejects('zero price', lambda: extract_prices(bad))
    bad = table.copy(); bad.iloc[0, 0] = np.inf
    rejects('infinite price', lambda: extract_prices(bad))
    future = table.copy(); future.index = list(future.index[:-1]) + [pd.Timestamp(END)]
    rejects('exclusive-end/future price', lambda: extract_prices(future))
    rejects('insufficient observed new-period coverage', lambda: extract_prices(table.head(10)))
    with tempfile.TemporaryDirectory() as name:
        temp = Path(name)
        with patch.dict(sys.modules, {'yfinance': None}):
            rejects('no lock fails before provider import or output creation', lambda: acquire(temp))
        assert not (temp / 'new_period').exists()
        checks.append('pre-lock acquisition refuses before any new-period artifact or provider import')
        rejects('relative escape', lambda: child(temp, '../escape.csv'))
        (temp / 'target').write_bytes(b'x')
        (temp / 'link').symlink_to(temp / 'target')
        rejects('symlink artifact', lambda: child(temp, 'link'))
        f = temp / 'immutable.csv'
        identity = write_new(f, b'Date,BIST100,USDTRY,EURTRY,GOLD\n')
        try:
            write_new(f, b'changed')
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing acquisition output was overwritten.')
        verify_file(f, identity)
        checks.append('exclusive immutable writes and SHA readback')
        partial = temp / 'partial'; partial.mkdir()
        (partial / 'provider_response.csv').write_text('partial')
        rejects('partial acquisition not downloaded again', lambda: verify_existing(partial, {}))

    # A complete synthetic metadata/checkpoint tree exercises the entire gate.
    # All checkpoint files contain opaque dummy bytes and are never loaded.
    with tempfile.TemporaryDirectory() as name:
        temp = Path(name)
        root = temp / 'project'; root.mkdir()
        old_root = temp / 'archived_run'; old_root.mkdir()

        def fixture(relative: str, payload: bytes) -> dict:
            path = root / relative; path.parent.mkdir(parents=True, exist_ok=True)
            return write_new(path, payload)

        plan_identity = fixture('protocol_plan_v1.json', json_bytes({'project_root': str(root)}))
        input_ids = {f'opaque_input_{i}.bin': fixture(f'inputs/opaque_input_{i}.bin', str(i).encode()) for i in range(21)}
        source_identity = fixture('inputs/' + SOURCE_08, b'synthetic-STL-source')
        manifest = {'status': 'LOCKED_BEFORE_TRAINING', 'run_root': str(old_root), 'input_identities': input_ids}
        old_manifest_identity = fixture('source_records/locked_training_manifest.json', json_bytes(manifest))
        old_jobs = []
        for arm in ('NS_base', 'FS_same_width', 'FS_budget_control'):
            for seed in (123, 777, 2026):
                filename = f'{arm}_{seed}.pt'
                cp = {'path': filename, **write_new(old_root / filename, b'opaque-checkpoint-' + filename.encode())}
                old_jobs.append({'arm': {'id': arm}, 'seed': seed, 'best_checkpoint': cp})
        old_identity = fixture('source_records/selected_checkpoints_frozen.json', json_bytes({
            'status': 'ALL_NINE_CHECKPOINTS_FROZEN_NO_TEST_ACCESS',
            'locked_training_manifest_sha256': old_manifest_identity['sha256'], 'completed_runs': old_jobs}))
        implementations = {}
        for filename in MANDATORY_CODE:
            payload = Path(__file__).read_bytes() if filename == Path(__file__).name else b'synthetic-code-' + filename.encode()
            implementations[filename] = fixture('implementation/' + filename, payload)['sha256']
        stl_cps = []
        for task in ('return', 'volatility'):
            for seed in (123, 777, 2026):
                relative = f'runs/stl_run_v1/{task}_{seed}.pt'
                stl_cps.append({'task': task, 'seed': seed, 'path': relative,
                    **fixture(relative, b'opaque-STL-' + relative.encode())})
        stl_relative = 'runs/stl_run_v1/selected_stl_checkpoints_frozen.json'
        stl_identity = fixture(stl_relative, json_bytes({'status': 'STL_CHECKPOINTS_FROZEN_BEFORE_NEW_DATA',
            'plan_sha256': plan_identity['sha256'],
            'training_implementation_sha256': implementations['run_stl_controls_v1.py'],
            'n_actual_fits': 6, 'test_arrays_loaded': False, 'new_period_accessed': False, 'checkpoints': stl_cps}))
        receipt_identity = fixture('stage_receipt_v1.json', json_bytes({'status': 'H2_EXTERNAL_PLAN_STAGED_NO_NEW_DATA',
            'identities': {'project_root': str(root), 'protocol_plan': plan_identity,
                'input_identities': {**input_ids, SOURCE_08: source_identity}},
            'training_started': False, 'new_period_downloaded': False,
            'test_arrays_loaded': False, 'final_analysis_lock_created': False}))
        fixture('analysis_locked_v1.json', json_bytes({'status': 'ANALYSIS_LOCKED_BEFORE_NEW_DATA',
            'protocol_plan_sha256': plan_identity['sha256'], 'implementation_sha256': implementations,
            'acquisition_runtime': RUNTIME_CONTRACT, 'stage_receipt_sha256': receipt_identity['sha256'],
            'existing_h2_checkpoints_manifest': {'path': 'source_records/selected_checkpoints_frozen.json', **old_identity},
            'selected_stl_checkpoints_manifest': {'path': stl_relative, **stl_identity}}))
        patched = {'PLAN_SHA': plan_identity['sha256'], 'OLD_RUN_ROOT': old_root,
            'OLD_MANIFEST_SHA': old_manifest_identity['sha256'], 'OLD_FROZEN_SHA': old_identity['sha256'],
            'SOURCE_08_SHA': source_identity['sha256']}
        with patch.dict(globals(), patched), patch.object(np, 'load', side_effect=AssertionError('An array/checkpoint was deserialized')):
            _, evidence = gate_before_prices(root)
            assert evidence['n_verified_inputs'] == 22 and evidence['n_verified_selected_checkpoints'] == 15
            checks.append('full metadata gate verifies22inputs and15opaque checkpoints without deserialization')
            changed = old_root / old_jobs[0]['best_checkpoint']['path']
            original = changed.read_bytes(); changed.write_bytes(original + b'changed')
            rejects('tampered old selected checkpoint', lambda: gate_before_prices(root))
            changed.write_bytes(original)
            changed = root / stl_cps[0]['path']
            original = changed.read_bytes(); changed.write_bytes(original + b'changed')
            rejects('tampered STL selected checkpoint', lambda: gate_before_prices(root))
            changed.write_bytes(original)
            changed = root / 'implementation/evaluate_external_h2_v1.py'
            original = changed.read_bytes(); changed.write_bytes(original + b'changed')
            rejects('changed implementation before provider access', lambda: gate_before_prices(root))
            changed.write_bytes(original)
            output = root / OUTPUT_REL; output.mkdir(parents=True)
            artifacts = {}
            for filename in ('acquisition_started.json', 'provider_response.csv', 'raw_prices.csv'):
                artifacts[filename] = write_new(output / filename, b'opaque-fixture-' + filename.encode())
            completed = {'status': 'ACQUISITION_COMPLETE', **evidence,
                'acquisition_runtime_contract': RUNTIME_CONTRACT,
                'assets': list(ASSETS), 'tickers': TICKERS,
                'download_parameters': BASE_PARAMETERS, 'effective_download_kwargs': DOWNLOAD_KWARGS,
                'provider': 'Yahoo Finance', 'price_field': 'Close', 'adjusted': False,
                'data_start': START, 'data_end_inclusive': LAST_DAY, 'artifacts': artifacts,
                'raw_file_sha256': artifacts['raw_prices.csv']['sha256']}
            write_new(output / 'acquisition_record.json', json_bytes(completed))
            with patch.dict(sys.modules, {'yfinance': None}), patch(__name__ + '.acquisition_environment', side_effect=AssertionError('No environment check/import needed for immutable reuse')):
                reused, was_reused = acquire(root)
            assert was_reused and reused['raw_file_sha256'] == completed['raw_file_sha256']
            checks.append('completed immutable snapshot reused without provider import, requests or package installation')
            (output / 'raw_prices.csv').write_bytes(b'changed')
            rejects('tampered existing acquired snapshot', lambda: acquire(root))
    return {'status': 'ACQUISITION_SELF_TEST_PASS_NO_NETWORK', 'checks': checks,
        'rejections': rejections, 'n_checks': len(checks), 'n_rejections': len(rejections),
        'real_price_requests': 0, 'models_fitted': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=DEFAULT_ROOT)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--self-test', action='store_true')
    modes.add_argument('--acquire', action='store_true', help='Explicitly acquire the locked new-period snapshot once.')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False, allow_nan=False))
        return
    record, reused = acquire(args.project_root)
    output = args.project_root / OUTPUT_REL
    print(json.dumps({'status': record['status'], 'snapshot_reused_without_provider_access': reused,
        'raw_file': str(output / 'raw_prices.csv'), 'raw_file_sha256': record['raw_file_sha256'],
        'acquisition_record': str(output / 'acquisition_record.json'),
        'n_new_period_union_rows': record['coverage']['n_new_period_union_rows'],
        'observed_new_period_rows_by_asset': {a: record['coverage']['assets'][a]['n_observed_in_new_test_period'] for a in ASSETS},
        'analysis_lock_sha256': record['analysis_lock_sha256'], 'models_evaluated': False,
        'old_test_arrays_loaded': False, 'old_project_writes': False}, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
