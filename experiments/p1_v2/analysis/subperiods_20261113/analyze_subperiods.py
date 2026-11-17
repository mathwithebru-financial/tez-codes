"""Default metadata/date preflight; --execute-analysis slices saved predictions only."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import sys
import numpy as np
from numeric_metrics import METRICS, CORE_PATH, metrics, ratio, primary_loss, macro_four, paired, period_masks, reconstruct_additive

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
RUN = BASE/'training/runs/p1_v2_mse_mae_20261113_full'
COMPARISON = RUN/'comparison'
OUT = RUN/'subperiods_20261113'
PROTOCOL = HERE/'subperiod_protocol.json'
PREFLIGHT = HERE/'subperiod_preflight.json'
ASSETS = ['BIST100', 'USDTRY', 'EURTRY', 'GOLD']
SEEDS = [123, 777, 2026]
MODELS = [f'{arm}_seed{seed}' for arm in ['mse', 'mae'] for seed in SEEDS]+['mse_ensemble', 'mae_ensemble', 'matched_naive']
SCOPES = {
    'primary': ('primary_return_mask', 'primary_vol20_mask'),
    'secondary_raw_target': ('secondary_return_raw_target_mask', 'secondary_vol20_raw_target_mask'),
    'diagnostic_all_model_inputs': ('strict_all_model_inputs_return_target_mask', 'strict_all_model_inputs_vol20_target_mask'),
    'diagnostic_asset_inputs': ('strict_asset_input_return_target_mask', 'strict_asset_input_vol20_target_mask'),
}
ARRAY_FILES = {m: 'predictions_raw_test_'+m+'.npy' for m in MODELS if m != 'matched_naive'}
ARRAY_FILES.update(truth='truth_raw_test.npy', dates='target_dates_test.npy', matched_naive='matched_naive_raw_test.npy')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def readj(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def writej(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def readcsv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def writecsv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def header(path):
    # Read an NPY header only: no np.load and no numeric payload access here.
    with path.open('rb') as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, order, dtype = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, order, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError('Unexpected saved NPY version')
    assert not dtype.hasobject and not order
    return list(shape), str(dtype)


def prepare():
    if sys.flags.optimize != 0:
        raise ValueError('Do not run Python with -O: source, date and reconstruction assertions must remain active')
    protocol = readj(PROTOCOL)
    assert [(p['id'], p['start'], p['end']) for p in protocol['disjoint_periods']] == [
        ('2022_partial', '2022-10-06', '2022-12-31'), ('2023', '2023-01-01', '2023-12-31'), ('2024', '2024-01-01', '2024-12-31')]
    assert protocol['full_reference'] == {'id': 'full', 'start': '2022-10-06', 'end': '2024-12-31', 'n_primary_per_asset_task': 584}
    assert protocol['numeric_rules']['quantile_method'] == 'numpy.quantile method=linear'
    assert protocol['populations']['individual_seeds'] == ['primary']
    assert protocol['populations']['ensemble_and_naive'] == list(SCOPES)
    manifest = readj(COMPARISON/'OUTPUT_MANIFEST.json')
    assert manifest['status'] == 'SUCCESS_SIX_FIXED_CHECKPOINTS_EVALUATED_ONCE_NO_TEST_SELECTION'
    lock_path = COMPARISON/'evaluation_input_lock.json'
    assert sha(lock_path) == manifest['files']['evaluation_input_lock.json']['sha256']
    lock = readj(lock_path)
    source_paths = [PROTOCOL, Path(__file__), HERE/'numeric_metrics.py', HERE/'verify_synthetic.py',
                    CORE_PATH, COMPARISON/'OUTPUT_MANIFEST.json', lock_path]
    for name in list(ARRAY_FILES.values())+['metrics_by_asset.csv', 'metrics_equal_asset_macro.csv']:
        p = COMPARISON/name; expected = manifest['files'][name]
        assert sha(p) == expected['sha256'] and p.stat().st_size == expected['size_bytes']
        source_paths.append(p)
    mask_path = BASE/'scoring/scoring_masks_bool.csv'
    assert sha(mask_path) == lock['input_sha256'][str(mask_path)]
    indices_path = BASE/'datasets/p1_v2/original_split_indices_test.npy'
    assert sha(indices_path) == lock['evaluation_input_files']['original_split_indices_test.npy']['sha256']
    source_paths += [mask_path, indices_path]
    rows = readcsv(mask_path)
    by_asset = {a: [r for r in rows if r['split'] == 'test' and r['asset'] == a] for a in ASSETS}
    dates = [r['target_date'] for r in by_asset[ASSETS[0]]]
    assert len(dates) == 584
    for asset in ASSETS:
        assert [r['target_date'] for r in by_asset[asset]] == dates
        assert [r['original_split_sample_index'] for r in by_asset[asset]] == [r['original_split_sample_index'] for r in by_asset[ASSETS[0]]]
        for r in by_asset[asset]:
            assert r['primary_return_mask'] == r['primary_vol20_mask'] == 'True'
    periods = period_masks(dates, protocol['disjoint_periods'], protocol['full_reference'])
    counts = {p: int(mask.sum()) for p, mask in periods.items()}
    assert counts == {'2022_partial': 62, '2023': 260, '2024': 262, 'full': 584}
    for model, name in ARRAY_FILES.items():
        shape, dtype = header(COMPARISON/name)
        assert (shape, dtype) == (([584], '<U10') if model == 'dates' else ([584, 8], 'float64'))
    assert header(indices_path) == ([584], 'int64')
    info = {'status': 'PREFLIGHT_ONLY_NO_PREDICTION_OR_TRUTH_ARRAY_LOAD',
            'protocol_sha256': sha(PROTOCOL), 'source_sha256': {p.relative_to(BASE).as_posix(): sha(p) for p in source_paths},
            'period_counts': counts, 'models': MODELS, 'populations': SCOPES,
            'expected_metric_rows': 576, 'expected_macro_rows': 144,
            'primary_seed_scope_only': True, 'np_load_calls': 0,
            'inference_training_or_checkpoint_loading': False,
            'post_review_boundaries_full_results_previously_known': True}
    return info, protocol, by_asset, dates, periods


def require_same(a, b, label):
    if a is None or b is None:
        assert a is None and b is None, label
    else:
        assert math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-14), label


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute-analysis', action='store_true')
    ap.add_argument('--refresh-preflight', action='store_true')
    args = ap.parse_args()
    assert not OUT.exists(), 'Only a fresh output directory is permitted; original comparison is never overwritten.'
    info, protocol, by_asset, dates, periods = prepare()
    if not args.execute_analysis:
        if PREFLIGHT.exists() and not args.refresh_preflight:
            assert readj(PREFLIGHT)['source_sha256'] == info['source_sha256'], 'Explicit preflight refresh required after source review changes'
        writej(PREFLIGHT, info)
        print(json.dumps({'status': info['status'], 'period_counts': info['period_counts'], 'np_load_calls': 0}))
        return
    assert not args.refresh_preflight
    frozen = readj(PREFLIGHT)
    assert frozen['source_sha256'] == info['source_sha256'], 'Inputs, protocol or code changed after preflight'
    protocol_bytes = PROTOCOL.read_bytes()
    # No numeric prediction/truth payload is accessed above this explicit gate.
    loaded = {k: np.load(COMPARISON/name, allow_pickle=False) for k, name in ARRAY_FILES.items()}
    original_indices = np.load(BASE/'datasets/p1_v2/original_split_indices_test.npy', allow_pickle=False)
    assert loaded.pop('dates').tolist() == dates
    assert original_indices.tolist() == [int(r['original_split_sample_index']) for r in by_asset[ASSETS[0]]]
    truth = loaded.pop('truth')
    for value in [truth]+list(loaded.values()):
        assert value.shape == (584, 8) and value.dtype == np.dtype('float64') and np.isfinite(value).all()
    assert np.all(loaded['matched_naive'][:, :4] == 0)
    for arm in ['mse', 'mae']:
        assert np.array_equal(loaded[arm+'_ensemble'], np.mean(np.stack([loaded[f'{arm}_seed{s}'] for s in SEEDS]), axis=0, dtype=np.float64))
    masks = {(scope, a, task): np.asarray([r[SCOPES[scope][ti]] == 'True' for r in by_asset[a]], dtype=bool)
             for scope in SCOPES for a in ASSETS for ti, task in enumerate(['return', 'volatility'])}
    rows, lookup = [], {}
    for period, pmask in periods.items():
        for model in MODELS:
            for scope in (['primary'] if '_seed' in model else SCOPES):
                for ai, asset in enumerate(ASSETS):
                    for ti, task in enumerate(['return', 'volatility']):
                        j = ai+4*ti; mask = pmask & masks[scope, asset, task]
                        result = metrics(truth[mask, j], loaded[model][mask, j], task)
                        naive = metrics(truth[mask, j], loaded['matched_naive'][mask, j], task)
                        value, status = ratio(primary_loss(result, task), primary_loss(naive, task), result['n'])
                        row = dict(period=period, model=model, population=scope, asset=asset, task=task, **result,
                                   matched_naive_primary_loss=primary_loss(naive, task),
                                   primary_loss_ratio_to_matched_naive=value, ratio_status=status)
                        rows.append(row); lookup[period, model, scope, asset, task] = row
    assert len(rows) == 576
    macros = []
    for period, model, scope, task in dict.fromkeys((r['period'], r['model'], r['population'], r['task']) for r in rows):
        four = [lookup[period, model, scope, a, task] for a in ASSETS]
        macros.append(dict(period=period, model=model, population=scope, task=task, **macro_four(four, ASSETS)))
    assert len(macros) == 144
    pair_rows = []
    for period in periods:
        for pair, mse_model, mae_model in [('ensemble', 'mse_ensemble', 'mae_ensemble')]+[(f'seed{s}', f'mse_seed{s}', f'mae_seed{s}') for s in SEEDS]:
            for scope in (['primary'] if pair.startswith('seed') else SCOPES):
                for asset in ASSETS:
                    for task in ['return', 'volatility']:
                        pair_rows.append(dict(period=period, pair=pair, population=scope, asset=asset, task=task,
                                              **paired(lookup[period, mse_model, scope, asset, task], lookup[period, mae_model, scope, asset, task], task)))
    reconstructions = []
    for row in [r for r in rows if r['period'] == 'full']:
        model, scope, asset, task = (row[k] for k in ['model', 'population', 'asset', 'task'])
        parts = [lookup[p['id'], model, scope, asset, task] for p in protocol['disjoint_periods']]
        result = reconstruct_additive(row, parts, task)
        j = ASSETS.index(asset)+(4 if task == 'volatility' else 0)
        pooled = np.concatenate([np.flatnonzero(periods[p['id']] & masks[scope, asset, task]) for p in protocol['disjoint_periods']])
        pooled_result = metrics(truth[pooled, j], loaded[model][pooled, j], task)
        for key in METRICS:
            require_same(pooled_result[key], row[key], 'Pooled full '+key)
        reconstructions.append(dict(model=model, population=scope, asset=asset, task=task, **result,
                                    pooled_R2_matches_full=True, pooled_quantiles_match_full=True,
                                    reconstruction_passed=True))
    reference_count = 0
    for old in readcsv(COMPARISON/'metrics_by_asset.csv'):
        if old['split'] != 'test':
            continue
        current = lookup['full', old['model'], old['population'], old['asset'], old['task']]
        assert current['n'] == int(old['n'])
        for key in ['MAE', 'MSE', 'RMSE', 'R2', 'Pinball_tau05', 'primary_loss_ratio_to_matched_naive']:
            require_same(current[key], float(old[key]) if old[key] else None, 'Existing full asset '+key)
        reference_count += 1
    assert reference_count == 144
    macro_lookup = {(r['period'], r['model'], r['population'], r['task']): r for r in macros}
    macro_reference_count = 0
    for old in readcsv(COMPARISON/'metrics_equal_asset_macro.csv'):
        if old['split'] != 'test':
            continue
        current = macro_lookup['full', old['model'], old['population'], old['task']]
        for key in ['MAE', 'MSE', 'RMSE', 'R2', 'Pinball_tau05', 'primary_loss_ratio_to_matched_naive']:
            require_same(current[key], float(old[key]) if old[key] else None, 'Existing full macro '+key)
        macro_reference_count += 1
    assert macro_reference_count == 36
    assignments = [{'target_date': date, 'original_split_sample_index': int(original_indices[i]),
                    'period': next(p['id'] for p in protocol['disjoint_periods'] if periods[p['id']][i]),
                    'full_reference': True} for i, date in enumerate(dates)]
    assert PROTOCOL.read_bytes() == protocol_bytes
    for name, digest in info['source_sha256'].items():
        assert sha(BASE/name) == digest, 'An input/source was modified during arithmetic'
    OUT.mkdir()
    (OUT/'subperiod_protocol.json').write_bytes(protocol_bytes)
    for name, values in [('metrics_by_asset.csv', rows), ('metrics_equal_asset_macro.csv', macros),
                         ('paired_arm_differences.csv', pair_rows), ('reconstruction_checks.csv', reconstructions),
                         ('period_assignment.csv', assignments)]:
        writecsv(OUT/name, values)
    receipt = {'status': 'SUCCESS_SAVED_PREDICTIONS_SUBPERIOD_ARITHMETIC_ONLY',
               'protocol_sha256_before_arithmetic': info['protocol_sha256'], 'protocol_bytes_unchanged': True,
               'preflight_sha256': sha(PREFLIGHT), 'source_sha256_before_and_after': info['source_sha256'],
               'all_sources_unchanged': True, 'period_counts': info['period_counts'],
               'metric_rows': len(rows), 'macro_rows': len(macros), 'paired_rows': len(pair_rows),
               'full_reconstruction_rows': len(reconstructions), 'existing_full_asset_rows_matched': reference_count,
               'all_full_reconstructions_passed': True,
               'existing_full_macro_rows_matched': macro_reference_count,
               'saved_ensemble_bitwise_identity_verified': True, 'daily_loss_tables_duplicated': False,
               'original_comparison_modified': False, 'torch_or_arch_imported': 'torch' in sys.modules or 'arch' in sys.modules,
               'model_checkpoint_load_inference_training_or_thesis_edit': False,
               'post_review_analysis_full_test_previously_known': True, 'statistical_inference_performed': False,
               'interpretation_limits': protocol['interpretation_limits']}
    assert not receipt['torch_or_arch_imported']
    writej(OUT/'analysis_receipt.json', receipt)
    writej(OUT/'OUTPUT_MANIFEST.json', {'status': receipt['status'], 'files': {
        p.name: {'sha256': sha(p), 'size_bytes': p.stat().st_size} for p in sorted(OUT.iterdir()) if p.is_file()}})
    print(json.dumps({k: receipt[k] for k in ['status', 'period_counts', 'metric_rows', 'macro_rows', 'paired_rows', 'all_sources_unchanged']}))


if __name__ == '__main__':
    main()
