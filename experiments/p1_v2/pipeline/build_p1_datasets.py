"""Build P1 V2 train-only scalers and complete L10 datasets; no model execution.

The locked validation contract deliberately uses pandas' default C CSV parser.
This is recorded rather than silently switching it to round_trip precision.
Only this standalone data pipeline and sklearn StandardScaler are executed.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import platform
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[1]
ASSETS = ['BIST100', 'USDTRY', 'EURTRY', 'GOLD']
FEATURES = [name for asset in ASSETS for name in (asset+'_LogRet', asset+'_Vol20')]
TARGETS = [asset+'_NextRet' for asset in ASSETS]+[asset+'_NextVol' for asset in ASSETS]
VERSION = 'P1_V2_STRICT_INFORMATION_BOUNDARY'
RAW_SHA = 'ab5f275d38dc98057b1cedcf58019adb26be7402c7ed5ae6ee3d6877b2444893'
COUNTS = {'train': 2712, 'val': 583, 'test': 584}
REASONS = {
    ('train', '2010-02-15'): 'INSUFFICIENT_INITIAL_FEATURE_HISTORY',
    ('train', '2020-07-09'): 'TRAIN_LABEL_AFTER_FIRST_VALIDATION_INFO_CUTOFF',
    ('val', '2022-10-05'): 'SELECTION_LABEL_AFTER_FIRST_TEST_INFO_CUTOFF',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype='<f8').tobytes()).hexdigest()


def dates_digest(dates):
    return hashlib.sha256(('\n'.join(map(str, dates))+'\n').encode('utf-8')).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def derive(raw):
    """Forward fill only, then float64 returns and sample rolling volatility."""
    clean = raw.ffill()
    assert not clean.isna().any().any() and (clean > 0).all().all()
    returns = np.log(clean/clean.shift(1))
    vol = returns.rolling(20, min_periods=20).std(ddof=1)*np.sqrt(252.)
    feature = np.column_stack([values for j in range(4)
                               for values in (returns.iloc[:, j], vol.iloc[:, j])]).astype(np.float64)
    # Indexed by the actual target realization date u, not by an input date.
    target = np.column_stack([returns.to_numpy(), vol.to_numpy()]).astype(np.float64)
    return clean, returns, vol, feature, target


def scaler_record(scaler, order, domain, kind):
    return {
        'class': 'sklearn.preprocessing.StandardScaler',
        'with_mean': True, 'with_std': True, 'variance_ddof': 0,
        'mean': scaler.mean_.tolist(), 'var': scaler.var_.tolist(),
        'scale': scaler.scale_.tolist(), 'n_samples_seen': int(scaler.n_samples_seen_),
        'input_dtype': 'float64', 'stats_dtype': str(scaler.mean_.dtype),
        'feature_order' if kind == 'x' else 'target_order': order,
        'fit_domain': domain,
    }


def scalar_stats(array):
    """Independent Python fsum mean and population variance, column by column."""
    n = len(array)
    means = [math.fsum(float(v) for v in array[:, j])/n for j in range(array.shape[1])]
    variances = [math.fsum((float(v)-means[j])**2 for v in array[:, j])/n
                 for j in range(array.shape[1])]
    return np.asarray(means), np.asarray(variances), np.sqrt(variances)


def scalar_prices_and_features(raw):
    """Different fill/log/std implementation over the declared parsed raw floats."""
    n = len(raw)
    values = raw.to_numpy(dtype=np.float64)
    filled = np.empty_like(values)
    for j in range(4):
        previous = None
        for i in range(n):
            value = float(values[i, j])
            if math.isnan(value):
                assert previous is not None
                value = previous
            filled[i, j] = value
            previous = value
    ret = np.full_like(filled, np.nan)
    vol = np.full_like(filled, np.nan)
    for j in range(4):
        for u in range(1, n):
            ret[u, j] = math.log(float(filled[u, j])/float(filled[u-1, j]))
        for u in range(20, n):
            window = [float(v) for v in ret[u-19:u+1, j]]
            mean = math.fsum(window)/20
            vol[u, j] = math.sqrt(math.fsum((v-mean)**2 for v in window)/19)*math.sqrt(252)
    features = np.column_stack([values for j in range(4) for values in (ret[:, j], vol[:, j])])
    targets = np.column_stack([ret, vol])
    return filled, features, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=BASE)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    base = args.project_root.resolve()
    out = args.output_dir.resolve() if args.output_dir else base/'datasets/p1_v2'
    out.mkdir(parents=True, exist_ok=True)
    raw_path = base/'inputs/raw_prices.csv'
    map_path = base/'protocol/proposed_lag1_date_map.csv'
    lock_path = base/'benchmarks/validation/selection_denominators_P1_validation_v2.json'
    source_paths = [raw_path, map_path, lock_path,
                    base/'inputs/data_source_manifest.json',
                    base/'validation/data_integrity_checks.json',
                    base/'protocol/strict_information_boundary_validation.json',
                    base/'pipeline/source_snapshots/01_rebuild_from_frozen_raw_v4.py',
                    base/'pipeline/source_snapshots/02_preprocessing_v4.py']
    source_hashes = {p.relative_to(base).as_posix(): sha(p) for p in source_paths}
    assert source_hashes['inputs/raw_prices.csv'] == RAW_SHA
    builder_sha = sha(__file__)
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    assert lock['version'] == 2 and lock['validation_rows'] == 583 and lock['output_order'] == TARGETS
    assert lock['source_hashes']['inputs/raw_prices.csv'] == RAW_SHA
    assert lock['source_hashes']['protocol/proposed_lag1_date_map.csv'] == sha(map_path)

    # Precision is locked by the existing float64 score contract, not guessed.
    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True,
                      dtype={a: 'float64' for a in ASSETS})
    dm = pd.read_csv(map_path)
    assert raw.shape == (3912, 4) and list(raw.columns) == ASSETS
    assert raw.index.is_monotonic_increasing and raw.index.is_unique
    assert raw.index.tz is None and (raw.index.dayofweek < 5).all()
    assert raw.isna().sum().to_dict() == dict(zip(ASSETS, [153, 5, 3, 141]))
    assert len(dm) == 3882 and not dm.duplicated(['split', 'original_split_sample_index']).any()
    assert dm.proposed_eligible.dtype == bool
    assert dm.training_or_selection_eligible.equals(dm.proposed_eligible)
    assert dm.groupby('split').proposed_eligible.sum().to_dict() == COUNTS
    excluded = dm[~dm.proposed_eligible]
    assert {(r.split, r.target_date): r.exclusion_reason for r in excluded.itertuples()} == REASONS
    assert not dm.proposal_executed_as_model_experiment.any()
    dates = np.asarray(raw.index.strftime('%Y-%m-%d'), dtype='<U10')
    position = {str(date): i for i, date in enumerate(dates)}
    clean, returns, vol, features, targets = derive(raw)
    scalar_clean, scalar_features, scalar_targets = scalar_prices_and_features(raw)
    assert np.array_equal(scalar_clean, clean.to_numpy())
    scalar_ret_diff = float(np.max(np.abs(scalar_targets[20:, :4]-targets[20:, :4])))
    scalar_vol_diff = float(np.max(np.abs(scalar_targets[20:, 4:]-targets[20:, 4:])))
    assert scalar_ret_diff < 1e-14 and scalar_vol_diff < 1e-12

    # Paired rows: anchor a -> X=F[a-1], Y=(r,V)[a+1].
    # Fit before L10 construction, preserving the specified original convention.
    last_train_target = dm.loc[(dm.split == 'train') & dm.proposed_eligible, 'target_date'].iloc[-1]
    fit_target_end = position[last_train_target]
    paired_anchors = np.arange(21, fit_target_end, dtype=np.int64)
    fit_x_positions, fit_y_positions = paired_anchors-1, paired_anchors+1
    assert len(paired_anchors) == 2721
    assert (dates[fit_x_positions[0]], dates[fit_x_positions[-1]]) == ('2010-02-01', '2020-07-06')
    assert (dates[fit_y_positions[0]], dates[fit_y_positions[-1]]) == ('2010-02-03', '2020-07-08')
    fit_X, fit_y = features[fit_x_positions].copy(), targets[fit_y_positions].copy()
    assert fit_X.shape == fit_y.shape == (2721, 8)
    assert np.isfinite(fit_X).all() and np.isfinite(fit_y).all()
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    assert not hasattr(x_scaler, 'mean_') and not hasattr(y_scaler, 'mean_')
    x_scaler.fit(fit_X)
    y_scaler.fit(fit_y)
    assert int(x_scaler.n_samples_seen_) == int(y_scaler.n_samples_seen_) == 2721
    assert (x_scaler.scale_ > 0).all() and (y_scaler.scale_ > 0).all()
    independent_stats = {}
    for kind, scaler, values in [('x', x_scaler, fit_X), ('y', y_scaler, fit_y)]:
        mean, variance, scale = scalar_stats(values)
        assert np.allclose(mean, scaler.mean_, rtol=1e-12, atol=1e-14)
        assert np.allclose(variance, scaler.var_, rtol=1e-12, atol=1e-14)
        assert np.allclose(scale, scaler.scale_, rtol=1e-12, atol=1e-14)
        independent_stats[kind] = {
            'mean_max_abs_difference': float(np.max(np.abs(mean-scaler.mean_))),
            'variance_max_abs_difference': float(np.max(np.abs(variance-scaler.var_))),
            'scale_max_abs_difference': float(np.max(np.abs(scale-scaler.scale_))),
            'n_rows': len(values), 'method': 'Python math.fsum, population variance ddof0',
        }

    # A genuine causal-prefix perturbation: alter every present raw suffix price
    # after the last permitted training label, recompute fill/features, and refit.
    perturbed = raw.copy()
    perturbed.iloc[fit_target_end+1:] = perturbed.iloc[fit_target_end+1:]*np.array([1.013, .973, 1.027, 1.041])
    _, _, _, pert_features, pert_targets = derive(perturbed)
    assert np.array_equal(pert_features[fit_x_positions], fit_X)
    assert np.array_equal(pert_targets[fit_y_positions], fit_y)
    assert not np.array_equal(pert_targets[fit_target_end+1:], targets[fit_target_end+1:], equal_nan=True)
    perturb_x = StandardScaler().fit(pert_features[fit_x_positions])
    perturb_y = StandardScaler().fit(pert_targets[fit_y_positions])
    for original, changed in [(x_scaler, perturb_x), (y_scaler, perturb_y)]:
        for attr in ['mean_', 'var_', 'scale_']:
            assert np.array_equal(getattr(original, attr), getattr(changed, attr))
        assert int(original.n_samples_seen_) == int(changed.n_samples_seen_)
    # Separate explicit row-selection probe, without using sklearn's selection.
    injected_x, injected_y = features.copy(), targets.copy()
    outside_x = np.ones(len(raw), dtype=bool); outside_x[fit_x_positions] = False
    outside_y = np.ones(len(raw), dtype=bool); outside_y[fit_y_positions] = False
    injected_x[outside_x] = 1e9
    injected_y[outside_y] = -1e9
    for original_values, injected, positions in [(fit_X, injected_x, fit_x_positions), (fit_y, injected_y, fit_y_positions)]:
        assert np.array_equal(original_values, injected[positions])
        for a, b in zip(scalar_stats(original_values), scalar_stats(injected[positions])):
            assert np.array_equal(a, b)

    arrays = {}
    def save_array(name, values, *, split=None, role):
        values = np.ascontiguousarray(values)
        assert not values.dtype.hasobject
        np.save(out/name, values, allow_pickle=False)
        reopened = np.load(out/name, allow_pickle=False)
        assert reopened.dtype == values.dtype and reopened.shape == values.shape
        assert np.array_equal(reopened, values, equal_nan=values.dtype.kind in 'fc')
        arrays[name] = {'sha256': sha(out/name), 'size_bytes': (out/name).stat().st_size,
                        'shape': list(values.shape), 'dtype': str(values.dtype),
                        'split': split, 'role': role}

    fit_domains = {
        'x': {'row_count': 2721, 'first_label': str(dates[fit_x_positions[0]]),
              'last_label': str(dates[fit_x_positions[-1]]), 'label_kind': 'input_feature_date',
              'position_array': 'fit_x_raw_positions.npy', 'date_array': 'fit_x_feature_dates.npy',
              'values_array': 'fit_X_raw.npy', 'fit_split': 'train_prewindow_only'},
        'y': {'row_count': 2721, 'first_label': str(dates[fit_y_positions[0]]),
              'last_label': str(dates[fit_y_positions[-1]]), 'label_kind': 'target_realization_date',
              'position_array': 'fit_y_raw_positions.npy', 'date_array': 'fit_y_target_dates.npy',
              'values_array': 'fit_y_raw.npy', 'fit_split': 'train_prewindow_only'},
    }
    scalers = {
        'protocol_version': VERSION, 'feature_order': FEATURES, 'output_order': TARGETS,
        'implementation': 'sklearn.preprocessing.StandardScaler', 'sklearn_version': sklearn.__version__,
        'fit_convention': 'Paired rows a=21..last_primary_training_target_position-1; X=F[a-1], y=T[a+1]. Fit before windows.',
        'precision': 'float64 fit and transform; float32 cast only after window and target scaling',
        'x': scaler_record(x_scaler, FEATURES, fit_domains['x'], 'x'),
        'y': scaler_record(y_scaler, TARGETS, fit_domains['y'], 'y'),
        'inverse_prediction_rule': 'scaled_prediction.astype(float64)*y.scale(float64)+y.mean(float64)',
        'fitted_objects_serialized': False, 'pickle_created_or_loaded': False,
    }
    write_json(out/'scalers.json', scalers)
    for kind, scaler in [('x', x_scaler), ('y', y_scaler)]:
        for name, value in [('mean', scaler.mean_), ('var', scaler.var_), ('scale', scaler.scale_)]:
            save_array(f'{kind}_scaler_{name}.npy', value, role='float64 train-only scaler statistic')
        save_array(f'{kind}_scaler_n_samples_seen.npy', np.asarray(scaler.n_samples_seen_, dtype=np.int64), role='train-only scaler fitted row count')
    for name, values, role in [
        ('fit_X_raw.npy', fit_X, 'float64 prewindow X fit matrix'),
        ('fit_y_raw.npy', fit_y, 'float64 prewindow y fit matrix'),
        ('fit_anchor_dates.npy', dates[paired_anchors], 'paired prewindow anchor date labels'),
        ('fit_x_feature_dates.npy', dates[fit_x_positions], 'X scaler fit feature date labels'),
        ('fit_y_target_dates.npy', dates[fit_y_positions], 'y scaler fit target date labels'),
        ('fit_x_raw_positions.npy', fit_x_positions, 'X scaler source raw index positions'),
        ('fit_y_raw_positions.npy', fit_y_positions, 'y scaler source raw index positions'),
    ]:
        save_array(name, values, role=role)

    # Transform only finite rows. The undefined leading history is kept as NaN
    # in memory and can never enter an eligible or diagnostic window.
    scaled_features = np.full_like(features, np.nan)
    valid_features = np.isfinite(features).all(axis=1)
    scaled_features[valid_features] = x_scaler.transform(features[valid_features])
    scaled_targets = np.full_like(targets, np.nan)
    valid_targets = np.isfinite(targets).all(axis=1)
    scaled_targets[valid_targets] = y_scaler.transform(targets[valid_targets])
    for scaler, values in [(x_scaler, fit_X), (y_scaler, fit_y)]:
        transformed = scaler.transform(values)
        assert np.max(np.abs(np.mean(transformed, axis=0))) < 1e-12
        assert np.max(np.abs(np.var(transformed, axis=0)-1)) < 1e-12

    # Keep every old map row, with an explicit position in active arrays or an
    # exclusion/diagnostic role. No silent row deletion or relabeling occurs.
    index_table = dm.copy()
    index_table.insert(0, 'original_global_sample_index', np.arange(len(dm), dtype=np.int64))
    index_table['dataset_group'] = 'excluded_without_window'
    index_table['dataset_row'] = -1
    index_table.loc[(index_table.split == 'train') & ~index_table.proposed_eligible & index_table.feature_window_eligible, 'dataset_group'] = 'train_boundary_excluded'
    sample_validation = {}
    all_window_price_cells = 0
    for split in ['train', 'val', 'test', 'val_boundary_diagnostic']:
        diagnostic = split == 'val_boundary_diagnostic'
        selected = dm.loc[dm.boundary_diagnostic_only] if diagnostic else dm.loc[(dm.split == split) & dm.proposed_eligible]
        assert len(selected) == (1 if diagnostic else COUNTS[split])
        target_pos = np.asarray([position[d] for d in selected.target_date], dtype=np.int64)
        input_end_pos = target_pos-2
        window_pos = input_end_pos[:, None] + np.arange(-9, 1, dtype=np.int64)[None, :]
        X64 = scaled_features[window_pos]
        y64 = targets[target_pos].copy()
        y_scaled64 = scaled_targets[target_pos]
        assert X64.shape == (len(selected), 10, 8) and y64.shape == (len(selected), 8)
        assert np.isfinite(X64).all() and np.isfinite(y64).all() and np.isfinite(y_scaled64).all()
        assert np.all(window_pos[:, -1] < target_pos-1) and np.all(window_pos >= 20)
        assert dates[window_pos[:, 0]].tolist() == selected.proposed_lag1_input_first.tolist()
        assert dates[window_pos[:, -1]].tolist() == selected.proposed_lag1_input_last.tolist()
        # Independent date-list reconstruction and numeric scalar validation.
        scalar_window_max = 0.
        scalar_target_max = 0.
        for i, row in enumerate(selected.itertuples()):
            u = position[row.target_date]
            scalar_dates = [str(dates[u-offset]) for offset in range(11, 1, -1)]
            assert scalar_dates == dates[window_pos[i]].tolist()
            assert scalar_dates[0] == row.proposed_lag1_input_first
            assert scalar_dates[-1] == row.proposed_lag1_input_last
            assert row.target_date not in scalar_dates
            assert row.return_target_start_label == str(dates[u-1])
            assert row.return_target_end_label == str(dates[u])
            assert row.vol20_first_return_label == str(dates[u-19])
            assert row.vol20_last_return_label == str(dates[u])
            # Union of actual f-20..f price supports across the ten features.
            support = sorted({p for f in range(u-11, u-1) for p in range(f-20, f+1)})
            assert support == list(range(u-31, u-1)) and len(support) == 30
            assert max(support) == u-2 and u not in support and u-1 not in support
            scalar_x = (scalar_features[window_pos[i]]-x_scaler.mean_)/x_scaler.scale_
            scalar_window_max = max(scalar_window_max, float(np.max(np.abs(scalar_x-X64[i]))))
            scalar_target_max = max(scalar_target_max, float(np.max(np.abs(scalar_targets[u]-y64[i]))))
            all_window_price_cells += len(support)*4
        assert scalar_window_max < 1e-10 and scalar_target_max < 1e-12
        X32 = X64.astype(np.float32)
        ys32 = y_scaled64.astype(np.float32)
        inverse_y32 = ys32.astype(np.float64)*y_scaler.scale_+y_scaler.mean_
        inverse_x32 = X32.astype(np.float64)*x_scaler.scale_+x_scaler.mean_
        original_X = features[window_pos]
        # Per-value bounds for rounding to float32, including operation roundoff.
        y_bounds = 2*np.finfo(np.float32).eps*np.maximum(1., np.abs(y_scaled64))*y_scaler.scale_+1e-14
        x_bounds = 2*np.finfo(np.float32).eps*np.maximum(1., np.abs(X64))*x_scaler.scale_+1e-14
        assert np.all(np.abs(inverse_y32-y64) <= y_bounds)
        assert np.all(np.abs(inverse_x32-original_X) <= x_bounds)
        assert np.max(np.abs(y_scaler.inverse_transform(y_scaled64)-y64)) < 1e-12
        save_array(f'X_{split}.npy', X32, split=split, role='scaled L10x8 input; float32')
        save_array(f'y_{split}_scaled.npy', ys32, split=split, role='scaled8 target; float32')
        save_array(f'y_{split}_raw.npy', y64, split=split, role='canonical raw scoring truth; float64')
        save_array(f'target_dates_{split}.npy', dates[target_pos], split=split, role='target realization date labels; no clock timestamp')
        save_array(f'input_end_dates_{split}.npy', dates[input_end_pos], split=split, role='last admitted feature date labels; no clock timestamp')
        save_array(f'input_start_dates_{split}.npy', dates[window_pos[:, 0]], split=split, role='first feature date labels')
        save_array(f'input_feature_dates_{split}.npy', dates[window_pos], split=split, role='complete ten feature date labels')
        save_array(f'original_split_indices_{split}.npy', selected.original_split_sample_index.to_numpy(np.int64), split=split, role='unchanged original split sample index')
        save_array(f'original_global_indices_{split}.npy', selected.index.to_numpy(np.int64), split=split, role='unchanged original map row index')
        save_array(f'target_raw_positions_{split}.npy', target_pos, split=split, role='source raw row index for target')
        save_array(f'input_end_raw_positions_{split}.npy', input_end_pos, split=split, role='source raw row index for last admitted feature')
        save_array(f'primary_scope_eligible_{split}.npy', np.full(len(selected), not diagnostic, dtype=bool), split=split, role='primary dataset/scoring scope; does not authorize test use for training or selection; diagnostic alwaysFalse')
        save_array(f'boundary_diagnostic_only_{split}.npy', np.full(len(selected), diagnostic, dtype=bool), split=split, role='post-selection-only validation boundary flag')
        index_table.loc[selected.index, 'dataset_group'] = split
        index_table.loc[selected.index, 'dataset_row'] = np.arange(len(selected))
        sample_validation[split] = {
            'count': len(selected), 'input_shape': list(X32.shape), 'target_shape': list(y64.shape),
            'first_target_date': str(dates[target_pos[0]]), 'last_target_date': str(dates[target_pos[-1]]),
            'first_input_end_date': str(dates[input_end_pos[0]]), 'last_input_end_date': str(dates[input_end_pos[-1]]),
            'target_dates_sha256': dates_digest(dates[target_pos]),
            'truth_sha256_float64_little_endian': numeric_digest(y64),
            'all_window_dates_and_source_price_supports_checked': True,
            'scalar_scaled_window_max_abs_difference': scalar_window_max,
            'scalar_raw_target_max_abs_difference': scalar_target_max,
            'inverse_scaled_float32_y_max_abs_error': float(np.max(np.abs(inverse_y32-y64))),
            'inverse_scaled_float32_X_max_abs_error': float(np.max(np.abs(inverse_x32-original_X))),
            'float32_inverse_within_per_value_bounds': True,
            'primary_dataset_scope': not diagnostic,
            'permitted_training_role': split == 'train',
            'permitted_checkpoint_selection_role': split == 'val',
            'actually_used_by_any_model': False,
        }

    assert sample_validation['val']['target_dates_sha256'] == lock['validation_target_dates_sha256']
    assert sample_validation['val']['truth_sha256_float64_little_endian'] == lock['scoring_truth_sha256_float64_little_endian']
    # Actual unique training feature rows exactly equal the prewindow X fit domain.
    active_train = dm[(dm.split == 'train') & dm.proposed_eligible]
    training_window_union = sorted({position[r.target_date]-offset for r in active_train.itertuples() for offset in range(2, 12)})
    assert np.array_equal(training_window_union, fit_x_positions)
    assert sample_validation['train']['last_target_date'] <= sample_validation['val']['first_input_end_date']
    assert sample_validation['val']['last_target_date'] <= sample_validation['test']['first_input_end_date']
    assert len(index_table) == 3882 and index_table.dataset_group.value_counts().to_dict() == {
        'train': 2712, 'val': 583, 'test': 584, 'val_boundary_diagnostic': 1,
        'excluded_without_window': 1, 'train_boundary_excluded': 1}
    index_table.to_csv(out/'all_original_sample_index.csv', index=False, encoding='utf-8-sig')
    examples = []
    for target_day in ['2010-02-16', '2020-07-10', '2022-10-05', '2022-10-06', '2022-10-10']:
        row = index_table[index_table.target_date == target_day].iloc[0]
        u = position[target_day]
        examples.append({
            'target_date': target_day, 'original_split': row.split,
            'dataset_group': row.dataset_group, 'dataset_row': int(row.dataset_row),
            'primary_scope_eligible': bool(row.proposed_eligible),
            'boundary_diagnostic_only': bool(row.boundary_diagnostic_only),
            'anchor_t_label': str(dates[u-1]), 'input_end_label': str(dates[u-2]),
            'input_feature_labels': dates[u-11:u-1].tolist(),
            'input_price_support_first': str(dates[u-31]),
            'input_price_support_last': str(dates[u-2]), 'input_price_rows_per_asset': 30,
            'return_target_price_labels': dates[u-1:u+1].tolist(),
            'vol20_target_price_support_first': str(dates[u-20]),
            'vol20_target_price_support_last': str(dates[u]),
            'raw_target_in_output_order_float64': targets[u].tolist(),
            'target_not_in_input_feature_or_price_support': True,
            'clock_availability_verified': False,
        })
    write_json(out/'window_examples.json', {'feature_order': FEATURES, 'output_order': TARGETS, 'examples': examples})
    (out/'README.txt').write_text(
        'P1 V2 — eğitim öncesi ölçekleyici ve tam pencere veri paketi\n\n'
        'pipeline/build_p1_datasets.py gerçekten çalıştırılmıştır. Gerçek sklearn '
        'StandardScaler fit ve veri dizisi üretimi yapılmıştır; model eğitimi, '
        'öğrenilmiş tahmin, GARCH uyarlaması veya tez düzenlemesi yapılmamıştır. '
        'Model kaynağı/checkpoint/pickle yüklenmemiştir. Özgün girdilerin hash\'leri değişmemiştir.\n\n'
        'Paket kökünde çalıştırma: python pipeline/build_p1_datasets.py\n'
        'Gerçek ortam sürümleri dataset_metadata.json içindedir.\n\n'
        'Aktif diziler: train 2712, val 583, test 584. X float32[N,10,8]; '
        'y_scaled float32[N,8]; y_raw float64[N,8]. Özellikler varlık başına '
        'LogRet/Vol20 sırasıyla iç içedir. Çıktılar önce dört getiri, sonra '
        'dört volatilitedir; kesin adlar metadata ve scalers JSON içinde verilmiştir.\n\n'
        'Ölçekleyici pencerelemeden önce 2721 eşlenmiş eğitim satırına fit edilir: '
        'X etiketleri 2010-02-01–2020-07-06, Y hedef etiketleri '
        '2010-02-03–2020-07-08. Bu sayı 2712 denetimli pencere sayısı değildir. '
        'Örtüşen pencerelerde tekrarlanan hücrelere fit yapılmamıştır. '
        'X fit alanı aktif eğitim pencerelerinin benzersiz özellik satırı birleşimine eşittir. '
        'Y fit alanı, belirtilen pencere-öncesi eşleme geleneğini korur. '
        'Doğrulama/test değerleri fit için kullanılmaz.\n\n'
        'scalers.json ve x/y_scaler_*.npy dosyaları float64 mean/var/scale '
        'ile n_samples_seen bilgisini saklar. Var ddof=0\'dır. Fit_X_raw ve fit_y_raw '
        '2721 satırlık fit matrisleridir; tarih/konum dizileri alanı açıklar. '
        'Gelecekteki ölçekli tahminin ham dönüşümü: '
        'prediction.astype(float64)*y.scale+y.mean. Puanlamada y_raw float64 '
        'kullanılır; float32 hedef veya ters çevrilmiş eğitim hedefi bunun yerine geçmez.\n\n'
        'Ham seri forward fill sonrası log(P/P_previous) ile getiriye; '
        '20 getirinin örnek standart sapması (ddof=1)*sqrt252 ile Vol20\'ye çevrilir. '
        'Kilitli V2 doğrulama sözleşmesi nedeniyle CSV pandas varsayılan C-parser '
        'hassasiyetiyle okunur; round_trip parser\'ına sessiz geçiş yapılmaz. '
        '583 doğrulama hedefinin ham float64 veri hash\'i ve tarih hash\'i '
        'önceden kilitlenen sözleşmeyle tam eşleşmiştir.\n\n'
        'Girdi F[t-10]..F[t-1], hedef r[t+1]/Vol20[t+1]\'dir. '
        'Girdi fiyat desteği t-30..t-1; getiri hedefi fiyatları t,t+1; '
        'Vol20 hedefi fiyatları t-19..t+1\'dir. Hedef başlangıcından girdi sonuna '
        'olan fark saklanmaz: son girdiden hedef etiketine iki ortak gözlem vardır. '
        'Bu, birikimli iki adımlı getiri hedefi anlamına gelmez.\n\n'
        'all_original_sample_index.csv, 3882 eski satırın tamamını nedenleriyle korur. '
        'İlk eğitim hedefi 2010-02-15 yetersiz geçmiş; 2020-07-09 eğitim sınırı '
        'nedeniyle dışarıdadır. 2022-10-05 doğrulama hedefi yalnız '
        'val_boundary_diagnostic adlı ayrı bir satırdır; checkpoint/model seçimi '
        'için kullanılmaz. Test 584 tarih olarak korunmuştur ve eğitim/seleksiyon '
        'için uygun sayılmaz. Primary_scope bayrağı yalnız ilgili ana veri/puanlama kapsamıdır.\n\n'
        'validation_report.json bağımsız scalar fiyat/getiri/volatilite, fsum '
        'istatistikleri, tüm pencere tarih/fiyat destekleri, float32 ters dönüşüm '
        'toleransları ve gelecek ham fiyatları değiştirme denetimlerini kaydeder. '
        'Gelecek değerleri değiştirip yeniden fit etmek mean/var/scale/nobs '
        'değerlerini bit düzeyinde değiştirmemiştir. Değiştirilmiş veri kaydedilmemiştir. '
        'Bütün NPY dosyaları allow_pickle=False ile yazılmış ve okunmuştur.\n\n'
        'Tarihler günlük ortak etiketlerdir; gerçek yayın veya seçilmiş tahmin '
        'saatleri değildir. Bir gözlem geriye alma tarihi erişilebilirlik varsayımıdır; '
        'saat düzeyinde sızıntısızlık veya canlı işlem sertifikası değildir. '
        'Hamda mevcut fiyat da yeni geçerli işlem kanıtı değildir. Ayrı scoring '
        'maskeleri ve BIST 8 Şubat 2023 olay bayrakları bu paket tarafından '
        'yeniden yorumlanmaz. P1 karşılaştırması bilgi, sınır örneklemi ve '
        'fit edilmiş ölçeklemenin ortak değişimidir; saf gecikme etkisi diye sunulmamalıdır.\n',
        encoding='utf-8')

    # Freeze source identities after all actual fits/transforms/checks.
    after_hashes = {p.relative_to(base).as_posix(): sha(p) for p in source_paths}
    assert source_hashes == after_hashes and builder_sha == sha(__file__)
    evidence = {
        'status': 'PASS_P1_V2_TRAIN_ONLY_SCALERS_AND_FULL_WINDOWS',
        'protocol_version': VERSION, 'source_hashes_before': source_hashes,
        'source_hashes_after': after_hashes, 'sources_unchanged': True,
        'scaler_fit_objects_initially_unfitted': True,
        'actual_implementation': 'sklearn.preprocessing.StandardScaler',
        'n_samples_seen_x_y': [2721, 2721], 'fit_domains': fit_domains,
        'independent_fsum_statistics': independent_stats,
        'scalar_ffill_matches_pandas_bitwise': True,
        'scalar_all_3912_rows_return_max_abs_difference': scalar_ret_diff,
        'scalar_all_3912_rows_vol20_max_abs_difference': scalar_vol_diff,
        'future_raw_suffix_perturbation': {
            'first_mutated_raw_date': str(dates[fit_target_end+1]),
            'method': 'All present suffix prices multiplied per asset; NaN locations preserved; ffill/features recomputed and both scalers refitted.',
            'suffix_values_actually_changed': True, 'fit_matrices_bitwise_unchanged': True,
            'refitted_mean_var_scale_n_seen_bitwise_unchanged': True,
            'perturbed_data_saved': False,
        },
        'independent_outside_fit_row_injection': {
            'all_nonfit_feature_rows': '1e9', 'all_nonfit_target_rows': '-1e9',
            'selected_matrices_and_independent_fsum_stats_bitwise_unchanged': True,
        },
        'all_window_source_price_cells_checked': all_window_price_cells,
        'sample_validation': sample_validation,
        'locked_validation_truth_and_dates_exact_match': True,
        'training_input_window_union_exactly_equals_X_fit_domain': True,
        'fit_before_window_not_fit_on_repeated_overlapping_windows': True,
        'all_original_map_rows_retained': 3882,
        'validation_boundary_diagnostic_not_in_primary_arrays': True,
        'all_npy_written_and_reloaded_allow_pickle_false': True,
        'no_torch_or_arch_imported': 'torch' not in sys.modules and 'arch' not in sys.modules,
        'actions': {'scaler_fit': True, 'full_window_dataset_created': True,
                    'model_training': False, 'model_inference': False, 'GARCH_fit': False,
                    'checkpoint_or_pickle_load': False, 'original_source_import_or_execution': False,
                    'network_acquisition': False, 'thesis_edit': False},
        'clock_availability_verified': False, 'clock_leak_count': None,
    }
    assert evidence['no_torch_or_arch_imported']
    write_json(out/'validation_report.json', evidence)
    metadata = {
        'status': evidence['status'], 'protocol_version': VERSION,
        'builder': 'pipeline/build_p1_datasets.py', 'builder_sha256': builder_sha,
        'split_counts': COUNTS, 'lookback': 10, 'n_features': 8, 'n_targets': 8,
        'feature_order': FEATURES, 'output_order': TARGETS, 'assets': ASSETS,
        'target_definition': {'return': 'log(Ptilde[t+1]/Ptilde[t])',
                              'vol20': 'sqrt252*sample_std(r[t-18]..r[t+1]), ddof1, 20returns'},
        'input_definition': 'F[t-10]..F[t-1]; latest feature t-1; unchanged targets at t+1',
        'raw_numeric_parser': 'pandas.read_csv default C-parser precision, float64 per asset, matching lockedV2validation contract',
        'raw_price_filling': 'forward fill only; no backfill',
        'feature_calculation_dtype': 'float64', 'training_array_dtype': 'float32',
        'raw_scoring_truth_dtype': 'float64', 'date_array_dtype': '<U10, date labels only',
        'scalers_file': 'scalers.json', 'scalers_sha256': sha(out/'scalers.json'),
        'fit_domains': fit_domains, 'arrays': arrays,
        'source_sha256': {'raw': sha(raw_path), 'map': sha(map_path), 'locked_denominators': sha(lock_path)},
        'source_file_sha256': source_hashes, 'source_files_unchanged': True,
        'locked_validation_contract_file': 'benchmarks/validation/selection_denominators_P1_validation_v2.json',
        'validation_truth_digest': sample_validation['val']['truth_sha256_float64_little_endian'],
        'validation_target_dates_digest': sample_validation['val']['target_dates_sha256'],
        'diagnostic_arrays_prefix': 'val_boundary_diagnostic',
        'diagnostic_target_date': '2022-10-05',
        'diagnostic_training_or_selection_eligible': False,
        'all_original_map_rows_file': 'all_original_sample_index.csv',
        'validation_report_file': 'validation_report.json',
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'pandas': pd.__version__, 'scikit_learn': sklearn.__version__},
        'limitations': [
            'Dates are common daily labels, not verified publication timestamps or a selected prediction hour.',
            'Lagging by one common observation is a declared availability assumption, not a historical clock-safety certificate.',
            'Primary targets retain the existing forward-filled series. Raw presence is not proof of a new valid trading event.',
            'P1 changes admitted input, boundary scope and fitted scaling jointly; it is not an isolated pure-delay comparison with P0.',
            'Training, learned prediction, GARCH fitting and thesis editing were not performed.',
        ],
        'actions': evidence['actions'],
    }
    write_json(out/'dataset_metadata.json', metadata)
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != 'artifact_manifest.json')
    write_json(out/'artifact_manifest.json', {
        'status': 'PASS', 'protocol_version': VERSION, 'source_sha256': source_hashes,
        'files': [{'file_name': p.name, 'size_bytes': p.stat().st_size, 'sha256': sha(p)} for p in files],
        'n_files_excluding_manifest': len(files), 'total_bytes_excluding_manifest': sum(p.stat().st_size for p in files),
    })
    print(json.dumps({'status': evidence['status'], 'split_counts': COUNTS,
                      'fit_rows_x_y': [2721, 2721],
                      'validation_truth_matches_lock': True,
                      'array_files': len(arrays), 'sources_unchanged': True,
                      'actions': evidence['actions']}))


if __name__ == '__main__':
    main()
