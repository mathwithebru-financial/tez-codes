"""Pure float64 descriptive arithmetic; no I/O, fitting, or model libraries."""
import math
import hashlib
import importlib.util
from pathlib import Path
import numpy as np

CORE_PATH = Path(__file__).resolve().parents[2]/'training/comparison/numeric_metrics.py'
CORE_SHA256 = '00d51c0a06ed5321ec1ab89abaa67d962653d7d492e79ee65b26a9e454b3ebbc'
if hashlib.sha256(CORE_PATH.read_bytes()).hexdigest() != CORE_SHA256:
    raise ValueError('Existing pure numeric core SHA mismatch')
_spec = importlib.util.spec_from_file_location('p1_existing_pure_numeric_core', CORE_PATH)
CORE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CORE)

METRICS = ('MAE', 'MSE', 'RMSE', 'R2', 'Pinball_tau05',
           'median_absolute_error', 'p90_absolute_error',
           'median_primary_loss', 'p90_primary_loss')


def metrics(truth, prediction, task):
    if task not in ('return', 'volatility'):
        raise ValueError('Unknown task')
    y = np.asarray(truth, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.ndim != 1 or y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Matching finite one-dimensional arrays required')
    n = len(y)
    result = dict(n=n, population_status='DEFINED' if n else 'UNDEFINED_EMPTY_POPULATION',
                  R2_status='UNDEFINED_EMPTY_POPULATION' if not n else 'UNDEFINED_INSUFFICIENT_OBSERVATIONS')
    result.update({key: None for key in METRICS})
    if not n:
        return result
    result.update(CORE.metrics(y, p, task=task))
    absolute = np.abs(y-p)
    result.update(median_absolute_error=float(np.quantile(absolute, .5, method='linear')),
                  p90_absolute_error=float(np.quantile(absolute, .9, method='linear')))
    factor = .5 if task == 'volatility' else 1.
    result['median_primary_loss'] = factor*result['median_absolute_error']
    result['p90_primary_loss'] = factor*result['p90_absolute_error']
    return result


def ratio(value, denominator, n):
    if not n or value is None or denominator is None:
        return None, 'UNDEFINED_EMPTY_POPULATION'
    if denominator == 0:
        return None, 'UNDEFINED_ZERO_DENOMINATOR'
    if not math.isfinite(value) or not math.isfinite(denominator) or denominator < 0:
        raise ValueError('Invalid nonnegative evaluation denominator')
    return value/denominator, 'DEFINED'


def primary_loss(record, task):
    return record['MAE' if task == 'return' else 'Pinball_tau05']


def macro_four(records, assets):
    if len(records) != 4 or {r['asset'] for r in records} != set(assets) or len(set(assets)) != 4:
        raise ValueError('Exactly one record for each of four assets required')
    out = {'aggregation': 'equal_weight_mean_of_exactly_four_asset_metrics',
           'n_by_asset': ';'.join(f'{a}:{next(r["n"] for r in records if r["asset"] == a)}' for a in assets)}
    for key in METRICS+('primary_loss_ratio_to_matched_naive',):
        values = [r[key] for r in records]
        defined = all(v is not None and math.isfinite(v) for v in values)
        out[key] = math.fsum(values)/4 if defined else None
        out[key+'_status'] = 'DEFINED_ALL_FOUR_ASSETS' if defined else 'UNDEFINED_AT_LEAST_ONE_ASSET'
    return out


def paired(mse_metrics, mae_metrics, task):
    if mse_metrics['n'] != mae_metrics['n']:
        raise ValueError('Paired populations differ')
    a = primary_loss(mse_metrics, task)
    b = primary_loss(mae_metrics, task)
    n = mse_metrics['n']
    change_ratio, status = ratio(None if a is None or b is None else b-a, a, n)
    return {'n': n, 'MSE_arm_primary_loss': a, 'MAE_arm_primary_loss': b,
            'mean_MSE_minus_MAE_primary_loss': None if a is None or b is None else a-b,
            'relative_change_MAE_minus_MSE_percent': None if change_ratio is None else 100*change_ratio,
            'relative_change_status': status,
            'negative_relative_change_favors': 'MAE', 'inferential_test_performed': False}


def period_masks(dates, periods, full):
    dates = np.asarray(dates)
    if dates.ndim != 1 or len(set(dates.tolist())) != len(dates) or dates.tolist() != sorted(dates.tolist()):
        raise ValueError('Unique sorted one-dimensional date labels required')
    if len({p['id'] for p in periods}) != len(periods):
        raise ValueError('Duplicate period IDs')
    coverage = np.zeros(len(dates), dtype=np.int64)
    result = {}
    for p in periods:
        if p['start'] > p['end']:
            raise ValueError('Reversed period')
        mask = (dates >= p['start']) & (dates <= p['end'])
        result[p['id']] = mask
        coverage += mask
    full_mask = (dates >= full['start']) & (dates <= full['end'])
    if not full_mask.all() or not np.all(coverage == 1):
        raise ValueError('Disjoint periods must cover every full-reference date exactly once')
    result[full['id']] = full_mask
    return result


def reconstruct_additive(full, parts, task):
    n = full['n']
    if sum(p['n'] for p in parts) != n:
        raise ValueError('Partition counts do not reconstruct full population')
    out = {'n_full': n, 'n_period_sum': sum(p['n'] for p in parts),
           'reconstructed_Pinball_tau05': None, 'Pinball_tau05_absolute_difference': None,
           'R2_and_quantiles_reconstruction': 'Recompute on pooled observations; never average period values'}
    for key in ('MAE', 'MSE')+ (('Pinball_tau05',) if task == 'volatility' else ()):
        rebuilt = math.fsum(p['n']*p[key] for p in parts if p['n'])/n if n else None
        difference = None if not n else abs(rebuilt-full[key])
        if difference is not None and not math.isclose(rebuilt, full[key], rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError('Weighted period reconstruction failed: '+key)
        out['reconstructed_'+key] = rebuilt
        out[key+'_absolute_difference'] = difference
    rebuilt_rmse = math.sqrt(out['reconstructed_MSE']) if n else None
    out['reconstructed_RMSE_from_MSE'] = rebuilt_rmse
    if n and not math.isclose(rebuilt_rmse, full['RMSE'], rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError('RMSE reconstruction failed')
    return out
