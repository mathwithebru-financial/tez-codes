"""Saved-array SHAP stability arithmetic. Default preflight never opens NPZ values."""
from pathlib import Path
import argparse
import csv
import hashlib
import itertools
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
RUN = BASE/'training/runs/p1_v2_mse_mae_20261113_full'
RESULTS = RUN/'shap_stability_20261113'
OUT = RESULTS/'stability'
PREFLIGHT = HERE/'saved_array_analysis_preflight.json'
METRICS_SHA = 'c3c31bd8f935b1505d573a47d67e21590fc93adb6bba443701ddcb36aa065588'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if sha(HERE/'stability_metrics.py') != METRICS_SHA:
    raise ValueError('Reviewed pure stability metrics source identity changed')
import stability_metrics as sm

PAIR_METRICS = ['spearman_midrank', 'kendall_tau_b', 'top3_jaccard', 'share_total_variation',
                'share_L1', 'normalized_L1', 'minimum_boundary_gap_share']
CORE_KEYS = ('phi_scaled', 'phi_raw', 'expected_scaled', 'expected_raw', 'prediction_scaled', 'prediction_raw')


def readj(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def writej(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def writecsv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def conditions(protocol):
    result = []
    for arm, seed, bg, mc in itertools.product(protocol['loss_arms'], protocol['model_seeds']+['ensemble'], protocol['backgrounds'], protocol['integration_seeds']):
        model = arm+('_ensemble' if seed == 'ensemble' else '_seed'+str(seed))
        result.append({'condition': f'{model}__{bg}__mc{mc}', 'arm': arm, 'model': model,
                       'model_seed': seed, 'model_scope': 'ensemble' if seed == 'ensemble' else 'individual_seed',
                       'background': bg, 'integration_seed': mc})
    assert len(result) == 64
    return result


def prepare():
    if sys.flags.optimize != 0:
        raise ValueError('Run without -O: source and array checks must stay active')
    plan_path = HERE/'locked_sample_plan.json'
    plan = readj(plan_path)
    assert sha(plan_path) == '184be7c13835fa7c48d8aea0be5d97652ee65395e4518daeba05524013d6ff12'
    scalers_path = BASE/'datasets/p1_v2/scalers.json'
    assert sha(scalers_path) == plan['source_sha256']['datasets/p1_v2/scalers.json']
    sources = {p.relative_to(BASE).as_posix(): sha(p) for p in [plan_path, scalers_path, HERE/'stability_metrics.py', Path(__file__)]}
    lock_path = RESULTS/'analysis_input_lock.json'
    if not lock_path.is_file():
        return ({'status': 'WAITING_FOR_ACTUAL_MAIN_INPUT_LOCK_AND_64_SAVED_CONDITIONS',
                 'source_sha256': sources, 'protocol_sha256': None, 'protocol_file': None,
                 'sample_plan_sha256': sha(plan_path), 'saved_conditions': 0,
                 'expected_output_pairs': 1408, 'expected_task_pairs': 352, 'np_load_calls': 0,
                 'model_torch_shap_import_or_execution': False}, None, plan, readj(scalers_path), [])
    lock = readj(lock_path)
    # Resolve the ACTUAL completed run's protocol, including a preserved V2
    # addendum. Never assume that an old pilot or a fixed V1 filename applies.
    if 'protocol_path' in lock:
        candidates = [Path(lock['protocol_path']).resolve()]
    else:
        candidates = [Path(name).resolve() for name, digest in lock['source_sha256'].items()
                      if digest == lock['protocol_sha256'] and Path(name).suffix == '.json']
    assert len(candidates) == 1 and candidates[0].parent == HERE
    protocol_path = candidates[0]
    assert sha(protocol_path) == lock['protocol_sha256']
    protocol = readj(protocol_path)
    assert sha(plan_path) == protocol['sample_plan_sha256']
    assert plan['explained']['n'] == 128 and plan['integration_seeds'] == protocol['integration_seeds']
    assert protocol['loss_arms'] == ['mse', 'mae'] and protocol['model_seeds'] == [123, 777, 2026]
    assert protocol['nsamples'] == 256 and protocol['background_rows_each'] == 64
    assert protocol['output_count'] == 8 and len(protocol['backgrounds']) == 4
    sources[protocol_path.relative_to(BASE).as_posix()] = sha(protocol_path)
    ready = (RESULTS/'analysis_receipt.json').is_file() and (RESULTS/'OUTPUT_MANIFEST.json').is_file()
    jobs = conditions(protocol)
    if ready:
        receipt, manifest = readj(RESULTS/'analysis_receipt.json'), readj(RESULTS/'OUTPUT_MANIFEST.json')
        assert receipt['status'] == manifest['status'] == 'SUCCESS_SHAP_INPUT_GRADIENT_STABILITY_ARRAYS_SAVED'
        assert receipt['model_jobs'] == 48 and receipt['ensemble_jobs'] == 16 and receipt['explained_n'] == 128
        assert receipt['nsamples'] == 256 and receipt['backgrounds'] == 4 and receipt['background_size'] == 64
        assert receipt['integration_seeds'] == protocol['integration_seeds']
        assert receipt['all_sources_unchanged'] and receipt['all_model_states_unchanged'] and not receipt['training_optimizer_or_fit']
        assert sha(lock_path) == receipt['input_lock_sha256']
        for p in [RESULTS/'analysis_receipt.json', lock_path]:
            assert sha(p) == manifest['files'][p.name]['sha256'] and p.stat().st_size == manifest['files'][p.name]['size_bytes']
        assert lock['protocol_sha256'] == sha(protocol_path) and lock['sample_plan_sha256'] == sha(plan_path)
        for p in [protocol_path, plan_path, scalers_path, HERE/'stability_metrics.py']:
            assert sha(p) == lock['source_sha256'][str(p)]
        for path in [RESULTS/'OUTPUT_MANIFEST.json', RESULTS/'analysis_receipt.json', lock_path]:
            sources[path.relative_to(BASE).as_posix()] = sha(path)
        for job in jobs:
            for suffix in ['.json', '.npz']:
                path = RESULTS/(job['condition']+suffix)
                record = manifest['files'][path.name]
                assert sha(path) == record['sha256'] and path.stat().st_size == record['size_bytes']
                sources[path.relative_to(BASE).as_posix()] = sha(path)
            record = readj(RESULTS/(job['condition']+'.json'))
            assert record['status'] == 'COMPLETE' and record['npz_sha256'] == sha(RESULTS/(job['condition']+'.npz'))
            assert record['input_lock_sha256'] == sha(lock_path)
            for key in ['arm', 'model_seed', 'background', 'integration_seed']:
                assert record[key] == job[key]
            if job['model_seed'] != 'ensemble':
                assert record['model_state_unchanged'] and record['parameter_gradients_none'] and record['nsamples'] == 256
    info = {'status': 'READY_SAVED_SHAP_ARRAYS_NO_NUMERIC_LOAD' if ready else 'WAITING_FOR_48_MODEL_AND_16_ENSEMBLE_ARRAYS',
            'source_sha256': sources, 'protocol_sha256': sha(protocol_path), 'sample_plan_sha256': sha(plan_path),
            'protocol_file': protocol_path.relative_to(BASE).as_posix(),
            'saved_conditions': 64 if ready else 0, 'expected_output_pairs': 1408, 'expected_task_pairs': 352,
            'np_load_calls': 0, 'model_torch_shap_import_or_execution': False}
    return info, protocol, plan, readj(scalers_path), jobs


def flatten_comparison(value):
    row = {k: value[k] for k in ['spearman_midrank', 'kendall_tau_b', 'rank_status', 'normalized_L1', 'share_L1', 'share_total_variation', 'share_status', 'top3_jaccard', 'top3_jaccard_depends_on_boundary_tie_break']}
    for side in ['left', 'right']:
        for key in ['top3', 'rank_order', 'total_importance', 'exact_tie_at_top3_boundary', 'third_fourth_gap', 'third_fourth_gap_share', 'tied_feature_pairs', 'top3_status', 'top3_without_boundary_tie_break_identifiable']:
            v = value[side][key]
            row[side+'_'+key] = json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
    gaps = [value[s]['third_fourth_gap_share'] for s in ['left', 'right']]
    row['minimum_boundary_gap_share'] = min(gaps) if all(v is not None for v in gaps) else None
    return row


def pair_specs(protocol, jobs):
    by_key = {(j['arm'], j['model_seed'], j['background'], j['integration_seed']): j for j in jobs}
    bgs, mcs, seeds = protocol['backgrounds'], protocol['integration_seeds'], protocol['model_seeds']
    specs = []
    for arm in protocol['loss_arms']:
        for seed, mc in itertools.product(seeds+['ensemble'], mcs):
            for left, right in itertools.combinations(bgs, 2):
                specs.append(('background', by_key[arm, seed, left, mc], by_key[arm, seed, right, mc]))
        for bg, mc in itertools.product(bgs, mcs):
            for left, right in itertools.combinations(seeds, 2):
                specs.append(('model_seed', by_key[arm, left, bg, mc], by_key[arm, right, bg, mc]))
        for seed, bg in itertools.product(seeds+['ensemble'], bgs):
            specs.append(('integration_MC', by_key[arm, seed, bg, mcs[0]], by_key[arm, seed, bg, mcs[1]]))
    assert {f: sum(item[0] == f for item in specs) for f in ['background', 'model_seed', 'integration_MC']} == {'background': 96, 'model_seed': 48, 'integration_MC': 32}
    return specs


def summaries(rows, unit):
    out = []
    groups = dict.fromkeys((r['factor'], r['arm'], r['model_scope'], r[unit]) for r in rows)
    for factor, arm, model_scope, label in groups:
        group = [r for r in rows if (r['factor'], r['arm'], r['model_scope'], r[unit]) == (factor, arm, model_scope, label)]
        for metric in PAIR_METRICS:
            values = [r[metric] for r in group if r[metric] is not None]
            out.append({'factor': factor, 'arm': arm, 'model_scope': model_scope, unit: label, 'metric': metric,
                        'n_pairs': len(group), 'n_valid': len(values), 'n_undefined': len(group)-len(values),
                        'minimum': min(values) if values else None,
                        'median': float(np.quantile(values, .5, method='linear')) if values else None,
                        'maximum': max(values) if values else None,
                        'pairs_with_top3_boundary_tie': sum(r['top3_jaccard_depends_on_boundary_tie_break'] for r in group),
                        'pairs_are_independent_replicates': False})
    return out


def verify_synthetic():
    plan = readj(HERE/'locked_sample_plan.json')
    protocol = {'loss_arms': ['mse', 'mae'], 'model_seeds': plan['model_training_seeds'],
                'backgrounds': [b['id'] for b in plan['backgrounds']], 'integration_seeds': plan['integration_seeds']}
    jobs = conditions(protocol)
    specs = pair_specs(protocol, jobs)
    for factor, left, right in specs:
        assert left['arm'] == right['arm']
        if factor == 'background':
            assert left['model_seed'] == right['model_seed'] and left['integration_seed'] == right['integration_seed'] and left['background'] != right['background']
        elif factor == 'model_seed':
            assert left['background'] == right['background'] and left['integration_seed'] == right['integration_seed'] and left['model_seed'] != right['model_seed'] and left['model_seed'] != 'ensemble' and right['model_seed'] != 'ensemble'
        else:
            assert left['background'] == right['background'] and left['model_seed'] == right['model_seed'] and left['integration_seed'] != right['integration_seed']
    features = ['f'+str(i) for i in range(8)]
    vector = np.arange(1., 9.)
    toy = []
    for a, b in [(vector, vector), (vector, vector[::-1]), (np.zeros(8), np.zeros(8))]:
        toy.append(dict(factor='toy', arm='mse', model_scope='individual_seed', output='toy_output', **flatten_comparison(sm.compare_importance(a, b, features))))
    summary = summaries(toy, 'output')
    assert all(r['n_pairs'] == 3 and r['n_valid'] == 2 and r['n_undefined'] == 1 for r in summary)
    rank = next(r for r in summary if r['metric'] == 'spearman_midrank')
    assert rank['minimum'] == -1 and rank['median'] == 0 and rank['maximum'] == 1
    phi = np.broadcast_to(vector.reshape(1, 1, 8, 1), (4, 10, 8, 8)).copy()
    scale = np.asarray([1., 2., 3., 4., .1, .2, .3, .4])
    raw = phi*scale.reshape(1, 1, 1, 8)
    task = sm.standardized_task_importance(sm.feature_importance(raw/scale.reshape(1, 1, 1, 8)))
    assert np.allclose(task, np.stack([vector, vector]), rtol=1e-14, atol=1e-14)
    ensemble = sm.ensemble_attributions(np.stack([phi, -phi, np.zeros_like(phi)]))
    assert np.all(sm.feature_importance(ensemble) == 0)
    assert np.any(np.mean(np.stack([sm.feature_importance(v) for v in [phi, -phi, np.zeros_like(phi)]]), axis=0) > 0)
    import io
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(toy[0])); writer.writeheader(); writer.writerows(toy)
    assert len(list(csv.DictReader(io.StringIO(stream.getvalue())))) == 3
    assert 'torch' not in sys.modules and 'shap' not in sys.modules
    result = {'status': 'PASS_SAVED_ARRAY_DRIVER_SYNTHETIC_ARITHMETIC', 'conditions': 64,
              'condition_pairs': len(specs), 'output_pairs': len(specs)*8, 'task_pairs': len(specs)*2,
              'other_factors_held_fixed': True, 'zero_importance_undefined_counts_preserved': True,
              'standardized_task_aggregation_verified': True, 'ensemble_signed_mean_before_abs_verified': True,
              'mixed_defined_undefined_csv_serialization_passed': True,
              'actual_saved_phi_or_model_values_read': False, 'source_sha256': sha(__file__)}
    writej(HERE/'saved_array_analysis_synthetic_verification.json', result)
    print(json.dumps(result))


def main():
    global RESULTS, OUT
    if sys.flags.optimize != 0:
        raise ValueError('Run without -O: validation assertions must stay active')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-analysis', action='store_true')
    parser.add_argument('--refresh-preflight', action='store_true')
    parser.add_argument('--verify-synthetic', action='store_true')
    parser.add_argument('--results-dir', type=Path, default=RESULTS,
                        help='Actual saved-array run directory; its input lock determines the protocol.')
    args = parser.parse_args()
    RESULTS = args.results_dir.resolve(); OUT = RESULTS/'stability'
    assert RESULTS.is_relative_to(RUN) and RESULTS != RUN
    if args.verify_synthetic:
        assert not args.execute_analysis and not args.refresh_preflight
        verify_synthetic(); return
    assert not OUT.exists(), 'Only a fresh stability child directory may be created'
    info, protocol, plan, scalers, jobs = prepare()
    if not args.execute_analysis:
        if PREFLIGHT.exists() and not args.refresh_preflight and readj(PREFLIGHT)['status'].startswith('READY'):
            assert readj(PREFLIGHT)['source_sha256'] == info['source_sha256']
        writej(PREFLIGHT, info); print(json.dumps({k: info[k] for k in ['status', 'saved_conditions', 'np_load_calls']})); return
    assert not args.refresh_preflight and info['status'] == 'READY_SAVED_SHAP_ARRAYS_NO_NUMERIC_LOAD'
    assert readj(PREFLIGHT)['source_sha256'] == info['source_sha256'], 'Refresh and review completed-array preflight first'
    features, outputs = plan['feature_order'], plan['output_order']
    mean, scale = np.asarray(scalers['y']['mean'], dtype=np.float64), np.asarray(scalers['y']['scale'], dtype=np.float64)
    cache, importance, tasks = {}, {}, {}
    global_rows, temporal_rows, task_rows, additivity_rows = [], [], [], []
    for job in jobs:
        with np.load(RESULTS/(job['condition']+'.npz'), allow_pickle=False) as file:
            assert all(file[key].dtype == np.dtype('float64') for key in CORE_KEYS)
            a = {key: np.asarray(file[key], dtype=np.float64) for key in CORE_KEYS}
            if job['model_seed'] != 'ensemble':
                assert file['explained_test_indices'].tolist() == plan['explained']['indices']
                assert file['background_train_indices'].tolist() == next(b['indices'] for b in plan['backgrounds'] if b['id'] == job['background'])
        assert a['phi_scaled'].shape == a['phi_raw'].shape == (128, 10, 8, 8)
        assert a['prediction_scaled'].shape == a['prediction_raw'].shape == (128, 8)
        assert a['expected_scaled'].shape == a['expected_raw'].shape == (8,)
        assert all(np.isfinite(value).all() for value in a.values())
        converted = sm.affine_output_transform(a['phi_scaled'], a['expected_scaled'], a['prediction_scaled'], mean, scale)
        assert all(np.allclose(value, a[key], rtol=1e-12, atol=1e-14) for value, key in zip(converted, ['phi_raw', 'expected_raw', 'prediction_raw']))
        cache[job['condition']] = a
        raw, standardized = sm.feature_importance(a['phi_raw']), sm.feature_importance(a['phi_scaled'])
        raw_temporal, std_temporal = sm.temporal_importance(a['phi_raw']), sm.temporal_importance(a['phi_scaled'])
        task = sm.standardized_task_importance(sm.feature_importance(a['phi_raw']/scale.reshape(1, 1, 1, 8)))
        importance[job['condition']], tasks[job['condition']] = raw, task
        for j, output in enumerate(outputs):
            self_comparison = sm.compare_importance(raw[j], raw[j], features)
            rank = {label: i+1 for i, label in enumerate(self_comparison['left']['rank_order'])}
            shares = self_comparison['left_shares']
            for k, feature in enumerate(features):
                global_rows.append(dict(job, output_index=j, output=output, feature_index=k, feature=feature,
                                        importance_raw=float(raw[j, k]), importance_standardized=float(standardized[j, k]),
                                        relative_feature_share=None if shares is None else shares[k], share_status=self_comparison['share_status'],
                                        rank_with_feature_order_tie_break=rank[feature], third_fourth_gap_share=self_comparison['left']['third_fourth_gap_share'],
                                        exact_tie_at_top3_boundary=self_comparison['left']['exact_tie_at_top3_boundary']))
            total_lag = float(np.sum(raw_temporal[j], dtype=np.float64))
            for lag, label in enumerate(protocol['attribution_contract']['lag_labels_oldest_to_newest']):
                temporal_rows.append(dict(job, output_index=j, output=output, lag_index=lag, lag=label,
                                          importance_raw=float(raw_temporal[j, lag]), importance_standardized=float(std_temporal[j, lag]),
                                          relative_lag_share=float(raw_temporal[j, lag]/total_lag) if total_lag > 0 else None,
                                          share_status='DEFINED' if total_lag > 0 else 'UNDEFINED_ZERO_TOTAL_IMPORTANCE'))
        for t, label in enumerate(['return', 'volatility']):
            total = float(np.sum(task[t], dtype=np.float64))
            for k, feature in enumerate(features):
                task_rows.append(dict(job, task=label, feature_index=k, feature=feature,
                                      importance_standardized_mean_four_outputs=float(task[t, k]),
                                      relative_feature_share=float(task[t, k]/total) if total > 0 else None,
                                      share_status='DEFINED' if total > 0 else 'UNDEFINED_ZERO_TOTAL_IMPORTANCE'))
        saved_residual = a['prediction_raw']-a['expected_raw']-np.sum(a['phi_raw'], axis=(1, 2), dtype=np.float64)
        affine_residual = converted[2]-converted[1]-np.sum(converted[0], axis=(1, 2), dtype=np.float64)
        for row in sm.additivity_diagnostics(a['phi_scaled'], a['prediction_scaled'], a['expected_scaled'], mean, scale, outputs):
            additivity_rows.append(dict(job, **row, saved_raw_vs_affine_residual_max_difference=float(np.max(np.abs(saved_residual[:, row['output_index']]-affine_residual[:, row['output_index']])))))
    for job in [j for j in jobs if j['model_seed'] == 'ensemble']:
        seed_values = [cache[f'{job["arm"]}_seed{s}__{job["background"]}__mc{job["integration_seed"]}'] for s in protocol['model_seeds']]
        for key in CORE_KEYS:
            assert np.array_equal(cache[job['condition']][key], np.mean(np.stack([a[key] for a in seed_values]), axis=0, dtype=np.float64)), 'Saved ensemble mean differs'
    output_pairs, task_pairs = [], []
    for factor, left, right in pair_specs(protocol, jobs):
        meta = {'factor': factor, 'arm': left['arm'], 'model_scope': left['model_scope'],
                'left_condition': left['condition'], 'right_condition': right['condition']}
        for j, output in enumerate(outputs):
            output_pairs.append(dict(meta, output_index=j, output=output, task='return' if j < 4 else 'volatility',
                                     **flatten_comparison(sm.compare_importance(importance[left['condition']][j], importance[right['condition']][j], features))))
        for t, task in enumerate(['return', 'volatility']):
            task_pairs.append(dict(meta, task=task, **flatten_comparison(sm.compare_importance(tasks[left['condition']][t], tasks[right['condition']][t], features))))
    assert [len(v) for v in [global_rows, temporal_rows, task_rows, additivity_rows, output_pairs, task_pairs]] == [4096, 5120, 1024, 512, 1408, 352]
    output_summary, task_summary = summaries(output_pairs, 'output'), summaries(task_pairs, 'task')
    for name, digest in info['source_sha256'].items():
        assert sha(BASE/name) == digest, 'Input/source changed during saved-array arithmetic'
    assert 'torch' not in sys.modules and 'shap' not in sys.modules and 'arch' not in sys.modules
    OUT.mkdir()
    for name, rows in [('global_feature_importance.csv', global_rows), ('temporal_importance.csv', temporal_rows),
                       ('standardized_task_feature_importance.csv', task_rows), ('local_additivity_diagnostics.csv', additivity_rows),
                       ('output_pairwise_stability.csv', output_pairs), ('task_pairwise_stability.csv', task_pairs),
                       ('output_factor_summaries.csv', output_summary), ('task_factor_summaries.csv', task_summary)]:
        writecsv(OUT/name, rows)
    summary = {'status': 'SUCCESS_SAVED_SHAP_STABILITY_DESCRIPTIVE_ARITHMETIC',
               'scope': {'conditions': 64, 'models': 6, 'derived_ensembles': 2, 'explained_n': 128, 'available_test_n': 584, 'backgrounds': 4, 'background_n_each': 64, 'integration_repeats': 2},
               'output_pair_count': 1408, 'task_pair_count': 352, 'output_factor_summaries': output_summary, 'task_factor_summaries': task_summary,
               'global_feature_aggregation': 'mean(abs(phi), axes=(sample,lag)); not sum_lag',
               'task_aggregation': 'phi_raw / training_y_scale, mean(abs) over sample/lag, then mean four standardized outputs; no raw-unit pooling',
               'saved_ensemble_six_core_arrays_bitwise_mean_verified': True,
               'additivity_rows': 512, 'standardized_additivity_mean_abs_range': [min(r['raw_over_y_scale_residual_mean_abs'] for r in additivity_rows), max(r['raw_over_y_scale_residual_mean_abs'] for r in additivity_rows)],
               'standardized_additivity_max_abs': max(r['raw_over_y_scale_residual_max_abs'] for r in additivity_rows),
               'stability_pass_fail_threshold': None, 'near_tie_binary_threshold': None, 'p_values_or_independent_pair_claims': False,
               'library_variance_is_standard_error_or_confidence_interval': False,
               'protocol_sha256': info['protocol_sha256'], 'sample_plan_sha256': info['sample_plan_sha256'],
               'protocol_file_resolved_from_actual_input_lock': info['protocol_file'],
               'source_sha256_before_and_after': info['source_sha256'], 'all_sources_unchanged': True,
               'new_SHAP_model_prediction_or_training': False, 'checkpoint_model_or_Torch_loaded': False,
               'thesis_edit': False, 'interpretation_limits': protocol['interpretation_limits']}
    writej(OUT/'SHAP_Kararlilik_Ozeti.json', summary)
    lines = ['SHAP kararlılığı — kaydedilmiş dizilerin betimsel karşılaştırması', '',
             '128/584 sabit test penceresi; eğitimden alınan dört 64 satırlık arka plan; iki MC entegrasyon tohumu. Altı model ve aynı koşuldaki üç tohum ortalamasından türetilen iki ansambl kullanıldı.',
             'Eğitim tohumları, arka plan örnekleme tohumları ve MC tohumları farklı rollerdir. Arka plan değişimi referans sorusunu da değiştirir.',
             'Özellik önemi örnek ve gecikme boyunca |phi| ortalamasıdır. Görev özetleri dört çıktıyı eğitim ölçeğiyle standartlaştırdıktan sonra ortalar. Ansambl için önce işaretli phi ortalanmıştır.', '',
             'Her çıktı ve görev için min/medyan/maksimum ile geçerli/tanımsız çift sayıları CSV/JSON tablolarındadır. Çiftler ortak modelleri ve örnekleri paylaştığından bağımsız tekrar sayılmaz.',
             '1408 çıktı çifti: arka plan 768, model tohumu 384, MC 256. Ayrıca 352 standartlaştırılmış görev çifti vardır. Evrensel kararlı/kararsız eşiği veya yeni p-değeri kullanılmadı.', '', 'Çıktılar birlikte özetlendiğinde (aynı faktör/kol/model türü içinde):']
    for factor, arm, scope in dict.fromkeys((r['factor'], r['arm'], r['model_scope']) for r in output_pairs):
        group = [r for r in output_pairs if (r['factor'], r['arm'], r['model_scope']) == (factor, arm, scope)]
        parts = []
        for metric in ['spearman_midrank', 'top3_jaccard', 'share_total_variation', 'normalized_L1']:
            v = [r[metric] for r in group if r[metric] is not None]
            parts.append(metric+': '+(f'medyan {np.median(v):.4g}, aralık [{min(v):.4g}, {max(v):.4g}], n={len(v)}/{len(group)}' if v else 'tanımsız'))
        lines.append(f'{factor} / {arm} / {scope}: '+'; '.join(parts))
    lines += ['', f"Yerel toplamsallık 512 koşul-çıktıda incelendi. Standartlaştırılmış ortalama mutlak artık aralığı {summary['standardized_additivity_mean_abs_range']}; en büyük mutlak artık {summary['standardized_additivity_max_abs']:.6g}. Sonlu MC'de tam eşitlik beklenmez; artık düzeltmesi veya sonuç sonrası eleme yapılmadı.",
              'Tam sıralama eşitlikleri, üçüncü-dördüncü özellik pay farkları ve bağ çözümüne duyarlı top3 durumları tabloda açık tutuldu. Küçük farklar için sonradan ikili yakınlık eşiği seçilmedi.',
              'SHAP kütüphanesinin varyans çıktısı standart hata veya güven aralığı olarak yorumlanmadı.',
              'Bulgular seçilen 128 pencere ve dört ampirik eğitim arka planına koşulludur; tüm 584 tarih, nedensellik veya tahmin üstünlüğü kanıtı değildir. P1 tarihsel erişilebilirlik saatleri hâlâ doğrulanmamıştır.',
              'Bu aşama yalnız mevcut dizilerde CPU aritmetiğidir: yeni SHAP, model/tahmin, eğitim veya tez düzenlemesi yapılmadı.']
    (OUT/'SHAP_Kararlilik_Ozeti.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    writej(OUT/'stability_analysis_receipt.json', {k: v for k, v in summary.items() if k not in ['output_factor_summaries', 'task_factor_summaries']})
    writej(OUT/'OUTPUT_MANIFEST.json', {'status': summary['status'], 'files': {p.name: {'sha256': sha(p), 'size_bytes': p.stat().st_size} for p in sorted(OUT.iterdir()) if p.is_file()}})
    print(json.dumps({k: summary[k] for k in ['status', 'output_pair_count', 'task_pair_count', 'additivity_rows', 'all_sources_unchanged']}))


if __name__ == '__main__':
    main()
