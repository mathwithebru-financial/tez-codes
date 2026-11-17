"""Lock date/index-only SHAP sampling; never load X values, y, or model weights."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
DATASET = BASE/'datasets/p1_v2'
PLAN = HERE/'locked_sample_plan.json'
BACKGROUND_SEEDS = (2026111301, 2026111302, 2026111303)
DEFAULT_INTEGRATION_SEEDS = (2026111321, 2026111322)
MODEL_SEEDS = (123, 777, 2026)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json_immutable(path, value):
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError('Existing locked sampling artifact differs; do not overwrite: '+str(path))
    else:
        path.write_bytes(payload)


def header(path):
    with path.open('rb') as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError('Unsupported NPY header version')
    if dtype.hasobject or fortran:
        raise ValueError('Unexpected X dtype/storage order')
    return list(shape), str(dtype)


def checked_source_metadata():
    if sys.flags.optimize != 0:
        raise ValueError('Source identity assertions require Python without -O')
    meta_path = DATASET/'dataset_metadata.json'
    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    assert metadata['protocol_version'] == 'P1_V2_STRICT_INFORMATION_BOUNDARY'
    assert metadata['split_counts'] == {'train': 2712, 'val': 583, 'test': 584}
    assert len(metadata['feature_order']) == len(metadata['output_order']) == 8
    sources = {meta_path: sha(meta_path)}
    names = ['X_train.npy', 'X_test.npy']
    names += [f'{stem}_{split}.npy' for split in ['train', 'test']
              for stem in ['target_dates', 'input_end_dates', 'original_split_indices']]
    for name in names:
        p = DATASET/name
        record = metadata['arrays'][name]
        assert sha(p) == record['sha256'] and p.stat().st_size == record['size_bytes']
        assert header(p) == (record['shape'], record['dtype'])
        sources[p] = record['sha256']
    scalers = DATASET/'scalers.json'
    assert sha(scalers) == metadata['scalers_sha256']
    sources[scalers] = sha(scalers)
    sources[Path(__file__).resolve()] = sha(__file__)
    # Read only safe label/index arrays; feature values and target values are not loaded.
    labels = {}
    for split, n in [('train', 2712), ('test', 584)]:
        labels[split] = {}
        for stem in ['target_dates', 'input_end_dates', 'original_split_indices']:
            v = np.load(DATASET/f'{stem}_{split}.npy', allow_pickle=False)
            assert v.shape == (n,)
            assert v.dtype == (np.dtype('int64') if stem == 'original_split_indices' else np.dtype('<U10'))
            assert v.tolist() == sorted(set(v.tolist()))
            labels[split][stem] = v
        assert np.all(labels[split]['input_end_dates'] < labels[split]['target_dates'])
    return metadata, sources, labels


def write_csv_immutable(path, rows):
    import io
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    payload = buffer.getvalue().encode('utf-8-sig')
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError('Existing sampling CSV differs; do not overwrite')
    else:
        path.write_bytes(payload)


def make_plan(explained_n, integration_seeds):
    metadata, sources, labels = checked_source_metadata()
    assert explained_n in (96, 128)
    assert len(integration_seeds) == len(set(integration_seeds)) == 2
    assert not set(integration_seeds) & set(BACKGROUND_SEEDS+MODEL_SEEDS)
    assert not set(BACKGROUND_SEEDS) & set(MODEL_SEEDS)
    explained = np.linspace(0, 583, explained_n, dtype=np.int64)
    reference = np.linspace(0, 2711, 64, dtype=np.int64)
    backgrounds = [{'id': 'linspace_reference', 'rng_seed': None, 'method': 'numpy.linspace(0,2711,64,dtype=int64)', 'indices': reference.tolist()}]
    for i, seed in enumerate(BACKGROUND_SEEDS, 1):
        # Explicit generator prevents a future default_rng algorithm change.
        indices = np.sort(np.random.Generator(np.random.PCG64(seed)).choice(2712, 64, replace=False))
        backgrounds.append({'id': f'random_bg_{i}', 'rng_seed': seed,
                            'method': 'Generator(PCG64(seed)).choice(2712,64,replace=False), sorted',
                            'indices': indices.tolist()})
    assert len(set(explained.tolist())) == explained_n and explained[0] == 0 and explained[-1] == 583
    for bg in backgrounds:
        assert len(set(bg['indices'])) == 64 and min(bg['indices']) >= 0 and max(bg['indices']) < 2712
    explained_rows = []
    for rank, i in enumerate(explained.tolist()):
        explained_rows.append({'explanation_position': rank, 'test_array_index': i,
                               'original_split_sample_index': int(labels['test']['original_split_indices'][i]),
                               'target_date_label': str(labels['test']['target_dates'][i]),
                               'input_end_date_label': str(labels['test']['input_end_dates'][i]),
                               'calendar_year_by_target': str(labels['test']['target_dates'][i])[:4]})
    background_rows = []
    for bg in backgrounds:
        for rank, i in enumerate(bg['indices']):
            background_rows.append({'background_id': bg['id'], 'background_position': rank,
                                    'background_rng_seed': bg['rng_seed'], 'train_array_index': i,
                                    'original_split_sample_index': int(labels['train']['original_split_indices'][i]),
                                    'target_date_label': str(labels['train']['target_dates'][i]),
                                    'input_end_date_label': str(labels['train']['input_end_dates'][i])})
    explained_file, background_file = HERE/'explained_test_indices_dates.csv', HERE/'background_indices_dates.csv'
    write_csv_immutable(explained_file, explained_rows)
    write_csv_immutable(background_file, background_rows)
    year_counts = {year: sum(r['calendar_year_by_target'] == year for r in explained_rows) for year in ['2022', '2023', '2024']}
    plan = {
        'schema_version': 1, 'status': 'LOCKED_BEFORE_SHAP_INDEX_AND_DATE_SELECTION_ONLY',
        'protocol_version': metadata['protocol_version'],
        'feature_order': metadata['feature_order'], 'output_order': metadata['output_order'],
        'model_training_seeds': list(MODEL_SEEDS), 'background_sampling_seeds': list(BACKGROUND_SEEDS),
        'integration_seeds': list(integration_seeds), 'seed_roles_pairwise_disjoint': True,
        'explained': {'n': explained_n, 'source_n': 584, 'split': 'test',
                      'indices': explained.tolist(), 'index_csv': explained_file.name,
                      'method': f'numpy.linspace(0,583,{explained_n},dtype=int64)',
                      'includes_both_endpoints': True, 'calendar_year_counts_by_target': year_counts,
                      'adjacent_index_gap_min': int(np.diff(explained).min()),
                      'adjacent_index_gap_max': int(np.diff(explained).max())},
        'backgrounds': [dict(bg, n=64, source_n=2712, split='train', index_csv=background_file.name) for bg in backgrounds],
        'background_overlap_counts': {a['id']: {b['id']: len(set(a['indices']) & set(b['indices'])) for b in backgrounds} for a in backgrounds},
        'source_sha256': {p.relative_to(BASE).as_posix(): digest for p, digest in sources.items()},
        'index_files': {p.name: {'sha256': sha(p), 'size_bytes': p.stat().st_size} for p in [explained_file, background_file]},
        'shared_across_all_six_models': True, 'shared_across_all_background_and_integration_conditions': True,
        'uniform_random_backgrounds_without_replacement_within_each_subset': True,
        'backgrounds_required_disjoint_from_each_other': False,
        'selection_uses_y_predictions_errors_or_SHAP_values': False,
        'X_feature_values_loaded_for_selection': False,
        'target_value_arrays_loaded': False, 'model_checkpoint_loaded_or_run': False,
        'new_feature_scaler_fit_or_training': False,
        'subset_limit': f'Results describe the fixed {explained_n}-of-584 explained sample, not an all584 analysis or inference across market regimes.',
        'date_limit': 'Dates are existing common daily labels; sample selection does not certify historical availability or fresh trading events.',
        'environment': {'numpy': np.__version__, 'background_bit_generator': 'PCG64'},
    }
    write_json_immutable(PLAN, plan)
    assert all(sha(p) == digest for p, digest in sources.items())
    return plan


def load_and_validate_sample_plan(plan_path, *, expected_plan_sha256, integration_seeds):
    """Runner entry: verify the protocol-pinned plan and every sampling source."""
    if sha(plan_path) != expected_plan_sha256:
        raise ValueError('Protocol-pinned sample plan SHA mismatch')
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    if plan['integration_seeds'] != list(integration_seeds):
        raise ValueError('Integration RNG roles do not match the locked sample plan')
    if set(integration_seeds) & set(plan['model_training_seeds']+plan['background_sampling_seeds']):
        raise ValueError('Model/background/integration seed roles overlap')
    for name, digest in plan['source_sha256'].items():
        if sha(BASE/name) != digest:
            raise ValueError('Sampling source identity changed: '+name)
    for name, record in plan['index_files'].items():
        path = Path(plan_path).parent/name
        if sha(path) != record['sha256'] or path.stat().st_size != record['size_bytes']:
            raise ValueError('Sampling index/date CSV identity changed')
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--explained-n', type=int, choices=[96, 128], default=128)
    parser.add_argument('--integration-seeds', type=int, nargs=2, default=DEFAULT_INTEGRATION_SEEDS)
    args = parser.parse_args()
    plan = make_plan(args.explained_n, args.integration_seeds)
    assert 'torch' not in sys.modules and 'arch' not in sys.modules
    print(json.dumps({'status': plan['status'], 'sample_plan_sha256': sha(PLAN),
                      'explained_n': plan['explained']['n'], 'year_counts': plan['explained']['calendar_year_counts_by_target'],
                      'backgrounds': 4, 'background_size': 64, 'integration_seeds': plan['integration_seeds'],
                      'X_or_y_values_or_models_loaded': False}))


if __name__ == '__main__':
    main()
