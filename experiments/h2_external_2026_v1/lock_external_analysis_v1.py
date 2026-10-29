#!/usr/bin/env python3
"""Freeze the final H2 external analysis before this project accesses new data.

Only standard-library operations are used. Checkpoints and old arrays are hashed
as opaque bytes; no models, arrays, prices, or historical modules are loaded.
This establishes the local project chronology, not proof that no inaccessible or
unsaved earlier experiment ever used a date in 2026. Run --self-test offline.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import errno
import hashlib
from io import StringIO
import json
import math
import os
from pathlib import Path
import platform
import tempfile

DEFAULT_ROOT = Path('/content/drive/MyDrive/tez_transformer_h2_external_2026_v1')
OLD_RUN_ROOT = Path('/content/drive/MyDrive/tez_transformer_h2_t4_v1/runs/t4_full_run_v1')
PROTECTED_ROOTS = tuple(Path('/content/drive/MyDrive') / x for x in
    ('tez_transformer_v4_repro', 'tez_transformer_h2_v1', 'tez_transformer_h2_t4_v1'))
PLAN_SHA = '0073b3bcc80a342d25f1034fef66d5eb353a9e4e5c14ae3719f57e98de81a659'
OLD_TRAINING_SHA = '517c06f3b1a5d71de9739f56ef6a2af7cf9a918ed2cf303a52d30bfc7b9f393a'
OLD_FROZEN_SHA = '12cf2481978546689f9087d7d97cb4d8325756a15f393a9c936a31e2a47876f4'
BUNDLE_SHA = '21abe6da20959ab958498b24bf0d4307595e4e251231e6005f9297c4ca8d4144'
SOURCE_06 = 'scripts/06_best_model_multiseed_v4.py'
SOURCE_08 = 'scripts/08B_learned_baselines_test_v4.py'
SOURCE_06_SHA = '35de2ee398699003dfef6be36b70c112fb2c0d1b1e9577cbf64bef58877e16d8'
SOURCE_08_SHA = 'df4fa2b29a6b522fc410f693a30520e045c144a15f7c7368af5eb6a1f0f12566'
STL_MANIFEST = 'runs/stl_run_v1/selected_stl_checkpoints_frozen.json'
STL_RUN_LOCK = 'runs/stl_run_v1/run_lock.json'
SEEDS = (123, 777, 2026)
TASKS = ('return', 'volatility')
ARMS = ('NS_base', 'FS_same_width', 'FS_budget_control')
REQUIRED_IMPLEMENTATIONS = frozenset(('run_stl_controls_v1.py',
    'prepare_external_sequences_v1.py', 'acquire_external_prices_v1.py',
    'evaluate_external_h2_v1.py', 'lock_external_analysis_v1.py'))
IMPLEMENTATIONS = REQUIRED_IMPLEMENTATIONS | {'stage_external_plan_v1.py'}
PACKAGES = {'torch': '2.11.0+cu128', 'numpy': '2.3.5', 'pandas': '2.2.3',
            'scipy': '1.17.0', 'scikit-learn': '1.6.1'}
ACQUISITION_RUNTIME = {
    'provider': 'Yahoo Finance', 'library': 'yfinance', 'library_version': '0.2.66',
    'python_version': '3.12.13', 'numpy_version': '2.3.5', 'pandas_version': '2.2.3',
    'isolated_from_training': True,
    'download_parameters': {'auto_adjust': False, 'back_adjust': False, 'repair': False, 'interval': '1d'},
    'download_start_inclusive': '2024-11-01', 'download_end_exclusive': '2026-10-29',
}
SEQ = 'data/sequences/baseline/lb10/'
OLD_INPUT_NAMES = {
    SOURCE_06, 'config/schema_v4.json', 'data/raw/raw_prices.csv',
    'data/sequences/baseline/scalers.pkl',
    'data/processed/selection_baseline_denominators_v4.json', SEQ + 'sequence_meta.json',
} | {SEQ + name + '_' + split + '.npy'
     for split in ('train', 'val', 'test')
     for name in ('X', 'y', 'anchor_dates', 'target_realization_dates')
} | {SEQ + 'y_' + split + '_raw.npy' for split in ('train', 'val', 'test')}
HISTORY_FIELDS = ('epoch', 'scaled_optimization_loss', 'train_primary_loss',
    'validation_primary_loss', 'validation_gap', 'train_task_score', 'validation_task_score',
    'batch_order_sha256', 'optimizer_steps', 'epoch_seconds')


class LockError(RuntimeError):
    pass


def no_symlinks(path: Path) -> None:
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise LockError(f'Symbolic links are forbidden: {path}')


def root_checked(root: Path) -> Path:
    no_symlinks(root)
    root = Path(root).resolve()
    if not root.is_dir() or any(root.is_relative_to(p) or p.is_relative_to(root) for p in PROTECTED_ROOTS):
        raise LockError('Dedicated existing project required; historical roots are protected.')
    return root


def relative_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise LockError('An artifact needs a safe root-relative path.')
    path = root / relative
    no_symlinks(path)
    resolved = path.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise LockError('Artifact escapes its root.')
    return resolved


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    no_symlinks(path)
    if not path.is_file():
        raise LockError(f'Missing regular file: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def valid_digest(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def check_identity(path: Path, record: dict) -> dict:
    if not isinstance(record, dict) or not valid_digest(record.get('sha256')):
        raise LockError(f'Missing valid SHA-256 identity: {path}')
    if digest_file(path) != record['sha256'] or ('size_bytes' in record and path.stat().st_size != record['size_bytes']):
        raise LockError(f'Artifact identity differs: {path}')
    return {'sha256': record['sha256'], 'size_bytes': path.stat().st_size}


def object_bytes(path: Path, expected_sha: str | None = None) -> tuple[dict, dict]:
    no_symlinks(path)
    if not path.is_file():
        raise LockError(f'Missing JSON record: {path}')
    data = path.read_bytes()
    sha = digest_bytes(data)
    if expected_sha is not None and sha != expected_sha:
        raise LockError(f'JSON identity differs: {path}')
    try:
        value = json.loads(data)
    except (ValueError, UnicodeDecodeError) as exc:
        raise LockError(f'Invalid JSON: {path}') from exc
    if not isinstance(value, dict):
        raise LockError(f'Expected JSON object: {path}')
    return value, {'sha256': sha, 'size_bytes': len(data)}


def _trust() -> dict:
    # No CLI or environment overrides: alternate values exist only in self-test fixtures.
    return {'plan': PLAN_SHA, 'old_training': OLD_TRAINING_SHA, 'old_frozen': OLD_FROZEN_SHA,
            'bundle': BUNDLE_SHA, 'source06': SOURCE_06_SHA, 'source08': SOURCE_08_SHA,
            'old_root': OLD_RUN_ROOT}


def runtime_check(runtime: dict) -> None:
    if not isinstance(runtime, dict) or runtime.get('python_version') != '3.12.13' or runtime.get('packages') != PACKAGES:
        raise LockError('STL training runtime package pins differ.')
    for key, value in (('cuda_runtime', '12.8'), ('cudnn_version', 91900),
        ('gpu_name', 'Tesla T4'), ('gpu_count', 1), ('device_capability', [7, 5])):
        if runtime.get(key) != value:
            raise LockError(f'STL training runtime differs: {key}')
    flags = runtime.get('determinism', {})
    expected = {'cublas_workspace_config': ':4096:8', 'deterministic_algorithms': True,
        'cudnn_deterministic': True, 'cudnn_benchmark': False, 'cuda_matmul_allow_tf32': False,
        'cudnn_allow_tf32': False, 'float32_matmul_precision': 'highest', 'amp_enabled': False}
    if flags != expected or runtime.get('threads') != 1 or not runtime.get('driver_version'):
        raise LockError('STL deterministic runtime evidence differs.')


def validate_history(data: bytes, completed: dict) -> list[str]:
    rows = list(csv.DictReader(StringIO(data.decode('utf-8'))))
    if len(rows) != 100 or [int(row['epoch']) for row in rows] != list(range(1, 101)):
        raise LockError('STL history must contain exactly 100 ordered epochs.')
    for row in rows:
        if any(not math.isfinite(float(row[key])) for key in HISTORY_FIELDS if key != 'batch_order_sha256'):
            raise LockError('Nonfinite STL history value.')
        if int(row['optimizer_steps']) != int(row['epoch']) * 43 or not valid_digest(row['batch_order_sha256']):
            raise LockError('STL optimizer budget or mini-batch order identity differs.')
    scores = [float(r['validation_task_score']) for r in rows]
    selected = min(range(100), key=lambda i: scores[i])
    if selected + 1 != completed['selected_epoch'] or scores[selected] != completed['selected_validation_task_score']:
        raise LockError('STL checkpoint violates earliest strict normalized-validation minimum.')
    return [row['batch_order_sha256'] for row in rows]


def validate_replay(replay: dict) -> None:
    expected = {'passed': True, 'n_checks': 12, 'relative_tolerance': 1e-5,
                'absolute_tolerance': 1e-8, 'test_arrays_loaded': False, 'new_period_accessed': False}
    if any(replay.get(k) != v or isinstance(v, bool) and replay.get(k) is not v for k, v in expected.items()):
        raise LockError('Six complete train/validation numerical replay proofs are required.')
    delta = replay.get('maximum_absolute_delta')
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta < 0:
        raise LockError('Invalid STL numerical replay delta.')


def collect_frozen_evidence(root: Path, *, trust: dict | None = None) -> dict:
    """No external-period path is opened, even for hashing, in this function."""
    trust = _trust() if trust is None else trust
    root = root_checked(root)
    auxiliary = []

    def remember(base, relative, record, scope='project'):
        identity = check_identity(relative_path(base, relative), record)
        auxiliary.append({'scope': scope, 'path': relative, **identity})
        return relative_path(base, relative)

    plan, plan_identity = object_bytes(relative_path(root, 'protocol_plan_v1.json'), trust['plan'])
    if plan.get('status') != 'STAGED_PLAN_NO_NEW_DATA' or plan.get('project_root') != str(root):
        raise LockError('The unchanged staged scientific plan and exact project root are required.')
    if plan.get('gates', {}).get('new_data_access_allowed_now') is not False:
        raise LockError('The staged plan must precede new data access.')
    remember(root, 'protocol_plan_v1.json', plan_identity)
    implementation = relative_path(root, 'implementation')
    present = {p.name for p in implementation.iterdir() if p.suffix == '.py'}
    if present != IMPLEMENTATIONS:
        raise LockError('Exactly the six reviewed implementation files must be present.')
    implementation_sha = {}
    for name in sorted(IMPLEMENTATIONS):
        path = relative_path(root, 'implementation/' + name)
        implementation_sha[name] = digest_file(path)
        remember(root, 'implementation/' + name, {'sha256': implementation_sha[name]})
    bundle, bundle_identity = object_bytes(relative_path(root, 'implementation/bundle_manifest_v1.json'), trust['bundle'])
    remember(root, 'implementation/bundle_manifest_v1.json', bundle_identity)
    if set(bundle.get('files', {})) != {'README.md', 'protocol_plan_v1.json',
            'stage_external_plan_v1.py', 'run_stl_controls_v1.py', 'prepare_external_sequences_v1.py'}:
        raise LockError('The original reviewed preparation bundle differs.')
    for name, identity in bundle['files'].items():
        remember(root, 'implementation/' + name, identity)
    old_training, old_training_id = object_bytes(relative_path(root, 'source_records/locked_training_manifest.json'), trust['old_training'])
    old_frozen, old_frozen_id = object_bytes(relative_path(root, 'source_records/selected_checkpoints_frozen.json'), trust['old_frozen'])
    remember(root, 'source_records/locked_training_manifest.json', old_training_id)
    remember(root, 'source_records/selected_checkpoints_frozen.json', old_frozen_id)
    old_root = Path(trust['old_root'])
    no_symlinks(old_root)
    if (old_training.get('status') != 'LOCKED_BEFORE_TRAINING'
        or old_training.get('run_root') != str(old_root)
        or old_training.get('test_arrays_loaded') is not False
        or old_frozen.get('status') != 'ALL_NINE_CHECKPOINTS_FROZEN_NO_TEST_ACCESS'
        or old_frozen.get('locked_training_manifest_sha256') != trust['old_training']
        or old_frozen.get('test_arrays_loaded') is not False):
        raise LockError('The original T4 provenance differs.')
    old_jobs = old_frozen.get('completed_runs')
    expected_old = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if not isinstance(old_jobs, list) or len(old_jobs) != 9 or {(j.get('arm', {}).get('id'), j.get('seed')) for j in old_jobs} != expected_old:
        raise LockError('Exactly nine distinct original arm/seed fits are required.')
    old_checkpoints = []
    for job in old_jobs:
        if job.get('status') != 'SUCCESS_NO_TEST_ACCESS' or job.get('test_arrays_loaded') is not False or job.get('epochs_ran') != 100:
            raise LockError('Original H2 training completion evidence differs.')
        for name in ('best_checkpoint', 'final_checkpoint', 'history'):
            item = job.get(name, {})
            path = item.get('path')
            remember(old_root, path, item, 'old_h2_run')
            if name == 'best_checkpoint':
                old_checkpoints.append({'arm': job['arm']['id'], 'seed': job['seed'], **item})
    inputs = old_training.get('input_identities')
    if not isinstance(inputs, dict) or set(inputs) != OLD_INPUT_NAMES or len(inputs) != 21:
        raise LockError('Exactly the original 21 frozen inputs are required.')
    copied = {}
    for name, identity in sorted(inputs.items()):
        remember(root, 'inputs/' + name, identity)
        copied[name] = dict(identity)
    if copied[SOURCE_06]['sha256'] != trust['source06']:
        raise LockError('The archived model definition source differs.')
    extra_path = remember(root, 'inputs/' + SOURCE_08, {'sha256': trust['source08']})
    copied[SOURCE_08] = {'sha256': trust['source08'], 'size_bytes': extra_path.stat().st_size}
    stage, stage_id = object_bytes(relative_path(root, 'stage_receipt_v1.json'))
    if stage.get('status') != 'H2_EXTERNAL_PLAN_STAGED_NO_NEW_DATA':
        raise LockError('The preparation receipt is missing or incomplete.')
    for key in ('training_started', 'new_period_downloaded', 'test_arrays_loaded', 'final_analysis_lock_created'):
        if stage.get(key) is not False:
            raise LockError('Preparation receipt must document no training or new data at staging.')
    staged = stage.get('identities', {})
    if (staged.get('project_root') != str(root) or staged.get('protocol_plan') != plan_identity
            or staged.get('input_identities') != copied):
        raise LockError('Preparation receipt does not match all 22 copied input identities.')
    for name, identity in staged.get('source_records', {}).items():
        remember(root, 'source_records/' + name, identity)
    if set(staged.get('source_records', {})) != {'locked_training_manifest.json', 'selected_checkpoints_frozen.json'}:
        raise LockError('Preparation source receipt records differ.')
    for name, identity in staged.get('bundle_identities', {}).items():
        remember(root, 'implementation/' + name, identity)
    if set(staged.get('bundle_identities', {})) != set(bundle['files']) | {'bundle_manifest_v1.json'}:
        raise LockError('The preparation receipt bundle differs.')
    remember(root, 'stage_receipt_v1.json', stage_id)
    stl, stl_identity = object_bytes(relative_path(root, STL_MANIFEST))
    remember(root, STL_MANIFEST, stl_identity)
    expected_flags = {'status': 'STL_CHECKPOINTS_FROZEN_BEFORE_NEW_DATA',
        'plan_sha256': trust['plan'], 'training_implementation_sha256': implementation_sha['run_stl_controls_v1.py'],
        'run_lock_path': STL_RUN_LOCK, 'n_actual_fits': 6, 'paired_seed_task_minibatch_orders_equal': True,
        'test_arrays_loaded': False, 'new_period_accessed': False, 'old_project_writes': False}
    for key, value in expected_flags.items():
        if stl.get(key) != value or isinstance(value, bool) and stl.get(key) is not value:
            raise LockError(f'STL final freeze evidence differs: {key}')
    runtime_check(stl.get('runtime_environment'))
    run_lock, run_lock_identity = object_bytes(relative_path(root, STL_RUN_LOCK), stl.get('run_lock_sha256'))
    remember(root, STL_RUN_LOCK, run_lock_identity)
    core = run_lock.get('core', {})
    expected_evidence = {'plan_sha256': trust['plan'], 'input_identities': copied,
        'old_locked_training_manifest_sha256': trust['old_training'],
        'old_selected_checkpoints_frozen_sha256': trust['old_frozen'],
        'training_implementation_sha256': implementation_sha['run_stl_controls_v1.py']}
    if (core.get('purpose') != 'STL_CONTROLS_FROZEN_BEFORE_NEW_DATA'
            or core.get('project_root') != str(root)
            or core.get('output_root') != str(root / 'runs/stl_run_v1')
            or core.get('runtime_environment') != stl['runtime_environment']
            or core.get('settings') != stl.get('settings')):
        raise LockError('STL run lock disagrees with its final freeze manifest.')
    for key, value in expected_evidence.items():
        if core.get('evidence', {}).get(key) != value:
            raise LockError(f'STL input or implementation provenance differs: {key}')
    for flags in (core, core.get('evidence', {})):
        if flags.get('test_arrays_loaded') is not False or flags.get('new_period_accessed') is not False:
            raise LockError('STL lock must precede all new-period access.')
    expected_jobs = {(task, seed) for task in TASKS for seed in SEEDS}
    smoke = core.get('initialization_smoke_proofs', {})
    if (smoke.get('status') != 'SYNTHETIC_MODELS_PASS_NO_FITTING' or smoke.get('models_checked') != 6
            or smoke.get('optimizer_steps') != 0 or smoke.get('real_data_predictions') is not False):
        raise LockError('Six synthetic initialization model proofs are required.')
    smoke_checks = smoke.get('checks', [])
    if len(smoke_checks) != 6 or {(c.get('task'), c.get('seed')) for c in smoke_checks} != expected_jobs:
        raise LockError('Synthetic initialization proofs do not cover all task/seed pairs.')
    for proof in smoke_checks:
        if (proof.get('parameters') != 27268 or proof.get('optimizer_steps') != 0
                or not valid_digest(proof.get('initial_state_sha256'))
                or any(proof.get(k) is not True for k in ('all_initial_tensors_equal_to_corresponding_template_branch',
                    'forward_backward_pass', 'branch_forward_equal_in_eval_mode'))):
            raise LockError('Synthetic branch-copy/forward/backward proof differs.')
    checkpoints = stl.get('checkpoints')
    if not isinstance(checkpoints, list) or len(checkpoints) != 6 or {(c.get('task'), c.get('seed')) for c in checkpoints} != expected_jobs:
        raise LockError('Exactly six distinct independent task/seed checkpoints are required.')
    replays = stl.get('train_validation_replays')
    if not isinstance(replays, list) or len(replays) != 6 or {(r.get('task'), r.get('seed')) for r in replays} != expected_jobs:
        raise LockError('Exactly six train/validation numerical replay proofs are required.')
    for replay in replays:
        validate_replay(replay)
    orders = {}
    stl_checkpoints = []
    for checkpoint in checkpoints:
        task, seed = checkpoint['task'], checkpoint['seed']
        folder = f'runs/stl_run_v1/jobs/{task}_seed{seed}'
        if checkpoint.get('parameters') != 27268 or checkpoint.get('completed_job_path') != folder + '/completed_job.json':
            raise LockError('STL checkpoint task directory or parameter count differs.')
        remember(root, checkpoint.get('path'), checkpoint)
        completed_path = remember(root, checkpoint['completed_job_path'], {'sha256': checkpoint.get('completed_job_sha256')})
        completed, _ = object_bytes(completed_path, checkpoint['completed_job_sha256'])
        for key, value in (('status', 'STL_TASK_TRAINING_COMPLETE_NO_NEW_DATA'), ('task', task), ('seed', seed),
            ('epochs_ran', 100), ('parameters', 27268), ('plan_sha256', trust['plan']),
            ('run_lock_sha256', run_lock_identity['sha256']), ('test_arrays_loaded', False), ('new_period_accessed', False)):
            if completed.get(key) != value or isinstance(value, bool) and completed.get(key) is not value:
                raise LockError(f'Completed STL job differs: {key}')
        status, status_id = object_bytes(relative_path(root, folder + '/job_status.json'))
        if status.get('status') != completed['status'] or status.get('task') != task or status.get('seed') != seed:
            raise LockError('STL completed-job status differs.')
        remember(root, folder + '/job_status.json', status_id)
        artifacts = completed.get('artifacts', {})
        if set(artifacts) != {'best_checkpoint', 'final_checkpoint', 'history', 'selected_metrics', 'final_metrics', 'initialization', 'replay'}:
            raise LockError('Completed STL artifact set differs.')
        for artifact in artifacts.values():
            p = remember(root, artifact.get('path'), artifact)
            if not p.is_relative_to(root / folder):
                raise LockError('STL artifact is outside its exact task directory.')
        best = artifacts['best_checkpoint']
        if any(best.get(k) != checkpoint.get(k) for k in ('path', 'sha256', 'size_bytes')):
            raise LockError('Frozen STL checkpoint differs from its completed job.')
        for key in ('selected_epoch', 'selected_validation_task_score'):
            if checkpoint.get(key) != completed.get(key):
                raise LockError('Frozen STL selection metadata differs.')
        history_path = relative_path(root, artifacts['history']['path'])
        history_bytes = history_path.read_bytes()
        if digest_bytes(history_bytes) != artifacts['history']['sha256']:
            raise LockError('STL history changed while being checked.')
        order = validate_history(history_bytes, completed)
        if seed in orders and order != orders[seed]:
            raise LockError('Paired independent tasks used different batch orders.')
        orders[seed] = order
        metrics, _ = object_bytes(relative_path(root, artifacts['selected_metrics']['path']), artifacts['selected_metrics']['sha256'])
        if (metrics.get('validation', {}).get('task_score') != checkpoint['selected_validation_task_score']
                or metrics.get('train', {}).get('primary_loss') != checkpoint.get('train_primary_loss')
                or metrics.get('validation', {}).get('primary_loss') != checkpoint.get('validation_primary_loss')):
            raise LockError('Selected STL metric identities differ from the final freeze.')
        proof, _ = object_bytes(relative_path(root, artifacts['initialization']['path']), artifacts['initialization']['sha256'])
        expected_proof = next(p for p in smoke_checks if p['task'] == task and p['seed'] == seed)
        initial_proof = {k: v for k, v in expected_proof.items()
                         if k not in ('forward_backward_pass', 'branch_forward_equal_in_eval_mode', 'optimizer_steps')}
        if proof != completed.get('initialization') or proof != initial_proof:
            raise LockError('STL initial branch-copy identities differ.')
        replay, _ = object_bytes(relative_path(root, artifacts['replay']['path']), artifacts['replay']['sha256'])
        validate_replay(replay)
        stl_checkpoints.append(dict(checkpoint))
    # Deduplicate identical identities so revalidation is deterministic and bounded.
    unique = {}
    for record in auxiliary:
        key = (record['scope'], record['path'])
        if key in unique and unique[key] != record:
            raise LockError('One artifact has conflicting frozen identities.')
        unique[key] = record
    return {'schema_version': 1, 'status': 'ANALYSIS_LOCKED_BEFORE_NEW_DATA',
        'project_root': str(root), 'protocol_plan_sha256': trust['plan'],
        'stage_receipt_sha256': stage_id['sha256'], 'implementation_sha256': implementation_sha,
        'selected_stl_checkpoints_manifest': {'path': STL_MANIFEST, **stl_identity},
        'existing_h2_checkpoints_manifest': {'path': 'source_records/selected_checkpoints_frozen.json', **old_frozen_id},
        'old_locked_training_manifest_sha256': trust['old_training'],
        'existing_h2_run_root': str(old_root), 'existing_h2_selected_checkpoints': old_checkpoints,
        'selected_stl_checkpoints': stl_checkpoints,
        'auxiliary_artifact_identities': [unique[key] for key in sorted(unique)],
        'acquisition_runtime': ACQUISITION_RUNTIME,
        'study_classification': plan['classification'], 'fixed_design': plan['design'],
        'fixed_new_period_data': plan['new_period_data'], 'fixed_outcomes': plan['outcomes'],
        'local_chronology_scope': 'No new_period directory exists at initial lock creation. This does not establish absence of data access outside this project or in deleted, unsaved or inaccessible work.',
        'training_started_by_this_code': False, 'old_test_arrays_deserialized_by_this_code': False,
        'new_period_accessed_by_this_code': False, 'old_project_writes': False}


def recheck_auxiliary(root: Path, core: dict) -> None:
    for record in core['auxiliary_artifact_identities']:
        base = root if record['scope'] == 'project' else Path(core['existing_h2_run_root'])
        check_identity(relative_path(base, record['path']), record)


def create_lock(root: Path, *, _test_trust: dict | None = None, _execution_check: bool = True) -> dict:
    root = root_checked(root)
    if _execution_check and Path(__file__).resolve() != relative_path(root, 'implementation/lock_external_analysis_v1.py'):
        raise LockError('Execute the staged lock implementation from this exact project.')
    path = relative_path(root, 'analysis_locked_v1.json')
    existing = object_bytes(path)[0] if path.exists() else None
    # Inspect path existence only. No external-period bytes or metadata are read.
    external = relative_path(root, 'new_period')
    if existing is None and external.exists():
        raise LockError('New-period directory already exists: cannot create a before-data lock retroactively.')
    core = collect_frozen_evidence(root, trust=_test_trust)
    if existing is not None:
        if not isinstance(existing.get('created_at_utc'), str) or not isinstance(existing.get('lock_writer_environment'), dict):
            raise LockError('Existing lock chronology metadata is incomplete.')
        compare = {k: v for k, v in existing.items() if k not in ('created_at_utc', 'lock_writer_environment')}
        if compare != core:
            raise LockError('Existing immutable analysis lock differs; no relock or overwrite performed.')
        recheck_auxiliary(root, core)
        return {'status': core['status'], 'analysis_lock_path': str(path), 'analysis_lock_sha256': digest_file(path),
                'lock_reused': True, 'n_existing_checkpoints': 9, 'n_stl_checkpoints': 6,
                'new_period_accessed_by_this_code': False, 'training_started_by_this_code': False}
    # All identities are rehashed immediately before the one exclusive output write.
    recheck_auxiliary(root, core)
    if external.exists() or path.exists():
        raise LockError('Project state changed before the exclusive analysis lock write.')
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'lock_writer_environment': {'python_version': platform.python_version(), 'platform': platform.system()}, **core}
    data = (json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    no_symlinks(path)
    with path.open('xb') as handle:
        handle.write(data)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError as exc:
            if exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
                raise
    if path.read_bytes() != data:
        raise LockError('New analysis lock failed exact byte readback; retain it for review.')
    return {'status': core['status'], 'analysis_lock_path': str(path), 'analysis_lock_sha256': digest_bytes(data),
            'lock_reused': False, 'n_existing_checkpoints': 9, 'n_stl_checkpoints': 6,
            'new_period_accessed_by_this_code': False, 'training_started_by_this_code': False}


def _json_write(path: Path, value: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    path.write_bytes(data)
    return {'sha256': digest_bytes(data), 'size_bytes': len(data)}


def _fixture(base: Path) -> tuple[Path, dict]:
    root, old = base / 'project', base / 'old_run'
    root.mkdir(parents=True)
    old.mkdir()
    ids = {}
    for name in sorted(OLD_INPUT_NAMES | {SOURCE_08}):
        p = root / 'inputs' / name
        p.parent.mkdir(parents=True, exist_ok=True)
        data = ('opaque original ' + name).encode()
        p.write_bytes(data)
        ids[name] = {'sha256': digest_bytes(data), 'size_bytes': len(data)}
    plan = {'status': 'STAGED_PLAN_NO_NEW_DATA', 'project_root': str(root),
        'gates': {'new_data_access_allowed_now': False}, 'classification': {'coverage_limit': 'fixture'},
        'design': {'seeds': list(SEEDS)}, 'new_period_data': {}, 'outcomes': {}}
    plan_id = _json_write(root / 'protocol_plan_v1.json', plan)
    impl = root / 'implementation'
    impl.mkdir()
    for name in IMPLEMENTATIONS:
        (impl / name).write_text('# frozen fixture ' + name + '\n')
    (impl / 'README.md').write_text('fixture readme\n')
    (impl / 'protocol_plan_v1.json').write_bytes((root / 'protocol_plan_v1.json').read_bytes())
    bundle_names = {'README.md', 'protocol_plan_v1.json', 'stage_external_plan_v1.py',
                    'run_stl_controls_v1.py', 'prepare_external_sequences_v1.py'}
    bundle_ids = {n: {'sha256': digest_file(impl/n), 'size_bytes': (impl/n).stat().st_size} for n in bundle_names}
    bundle_id = _json_write(impl/'bundle_manifest_v1.json', {'files': bundle_ids})
    original_inputs = {k:v for k,v in ids.items() if k != SOURCE_08}
    train = {'status': 'LOCKED_BEFORE_TRAINING', 'run_root': str(old),
             'input_identities': original_inputs, 'test_arrays_loaded': False}
    train_id = _json_write(root/'source_records/locked_training_manifest.json', train)
    old_jobs = []
    for arm in ARMS:
        for seed in SEEDS:
            j = {'status': 'SUCCESS_NO_TEST_ACCESS', 'arm': {'id': arm}, 'seed': seed,
                 'epochs_ran': 100, 'test_arrays_loaded': False}
            for key, name in (('best_checkpoint','best_checkpoint.pt'), ('final_checkpoint','final_checkpoint.pt'), ('history','history.csv')):
                path = f'runs/{arm}_seed{seed}/{name}'
                p = old/path
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(('opaque old ' + path).encode())
                j[key] = {'path': path, 'sha256': digest_file(p)}
            old_jobs.append(j)
    frozen = {'status': 'ALL_NINE_CHECKPOINTS_FROZEN_NO_TEST_ACCESS',
        'locked_training_manifest_sha256': train_id['sha256'], 'test_arrays_loaded': False, 'completed_runs': old_jobs}
    frozen_id = _json_write(root/'source_records/selected_checkpoints_frozen.json', frozen)
    staged = {'status': 'H2_EXTERNAL_PLAN_STAGED_NO_NEW_DATA', 'training_started': False,
        'new_period_downloaded': False, 'test_arrays_loaded': False, 'final_analysis_lock_created': False,
        'identities': {'project_root': str(root), 'protocol_plan': plan_id, 'input_identities': ids,
            'source_records': {'locked_training_manifest.json': train_id, 'selected_checkpoints_frozen.json': frozen_id},
            'bundle_identities': {**bundle_ids, 'bundle_manifest_v1.json': bundle_id}}}
    _json_write(root/'stage_receipt_v1.json', staged)
    runtime = {'python_version':'3.12.13', 'packages': PACKAGES, 'cuda_runtime':'12.8',
        'cudnn_version':91900, 'gpu_name':'Tesla T4', 'gpu_count':1, 'device_capability':[7,5],
        'threads':1, 'driver_version':'fixture-driver', 'determinism': {
        'cublas_workspace_config':':4096:8', 'deterministic_algorithms':True, 'cudnn_deterministic':True,
        'cudnn_benchmark':False, 'cuda_matmul_allow_tf32':False, 'cudnn_allow_tf32':False,
        'float32_matmul_precision':'highest', 'amp_enabled':False}}
    smoke_checks = []
    for task in TASKS:
        for seed in SEEDS:
            smoke_checks.append({'task': task, 'seed': seed, 'parameters':27268,
                'initial_state_sha256': 'a'*64, 'prefix_map': [],
                'all_initial_tensors_equal_to_corresponding_template_branch':True,
                'forward_backward_pass':True, 'branch_forward_equal_in_eval_mode':True, 'optimizer_steps':0})
    smoke = {'status':'SYNTHETIC_MODELS_PASS_NO_FITTING', 'models_checked':6, 'optimizer_steps':0,
             'real_data_predictions':False, 'checks':smoke_checks}
    training_sha = digest_file(impl/'run_stl_controls_v1.py')
    settings = {'fixture': True}
    core = {'purpose':'STL_CONTROLS_FROZEN_BEFORE_NEW_DATA', 'project_root':str(root),
        'output_root':str(root/'runs/stl_run_v1'), 'runtime_environment':runtime, 'settings':settings,
        'initialization_smoke_proofs':smoke, 'test_arrays_loaded':False, 'new_period_accessed':False,
        'evidence':{'plan_sha256':plan_id['sha256'], 'input_identities':ids,
            'old_locked_training_manifest_sha256':train_id['sha256'], 'old_selected_checkpoints_frozen_sha256':frozen_id['sha256'],
            'training_implementation_sha256':training_sha, 'test_arrays_loaded':False, 'new_period_accessed':False}}
    lock_id = _json_write(root/STL_RUN_LOCK, {'core':core})
    checkpoints, replays = [], []
    for task in TASKS:
        for seed in SEEDS:
            folder = f'runs/stl_run_v1/jobs/{task}_seed{seed}'
            p = root/folder
            p.mkdir(parents=True)
            proof = next({k:v for k,v in pr.items() if k not in ('forward_backward_pass','branch_forward_equal_in_eval_mode','optimizer_steps')}
                         for pr in smoke_checks if pr['task']==task and pr['seed']==seed)
            _json_write(p/'initialization.json', proof)
            (p/'best_checkpoint.pt').write_bytes(f'opaque {task}{seed}'.encode())
            (p/'final_checkpoint.pt').write_bytes(f'opaque final{task}{seed}'.encode())
            metrics = {'train': {'primary_loss':0.1}, 'validation':{'primary_loss':0.2,'task_score':0.01}}
            _json_write(p/'selected_train_validation_metrics.json', metrics)
            _json_write(p/'final_train_validation_metrics.json', metrics)
            replay = {'passed':True, 'n_checks':12, 'maximum_absolute_delta':0.0, 'relative_tolerance':1e-5,
                      'absolute_tolerance':1e-8, 'test_arrays_loaded':False, 'new_period_accessed':False}
            _json_write(p/'train_validation_replay.json', replay)
            rows=[]
            for epoch in range(1,101):
                row={k:0.1 for k in HISTORY_FIELDS}
                row.update(epoch=epoch, optimizer_steps=43*epoch, validation_task_score=1/epoch,
                           batch_order_sha256=digest_bytes(f'order{seed}/{epoch}'.encode()))
                rows.append(row)
            with (p/'history.csv').open('w',newline='') as h:
                w=csv.DictWriter(h,fieldnames=HISTORY_FIELDS);w.writeheader();w.writerows(rows)
            names={'best_checkpoint':'best_checkpoint.pt','final_checkpoint':'final_checkpoint.pt','history':'history.csv',
                   'selected_metrics':'selected_train_validation_metrics.json','final_metrics':'final_train_validation_metrics.json',
                   'initialization':'initialization.json','replay':'train_validation_replay.json'}
            artifacts={k:{'path':folder+'/'+n,'sha256':digest_file(p/n),'size_bytes':(p/n).stat().st_size} for k,n in names.items()}
            completed={'status':'STL_TASK_TRAINING_COMPLETE_NO_NEW_DATA','task':task,'seed':seed,'epochs_ran':100,
                'parameters':27268,'plan_sha256':plan_id['sha256'],'run_lock_sha256':lock_id['sha256'],
                'test_arrays_loaded':False,'new_period_accessed':False,'selected_epoch':100,
                'selected_validation_task_score':0.01,'initialization':proof,'artifacts':artifacts}
            cid=_json_write(p/'completed_job.json',completed)
            _json_write(p/'job_status.json', {'status':completed['status'],'task':task,'seed':seed})
            checkpoints.append({'task':task,'seed':seed,**artifacts['best_checkpoint'],'selected_epoch':100,
                'parameters':27268,'selected_validation_task_score':0.01,'train_primary_loss':0.1,
                'validation_primary_loss':0.2,'completed_job_path':folder+'/completed_job.json','completed_job_sha256':cid['sha256']})
            replays.append({'task':task,'seed':seed,**replay})
    stl={'status':'STL_CHECKPOINTS_FROZEN_BEFORE_NEW_DATA','plan_sha256':plan_id['sha256'],
        'training_implementation_sha256':training_sha,'run_lock_path':STL_RUN_LOCK,'run_lock_sha256':lock_id['sha256'],
        'n_actual_fits':6,'paired_seed_task_minibatch_orders_equal':True,'test_arrays_loaded':False,
        'new_period_accessed':False,'old_project_writes':False,'runtime_environment':runtime,'settings':settings,
        'checkpoints':checkpoints,'train_validation_replays':replays}
    _json_write(root/STL_MANIFEST,stl)
    trust={'plan':plan_id['sha256'],'old_training':train_id['sha256'],'old_frozen':frozen_id['sha256'],
           'bundle':bundle_id['sha256'],'source06':ids[SOURCE_06]['sha256'],'source08':ids[SOURCE_08]['sha256'],'old_root':old}
    return root,trust


def self_test() -> dict:
    checks=[]
    with tempfile.TemporaryDirectory(prefix='h2_analysis_lock_selftest_') as tmp:
        base=Path(tmp)
        root,trust=_fixture(base/'valid')
        old_bytes={str(p):p.read_bytes() for p in trust['old_root'].rglob('*') if p.is_file()}
        first=create_lock(root,_test_trust=trust,_execution_check=False)
        checks.append('fifteen_opaque_checkpoints_and_complete_provenance')
        before=(root/'analysis_locked_v1.json').read_bytes()
        again=create_lock(root,_test_trust=trust,_execution_check=False)
        assert first['lock_reused'] is False and again['lock_reused'] is True and before==(root/'analysis_locked_v1.json').read_bytes()
        checks.append('idempotent_exact_lock_bytes')
        (root/'new_period').mkdir()
        assert create_lock(root,_test_trust=trust,_execution_check=False)['lock_reused'] is True
        checks.append('existing_lock_revalidates_without_retroactive_relock')
        assert all(Path(p).read_bytes()==data for p,data in old_bytes.items())
        checks.append('all_old_files_unchanged')
        scenarios={
            'new_period_before_initial_lock': lambda r,t:(r/'new_period').mkdir(),
            'old_checkpoint_tamper': lambda r,t:(t['old_root']/'runs/NS_base_seed123/best_checkpoint.pt').write_bytes(b'changed'),
            'stl_checkpoint_tamper': lambda r,t:(r/'runs/stl_run_v1/jobs/return_seed123/best_checkpoint.pt').write_bytes(b'changed'),
            'implementation_tamper': lambda r,t:(r/'implementation/run_stl_controls_v1.py').write_text('changed'),
            'input_tamper': lambda r,t:(r/'inputs'/SEQ/'X_test.npy').write_bytes(b'changed opaque array'),
            'stage_receipt_tamper': lambda r,t:(r/'stage_receipt_v1.json').write_text('{}'),
            'stl_manifest_incomplete': lambda r,t:_json_write(r/STL_MANIFEST,{'status':'partial'}),
            'missing_evaluation_code': lambda r,t:(r/'implementation/evaluate_external_h2_v1.py').unlink(),
            'symlink_checkpoint': lambda r,t:((r/'runs/stl_run_v1/jobs/return_seed123/best_checkpoint.pt').unlink(),(r/'runs/stl_run_v1/jobs/return_seed123/best_checkpoint.pt').symlink_to('/etc/hosts')),
        }
        for name,mutation in scenarios.items():
            r,t=_fixture(base/name);mutation(r,t)
            try:create_lock(r,_test_trust=t,_execution_check=False)
            except (LockError,FileNotFoundError):pass
            else:raise AssertionError('Expected rejection: '+name)
            assert not (r/'analysis_locked_v1.json').exists()
            checks.append('reject_'+name)
        # Existing lock drift must not alter the historical lock bytes.
        (root/'implementation/evaluate_external_h2_v1.py').write_text('changed new evaluation code')
        try:create_lock(root,_test_trust=trust,_execution_check=False)
        except LockError:pass
        else:raise AssertionError('Existing lock accepted code drift.')
        assert before==(root/'analysis_locked_v1.json').read_bytes()
        checks.append('reject_drift_without_rewriting_existing_lock')
        try:relative_path(root,'../outside')
        except LockError:pass
        else:raise AssertionError('Path traversal accepted.')
        checks.append('reject_path_traversal')
        # Distinct minima and ties: an earlier equal minimum is mandatory.
        rows=[]
        for epoch in range(1,101):
            row={k:0.1 for k in HISTORY_FIELDS};row.update(epoch=epoch,optimizer_steps=43*epoch,
                validation_task_score=0.01 if epoch in (2,3) else 1.0,batch_order_sha256='a'*64);rows.append(row)
        h=StringIO();w=csv.DictWriter(h,fieldnames=HISTORY_FIELDS);w.writeheader();w.writerows(rows)
        validate_history(h.getvalue().encode(),{'selected_epoch':2,'selected_validation_task_score':0.01})
        try:validate_history(h.getvalue().encode(),{'selected_epoch':3,'selected_validation_task_score':0.01})
        except LockError:pass
        else:raise AssertionError('Later equal minimum accepted.')
        checks.append('earliest_strict_validation_minimum')
    return {'status':'ANALYSIS_LOCK_SELF_TEST_PASS','n_checks':len(checks),'checks':checks,
            'real_data_accessed':False,'training_started':False}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=DEFAULT_ROOT)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--lock',action='store_true',help='Verify frozen controls and exclusively create the final before-data analysis lock.')
    modes.add_argument('--self-test',action='store_true',help='Offline temporary synthetic fixtures; no project access.')
    args=parser.parse_args()
    result=self_test() if args.self_test else create_lock(args.project_root)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':
    main()
