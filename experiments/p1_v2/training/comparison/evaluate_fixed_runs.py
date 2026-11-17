"""Six fixed P1V2 checkpoints only. Default is metadata preflight, never inference.

Explicit --execute-evaluation is required. Test arrays are opened only after all
six successful receipts, histories, checkpoint hashes and paired initial states
have been checked. Only a fresh <runs-root>/comparison directory may be written.
"""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
import math
import platform
import sys
from dataclasses import asdict

import numpy as np

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent
BASE = TRAINING.parent
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))
from config import ASSETS, FEATURE_ORDER, OUTPUT_ORDER, PROTOCOL_VERSION, SEEDS, fixed_arms
from data_contract import inverse_y_float64
try:
    from .numeric_metrics import errors_and_losses, metrics, raw_seed_ensemble, descriptive_difference
except ImportError:
    from numeric_metrics import errors_and_losses, metrics, raw_seed_ensemble, descriptive_difference

EXPECTED_COUNTS = {'train': 2712, 'val': 583, 'test': 584}
DEFAULT_RUNS = TRAINING/'runs/p1_v2_mse_mae_20261113_full'
DATASET = BASE/'datasets/p1_v2'
SCOPES = {
    'primary': ('primary_return_mask', 'primary_vol20_mask'),
    'secondary_raw_target': ('secondary_return_raw_target_mask', 'secondary_vol20_raw_target_mask'),
    'diagnostic_all_model_inputs': ('strict_all_model_inputs_return_target_mask', 'strict_all_model_inputs_vol20_target_mask'),
    'diagnostic_asset_inputs': ('strict_asset_input_return_target_mask', 'strict_asset_input_vol20_target_mask'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def reference(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'size_bytes': path.stat().st_size}


def write_csv(path, rows):
    rows = list(rows)
    require(bool(rows), 'Refuse an empty CSV: '+str(path))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def source_preflight(runs_root):
    """Read JSON/CSV/bytes only; no NPY, Torch, checkpoint deserialization or model."""
    require(sys.flags.optimize == 0, 'Python -O is unsupported: canonical score assertions must remain active')
    runs_root = Path(runs_root).resolve()
    require((TRAINING/'runs').resolve() in runs_root.parents, 'Runs must be in this new training/runs workspace')
    contract = read_json(HERE/'comparison_protocol.json')
    for filename, expected in contract['frozen_training_code_sha256'].items():
        require(sha(TRAINING/filename) == expected, 'Frozen production code changed: '+filename)
    metadata_path = DATASET/'dataset_metadata.json'
    metadata = read_json(metadata_path)
    require(metadata['protocol_version'] == PROTOCOL_VERSION and metadata['split_counts'] == EXPECTED_COUNTS, 'Dataset protocol/count mismatch')
    require(metadata['feature_order'] == list(FEATURE_ORDER) and metadata['output_order'] == list(OUTPUT_ORDER), 'Dataset order mismatch')
    source_paths = {'raw': BASE/'inputs/raw_prices.csv', 'map': BASE/'protocol/proposed_lag1_date_map.csv',
                    'locked_denominators': BASE/'benchmarks/validation/selection_denominators_P1_validation_v2.json'}
    for key, path in source_paths.items():
        require(sha(path) == metadata['source_sha256'][key], 'Dataset source changed: '+key)
    scalers = read_json(DATASET/'scalers.json')
    require(sha(DATASET/'scalers.json') == metadata['scalers_sha256'], 'Scaler identity changed')
    require(scalers['output_order'] == list(OUTPUT_ORDER) and scalers['feature_order'] == list(FEATURE_ORDER), 'Scaler order changed')
    for kind in ('x', 'y'):
        require(scalers[kind]['n_samples_seen'] == 2721, 'Scaler fit population changed')
        for key in ('mean', 'var', 'scale'):
            values = np.asarray(scalers[kind][key], dtype=np.float64)
            require(values.shape == (8,) and np.isfinite(values).all(), 'Invalid scaler statistics')
        require(np.all(np.asarray(scalers[kind]['scale']) > 0), 'Invalid scaler scale')
    paths = [metadata_path, DATASET/'scalers.json', *source_paths.values(),
             BASE/'benchmarks/validation/validation_score_contract.py',
             BASE/'scoring/scoring_masks_bool.csv', BASE/'scoring/scoring_policy_v2.json',
             HERE/'comparison_protocol.json', Path(__file__), HERE/'numeric_metrics.py']
    # NPY identities are hashed only in explicit execution; no test array bytes
    # are opened during this default metadata preflight.
    policy = read_json(BASE/'scoring/scoring_policy_v2.json')
    require(policy['version'] == 2 and policy['primary_population']['test'] == 584 and policy['primary_population']['validation'] == 583, 'Scoring policy is not V2')
    identity = {str(p.resolve()): sha(p) for p in paths}
    expected = contract['frozen_inputs_sha256']
    for rel, digest in expected.items():
        require(sha(BASE/rel) == digest, 'Comparison input differs from declared contract: '+rel)
    return {'metadata': metadata, 'scalers': scalers, 'source_paths': source_paths,
            'identity': identity, 'runs_root': runs_root, 'contract': contract}


def six_fixed_runs(preflight):
    """Verify aggregate receipt and all six run receipts/histories/bytes first."""
    root = preflight['runs_root']
    receipt_path = root/'training_receipt.json'
    if not receipt_path.is_file():
        return None
    receipt = read_json(receipt_path)
    require(receipt['status'] == 'DECLARED_TRAINING_RUNS_FINISHED_CHECKPOINTS_FIXED_NO_TEST', 'Aggregate training is not complete')
    require(receipt['planned_run_count'] == 6 and len(receipt['runs']) == 6, 'Exactly six fixed runs are required')
    require(receipt['test_arrays_loaded'] is False and receipt['test_scores_computed'] is False, 'Training accessed test data')
    configs = {c.return_loss: asdict(c) for c in fixed_arms()}
    found = {}
    for run in receipt['runs']:
        arm = run['arm']['return_loss'];seed = run['seed']
        require(arm in configs and run['arm'] == configs[arm] and seed in SEEDS, 'A run is not the exact fixed configuration')
        key = (arm, seed)
        require(key not in found, 'Duplicate fixed run')
        folder = root/(arm+'_seed'+str(seed))
        require(folder.is_dir() and not folder.is_symlink(), 'Missing or symlinked own run directory')
        manifest = folder/'run_manifest.json';history = folder/'history.csv';checkpoint = folder/'best_checkpoint.pt'
        require(read_json(manifest) == run, 'Run receipt differs from aggregate: '+str(key))
        require(run['status'] == 'SUCCESS_VALIDATION_CHECKPOINT_FIXED_NO_TEST_ACCESS', 'Run not fixed successfully')
        require(run['test_arrays_loaded'] is False and run['test_scores_computed'] is False, 'Run reports test access')
        require(Path(run['best_checkpoint_path']).resolve() == checkpoint.resolve() and not checkpoint.is_symlink(), 'Checkpoint must be the exact new own-run best_checkpoint.pt')
        require(sha(checkpoint) == run['best_checkpoint_sha256'], 'Checkpoint hash mismatch')
        require(run['parameter_counts'] == {'total': 54536, 'return_specific': 27268, 'vol_specific': 27268, 'shared': 0}, 'Unexpected architecture parameter count')
        identity = run['dataset_identity'];metadata = preflight['metadata']
        require(identity['dataset_metadata_sha256'] == sha(DATASET/'dataset_metadata.json') and identity['scalers_sha256'] == metadata['scalers_sha256'], 'Run used different dataset/scaler')
        require(identity['denominator_sha256'] == metadata['source_sha256']['locked_denominators'], 'Run used another selection denominator')
        require(identity['validation_score_source_sha256'] == sha(BASE/'benchmarks/validation/validation_score_contract.py'), 'Run used another validation scorer')
        require(identity['test_arrays_read'] is False and identity['fit_counts'] == {'train': 2712, 'val': 583}, 'Run fit scope mismatch')
        rows = read_rows(history)
        require(len(rows) == run['epochs_ran'] and 45 <= len(rows) <= 100, 'Incomplete/invalid full training history')
        require([int(r['epoch']) for r in rows] == list(range(1, len(rows)+1)), 'Epoch history sequence mismatch')
        scores = [float(r['validation_score']) for r in rows]
        require(all(math.isfinite(v) for v in scores), 'Nonfinite history score')
        first_best = int(np.argmin(np.asarray(scores, dtype=np.float64)))+1
        require(first_best == run['best_epoch'] and scores[first_best-1] == run['best_validation_score'], 'Best checkpoint is not first minimum validation score')
        if len(rows) < 100:
            require(int(rows[-1]['no_improve']) >= 15, 'Early-stop history does not meet patience')
        require(run['runtime']['device'] == 'cuda', 'Declared six-run comparison requires CUDA runs')
        datetime.datetime.fromisoformat(run['completed_at_utc'])
        found[key] = {'run': run, 'folder': folder, 'checkpoint': checkpoint,
                      'manifest': reference(manifest), 'history': reference(history),
                      'checkpoint_reference': reference(checkpoint)}
    require(set(found) == {(a,s) for a in configs for s in SEEDS}, 'Missing arm/seed combination')
    for seed in SEEDS:
        a,b = (found[(arm,seed)]['run'] for arm in ('mse','mae'))
        require(a['initial_model_sha256'] == b['initial_model_sha256'], 'Paired initial model state differs')
        require(a['runtime'] == b['runtime'] and a['dataset_identity'] == b['dataset_identity'], 'Paired environment/data differ')
    return {'receipt': reference(receipt_path), 'runs': found}


def checked_array(stem, shape, dtype, metadata):
    path = DATASET/(stem+'.npy');record = metadata['arrays'][path.name]
    require(sha(path) == record['sha256'] and path.stat().st_size == record['size_bytes'], 'Array identity changed: '+stem)
    array = np.load(path, allow_pickle=False)
    require(array.shape == shape and array.dtype == np.dtype(dtype), 'Array shape/dtype mismatch: '+stem)
    require(list(array.shape) == record['shape'] and str(array.dtype) == record['dtype'], 'Array metadata mismatch: '+stem)
    if array.dtype.kind == 'f':
        require(np.isfinite(array).all(), 'Nonfinite dataset array')
    return array


def load_evaluation_arrays(preflight):
    result = {}
    for split in ('val','test'):
        n = EXPECTED_COUNTS[split]
        result[split] = {
            'X': checked_array('X_'+split, (n,10,8), 'float32', preflight['metadata']),
            'truth': checked_array('y_'+split+'_raw', (n,8), 'float64', preflight['metadata']),
            'dates': checked_array('target_dates_'+split, (n,), '<U10', preflight['metadata']),
            'input_end': checked_array('input_end_dates_'+split, (n,), '<U10', preflight['metadata']),
            'original_indices': checked_array('original_split_indices_'+split, (n,), 'int64', preflight['metadata'])}
        dates = result[split]['dates'].tolist()
        require(dates == sorted(set(dates)), 'Evaluation date order mismatch')
        require(np.all(result[split]['input_end'] < result[split]['dates']), 'Input boundary mismatch')
    require(result['val']['dates'][-1] == '2022-10-04' and result['test']['dates'][0] == '2022-10-06', 'Strict V2 evaluation boundary mismatch')
    require(result['val']['dates'][-1] <= result['test']['input_end'][0], 'Selection label exceeds first test information cutoff')
    return result


def mask_arrays(split, dates, original_indices):
    rows = [r for r in read_rows(BASE/'scoring/scoring_masks_bool.csv') if r['split'] == split]
    keyed = {}
    for row in rows:
        key = (int(row['original_split_sample_index']), row['asset'])
        require(key not in keyed, 'Duplicate scoring-mask date/asset key')
        keyed[key] = row
    masks = {};flags = {}
    all_columns = [c for pair in SCOPES.values() for c in pair]
    event_columns = ['known_cancelled_session_date_in_return_target_support', 'known_cancelled_session_date_in_vol20_target_support']
    for column in all_columns+event_columns:
        values = np.empty((len(dates),4), dtype=bool)
        for i,date in enumerate(dates):
            for j,asset in enumerate(ASSETS):
                row = keyed[(int(original_indices[i]),asset)]
                require(row['target_date'] == str(date), 'Canonical mask target-date/key alignment mismatch')
                require(row['protocol_version'] == PROTOCOL_VERSION and row['proposed_eligible'] == 'True', 'Mask row is not primary eligible')
                require(row[column] in ('True','False'), 'Invalid Boolean scoring mask')
                values[i,j] = row[column] == 'True'
        (flags if column in event_columns else masks)[column] = values
    require(masks['primary_return_mask'].all() and masks['primary_vol20_mask'].all(), 'Primary masks must retain every fixed target')
    if split == 'test':
        expected = read_json(BASE/'planning/full_training_20261113/post_fixed_checkpoint_scoring_contract.json')
        for short,column in expected['mask_columns'].items():
            for j,asset in enumerate(ASSETS):
                require(int(masks[column][:,j].sum()) == expected['test_counts'][asset][short], 'Locked test mask count changed')
    return masks,flags


def naive_predictions(arrays):
    """Exact locked raw parser/ffill/Vol20 at latest admitted label, not X32 inverse."""
    import pandas as pd
    raw = pd.read_csv(BASE/'inputs/raw_prices.csv', index_col=0, parse_dates=True,
                      dtype={a:'float64' for a in ASSETS})
    require(raw.shape == (3912,4) and raw.columns.tolist() == list(ASSETS), 'Raw shape/order mismatch')
    prices = raw.ffill();returns = np.log(prices/prices.shift(1))
    vol = returns.rolling(20,min_periods=20).std(ddof=1)*np.sqrt(252.)
    truth = np.column_stack([returns.to_numpy(),vol.to_numpy()])
    positions = {d:i for i,d in enumerate(raw.index.strftime('%Y-%m-%d'))}
    result = {}
    for split,data in arrays.items():
        target = [positions[str(d)] for d in data['dates']]
        admitted = [positions[str(d)] for d in data['input_end']]
        require(np.array_equal(truth[target], data['truth']), 'Canonical raw truth differs from saved float64 truth')
        result[split] = np.column_stack([np.zeros((len(target),4),dtype=np.float64), vol.to_numpy()[admitted]])
        require(np.isfinite(result[split]).all(), 'Nonfinite naive forecast')
    return result


def load_fixed_checkpoint_states(fixed, torch):
    """Only six exact, hashed, newly trained own checkpoint files; weights_only mandatory."""
    states = {}
    for key,entry in fixed['runs'].items():
        path = entry['checkpoint'];run = entry['run']
        require(sha(path) == entry['checkpoint_reference']['sha256'], 'Checkpoint changed before safe load')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        for field in ('arm','seed','best_epoch','dataset_identity','initial_model_sha256'):
            require(checkpoint[field] == run[field], 'Checkpoint receipt mismatch: '+field)
        require(checkpoint['validation_score'] == run['best_validation_score'], 'Checkpoint validation score mismatch')
        require(checkpoint['test_arrays_loaded'] is False and checkpoint['test_scores_computed'] is False, 'Checkpoint reports test access')
        state = checkpoint['model_state_dict']
        require(isinstance(state, dict) and state and all(isinstance(v,torch.Tensor) and torch.isfinite(v).all() for v in state.values()), 'Invalid model state')
        states[key] = {'state': state, 'score_object': checkpoint['validation_score_object']}
    return states


def predict_raw(model, X, scalers, torch, device):
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0,len(X),64):
            tensor = torch.from_numpy(X[start:start+64]).to(device)
            prediction = model(tensor).detach().cpu().numpy()
            require(prediction.shape == (len(tensor),8) and np.isfinite(prediction).all(), 'Invalid learned prediction')
            chunks.append(prediction)
    return inverse_y_float64(np.vstack(chunks), scalers)


def numeric_outputs(output, arrays, predictions, naive, masks_by_split):
    metric_rows=[];daily_rows=[];difference_rows=[];difference_summary=[];macros=[]
    for split,data in arrays.items():
        models = dict(predictions[split]);models['matched_naive'] = naive[split]
        masks,flags = masks_by_split[split]
        for label,prediction in models.items():
            err,absolute,squared,pin = errors_and_losses(data['truth'], prediction)
            for j,asset in enumerate(ASSETS):
                for task,column in [('return',j),('volatility',j+4)]:
                    for i,date in enumerate(data['dates']):
                        daily_rows.append({'split':split,'model':label,'target_date':str(date),'input_end_date':str(data['input_end'][i]),
                            'asset':asset,'task':task,'truth':float(data['truth'][i,column]),'prediction':float(prediction[i,column]),
                            'error_true_minus_pred':float(err[i,column]),'absolute_error':float(absolute[i,column]),
                            'squared_error':float(squared[i,column]),'pinball_tau05':float(pin[i,column]) if task=='volatility' else None,
                            'secondary_raw_target':bool(masks[SCOPES['secondary_raw_target'][task=='volatility']][i,j]),
                            'diagnostic_all_model_inputs':bool(masks[SCOPES['diagnostic_all_model_inputs'][task=='volatility']][i,j]),
                            'diagnostic_asset_inputs':bool(masks[SCOPES['diagnostic_asset_inputs'][task=='volatility']][i,j]),
                            'known_cancelled_session_in_target_support':bool(flags['known_cancelled_session_date_in_'+('return' if task=='return' else 'vol20')+'_target_support'][i,j])})
                    scopes = SCOPES if label.endswith('_ensemble') or label=='matched_naive' else {'primary':SCOPES['primary']}
                    for scope,columns in scopes.items():
                        mask = masks[columns[task=='volatility']][:,j]
                        if not mask.any():
                            metric_rows.append({'split':split,'model':label,'population':scope,'asset':asset,'task':task,'n':0,
                                'MAE':None,'MSE':None,'RMSE':None,'Pinball_tau05':None,'R2':None,'R2_status':'EMPTY_POPULATION',
                                'primary_loss_ratio_to_matched_naive':None,'ratio_status':'EMPTY_POPULATION'})
                            continue
                        obj = metrics(data['truth'][mask,column], prediction[mask,column], task=task)
                        baseline = metrics(data['truth'][mask,column], naive[split][mask,column], task=task)
                        loss = 'MAE' if task=='return' else 'Pinball_tau05';denominator=baseline[loss]
                        metric_rows.append({'split':split,'model':label,'population':scope,'asset':asset,'task':task,**obj,
                            'primary_loss_ratio_to_matched_naive':obj[loss]/denominator if denominator>0 else None,
                            'ratio_status':'DEFINED' if denominator>0 else 'UNDEFINED_NONPOSITIVE_DENOMINATOR'})
        for seed in (*SEEDS,'ensemble'):
            left = models['mse_'+('ensemble' if seed=='ensemble' else 'seed'+str(seed))]
            right = models['mae_'+('ensemble' if seed=='ensemble' else 'seed'+str(seed))]
            for j,asset in enumerate(ASSETS):
                for task,column in [('return',j),('volatility',j+4)]:
                    diff,summary = descriptive_difference(left[:,column],right[:,column],data['truth'][:,column],task=task)
                    for date,value in zip(data['dates'],diff):
                        difference_rows.append({'split':split,'paired_seed_or_ensemble':seed,'target_date':str(date),'asset':asset,'task':task,
                            'loss':'MAE' if task=='return' else 'Pinball_tau05','MSE_minus_MAE_loss':float(value)})
                    difference_summary.append({'split':split,'paired_seed_or_ensemble':seed,'population':'primary','asset':asset,'task':task,**summary})
    # Equal-asset means are separate from pooled daily observations. Secondary
    # scopes use each asset's predeclared support, with each n kept in metrics.
    keys = sorted({(r['split'],r['model'],r['population'],r['task']) for r in metric_rows})
    for split,model,population,task in keys:
        rows=[r for r in metric_rows if (r['split'],r['model'],r['population'],r['task'])==(split,model,population,task)]
        require(len(rows)==4, 'Macro must retain exactly four asset entries')
        values={}
        for field in ('MAE','MSE','RMSE','Pinball_tau05','R2','primary_loss_ratio_to_matched_naive'):
            values[field] = float(np.mean([r[field] for r in rows],dtype=np.float64)) if all(r[field] is not None for r in rows) else None
        macros.append({'split':split,'model':model,'population':population,'task':task,'aggregation':'equal_weight_mean_of_four_asset_metrics',
            'n_by_asset':';'.join(r['asset']+':'+str(r['n']) for r in rows),**values})
    write_csv(output/'metrics_by_asset.csv',metric_rows);write_csv(output/'metrics_equal_asset_macro.csv',macros)
    write_csv(output/'daily_errors_and_losses.csv',daily_rows)
    write_csv(output/'paired_loss_differences_daily.csv',difference_rows)
    write_csv(output/'paired_loss_differences_descriptive.csv',difference_summary)
    return metric_rows,macros,difference_summary


def execute(preflight, fixed, device_name):
    """Explicit only. No optimizer, backward, fit, selection, or extra training."""
    output = preflight['runs_root']/'comparison'
    require(not output.exists(), 'Comparison directory already exists; refuse repeat test access or overwrite')
    # All six manifest/history/hash checks above precede even importing torch.
    import torch
    require(device_name != 'cuda' or torch.cuda.is_available(), 'CUDA unavailable; no implicit CPU fallback')
    require(str(torch.__version__) == next(iter(fixed['runs'].values()))['run']['runtime']['torch'], 'Inference Torch version differs from fixed training environment')
    require(device_name == 'cuda', 'This fixed CUDA comparison has no CPU inference fallback')
    runtime = next(iter(fixed['runs'].values()))['run']['runtime']
    require(torch.version.cuda == runtime['cuda_build'] and torch.cuda.get_device_name(0) == runtime['gpu_name'], 'Inference CUDA/GPU differs from fixed training environment')
    device = torch.device(device_name)
    states = load_fixed_checkpoint_states(fixed,torch)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    output.mkdir(exist_ok=False)
    lock = {'status':'ALL_SIX_FIXED__EVALUATION_STARTED_NO_TEST_SELECTION','started_at_utc':utcnow(),
        'test_access_permitted_only_after_all_six_fixed':True,'six_training_receipt':fixed['receipt'],
        'runs':{a+'_seed'+str(s):{k:v for k,v in e.items() if k in ('manifest','history','checkpoint_reference')} for (a,s),e in fixed['runs'].items()},
        'input_sha256':preflight['identity'],'code_sha256':{p.name:sha(p) for p in (Path(__file__),HERE/'numeric_metrics.py')},
        'evaluation_input_files':{stem+'.npy':preflight['metadata']['arrays'][stem+'.npy'] for split in ('val','test') for stem in ('X_'+split,'y_'+split+'_raw','target_dates_'+split,'input_end_dates_'+split,'original_split_indices_'+split)},
        'checkpoint_or_arm_selection_by_test':False,'test_evaluation_repetition_allowed':False}
    write_json(output/'evaluation_input_lock.json',lock)
    try:
        arrays = load_evaluation_arrays(preflight)
        masks = {split:mask_arrays(split,data['dates'],data['original_indices']) for split,data in arrays.items()}
        naive = naive_predictions(arrays)
        from models import build_selected_model,parameter_counts
        from data_contract import load_score_function
        scorer = load_score_function();predictions={'val':{},'test':{}};score_rows=[];roundtrip=[]
        for arm in ('mse','mae'):
            for seed in SEEDS:
                label=arm+'_seed'+str(seed);entry=fixed['runs'][(arm,seed)]
                model=build_selected_model().to(device)
                require(parameter_counts(model)==entry['run']['parameter_counts'], 'Loaded architecture mismatch')
                model.load_state_dict(states[(arm,seed)]['state'],strict=True)
                prediction=predict_raw(model,arrays['val']['X'],preflight['scalers'],torch,device)
                predictions['val'][label]=prediction
                np.save(output/('predictions_raw_val_'+label+'.npy'),prediction,allow_pickle=False)
                obj=scorer(arrays['val']['truth'],predictions['val'][label],arrays['val']['dates'].tolist(),
                    output_order=list(OUTPUT_ORDER),denominator_path=preflight['source_paths']['locked_denominators'])
                old=states[(arm,seed)]['score_object']
                delta=abs(obj['validation_score']-entry['run']['best_validation_score'])
                require(delta<=1e-12, 'Fixed checkpoint validation score did not reproduce within1e-12')
                numeric_differences=[abs(obj[k]-old[k]) for k in ('validation_score','mean_return_ratio','mean_volatility_ratio')]
                numeric_differences += [abs(obj['per_asset'][a][k]-old['per_asset'][a][k]) for a in ASSETS for k in ('return_MAE','vol_Pinball_tau05','return_ratio','vol_ratio')]
                require(max(numeric_differences)<=1e-12, 'Per-asset validation checkpoint reproduction failed')
                roundtrip.append({'model':label,'score_absolute_difference_from_training':delta,'tolerance':1e-12,
                    'maximum_numeric_score_object_difference':max(numeric_differences),
                    'full_score_object_equal':obj==old})
                score_rows.append({'model':label,**{k:obj[k] for k in ('validation_score','mean_return_ratio','mean_volatility_ratio','n_per_output')}})
                del model
        # All six fixed models must first reproduce their selection-validation
        # score. No test forward or test-derived choice can precede this gate.
        for arm in ('mse','mae'):
            for seed in SEEDS:
                label=arm+'_seed'+str(seed)
                model=build_selected_model().to(device)
                model.load_state_dict(states[(arm,seed)]['state'],strict=True)
                prediction=predict_raw(model,arrays['test']['X'],preflight['scalers'],torch,device)
                predictions['test'][label]=prediction
                np.save(output/('predictions_raw_test_'+label+'.npy'),prediction,allow_pickle=False)
                del model
        for arm in ('mse','mae'):
            for split in ('val','test'):
                ensemble=raw_seed_ensemble([predictions[split][arm+'_seed'+str(seed)] for seed in SEEDS])
                predictions[split][arm+'_ensemble']=ensemble
                np.save(output/('predictions_raw_'+split+'_'+arm+'_ensemble.npy'),ensemble,allow_pickle=False)
            obj=scorer(arrays['val']['truth'],predictions['val'][arm+'_ensemble'],arrays['val']['dates'].tolist(),
                output_order=list(OUTPUT_ORDER),denominator_path=preflight['source_paths']['locked_denominators'])
            score_rows.append({'model':arm+'_ensemble',**{k:obj[k] for k in ('validation_score','mean_return_ratio','mean_volatility_ratio','n_per_output')}})
        write_csv(output/'validation_scores_fixed_checkpoints.csv',score_rows)
        for split,data in arrays.items():
            np.save(output/('truth_raw_'+split+'.npy'),data['truth'],allow_pickle=False)
            np.save(output/('target_dates_'+split+'.npy'),data['dates'],allow_pickle=False)
            np.save(output/('matched_naive_raw_'+split+'.npy'),naive[split],allow_pickle=False)
        metric_rows,macros,differences=numeric_outputs(output,arrays,predictions,naive,masks)
        for path,digest in preflight['identity'].items():require(sha(path)==digest, 'Input changed during evaluation: '+path)
        for entry in fixed['runs'].values():
            for field in ('manifest','history','checkpoint_reference'):
                ref=entry[field];require(sha(ref['path'])==ref['sha256'],'Fixed run evidence changed during evaluation')
        summary={'status':'SUCCESS_SIX_FIXED_CHECKPOINTS_EVALUATED_ONCE_NO_TEST_SELECTION','completed_at_utc':utcnow(),
            'input_lock_sha256':sha(output/'evaluation_input_lock.json'),'runtime':{'python':sys.version,'platform':platform.platform(),
                'numpy':np.__version__,'torch':str(torch.__version__),'device':str(device),'cuda_build':torch.version.cuda},
            'output_order':list(OUTPUT_ORDER),'seeds':list(SEEDS),'primary_counts':{'validation':583,'test':584},
            'prediction_dtype':'float64','ensemble':'arithmetic mean of three raw float64 seed predictions',
            'checkpoint_validation_reproduction':roundtrip,'validation_scores':score_rows,
            'metrics_by_asset_rows':len(metric_rows),'macro_rows':len(macros),'paired_difference_summary_rows':len(differences),
            'test_evaluation_performed':True,'test_checkpoint_or_arm_selection':False,'training_performed':False,
            'GARCH_fit':False,'SHAP':False,'new_price_download':False,'thesis_edit':False,'inferential_tests_performed':False,
            'limitations':['Three seeds are not1752independenttestobservations; daily paired differences are descriptive.',
                'P1MSEversusP1MAE changes onlyreturntrainingloss under matched protocol;fixed0.7/0.3 does not equate effectivegradients.',
                'Secondarysupport masks do not prove freshtradeorhistoricalavailability; eventflags never autoexclude.',
                'P0versusP1 is acombinedprotocolchange, not isolated delay or certifiedclockcorrection.'],
            'outputs':{p.name:{'sha256':sha(p),'size_bytes':p.stat().st_size} for p in output.iterdir() if p.is_file()}}
        write_json(output/'comparison_summary.json',summary)
        report='P1 V2 — MSE ve MAE eğitim kollarının sabit modellerle karşılaştırılması\n\n'
        report+='Her iki koldaki üç kontrol noktası, yalnız 583 gözlemlik doğrulama kümesinden seçildikten sonra test kümesi değerlendirildi. Test sonuçlarıyla model, tohum veya kayıp kolu seçilmedi.\n\n'
        report+='Ana test kapsamı, her varlık ve görev için aynı 584 hedeftir. İkincil ham fiyat desteği maskeleri ve tüm girdi desteği tanıları ayrı satırlarda, gözlem sayılarıyla verilmiştir. Ham fiyatın bulunması yeni işlem veya tarihsel erişilebilirlik kanıtı sayılmaz.\n\n'
        report+='Tahminler önce float64 ile ham ölçeğe çevrildi; üç tohumun ham tahminleri eşit ağırlıkla ortalandı. Getiri sıfır tahmini ve son kabul edilen t−1 durumundan Vol20 kalıcılık tahmini aynı tarihlerde puanlandı.\n\n'
        report+='Ana test kapsamındaki eşit varlık ortalamaları:\n'
        for model in ('mse_ensemble','mae_ensemble','matched_naive'):
            for task in ('return','volatility'):
                row=next(r for r in macros if r['split']=='test' and r['model']==model and r['population']=='primary' and r['task']==task)
                key='MAE' if task=='return' else 'Pinball_tau05'
                report+=f"  {model}, {'getiri MAE' if task=='return' else 'volatilite pinball (τ=0,5)'}: {row[key]:.10g}; eşit varlık hata oranı: {row['primary_loss_ratio_to_matched_naive']:.10g}.\n"
        report+='Bu ortalamalar farklı büyüklükteki ham hata sayılarını ve aynı kapsamda hesaplanan naif oranlarını ayrı gösterir.\n\n'
        report+='metrics_by_asset.csv ve metrics_equal_asset_macro.csv hata ölçülerini; validation_scores_fixed_checkpoints.csv kanonik doğrulama skorlarını; daily_errors_and_losses.csv günlük hesapları içerir. paired_loss_differences_* dosyalarında pozitif MSE−MAE farkı, MSE kolunda daha büyük değerlendirme kaybı demektir. Bu farklar betimseldir; yeni anlamlılık testi yapılmadı. Üç tohum bağımsız 1.752 test gözlemi olarak ele alınmadı.\n\n'
        report+='MSE–MAE karşılaştırmasında diğer ayarlar aynıdır; 0,7/0,3 ağırlıklarının korunması etkin gradyan dengesini eşitlemez. NoSharing dalları parametre paylaşmasa da küresel gradyan kırpma ve ortak doğrulama skoru ile kontrol noktası seçimi, getiri kaybı değiştiğinde volatilite sonuçlarını da etkileyebilir. Bu değişim tek başına ortak temsil veya pozitif görev aktarımı kanıtı değildir.\n\n'
        report+='P0–P1 farkı bilgi kesiti, sınır kapsamı, ölçekleme ve sayısal hassasiyetin ortak değişimidir. Mevcut test dönemi yeniden kullanılmıştır; bu yeni bir dış dönem değildir. Bu çıktı saat düzeyinde sızıntısızlık kanıtı veya tez değişikliği değildir.\n'
        (output/'Karsilastirma_Notu.txt').write_text(report,encoding='utf-8')
        write_json(output/'OUTPUT_MANIFEST.json',{'status':summary['status'],'files':{p.name:reference(p) for p in output.iterdir() if p.is_file()}})
        return summary
    except Exception as exc:
        write_json(output/'evaluation_failure.json',{'status':'FAILED_RETAINED_AUDIT_NO_SILENT_REPEAT','at_utc':utcnow(),
            'error':repr(exc),'test_arrays_may_have_been_read':True,'automatic_retry_or_new_test_selection_allowed':False})
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root',type=Path,default=DEFAULT_RUNS)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--execute-evaluation',action='store_true')
    args=parser.parse_args(argv)
    preflight=source_preflight(args.runs_root)
    fixed=six_fixed_runs(preflight)
    plan={'status':'READY_ALL_SIX_FIXED__DRY_RUN_NO_MODELS_OR_TEST_ARRAYS' if fixed else 'WAITING_FOR_ALL_SIX_FIXED__DRY_RUN',
        'default_dry_run':not args.execute_evaluation,'runs_root':str(preflight['runs_root']),
        'six_fixed':fixed is not None,'torch_imported':False,'model_constructed':False,'checkpoint_deserialized':False,
        'test_arrays_read':False,'learned_inference':False,'training':False,'test_selection':False,
        'future_output':str(preflight['runs_root']/'comparison')}
    if not args.execute_evaluation:
        print(json.dumps(plan,ensure_ascii=False,indent=2));return 0
    require(fixed is not None,'All six fixed successful checkpoints are required before evaluation')
    summary=execute(preflight,fixed,args.device)
    print(json.dumps({'status':summary['status'],'output':plan['future_output'],'training':False,'test_selection':False},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
