"""Default metadata-only preflight. Explicit pilot/main flags permit input-gradient SHAP.

No fitting, optimizer, backward into model parameters, or thesis writes. Only six
new own P1 V2 checkpoints are loaded, with weights_only=True, after identity gates.
"""
from pathlib import Path
import argparse
import contextlib
import datetime
import hashlib
import importlib.util
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
RUN=BASE/'training/runs/p1_v2_mse_mae_20261113_full'
DATASET=BASE/'datasets/p1_v2'
PROTOCOL=HERE/'shap_stability_protocol_v2.json'
MODEL_SEEDS=(123,777,2026)
MC_SEEDS=(2026111321,2026111322)
LAGS=[f't-{i}' for i in range(10,0,-1)]


def require(condition,message):
    if not condition:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def readj(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def writej(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def numeric_sha(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def module_at(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def preflight():
    require(sys.flags.optimize==0,'Python -O is unsupported')
    require(PROTOCOL.is_file(),'Root SHAP protocol is not yet ready')
    protocol=readj(PROTOCOL)
    require(protocol['version']==2,'Only the transparent V2 gradient-compatibility protocol is executable')
    require(protocol['gradient_adapter']['module_file']=='gradient_compatible_adapter.py','Unexpected numerical adapter')
    require(protocol['gradient_adapter']['forward_tolerance_unchanged']==protocol['prediction_match_max_abs_standardized'],'Gradient adapter changed the original prediction tolerance')
    validation=HERE/protocol['gradient_adapter']['validation_file']
    require(validation.parent.resolve()==HERE and sha(validation)==protocol['gradient_adapter']['validation_sha256'],'All-six forward diagnostic identity changed')
    require(readj(validation)['all_checks_passed'] is True,'All-six test/background/interpolant forward checks did not pass')
    require(protocol['nsamples']==256 and protocol['integration_seeds']==list(MC_SEEDS),'Integration protocol mismatch')
    require(protocol['expected_shap_version']=='0.52.0','SHAP version must be the declared0.52.0')
    require(protocol['allowed_batch_sizes']==[64,256],'Unexpected technical batch alternatives')
    require(0<protocol['prediction_match_max_abs_standardized']<=1e-5,'Invalid prediction agreement bound')
    plan_path=HERE/protocol['sample_plan_file']
    require(plan_path.parent.resolve()==HERE,'Sample plan must be in the new analysis workspace')
    selector=module_at('new_p1_shap_sample_selection',HERE/'sample_selection.py')
    plan=selector.load_and_validate_sample_plan(plan_path,expected_plan_sha256=protocol['sample_plan_sha256'],integration_seeds=MC_SEEDS)
    require(plan['explained']['n']==128 and plan['explained']['source_n']==584,'Only the fixed128-of584 explanation plan is permitted')
    require(len(plan['backgrounds'])==4 and all(b['n']==64 and b['split']=='train' for b in plan['backgrounds']),'Expected four training-only64-row backgrounds')
    # The already reviewed comparator has a script-mode sibling import. Load
    # that sibling in a temporary namespace/path without changing its source
    # or accidentally using a different analysis module named numeric_metrics.
    previous_numeric=sys.modules.pop('numeric_metrics',None);previous_path=list(sys.path)
    sys.path.insert(0,str(BASE/'training/comparison'))
    try:compare=module_at('prepared_p1_six_fixed_checkpoint_contract',BASE/'training/comparison/evaluate_fixed_runs.py')
    finally:
        sys.path[:]=previous_path;sys.modules.pop('numeric_metrics',None)
        if previous_numeric is not None:sys.modules['numeric_metrics']=previous_numeric
    # The model and data-contract modules are imported only beyond explicit
    # execution gates; keep their reviewed training directory discoverable.
    if str(BASE/'training') not in sys.path:sys.path.insert(0,str(BASE/'training'))
    data=compare.source_preflight(RUN);fixed=compare.six_fixed_runs(data)
    require(fixed is not None,'All six fixed checkpoints are required')
    comp_manifest=readj(RUN/'comparison/OUTPUT_MANIFEST.json')
    require(comp_manifest['status']=='SUCCESS_SIX_FIXED_CHECKPOINTS_EVALUATED_ONCE_NO_TEST_SELECTION','Existing comparison did not complete')
    sources={str(PROTOCOL):sha(PROTOCOL),str(plan_path):sha(plan_path),
        str(HERE/'sample_selection.py'):sha(HERE/'sample_selection.py'),str(Path(__file__)):sha(__file__),
        str(HERE/'stability_metrics.py'):sha(HERE/'stability_metrics.py'),
        str(HERE/'gradient_compatible_adapter.py'):sha(HERE/'gradient_compatible_adapter.py'),
        str(HERE/'verify_v2_candidate_batch128.py'):sha(HERE/'verify_v2_candidate_batch128.py'),
        str(validation):sha(validation),str(HERE/'method_addendum_v2.json'):sha(HERE/'method_addendum_v2.json'),
        str(RUN/'training_receipt.json'):sha(RUN/'training_receipt.json'),
        str(RUN/'comparison/OUTPUT_MANIFEST.json'):sha(RUN/'comparison/OUTPUT_MANIFEST.json')}
    environment_path=HERE/'environment/environment_receipt.json'
    environment=readj(environment_path)
    require(environment['status']=='PASS_SHAP_EXTENSION_EXISTING_PACKAGES_UNCHANGED','SHAP environment extension did not pass')
    library=Path(environment['shap_gradient_source_path'])
    require(sha(library)==environment['shap_gradient_source_sha256'],'Installed SHAP gradient source changed')
    sources[str(environment_path)]=sha(environment_path);sources[str(library)]=sha(library)
    for name,digest in protocol['source_hashes'].items():
        require(sha(BASE/name)==digest,'Protocol source changed: '+name);sources[str(BASE/name)]=digest
    for name,digest in plan['source_sha256'].items():sources[str(BASE/name)]=digest
    addendum=readj(HERE/'method_addendum_v2.json')
    require(addendum['new_protocol']['sha256']==sha(PROTOCOL),'Technical method addendum is bound to another V2 protocol')
    for arm in ('mse','mae'):
        for seed in MODEL_SEEDS:
            path=RUN/'comparison'/f'predictions_raw_test_{arm}_seed{seed}.npy'
            require(sha(path)==comp_manifest['files'][path.name]['sha256'],'Saved prediction identity changed')
            sources[str(path)]=sha(path)
    return {'protocol':protocol,'plan':plan,'compare':compare,'data':data,'fixed':fixed,'sources':sources,
        'sample_plan_sha256':sha(plan_path),'protocol_sha256':sha(PROTOCOL),'environment_receipt':environment}


def common_paths(n,nsamples,rseed,background_n=64):
    """Match SHAP0.52MT19937 choice-then-uniform order; reused for each output."""
    rng=np.random.RandomState(rseed)
    slots=np.empty((n,nsamples),dtype=np.int64);alpha=np.empty((n,nsamples),dtype=np.float64)
    for i in range(n):
        for j in range(nsamples):
            slots[i,j]=rng.choice(background_n);alpha[i,j]=rng.uniform()
    return slots,alpha


def canonical_phi(value,n):
    if isinstance(value,list):
        require(len(value)==8,'Expected exactly eight output arrays')
        value=np.stack([np.asarray(v,dtype=np.float64) for v in value],axis=-1)
    value=np.asarray(value,dtype=np.float64)
    require(value.shape==(n,10,8,8) and np.isfinite(value).all(),'Expected finite[sample,lag,feature,output] array')
    return value


def save_npz(path,values):
    path=Path(path);require(not path.exists(),'Refuse attribution overwrite')
    temporary=path.with_suffix('.npz.part')
    require(not temporary.exists(),'Incomplete prior NPZ requires explicit review')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**values)
    temporary.replace(path)


def check_sources(payload):
    for path,digest in payload['sources'].items():require(sha(path)==digest,'Source changed: '+path)


@contextlib.contextmanager
def input_gradient_backend(torch):
    previous=torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            yield
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def environment_and_data(payload):
    import torch
    import shap
    require(torch.cuda.is_available(),'CUDA required; no CPU fallback')
    require(shap.__version__==payload['protocol']['expected_shap_version'],'Unexpected installed SHAP version')
    for name,version in payload['environment_receipt']['packages_after'].items():
        require(importlib.metadata.version(name)==version,'Execution package changed: '+name)
    ref=next(iter(payload['fixed']['runs'].values()))['run']['runtime']
    require(np.__version__==ref['numpy'],'NumPy differs from fixed training environment')
    require(str(torch.__version__)==ref['torch'] and torch.version.cuda==ref['cuda_build'] and torch.cuda.get_device_name(0)==ref['gpu_name'],'CurrentCUDAenvironment differs from six-run environment')
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    metadata=payload['data']['metadata'];compare=payload['compare']
    Xtrain=compare.checked_array('X_train',(2712,10,8),'float32',metadata)
    Xtest=compare.checked_array('X_test',(584,10,8),'float32',metadata)
    # y_test is deliberately not read: explanation sample selection and model
    # explanation do not require target values or error-based choices.
    state=compare.load_fixed_checkpoint_states(payload['fixed'],torch)
    sample_indices=np.asarray(payload['plan']['explained']['indices'],dtype=np.int64)
    X=Xtest[sample_indices]
    predictions={}
    for arm in ('mse','mae'):
        for seed in MODEL_SEEDS:
            path=RUN/'comparison'/f'predictions_raw_test_{arm}_seed{seed}.npy'
            saved=np.load(path,allow_pickle=False)
            require(saved.shape==(584,8) and saved.dtype==np.float64 and np.isfinite(saved).all(),'Invalid saved prediction')
            predictions[(arm,seed)]=saved[sample_indices]
    gradient_source=Path(shap.explainers._gradient.__file__).resolve()
    require(sha(gradient_source)==payload['environment_receipt']['shap_gradient_source_sha256'],'Imported SHAP gradient implementation differs from verified installation')
    environment={'python':sys.version,'numpy':np.__version__,'torch':str(torch.__version__),'shap':shap.__version__,
        'cuda_build':torch.version.cuda,'gpu_name':torch.cuda.get_device_name(0),
        'gradient_library_source_path':str(gradient_source),'gradient_library_source_sha256':sha(gradient_source)}
    return torch,shap,state,Xtrain,X,predictions,environment


def build_frozen_model(state,torch):
    from models import build_selected_model
    from train_p1 import state_hash
    model=build_selected_model().to('cuda')
    model.load_state_dict(state,strict=True);model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    require(all(not p.requires_grad and p.grad is None for p in model.parameters()),'Model parameter gradients must be disabled')
    return model,state_hash(model)


def scaled_prediction(model,values,torch):
    with torch.no_grad():
        result=model(torch.from_numpy(np.asarray(values,dtype=np.float32)).to('cuda')).detach().cpu().numpy()
    require(result.shape==(len(values),8) and np.isfinite(result).all(),'Invalid model output')
    return np.asarray(result,dtype=np.float64)


def verify_v2_forward_receipt(payload):
    path=HERE/'v2_batch128_equivalence.json'
    value=readj(path)
    require(value['status']=='PASS_V2_ALL_SIX_BATCH128_SAVED_PREDICTIONS' and value['model_count']==6,'All six B128 candidate forwards must pass before new SHAP pilot/main')
    require(value['protocol_sha256']==payload['protocol_sha256'] and value['sample_plan_sha256']==payload['sample_plan_sha256'],'B128 verification refers to another protocol')
    require(value['source_sha256']==payload['sources'],'B128 verification source identities differ')
    require(value['input_or_parameter_gradients_computed'] is False and value['all_model_states_and_original_objects_unchanged'] is True,'B128 diagnostic exceeded read-only forward scope')
    return {'path':str(path),'sha256':sha(path)}


def one_attribution(model,X,background,mcseed,batch_size,torch,shap,scalers,metrics):
    tensor=torch.from_numpy(np.asarray(X,dtype=np.float32)).to('cuda')
    bg=torch.from_numpy(np.asarray(background,dtype=np.float32)).to('cuda')
    torch.manual_seed(mcseed);torch.cuda.manual_seed_all(mcseed)
    np.random.seed(mcseed)
    torch.cuda.synchronize();start=time.perf_counter()
    from gradient_compatible_adapter import compatible_encoder_input_gradients
    with compatible_encoder_input_gradients(model,torch):
        prediction=scaled_prediction(model,X,torch)
        expected=np.mean(scaled_prediction(model,background,torch),axis=0,dtype=np.float64)
        explainer=shap.GradientExplainer(model,bg,batch_size=batch_size,local_smoothing=0.)
        value=explainer.shap_values(tensor,nsamples=256,ranked_outputs=None,rseed=mcseed,return_variances=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    require(isinstance(value,tuple) and len(value)==2,'Unexpected GradientExplainer result')
    phi=canonical_phi(value[0],len(X));diagnostic=canonical_phi(value[1],len(X))
    require(np.all(diagnostic>=0),'Negative library variance diagnostic')
    ymean=np.asarray(scalers['y']['mean'],dtype=np.float64);yscale=np.asarray(scalers['y']['scale'],dtype=np.float64)
    rawphi,rawexpected,rawprediction=metrics.affine_output_transform(phi,expected,prediction,ymean,yscale)
    return {'phi_scaled':phi,'phi_raw':rawphi,'expected_scaled':expected,'expected_raw':rawexpected,
        'prediction_scaled':prediction,'prediction_raw':rawprediction,
        'library_variance_diagnostic_scaled':diagnostic,
        'library_variance_diagnostic_raw':diagnostic*yscale.reshape(1,1,1,8)**2},elapsed


def prediction_agreement(raw,saved,scale,limit):
    differences=np.abs(np.asarray(raw,dtype=np.float64)-np.asarray(saved,dtype=np.float64))
    normalized=differences/np.asarray(scale,dtype=np.float64)
    require(np.max(normalized)<=limit,'Gradient-compatible backend changes saved predictions beyond declared tolerance')
    return {'maximum_abs_raw_difference':float(np.max(differences)),
        'maximum_abs_difference_over_target_scale':float(np.max(normalized)),'declared_normalized_tolerance':limit}


def diagnostic_record(values,payload,metrics):
    y=payload['data']['scalers']['y']
    return {'feature_importance_scaled':metrics.feature_importance(values['phi_scaled']).tolist(),
        'feature_importance_raw':metrics.feature_importance(values['phi_raw']).tolist(),
        'temporal_importance_scaled':metrics.temporal_importance(values['phi_scaled']).tolist(),
        'additivity_diagnostics':metrics.additivity_diagnostics(values['phi_scaled'],values['prediction_scaled'],values['expected_scaled'],
            np.asarray(y['mean']),np.asarray(y['scale']),payload['plan']['output_order'])}


def log_event(path,event):
    with Path(path).open('a',encoding='utf-8') as stream:
        stream.write(json.dumps({'at_utc':now(),**event},ensure_ascii=False,allow_nan=False)+'\n');stream.flush()


def pilot_worker(payload,output):
    torch,shap,states,Xtrain,X,saved,environment=environment_and_data(payload)
    metrics=module_at('new_p1_shap_stability_numeric',HERE/'stability_metrics.py')
    model,model_hash=build_frozen_model(states[('mse',123)]['state'],torch)
    bg=payload['plan']['backgrounds'][0];require(bg['id']=='linspace_reference','Pilot background must be deterministic reference')
    X=X[:4];background=Xtrain[np.asarray(bg['indices'])]
    timings=[];phis=[]
    from train_p1 import state_hash
    from gradient_compatible_adapter import compatible_encoder_input_gradients,ADAPTER_ID
    with compatible_encoder_input_gradients(model,torch):
        raw=payload['compare'].inverse_y_float64(scaled_prediction(model,X,torch),payload['data']['scalers'])
    prediction_agreement(raw,saved[('mse',123)][:4],payload['data']['scalers']['y']['scale'],payload['protocol']['prediction_match_max_abs_standardized'])
    for batch_size in (64,256):
        # Both implementations use exactly the same4samples, background andMC
        # paths. nsamples/output/sample choices never depend on pilot results.
        torch.cuda.reset_peak_memory_stats()
        try:
            values,elapsed=one_attribution(model,X,background,MC_SEEDS[0],batch_size,torch,shap,payload['data']['scalers'],metrics)
        except torch.cuda.OutOfMemoryError as exc:
            require(batch_size==256,'Reference batch64 exhausted GPU memory')
            torch.cuda.empty_cache();timings.append({'batch_size':256,'status':'OUT_OF_MEMORY_RETAIN_BATCH64','error':str(exc)})
            break
        agreement=prediction_agreement(values['prediction_raw'],saved[('mse',123)][:4],payload['data']['scalers']['y']['scale'],payload['protocol']['prediction_match_max_abs_standardized'])
        require(state_hash(model)==model_hash and all(p.grad is None and not p.requires_grad for p in model.parameters()),'Pilot changed model state/parameter gradients')
        save_npz(output/f'pilot_batch{batch_size}.npz',values)
        peak=int(torch.cuda.max_memory_allocated());total=int(torch.cuda.get_device_properties(0).total_memory)
        if peak>=.95*total:
            require(batch_size==256,'Reference batch64 used over95percent of GPU memory')
            timings.append({'batch_size':256,'status':'NEAR_MEMORY_LIMIT_RETAIN_BATCH64',
                'peak_allocated_gpu_bytes':peak,'total_gpu_bytes':total,'seconds':elapsed})
            log_event(output/'progress.jsonl',{'status':'PILOT_BATCH256_MEMORY_GUARD_RETAIN64',**timings[-1]})
            break
        timings.append({'batch_size':batch_size,'status':'FINITE_STATE_AND_PREDICTION_PASS','explained_samples':4,'seconds':elapsed,
            'peak_allocated_gpu_bytes':peak,'total_gpu_bytes':total,
            'rough_48job_N128_seconds':elapsed*48*128/4,'rough_48job_N584_seconds':elapsed*48*584/4,
            'linear_projection_is_not_runtime_guarantee':True,'prediction_agreement':agreement})
        phis.append(values['phi_scaled']);log_event(output/'progress.jsonl',{'status':'PILOT_BATCH_COMPLETE',**timings[-1]})
    check_sources(payload)
    phi_difference=float(np.max(np.abs(phis[0]-phis[1]))) if len(phis)==2 else None
    phi_limit=payload['protocol']['pilot_batch_phi_max_abs_scaled_difference']
    result={'status':'PASS_BOUNDED_INPUT_GRADIENT_PILOT_ONLY','environment':environment,'timings':timings,
        'adapter_id':ADAPTER_ID,'input_lock_sha256':sha(output/'pilot_input_lock.json'),
        'source_sha256':payload['sources'],'all_original_encoder_and_head_objects_restored':True,
        'same_path_scaled_phi_max_abs_batch64_vs256':phi_difference,'phi_agreement_tolerance':phi_limit,
        'batch256_eligible':len(phis)==2 and phi_difference<=phi_limit,
        'sample_count':4,'nsamples':256,'output_count':8,'sample_plan_sha256':payload['sample_plan_sha256'],
        'protocol_sha256':payload['protocol_sha256'],'all_model_parameters_frozen_and_unchanged':True,
        'model_training_or_optimizer':False,'model_input_gradients_computed':True,'main_analysis_started':False,
        'batch_choice_requires_separate_technical_decision_before_main':True}
    writej(output/'pilot_result.json',result)


def launch_pilot(payload,args):
    forward_receipt=verify_v2_forward_receipt(payload)
    output=(HERE/'pilot_runs'/args.pilot_name).resolve()
    require(output.parent==(HERE/'pilot_runs').resolve() and args.pilot_name not in ('','.', '..'),'Pilot path must be a fresh direct child')
    require(not output.exists(),'Pilot output exists; no implicit repeat')
    output.mkdir(parents=True)
    writej(output/'pilot_input_lock.json',{'protocol_sha256':payload['protocol_sha256'],'sample_plan_sha256':payload['sample_plan_sha256'],'source_sha256':payload['sources'],'external_timeout_seconds':180,'B128_forward_verification':forward_receipt})
    command=[sys.executable,str(Path(__file__).resolve()),'--internal-pilot-worker','--pilot-name',args.pilot_name]
    env=dict(os.environ);env['P1_SHAP_PILOT_WORKER']='1'
    started=time.perf_counter()
    with (output/'worker.log').open('w',encoding='utf-8') as log:
        try:
            process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=180,check=False)
            require(process.returncode==0,'Pilot worker failed; see worker.log')
            require(readj(output/'pilot_result.json')['status']=='PASS_BOUNDED_INPUT_GRADIENT_PILOT_ONLY','Pilot did not pass')
            result=readj(output/'pilot_result.json')
            require(result['input_lock_sha256']==sha(output/'pilot_input_lock.json') and result['source_sha256']==payload['sources'],'Pilot worker source/input lock mismatch')
            check_sources(payload)
            status='PASS_BOUNDED_INPUT_GRADIENT_PILOT_ONLY';timed_out=False
        except subprocess.TimeoutExpired:
            status='PILOT_TIMEOUT_NO_MAIN_ANALYSIS';timed_out=True
        except Exception as exc:
            writej(output/'parent_result.json',{'status':'PILOT_FAILED_NO_MAIN_ANALYSIS','error':repr(exc),'seconds':time.perf_counter()-started});raise
    writej(output/'parent_result.json',{'status':status,'timed_out':timed_out,'seconds':time.perf_counter()-started,'external_timeout_seconds':180,'training':False})
    require(not timed_out,'Pilot exceeded external180-second limit')
    print(json.dumps({'status':status,'output':str(output)},ensure_ascii=False))


def main_analysis(payload,args):
    forward_receipt=verify_v2_forward_receipt(payload)
    decision=readj(HERE/'technical_batch_decision.json')
    require(decision['status']=='APPROVED_TECHNICAL_BATCH_AFTER_PILOT' and decision['batch_size'] in (64,256),'Pilot-based batch decision is not ready')
    require(decision['protocol_sha256']==payload['protocol_sha256'] and decision['sample_plan_sha256']==payload['sample_plan_sha256'],'Technical decision refers to another protocol')
    require(decision['source_sha256']==payload['sources'],'Code/input/environment identities changed after pilot decision')
    pilot_path=Path(decision['pilot_result_path']).resolve()
    require((HERE/'pilot_runs').resolve() in pilot_path.parents and pilot_path.name=='pilot_result.json','Decision pilot must be a new own pilot result')
    require(sha(pilot_path)==decision['pilot_result_sha256'],'Decision pilot result identity changed')
    pilot=readj(pilot_path)
    require(pilot['status']=='PASS_BOUNDED_INPUT_GRADIENT_PILOT_ONLY' and pilot['protocol_sha256']==payload['protocol_sha256'] and pilot['sample_plan_sha256']==payload['sample_plan_sha256'],'Actual pilot did not pass this exact protocol')
    require(readj(pilot_path.parent/'pilot_input_lock.json')['source_sha256']==payload['sources'],'Pilot used other code/input/environment identities')
    require(pilot['input_lock_sha256']==sha(pilot_path.parent/'pilot_input_lock.json') and pilot['source_sha256']==payload['sources'],'Pilot worker was not bound to its parent source lock')
    require(readj(pilot_path.parent/'parent_result.json')['status']=='PASS_BOUNDED_INPUT_GRADIENT_PILOT_ONLY','Pilot parent/watchdog did not pass')
    require(decision['batch_size']==64 or pilot['batch256_eligible'] is True,'Batch256 failed the declared pilot agreement/memory checks')
    output=RUN/'shap_stability_20261113'
    lock={'protocol_sha256':payload['protocol_sha256'],'sample_plan_sha256':payload['sample_plan_sha256'],
        'source_sha256':payload['sources'],'batch_size':decision['batch_size'],'technical_batch_decision_sha256':sha(HERE/'technical_batch_decision.json'),
        'B128_forward_verification':forward_receipt,
        'environment_identity':{'receipt_sha256':sha(HERE/'environment/environment_receipt.json'),
            'expected_package_versions':payload['environment_receipt']['packages_after'],
            'shap_gradient_source_sha256':payload['environment_receipt']['shap_gradient_source_sha256'],
            'training_runtime':next(iter(payload['fixed']['runs'].values()))['run']['runtime']}}
    if output.exists():
        require(args.resume and readj(output/'analysis_input_lock.json')==lock,'Only exact-identity explicit resume is allowed')
        require(not (output/'analysis_receipt.json').exists(),'Completed analysis cannot be implicitly repeated')
    else:
        require(not args.resume,'Cannot resume a missing output');output.mkdir();writej(output/'analysis_input_lock.json',lock)
    torch,shap,states,Xtrain,X,saved,environment=environment_and_data(payload)
    metrics=module_at('new_p1_shap_stability_numeric',HERE/'stability_metrics.py')
    from train_p1 import state_hash
    n=len(X);yscale=payload['data']['scalers']['y']['scale'];completed=[]
    for mc in MC_SEEDS:
        slots,alpha=common_paths(n,256,mc)
        path=output/f'common_MC_paths_{mc}.npz'
        if not path.exists():save_npz(path,{'background_slot':slots,'interpolation_alpha':alpha})
        else:
            with np.load(path,allow_pickle=False) as value:
                require(np.array_equal(value['background_slot'],slots) and np.array_equal(value['interpolation_alpha'],alpha),'Saved commonMC paths changed')
    for arm in ('mse','mae'):
        for seed in MODEL_SEEDS:
            model,initial_hash=build_frozen_model(states[(arm,seed)]['state'],torch)
            # Gate before any SHAP gradients for this checkpoint.
            from gradient_compatible_adapter import compatible_encoder_input_gradients
            with compatible_encoder_input_gradients(model,torch):raw=payload['compare'].inverse_y_float64(scaled_prediction(model,X,torch),payload['data']['scalers'])
            agreement=prediction_agreement(raw,saved[(arm,seed)],yscale,payload['protocol']['prediction_match_max_abs_standardized'])
            for bg in payload['plan']['backgrounds']:
                for mc in MC_SEEDS:
                    label=f'{arm}_seed{seed}__{bg["id"]}__mc{mc}'
                    npz=output/(label+'.npz');record=output/(label+'.json')
                    if record.exists():
                        prior=readj(record);require(prior['status']=='COMPLETE' and prior['npz_sha256']==sha(npz) and prior['input_lock_sha256']==sha(output/'analysis_input_lock.json'),'Completed SHAP job identity mismatch')
                        completed.append(prior);continue
                    require(not npz.exists(),'Uncommitted NPZ requires review before resume')
                    log_event(output/'progress.jsonl',{'status':'JOB_STARTED','job':label})
                    values,elapsed=one_attribution(model,X,Xtrain[np.asarray(bg['indices'])],mc,decision['batch_size'],torch,shap,payload['data']['scalers'],metrics)
                    require(state_hash(model)==initial_hash and all(p.grad is None and not p.requires_grad for p in model.parameters()),'SHAP changed model state or parameter gradients')
                    save_npz(npz,{**values,'explained_test_indices':np.asarray(payload['plan']['explained']['indices'],dtype=np.int64),
                        'background_train_indices':np.asarray(bg['indices'],dtype=np.int64)})
                    item={'status':'COMPLETE','arm':arm,'model_seed':seed,'background':bg['id'],'integration_seed':mc,
                        'gradient_adapter_id':'temporary_encoder_only_tanh_GELU_MHAoff_SDPAmath_v2',
                        'original_encoder_activation_objects_and_head_modules_restored':True,
                        'axis_order':['sample','lag','feature','output'],'shape':[n,10,8,8],'feature_order':payload['plan']['feature_order'],
                        'output_order':payload['plan']['output_order'],'lag_labels':LAGS,'nsamples':256,'batch_size':decision['batch_size'],
                        'seconds':elapsed,'prediction_agreement':agreement,'model_state_unchanged':True,'parameter_gradients_none':True,
                        'npz_sha256':sha(npz),'input_lock_sha256':sha(output/'analysis_input_lock.json'),
                        'library_variance_diagnostic_is_not_SE_or_CI':True,**diagnostic_record(values,payload,metrics)}
                    writej(record,item);completed.append(item)
                    log_event(output/'progress.jsonl',{'status':'JOB_COMPLETE','job':label,'seconds':elapsed});print(label,flush=True)
            del model
    ensemble_records=[]
    for arm in ('mse','mae'):
        for bg in payload['plan']['backgrounds']:
            for mc in MC_SEEDS:
                arrays=[]
                for seed in MODEL_SEEDS:
                    with np.load(output/f'{arm}_seed{seed}__{bg["id"]}__mc{mc}.npz',allow_pickle=False) as value:
                        arrays.append({key:value[key] for key in ('phi_scaled','phi_raw','expected_scaled','expected_raw','prediction_scaled','prediction_raw')})
                values={key:np.mean(np.stack([v[key] for v in arrays]),axis=0,dtype=np.float64) for key in arrays[0]}
                label=f'{arm}_ensemble__{bg["id"]}__mc{mc}';npz=output/(label+'.npz');record=output/(label+'.json')
                if not npz.exists():save_npz(npz,values)
                else:
                    with np.load(npz,allow_pickle=False) as value:
                        require(all(np.array_equal(value[key],values[key]) for key in values),'Prior ensemble differs from common-path mean')
                item={'status':'COMPLETE','arm':arm,'model_seed':'ensemble','background':bg['id'],'integration_seed':mc,
                    'ensemble_rule':'Mean signed phi across3seed models BEFORE absolute importance; same background/MCpaths',
                    'npz_sha256':sha(npz),'input_lock_sha256':sha(output/'analysis_input_lock.json'),**diagnostic_record(values,payload,metrics)}
                if record.exists():require(readj(record)==item,'Prior ensemble record changed')
                else:writej(record,item)
                ensemble_records.append(item)
    check_sources(payload)
    receipt={'status':'SUCCESS_SHAP_INPUT_GRADIENT_STABILITY_ARRAYS_SAVED','environment':environment,
        'gradient_protocol_version':2,'gradient_adapter_id':'temporary_encoder_only_tanh_GELU_MHAoff_SDPAmath_v2',
        'adapter_limit':'Finite forward agreement is not a global function or fused-kernel derivative identity proof.',
        'original_encoder_activation_objects_and_head_modules_restored':True,
        'model_jobs':len(completed),'ensemble_jobs':len(ensemble_records),'explained_n':n,'full_test_n':584,
        'sample_scope':'Fixed128-of584chronologicalcoverage;notall584','backgrounds':4,'background_size':64,
        'integration_seeds':list(MC_SEEDS),'nsamples':256,'batch_size':decision['batch_size'],
        'input_lock_sha256':sha(output/'analysis_input_lock.json'),'all_sources_unchanged':True,
        'all_model_states_unchanged':True,'parameter_gradients_computed':False,'input_gradients_computed':True,
        'training_optimizer_or_fit':False,'thesis_edit':False,'no_result_based_sample_or_nsamples_selection':True,
        'library_variance_diagnostic_caution':'SHAP0.52returns var(samples)/sqrt(nsamples), notvar(mean),notSEorCI.'}
    require(len(completed)==48 and len(ensemble_records)==16,'Incomplete factorial design')
    writej(output/'analysis_receipt.json',receipt)
    writej(output/'OUTPUT_MANIFEST.json',{'status':receipt['status'],'files':{p.name:{'sha256':sha(p),'size_bytes':p.stat().st_size} for p in output.iterdir() if p.is_file()}})
    print(json.dumps({'status':receipt['status'],'model_jobs':48,'ensemble_jobs':16,'output':str(output)}))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--execute-pilot',action='store_true')
    mode.add_argument('--execute-analysis',action='store_true')
    mode.add_argument('--internal-pilot-worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--pilot-name',default='pilot_mse123_first4_batch64_256_v2')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    payload=preflight()
    if args.internal_pilot_worker:
        require(os.environ.get('P1_SHAP_PILOT_WORKER')=='1','Pilot worker is internal only')
        target=(HERE/'pilot_runs'/args.pilot_name).resolve()
        require(target.parent==(HERE/'pilot_runs').resolve() and target.is_dir(),'Invalid pilot output')
        lock=readj(target/'pilot_input_lock.json')
        require(lock['source_sha256']==payload['sources'] and lock['protocol_sha256']==payload['protocol_sha256'] and lock['sample_plan_sha256']==payload['sample_plan_sha256'],'Parent/worker source lock equality failed before model load')
        require(lock['B128_forward_verification']==verify_v2_forward_receipt(payload),'Parent/worker B128 verification differs')
        pilot_worker(payload,target);return 0
    if args.execute_pilot:launch_pilot(payload,args);return 0
    if args.execute_analysis:main_analysis(payload,args);return 0
    require(not args.resume,'Resume requires explicit analysis execution')
    print(json.dumps({'status':'PREFLIGHT_ONLY_NO_TORCH_SHAP_OR_ARRAY_LOAD','explained_n':128,'backgrounds':4,'MC_repeats':2,
        'gradient_protocol_version':2,'gradient_adapter_source_sha256':sha(HERE/'gradient_compatible_adapter.py'),
        'model_jobs':48,'ensemble_jobs':16,'protocol_sha256':payload['protocol_sha256'],'sample_plan_sha256':payload['sample_plan_sha256'],
        'model_or_checkpoint_loaded':False,'input_gradients_computed':False,'training':False},indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
