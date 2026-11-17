"""V3 explicit computational pilot against cached successful SHAP0.52 V2 values.

Default metadata-only preflight. Frozen V2 source/functions are reused read-only;
the V3 source identity extension is separate and never monkeypatches the base.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
PROTOCOL=HERE/'shap_stability_protocol_v3.json'
V2_RUNNER_SHA='06be0d979c33968e55a04fd2ce73f9cefd69db2cafc44c58d8eaee9f62b4afe5'
V2_ADAPTER_SHA='70382426dc99f3125fe78acc2d9c5c605bbfcfe84439d7407f4c94dce9dec347'


def require(condition,message):
    if not condition:raise ValueError(message)


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def prepare_v3():
    import hashlib
    require(hashlib.sha256((HERE/'run_shap_stability.py').read_bytes()).hexdigest()==V2_RUNNER_SHA,'Frozen V2 runner changed')
    require(hashlib.sha256((HERE/'gradient_compatible_adapter.py').read_bytes()).hexdigest()==V2_ADAPTER_SHA,'Frozen V2 adapter changed')
    base=load_module('frozen_p1_SHAP_v2_helpers',HERE/'run_shap_stability.py')
    original=base.preflight()
    # CRITICAL: V2 proof is checked against its original identity BEFORE the
    # V3 source dictionary is extended. No original functions/data are changed.
    B128=base.verify_v2_forward_receipt(original)
    p=base.readj(PROTOCOL)
    base.require(p['version']==3 and p['nsamples']==256 and p['integration_seeds']==list(base.MC_SEEDS),'V3 estimator design changed')
    base.require(p['allowed_vectorized_batch_sizes']==[256,1024] and p['phi_agreement_max_abs_scaled']==1e-5,'V3 technical controls changed')
    base.require(p['base_v2_protocol_sha256']==original['protocol_sha256'] and p['sample_plan_sha256']==original['sample_plan_sha256'],'V3 reference protocol/plan differs')
    base.require(p['expected_engine_file']=='vectorized_expected_gradients.py','Unexpected EG engine')
    reference=HERE/p['reference_pilot']['file'];reference_phi=HERE/p['reference_pilot']['batch256_phi_file']
    base.require(base.sha(reference)==p['reference_pilot']['sha256'] and base.sha(reference_phi)==p['reference_pilot']['batch256_phi_sha256'],'Cached successful V2 pilot identity differs')
    v2pilot=base.readj(reference);v2lock=base.readj(reference.parent/'pilot_input_lock.json')
    base.require(v2pilot['status']=='PASS_BOUNDED_INPUT_GRADIENT_PILOT_ONLY' and v2pilot['batch256_eligible'] is True,'Reference V2 pilot did not pass')
    base.require(v2pilot['source_sha256']==v2lock['source_sha256']==original['sources'],'V2 source/proof identity differs')
    base.require(v2pilot['input_lock_sha256']==base.sha(reference.parent/'pilot_input_lock.json'),'V2 pilot parent/worker identity differs')
    reviewed=HERE/'independent_pilot_review_v2.json';review=base.readj(reviewed)
    base.require(review['status']=='PASS_INDEPENDENT_V2_PILOT_RECORDS_AND_SAVED_ARRAYS' and review['pilot_result_sha256']==base.sha(reference),'Successful cached V2 pilot independent review missing')
    sources=dict(original['sources'])
    for path in (PROTOCOL,HERE/'method_addendum_v3.json',HERE/'vectorized_expected_gradients.py',Path(__file__),
                 HERE/'run_shap_stability_vectorized.py',reference,reference_phi,reviewed,
                 reference.parent/'pilot_input_lock.json',reference.parent/'parent_result.json',Path(B128['path'])):
        sources[str(path.resolve())]=base.sha(path)
    for rel,digest in p['source_hashes'].items():
        base.require(base.sha(BASE/rel)==digest,'V3 protocol source differs: '+rel);sources[str((BASE/rel).resolve())]=digest
    addendum=base.readj(HERE/'method_addendum_v3.json')
    base.require(addendum['new_protocol_sha256']==base.sha(PROTOCOL),'V3 addendum/protocol identity differs')
    return {'base':base,'base_payload':original,'protocol':p,'protocol_sha256':base.sha(PROTOCOL),
        'sources':sources,'reference_phi':reference_phi,'B128_forward_verification':B128,
        'sample_plan_sha256':original['sample_plan_sha256']}


def check_sources(payload):
    for path,digest in payload['sources'].items():payload['base'].require(payload['base'].sha(path)==digest,'V3 source changed: '+path)


def estimator_values(model,X,background,mcseed,batch_size,torch,payload,metrics):
    base=payload['base'];original=payload['base_payload']
    engine=load_module('p1_vectorized_finite_expected_gradients',HERE/'vectorized_expected_gradients.py')
    from gradient_compatible_adapter import compatible_encoder_input_gradients
    torch.manual_seed(mcseed);torch.cuda.manual_seed_all(mcseed)
    torch.cuda.synchronize();start=time.perf_counter()
    with compatible_encoder_input_gradients(model,torch):
        prediction=base.scaled_prediction(model,X,torch)
        expected=np.mean(base.scaled_prediction(model,background,torch),axis=0,dtype=np.float64)
        value=engine.vectorized_expected_gradients(model,X,background,rseed=mcseed,nsamples=256,batch_size=batch_size,device='cuda')
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    phi=base.canonical_phi(value['phi_scaled'],len(X))
    diagnostic=base.canonical_phi(value['library_variance_diagnostic_scaled'],len(X))
    base.require(np.all(diagnostic>=0),'Negative finite-MC diagnostic')
    y=original['data']['scalers']['y'];scale=np.asarray(y['scale'],dtype=np.float64)
    rawphi,rawexpected,rawprediction=metrics.affine_output_transform(phi,expected,prediction,np.asarray(y['mean']),scale)
    values={'phi_scaled':phi,'phi_raw':rawphi,'expected_scaled':expected,'expected_raw':rawexpected,
        'prediction_scaled':prediction,'prediction_raw':rawprediction,
        'library_variance_diagnostic_scaled':diagnostic,'library_variance_diagnostic_raw':diagnostic*scale.reshape(1,1,1,8)**2}
    metadata={k:v for k,v in value.items() if k not in ('phi_scaled','library_variance_diagnostic_scaled')}
    return values,elapsed,metadata


def worker(payload,output):
    base=payload['base'];original=payload['base_payload']
    torch,shap,states,Xtrain,X,saved,environment=base.environment_and_data(original)
    metrics=load_module('p1_stability_numeric_metrics',HERE/'stability_metrics.py')
    model,initial=base.build_frozen_model(states[('mse',123)]['state'],torch)
    from train_p1 import state_hash
    from gradient_compatible_adapter import compatible_encoder_input_gradients
    X=X[:4];background=Xtrain[np.asarray(original['plan']['backgrounds'][0]['indices'])]
    with np.load(payload['reference_phi'],allow_pickle=False) as cached:
        reference={name:cached[name] for name in ('phi_scaled','library_variance_diagnostic_scaled','prediction_raw','expected_raw')}
    base.require(reference['phi_scaled'].shape==(4,10,8,8),'Cached reference is not the fixed four-example pilot')
    with compatible_encoder_input_gradients(model,torch):
        raw=original['compare'].inverse_y_float64(base.scaled_prediction(model,X,torch),original['data']['scalers'])
    agreement=base.prediction_agreement(raw,saved[('mse',123)][:4],original['data']['scalers']['y']['scale'],payload['protocol']['prediction_match_max_abs_standardized'])
    results=[]
    for batch in (256,1024):
        torch.cuda.reset_peak_memory_stats()
        try:values,elapsed,engine_metadata=estimator_values(model,X,background,base.MC_SEEDS[0],batch,torch,payload,metrics)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache();results.append({'batch_size':batch,'status':'OUT_OF_MEMORY','eligible':False,'error':str(exc)});continue
        base.require(state_hash(model)==initial and all(not p.requires_grad and p.grad is None for p in model.parameters()),'Optimized estimator changed model state/gradients')
        difference=float(np.max(np.abs(values['phi_scaled']-reference['phi_scaled'])))
        diagnostic_difference=float(np.max(np.abs(values['library_variance_diagnostic_scaled']-reference['library_variance_diagnostic_scaled'])))
        peak=int(torch.cuda.max_memory_allocated());total=int(torch.cuda.get_device_properties(0).total_memory)
        eligible=difference<=payload['protocol']['phi_agreement_max_abs_scaled'] and peak<.95*total
        base.save_npz(output/f'vectorized_batch{batch}.npz',values)
        item={'batch_size':batch,'status':'CHECKS_PASS' if eligible else 'CHECKS_FAIL_RETAINED','eligible':eligible,
            'seconds':elapsed,'phi_max_abs_scaled_difference_from_cached_SHAP052':difference,
            'library_variance_diagnostic_max_abs_difference':diagnostic_difference,
            'original_phi_tolerance':payload['protocol']['phi_agreement_max_abs_scaled'],
            'peak_allocated_gpu_bytes':peak,'total_gpu_bytes':total,'prediction_agreement':agreement,
            'rough_48job_N128_seconds':elapsed*48*128/4,'linear_projection_is_not_runtime_guarantee':True,
            'engine_metadata':engine_metadata}
        results.append(item);base.log_event(output/'progress.jsonl',{'status':'OPTIMIZED_PILOT_BATCH_FINISHED',**item})
    check_sources(payload)
    result={'status':'PASS_VECTORIZED_EG_EQUIVALENCE_PILOT' if any(r['eligible'] for r in results) else 'FAIL_VECTORIZED_EG_EQUIVALENCE_PILOT_RETAINED',
        'environment':environment,'timings':results,'sample_count':4,'nsamples':256,'output_count':8,
        'model':'mse_seed123','background':'linspace_reference','integration_seed':base.MC_SEEDS[0],
        'protocol_sha256':payload['protocol_sha256'],'sample_plan_sha256':payload['sample_plan_sha256'],
        'input_lock_sha256':base.sha(output/'pilot_input_lock.json'),'source_sha256':payload['sources'],
        'reference_SHAP052_NPZ_sha256':base.sha(payload['reference_phi']),
        'all_model_states_and_original_activation_objects_unchanged':True,'parameter_gradients_none':True,
        'input_gradients_computed':True,'optimizer_training_or_thesis_edit':False,'main_analysis_started':False,
        'main_estimator_is_optimized_finite_expected_gradients_not_direct_library_call':True}
    base.writej(output/'pilot_result.json',result)


def launch(payload,name):
    base=payload['base'];target=(HERE/'pilot_runs'/name).resolve()
    base.require(target.parent==(HERE/'pilot_runs').resolve() and name not in ('','.','..'),'Invalid optimized pilot directory')
    base.require(not target.exists(),'Preserve earlier pilot; require a fresh directory')
    target.mkdir(parents=True)
    base.writej(target/'pilot_input_lock.json',{'protocol_sha256':payload['protocol_sha256'],'sample_plan_sha256':payload['sample_plan_sha256'],
        'source_sha256':payload['sources'],'B128_forward_verification':payload['B128_forward_verification'],'timeout_seconds':180})
    started=time.perf_counter();environment=dict(os.environ);environment['P1_VECTOR_EG_PILOT_WORKER']='1'
    command=[sys.executable,str(Path(__file__).resolve()),'--internal-worker','--pilot-name',name]
    with (target/'worker.log').open('w',encoding='utf-8') as log:
        try:
            completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=environment,timeout=180,check=False)
            base.require(completed.returncode==0,'Optimized pilot worker failed; see retained log')
            result=base.readj(target/'pilot_result.json')
            base.require(result['status']=='PASS_VECTORIZED_EG_EQUIVALENCE_PILOT','No optimized pilot batch passed')
            base.require(result['input_lock_sha256']==base.sha(target/'pilot_input_lock.json') and result['source_sha256']==payload['sources'],'Optimized parent/worker identity differs')
            check_sources(payload)
            record={'status':'PASS_VECTORIZED_EG_EQUIVALENCE_PILOT','seconds':time.perf_counter()-started,'timed_out':False}
        except subprocess.TimeoutExpired:
            record={'status':'VECTORIZED_PILOT_TIMEOUT_RETAINED','seconds':time.perf_counter()-started,'timed_out':True}
        except Exception as exc:
            base.writej(target/'parent_result.json',{'status':'VECTORIZED_PILOT_FAILED_RETAINED','error':repr(exc),'seconds':time.perf_counter()-started});raise
    base.writej(target/'parent_result.json',record)
    base.require(record['status']=='PASS_VECTORIZED_EG_EQUIVALENCE_PILOT','Optimized pilot timed out')
    print(json.dumps({'status':record['status'],'output':str(target)}))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--execute-pilot',action='store_true')
    group.add_argument('--internal-worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--pilot-name',default='pilot_vectorized_EG_v3_first4')
    args=parser.parse_args(argv);payload=prepare_v3();base=payload['base']
    if args.internal_worker:
        base.require(os.environ.get('P1_VECTOR_EG_PILOT_WORKER')=='1','Internal worker only')
        target=(HERE/'pilot_runs'/args.pilot_name).resolve()
        base.require(target.parent==(HERE/'pilot_runs').resolve() and target.is_dir(),'Invalid worker output')
        lock=base.readj(target/'pilot_input_lock.json')
        base.require(lock['source_sha256']==payload['sources'] and lock['protocol_sha256']==payload['protocol_sha256'] and lock['sample_plan_sha256']==payload['sample_plan_sha256'],'Parent/worker source lock differs before model load')
        worker(payload,target);return 0
    if args.execute_pilot:launch(payload,args.pilot_name);return 0
    print(json.dumps({'status':'V3_VECTOR_EG_PREFLIGHT_ONLY_NO_ARRAYS_MODELS_OR_GRADIENTS','protocol_sha256':payload['protocol_sha256'],
        'sample_plan_sha256':payload['sample_plan_sha256'],'model_jobs':48,'explained_n':128,'nsamples':256,
        'base_V2_proof_checked_before_identity_extension':True,'Torch_SHAP_or_NPY_loaded':False},indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
