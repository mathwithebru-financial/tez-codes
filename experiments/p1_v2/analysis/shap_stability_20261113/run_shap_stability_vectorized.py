"""Explicit V3 main using the same finite Expected Gradients estimator, vectorized.

Frozen V2 helper source is reused without modification. Default preflight only.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from pilot_vectorized_expected_gradients import prepare_v3,check_sources,estimator_values,load_module,HERE


def validated_decision(payload):
    base=payload['base'];path=HERE/'technical_batch_decision_v3.json';decision=base.readj(path)
    base.require(decision['status']=='APPROVED_VECTORIZED_BATCH_AFTER_EQUIVALENCE_PILOT' and decision['batch_size'] in (256,1024),'Optimized technical decision not ready')
    base.require(decision['protocol_sha256']==payload['protocol_sha256'] and decision['source_sha256']==payload['sources'],'Optimized technical decision source differs')
    result_path=Path(decision['pilot_result_path']).resolve()
    base.require((HERE/'pilot_runs').resolve() in result_path.parents and result_path.name=='pilot_result.json','Decision must reference a new optimized pilot')
    base.require(base.sha(result_path)==decision['pilot_result_sha256'],'Optimized pilot result changed')
    result=base.readj(result_path);lock_path=result_path.parent/'pilot_input_lock.json'
    base.require(result['status']=='PASS_VECTORIZED_EG_EQUIVALENCE_PILOT' and result['protocol_sha256']==payload['protocol_sha256'] and result['source_sha256']==payload['sources'],'Actual optimized pilot did not pass this source')
    base.require(result['input_lock_sha256']==base.sha(lock_path) and base.readj(lock_path)['source_sha256']==payload['sources'],'Optimized pilot parent/worker lock mismatch')
    base.require(base.readj(result_path.parent/'parent_result.json')['status']=='PASS_VECTORIZED_EG_EQUIVALENCE_PILOT','Optimized pilot parent did not pass')
    matching=[r for r in result['timings'] if r['batch_size']==decision['batch_size']]
    base.require(len(matching)==1 and matching[0]['eligible'] is True,'Chosen optimized batch failed equivalence or memory checks')
    review_path=Path(decision['independent_pilot_review_path']).resolve();review=base.readj(review_path)
    base.require(base.sha(review_path)==decision['independent_pilot_review_sha256'],'Independent optimized pilot review changed')
    base.require(review['status']=='PASS_INDEPENDENT_V3_VECTORIZED_PILOT' and review['pilot_result_sha256']==base.sha(result_path),'Independent optimized pilot review did not pass the actual result')
    return decision,{'technical_decision_sha256':base.sha(path),'pilot_result_path':str(result_path),'pilot_result_sha256':base.sha(result_path),
        'independent_review_path':str(review_path),'independent_review_sha256':base.sha(review_path)}


def execute(payload,resume):
    base=payload['base'];original=payload['base_payload'];decision,proof=validated_decision(payload)
    output=base.RUN/'shap_stability_20261113'
    lock={'method':'Vectorized finite Expected Gradients crosschecked against cachedSHAP0.52V2pilot',
        'protocol_version':3,'protocol_sha256':payload['protocol_sha256'],'sample_plan_sha256':payload['sample_plan_sha256'],
        'source_sha256':payload['sources'],'batch_size':decision['batch_size'],'optimization_validation':proof,
        'base_V2_B128_verification':payload['B128_forward_verification'],
        'environment_receipt_sha256':base.sha(HERE/'environment/environment_receipt.json'),
        'expected_package_versions':original['environment_receipt']['packages_after']}
    if output.exists():
        base.require(resume and base.readj(output/'analysis_input_lock.json')==lock,'Only explicit same-identity resume is allowed')
        base.require(not (output/'analysis_receipt.json').exists(),'Completed analysis must not be repeated')
    else:
        base.require(not resume,'Cannot resume missing analysis');output.mkdir();base.writej(output/'analysis_input_lock.json',lock)
    torch,shap,states,Xtrain,X,saved,environment=base.environment_and_data(original)
    metrics=load_module('p1_stability_v3_numeric_metrics',HERE/'stability_metrics.py')
    from train_p1 import state_hash
    from gradient_compatible_adapter import compatible_encoder_input_gradients
    n=len(X);yscale=original['data']['scalers']['y']['scale'];completed=[]
    for mc in base.MC_SEEDS:
        slots,alpha=base.common_paths(n,256,mc);target=output/f'common_MC_paths_{mc}.npz'
        if target.exists():
            with np.load(target,allow_pickle=False) as values:
                base.require(np.array_equal(values['background_slot'],slots) and np.array_equal(values['interpolation_alpha'],alpha),'Prior common MC draws changed')
        else:base.save_npz(target,{'background_slot':slots,'interpolation_alpha':alpha})
    for arm in ('mse','mae'):
        for seed in base.MODEL_SEEDS:
            model,initial=base.build_frozen_model(states[(arm,seed)]['state'],torch)
            with compatible_encoder_input_gradients(model,torch):
                raw=original['compare'].inverse_y_float64(base.scaled_prediction(model,X,torch),original['data']['scalers'])
            agreement=base.prediction_agreement(raw,saved[(arm,seed)],yscale,payload['protocol']['prediction_match_max_abs_standardized'])
            for bg in original['plan']['backgrounds']:
                for mc in base.MC_SEEDS:
                    label=f'{arm}_seed{seed}__{bg["id"]}__mc{mc}';npz=output/(label+'.npz');record=output/(label+'.json')
                    if record.exists():
                        prior=base.readj(record);base.require(prior['status']=='COMPLETE' and prior['npz_sha256']==base.sha(npz) and prior['input_lock_sha256']==base.sha(output/'analysis_input_lock.json'),'Completed optimized job identity differs')
                        completed.append(prior);continue
                    base.require(not npz.exists(),'Uncommitted NPZ requires review')
                    base.log_event(output/'progress.jsonl',{'status':'JOB_STARTED','job':label,'implementation':'vectorized_expected_gradients'})
                    values,elapsed,engine_metadata=estimator_values(model,X,Xtrain[np.asarray(bg['indices'])],mc,decision['batch_size'],torch,payload,metrics)
                    base.require(state_hash(model)==initial and all(p.grad is None and not p.requires_grad for p in model.parameters()),'Optimized SHAP changed model state/gradients')
                    base.save_npz(npz,{**values,'explained_test_indices':np.asarray(original['plan']['explained']['indices'],dtype=np.int64),
                        'background_train_indices':np.asarray(bg['indices'],dtype=np.int64)})
                    item={'status':'COMPLETE','arm':arm,'model_seed':seed,'background':bg['id'],'integration_seed':mc,
                        'implementation':'vectorized_expected_gradients_crosschecked_SHAP052','direct_GradientExplainer_library_call':False,
                        'axis_order':['sample','lag','feature','output'],'shape':[n,10,8,8],'feature_order':original['plan']['feature_order'],
                        'output_order':original['plan']['output_order'],'lag_labels':base.LAGS,'nsamples':256,'batch_size':decision['batch_size'],
                        'seconds':elapsed,'prediction_agreement':agreement,'model_state_unchanged':True,'parameter_gradients_none':True,
                        'original_encoder_and_head_objects_restored':True,'engine_metadata':engine_metadata,
                        'npz_sha256':base.sha(npz),'input_lock_sha256':base.sha(output/'analysis_input_lock.json'),
                        'library_variance_diagnostic_is_not_SE_or_CI':True,**base.diagnostic_record(values,original,metrics)}
                    base.writej(record,item);completed.append(item)
                    base.log_event(output/'progress.jsonl',{'status':'JOB_COMPLETE','job':label,'seconds':elapsed});print(label,flush=True)
            del model
    ensembles=[]
    for arm in ('mse','mae'):
        for bg in original['plan']['backgrounds']:
            for mc in base.MC_SEEDS:
                arrays=[]
                for seed in base.MODEL_SEEDS:
                    with np.load(output/f'{arm}_seed{seed}__{bg["id"]}__mc{mc}.npz',allow_pickle=False) as value:
                        arrays.append({key:value[key] for key in ('phi_scaled','phi_raw','expected_scaled','expected_raw','prediction_scaled','prediction_raw')})
                values={key:np.mean(np.stack([a[key] for a in arrays]),axis=0,dtype=np.float64) for key in arrays[0]}
                label=f'{arm}_ensemble__{bg["id"]}__mc{mc}';npz=output/(label+'.npz');record=output/(label+'.json')
                if npz.exists():
                    with np.load(npz,allow_pickle=False) as old:base.require(all(np.array_equal(old[key],value) for key,value in values.items()),'Prior optimized ensemble differs')
                else:base.save_npz(npz,values)
                item={'status':'COMPLETE','arm':arm,'model_seed':'ensemble','background':bg['id'],'integration_seed':mc,
                    'implementation':'mean_of_three_common_path_vectorized_expected_gradients',
                    'ensemble_rule':'Mean signed phi FIRST, then absolute importance; same background and MCpaths',
                    'npz_sha256':base.sha(npz),'input_lock_sha256':base.sha(output/'analysis_input_lock.json'),**base.diagnostic_record(values,original,metrics)}
                if record.exists():base.require(base.readj(record)==item,'Prior optimized ensemble record differs')
                else:base.writej(record,item)
                ensembles.append(item)
    check_sources(payload)
    base.require(len(completed)==48 and len(ensembles)==16,'Incomplete fixed factorial design')
    receipt={'status':'SUCCESS_SHAP_INPUT_GRADIENT_STABILITY_ARRAYS_SAVED','environment':environment,
        'gradient_protocol_version':3,'estimator_implementation':'vectorized_expected_gradients_crosschecked_SHAP052',
        'direct_GradientExplainer_library_call_in_main':False,'optimization_validation':proof,
        'model_jobs':48,'ensemble_jobs':16,'explained_n':128,'full_test_n':584,'backgrounds':4,'background_size':64,
        'integration_seeds':list(base.MC_SEEDS),'nsamples':256,'batch_size':decision['batch_size'],
        'input_lock_sha256':base.sha(output/'analysis_input_lock.json'),'all_sources_unchanged':True,
        'all_model_states_unchanged':True,'original_encoder_and_head_objects_restored':True,
        'parameter_gradients_computed':False,'input_gradients_computed':True,'training_optimizer_or_fit':False,'thesis_edit':False,
        'sample_scope':'Fixed128-of584chronologicalcoverage;notall584','no_result_based_sample_or_nsamples_selection':True,
        'library_variance_diagnostic_caution':'var(samples)/sqrt(nsamples), notSEorCI.',
        'adapter_limit':'Finite forward agreement is not global fused-function/gradient identity.'}
    base.writej(output/'analysis_receipt.json',receipt)
    base.writej(output/'OUTPUT_MANIFEST.json',{'status':receipt['status'],'files':{p.name:{'sha256':base.sha(p),'size_bytes':p.stat().st_size} for p in output.iterdir() if p.is_file()}})
    print(json.dumps({'status':receipt['status'],'model_jobs':48,'ensemble_jobs':16,'output':str(output)}))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-analysis',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv);payload=prepare_v3()
    if not args.execute_analysis:
        payload['base'].require(not args.resume,'Resume requires explicit analysis execution')
        print(json.dumps({'status':'V3_MAIN_PREFLIGHT_ONLY_NO_MODELS_ARRAYS_OR_GRADIENTS','protocol_sha256':payload['protocol_sha256'],
            'sample_plan_sha256':payload['sample_plan_sha256'],'base_V2_proof_checked_before_extension':True,'model_jobs':48,'ensemble_jobs':16},indent=2));return 0
    execute(payload,args.resume);return 0


if __name__=='__main__':raise SystemExit(main())
