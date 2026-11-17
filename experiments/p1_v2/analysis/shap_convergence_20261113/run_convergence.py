"""Post-analysis targeted numerical SHAP diagnostic; never train or edit a thesis.

Default validates the locked metadata only. --prepare locks the diagnostic
selection before gradients. --execute requires that exact new plan and sources.
"""
from pathlib import Path
import argparse, csv, datetime, hashlib, importlib.util, json, sys, time
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
OLD=HERE.parent/'shap_stability_20261113'
RUN=BASE/'training/runs/p1_v2_mse_mae_20261113_full'
PRIOR=RUN/'shap_stability_20261113'
OUT=RUN/'shap_convergence_20261113'
PLAN=HERE/'locked_convergence_plan.json'
STEPS=(256,1024,4096)
REPEATS=(('original_path_continuation',None),('fresh_1',2026111341),('fresh_2',2026111342))


def require(ok,msg):
    if not ok: raise ValueError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);require(not p.exists(),'Refuse overwrite: '+str(p))
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def mod(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def api():
    if str(OLD) not in sys.path: sys.path.insert(0,str(OLD))
    v3=mod('unchanged_shap_v3_followup_api',OLD/'pilot_vectorized_expected_gradients.py')
    return v3.prepare_v3()
def manifest_files():
    prior=read(BASE/'WORKSPACE_MANIFEST.json')
    protected={str((BASE/r['path']).resolve()):r['sha256'] for r in prior['files']}
    protected[str((BASE/'WORKSPACE_MANIFEST.json').resolve())]=sha(BASE/'WORKSPACE_MANIFEST.json')
    for r in read(OLD/'protected_sources_before.json')['files']:
        p=str(Path(r['path']).resolve())
        require(p not in protected or protected[p]==r['sha256'],'Protection identity conflict')
        protected[p]=r['sha256']
    return protected
def check(protected):
    for p,h in protected.items(): require(sha(p)==h,'Existing source/result changed: '+p)


def prepare():
    require(not PLAN.exists() and not OUT.exists(),'Keep earlier plan and results unchanged')
    p=api();base=p['base'];original=p['base_payload']
    qa=read(OLD/'independent_final_shap_review_v3.json')
    require(qa['status']=='PASS_INDEPENDENT_V3_SHAP_ARRAYS_AND_STABILITY_TABLES','Prior independent QA missing')
    require(qa['analysis_receipt_sha256']==sha(PRIOR/'analysis_receipt.json'),'Prior QA identity mismatch')
    rows=list(csv.DictReader((PRIOR/'stability/local_additivity_diagnostics.csv').open(encoding='utf-8-sig')))
    dates=list(csv.DictReader((OLD/'explained_test_indices_dates.csv').open(encoding='utf-8-sig')))
    conditions=[]
    for arm in ('mse','mae'):
        r=max((r for r in rows if r['arm']==arm and r['model_scope']=='individual_seed'),
              key=lambda r:float(r['scaled_residual_max_abs']))
        label=r['condition'];j=int(r['output_index'])
        with np.load(PRIOR/(label+'.npz'),allow_pickle=False) as z:
            residual=z['prediction_scaled']-z['expected_scaled'][None,:]-z['phi_scaled'].sum(axis=(1,2))
        worst=int(np.argmax(np.abs(residual[:,j])))
        require(abs(float(abs(residual[worst,j]))-float(r['scaled_residual_max_abs']))<1e-12,'Selected residual differs from prior table')
        positions=list(dict.fromkeys([worst,0,64,127]))
        conditions.append({'arm':arm,'model_seed':int(r['model_seed']),'condition':label,
            'background':r['background'],'original_integration_seed':int(r['integration_seed']),
            'selected_worst_output_index':j,'selected_worst_output':r['output'],
            'selected_worst_explanation_position':worst,'prior_max_abs_scaled_residual':float(r['scaled_residual_max_abs']),
            'positions':positions,'dates':[dates[i] for i in positions],
            'prior_npz_sha256':sha(PRIOR/(label+'.npz'))})
    protected=manifest_files();check(protected)
    sources={str(p.resolve()):sha(p) for p in (Path(__file__),HERE/'explicit_path_gradients.py')}
    plan={'status':'LOCKED_POST_ANALYSIS_TARGETED_NUMERICAL_DIAGNOSTIC_BEFORE_GRADIENTS',
        'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'selection':'One maximum-residual individual condition per training arm; worst date plus fixed positions0/64/127; outcome-selected diagnostic, not representative inference.',
        'conditions':conditions,'nsamples':list(STEPS),'integration_repeats':[
            {'id':k,'seed':s,'skip_rule':'original_position*256 choice-then-uniform draws' if s is None else 'zero; shared path across cases'} for k,s in REPEATS],
        'path_rule':'Per case one4096draw stream; each256/1024/4096 prefix is nested. Original stream skips position*256 to replay old256exactly, then continues.',
        'all_outputs_computed':8,'batch_size':1024,'background_n':64,
        'technical_replay_phi_max_abs_scaled_tolerance':1e-5,
        'no_monotone_residual_or_statistical_pass_threshold':True,
        'decomposition':'R_total=f(x)-mean64f(B)-sumphi; R_bg=mean_sampled_f(B)-mean64f(B); R_path=f(x)-mean_sampled_f(B)-sumphi; R_total=R_bg+R_path.',
        'no_training_thesis_or_original_edits':True,'source_sha256':sources,
        'prior_v3_protocol_sha256':p['protocol_sha256'],'prior_sample_plan_sha256':p['sample_plan_sha256'],
        'prior_qa_sha256':sha(OLD/'independent_final_shap_review_v3.json'),
        'prior_workspace_manifest_sha256':sha(BASE/'WORKSPACE_MANIFEST.json'),
        'protected_files':protected}
    write(PLAN,plan)
    print(json.dumps({'status':plan['status'],'cases':sum(len(c['positions']) for c in conditions),
        'conditions':[{k:c[k] for k in ('condition','selected_worst_output','selected_worst_explanation_position','prior_max_abs_scaled_residual')} for c in conditions],
        'protected_file_count':len(protected)}))


def preflight():
    plan=read(PLAN)
    require(plan['status']=='LOCKED_POST_ANALYSIS_TARGETED_NUMERICAL_DIAGNOSTIC_BEFORE_GRADIENTS','Wrong plan')
    require(plan['nsamples']==list(STEPS) and plan['batch_size']==1024 and plan['all_outputs_computed']==8,'Plan changed')
    for p,h in plan['source_sha256'].items(): require(sha(p)==h,'Locked new source changed: '+p)
    check(plan['protected_files']);p=api()
    require(p['protocol_sha256']==plan['prior_v3_protocol_sha256'] and p['sample_plan_sha256']==plan['prior_sample_plan_sha256'],'Prior method changed')
    return plan,p


def execute():
    plan,p=preflight();require(not OUT.exists(),'Refuse overwrite/restart completed or incomplete diagnostic')
    OUT.mkdir();write(OUT/'input_lock.json',{'plan_sha256':sha(PLAN),'source_sha256':plan['source_sha256'],
        'prior_v3_sources':p['sources']})
    original=p['base_payload'];base=p['base'];engine=mod('new_explicit_convergence_paths',HERE/'explicit_path_gradients.py')
    torch,shap,states,Xtrain,X,saved,environment=base.environment_and_data(original)
    from train_p1 import state_hash
    from gradient_compatible_adapter import compatible_encoder_input_gradients
    records=[];technical=[];model_checks=[];started=time.perf_counter()
    for c in plan['conditions']:
        arm,seed=c['arm'],c['model_seed'];model,initial=base.build_frozen_model(states[(arm,seed)]['state'],torch)
        bgplan=next(b for b in original['plan']['backgrounds'] if b['id']==c['background'])
        bg=Xtrain[np.asarray(bgplan['indices'],dtype=np.int64)]
        with np.load(PRIOR/(c['condition']+'.npz'),allow_pickle=False) as z:
            old={k:z[k].copy() for k in ('phi_scaled','prediction_scaled','expected_scaled')}
        with np.load(PRIOR/f'common_MC_paths_{c["original_integration_seed"]}.npz',allow_pickle=False) as z:
            oldslots=z['background_slot'].copy();oldalpha=z['interpolation_alpha'].copy()
        with compatible_encoder_input_gradients(model,torch):
            pred=base.scaled_prediction(model,X,torch);bgpred=base.scaled_prediction(model,bg,torch)
            expected=bgpred.mean(axis=0,dtype=np.float64)
            raw=original['compare'].inverse_y_float64(pred,original['data']['scalers'])
            agreement=base.prediction_agreement(raw,saved[(arm,seed)],original['data']['scalers']['y']['scale'],1e-5)
            require(np.max(np.abs(pred-old['prediction_scaled']))<=1e-5 and np.array_equal(expected,old['expected_scaled']),'Prior endpoint predictions changed')
            for i in c['positions']:
                for rep,freshseed in REPEATS:
                    rseed=c['original_integration_seed'] if freshseed is None else freshseed
                    skip=i*256 if freshseed is None else 0
                    slots,alpha=engine.path_stream(4096,rseed,skip)
                    if freshseed is None:
                        require(np.array_equal(slots[:256],oldslots[i]) and np.array_equal(alpha[:256],oldalpha[i]),'Original256 paths did not replay exactly')
                    contributions=engine.explicit_contributions(model,X[i],bg,slots,alpha,torch)
                    if freshseed is None:
                        diff=float(np.max(np.abs(contributions[:256].mean(axis=0)-old['phi_scaled'][i])))
                        require(diff<=1e-5,'Original256 attribution failed technical replay')
                        technical.append({'condition':c['condition'],'position':i,'phi256_replay_max_abs_difference_scaled':diff})
                    filename=f'{arm}_seed{seed}__position{i}__{rep}.npz'
                    base.save_npz(OUT/filename,{'contributions_scaled':contributions,'slots':slots,'alpha':alpha,
                        'prediction_scaled':pred[i],'background_prediction_scaled':bgpred})
                    for n in STEPS:
                        phi=contributions[:n].mean(axis=0,dtype=np.float64)
                        psum=phi.sum(axis=(0,1));sample_dbg=bgpred[slots[:n]].mean(axis=0,dtype=np.float64)
                        total=pred[i]-expected-psum;bgterm=sample_dbg-expected;pathterm=pred[i]-sample_dbg-psum
                        require(np.allclose(total,bgterm+pathterm,rtol=0,atol=1e-12),'Residual decomposition failed')
                        for j,name in enumerate(original['plan']['output_order']):
                            records.append({'arm':arm,'model_seed':seed,'condition':c['condition'],'background':c['background'],
                                'position':i,'target_date':next(x['target_date_label'] for x in c['dates'] if int(x['explanation_position'])==i),
                                'is_condition_worst_date':i==c['selected_worst_explanation_position'],
                                'is_selected_worst_output':j==c['selected_worst_output_index'],'repeat':rep,'integration_seed':rseed,'skip_draw_pairs':skip,
                                'nsamples':n,'output_index':j,'output':name,'prediction_scaled':float(pred[i,j]),'expected_scaled':float(expected[j]),
                                'sum_phi_scaled':float(psum[j]),'residual_scaled':float(total[j]),'abs_residual_scaled':float(abs(total[j])),
                                'background_sampling_residual_scaled':float(bgterm[j]),'path_sampling_residual_scaled':float(pathterm[j]),
                                'residual_raw':float(total[j]*original['data']['scalers']['y']['scale'][j]),
                                'npz_file':filename})
                    print(json.dumps({'status':'CASE_REPEAT_COMPLETE','condition':c['condition'],'position':i,'repeat':rep}),flush=True)
        require(state_hash(model)==initial and all(not q.requires_grad and q.grad is None for q in model.parameters()),'Model weights or parameter gradients changed')
        model_checks.append({'condition':c['condition'],'state_sha256_before':initial,
            'state_sha256_after':state_hash(model),'saved_prediction_agreement':agreement,
            'adapter_context_original_objects_restoration_checks_passed':True,
            'all_parameter_gradients_disabled_and_none':True})
        del model
    with (OUT/'convergence_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    check(plan['protected_files'])
    for path,digest in plan['source_sha256'].items():require(sha(path)==digest,'New method changed during execution')
    p['base'].check_sources(p['base_payload'])
    receipt={'status':'SUCCESS_FIXED_MODEL_TARGETED_SHAP_CONVERGENCE','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'plan_sha256':sha(PLAN),'input_lock_sha256':sha(OUT/'input_lock.json'),'environment':environment,
        'conditions':len(plan['conditions']),'selected_cases':sum(len(c['positions']) for c in plan['conditions']),
        'stream_repeats':3,'draw_counts':list(STEPS),'output_count':8,'metric_rows':len(records),
        'technical_replays':technical,'model_checks':model_checks,'all_existing_protected_files_unchanged':True,'protected_file_count':len(plan['protected_files']),
        'all_model_states_unchanged':True,'original_activation_objects_restored':True,'input_gradients_computed':True,
        'parameter_gradients_computed':False,'optimizer_steps':0,'new_training':False,'thesis_edit':False,
        'seconds':time.perf_counter()-started,'no_residual_filter_correction_or_pass_threshold':True,
        'limits':['Outcome-selected stress diagnostic plus fixed chronological controls, not a representative new test or model selection.',
            'Nested draws and shared streams are dependent; no p values or monotone convergence claim.',
            'Finite forward agreement of the prior numerical adapter is not universal function/gradient equality.',
            'No old SHAP result, thesis, forecast performance or historical availability conclusion was changed.']}
    write(OUT/'convergence_receipt.json',receipt)
    write(OUT/'OUTPUT_MANIFEST.json',{'status':receipt['status'],'files':{f.name:{'sha256':sha(f),'size_bytes':f.stat().st_size} for f in OUT.iterdir() if f.is_file()}})
    print(json.dumps({'status':receipt['status'],'rows':len(records),'seconds':receipt['seconds']}))


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group()
    g.add_argument('--prepare',action='store_true');g.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.prepare:prepare()
    elif a.execute:execute()
    else:
        plan,_=preflight()
        print(json.dumps({'status':'READY_NO_MODELS_OR_GRADIENTS_EXECUTED','cases':sum(len(c['positions']) for c in plan['conditions'])}))
if __name__=='__main__':main()
