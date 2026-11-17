"""Default PLAN/PREFLIGHT ONLY. Training requires explicit --execute-train.

This prepared CLI has not been authorized to train in the preparation turn.
No test arrays or test metrics are read by this program, even during training.
"""
from pathlib import Path
import argparse,copy,csv,hashlib,json,math,platform,random,sys,datetime,importlib.metadata
import numpy as np
try:
    from .config import ArmConfig,SEEDS,OUTPUT_ORDER,fixed_arms,should_stop,is_improvement
    from .data_contract import DEFAULT_DATASET,DEFAULT_DENOMINATORS,load_fit_data,inverse_y_float64,sha
except ImportError:
    from config import ArmConfig,SEEDS,OUTPUT_ORDER,fixed_arms,should_stop,is_improvement
    from data_contract import DEFAULT_DATASET,DEFAULT_DENOMINATORS,load_fit_data,inverse_y_float64,sha
from dataclasses import asdict

HERE=Path(__file__).resolve().parent
def utcnow():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def set_seed(seed,torch):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends,'cudnn'):
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def state_hash(model):
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        value=tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode('utf-8'));digest.update(str(value.dtype).encode('ascii'))
        digest.update(str(value.shape).encode('ascii'));digest.update(value.tobytes())
    return digest.hexdigest()

def make_loaders(data,seed,torch):
    from torch.utils.data import DataLoader,TensorDataset
    train=TensorDataset(torch.tensor(data['X_train'],dtype=torch.float32),torch.tensor(data['y_train_scaled'],dtype=torch.float32))
    val=TensorDataset(torch.tensor(data['X_val'],dtype=torch.float32),torch.tensor(data['y_val_scaled'],dtype=torch.float32))
    generator=torch.Generator();generator.manual_seed(int(seed))
    return (DataLoader(train,batch_size=64,shuffle=True,drop_last=False,generator=generator),
            DataLoader(val,batch_size=64,shuffle=False,drop_last=False))

def write_history(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def execute_one_run(config,seed,payload,device_name,run_dir):
    # Called ONLY after main's explicit execution gate; never called in preflight.
    if not isinstance(config,ArmConfig) or asdict(config) not in [asdict(c) for c in fixed_arms()]:
        raise ValueError('Execution accepts only an exact fixed arm; custom controls are not silently ignored')
    if seed not in SEEDS:raise ValueError('Execution seed must be one of123/777/2026')
    import torch
    try:
        from .models import build_selected_model,parameter_counts
        from .losses import task_losses
    except ImportError:
        from models import build_selected_model,parameter_counts
        from losses import task_losses
    if device_name=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback. CPU requires explicit --device cpu.')
    device=torch.device(device_name)
    set_seed(seed,torch)
    arrays=payload['arrays']
    train_loader,val_loader=make_loaders(arrays,seed,torch)
    model=build_selected_model().to(device)
    counts=parameter_counts(model);initial_hash=state_hash(model)
    optimizer_params=list(model.parameters())
    optimizer=torch.optim.AdamW(optimizer_params,lr=1e-3,weight_decay=1e-4)
    best_score=math.inf;best_epoch=0;best_state=None;best_score_object=None;no_improve=0;history=[]
    run_dir.mkdir(parents=True,exist_ok=False)
    run_meta={'status':'TRAINING_STARTED_AFTER_EXPLICIT_FLAG','started_at_utc':utcnow(),'arm':asdict(config),'seed':seed,
      'initial_model_sha256':initial_hash,'parameter_counts':counts,'dataset_identity':payload['identity'],
      'runtime':{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,
                 'device':str(device),'cuda_build':torch.version.cuda,'gpu_name':torch.cuda.get_device_name(0) if device.type=='cuda' else None},
      'optimizer_defaults':optimizer.defaults,'scheduler':None,'test_arrays_loaded':False,'test_scores_computed':False}
    (run_dir/'run_manifest.json').write_text(json.dumps(run_meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    try:
        for epoch in range(1,101):
            model.train();sums=np.zeros(3,dtype=np.float64);n_obs=0
            for X,y in train_loader:
                X=X.to(device);y=y.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred=model(X)
                total,ret,vol=task_losses(pred,y,config.return_loss)
                if not torch.isfinite(total):raise RuntimeError('Nonfinite training loss')
                total.backward()
                torch.nn.utils.clip_grad_norm_(optimizer_params,1.0)
                optimizer.step()
                n=len(X);sums+=np.array([total.item(),ret.item(),vol.item()])*n;n_obs+=n
            model.eval();scaled_predictions=[];val_sums=np.zeros(2,dtype=np.float64);val_n=0
            with torch.no_grad():
                for X,y in val_loader:
                    X=X.to(device);y=y.to(device);pred=model(X)
                    _,ret,vol=task_losses(pred,y,config.return_loss)
                    scaled_predictions.append(pred.detach().cpu().numpy())
                    n=len(X);val_sums+=np.array([ret.item(),vol.item()])*n;val_n+=n
            scaled=np.vstack(scaled_predictions)
            raw=inverse_y_float64(scaled,payload['scalers'])
            if sha(payload['identity']['denominator_path'])!=payload['identity']['denominator_sha256']:
                raise RuntimeError('Locked validation denominators changed during the run')
            score_obj=payload['scorer'](arrays['y_val_raw'],raw,arrays['target_dates_val'].tolist(),
                 output_order=list(OUTPUT_ORDER),denominator_path=payload['identity']['denominator_path'])
            score=float(score_obj['validation_score'])
            if not math.isfinite(score):raise RuntimeError('Nonfinite validation score')
            if is_improvement(score,best_score):
                best_score=score;best_epoch=epoch;no_improve=0
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                best_score_object=copy.deepcopy(score_obj)
            else:no_improve+=1
            history.append({'epoch':epoch,'train_total_loss_scaled':float(sums[0]/n_obs),
              'train_return_loss_scaled':float(sums[1]/n_obs),'train_vol_pinball_scaled':float(sums[2]/n_obs),
              'val_return_loss_scaled':float(val_sums[0]/val_n),'val_vol_pinball_scaled':float(val_sums[1]/val_n),
              'val_equal_weight_proxy_loss':float(.5*np.sum(val_sums/val_n)),
              'validation_score':score,'mean_return_ratio':score_obj['mean_return_ratio'],
              'mean_volatility_ratio':score_obj['mean_volatility_ratio'],'best_epoch_so_far':best_epoch,'no_improve':no_improve})
            if should_stop(epoch,no_improve):break
        if best_state is None:raise RuntimeError('No finite validation-selected checkpoint')
        # No torch.load, test arrays or test evaluation. Only newly trained
        # state is written after this run's validation selection is finished.
        checkpoint=run_dir/'best_checkpoint.pt'
        torch.save({'model_state_dict':best_state,'arm':asdict(config),'seed':seed,'best_epoch':best_epoch,
          'validation_score':best_score,'validation_score_object':best_score_object,'dataset_identity':payload['identity'],
          'initial_model_sha256':initial_hash,'test_arrays_loaded':False,'test_scores_computed':False},checkpoint)
        write_history(run_dir/'history.csv',history)
        run_meta.update({'status':'SUCCESS_VALIDATION_CHECKPOINT_FIXED_NO_TEST_ACCESS','completed_at_utc':utcnow(),
          'best_epoch':best_epoch,'epochs_ran':len(history),'best_validation_score':best_score,
          'best_checkpoint_path':str(checkpoint),'best_checkpoint_sha256':sha(checkpoint)})
        (run_dir/'run_manifest.json').write_text(json.dumps(run_meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        return run_meta
    except Exception as exc:
        run_meta.update({'status':'FAILED_NO_TEST_ACCESS','error':repr(exc),'completed_at_utc':utcnow()})
        (run_dir/'run_manifest.json').write_text(json.dumps(run_meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        raise

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir',type=Path,default=DEFAULT_DATASET)
    parser.add_argument('--denominators',type=Path,default=DEFAULT_DENOMINATORS)
    parser.add_argument('--plan-output',type=Path,default=HERE/'preflight_training_manifest.json')
    parser.add_argument('--output-dir',type=Path,default=HERE/'runs/p1_v2_matched_loss')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--arm',choices=['both','mse','mae'],default='both')
    parser.add_argument('--seed',type=int,choices=list(SEEDS),default=None)
    parser.add_argument('--execute-train',action='store_true',help='Explicit training gate. DO NOT use while training remains prohibited.')
    args=parser.parse_args(argv)
    arms=[c for c in fixed_arms() if args.arm=='both' or c.return_loss==args.arm]
    seeds=SEEDS if args.seed is None else (args.seed,)
    plan={'status':'PREFLIGHT_ONLY_NOT_TRAINED','created_at_utc':utcnow(),'default_mode':'dry_run',
      'execute_flag_supplied':bool(args.execute_train),'training_authorization_inferred_from_this_file':False,
      'requested_device':args.device,'cuda_requirement':'Execution oncuda fails ifunavailable; CPUrequires explicit--devicecpu and is a documented platformchange.',
      'arms':[asdict(c) for c in arms],'seeds':list(seeds),'planned_run_count':len(arms)*len(seeds),
      'expected_batches_per_epoch':{'train':43,'train_last_batch':24,'val':10,'val_last_batch':7},
      'only_between_arm_config_difference':'return_loss','test_arrays_loaded':False,'test_scores_computed':False,
      'code_sha256':{p.name:sha(p) for p in (HERE/'models.py',HERE/'losses.py',HERE/'config.py',HERE/'data_contract.py',Path(__file__))},
      'actions':{'model_constructed':False,'model_forward':False,'backward':False,'optimizer_step':False,'checkpoint_written':False},
      'notes':['Globalgradientclip and joint checkpoint selection can couple the otherwise disjointNoSharing branches.',
        'Fixed0.7/0.3 coefficients do not equate effective taskgradient balance whenMSEis replaced byMAE.',
        'Rawpredictioninverse and583row selection arithmetic arefloat64; training tensors arefloat32.',
        'Architecture/source equality does not assert bitwise reproduction of archivedGPUruns underanother PyTorchversion/device.']}
    try:plan['installed_torch_version']=importlib.metadata.version('torch')
    except importlib.metadata.PackageNotFoundError:plan['installed_torch_version']=None
    try:
        payload=load_fit_data(args.dataset_dir,args.denominators)
        plan['preflight']='PASS';plan['dataset_identity']=payload['identity']
    except (FileNotFoundError,ValueError,KeyError,AssertionError) as exc:
        plan['status']='PREFLIGHT_FAILED_NOT_TRAINED';plan['preflight']='FAIL';plan['reason']=str(exc)
        args.plan_output.parent.mkdir(parents=True,exist_ok=True)
        args.plan_output.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(plan,ensure_ascii=False,indent=2));return 2
    args.plan_output.parent.mkdir(parents=True,exist_ok=True)
    args.plan_output.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if not args.execute_train:
        print(json.dumps({'status':plan['status'],'preflight':'PASS','planned_runs':plan['planned_run_count'],
           'training_started':False,'test_arrays_read':False,'plan':str(args.plan_output.resolve())},indent=2));return 0
    # EXPLICIT execution gate: preparation and default invocations never reach
    # imports/models/optimizers below. This branch was not run during authoring.
    if plan['installed_torch_version'] is None:raise RuntimeError('PyTorch is required for explicitly authorized training')
    output=args.output_dir.resolve();dataset=args.dataset_dir.resolve()
    allowed_runs=(HERE/'runs').resolve()
    if allowed_runs not in output.parents:
        raise RuntimeError('Training output must be a fresh subdirectory of this new training/runs workspace')
    if output==dataset or dataset in output.parents or output in dataset.parents:
        raise RuntimeError('Training outputs must be separate from dataset inputs')
    if output.exists():raise FileExistsError('Fresh output directory required; no implicit resume/overwrite')
    import torch
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; defaultdry-run is supported but execution requiresCUDAor explicit--devicecpu')
    output.mkdir(parents=True,exist_ok=False)
    results=[]
    for config in arms:
        for seed in seeds:
            result=execute_one_run(config,seed,payload,args.device,output/(config.return_loss+'_seed'+str(seed)))
            results.append(result)
    # Retain both declared arms. No test-error-based winner or arm selection.
    receipt={'status':'DECLARED_TRAINING_RUNS_FINISHED_CHECKPOINTS_FIXED_NO_TEST','finished_at_utc':utcnow(),
      'planned_run_count':len(arms)*len(seeds),'runs':results,'test_arrays_loaded':False,'test_scores_computed':False,
      'future_test_policy':'Evaluateeacharm afterits three checkpoints arefixed; arithmeticmean3seedrawpredictions. Separate testprogram not supplied bythis fitCLI.'}
    (output/'training_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':receipt['status'],'runs':len(results),'test_access':False},indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
