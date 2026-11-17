"""Strict primary validation score. No fitting, network or model import."""
from pathlib import Path
import csv,hashlib,json
import numpy as np

ASSETS=['BIST100','USDTRY','EURTRY','GOLD']
OUTPUT_ORDER=[a+'_NextRet' for a in ASSETS]+[a+'_NextVol' for a in ASSETS]

def reference_arrays(folder=None):
    """Read reference truth and allowed naive values; not learned predictions."""
    folder=Path(folder) if folder is not None else Path(__file__).resolve().parent
    with (folder/'validation_naive_daily_terms.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    by_asset={a:[r for r in rows if r['asset']==a] for a in ASSETS}
    dates=[r['target_date'] for r in by_asset[ASSETS[0]]]
    assert all([r['target_date'] for r in by_asset[a]]==dates for a in ASSETS)
    truth=np.column_stack([[float(r['return_true']) for r in by_asset[a]] for a in ASSETS]+[[float(r['vol20_true']) for r in by_asset[a]] for a in ASSETS])
    naive=np.column_stack([np.zeros(len(dates),dtype=np.float64) for a in ASSETS]+[[float(r['vol_persistence_P1']) for r in by_asset[a]] for a in ASSETS])
    return dates,np.asarray(truth,dtype=np.float64),np.asarray(naive,dtype=np.float64)

def primary_validation_score(y_true,y_pred,target_dates,*,output_order,denominator_path=None):
    """No custom mask: selection uses the fixed strict-F_k dates and8outputs.

    A secondary quality-subset summary must be reported separately and is not
    accepted here as a checkpoint selection population.
    """
    path=Path(denominator_path) if denominator_path is not None else Path(__file__).resolve().parent/'selection_denominators_P1_validation_v2.json'
    d=json.loads(path.read_text(encoding='utf-8'))
    assert list(output_order)==d['output_order']==OUTPUT_ORDER,'Output order mismatch'
    truth=np.asarray(y_true,dtype=np.float64)
    pred=np.asarray(y_pred,dtype=np.float64)
    n=d['validation_rows']
    assert d['version']==2 and n==583
    assert list(truth.shape)==list(pred.shape)==d['truth_shape']==[n,8],'Shape mismatch'
    assert np.isfinite(truth).all() and np.isfinite(pred).all(),'Nonfinite truth/prediction'
    dates=list(target_dates)
    assert len(dates)==n and len(set(dates))==n
    digest=hashlib.sha256(('\n'.join(dates)+'\n').encode('utf-8')).hexdigest()
    assert digest==d['validation_target_dates_sha256'],'Target dates/order mismatch'
    truthdigest=hashlib.sha256(np.ascontiguousarray(truth,dtype='<f8').tobytes()).hexdigest()
    assert truthdigest==d['scoring_truth_sha256_float64_little_endian'],'Truth values/dtype-boundary mismatch'
    errors=truth-pred
    ret_mae=np.mean(np.abs(errors[:,:4]),axis=0,dtype=np.float64)
    vol_pin=.5*np.mean(np.abs(errors[:,4:]),axis=0,dtype=np.float64)
    rden=np.asarray([d['return_denominator'][a]['value'] for a in ASSETS],dtype=np.float64)
    vden=np.asarray([d['volatility_denominator'][a]['value'] for a in ASSETS],dtype=np.float64)
    assert np.all(np.isfinite(rden)&(rden>0)) and np.all(np.isfinite(vden)&(vden>0))
    rr=ret_mae/rden;vr=vol_pin/vden
    score=.5*np.mean(rr,dtype=np.float64)+.5*np.mean(vr,dtype=np.float64)
    return {'validation_score':float(score),'mean_return_ratio':float(np.mean(rr,dtype=np.float64)),
      'mean_volatility_ratio':float(np.mean(vr,dtype=np.float64)),
      'per_asset':{a:{'return_MAE':float(ret_mae[i]),'vol_Pinball_tau05':float(vol_pin[i]),'return_ratio':float(rr[i]),'vol_ratio':float(vr[i])} for i,a in enumerate(ASSETS)},
      'population':'ALL_PRIMARY_STRICT_F_K_SELECTION','n_per_output':n,'dtype':'float64','scoring_function_runs_a_model':False}

def check_contract():
    dates,truth,naive=reference_arrays()
    s0=primary_validation_score(truth,truth,dates,output_order=OUTPUT_ORDER)
    s1=primary_validation_score(truth,naive,dates,output_order=OUTPUT_ORDER)
    assert s0['validation_score']==0.
    assert abs(s1['validation_score']-1.)<1e-14
    rejects={}
    cases={'reversed_dates':(truth,naive,list(reversed(dates)),OUTPUT_ORDER),
      'wrong_output_order':(truth,naive,dates,list(reversed(OUTPUT_ORDER))),
      'rounded_truth':(truth.astype('float32'),naive,dates,OUTPUT_ORDER),
      'short_population':(truth[:-1],naive[:-1],dates[:-1],OUTPUT_ORDER)}
    for name,(yt,yp,dt,order) in cases.items():
        try:primary_validation_score(yt,yp,dt,output_order=order)
        except AssertionError:rejects[name]=True
        else:raise AssertionError('Invalid input not rejected: '+name)
    return {'status':'PASS','perfect_reference_score':s0['validation_score'],
      'matched_naive_reference_score':s1['validation_score'],'invalid_inputs_rejected':rejects,
      'training_or_learned_model_inference':False,'synthetic_contract_check_only':True}

if __name__=='__main__':
    result=check_contract()
    (Path(__file__).resolve().parent/'score_contract_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
