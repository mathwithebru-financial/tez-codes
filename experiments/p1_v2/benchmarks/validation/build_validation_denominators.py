"""P1 validation naive denominators only. No training/scaler/model import.

Raw scoring is explicitly float64. Existing P0 results are never overwritten.
"""
from pathlib import Path
import csv,hashlib,json,math,platform,sys
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
RAW=BASE/'inputs/raw_prices.csv'
MAP=BASE/'protocol/proposed_lag1_date_map.csv'
ASSETS=['BIST100','USDTRY','EURTRY','GOLD']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
before={p.relative_to(BASE).as_posix():sha(p) for p in [RAW,MAP]}
assert before['inputs/raw_prices.csv']=='ab5f275d38dc98057b1cedcf58019adb26be7402c7ed5ae6ee3d6877b2444893'
raw=pd.read_csv(RAW,index_col=0,parse_dates=True,dtype={a:'float64' for a in ASSETS})
with MAP.open(encoding='utf-8-sig',newline='') as f:allrows=list(csv.DictReader(f))
val=[r for r in allrows if r['split']=='val' and r['proposed_eligible']=='True']
NVAL=len(val)
assert NVAL==583
assert val[-1]['target_date']=='2022-10-04'
prices=raw.ffill();rets=np.log(prices/prices.shift(1));vol=rets.rolling(20,min_periods=20).std(ddof=1)*np.sqrt(252.)
positions={d.strftime('%Y-%m-%d'):i for i,d in enumerate(raw.index)}
present=raw.notna().to_numpy()
terms=[]
for row in val:
 j=positions[row['target_date']];k=j-2
 assert row['proposed_lag1_input_last']==raw.index[k].strftime('%Y-%m-%d')
 assert row['return_target_start_label']==raw.index[j-1].strftime('%Y-%m-%d')
 assert raw.index[j]>raw.index[k]
 # A10x8 input has30distinct raw price rows, across all4assets.
 all_input_raw=bool(present[k-29:k+1,:].all())
 for ai,a in enumerate(ASSETS):
  yret=float(rets.iloc[j,ai]);yvol=float(vol.iloc[j,ai])
  vp0=float(vol.iloc[j-1,ai]);vp1=float(vol.iloc[k,ai])
  raw_ret=bool(present[j-1:j+1,ai].all())
  raw_vol=bool(present[j-20:j+1,ai].all())
  terms.append({'asset':a,'target_date':row['target_date'],
   'return_start_label':row['return_target_start_label'],'P0_last_information_label':raw.index[j-1].strftime('%Y-%m-%d'),
   'P1_last_information_label':raw.index[k].strftime('%Y-%m-%d'),
   'return_true':yret,'vol20_true':yvol,'return_zero':0.,
   'return_persistence_P0':float(rets.iloc[j-1,ai]),'return_persistence_P1':float(rets.iloc[k,ai]),
   'vol_persistence_P0':vp0,'vol_persistence_P1':vp1,
   'return_zero_absolute_error':abs(yret),'vol_P0_pinball_tau05':.5*abs(yvol-vp0),
   'vol_P1_pinball_tau05':.5*abs(yvol-vp1),
   'target_return_raw_support':raw_ret,'target_vol20_raw_support':raw_vol,
   'full_model_input_raw_support':all_input_raw})
df=pd.DataFrame(terms)
assert df.shape[0]==NVAL*4
df.to_csv(OUT/'validation_naive_daily_terms.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
summaries=[];primary_ret={};primary_vol={};compat=[]
def finite_mean(x):
 arr=np.asarray(x,dtype=np.float64)
 if not len(arr):return None,'UNDEFINED_EMPTY_MASK'
 assert np.isfinite(arr).all()
 value=float(np.mean(arr,dtype=np.float64))
 return (value,'VALID_POSITIVE') if value>0 else (value,'UNDEFINED_ZERO_DENOMINATOR')
for a in ASSETS:
 sub=df[df.asset==a]
 yret=sub.return_true.to_numpy(dtype=np.float64);yvol=sub.vol20_true.to_numpy(dtype=np.float64)
 vp1=sub.vol_persistence_P1.to_numpy(dtype=np.float64)
 for task in ['return','volatility']:
  target_mask=sub.target_return_raw_support.to_numpy(bool) if task=='return' else sub.target_vol20_raw_support.to_numpy(bool)
  masks={'ALL_PRIMARY':np.ones(len(sub),dtype=bool),
   'TARGET_RAW_SUPPORT_ONLY':target_mask,
   'FULL_MODEL_INPUT_AND_TARGET_RAW_SUPPORT':target_mask & sub.full_model_input_raw_support.to_numpy(bool)}
  for maskname,mask in masks.items():
   e1=sub.return_zero_absolute_error.to_numpy(np.float64) if task=='return' else sub.vol_P1_pinball_tau05.to_numpy(np.float64)
   e0=sub.return_zero_absolute_error.to_numpy(np.float64) if task=='return' else sub.vol_P0_pinball_tau05.to_numpy(np.float64)
   p1,status=finite_mean(e1[mask]);p0,_=finite_mean(e0[mask]);n=int(mask.sum())
   summaries.append({'asset':a,'task':task,'scoring_population':maskname,'n':n,
    'P0_recomputed_F64_denominator':p0,'P1_F64_denominator':p1,
    'P1_over_P0':p1/p0 if p0 and p1 is not None else None,'status':status,
    'used_for_checkpoint_selection':maskname=='ALL_PRIMARY'})
   if maskname=='ALL_PRIMARY':
    assert status=='VALID_POSITIVE' and n==NVAL
    entry={'baseline':'ReturnZero' if task=='return' else 'VolPersistence_P1_last_admitted',
     'metric':'MAE' if task=='return' else 'PinballLoss_tau_0.5','value':p1,'n':n,
     'population':'ALL_PRIMARY','dtype':'float64','source_of_information':'constant0' if task=='return' else 'Vol20[t-1] predicting Vol20[t+1]'}
    (primary_ret if task=='return' else primary_vol)[a]=entry
    if task=='return':assert p0==p1
 # Compatibility diagnostics only: never used to change fixed scoring policy.
 compat.append({'asset':a,'return_denominator_F64':float(np.mean(np.abs(yret))),
  'return_denominator_round_target_float32_then_promote':float(np.mean(np.abs(yret.astype('float32').astype('float64')))),
  'vol_denominator_F64':float(np.mean(.5*np.abs(yvol-vp1))),
  'vol_denominator_round_target_only_float32':float(np.mean(.5*np.abs(yvol.astype('float32').astype('float64')-vp1))),
  'vol_denominator_round_both_float32':float(np.mean(.5*np.abs(yvol.astype('float32').astype('float64')-vp1.astype('float32').astype('float64'))))})
pd.DataFrame(summaries).to_csv(OUT/'validation_denominator_summary.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
pd.DataFrame(compat).to_csv(OUT/'float32_compatibility_diagnostics.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
# Independent scalar reductions; hand example shows the half-MAE factor.
scalar_max=0.
for a in ASSETS:
 rows=[r for r in terms if r['asset']==a]
 d_ret=math.fsum(abs(r['return_true']) for r in rows)/NVAL
 d_vol=math.fsum(.5*abs(r['vol20_true']-r['vol_persistence_P1']) for r in rows)/NVAL
 scalar_max=max(scalar_max,abs(d_ret-primary_ret[a]['value']),abs(d_vol-primary_vol[a]['value']))
assert scalar_max<1e-14
assert abs(.5*(abs(.20-.10)+abs(.30-.50))/2-.075)<1e-15
after={p.relative_to(BASE).as_posix():sha(p) for p in [RAW,MAP]};assert before==after
target_dates=[r['target_date'] for r in val]
truth=np.column_stack([df[df.asset==a].return_true.to_numpy(np.float64) for a in ASSETS]+[df[df.asset==a].vol20_true.to_numpy(np.float64) for a in ASSETS])
date_digest=hashlib.sha256(('\n'.join(target_dates)+'\n').encode('utf-8')).hexdigest()
truth_digest=hashlib.sha256(np.ascontiguousarray(truth,dtype='<f8').tobytes()).hexdigest()
spec={'date':'2026-11-13','version':2,'status':'LOCKED_STRICT_F_K_VALIDATION_ONLY_F64_DENOMINATORS__NO_MODEL_RUN',
 'protocol':'P1lag1_strict_information_boundary_corrected','split':'selection_validation_only','validation_rows':NVAL,
 'validation_target_start':val[0]['target_date'],'validation_target_end':val[-1]['target_date'],
 'P1_last_admitted_start':df.P1_last_information_label.min(),'P1_last_admitted_end':df.P1_last_information_label.max(),
 'return_denominator':primary_ret,'volatility_denominator':primary_vol,
 'selection_score_rule':'0.5*mean_asset(model_return_MAE/Dret)+0.5*mean_asset(model_vol_pinball_tau05/Dvol)',
 'shared_for_return_loss_arms':['MSE','MAE'],
 'validation_target_dates_sha256':date_digest,'scoring_truth_sha256_float64_little_endian':truth_digest,
 'output_order':[a+'_NextRet' for a in ASSETS]+[a+'_NextVol' for a in ASSETS],
 'truth_shape':[NVAL,8],
 'target_scoring_dtype':'float64 canonical raw→ffill→log→20rollingstd(ddof1)*sqrt252; no target rounding',
 'future_prediction_scoring_dtype':'Promote inverse-scaled predictions tofloat64 for reductions; do not round float64truth tofloat32 for scoring',
 'future_training_tensor_dtype':'float32 may remain; no tensor/scaler created here',
 'legacy_dtype_boundary':'Original03 denominator CSVs use.astype(float), while06evaluates stored y_val_raw float32; old publishedP0 results unchanged. NewP0_F64_recomputed is a diagnostic reference, not a replacement.',
 'numeric_provenance':'pandas.read_csv with per-asset float64 and parse_dates, default C-parser precision; ffill then np.log(price/shift1), pandas.rolling20.std(ddof1)*sqrt252; reductions float64. Serialized daily terms use17significant digits and stdlibfloat round-trip for scoring reference.',
 'primary_mask':'All583strict-F_k eligible selection-validation targets; Oct5boundarylabel excluded before values calculated. Test584unchanged.',
 'boundary_rule':'traininglabels<=firstvalidationadmittedcutoff;selectionlabels<=firsttestadmittedcutoff. Val2022-10-05ispost-selectiondiagnostic only, never a denominator or checkpoint-selection row.',
 'secondary_masks':'Raw target support and full-model-input+target support are diagnostic populations only; each needs matched baseline denominator. Not used for checkpoint/model selection.',
 'undefined_denominator_rule':'Return null/status for empty or zero denominators; never epsilon-fill or silently omit an asset from primary equal-asset score',
 'no_test_used_for_denominator_values':True,
 'test_dates_in_date_map_metadata':'Input map includes split labels; only validation rows selected before numerical denominator evaluation. No test metric or value used.',
 'source_hashes':before,'canonical_source_unchanged':True,
 'source_code_reference':['03_baseline_sanity_v4.py:496–532,702–731','06_best_model_multiseed_v4.py:310–358,1228–1244'],
 'actions':{'neural_training':False,'GARCH_fit':False,'learned_model_inference':False,'scaler_fit':False,'deterministic_naive_calculation':True,'thesis_write':False}}
(OUT/'selection_denominators_P1_validation_v2.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'selection_denominators_P1_validation_v1.json').write_text(json.dumps({'status':'SUPERSEDED_NOT_FOR_MODEL_SELECTION','reason':'V1didnotpurgeupstreamtrainingandselectionlabelsforstrictF_k','current_file':'selection_denominators_P1_validation_v2.json','historical_snapshot':'v1_superseded/selection_denominators_P1_validation_v1.json'},indent=2),encoding='utf-8')
verification={'status':'PASS','version':2,'n_validation':NVAL,'n_asset_dates':len(terms),'positive_primary_denominators':8,
 'P0_P1_return_zero_denominators_exactly_equal':True,'maximum_scalar_reduction_difference':scalar_max,
 'source_hashes_before':before,'source_hashes_after':after,'hand_example_tau05_pinball':.075,
 'hand_example_explanation':'truth[.20,.30],prediction[.10,.50]:abs errors[.10,.20],MAE.15,pinball.075',
 'primary_summary':[r for r in summaries if r['scoring_population']=='ALL_PRIMARY'],
 'output_sha256':{n:sha(OUT/n) for n in ['validation_naive_daily_terms.csv','validation_denominator_summary.csv','float32_compatibility_diagnostics.csv','selection_denominators_P1_validation_v2.json']},
 'runtime':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__},'training_or_fit':False}
(OUT/'denominator_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':'PASS','positive_primary_denominators':8,'primary':verification['primary_summary'],'max_scalar_difference':scalar_max},ensure_ascii=True,indent=2))
