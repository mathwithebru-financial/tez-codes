"""Standalone small raw-data/return/volatility/window checks.

Requires Python, NumPy and pandas only. No archived source execution, feature
dataset export, scaler fitting, model/tensor artifact creation or deserialization.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import platform
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parents[1]
EXPECTED_SHA='ab5f275d38dc98057b1cedcf58019adb26be7402c7ed5ae6ee3d6877b2444893'
ASSETS=['BIST100','USDTRY','EURTRY','GOLD']
MISSING={'BIST100':153,'USDTRY':5,'EURTRY':3,'GOLD':141}
FEATURE_COLUMNS=[column for asset in ASSETS for column in [asset+'_LogRet',asset+'_Vol20']]
TARGET_COLUMNS=[asset+'_NextRet' for asset in ASSETS]+[asset+'_NextVol' for asset in ASSETS]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def iso(value):return str(pd.Timestamp(value).date())
def scalar_vol(values):
    mean=math.fsum(values)/len(values)
    variance=math.fsum((v-mean)**2 for v in values)/(len(values)-1)
    return math.sqrt(variance)*math.sqrt(252)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,default=BASE/'inputs/raw_prices.csv')
    parser.add_argument('--output-dir',type=Path,default=BASE/'validation')
    args=parser.parse_args();raw_path=args.raw.resolve();out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    raw_before=sha(raw_path)
    assert raw_before==EXPECTED_SHA,'Frozen raw source hash mismatch'
    # Explicit round-trip parsing matches Python float() of original CSV literals.
    raw=pd.read_csv(raw_path,index_col=0,parse_dates=True,float_precision='round_trip')
    assert list(raw.columns)==ASSETS and raw.shape==(3912,4)
    assert raw.index.is_unique and raw.index.is_monotonic_increasing
    assert raw.index.tz is None and raw.index.equals(raw.index.normalize())
    assert (raw.index.weekday<5).all()
    assert iso(raw.index[0])=='2010-01-04' and iso(raw.index[-1])=='2024-12-31'
    assert raw.isna().sum().to_dict()==MISSING
    observed=raw.to_numpy();present_values=observed[~np.isnan(observed)]
    assert np.isfinite(present_values).all() and (present_values>0).all()

    # Independent standard-library forward fill; no pandas fill used here.
    scalar_clean=[];scalar_dates=[];last=[None]*4;leading=[0]*4
    with raw_path.open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle)
        assert reader.fieldnames[1:]==ASSETS
        date_column=reader.fieldnames[0]
        for source_row in reader:
            scalar_dates.append(pd.Timestamp(source_row[date_column]))
            values=[]
            for j,asset in enumerate(ASSETS):
                token=source_row[asset].strip()
                value=float(token) if token else None
                if value is not None:last[j]=value
                elif last[j] is None:leading[j]+=1
                values.append(last[j] if last[j] is not None else float('nan'))
            scalar_clean.append(values)
    clean=raw.ffill();scalar_clean=np.asarray(scalar_clean,dtype=np.float64)
    assert pd.DatetimeIndex(scalar_dates).equals(raw.index)
    assert np.array_equal(scalar_clean,clean.to_numpy()),'Independent scalar/pandas ffill mismatch'
    assert not any(leading) and int(clean.isna().sum().sum())==0
    raw_present=raw.notna()
    assert np.array_equal(raw.to_numpy()[raw_present.to_numpy()],clean.to_numpy()[raw_present.to_numpy()])
    changed_by_fill=raw.isna() & clean.notna()
    assert changed_by_fill.sum().to_dict()==MISSING

    returns=np.log(clean/clean.shift(1))
    vol20=returns.rolling(20,min_periods=20).std(ddof=1)*np.sqrt(252)
    feature=pd.DataFrame(index=raw.index)
    target=pd.DataFrame(index=raw.index)
    for asset in ASSETS:
        feature[asset+'_LogRet']=returns[asset]
        feature[asset+'_Vol20']=vol20[asset]
    for asset in ASSETS:target[asset+'_NextRet']=returns[asset].shift(-1)
    for asset in ASSETS:target[asset+'_NextVol']=vol20[asset].shift(-1)
    assert list(feature.columns)==FEATURE_COLUMNS and list(target.columns)==TARGET_COLUMNS
    paired=feature.notna().all(axis=1) & target.notna().all(axis=1)
    assert int(paired.sum())==3891
    assert feature.loc[paired].shape==(3891,8) and target.loc[paired].shape==(3891,8)
    assert iso(feature.index[paired][0])=='2010-02-01'
    assert iso(feature.index[paired][-1])=='2024-12-30'
    raw_present_equal=raw_present & raw_present.shift(1,fill_value=False) & clean.eq(clean.shift(1))
    filled_current_zero=changed_by_fill & returns.eq(0)
    assert filled_current_zero.sum().to_dict()==MISSING

    samples=[];date_rows=[];scalar_comparisons=[]
    for anchor_text,target_text in [('2022-10-05','2022-10-06'),('2022-10-07','2022-10-10')]:
        anchor=pd.Timestamp(anchor_text);realization=pd.Timestamp(target_text)
        a=raw.index.get_loc(anchor);u=raw.index.get_loc(realization)
        assert u==a+1,'Expected next common-calendar observation'
        positions=list(range(a-9,a+1));dates=raw.index[positions]
        assert len(dates)==10 and dates[-1]==anchor and realization not in dates
        assert dates.max()<realization and np.isfinite(feature.iloc[positions].to_numpy()).all()
        assert raw.index[a-29]<dates[0] and raw.index[a]==anchor
        for position_index,position in enumerate(positions):
            date_rows.append({'anchor_date':anchor_text,'target_realization_date':target_text,
                'window_position':position_index,'feature_date':iso(raw.index[position]),
                'feature_latest_raw_price_date':iso(raw.index[position]),
                'feature_vol20_first_raw_price_date':iso(raw.index[position-20]),
                'target_date_in_input_window':False})
        for asset in ASSETS:
            j=ASSETS.index(asset)
            for pos in sorted(set(positions+[u])):
                r_scalar=math.log(scalar_clean[pos,j]/scalar_clean[pos-1,j])
                history=[math.log(scalar_clean[k,j]/scalar_clean[k-1,j]) for k in range(pos-19,pos+1)]
                v_scalar=scalar_vol(history)
                rd=abs(r_scalar-float(returns.iloc[pos,j]));vd=abs(v_scalar-float(vol20.iloc[pos,j]))
                assert rd<1e-14 and vd<1e-12
                scalar_comparisons.append({'sample_anchor_date':anchor_text,'date':iso(raw.index[pos]),'asset':asset,
                    'scalar_vs_numpy_logreturn_abs_difference':rd,
                    'scalar_sample_std_vs_pandas_vol20_abs_difference':vd})
            assert float(target.loc[anchor,asset+'_NextRet'])==float(returns.loc[realization,asset])
            assert float(target.loc[anchor,asset+'_NextVol'])==float(vol20.loc[realization,asset])
        samples.append({'anchor_date':anchor_text,'target_realization_date':target_text,
            'input_shape':[10,8],'target_shape':[8],
            'input_feature_dates':[iso(d) for d in dates],
            'first_input_feature_values':feature.loc[dates[0]].to_dict(),
            'last_input_feature_values':feature.loc[anchor].to_dict(),
            'target_values_in_declared_order':target.loc[anchor].to_dict(),
            'combined_input_raw_price_support':[iso(raw.index[a-29]),anchor_text],
            'target_return_price_support':[anchor_text,target_text],
            'target_vol20_price_support':[iso(raw.index[u-20]),target_text],
            'target_is_next_common_observation':True,
            'calendar_days_anchor_to_target':int((realization-anchor).days),
            'target_date_absent_from_feature_window_and_price_support':True,
            'clock_availability_or_trade_validity_certified':False})

    source_manifest=BASE/'inputs/data_source_manifest.json'
    provenance=json.loads(source_manifest.read_text(encoding='utf-8')) if source_manifest.is_file() else None
    if provenance:
        for record in provenance['records']:
            staged=BASE/record['destination_relative_path']
            assert sha(staged)==record['destination_sha256']
    raw_after=sha(raw_path);assert raw_before==raw_after
    result={'status':'PASS_STANDALONE_SMALL_DATA_CHECKS','raw_sha256':raw_before,
        'raw_sha256_after':raw_after,'raw_unchanged':True,'raw_shape':[3912,4],
        'period':['2010-01-04','2024-12-31'],'columns':ASSETS,
        'date_index':{'monotonic':True,'unique':True,'Monday_through_Friday_only':True,
            'date_only_timezone_naive':True,'warning':'Weekday labels are not proof that every asset had an open session.'},
        'missing_before_fill':MISSING,'forward_filled_counts':changed_by_fill.sum().to_dict(),
        'leading_missing_counts':dict(zip(ASSETS,leading)),'remaining_missing_after_fill':clean.isna().sum().to_dict(),
        'independent_scalar_ffill_matches_pandas_bitwise':True,
        'raw_present_prices_unchanged_by_fill':True,
        'zero_returns_from_current_price_ffill':filled_current_zero.sum().to_dict(),
        'zero_returns_with_both_raw_endpoints_present_equal':raw_present_equal.sum().to_dict(),
        'raw_present_equal_price_caution':'Provider equality is not proof of a new unchanged traded close.',
        'return_formula':'log(P_u/P_previous_common_row)',
        'vol20_formula':'sample std of20 log returns, ddof=1, multiplied bysqrt252; uses21 prices',
        'feature_columns':FEATURE_COLUMNS,'target_columns':TARGET_COLUMNS,
        'paired_valid_in_memory_shape':{'features':[3891,8],'targets':[3891,8]},
        'independent_scalar_checks':{'comparisons':len(scalar_comparisons),
            'unique_asset_date_pairs':len({(r['asset'],r['date']) for r in scalar_comparisons}),
            'maximum_return_abs_difference':max(r['scalar_vs_numpy_logreturn_abs_difference'] for r in scalar_comparisons),
            'maximum_vol20_abs_difference':max(r['scalar_sample_std_vs_pandas_vol20_abs_difference'] for r in scalar_comparisons)},
        'sample_checks':samples,
        'limitations':['Daily-index target exclusion is verified. Historical provider bar/publication timestamps and a forecast cutoff are not verified; this is not a clock-level no-leakage certificate.',
            'Current data filling is forward-only; no backward fill was executed. The copied historical v3 source may contain other preparation operations and was not executed.',
            'No full feature/target/tensor dataset was exported, no scaler fitted/loaded, no model/checkpoint/pickle imported, and no training/inference performed.'],
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},
        'source_staging_manifest_status':provenance['status'] if provenance else 'NOT_PRESENT_STANDALONE_RAW_CHECK_ONLY',
        'data_or_source_modified':False,'new_tensor_artifacts_created':False,
        'scaler_fit_or_model_deserialization':False}
    pd.DataFrame(date_rows).to_csv(out/'small_window_date_map.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(scalar_comparisons).to_csv(out/'scalar_numeric_comparisons.csv',index=False,encoding='utf-8-sig')
    (out/'data_integrity_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (out/'sample_window_checks.json').write_text(json.dumps({'feature_columns':FEATURE_COLUMNS,'target_columns':TARGET_COLUMNS,'samples':samples},ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'raw_shape':result['raw_shape'],'missing':MISSING,
        'independent_scalar_checks':result['independent_scalar_checks'],'raw_unchanged':True,'sample_pairs':[(r['anchor_date'],r['target_realization_date']) for r in samples]}))

if __name__=='__main__':main()
