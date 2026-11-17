"""Standalone P1 V2 scoring-mask audit from raw presence and the proposed date map.

Masks are metadata for scoring. No rows/targets are removed, no scaler/model is
loaded, no numeric targets regenerated, and no fit/inference is performed.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parents[1]
ASSETS=['BIST100','USDTRY','EURTRY','GOLD']
EXPECTED_RAW_SHA='ab5f275d38dc98057b1cedcf58019adb26be7402c7ed5ae6ee3d6877b2444893'
PROTOCOL_VERSION='P1_V2_STRICT_INFORMATION_BOUNDARY'
EXPECTED_ELIGIBLE={'train':2712,'val':583,'test':584}
EXPECTED_EXCLUDED={('train','2010-02-15'),('train','2020-07-09'),('val','2022-10-05')}
EVENT_DATE=pd.Timestamp('2023-02-08')
EVENT_URL='https://www.kap.org.tr/tr/Bildirim/1111497'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def iso(day):return str(pd.Timestamp(day).date())
def proportion(n,d):return 100*n/d if d else None

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw',type=Path,default=BASE/'inputs/raw_prices.csv')
    ap.add_argument('--date-map',type=Path,default=BASE/'protocol/proposed_lag1_date_map.csv')
    ap.add_argument('--output-dir',type=Path,default=BASE/'scoring')
    args=ap.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    paths=[args.raw.resolve(),args.date_map.resolve(),BASE/'inputs/data_source_manifest.json',BASE/'validation/data_integrity_checks.json']
    before={str(p.resolve()):sha(p) for p in paths}
    assert sha(args.raw)==EXPECTED_RAW_SHA
    raw=pd.read_csv(args.raw,index_col=0,parse_dates=True,float_precision='round_trip')
    dm=pd.read_csv(args.date_map)
    assert list(raw.columns)==ASSETS and raw.shape==(3912,4)
    assert raw.index.is_unique and raw.index.is_monotonic_increasing
    assert dm.proposed_eligible.dtype==bool
    assert dm.feature_window_eligible.dtype==bool
    assert dm.training_or_selection_eligible.equals(dm.proposed_eligible)
    assert dm.boundary_diagnostic_only.dtype==bool
    assert len(dm)==3882 and not dm.duplicated(['split','original_split_sample_index']).any()
    assert not dm.proposal_executed_as_model_experiment.any()
    assert dm.groupby('split').proposed_eligible.sum().to_dict()==EXPECTED_ELIGIBLE
    excluded=dm.loc[~dm.proposed_eligible].copy()
    assert set(zip(excluded.split,excluded.target_date))==EXPECTED_EXCLUDED
    assert excluded.exclusion_reason.notna().all() and excluded.exclusion_reason.nunique()==3
    assert dm.loc[dm.proposed_eligible,'exclusion_reason'].isna().all()
    assert dm.loc[dm.boundary_diagnostic_only,['split','target_date']].to_dict(orient='records')==[{'split':'val','target_date':'2022-10-05'}]
    eligible_dm=dm.loc[dm.proposed_eligible]
    boundary_checks=[]
    for previous,next_split in [('train','val'),('val','test')]:
        last_label=str(eligible_dm.loc[eligible_dm.split==previous,'target_date'].max())
        first_next_input_end=str(eligible_dm.loc[eligible_dm.split==next_split,'proposed_lag1_input_last'].iloc[0])
        assert pd.Timestamp(last_label)<=pd.Timestamp(first_next_input_end)
        boundary_checks.append({'previous_split':previous,'next_split':next_split,
            'max_primary_target_label':last_label,'first_next_split_input_end':first_next_input_end,
            'target_label_not_after_first_next_input_end':True})

    present=raw.notna().to_numpy()
    # Independent stdlib presence parsing; no float arithmetic/price filling.
    scalar_dates=[];scalar_present=[]
    with args.raw.open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle);date_column=reader.fieldnames[0]
        assert reader.fieldnames[1:]==ASSETS
        for row in reader:
            scalar_dates.append(pd.Timestamp(row[date_column]))
            scalar_present.append([bool(row[a].strip()) for a in ASSETS])
    assert pd.DatetimeIndex(scalar_dates).equals(raw.index)
    assert np.array_equal(present,np.asarray(scalar_present,dtype=bool))
    missing_prefix=np.vstack([np.zeros((1,4),dtype=np.int64),np.cumsum(~np.asarray(scalar_present,dtype=bool),axis=0)])
    def prefix_complete(start,end,j):
        return start>=0 and end<len(raw) and int(missing_prefix[end+1,j]-missing_prefix[start,j])==0

    records=[];bounds=[];independent_checks=0
    for original_global_index,row in enumerate(dm.itertuples(index=False)):
        u=raw.index.get_loc(pd.Timestamp(row.target_date));t=raw.index.get_loc(pd.Timestamp(row.return_target_start_label))
        ret_end=raw.index.get_loc(pd.Timestamp(row.return_target_end_label))
        first_feature=raw.index.get_loc(pd.Timestamp(row.proposed_lag1_input_first))
        last_feature=raw.index.get_loc(pd.Timestamp(row.proposed_lag1_input_last))
        vol_first_return=raw.index.get_loc(pd.Timestamp(row.vol20_first_return_label))
        vol_last_return=raw.index.get_loc(pd.Timestamp(row.vol20_last_return_label))
        eligible=bool(row.proposed_eligible)
        history_support_available=bool(first_feature>=20)
        exclusion_reason='' if pd.isna(row.exclusion_reason) else str(row.exclusion_reason)
        assert bool(row.feature_window_eligible)==history_support_available
        assert bool(row.training_or_selection_eligible)==eligible
        assert t==u-1 and ret_end==u
        assert last_feature==t-1 and first_feature==t-10 and last_feature-first_feature+1==10
        assert vol_first_return==u-19 and vol_last_return==u
        assert not eligible or history_support_available
        assert int(row.proposed_horizon_common_rows_from_last_input_to_target)==u-last_feature==2
        # Derive supports from each return and feature definition, then compare offsets.
        return_support=sorted({t,u})
        vol_return_positions=range(vol_first_return,vol_last_return+1)
        vol_support=sorted({price for r in vol_return_positions for price in [r-1,r]})
        input_support=sorted({price for f in range(first_feature,last_feature+1) for r in range(f-19,f+1) for price in [r-1,r]})
        assert return_support==list(range(t,u+1)) and len(return_support)==2
        assert vol_support==list(range(t-19,t+2)) and len(vol_support)==21
        assert input_support==list(range(t-30,t)) and len(input_support)==30
        assert max(input_support)<t<u
        assert (min(input_support)>=0)==history_support_available
        all_input_complete=bool(present[input_support].all()) if history_support_available else False
        independent_all_input=history_support_available and all(prefix_complete(t-30,t-1,j) for j in range(4))
        assert all_input_complete==independent_all_input;independent_checks+=1
        bounds.append({'original_global_sample_index':original_global_index,'split':row.split,
            'original_split_sample_index':int(row.original_split_sample_index),'proposed_eligible':eligible,
            'exclusion_reason':exclusion_reason,'input_history_support_available':history_support_available,
            'feature_window_eligible':bool(row.feature_window_eligible),
            'training_or_selection_eligible':bool(row.training_or_selection_eligible),
            'boundary_diagnostic_only':bool(row.boundary_diagnostic_only),
            'anchor_label_t':iso(raw.index[t]),'target_date_u':iso(raw.index[u]),
            'input_feature_start':iso(raw.index[first_feature]),'input_feature_end':iso(raw.index[last_feature]),
            'input_raw_price_start':iso(raw.index[min(input_support)]) if history_support_available else None,
            'input_raw_price_end':iso(raw.index[max(input_support)]),
            'input_raw_price_count':30,'input_asset_price_cells':120,
            'return_target_raw_price_start':iso(raw.index[t]),'return_target_raw_price_end':iso(raw.index[u]),
            'vol20_target_raw_price_start':iso(raw.index[min(vol_support)]),'vol20_target_raw_price_end':iso(raw.index[u]),
            'vol20_raw_price_count':21,'vol20_return_count':20})
        for j,asset in enumerate(ASSETS):
            target_price_present=bool(present[u,j])
            ret_raw_complete=bool(present[return_support,j].all())
            vol_raw_complete=bool(present[vol_support,j].all())
            asset_input_complete=bool(present[input_support,j].all()) if history_support_available else False
            assert ret_raw_complete==prefix_complete(t,u,j)
            assert vol_raw_complete==prefix_complete(t-19,u,j)
            assert asset_input_complete==(history_support_available and prefix_complete(t-30,t-1,j))
            independent_checks+=3
            event_position=raw.index.get_loc(EVENT_DATE)
            event_ret=asset=='BIST100' and event_position in return_support
            event_vol=asset=='BIST100' and event_position in vol_support
            event_asset_input=asset=='BIST100' and event_position in input_support
            event_all_input=event_position in input_support
            records.append({'original_global_sample_index':original_global_index,'split':row.split,
                'original_split_sample_index':int(row.original_split_sample_index),'asset':asset,
                'anchor_label_t':iso(raw.index[t]),'target_date':iso(raw.index[u]),
                'proposed_input_first':iso(raw.index[first_feature]),'proposed_input_last':iso(raw.index[last_feature]),
                'proposed_eligible':eligible,
                'protocol_version':PROTOCOL_VERSION,'exclusion_reason':exclusion_reason,
                'input_history_support_available':history_support_available,
                'feature_window_eligible':bool(row.feature_window_eligible),
                'training_or_selection_eligible':bool(row.training_or_selection_eligible),
                'boundary_diagnostic_only':bool(row.boundary_diagnostic_only),
                'primary_return_mask':eligible,'primary_vol20_mask':eligible,
                'raw_present_target_price_only':target_price_present,
                'raw_present_both_return_endpoints':ret_raw_complete,
                'raw_present_all21_vol20_target_prices':vol_raw_complete,
                'raw_present_all30_asset_input_prices':asset_input_complete,
                'raw_present_all120_model_input_price_cells':all_input_complete,
                'secondary_return_raw_target_mask':eligible and ret_raw_complete,
                'secondary_vol20_raw_target_mask':eligible and vol_raw_complete,
                'strict_all_model_inputs_return_target_mask':eligible and all_input_complete and ret_raw_complete,
                'strict_all_model_inputs_vol20_target_mask':eligible and all_input_complete and vol_raw_complete,
                'strict_asset_input_return_target_mask':eligible and asset_input_complete and ret_raw_complete,
                'strict_asset_input_vol20_target_mask':eligible and asset_input_complete and vol_raw_complete,
                'known_cancelled_session_date_in_return_target_support':bool(event_ret),
                'known_cancelled_session_date_in_vol20_target_support':bool(event_vol),
                'known_cancelled_session_date_in_asset_input_support':bool(event_asset_input),
                'known_cancelled_BIST_date_in_model_input_support':bool(event_all_input),
                'event_quality_flags_applied_as_exclusions':False,
                'raw_presence_certifies_trade_or_clock_availability':False})

    masks=pd.DataFrame(records);support=pd.DataFrame(bounds)
    assert len(masks)==3882*4 and len(support)==3882
    assert not masks.duplicated(['split','original_split_sample_index','asset']).any()
    bool_columns=[c for c in masks if masks[c].dtype==bool]
    assert masks.primary_return_mask.equals(masks.proposed_eligible)
    assert masks.primary_vol20_mask.equals(masks.proposed_eligible)
    assert int((~masks.proposed_eligible).sum())==12
    count_rows=[]
    for split in ['train','val','test']:
        for asset in ASSETS:
            ss=masks[(masks.split==split)&(masks.asset==asset)];eligible_count=int(ss.proposed_eligible.sum())
            for task,primary,secondary,strict_global,strict_asset,quality in [
                ('return','primary_return_mask','secondary_return_raw_target_mask','strict_all_model_inputs_return_target_mask','strict_asset_input_return_target_mask','known_cancelled_session_date_in_return_target_support'),
                ('vol20','primary_vol20_mask','secondary_vol20_raw_target_mask','strict_all_model_inputs_vol20_target_mask','strict_asset_input_vol20_target_mask','known_cancelled_session_date_in_vol20_target_support')]:
                count_rows.append({'split':split,'asset':asset,'task':task,'original_mapped_samples':len(ss),
                    'p1_eligible_samples':eligible_count,'primary_all_eligible_count':int(ss[primary].sum()),
                    'secondary_raw_target_support_count':int(ss[secondary].sum()),
                    'secondary_pct_of_eligible':proportion(int(ss[secondary].sum()),eligible_count),
                    'strict_all_model_inputs_plus_task_target_count':int(ss[strict_global].sum()),
                    'strict_all_model_inputs_pct_of_eligible':proportion(int(ss[strict_global].sum()),eligible_count),
                    'strict_asset_inputs_plus_task_target_count':int(ss[strict_asset].sum()),
                    'target_price_only_raw_present_count':int((ss.proposed_eligible&ss.raw_present_target_price_only).sum()),
                    'known_event_quality_flag_target_count_without_exclusion':int((ss.proposed_eligible&ss[quality]).sum())})

    # Concrete examples; they demonstrate why raw presence and actual event validity differ.
    case_dates=['2010-02-15','2020-07-09','2022-10-05','2022-10-06','2022-10-10','2023-02-08','2023-02-09','2023-02-15','2023-12-25','2024-12-25']
    examples=[]
    for day in case_dates:
        r=masks[(masks.asset=='BIST100')&(masks.target_date==day)].iloc[0].to_dict()
        u=raw.index.get_loc(pd.Timestamp(day));r.update({'raw_previous_date':iso(raw.index[u-1]),
            'raw_previous_BIST_price':None if pd.isna(raw.iloc[u-1,0]) else float(raw.iloc[u-1,0]),
            'raw_target_BIST_price':None if pd.isna(raw.iloc[u,0]) else float(raw.iloc[u,0]),
            'raw_BIST_endpoints_present_and_equal':bool(pd.notna(raw.iloc[u-1,0]) and pd.notna(raw.iloc[u,0]) and raw.iloc[u-1,0]==raw.iloc[u,0])})
        examples.append(r)
    feb8=next(r for r in examples if r['target_date']=='2023-02-08')
    assert feb8['raw_BIST_endpoints_present_and_equal'] and feb8['secondary_return_raw_target_mask']
    assert feb8['known_cancelled_session_date_in_return_target_support']
    feb9=next(r for r in examples if r['target_date']=='2023-02-09')
    assert feb9['primary_return_mask'] and not feb9['secondary_return_raw_target_mask']
    feb15=next(r for r in examples if r['target_date']=='2023-02-15')
    assert feb15['raw_present_target_price_only'] and not feb15['secondary_return_raw_target_mask']

    masks.to_csv(out/'scoring_masks_bool.csv',index=False,encoding='utf-8-sig')
    support.to_csv(out/'derived_support_date_map.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(count_rows).to_csv(out/'scoring_mask_counts.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(examples).to_csv(out/'scoring_mask_examples.csv',index=False,encoding='utf-8-sig')
    excluded.to_csv(out/'protocol_excluded_rows.csv',index=False,encoding='utf-8-sig')
    after={str(p.resolve()):sha(p) for p in paths};assert before==after
    report={'status':'PASS_P1_V2_SCORING_MASKS_BUILT_WITHOUT_FILTERING_OR_MODEL_EXECUTION',
        'protocol_version':PROTOCOL_VERSION,'supersedes':'P1 V1 eligible train2713/val584/test584; archived under v1_superseded/',
        'design_scope':'New proposed strict-information-boundary design; not an original-thesis implementation error finding.',
        'p1_eligible_counts':EXPECTED_ELIGIBLE,'mapped_original_samples':3882,'eligible_original_samples':sum(EXPECTED_ELIGIBLE.values()),
        'excluded_original_rows':excluded.to_dict(orient='records'),'strict_information_boundary_checks':boundary_checks,
        'mask_table_rows':len(masks),'boolean_columns':bool_columns,
        'derived_offsets':{'anchor':'t=u-1','input_feature_rows':'t-10..t-1 (10 rows)',
            'feature_vol20_at_f':'20 returns f-19..f, needing prices f-20..f',
            'combined_input_raw_prices':'t-30..t-1 (30 per asset;120 across all four model assets)',
            'return_target_raw_prices':'t,t+1 (2)',
            'vol20_target_returns':'t-18..t+1 (20)',
            'vol20_target_raw_prices':'t-19..t+1 (21)'},
        'rules':{'primary':'All P1 eligible imputed-series targets; no raw-presence or event-quality exclusion.',
            'secondary_return':'P1 eligible AND both raw return price endpoints present.',
            'secondary_vol20':'P1 eligible AND all21 raw prices for the20 target returns present.',
            'optional_strict_global':'Secondary asset/task mask AND all120 raw price cells supporting the actual eight-feature model input present.',
            'optional_strict_asset':'Secondary asset/task mask AND all30 own-asset input prices present; diagnostic only, not a claim that the full model input is raw-complete.',
            'selection_policy':'Masks must not select a best model, change training loss rows, or silently replace primary scoring. Every comparison within a mask must share exact asset/task dates.'},
        'known_event_quality_context':{'asset':'BIST100','date':'2023-02-08','source_url':EVENT_URL,
            'fact':'All8Feb trades cancelled; suspension started11:00 and continued through14Feb evening.',
            'interpretation':'A raw-present equal value may be carried/corrected. No assertion that the numerical carried value is wrong; raw presence is not a new-trade or timestamp certificate.',
            'mask_application':'Separate support-level context flags only; no primary/secondary/strict mask automatically excludes this date.',
            'other_raw_present_dates_event_quality_certified':False},
        'independent_validation':{'presence_csv_stdlib_vs_pandas_equal':True,
            'direct_slice_vs_independent_prefix_mask_checks':independent_checks,
            'support_sets_derived_by_return_endpoint_enumeration':True,
            'all_proposed_offsets_match_actual_date_map':True,'feb8_raw_mask_true_but_event_flag_true':True,
            'feb9_primary_true_secondary_false':True,'feb15_target_raw_present_but_return_endpoint_mask_false':True},
        'counts':count_rows,'example_records':examples,'input_sha256_before':before,'input_sha256_after':after,
        'sources_unchanged':True,'rows_deleted':0,'target_arrays_created_or_changed':False,
        'clock_availability_verified':False,'clock_leak_count':None,
        'training_scaler_fit_inference_model_garch_or_thesis_edit':False}
    (out/'scoring_masks_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'p1_eligible_counts':EXPECTED_ELIGIBLE,
        'bool_table_rows':len(masks),'independent_checks':independent_checks,'sources_unchanged':True,
        'test_counts':[r for r in count_rows if r['split']=='test']}))

if __name__=='__main__':main()
