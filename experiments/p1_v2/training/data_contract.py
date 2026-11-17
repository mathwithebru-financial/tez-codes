"""Read NEW P1V2 train/primary-val arrays only. No torch/model/test-array I/O."""
from pathlib import Path
import hashlib,json,datetime,importlib.util,sys
import numpy as np
try:
    from .config import FEATURE_ORDER,OUTPUT_ORDER,PROTOCOL_VERSION
except ImportError:
    from config import FEATURE_ORDER,OUTPUT_ORDER,PROTOCOL_VERSION

BASE=Path(__file__).resolve().parents[1]
DEFAULT_DATASET=BASE/'datasets/p1_v2'
DEFAULT_DENOMINATORS=BASE/'benchmarks/validation/selection_denominators_P1_validation_v2.json'
SCORE_SOURCE=BASE/'benchmarks/validation/validation_score_contract.py'
EXPECTED_COUNTS={'train':2712,'val':583,'test':584}
FIT_FILE_NAMES=tuple(name for split in ('train','val') for name in (
    'X_'+split+'.npy','y_'+split+'_scaled.npy','y_'+split+'_raw.npy',
    'target_dates_'+split+'.npy','input_end_dates_'+split+'.npy'))

class DataContractError(ValueError):pass
def require(condition,message):
    if not condition:raise DataContractError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_score_function():
    # This is a newly authored pure numeric scorer, never an archived script.
    spec=importlib.util.spec_from_file_location('p1_v2_fixed_validation_score_contract',SCORE_SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.primary_validation_score

def inverse_y_float64(scaled,scalers):
    return np.asarray(scaled,dtype=np.float64)*np.asarray(scalers['y']['scale'],dtype=np.float64)+np.asarray(scalers['y']['mean'],dtype=np.float64)

def load_fit_data(dataset_dir=DEFAULT_DATASET,denominator_path=DEFAULT_DENOMINATORS):
    require(sys.flags.optimize==0,'Do not run Python with-O: canonical scorer assertions must remain active')
    folder=Path(dataset_dir).resolve();denom=Path(denominator_path).resolve()
    metadata_path=folder/'dataset_metadata.json';scalers_path=folder/'scalers.json'
    metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
    require(metadata['protocol_version']==PROTOCOL_VERSION,'Dataset protocol version mismatch')
    require(metadata['feature_order']==list(FEATURE_ORDER),'Feature order mismatch')
    require(metadata['output_order']==list(OUTPUT_ORDER),'Output order mismatch')
    require(metadata['split_counts']==EXPECTED_COUNTS,'Strict split counts mismatch')
    source_paths={'raw':BASE/'inputs/raw_prices.csv','map':BASE/'protocol/proposed_lag1_date_map.csv',
                  'locked_denominators':denom}
    for key,path in source_paths.items():
        require(metadata['source_sha256'][key]==sha(path),'Dataset source identity mismatch: '+key)
    require(metadata['scalers_file']=='scalers.json','Unexpected scaler JSON name')
    require(sha(scalers_path)==metadata['scalers_sha256'],'Scaler JSON hash mismatch')
    scalers=json.loads(scalers_path.read_text(encoding='utf-8'))
    require(scalers['feature_order']==list(FEATURE_ORDER) and scalers['output_order']==list(OUTPUT_ORDER),'Scaler orders mismatch')
    for kind,order,key in [('x',FEATURE_ORDER,'feature_order'),('y',OUTPUT_ORDER,'target_order')]:
        obj=scalers[kind]
        require(obj['n_samples_seen']==2721,'Scaler fit must use exactly2721 P1V2 pre-window training pairs')
        require(obj[key]==list(order),'Per-scaler order mismatch')
        for stat in ('mean','var','scale'):
            values=np.asarray(obj[stat],dtype=np.float64)
            require(values.shape==(8,) and np.isfinite(values).all(),'Invalid scaler '+kind+' '+stat)
        require(np.all(np.asarray(obj['var'])>=0) and np.all(np.asarray(obj['scale'])>0),'Invalid scaler variance/scale')
    arr={};files_read=[str(metadata_path),str(scalers_path),str(denom),str(SCORE_SOURCE)]+[str(p) for p in source_paths.values() if p!=denom]
    for split in ('train','val'):
        n=EXPECTED_COUNTS[split]
        entries={
          'X_'+split:((n,10,8),np.dtype('float32')),
          'y_'+split+'_scaled':((n,8),np.dtype('float32')),
          'y_'+split+'_raw':((n,8),np.dtype('float64')),
          'target_dates_'+split:((n,),np.dtype('<U10')),
          'input_end_dates_'+split:((n,),np.dtype('<U10'))}
        for stem,(shape,dtype) in entries.items():
            filename=stem+'.npy';path=folder/filename;record=metadata['arrays'][filename]
            require(sha(path)==record['sha256'] and path.stat().st_size==record['size_bytes'],'Dataset array identity mismatch: '+filename)
            value=np.load(path,allow_pickle=False)
            require(value.shape==shape and value.dtype==dtype,'Shape/dtype mismatch: '+filename)
            require(list(value.shape)==record['shape'] and str(value.dtype)==record['dtype'],'Metadata shape/dtype mismatch: '+filename)
            if value.dtype.kind=='f':require(np.isfinite(value).all(),'Nonfinite array: '+filename)
            else:
                dates=value.tolist()
                for date in dates:datetime.date.fromisoformat(date)
                require(dates==sorted(set(dates)),'Date order/uniqueness mismatch: '+filename)
            arr[stem]=value;files_read.append(str(path))
        require(np.all(arr['input_end_dates_'+split]<arr['target_dates_'+split]),'Input is not earlier than target')
    require(arr['target_dates_train'][-1]=='2020-07-08','Training-label boundary mismatch')
    require(arr['input_end_dates_val'][0]=='2020-07-08','First validation information cutoff mismatch')
    require(arr['target_dates_val'][-1]=='2022-10-04','Validation selection-label boundary mismatch')
    require(arr['target_dates_train'][-1]<=arr['input_end_dates_val'][0],'Training labels exceed first validation cutoff')
    scorer=load_score_function()
    identity_score=scorer(arr['y_val_raw'],arr['y_val_raw'],arr['target_dates_val'].tolist(),
                          output_order=list(OUTPUT_ORDER),denominator_path=denom)
    require(identity_score['validation_score']==0.0,'Canonical validation truth/denominator identity check failed')
    # Train/validation truth precision and scaler quantization are reported,
    # never used as a replacement for the canonical float64 raw truth.
    inverse_errors={split:float(np.max(np.abs(inverse_y_float64(arr['y_'+split+'_scaled'],scalers)-arr['y_'+split+'_raw']))) for split in ('train','val')}
    identity={'dataset_metadata_path':str(metadata_path),'dataset_metadata_sha256':sha(metadata_path),
      'scalers_sha256':sha(scalers_path),'denominator_path':str(denom),'denominator_sha256':sha(denom),
      'validation_score_source_path':str(SCORE_SOURCE),'validation_score_source_sha256':sha(SCORE_SOURCE),
      'protocol_version':PROTOCOL_VERSION,'files_read':files_read,'test_arrays_read':False,
      'fit_counts':{k:EXPECTED_COUNTS[k] for k in ('train','val')},
      'canonical_identity_score':identity_score['validation_score'],
      'float32_scaled_truth_inverse_error_diagnostic':inverse_errors,
      'inverse_prediction_rule':'castscaledpredictiontofloat64 before multiplyingfloat64y_scale andaddingfloat64y_mean'}
    return {'arrays':arr,'scalers':scalers,'metadata':metadata,'identity':identity,'scorer':scorer}
