"""Pure information-cutoff adapter. No arch import/fit or file mutation.

Only a separately supplied, explicitly valid fit state may reach the numerical
CDF bridge. This checks declarations against exact input identities; it cannot
prove that an external fitter truthfully used those declared inputs.
"""
from dataclasses import dataclass
from pathlib import Path
import csv,io,json,hashlib,datetime
import numpy as np
from two_increment_vol20_bridge import Garch11Parameters,median_cdf_or_missing

CANONICAL_RAW_SHA256='ab5f275d38dc98057b1cedcf58019adb26be7402c7ed5ae6ee3d6877b2444893'
ASSETS=('BIST100','USDTRY','EURTRY','GOLD')
HISTORY_START_DATE='2010-02-01' # Original08C expanding history begins at first v4feature row.

class InformationCutoffError(ValueError):pass

def digest_bytes(data):return hashlib.sha256(data).hexdigest()
def digest_object(data):return digest_bytes(json.dumps(data,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8'))
def vector_sha(a):return digest_bytes(np.asarray(a,dtype='<f8').tobytes(order='C'))

@dataclass(frozen=True)
class PreparedCase:
    asset:str
    source_raw_sha256:str
    map_row_sha256:str
    cutoff_date:str
    cutoff_raw_index:int
    target_date:str
    target_raw_index:int
    history_dates:tuple
    history_decimal:np.ndarray
    known18_dates:tuple
    known18_decimal:np.ndarray
    history_sha256:str
    known18_sha256:str
    context_sha256:str

    def summary(self):
        return {'asset':self.asset,'source_raw_sha256':self.source_raw_sha256,'map_row_sha256':self.map_row_sha256,
          'cutoff_date':self.cutoff_date,'cutoff_raw_index':self.cutoff_raw_index,'target_date':self.target_date,
          'target_raw_index':self.target_raw_index,'horizon_common_positions':self.target_raw_index-self.cutoff_raw_index,
          'history_first_date':self.history_dates[0],'history_last_date':self.history_dates[-1],
          'history_nobs':len(self.history_decimal),'history_sha256':self.history_sha256,
          'known18_first_date':self.known18_dates[0],'known18_last_date':self.known18_dates[-1],
          'known18_nobs':len(self.known18_decimal),'known18_sha256':self.known18_sha256,
          'context_sha256':self.context_sha256,'clock_availability_verified':False,
          'future_numeric_prices_parsed':False}

def context_digest(asset,raw_sha,map_sha,kdate,k,jdate,j,hdates,hsha,kdates,ksha):
    return digest_object({'asset':asset,'raw_sha':raw_sha,'map_sha':map_sha,'cutoff_date':kdate,'cutoff_raw_index':k,
      'target_date':jdate,'target_raw_index':j,'history_dates':list(hdates),'history_sha':hsha,
      'known18_dates':list(kdates),'known18_sha':ksha})

def prepare_case(raw_csv,map_row,asset,*,expected_raw_sha256=CANONICAL_RAW_SHA256,history_start_date=HISTORY_START_DATE):
    """Parse numeric prices ONLY through k; later rows supply date labels only.

    The full raw bytes are hashed for source identity. Bytes or a CSV path are
    accepted to permit pure in-memory identity-rejection tests. No source edits.
    History follows original08C's first-feature-date convention, not a newly
    lengthened history from the first raw price. known18 comes from raw prefix.
    """
    payload=raw_csv if isinstance(raw_csv,bytes) else Path(raw_csv).read_bytes()
    raw_sha=digest_bytes(payload)
    if raw_sha!=expected_raw_sha256:raise InformationCutoffError('Canonical raw source SHA256 mismatch')
    if asset not in ASSETS:raise InformationCutoffError('Unsupported asset')
    if str(map_row.get('proposed_eligible')).lower()!='true':raise InformationCutoffError('P1 row is ineligible')
    if 'training_or_selection_eligible' in map_row and str(map_row['training_or_selection_eligible']).lower()!='true':
        raise InformationCutoffError('Strict primary training/selection population excludes this row')
    if 'feature_window_eligible' in map_row and str(map_row['feature_window_eligible']).lower()!='true':
        raise InformationCutoffError('Feature-window population excludes this row')
    if int(map_row.get('proposed_horizon_common_rows_from_last_input_to_target',-1))!=2:
        raise InformationCutoffError('P1 horizon must equal2 common observation positions')
    kdate=str(map_row['proposed_lag1_input_last']);jdate=str(map_row['target_date'])
    datetime.date.fromisoformat(kdate);datetime.date.fromisoformat(jdate)
    reader=csv.DictReader(io.StringIO(payload.decode('utf-8-sig')))
    if reader.fieldnames!=['Date',*ASSETS]:raise InformationCutoffError('Raw CSV schema mismatch')
    dates=[];prefix_prices=[];last=None
    for row in reader:
        date=row['Date'];datetime.date.fromisoformat(date)
        if dates and date<=dates[-1]:raise InformationCutoffError('Calendar must be strictly ordered and unique')
        dates.append(date)
        if date<=kdate:
            token=row[asset].strip()
            if token:
                value=float(token)
                if not np.isfinite(value) or value<=0:raise InformationCutoffError('Invalid admitted price')
                last=value
            if last is None:raise InformationCutoffError('Leading missing price cannot be backfilled')
            prefix_prices.append(last)
        # No float()/price read is performed for later calendar rows.
    if kdate not in dates or jdate not in dates:raise InformationCutoffError('Mapped date absent from raw calendar')
    k=dates.index(kdate);j=dates.index(jdate)
    if j!=k+2 or len(prefix_prices)!=k+1:raise InformationCutoffError('Mapped cutoff/target indices disagree')
    if k-17<1:raise InformationCutoffError('Insufficient past returns for known18')
    if map_row.get('return_target_start_label')!=dates[j-1] or map_row.get('return_target_end_label')!=jdate:
        raise InformationCutoffError('Single-increment return target labels disagree')
    if map_row.get('vol20_first_return_label')!=dates[j-19] or map_row.get('vol20_last_return_label')!=jdate:
        raise InformationCutoffError('Vol20 target window labels disagree')
    if map_row.get('proposed_lag1_input_first')!=dates[k-9]:raise InformationCutoffError('L10 input window labels disagree')
    prices=np.asarray(prefix_prices,dtype=np.float64)
    returns=np.log(prices[1:]/prices[:-1])
    start=dates.index(history_start_date)
    if start<1 or start>k:raise InformationCutoffError('Invalid original08C history start')
    history=returns[start-1:k].copy();hdates=tuple(dates[start:k+1])
    known=returns[k-18:k].copy();kdates=tuple(dates[k-17:k+1])
    if len(known)!=18 or not np.isfinite(history).all() or not np.isfinite(known).all():
        raise InformationCutoffError('Invalid admitted return vectors')
    if hdates[-1]!=kdate or kdates[-1]!=kdate or kdates[0]!=dates[j-19]:
        raise InformationCutoffError('Cutoff or known18 endpoints disagree')
    hs=vector_sha(history);ks=vector_sha(known);ms=digest_object(dict(map_row))
    cs=context_digest(asset,raw_sha,ms,kdate,k,jdate,j,hdates,hs,kdates,ks)
    history.setflags(write=False);known.setflags(write=False)
    return PreparedCase(asset,raw_sha,ms,kdate,k,jdate,j,hdates,history,kdates,known,hs,ks,cs)

def validate_supplied_fit_then_bridge(context,fit_state,*,primary_callback=median_cdf_or_missing):
    """Validate information declarations before a caller-supplied primary callback.

    No fitting occurs. fit_state must bind asset/model, admitted-date/index,
    nobs and hashes; a forecast generated at another cutoff is rejected.
    """
    try:
        if not isinstance(context,PreparedCase):raise InformationCutoffError('PreparedCase required')
        if not isinstance(fit_state,dict):raise InformationCutoffError('Explicit fit-state dictionary required')
        hsha=vector_sha(context.history_decimal);ksha=vector_sha(context.known18_decimal)
        cs=context_digest(context.asset,context.source_raw_sha256,context.map_row_sha256,context.cutoff_date,
           context.cutoff_raw_index,context.target_date,context.target_raw_index,context.history_dates,hsha,
           context.known18_dates,ksha)
        if hsha!=context.history_sha256 or ksha!=context.known18_sha256 or cs!=context.context_sha256:
            raise InformationCutoffError('Prepared context changed after its identity was recorded')
        if context.target_raw_index!=context.cutoff_raw_index+2 or len(context.known18_decimal)!=18:
            raise InformationCutoffError('Prepared horizon/known-count mismatch')
        if max(context.history_dates)!=context.cutoff_date or max(context.known18_dates)!=context.cutoff_date:
            raise InformationCutoffError('History contains a date beyond admitted cutoff')
        required={'asset':context.asset,'source_raw_sha256':context.source_raw_sha256,
          'context_sha256':context.context_sha256,'history_sha256':hsha,
          'date_last_fit':context.cutoff_date,'last_fit_raw_index':context.cutoff_raw_index,
          'history_start_date':context.history_dates[0],'nobs':len(context.history_decimal),
          'h1_information_date':context.cutoff_date,'h1_forecast_horizon':1}
        for name,expected in required.items():
            if fit_state.get(name)!=expected:raise InformationCutoffError('Supplied fit mismatch: '+name)
        if fit_state.get('status')!='OK' or fit_state.get('convergence_ok') is not True or fit_state.get('feasibility_ok') is not True:
            return {'status':'FIT_UNRESOLVED','point':None,'fallback_used':False,'callback_called':False}
        params=fit_state.get('parameters')
        if isinstance(params,dict):params=Garch11Parameters(**params)
        if not isinstance(params,Garch11Parameters):raise InformationCutoffError('Explicit scalar parameters required')
        params.validate()
        if fit_state.get('model')!=params.model:raise InformationCutoffError('Supplied model identifier mismatch')
        h1=float(fit_state['h1_percent2'])
        if not np.isfinite(h1) or h1<=0:raise InformationCutoffError('Invalid h1 variance')
    except (InformationCutoffError,ValueError,TypeError,KeyError) as exc:
        return {'status':'INFORMATION_OR_FIT_STATE_REJECTED','point':None,'reason':str(exc),'fallback_used':False,'callback_called':False}
    try:
        result=primary_callback(fit_ok=True,known18_decimal=context.known18_decimal,params=params,h1_percent2=h1)
        if not isinstance(result,dict):raise InformationCutoffError('Primary callback returned no structured status')
        result=dict(result);result['callback_called']=True;result['context_sha256']=context.context_sha256
        if result.get('fallback_used') is not False:raise InformationCutoffError('Primary callback permits fallback')
        if result.get('point') is not None and (result.get('status')!='NUMERICALLY_CHECKED_CDF_MEDIAN' or not np.isfinite(result['point']) or result['point']<0):
            raise InformationCutoffError('Primary callback supplied an invalid or unguarded point')
        return result
    except (InformationCutoffError,ValueError,TypeError,KeyError,RuntimeError,FloatingPointError,OverflowError) as exc:
        return {'status':'PRIMARY_CALLBACK_FAILED','point':None,'reason':str(exc),'fallback_used':False,'callback_called':True}
