"""Independent list/fsum implementation for saved-array subperiod QA only."""
import math

METRIC_FIELDS=('MAE','MSE','RMSE','R2','Pinball_tau05','median_absolute_error',
 'p90_absolute_error','median_primary_loss','p90_primary_loss')
def quantile_linear(values,p):
    values=sorted(values)
    if not values:return None
    point=(len(values)-1)*p;lo=math.floor(point);hi=math.ceil(point)
    return values[lo]+(point-lo)*(values[hi]-values[lo])
def metric_reference(truth,prediction,task):
    truth=[float(v) for v in truth];prediction=[float(v) for v in prediction]
    if len(truth)!=len(prediction):raise ValueError('Unmatched populations')
    n=len(truth)
    if not n:return {'n':0,**{k:None for k in METRIC_FIELDS},'SSE':0.,'SST':0.,'truth_mean':None,'primary_loss':None}
    error=[y-p for y,p in zip(truth,prediction)];absolute=[abs(v) for v in error]
    squared=[v*v for v in error];sse=math.fsum(squared);mse=sse/n
    mu=math.fsum(truth)/n;sst=math.fsum((v-mu)**2 for v in truth)
    primary=[.5*v for v in absolute] if task=='volatility' else absolute
    return {'n':n,'MAE':math.fsum(absolute)/n,'MSE':mse,'RMSE':math.sqrt(mse),
      'R2':1-sse/sst if n>=2 and sst>0 else None,
      'Pinball_tau05':math.fsum(primary)/n if task=='volatility' else None,
      'median_absolute_error':quantile_linear(absolute,.5),'p90_absolute_error':quantile_linear(absolute,.9),
      'median_primary_loss':quantile_linear(primary,.5),'p90_primary_loss':quantile_linear(primary,.9),
      'SSE':sse,'SST':sst,'truth_mean':mu,'primary_loss':math.fsum(primary)/n}
