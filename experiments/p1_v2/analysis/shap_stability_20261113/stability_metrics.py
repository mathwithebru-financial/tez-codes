"""Pure numerical SHAP diagnostics. No Torch, SHAP, weights, or file access.

Canonical phi axes: [explained sample, lag, input feature, output]. Eight
outputs remain separate. Original global importance is mean(abs(phi), N, lag).
"""
import math
import numpy as np

def _finite(value, name):
    a=np.asarray(value,dtype=np.float64)
    if not np.isfinite(a).all():raise ValueError(name+' must be finite')
    return a

def _phi(value):
    a=_finite(value,'phi')
    if a.ndim!=4 or min(a.shape[:2])<1 or a.shape[2:]!=(8,8):
        raise ValueError('phi must be [N>=1,L>=1,8 features,8 outputs]')
    return a

def _output_vectors(y_mean,y_scale):
    mean=_finite(y_mean,'y_mean');scale=_finite(y_scale,'y_scale')
    if mean.shape!=(8,) or scale.shape!=(8,) or np.any(scale<=0):
        raise ValueError('Eight means and strictly positive output scales required')
    return mean,scale

def feature_importance(phi_scaled):
    """Return [output, feature], original mean(abs(phi), sample AND lag)."""
    return np.mean(np.abs(_phi(phi_scaled)),axis=(0,1),dtype=np.float64).T

def temporal_importance(phi_scaled):
    """Return [output, lag], mean(abs(phi), sample AND feature)."""
    return np.mean(np.abs(_phi(phi_scaled)),axis=(0,2),dtype=np.float64).T

def feature_lag_importance(phi_scaled):
    """Return [output, lag, feature], mean(abs(phi), sample)."""
    return np.mean(np.abs(_phi(phi_scaled)),axis=0,dtype=np.float64).transpose(2,0,1)

def standardized_task_importance(feature_importance_scaled):
    """[2,8]: average four standardized outputs, THEN optionally normalize."""
    a=_finite(feature_importance_scaled,'standardized feature importance')
    if a.shape!=(8,8) or np.any(a<0):raise ValueError('Importance must be nonnegative [8 outputs,8 features]')
    return np.stack([np.mean(a[:4],axis=0,dtype=np.float64),np.mean(a[4:],axis=0,dtype=np.float64)])

def ensemble_attributions(phi_by_seed):
    """Mean phi FIRST. Call only with three models sharing background/path draws."""
    values=_finite(phi_by_seed,'three-seed phi')
    if values.ndim!=5 or values.shape[0]!=3:raise ValueError('Three [N,L,8,8] arrays required')
    for a in values:_phi(a)
    return np.mean(values,axis=0,dtype=np.float64)

def affine_output_transform(phi_scaled,expected_scaled,prediction_scaled,y_mean,y_scale):
    """scale phi; scale+intercept expected/prediction. Never add intercept to phi."""
    phi=_phi(phi_scaled);expected=_finite(expected_scaled,'expected output');prediction=_finite(prediction_scaled,'prediction')
    mean,scale=_output_vectors(y_mean,y_scale)
    if expected.shape!=(8,) or prediction.shape!=(phi.shape[0],8):raise ValueError('Expected [8] and prediction [N,8] required')
    return (phi*scale.reshape(1,1,1,8),expected*scale+mean,prediction*scale+mean)

def variance_diagnostic_raw(variance_scaled,y_scale):
    """Affine scale squared only; SHAP implementation statistic is NOT an SE/CI."""
    a=_phi(variance_scaled);_,scale=_output_vectors(np.zeros(8),y_scale)
    if np.any(a<0):raise ValueError('Variance implementation diagnostic must be nonnegative')
    return a*scale.reshape(1,1,1,8)**2

def additivity_diagnostics(phi_scaled,prediction_scaled,expected_scaled,y_mean,y_scale,output_labels=None):
    """One row/output; residual = prediction - expected - sum_lag,feature(phi)."""
    phi=_phi(phi_scaled);prediction=_finite(prediction_scaled,'prediction');expected=_finite(expected_scaled,'expected')
    phi_raw,expected_raw,pred_raw=affine_output_transform(phi,expected,prediction,y_mean,y_scale)
    _,scale=_output_vectors(y_mean,y_scale)
    labels=list(output_labels) if output_labels is not None else [str(i) for i in range(8)]
    if len(labels)!=8 or len(set(labels))!=8:raise ValueError('Eight unique output labels required')
    resid_scaled=prediction-expected-np.sum(phi,axis=(1,2),dtype=np.float64)
    resid_raw=pred_raw-expected_raw-np.sum(phi_raw,axis=(1,2),dtype=np.float64)
    resid_normalized=resid_raw/scale
    result=[]
    for j,label in enumerate(labels):
        row={'output_index':j,'output':label,'n':phi.shape[0],
             'expected_scaled':float(expected[j]),'expected_raw':float(expected_raw[j]),
             'raw_over_scale_vs_scaled_residual_max_difference':float(np.max(np.abs(resid_normalized[:,j]-resid_scaled[:,j])))}
        for name,values in [('scaled',resid_scaled[:,j]),('raw',resid_raw[:,j]),('raw_over_y_scale',resid_normalized[:,j])]:
            absolute=np.abs(values)
            row.update({f'{name}_residual_signed_mean':float(np.mean(values,dtype=np.float64)),
                        f'{name}_residual_signed_median':float(np.quantile(values,.5,method='linear')),
                        f'{name}_residual_mean_abs':float(np.mean(absolute,dtype=np.float64)),
                        f'{name}_residual_median_abs':float(np.quantile(absolute,.5,method='linear')),
                        f'{name}_residual_p95_abs':float(np.quantile(absolute,.95,method='linear')),
                        f'{name}_residual_rmse':float(np.sqrt(np.mean(values*values,dtype=np.float64))),
                        f'{name}_residual_max_abs':float(np.max(absolute))})
        result.append(row)
    return result

def _top3_info(values,labels):
    # Exact ties resolve by the supplied canonical FEATURE_ORDER, never random.
    order=sorted(range(8),key=lambda j:(-float(values[j]),j));total=float(np.sum(values,dtype=np.float64))
    exact_boundary_tie=bool(values[order[2]]==values[order[3]])
    gap=float(values[order[2]]-values[order[3]])
    unique,counts=np.unique(values,return_counts=True)
    return {'top3':[labels[j] for j in order[:3]],'rank_order':[labels[j] for j in order],
            'total_importance':total,'exact_tie_at_top3_boundary':exact_boundary_tie,
            'top3_without_boundary_tie_break_identifiable':not exact_boundary_tie and total>0,
            'third_fourth_gap':gap,'third_fourth_gap_share':gap/total if total>0 else None,
            'all_features_exactly_tied':len(unique)==1,
            'positive_importance_features':int(np.count_nonzero(values>0)),
            'top3_includes_zero_importance':bool(np.any(values[order[:3]]==0)),
            'tied_feature_pairs':int(sum(int(c)*(int(c)-1)//2 for c in counts)),
            'top3_status':'DEFINED_DETERMINISTIC_FEATURE_ORDER_TIES' if total>0 else 'UNINFORMATIVE_ALL_ZERO_IMPORTANCE'}

def compare_importance(left,right,feature_labels):
    """Eight-feature coefficient/share diagnostics. No significance p-values.

    normalized_L1=sum|a-b|/(sum a+sum b) preserves magnitude sensitivity.
    share_L1=sum|a/sum(a)-b/sum(b)|; share_total_variation=share_L1/2.
    All-zero share/rank cases are explicit undefined; no epsilon substitution.
    """
    a=_finite(left,'left importance');b=_finite(right,'right importance');labels=list(feature_labels)
    if a.shape!=(8,) or b.shape!=(8,) or np.any(a<0) or np.any(b<0) or len(labels)!=8 or len(set(labels))!=8:
        raise ValueError('Two nonnegative vectors and eight unique canonical feature labels required')
    sa=float(np.sum(a,dtype=np.float64));sb=float(np.sum(b,dtype=np.float64))
    if not math.isfinite(sa+sb):raise ValueError('Importance sums overflow')
    topa=_top3_info(a,labels);topb=_top3_info(b,labels)
    ranks_defined=bool(np.ptp(a)>0 and np.ptp(b)>0)
    if ranks_defined:
        from scipy.stats import rankdata,kendalltau
        ra=rankdata(a,method='average');rb=rankdata(b,method='average')
        ra=ra-np.mean(ra);rb=rb-np.mean(rb)
        spearman=float(np.dot(ra,rb)/math.sqrt(float(np.dot(ra,ra))*float(np.dot(rb,rb))))
        tau=float(kendalltau(a,b,variant='b',method='asymptotic').statistic)
        if not math.isfinite(spearman) or not math.isfinite(tau):raise ValueError('Nonfinite rank coefficient')
        spearman=min(1.,max(-1.,spearman));tau=min(1.,max(-1.,tau))
    else:spearman=tau=None
    shares_defined=sa>0 and sb>0
    share_l1=float(np.sum(np.abs(a/sa-b/sb),dtype=np.float64)) if shares_defined else None
    top3_jaccard=len(set(topa['top3'])&set(topb['top3']))/len(set(topa['top3'])|set(topb['top3'])) if shares_defined else None
    return {'spearman_midrank':spearman,'kendall_tau_b':tau,
            'rank_status':'DEFINED' if ranks_defined else 'UNDEFINED_CONSTANT_IMPORTANCE_VECTOR',
            'normalized_L1':float(np.sum(np.abs(a-b),dtype=np.float64))/(sa+sb) if sa+sb>0 else None,
            'normalized_L1_formula':'sum_abs_difference/(sum_left+sum_right)',
            'share_L1':share_l1,'share_total_variation':share_l1/2 if share_l1 is not None else None,
            'share_status':'DEFINED' if shares_defined else 'UNDEFINED_ZERO_TOTAL_IMPORTANCE',
            'top3_jaccard':top3_jaccard,'top3_tie_break':'Canonical FEATURE_ORDER supplied to function',
            'top3_jaccard_depends_on_boundary_tie_break':topa['exact_tie_at_top3_boundary'] or topb['exact_tie_at_top3_boundary'],
            'left':topa,'right':topb,
            'left_shares':(a/sa).tolist() if sa>0 else None,'right_shares':(b/sb).tolist() if sb>0 else None,
            'near_tie_binary_threshold_used':False,'p_values_or_significance_claims':False}
