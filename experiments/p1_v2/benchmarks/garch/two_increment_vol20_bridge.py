"""P1 numerical bridge only: no arch import, fit, market I/O, or model artifacts.

Given 18 already-admitted DECIMAL returns, fitted/manual scalar parameters and
the first forecast variance in PERCENT-return squared units, construct the
conditional distribution of the ORIGINAL 20-return sample volatility target.
The primary adapter is a guarded deterministic conditional-CDF/root median.
CRN Monte Carlo quantiles remain an independent approximation diagnostic.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from scipy.special import stdtr, stdtrit, gammaln
from scipy.integrate import quad, IntegrationWarning
from scipy.optimize import brentq
import warnings

ANNUALIZATION=252.0
NEGATIVE_VARIANCE_TOL=1e-14

class BridgeInputError(ValueError):
    pass

@dataclass(frozen=True)
class Garch11Parameters:
    model: str
    omega_percent2: float
    alpha: float
    beta: float
    gamma: float
    nu: float

    def validate(self, *, synthetic_degenerate=False):
        vals=[self.omega_percent2,self.alpha,self.beta,self.gamma,self.nu]
        if self.model not in {'GARCH_1_1','GJR_GARCH_1_1'} or not all(math.isfinite(float(x)) for x in vals):
            raise BridgeInputError('Unsupported model or nonfinite scalar parameter')
        if self.model=='GARCH_1_1' and self.gamma!=0.0:
            raise BridgeInputError('GARCH o=0 requires gamma=0')
        if self.omega_percent2<0 or self.alpha<0 or self.beta<0 or self.alpha+self.gamma<0:
            raise BridgeInputError('Variance positivity constraints fail')
        # Actual local arch8.0.0 static fit bounds/linear constraints. Omega
        # bounds also depend on the fitted history and CANNOT be checked here.
        if not (0<=self.alpha<=1 and 0<=self.beta<=1 and -1<=self.gamma<=2):
            raise BridgeInputError('Static arch volatility bounds fail')
        if self.alpha+0.5*self.gamma+self.beta>1.0:
            raise BridgeInputError('arch symmetric-t persistence constraint fails')
        if not (2.05<=self.nu<=500.0):
            raise BridgeInputError('Local arch StudentsT fit bounds require2.05<=nu<=500')
        if not synthetic_degenerate and self.omega_percent2<=0:
            raise BridgeInputError('Zero omega is allowed only in labelled synthetic boundary checks')

    def arch_parameter_order(self):
        if self.model=='GARCH_1_1':
            return {'names':['omega','alpha[1]','beta[1]','nu'],
                    'values':[self.omega_percent2,self.alpha,self.beta,self.nu]}
        return {'names':['omega','alpha[1]','gamma[1]','beta[1]','nu'],
                'values':[self.omega_percent2,self.alpha,self.gamma,self.beta,self.nu]}

def known_summary(known18_decimal):
    a=np.asarray(known18_decimal,dtype=np.float64)
    if a.shape!=(18,) or not np.isfinite(a).all():
        raise BridgeInputError('Exactly18 finite admitted DECIMAL returns are required')
    return a,float(np.sum(a,dtype=np.float64)),float(np.sum(a*a,dtype=np.float64))

def _nonnegative(a,name):
    a=np.asarray(a,dtype=np.float64)
    if not np.isfinite(a).all() or np.any(a < -NEGATIVE_VARIANCE_TOL):
        raise BridgeInputError(name+' is nonfinite or materially negative')
    return np.maximum(a,0.0)

def standardized_t_from_uniform(uniforms,nu):
    u=np.asarray(uniforms,dtype=np.float64)
    if u.ndim!=2 or u.shape[1]!=2 or not np.isfinite(u).all() or np.any((u<=0)|(u>=1)):
        raise BridgeInputError('CRN uniforms must have shape(N,2) strictly inside(0,1)')
    if not math.isfinite(nu) or nu<=2:
        raise BridgeInputError('Unit-variance Student-t requires nu>2')
    # t_nu has variance nu/(nu-2); this scaling matches arch StudentsT.
    z=stdtrit(nu,u)*math.sqrt((nu-2.0)/nu)
    if not np.isfinite(z).all():raise BridgeInputError('Student-t transform is nonfinite')
    return z

def simulate_two_increments(params,h1_percent2,z,*,synthetic_degenerate=False):
    params.validate(synthetic_degenerate=synthetic_degenerate)
    h1=float(h1_percent2)
    if not math.isfinite(h1) or h1<0 or (h1==0 and not synthetic_degenerate):
        raise BridgeInputError('First variance must be positive PERCENT-return squared')
    z=np.asarray(z,dtype=np.float64)
    if z.ndim!=2 or z.shape[1]!=2 or not np.isfinite(z).all():
        raise BridgeInputError('Standardized innovations must have shape(N,2)')
    r1_percent=math.sqrt(h1)*z[:,0]
    # Combine alpha+gamma for negative shocks: negative gamma can be valid.
    shock_coefficient=params.alpha+params.gamma*(r1_percent<0)
    h2_percent2=params.omega_percent2+shock_coefficient*(r1_percent*r1_percent)+params.beta*h1
    h2_percent2=_nonnegative(h2_percent2,'pathwise second-step variance')
    r2_percent=np.sqrt(h2_percent2)*z[:,1]
    increments_decimal=np.column_stack((r1_percent/100.0,r2_percent/100.0))
    if not np.isfinite(increments_decimal).all():raise BridgeInputError('Increment path is nonfinite')
    return increments_decimal,h2_percent2

def vol20_from_increments(known18_decimal,increments_decimal):
    a,S,_=known_summary(known18_decimal)
    r=np.asarray(increments_decimal,dtype=np.float64)
    if r.ndim!=2 or r.shape[1]!=2 or not np.isfinite(r).all():
        raise BridgeInputError('Two future DECIMAL returns per path are required')
    # Centered Welford combination is algebraically identical to
    # (Q18+R1²+R2²-(S18+R1+R2)²/20)/19, but avoids subtracting
    # nearly equal sums for constant/nearly constant return windows.
    c18=S/18.0
    sse18=float(np.sum((a-c18)**2,dtype=np.float64))
    c19=(S+r[:,0])/19.0
    sse20=sse18+(18.0/19.0)*(r[:,0]-c18)**2+(19.0/20.0)*(r[:,1]-c19)**2
    sample_variance=sse20/19.0
    sample_variance=_nonnegative(sample_variance,'Vol20 sample variance')
    return np.sqrt(ANNUALIZATION*sample_variance)

def common_uniforms(max_n,seed):
    # SeedSequence/PCG64 is explicit. These are simulation randomness seeds,
    # separate from neural training seeds123/777/2026.
    rng=np.random.Generator(np.random.PCG64(int(seed)))
    u=rng.random((int(max_n),2),dtype=np.float64)
    # A possible exact0 is replaced by the adjacent representable positive
    # number, avoiding an infinite inverse CDF;1 is never produced by PCG64.
    u=np.maximum(u,np.nextafter(0.0,1.0))
    return u

def median_crn(known18_decimal,params,h1_percent2,*,
               n_grid=(4096,16384,65536),seeds=(17011,27011,37011,47011),
               absolute_tolerance=2e-4,relative_tolerance=2e-3):
    """Return a point ONLY on observed numerical stability, never a guarantee.

    All grid sizes are prefixes of the same per-replicate uniforms; the same
    seeds can be reused across model specifications for common random numbers.
    The tolerance is in annualized decimal Vol20 units. A hard N budget does
    not certify a population median. If insufficiently stable, point=None.
    """
    known_summary(known18_decimal);params.validate()
    ns=tuple(int(n) for n in n_grid)
    if len(ns)<3 or ns!=tuple(sorted(set(ns))) or ns[0]<2 or len(set(seeds))<2 or len(set(seeds))!=len(seeds):
        raise BridgeInputError('Need>=3 increasing N levels and>=2 distinct replication seeds')
    if not (math.isfinite(absolute_tolerance) and math.isfinite(relative_tolerance)) or absolute_tolerance<=0 or relative_tolerance<0:
        raise BridgeInputError('Invalid numeric tolerances')
    all_vols=[]
    for seed in seeds:
        z=standardized_t_from_uniform(common_uniforms(ns[-1],seed),params.nu)
        r,_=simulate_two_increments(params,h1_percent2,z)
        all_vols.append(vol20_from_increments(known18_decimal,r))
    stages=[]
    for n in ns:
        per_rep=[float(np.quantile(v[:n],0.5,method='inverted_cdf')) for v in all_vols]
        pooled=np.concatenate([v[:n] for v in all_vols])
        point=float(np.quantile(pooled,0.5,method='inverted_cdf'))
        tol=max(float(absolute_tolerance),float(relative_tolerance)*abs(point))
        delta=abs(point-stages[-1]['pooled_median']) if stages else None
        stages.append({'n_per_replication':n,'n_total':len(pooled),'replicate_medians':per_rep,
          'pooled_median':point,'tolerance':tol,'change_from_previous':delta,
          'maximum_replication_deviation':max(abs(x-point) for x in per_rep)})
    last=stages[-1];tol=last['tolerance']
    stable=(all(s['change_from_previous']<=tol for s in stages[-2:])
            and last['maximum_replication_deviation']<=tol)
    return {'status':'NUMERICALLY_STABLE_IN_THIS_CHECK' if stable else 'NUMERICAL_UNRESOLVED',
      'point':last['pooled_median'] if stable else None,'last_quantile_diagnostic':last['pooled_median'],
      'stages':stages,'seeds':list(seeds),'quantile_method':'inverted_cdf(lower empirical median)',
      'absolute_tolerance':absolute_tolerance,'relative_tolerance':relative_tolerance,
      'meaning':'Observed prefix/replication stability only; no finite-N universal error guarantee.',
      'no_fit_or_market_prediction':True}

def fit_bridge_or_missing(*,fit_ok,known18_decimal=None,params=None,h1_percent2=None,**kwargs):
    """MC DIAGNOSTIC adapter; primary predictions use median_cdf_or_missing.

    Never converts failed fit/numerics to persistence.
    """
    if not fit_ok:
        return {'status':'FIT_UNRESOLVED','point':None,'fallback_used':False}
    try:
        result=median_crn(known18_decimal,params,h1_percent2,**kwargs)
        result['fallback_used']=False
        return result
    except (BridgeInputError,ValueError,FloatingPointError,OverflowError) as exc:
        return {'status':'INVALID_BRIDGE_INPUT_OR_NUMERICS','point':None,
                'reason':str(exc),'fallback_used':False}

def conditional_vol_cdf(vol,known18_decimal,params,h1_percent2,*,epsabs=1e-9,epsrel=1e-9):
    """Independent deterministic1D integral of the exact two-increment CDF.

    For synthetic validation, integrates out R2 conditionally on R1. This is
    not a fitted-model call, nor an empirical future realization.
    """
    a,S18,_=known_summary(known18_decimal);params.validate()
    h1=float(h1_percent2)
    if h1<=0 or not math.isfinite(h1):raise BridgeInputError('Nondegenerate h1 required for CDF integral')
    if not math.isfinite(vol):raise BridgeInputError('Finite Vol20 threshold required')
    if vol<0:return 0.0,0.0
    v=vol*vol/ANNUALIZATION
    sse18=float(np.sum((a-S18/18.0)**2,dtype=np.float64))
    c0=sse18/19.0
    if v<=c0:return 0.0,0.0
    center=S18/18.0
    radius=math.sqrt((v-c0)*19.0**2/18.0)
    low,high=center-radius,center+radius
    sigma1_dec=math.sqrt(h1)/100.0
    tscale=math.sqrt((params.nu-2.0)/params.nu)
    lognorm=gammaln((params.nu+1)/2)-gammaln(params.nu/2)-0.5*math.log(math.pi*(params.nu-2))
    def integrand(r1_dec):
        c19=(S18+r1_dec)/19.0
        C19=c0+18.0/(19.0**2)*(r1_dec-center)**2
        q2=math.sqrt(max(0.0,20.0*(v-C19)))
        r1p=100.0*r1_dec
        h2=params.omega_percent2+(params.alpha+(params.gamma if r1p<0 else 0.0))*r1p*r1p+params.beta*h1
        if h2<0:raise BridgeInputError('Invalid conditional second variance')
        if h2==0:
            conditional_mass=float(abs(c19)<=q2)
        else:
            sigma2_dec=math.sqrt(h2)/100.0
            conditional_mass=float(stdtr(params.nu,(c19+q2)/(sigma2_dec*tscale))-
                                   stdtr(params.nu,(c19-q2)/(sigma2_dec*tscale)))
        z=r1_dec/sigma1_dec
        f1=math.exp(lognorm-(params.nu+1)/2*math.log1p(z*z/(params.nu-2)))/sigma1_dec
        if conditional_mass < -1e-8 or conditional_mass > 1.0+1e-8:
            raise BridgeInputError('Conditional CDF probability materially outside[0,1]')
        return min(1.0,max(0.0,conditional_mass))*f1
    points=[0.0] if low<0<high else None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',IntegrationWarning)
        mass,error=quad(integrand,low,high,epsabs=epsabs,epsrel=epsrel,limit=300,points=points)
    if caught or not math.isfinite(mass) or error>2e-8 or mass < -1e-8 or mass > 1.0+1e-8:
        raise BridgeInputError('CDF integration unresolved: '+str([str(w.message) for w in caught]))
    return float(min(1.0,max(0.0,mass))),float(error)

def median_quadrature_reference(known18_decimal,params,h1_percent2,*,epsabs=1e-9,epsrel=1e-9,
                                root_xtol=1e-10,root_rtol=1e-10):
    """Deterministic population-CDF solver; numerical errors still apply."""
    a,_,_=known_summary(known18_decimal)
    upper=max(0.01,float(np.std(a,ddof=1)*math.sqrt(ANNUALIZATION))*2,
              math.sqrt(float(h1_percent2))/100*math.sqrt(ANNUALIZATION))
    doublings=0
    def cdf(v):return conditional_vol_cdf(v,a,params,h1_percent2,epsabs=epsabs,epsrel=epsrel)
    while cdf(upper)[0]<0.5:
        upper*=2;doublings+=1
        if doublings>50 or not math.isfinite(upper):raise BridgeInputError('Median bracket unresolved')
    root=brentq(lambda v:cdf(v)[0]-0.5,0.0,upper,xtol=root_xtol,rtol=root_rtol,maxiter=200)
    mass,error=cdf(root)
    return {'median':float(root),'cdf_at_median':mass,'cdf_residual':abs(mass-0.5),
            'quadrature_error_estimate':error,'root_xtol':root_xtol,'root_rtol':root_rtol,
            'quad_epsabs':epsabs,'quad_epsrel':epsrel,
            'limit':'Independent numerical check, not an exact symbolic/finite-N accuracy certificate.'}

def median_cdf_or_missing(*,fit_ok,known18_decimal=None,params=None,h1_percent2=None):
    """Primary P1 numerical bridge; NO fit is performed by this adapter.

    Two deterministic solves use predeclared tolerances. A passing guard is
    numerical evidence, not rigorous interval-arithmetic certification for all
    fitted parameter sets. Any unresolved step yields point=None, no fallback.
    """
    if not fit_ok:
        return {'status':'FIT_UNRESOLVED','point':None,'fallback_used':False,'primary_method':'conditional_CDF_root'}
    try:
        if not isinstance(params,Garch11Parameters):raise BridgeInputError('Explicit scalar GARCH parameters required')
        params.validate();known_summary(known18_decimal)
        base=median_quadrature_reference(known18_decimal,params,h1_percent2,
              epsabs=1e-9,epsrel=1e-9,root_xtol=1e-10,root_rtol=1e-10)
        strict=median_quadrature_reference(known18_decimal,params,h1_percent2,
              epsabs=1e-11,epsrel=1e-11,root_xtol=1e-12,root_rtol=1e-12)
        delta=abs(base['median']-strict['median'])
        tolerance=max(1e-7,1e-5*abs(strict['median']))
        passed=(all(x['cdf_residual']<=1e-7 and x['quadrature_error_estimate']<=2e-8 for x in (base,strict))
                and delta<=tolerance)
        return {'status':'NUMERICALLY_CHECKED_CDF_MEDIAN' if passed else 'NUMERICAL_UNRESOLVED',
          'point':strict['median'] if passed else None,'primary_method':'conditional_CDF_root',
          'base_solve':base,'stricter_repeat':strict,'repeat_difference':delta,
          'repeat_tolerance':tolerance,'fallback_used':False,
          'guard_status':'provisional_predeclared_numeric_guard_not_a_universal_error_bound'}
    except (BridgeInputError,ValueError,TypeError,FloatingPointError,OverflowError,RuntimeError) as exc:
        return {'status':'INVALID_OR_UNRESOLVED_CDF_BRIDGE','point':None,'reason':str(exc),
                'fallback_used':False,'primary_method':'conditional_CDF_root'}
