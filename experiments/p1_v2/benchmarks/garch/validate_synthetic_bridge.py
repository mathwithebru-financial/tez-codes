"""Synthetic checks + protocol receipt only. No market arrays/models are read."""
from pathlib import Path
import json,hashlib,sys,platform,datetime,ast,math
from dataclasses import asdict
import numpy as np
import scipy
import two_increment_vol20_bridge as b

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
MAIN=ROOT/'revision_20261112/archive/main_project'
SITE=Path(sys.executable).parent/'Lib/site-packages'
sources={
 'original08C':MAIN/'scripts/08C_garch_baselines_test_v4.py',
 'original08C_protocol':MAIN/'config/08C_garch_protocol_lock_v4.json',
 'original08C_summary':MAIN/'results/baselines/garch/garch_baseline_summary_v4.json',
 'original08C_R':MAIN/'scripts/08C_R_garch_rescue_run_v4.py',
 'original_closure':MAIN/'results/baselines/garch/08C_closure/08C_08C_R_closure_record_v4.json',
 'arch_volatility':SITE/'arch/univariate/volatility.py',
 'arch_distribution':SITE/'arch/univariate/distribution.py',
 'arch_metadata':SITE/'arch-8.0.0.dist-info/METADATA'}
text={k:p.read_text(encoding='utf-8-sig') for k,p in sources.items()}
identities={k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for k,p in sources.items()}
def evidence(k,start,end):
    return {'source_id':k,'path':str(sources[k]),'line_start':start,'line_end':end,
            'excerpt':'\n'.join(text[k].splitlines()[start-1:end])}
refs={
 'old_horizon1':evidence('original08C',979,993),
 'old_history_through_anchor':evidence('original08C',1090,1095),
 'old_known19_formula':evidence('original08C',1227,1263),
 'old_specs':evidence('original08C_protocol',260,295),
 'old_target_ddof_annualization':evidence('original08C_protocol',397,403),
 'old_recorded_arch8':evidence('original08C_summary',123,129),
 'full_data_dependent_feasibility':evidence('original08C_R',324,415),
 'old_final_scope':evidence('original_closure',113,157),
 'arch8_recursion_equation':evidence('arch_volatility',1013,1024),
 'arch8_vol_bounds_constraints':evidence('arch_volatility',1091,1132),
 'arch8_pathwise_recursion':evidence('arch_volatility',1336,1379),
 'arch8_t_fit_bounds':evidence('arch_distribution',482,508),
 'arch8_t_standardization':evidence('arch_distribution',585,636),
 'arch8_distribution_parameter_shape':evidence('arch_distribution',502,508),
 'arch8_version':evidence('arch_metadata',1,4)}

known=np.array([-.011,.006,.004,-.007,.010,-.003,.002,.013,-.005,.008,-.009,.001,-.004,.007,.003,-.006,.012,-.002],dtype=np.float64)
checks={};cases=[]
# A hand path verifies sign, percentage units and the ddof=1 algebra separately.
hand=b.Garch11Parameters('GJR_GARCH_1_1',.1,.1,.7,.1,8.)
hand_z=np.array([[-2.,.5],[2.,.5]],dtype=np.float64)
hand_r,hand_h2=b.simulate_two_increments(hand,1.,hand_z)
hand_v=b.vol20_from_increments(np.zeros(18),hand_r)
hand_brute=np.std(np.column_stack((np.zeros((2,18)),hand_r)),axis=1,ddof=1)*math.sqrt(252)
assert np.allclose(hand_h2,[1.6,1.2],rtol=0,atol=1e-14)
assert np.allclose(hand_r[:,0],[-.02,.02],rtol=0,atol=1e-15)
assert np.allclose(hand_v,hand_brute,rtol=0,atol=2e-14)
checks['hand_gjr_sign_percent_and_vol20']={'pass':True,'parameters':asdict(hand),'h1_percent2':1.,
 'standardized_shocks':hand_z.tolist(),'h2_percent2':hand_h2.tolist(),'returns_decimal':hand_r.tolist(),
 'vol20_formula':hand_v.tolist(),'vol20_brute_ddof1':hand_brute.tolist(),
 'negative_minus_positive_h2':float(hand_h2[0]-hand_h2[1]),'expected_gamma_times_r1_percent_squared':.4}

params_cases=[
 (b.Garch11Parameters('GARCH_1_1',.04,.07,.88,0.,8.),1.44),
 (b.Garch11Parameters('GJR_GARCH_1_1',.03,.05,.84,.12,8.),1.44),
 (b.Garch11Parameters('GJR_GARCH_1_1',.04,.15,.80,-.10,10.),.81)]
u=b.common_uniforms(4096,17011)
for p,h1 in params_cases:
    p.validate()
    z=b.standardized_t_from_uniform(u,p.nu)
    r,h2=b.simulate_two_increments(p,h1,z)
    fast=b.vol20_from_increments(known,r)
    full=np.column_stack((np.broadcast_to(known,(len(r),18)),r))
    brute=np.std(full,axis=1,ddof=1)*math.sqrt(252)
    maxdiff=float(np.max(np.abs(fast-brute)))
    assert maxdiff<2e-14,maxdiff
    # Independent conversion: decimal recurrence must agree with percent2/10000.
    h2_dec=p.omega_percent2/10000+(p.alpha+p.gamma*(r[:,0]<0))*r[:,0]**2+p.beta*h1/10000
    unitdiff=float(np.max(np.abs(h2/10000-h2_dec)))
    assert unitdiff<2e-18,unitdiff
    median=b.median_crn(known,p,h1)
    ref=b.median_quadrature_reference(known,p,h1)
    primary=b.median_cdf_or_missing(fit_ok=True,known18_decimal=known,params=p,h1_percent2=h1)
    assert primary['status']=='NUMERICALLY_CHECKED_CDF_MEDIAN' and primary['point'] is not None,primary
    error=abs(median['last_quantile_diagnostic']-ref['median'])
    tol=median['stages'][-1]['tolerance']
    assert median['point'] is not None,median
    assert error<=tol,(error,tol)
    assert ref['cdf_residual']<1e-7,ref
    # On the exact finite sample, the lower median minimizes absolute loss.
    z_small=b.standardized_t_from_uniform(b.common_uniforms(2048,27011),p.nu)
    r_small,_=b.simulate_two_increments(p,h1,z_small)
    vsmall=b.vol20_from_increments(known,r_small)
    m=float(np.quantile(vsmall,.5,method='inverted_cdf'))
    mae=float(np.mean(np.abs(vsmall-m)))
    pin=float(np.mean(np.maximum(.5*(vsmall-m),-.5*(vsmall-m))))
    plus=float(np.mean(np.abs(vsmall-(m+.001))))
    minus=float(np.mean(np.abs(vsmall-(m-.001))))
    assert abs(pin-.5*mae)<1e-16 and mae<=min(plus,minus)+1e-16
    cases.append({'parameters':asdict(p),'arch_vector_order':p.arch_parameter_order(),'h1_percent2':h1,
      'formula_vs_brute_std_max_abs':maxdiff,'percent_to_decimal_recursion_max_abs':unitdiff,
      'h2_path_min':float(np.min(h2)),'h2_path_max':float(np.max(h2)),
      'crn_quantile':median,'independent_conditional_CDF_quadrature':ref,'primary_CDF_adapter':primary,
      'MC_minus_reference_abs':error,'within_declared_numeric_tolerance':error<=tol,
      'sample_pinball_equals_half_MAE':{'mae':mae,'pinball':pin,'mae_at_median_plus_0p001':plus,
                                      'mae_at_median_minus_0p001':minus,'pass':True}})
checks['three_synthetic_cases']={'pass':True,'cases':cases}

# Deterministic zero-innovation boundary is not a valid positive-omega fitted model.
boundary=b.Garch11Parameters('GARCH_1_1',0.,0.,0.,0.,8.)
r0,h20=b.simulate_two_increments(boundary,0.,np.ones((4,2)),synthetic_degenerate=True)
zero_v=b.vol20_from_increments(np.zeros(18),r0)
constant_v=b.vol20_from_increments(np.full(18,.007),np.full((4,2),.007))
assert np.array_equal(r0,np.zeros((4,2))) and np.array_equal(zero_v,np.zeros(4))
assert np.max(np.abs(constant_v))<2e-14
checks['deterministic_constant_limiting_case']={'pass':True,'zero_h1_omega_future_increments':r0.tolist(),
 'all20_zero_return_vol':zero_v.tolist(),'all20_constant0p007_return_vol':constant_v.tolist(),
 'limit':'Zero omega/h1 is explicitly synthetic boundary only; helper rejects it for an ordinary fitted bridge.'}

failed=b.fit_bridge_or_missing(fit_ok=False)
invalid=b.fit_bridge_or_missing(fit_ok=True,known18_decimal=np.zeros(17),params=params_cases[0][0],h1_percent2=1.)
unresolved=b.fit_bridge_or_missing(fit_ok=True,known18_decimal=known,params=params_cases[1][0],h1_percent2=1.44,
                                  n_grid=(64,128,256),seeds=(17011,27011),absolute_tolerance=1e-15,relative_tolerance=0.)
assert failed['point'] is None and not failed['fallback_used']
assert invalid['point'] is None and not invalid['fallback_used']
assert unresolved['status']=='NUMERICAL_UNRESOLVED' and unresolved['point'] is None and not unresolved['fallback_used']
checks['failed_fit_invalid_shape_and_numeric_budget_failure']={'pass':True,'fit_failure':failed,
 'invalid_input':invalid,'forced_numeric_nonconvergence':unresolved}
primary_failed=b.median_cdf_or_missing(fit_ok=False)
primary_invalid=b.median_cdf_or_missing(fit_ok=True,known18_decimal=np.zeros(17),params=params_cases[0][0],h1_percent2=1.)
assert primary_failed['point'] is None and primary_invalid['point'] is None
# Inject a solver diagnostic failure to prove propagation, not to fabricate
# a population result. Restore the real solver before further computations.
saved_solver=b.median_quadrature_reference
try:
    def failed_solver(*args,**kwargs):
        return {'median':.1,'cdf_residual':1e-3,'quadrature_error_estimate':0.}
    b.median_quadrature_reference=failed_solver
    primary_numeric_fail=b.median_cdf_or_missing(fit_ok=True,known18_decimal=known,params=params_cases[0][0],h1_percent2=1.)
finally:b.median_quadrature_reference=saved_solver
assert primary_numeric_fail['status']=='NUMERICAL_UNRESOLVED' and primary_numeric_fail['point'] is None
saved_quad=b.quad
mass_rejected=False
try:
    b.quad=lambda *args,**kwargs:(1.0001,0.)
    try:b.conditional_vol_cdf(.3,known,params_cases[0][0],1.)
    except b.BridgeInputError:mass_rejected=True
finally:b.quad=saved_quad
assert mass_rejected
checks['primary_CDF_adapter_and_rejection_guards']={'pass':True,
 'three_case_results':[c['primary_CDF_adapter'] for c in cases],
 'fit_failure':primary_failed,'shape_failure':primary_invalid,'injected_solver_residual_failure':primary_numeric_fail,
 'injected_mass_above1p00000001_rejected':mass_rejected,
 'limit':'Injected failures test missing/no-fallback control flow; they are not actual fitted observations.'}
zstats=b.standardized_t_from_uniform(b.common_uniforms(65536,17011),8.)
checks['standardized_t_synthetic_moments']={'pass':True,'theoretical_variance':1.,'nu':8.,'n_per_dimension':65536,
 'sample_mean_by_dim':np.mean(zstats,axis=0).tolist(),'sample_variance_by_dim':np.var(zstats,axis=0).tolist(),
 'limit':'Sample moments are diagnostics; exact standardization follows the scale formula, not these noisy estimates.'}
assert np.max(np.abs(np.mean(zstats,axis=0)))<.03
assert np.max(np.abs(np.var(zstats,axis=0)-1))<.04

closure=json.loads(text['original_closure'])
protocol={
 'id':'P1_GARCH_TWO_INCREMENT_VOL20_BRIDGE_PROTOTYPE_V1','date':'2026-11-13',
 'status':'SYNTHETIC_VALIDATED_NUMERICAL_PROTOTYPE_NOT_A_FITTED_P1_BENCHMARK',
 'training_or_market_fit_authorized':False,
 'target_and_admission':{'original_P0_target_ret':'r[j]=r[t+1], a single endpoint-to-endpoint increment, not r[k+1]+r[k+2]',
  'P1_latest_admitted_index':'k=t-1','target_index':'j=t+1=k+2','horizon':'2 common ordered observations, not2calendar/trading days',
  'known_return_window':'r[j-19],...,r[k] = r[k-17],...,r[k], exactly18',
  'not_admitted_increments':['R1=r[k+1]','R2=r[k+2]=r[j]'],
  'target':'Vol20[j]=sqrt(252)*std(r[j-19],...,r[j],ddof=1)',
  'calendar_and_provider_limit':'Existing union-calendar/ffill target remains a constructed retrospective target. Lag1 is not a historical clock-availability certificate.'},
 'exact_distributional_derivation':{
  'innovation':'Z1,Z2 iid standardizedStudentT(nu), E[Z]=0,Var[Z]=1; Z=sqrt((nu-2)/nu)*T_nu',
  'parameters':'ZeroMean, GARCH/GJR(1,1),power2,StudentsT,rescaleFalse; parameter estimates treated as fixed conditional plug-in values, no parameter uncertainty integration',
  'h1':'h[k+1|F_k] in percentage-return squared, passed from a valid fit/filter using ONLY returns through k',
  'R1_percent':'sqrt(h1_percent2)*Z1',
  'h2_percent2_given_R1':'omega_percent2+(alpha+gamma*I[R1_percent<0])*R1_percent^2+beta*h1_percent2; gamma=0 forGARCH',
  'R2_percent':'sqrt(h2_percent2_given_R1)*Z2',
  'dependence':'Z1,Z2 independent innovations; R1,R2 not independent because h2 depends onR1. Replacing h2 by its expected2step variance loses the joint predictive law.',
  'unit_conversion':'R_decimal=R_percent/100; h_decimal=h_percent2/10000; omega_decimal=omega_percent2/10000. Alpha,beta,gamma,nu unchanged.',
  'S18':'sum18known decimal returns','Q18':'sum18known squared decimal returns',
  'sample_variance20':'V(R1,R2)=[Q18+R1_dec²+R2_dec²-(S18+R1_dec+R2_dec)²/20]/19',
      'stable_implementation':'Equivalent centered/Welford update: SSE20=SSE18+(18/19)*(R1-c18)^2+(19/20)*(R2-c19)^2;V20=SSE20/19. Avoids cancellation nearconstant windows.',
  'conditional_median_target':'m=Q0.5[ sqrt(252*V(R1,R2)) | F_k, fitted_parameters ]',
  'population_monotone_property':'Population median of sqrt(252V) is sqrt(252*median(V)). Finite interpolated sample medians need not commute; the chosen inverted_cdf order statistic does.',
  'one_dimensional_CDF_identity':{
   'S19':'S18+R1_dec','Q19':'Q18+R1_dec²','c19':'S19/19','C19':'(Q19-S19²/19)/19',
   'V_conditional_on_R1':'C19+(R2_dec-c19)²/20',
   'F_V(v)':'Integral f_R1(x)*I[v>=C19(x)]*[F_R2|x(c19(x)+sqrt(20(v-C19(x))))-F_R2|x(c19(x)-sqrt(20(v-C19(x))))]dx',
   'reference_solver':'Synthetic only: scipy.integrate.quad over the finite admissibleR1 interval, split at0; brentq root ofF_V(m²/252)=.5.'}},
 'shape_and_parameter_checks':{
  'known_returns_shape':[18],'innovation_and_increment_path_shapes':['N',2],
  'GARCH_arch_parameter_names':['omega','alpha[1]','beta[1]','nu'],
  'GJR_arch_parameter_names':['omega','alpha[1]','gamma[1]','beta[1]','nu'],
  'linear_constraints':'omega>=0;alpha>=0;beta>=0;alpha+gamma>=0;alpha+.5gamma+beta<=1;gamma=0forGARCH',
  'local_arch8_static_bounds':'alpha,beta in[0,1]; gamma in[-1,2] forp=o=1;nu in[2.05,500]',
  'nu_math_vs_fit':'nu>2 mathematically defines finite unit variance; actual localarch8fitting bounds are stricter[2.05,500]',
  'omega_fit_bounds':'[1e-8*v,10*v],v=mean(abs(fit_residuals)^power); requires full fitted history and is not verified by this bridge',
  'h1_validity':'positive finite in production; h1=omega=0 permitted only in labelled synthetic boundary test',
  'complete_arch_feasibility':'A future fitting adapter must retain exact data-dependent arch feasibility checks from original08C-R; this helper checks only stated scalar/static conditions.',
  'negative_gamma':'Allowed when alpha+gamma>=0 and other constraints hold; validated withgamma=-0.10.'},
 'numerical_policy':{
  'primary_method':'deterministic_conditional_CDF_integration_and_root',
  'primary_adapter':'median_cdf_or_missing',
  'primary_predeclared_guard':{'base_quad_epsabs':1e-9,'base_quad_epsrel':1e-9,'base_root_xtol':1e-10,'base_root_rtol':1e-10,
   'repeat_quad_epsabs':1e-11,'repeat_quad_epsrel':1e-11,'repeat_root_xtol':1e-12,'repeat_root_rtol':1e-12,
   'cdf_residual_max':1e-7,'quadrature_error_estimate_max':2e-8,
   'repeat_median_difference_max':'max(1e-7,1e-5*abs(stricter_median)), annualized decimalVol20',
   'material_probability_violation':'mass<-1e-8 ormass>1+1e-8 =>reject, never silentlyclip',
   'on_failure':'point=None, explicitinvalid/unresolved status, nofallback',
   'interpretation':'Provisional numerical stability/error-estimate guard selected before any marketfits/testmetrics; not rigorous interval-arithmetic population-median certification.'},
  'MC_role':'Independent synthetic diagnostic only; theCRNpoint is not the primary P1benchmark prediction',
  'algorithm':'PCG64fixed-seed common uniforms(N,2), inverseStudentT CDF withunitvariance scale, pathwise2step recursion, exactVol20 formula, pooled empirical median',
  'simulation_seeds':[17011,27011,37011,47011],
  'training_seeds_are_different':'123/777/2026 belong to model training; not used as an implicit MC tolerance choice',
  'N_levels_per_replication':[4096,16384,65536],
  'common_random_numbers':'Within each replication, allNlevels use prefixes of the same stream; same streams can be reused acrossGARCH/GJR specification at a given experiment. Explicit stream keys must be retained in any future adapter.',
  'quantile_method':'numpy.quantile(q=.5,method=inverted_cdf): lower empirical median, a valid finite-sample MAE/pinball0.5 minimizer',
  'acceptance_tolerance_units':'annualized decimalVol20',
  'absolute_tolerance':2e-4,'relative_tolerance':2e-3,
  'acceptance':'At the maximum grid: each of the last2successive pooled-quantile changes<=max(atol,rtol*abs(point)), and every replication median deviates from pooledpoint by<=that tolerance.',
  'failure':'NUMERICAL_UNRESOLVED withpoint=None; diagnostics retained, no persistence fallback. No observed stable result means no eligible benchmark prediction.',
  'hard_budget_limit':'A fixed finiteN and passing replication/prefix diagnostics DO NOT prove a uniform or exact population-quantile error bound. Synthetic reference agreement applies only to the tested parameters; future fit cases need the same guards and may remain unresolved.',
  'pinball_error_propagation':'For tau=.5, |rho(y-mhat)-rho(y-m)|<=.5*|mhat-m|. If an actual quantile error boundepsilon were established, mean pinball error is bounded by.5*mean(epsilon); a ratio error is additionally divided by that asset baseline denominator. Observed replication stability alone is not such a bound.',
  'before_market_freeze_gate':'PrimaryCDF/root tolerances and stricter-repeat guards are explicitly predeclared here, independently of markettestmetrics. MC tolerances remain SYNTHETIC diagnostic settings and do not select/replace the primaryCDFpoint. No exact universal quantileerror certificate is supplied; passing3cases does not certify allfits and any future failedguard remains unresolved.',
  'reference_tolerances':'quad epsabs=epsrel=1e-9,reportederror<=2e-8; brentq xtol=rtol=1e-10; syntheticCDFresidual<1e-7',
  'identity_tolerance':'Vol20 formula vsbrute-array std maxabs<2e-14; percent/decimal recursion maxabs<2e-18',
  'negative_variance_handling':'Values below-1e-14 fail; tiny negative roundoff in[-1e-14,0) may be set tozero only; never repair substantive invalid paths.'},
 'future_fit_pipeline_NOT_EXECUTED':[
  'Take root-approved P1date/mask contract. Resolve latest admitted k=t-1 and targetj=k+2 by exact ordered indices; assert known18end==k and no later row is read.',
  'Fit expanding univariate admitted returns throughk only;meanZero,p1,q1,o0or1,power2,StudentsT,rescaleFalse,percentage units. Preserve inherited validation-independent retry/start policy before execution; this file authorizes no fit.',
  'Accept fitted parameters/h1only after convergence, finite outputs, exactarch constraints/data-dependent bounds andnu checks. Invalidfit=>explicit missing status; no parameter/model/persistence fallback.',
  'For each validfit, integrate the joint two-increment conditionalCDF and solve itsVol20median through median_cdf_or_missing, including the stricter-repeat guard. CRNpath quantiles are independent diagnostics only. Reject unresolved bridges instead of publishing the last diagnostic quantile.',
  'Scoring masks select metric rows but do not silently compress the fitted calendar or change return windows. Any such history deletion is a separate data/model protocol requiring a newly derived horizon.',
  'Compare on identical targetdates and scoring masks, retain complete-pair coverage and numerical failures. Use the targetmedian forpinball0.5; directdailyconditional sigma remains a different diagnostic target.',
  'Before any execution, root must pin numericalbudget/tolerance and exception policy without testperformance tuning; test observed errors must not determineMCbudget, models or retries.'
 ],
 'old_provenance_preserved':{'pipeline_status':closure['pipeline_status'],
  'parent_unresolved':574,'successful_secondary_rescues':268,'remaining_unresolved':306,
  'rescue_merged_new_series':False,'canonical_parent_retained':True,
  'scope':closure['scientific_interpretation']['scope_limitation'],
  'not_claimed':'No newP1GARCH fit, no repair ofhistoricalUSDTRYfailures, no newmarket prediction.'},
 'source_evidence':refs,'source_identities':identities,
 'local_reference_scope':'Original run recordsarch8.0.0; inspected installedarch8.0.0 source bytext only. Same version is not asserted as byte-identical to the historical installed wheel.',
 'actions':{'original_code_import_or_execution':False,'arch_import':False,'fit':False,'training':False,
  'market_prediction':False,'raw_market_data_read':False,'scaler_fit':False,'checkpoint_read':False,
  'new_numpy_tensor_datasets':False,'thesis_edit':False,'synthetic_numeric_checks':True}}
for k,p in sources.items():assert hashlib.sha256(p.read_bytes()).hexdigest()==identities[k]['sha256']
# Confirm the new numerical helper cannot import arch or original project code.
tree=ast.parse((HERE/'two_increment_vol20_bridge.py').read_text(encoding='utf-8'))
imports=[]
for n in ast.walk(tree):
    if isinstance(n,ast.Import):imports.extend(x.name for x in n.names)
    if isinstance(n,ast.ImportFrom):imports.append(n.module or '')
assert not any(x.split('.')[0] in {'arch','torch','pickle','joblib','pandas'} for x in imports)
assert 'arch' not in sys.modules and 'torch' not in sys.modules
receipt={'status':'PASS_SYNTHETIC_ONLY','date':'2026-11-13','primary_numerical_method':'deterministic_conditional_CDF_root',
 'arch_actually_imported':False,'torch_actually_imported':False,'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__},
 'checks':checks,'helper_imports':imports,'original_sources_unchanged':True,'source_identities':identities,
 'scope':protocol['actions'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'helper_sha256':hashlib.sha256((HERE/'two_increment_vol20_bridge.py').read_bytes()).hexdigest(),
 'limit':'Synthetic bridge validation does not establish P1fit coverage, market forecasts or exactfiniteNquantile accuracy for every fitted parameter set.'}
(HERE/'garch_P1_bridge_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(HERE/'synthetic_validation_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
report='''P1 / GARCH İKİ GELECEK GETİRİ KÖPRÜSÜ — 13 Kasım 2026

Bu paket yalnız sayısal köprü ve sentetik doğrulamadır. arch import/fit, piyasa tahmini, model/scaler açma, yeni piyasa veri dosyası veya tez değişikliği yoktur.

İndeks: k=t−1 son kabul edilen gözlem; j=t+1=k+2 hedeftir. Getiri hedefi yalnız r[j]’dir; iki getirinin toplamı değildir. Vol20[j] penceresi r[j−19]…r[j] olduğundan r[k−17]…r[k] arasındaki 18 getiri bilinir; R1=r[k+1] ve R2=r[k+2] henüz kabul edilmemiştir. İki adım iki ortak takvim gözlemidir; 48 saat veya iki işlem günü garantisi değildir.

Yüzde biriminde iki adımlı ortak dağılım:
Z1,Z2 iid standardized Student-t(nu), Z=sqrt((nu−2)/nu)*T_nu.
R1_pct=sqrt(h1_pct²)*Z1.
h2_pct²=omega_pct²+(alpha+gamma*1[R1_pct<0])*R1_pct²+beta*h1_pct².
R2_pct=sqrt(h2_pct²)*Z2. GARCH için gamma=0.
Z1 ve Z2 bağımsız yeniliklerdir; R1 ve R2 bağımsız değildir. h2, ilk gerçekleşen şoka bağlıdır. Tek bir beklenen ikinci-adım varyansını her yola koymak aynı ortak dağılımı vermez.

R1,R2 daha sonra 100’e bölünerek ondalık getiriye çevrilir; yüzde-varyans ve omega 10000’e bölünür. S18=sum(18r), Q18=sum(18r²):
V20(R1,R2)=[Q18+R1²+R2²−(S18+R1+R2)²/20]/19.
Vol20=sqrt(252*V20). Ana nokta tahmini bu değişkenin koşullu medyanıdır. sqrt(252*h2) günlük sigma hedefidir; Vol20 yerine geçmez.

Bağımsız kontrol için R1 sabitken S19=S18+R1, Q19=Q18+R1², c19=S19/19, C19=(Q19−S19²/19)/19 yazılır. Böylece V20=C19+(R2−c19)²/20. CDF, R1 dağılımı üzerinden R2’nin c19±sqrt(20(v−C19)) aralığına düşme olasılığının integralidir. Sentetik referans bunu tek boyutlu quad integrali ve brentq medyan kökü ile çözer.

Birincil sayısal yöntem: median_cdf_or_missing iki koşullu CDF/root çözümü yapar. İlk çözüm quad epsabs=epsrel=1e−9, root xtol=rtol=1e−10; daha sıkı tekrar 1e−11 ve 1e−12 kullanır. Her ikisinde CDF kök artığı <=1e−7, quad hata tahmini <=2e−8 ve iki medyan farkı <=max(1e−7,1e−5*|medyan|) olmalıdır. Olasılık −1e−8’den küçük veya 1+1e−8’den büyükse reddedilir; sessizce aralığa çekilmez. Bir adım başarısızsa point=None ve açıklanan eksik statü, fallback yoktur. Bu önceden belirlenmiş geçici sayısal koruma kuralıdır; bütün parametrelerde kesin interval-aritmetik hata sertifikası değildir.

CRN yaklaşımı yalnız bağımsız sentetik tanıdır: dört ayrı PCG64 akışı, her akışta iki uniform sütunu ve standartlaştırılmış t ters CDF’si. N=4096/16384/65536 düzeyleri aynı akışların öneklerini kullanır. Aynı akışlar GARCH/GJR karşılaştırmasında ortak rastgele sayılar sağlar. Numpy inverted_cdf alt ampirik medyan seçildi; ampirik MAE/pinball(0.5) minimizer’ıdır ve monoton karekök dönüşümüyle sıra istatistiği korunur. linear yöntemi de medyan minimizer’ı olabilir; fakat çift örnek sayısında interpolasyonun karekökle yer değiştirmesi tam değildir. MC noktası birincil P1 benchmark tahmini değildir.

Sayısal kural: yıllıklandırılmış ondalık Vol20 biriminde tol=max(0.0002,0.002*|medyan|). Son iki N-artışındaki medyan farkı ve son düzeyde tekrar medyanlarının ortak medyandan sapması bu toleransı aşmamalı. Aksi hâlde NUMERICAL_UNRESOLVED ve point=None; son kuantil teşhis olarak tutulur, persistence ile doldurulmaz. Sabit N veya gözlenen yakınlık bütün parametreler için kesin hata garantisi değildir. Prototip yalnız belirtilen sentetik örneklerde sınandı; sonraki gerçek fitler de başarısız kalabilir.

Bu MC toleransları yalnız sentetik tanı ayarlarıdır; gerçek benchmark farklarına göre küçük sayılacakları henüz gösterilmedi. Pinball(0.5) için |kayıp(mhat)−kayıp(m)|<=0.5*|mhat−m|. Gerçek bir medyan hata sınırı epsilon kurulursa ortalama pinball etkisi en fazla 0.5*mean(epsilon), oran üzerindeki etki ayrıca ilgili paydasına bölünür. Önek/tekrar kararlılığı tek başına böyle bir hata sınırı değildir. Birincil CDF/root ve daha sıkı tekrar kuralları ise burada, hiçbir piyasa fitine veya test hatasına bakmadan belirlenmiştir. Sonraki gerçek fitlerden biri korumayı geçemezse eksik kalır; MC değeriyle değiştirilmez. Burada genel kesin medyan hata sertifikası verilmemiştir.

Yerel arch8.0.0 metninde alpha>=0, beta>=0, alpha+gamma>=0, alpha+0.5gamma+beta<=1; gamma negatif olabilir. Nu’nun matematiksel koşulu >2, gerçek yerel fit sınırı [2.05,500]. Omega sınırı veriyle değişir; tüm geçmişi kullanan gerçek arch feasibility kontrolü bu salt köprü tarafından doğrulanmaz. Gelecekteki adapter bunu ayrıca uygulamak zorundadır.

Sentetik kontroller: el hesabı GJR işareti/yüzde dönüşümü; üç GARCH/GJR parametre kümesinde kısa formül ile 20 sütunlu std(ddof=1) eşitliği; bağımsız koşullu CDF medyanı; negatif gamma; sabit sıfır varyans sınırı; bütün getiriler sabitken sıfır volatilite; pinball(0.5)=yarım MAE; fit/şekil/sayısal başarısızlıkta eksik çıktı ve sıfır fallback. Ayrıntılı sayılar synthetic_validation_receipt.json içindedir.

Sonraki fit akışı burada çalıştırılmadı: tarih ve mask sözleşmesini al; yalnız k dahil geçmişe yüzde getiriyle fit et; geçerli yakınsama, parametre ve h1 kontrolünden sonra iki adımlı köprüyü uygula. Scoring mask satırları ölçümden çıkarabilir, ancak sessizce geçmişi veya takvim indekslerini sıkıştırmamalı. Ortak hedef tarihleri ve eksiksiz çift kapsamı korunmalı. Model, retry veya MC bütçesi test hatasına bakılarak seçilmemeli.

Eski 08C/08C-R zinciri değişmedi: parent canonical; 574 başarısız gözlemden 268 ek geçerli kurtarma, 306 kalan; rescue ile yeni birleşik tahmin serisi üretilmedi. Nihai durum CLOSED_WITH_DOCUMENTED_SCOPE_LIMITATION; yalnız iki USDTRY GARCH tam-seri karşılaştırması hariç. Bu sentetik paket eski kapsamı tamir ettiğini veya P1 GARCH sonucunu ürettiğini iddia etmez.
'''
for c in cases:
    report+='\n'+c['parameters']['model']+' gamma='+str(c['parameters']['gamma'])+': CRN median='+str(c['crn_quantile']['point'])+', CDF reference='+str(c['independent_conditional_CDF_quadrature']['median'])+', |difference|='+str(c['MC_minus_reference_abs'])+'\n'
report+='\nKaynak satırları ve hashler garch_P1_bridge_protocol.json içinde. Kaynaklar önce/sonra aynı hash değerindedir.\n'
(HERE/'garch_P1_bridge_protocol.txt').write_text(report,encoding='utf-8')
print(json.dumps({'status':receipt['status'],'cases':[{'model':c['parameters']['model'],'gamma':c['parameters']['gamma'],
 'MC_median':c['crn_quantile']['point'],'reference_median':c['independent_conditional_CDF_quadrature']['median'],
 'absolute_difference':c['MC_minus_reference_abs'],'brute_std_max_difference':c['formula_vs_brute_std_max_abs']} for c in cases],
 'checks_passed':all(x['pass'] for x in checks.values()),'original_sources_unchanged':True,
 'arch_imported':'arch' in sys.modules,'market_fit_or_training':False},ensure_ascii=False,indent=2))
