# PORTABLE DERIVATIVE: archived Nov10 numerical formulas retained.
# Path/CLI/output/provenance changes are enumerated in post_review/path_relocations.json.
# Does not train models, execute checkpoints, recompute SHAP sampling or download data.
"""Post-hoc H1 return HAC supplement. Read frozen originals; write isolated outputs.

Run after downloading the three input files: python run_analysis.py
No model fitting, inference, price download or original-file writes occur here.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd
import scipy
from scipy.stats import t, norm, chi2
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import acf

import argparse
_parser = argparse.ArgumentParser(description="Portable fixed-loss return HAC supplement; no training/inference")
_parser.add_argument("--archive-root", type=Path, default=Path(__file__).resolve().parent.parent)
_parser.add_argument("--output-dir", type=Path)
_args = _parser.parse_args()
ARCHIVE = _args.archive_root.resolve()
REFERENCE = ARCHIVE / "post_review/return_robustness"
HERE = (_args.output_dir or ARCHIVE / "verification/post_review_return_hac").resolve()
if HERE == ARCHIVE or ARCHIVE.is_relative_to(HERE) or HERE.is_relative_to(ARCHIVE / "main_project") or HERE.is_relative_to(ARCHIVE / "h2_project") or HERE.is_relative_to(ARCHIVE / "post_review"):
    raise RuntimeError("Use a new output directory outside immutable source/input folders")
HERE.mkdir(parents=True, exist_ok=False)
WORKSPACE = ARCHIVE
INPUTS = REFERENCE / "inputs"
ASSETS = ["BIST100", "USDTRY", "EURTRY", "GOLD"]
N = 584

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save_json(name, data):
    (HERE / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    corrected = np.empty(len(p))
    corrected[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    np.testing.assert_allclose(corrected, multipletests(p, method="holm")[1], rtol=1e-13, atol=1e-15)
    return corrected

def decision(mean, p):
    if p > .05:
        return "NOT_SIGNIFICANT"
    return "SIGNIFICANT_FINAL_BETTER" if mean > 0 else "SIGNIFICANT_BASELINE_BETTER"

def main():
    plan_bytes = (REFERENCE / "plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    assert plan["bartlett_bandwidths"] == [3, 6, 10, 19]
    expected = {
        "naive_baseline_loss_series_v4.npz": "97633ac36d1e7d3a4266aef753902d5b79796baaadc7bad0d8c0640efb06477c",
        "learned_baseline_loss_series_v4.npz": "34ea36086107c4e2fbc3304c4507272d2ce24a8c84c22e2cda6636f9a4ecea0f",
    }
    for name, sha in expected.items():
        assert digest(INPUTS / name) == sha, name
    with np.load(INPUTS / "naive_baseline_loss_series_v4.npz", allow_pickle=False) as f:
        naive = {k: f[k].astype(np.float64) for k in f.files}
    with np.load(INPUTS / "learned_baseline_loss_series_v4.npz", allow_pickle=False) as f:
        learned = {k: f[k].astype(np.float64) for k in f.files}
    original = pd.read_csv(INPUTS / "09_dm_harvey_holm_results_v4.csv", float_precision="round_trip")
    original = original.loc[original.task.eq("return")].reset_index(drop=True)
    assert len(original) == 20 and original.comparison_id.is_unique
    assert original.holm_family.eq("RETURN_PRIMARY").all() and original.n.eq(N).all()
    inventory = [(a, m) for a in ASSETS for m in ["ReturnZero", "ReturnPersistence"]]
    inventory += [(a, m) for a in ASSETS for m in ["SingleTaskTransformer", "SingleTaskLSTM", "XGBoost"]]
    assert list(zip(original.asset, original.baseline_model)) == inventory

    truth_path = WORKSPACE / "main_project/results/final_test/final_test_y_true_raw_v4.npy"
    pred_path = WORKSPACE / "main_project/results/final_test/pred_final_ensemble_raw_v4.npy"
    raw_path = WORKSPACE / "main_project/data/raw/raw_prices.csv"
    assert digest(truth_path) == "a37034c833365b89302dbf0e3d57c29dc2eaf6ada91bfd1fae16fd56f112adb8"
    assert digest(pred_path) == "17f6c8ca42d833118e057b72b6f481c8cde5c77501ab737559f67b9a6e3e13a1"
    assert digest(raw_path) == "ab5f275d38dc98057b1cedcf58019adb26be7402c7ed5ae6ee3d6877b2444893"
    truth = np.load(truth_path, allow_pickle=False)
    pred = np.load(pred_path, allow_pickle=False)
    assert truth.shape == pred.shape == (N, 8)
    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)[ASSETS]
    target_dates = raw.index[-N:]
    anchor_dates = raw.index[-N-1:-1]
    returns = np.log(raw.ffill() / raw.ffill().shift())
    np.testing.assert_array_equal(returns.loc[target_dates].to_numpy().astype("float32"), truth[:, :4])
    dates = pd.DataFrame({"test_index": np.arange(N), "anchor_date": anchor_dates.strftime("%Y-%m-%d"),
                          "target_realization_date": target_dates.strftime("%Y-%m-%d")})
    dates.to_csv(HERE / "test_dates.csv", index=False)
    original_final_loss = np.abs(truth[:, :4] - pred[:, :4]).astype(np.float64)
    for j, a in enumerate(ASSETS):
        np.testing.assert_array_equal(naive[f"final_return_abs__{a}"], original_final_loss[:, j])
        np.testing.assert_array_equal(naive[f"return_zero_abs__{a}"], np.abs(truth[:, j]).astype(np.float64))

    pairs = []
    differentials = dates.copy()
    for row in original.itertuples():
        final = naive[f"final_return_abs__{row.asset}"]
        if row.baseline_family == "naive":
            key = f"{'return_zero_abs' if row.baseline_model == 'ReturnZero' else 'return_persistence_abs'}__{row.asset}"
            baseline = naive[key]
        else:
            key = f"{row.baseline_model}__return__{row.asset}"
            baseline = learned[key]
        assert final.shape == baseline.shape == (N,)
        assert np.isfinite(final).all() and np.isfinite(baseline).all()
        d = baseline - final
        assert not np.all(d == d[0]), "Nonzero constant differential is not testable"
        pairs.append((row, final, baseline, d))
        differentials[row.comparison_id] = d
    differentials.to_csv(HERE / "loss_differentials.csv", index=False, float_format="%.17g")

    result_rows, acf_rows, diag_rows = [], [], []
    internal = {"hac_variance_max_abs_diff": 0., "acf_max_abs_diff": 0., "ljung_box_Q_max_abs_diff": 0.,
                "ljung_box_p_max_abs_diff": 0., "L0_numeric_max_abs_diff": {}, "L0_all_decisions_match": True}
    for row, final, baseline, d in pairs:
        mean = float(d.mean())
        centered = d - mean
        gammas = np.array([np.dot(centered[k:], centered[:N-k]) / N for k in range(21)])
        assert gammas[0] > 0
        rho = gammas / gammas[0]
        sm_acf = acf(d, nlags=20, adjusted=False, fft=False)
        internal["acf_max_abs_diff"] = max(internal["acf_max_abs_diff"], float(np.max(np.abs(rho-sm_acf))))
        np.testing.assert_allclose(rho, sm_acf, atol=1e-13, rtol=1e-12)
        for lag in range(1, 21):
            acf_rows.append({"comparison_id": row.comparison_id, "asset": row.asset, "baseline_model": row.baseline_model,
                             "lag": lag, "autocovariance_denominator_T": float(gammas[lag]), "acf": float(rho[lag])})
        diag = {"comparison_id": row.comparison_id, "asset": row.asset, "baseline_model": row.baseline_model,
                "acf1": float(rho[1]), "acf5": float(rho[5]), "acf10": float(rho[10]), "acf20": float(rho[20]),
                "max_abs_acf_1_20": float(np.max(np.abs(rho[1:])))}
        sm_lb = acorr_ljungbox(d, lags=[5, 10, 20], model_df=0, return_df=True)
        for lag in [5, 10, 20]:
            q = float(N * (N+2) * np.sum(rho[1:lag+1]**2 / (N-np.arange(1, lag+1))))
            p = float(chi2.sf(q, lag))
            internal["ljung_box_Q_max_abs_diff"] = max(internal["ljung_box_Q_max_abs_diff"], abs(q-float(sm_lb.loc[lag,"lb_stat"])))
            internal["ljung_box_p_max_abs_diff"] = max(internal["ljung_box_p_max_abs_diff"], abs(p-float(sm_lb.loc[lag,"lb_pvalue"])))
            np.testing.assert_allclose(q, sm_lb.loc[lag,"lb_stat"], rtol=1e-12, atol=1e-12)
            diag[f"lb{lag}_Q"] = q
            diag[f"lb{lag}_p"] = p
        diag_rows.append(diag)
        ols = sm.OLS(d, np.ones((N, 1))).fit()
        for L in [0] + plan["bartlett_bandwidths"]:
            omega = float(gammas[0] + 2*np.sum((1-np.arange(1,L+1)/(L+1))*gammas[1:L+1]))
            assert omega > 0 and np.isfinite(omega), (row.comparison_id, L, omega)
            variance = omega / N
            alternative = float(cov_hac(ols, nlags=L, use_correction=False)[0,0])
            internal["hac_variance_max_abs_diff"] = max(internal["hac_variance_max_abs_diff"], abs(variance-alternative))
            np.testing.assert_allclose(variance, alternative, rtol=2e-12, atol=1e-20)
            se = float(np.sqrt(variance))
            dm = mean / se
            factor = float(np.sqrt((N-1)/N))
            dm_hln = dm * factor
            result_rows.append({"comparison_id": row.comparison_id, "task": "return", "asset": row.asset,
                "baseline_family": row.baseline_family, "baseline_model": row.baseline_model,
                "loss_function": "absolute_error", "holm_family": "RETURN_PRIMARY", "n": N, "bandwidth": L,
                "forecast_horizon": 1, "mean_final_loss": float(final.mean()), "mean_baseline_loss": float(baseline.mean()),
                "mean_loss_differential": mean, "median_loss_differential": float(np.median(d)),
                "relative_loss_reduction_pct": float(100*mean/baseline.mean()),
                "loss_differential_gamma0": float(gammas[0]), "long_run_variance": omega,
                "dm_standard_error": se, "dm_raw_statistic": dm, "dm_raw_p_normal_two_sided": float(2*norm.sf(abs(dm))),
                "hln_correction_factor": factor, "dm_hln_statistic": dm_hln,
                "dm_hln_p_t_two_sided": float(2*t.sf(abs(dm_hln), N-1)),
                "variance_policy_case": "STANDARD_POSITIVE_VARIANCE"})
    result = pd.DataFrame(result_rows)
    for L in [0]+plan["bartlett_bandwidths"]:
        mask = result.bandwidth.eq(L)
        result.loc[mask, "holm_adjusted_p"] = holm(result.loc[mask, "dm_hln_p_t_two_sided"])
    result["null_rejected_holm"] = result.holm_adjusted_p.le(.05)
    result["interpretation"] = [decision(m,p) for m,p in zip(result.mean_loss_differential, result.holm_adjusted_p)]
    l0 = result.loc[result.bandwidth.eq(0)].reset_index(drop=True)
    numeric = [c for c in l0.columns if c in original and pd.api.types.is_numeric_dtype(l0[c]) and c not in ["null_rejected_holm"]]
    for col in numeric:
        diff = float(np.max(np.abs(l0[col].to_numpy()-original[col].to_numpy())))
        internal["L0_numeric_max_abs_diff"][col] = diff
        np.testing.assert_allclose(l0[col], original[col], atol=3e-12, rtol=3e-12, err_msg=col)
    assert l0.interpretation.tolist() == original.interpretation.tolist()
    assert l0.null_rejected_holm.tolist() == original.null_rejected_holm.tolist()
    baseline_decisions = dict(zip(l0.comparison_id, l0.interpretation))
    result["original_L0_decision"] = result.comparison_id.map(baseline_decisions)
    result["changed_vs_L0"] = result.interpretation.ne(result.original_L0_decision)
    result.to_csv(HERE / "dm_results_all_bandwidths.csv", index=False, float_format="%.17g")
    counts=[]
    for L,g in result.groupby("bandwidth", sort=True):
        counts.append({"bandwidth":int(L), "n_comparisons":len(g), "final_better":int(g.interpretation.eq("SIGNIFICANT_FINAL_BETTER").sum()),
            "baseline_better":int(g.interpretation.eq("SIGNIFICANT_BASELINE_BETTER").sum()),
            "not_significant":int(g.interpretation.eq("NOT_SIGNIFICANT").sum()), "changed_vs_L0":int(g.changed_vs_L0.sum())})
    pd.DataFrame(counts).to_csv(HERE / "decision_counts.csv", index=False)
    matrix=original[["comparison_id","asset","baseline_model","mean_loss_differential"]].copy()
    for L in [0]+plan["bartlett_bandwidths"]:
        g=result.loc[result.bandwidth.eq(L)].set_index("comparison_id")
        matrix[f"holm_p_L{L}"] = matrix.comparison_id.map(g.holm_adjusted_p)
        matrix[f"decision_L{L}"] = matrix.comparison_id.map(g.interpretation)
    matrix.to_csv(HERE / "pvalue_matrix.csv", index=False, float_format="%.17g")
    result.loc[result.changed_vs_L0].to_csv(HERE / "changed_decisions.csv", index=False, float_format="%.17g")
    pd.DataFrame(acf_rows).to_csv(HERE / "acf.csv", index=False, float_format="%.17g")
    diagnostics = pd.DataFrame(diag_rows)
    diagnostic_ps=np.array([r[f"lb{m}_p"] for r in diag_rows for m in [5,10,20]])
    adjusted=holm(diagnostic_ps).reshape(20,3)
    for j,m in enumerate([5,10,20]):
        diagnostics[f"lb{m}_holm_p_60"] = adjusted[:,j]
    diagnostics.to_csv(HERE / "dependence_diagnostics.csv", index=False, float_format="%.17g")
    diagnostic_counts={str(m):int(diagnostics[f"lb{m}_holm_p_60"].le(.05).sum()) for m in [5,10,20]}
    input_records=[]
    for p in [REFERENCE/"plan.json", INPUTS/"naive_baseline_loss_series_v4.npz", INPUTS/"learned_baseline_loss_series_v4.npz", INPUTS/"09_dm_harvey_holm_results_v4.csv", truth_path,pred_path,raw_path,Path(__file__)]:
        input_records.append({"path":str(p.relative_to(WORKSPACE)),"sha256":digest(p),"size_bytes":p.stat().st_size})
    summary={"label":"post-hoc fixed-loss H1 return serial-dependence supplement", "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "plan_sha256":hashlib.sha256(plan_bytes).hexdigest(), "old_results_known":True,
        "n":N, "comparison_count":20, "orientation":"baseline-final", "bandwidths":[0]+plan["bartlett_bandwidths"],
        "first_anchor":dates.anchor_date.iloc[0],"last_anchor":dates.anchor_date.iloc[-1],
        "first_target":dates.target_realization_date.iloc[0],"last_target":dates.target_realization_date.iloc[-1],
        "decision_counts":counts,"ljung_box_holm60_rejections_by_lag":diagnostic_counts,
        "ljung_box_any_rejection_holm60_count":int((adjusted<=.05).any(axis=1).sum()),
        "checks":internal,"inputs":input_records,
        "environment":{"python":sys.version,"executable":sys.executable,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"statsmodels":statsmodels.__version__},
        "limits": ["Post-hoc supplement: old main results already known; no unseen validation.","No bandwidth selected by significance.",
          "Per-bandwidth Holm controls a 20-comparison family under valid marginal p-values; no joint guarantee across bandwidths.",
          "HLN with h1 and t583 is retained for comparability; no exact finite-sample validity is claimed under arbitrary dependence.",
          "HAC and Ljung-Box do not repair nonstationarity, source prices, common-calendar filling, validation selection or clock-time availability.",
          "ACF/Ljung-Box use common-calendar lags, not elapsed-day lags; diagnostic rejections can reflect heteroskedasticity or structural changes."]}
    save_json("analysis_summary.json",summary)
    print(json.dumps({"decision_counts":counts,"ljung_box_holm60":diagnostic_counts,"L0_max_abs_diff":max(internal["L0_numeric_max_abs_diff"].values())},ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()

# Current numeric-run provenance is separate from the preserved Nov10 method plan.
_record = {"status":"PORTABLE_RETURN_HAC_REPLAY_COMPLETE", "timestamp_utc":datetime.now(timezone.utc).isoformat(),
    "script_sha256":digest(Path(__file__)), "source_sha256":digest(REFERENCE / "source/run_analysis.py"),
    "no_new_training":True, "no_new_inference":True, "no_new_SHAP_sampling":True,
    "archive_root":str(ARCHIVE), "output_dir":str(HERE),
    "inputs_root":str(INPUTS), "path_changes_record":"post_review/path_relocations.json"}
save_json("portable_run_record.json", _record)
