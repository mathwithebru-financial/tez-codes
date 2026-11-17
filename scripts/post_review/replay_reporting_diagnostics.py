# PORTABLE DERIVATIVE: archived Nov10 numerical formulas retained.
# Path/CLI/output/provenance changes are enumerated in post_review/path_relocations.json.
# Does not train models, execute checkpoints, recompute SHAP sampling or download data.
"""Read frozen artifacts; derive reporting diagnostics without model fitting."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

import argparse
_parser = argparse.ArgumentParser(description="Frozen reporting/SHAP diagnostics only; no fitting or SHAP sampling")
_parser.add_argument("--archive-root", type=Path, default=Path(__file__).resolve().parent.parent)
_parser.add_argument("--output-dir", type=Path)
_args = _parser.parse_args()
ROOT = _args.archive_root.resolve()
OUT = (_args.output_dir or ROOT / "verification/post_review_reporting").resolve()
if OUT == ROOT or ROOT.is_relative_to(OUT) or any(OUT.is_relative_to(ROOT / d) for d in ("main_project", "h2_project", "post_review", "tools")):
    raise RuntimeError("Use a new output directory outside immutable source/input folders")
OUT.mkdir(parents=True, exist_ok=False)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


assets = ["BIST100", "USDTRY", "EURTRY", "GOLD"]
raw_path = ROOT / "main_project/data/raw/raw_prices.csv"
raw_rows = read_csv(raw_path)
dates = [r["Date"] for r in raw_rows]
raw = np.array([[float(r[a]) if r[a] else np.nan for a in assets] for r in raw_rows])
clean = raw.copy()
for i in range(1, len(clean)):
    absent = np.isnan(clean[i])
    clean[i, absent] = clean[i-1, absent]
assert np.isfinite(clean).all()
ret = np.full_like(clean, np.nan)
ret[1:] = np.log(clean[1:] / clean[:-1])
vol = np.full_like(clean, np.nan)
for i in range(20, len(clean)):
    vol[i] = np.std(ret[i-19:i+1], axis=0, ddof=1) * np.sqrt(252)
target_positions = np.arange(len(raw)-584, len(raw))
anchors = target_positions-1
target_dates = [dates[i] for i in target_positions]
anchor_dates = [dates[i] for i in anchors]
truth = np.load(ROOT / "main_project/results/final_test/final_test_y_true_raw_v4.npy", allow_pickle=False)
assert np.array_equal(np.column_stack([ret[target_positions], vol[target_positions]]).astype("float32"), truth)
scaler = json.loads((ROOT / "post_review/reporting/inputs/training_scaler_values.json").read_text(encoding="utf-8"))
shap_path = ROOT / "main_project/results/shap_analysis/10_shap_values_ensemble_raw_v4.npz"
assert hashlib.sha256(shap_path.read_bytes()).hexdigest() == "2bf3f66463ce14df9109fc83b55f0c3423930f9a73c1bcc760774f40cf05bcfd"
with np.load(shap_path, allow_pickle=False) as z:
    phi = z["shap_raw"].astype("float64")
    pred = z["official_predictions_raw"].copy()
    base = z["expected_raw"].copy()
    residual = z["additivity_residual_raw"].copy()
    feature_names = z["feature_names"].tolist()
    target_names = z["target_names"].tolist()
    lag_names = z["lag_names"].tolist()
    local_indices = z["local_sample_indices"].tolist()
assert np.array_equal(pred, np.load(ROOT / "main_project/results/final_test/pred_final_ensemble_raw_v4.npy", allow_pickle=False))
assert np.array_equal(pred - (base + phi.sum(axis=(1, 2))), residual)
assert phi.shape == (584, 10, 8, 8)

diag = []
for j, name in enumerate(target_names):
    ae = np.abs(residual[:, j])
    sd = float(scaler["y_scale"][j])
    diag.append({"target": name, "n": 584, "train_target_sd": sd,
                 "mean_abs_residual": float(ae.mean()), "max_abs_residual": float(ae.max()),
                 "rmse_residual": float(np.sqrt(np.mean(residual[:, j]**2))),
                 "mean_abs_residual_per_train_sd": float(ae.mean()/sd),
                 "max_abs_residual_per_train_sd": float(ae.max()/sd)})
write_csv("shap_output_diagnostics.csv", diag)

features = np.empty((len(raw), 8))
features[:, 0::2], features[:, 1::2] = ret, vol
local_summary, local_terms = [], []
for i in local_indices:
    for j in (0, 4):  # same BIST asset, one return and one volatility output
        flat = phi[i, :, :, j].reshape(-1)
        chosen = np.argsort(-np.abs(flat), kind="stable")[:5]
        top_sum = float(flat[chosen].sum())
        local_summary.append({"test_index_zero_based": i, "anchor_date": anchor_dates[i],
                              "target_date": target_dates[i], "target": target_names[j],
                              "truth": float(truth[i, j]), "prediction": float(pred[i, j]),
                              "background": float(base[j]), "all_contributions": float(flat.sum()),
                              "top5_contributions": top_sum,
                              "other75_contributions": float(flat.sum()-top_sum),
                              "residual_prediction_minus_reconstruction": float(residual[i, j])})
        for rank, k in enumerate(chosen, 1):
            lag, feat = divmod(int(k), 8)
            raw_position = anchors[i] - 9 + lag
            local_terms.append({"test_index_zero_based": i, "target_date": target_dates[i],
                                "target": target_names[j], "rank": rank,
                                "feature": feature_names[feat], "lag": lag_names[lag],
                                "feature_date": dates[raw_position],
                                "feature_value_raw": float(features[raw_position, feat]),
                                "signed_contribution": float(flat[k])})
write_csv("shap_local_summary.csv", local_summary)
write_csv("shap_local_top5.csv", local_terms)

metric_rows = []
for r in read_csv(ROOT / "post_review/reporting/inputs/naive_metrics.csv"):
    if r["model_label"] != "FinalWinner_3SeedEnsemble":
        metric_rows.append({"model":r["model_label"], "task":r["task"], "asset":r["asset"],
                            **{m:float(r[m]) if r[m] else None for m in ["MAE","RMSE","R2","PinballLoss_tau_0.5"]}})
for r in read_csv(ROOT / "post_review/reporting/inputs/learned_metrics.csv"):
    if r["prediction_type"] == "ensemble_primary":
        metric_rows.append({"model":r["model"], "task":r["task"], "asset":r["asset"],
                            **{m:float(r[m]) if r[m] else None for m in ["MAE","RMSE","R2","PinballLoss_tau_0.5"]}})
for j, name in enumerate(target_names):
    e = pred[:, j]-truth[:, j].astype("float64")
    metric_rows.append({"model":"FinalWinner_3SeedEnsemble", "task":"return" if j<4 else "volatility",
                        "asset":assets[j%4], "MAE":float(np.abs(e).mean()),
                        "RMSE":float(np.sqrt(np.mean(e**2))),
                        "R2":float(1-np.sum(e**2)/np.sum((truth[:,j]-truth[:,j].mean(dtype="float64"))**2)),
                        "PinballLoss_tau_0.5":None if j<4 else float(.5*np.abs(e).mean())})
assert len(metric_rows)==44, (len(metric_rows),[(r["model"],r["task"]) for r in metric_rows])
write_csv("absolute_asset_metrics.csv", metric_rows)
macro=[]
for model in dict.fromkeys(r["model"] for r in metric_rows):
    for task in ("return","volatility"):
        rr=[r for r in metric_rows if r["model"]==model and r["task"]==task]
        if not rr:continue
        assert len(rr)==4
        metric="MAE" if task=="return" else "PinballLoss_tau_0.5"
        macro.append({"model":model,"task":task,"asset_count":4,"metric":metric,
                      "macro_loss":float(np.mean([r[metric] for r in rr]))})
write_csv("absolute_task_metrics.csv", macro)

effects=[]
for asset, model in [("USDTRY","XGBoost"),("EURTRY","SingleTaskTransformer")]:
    r=next(r for r in metric_rows if r["model"]==model and r["task"]=="volatility" and r["asset"]==asset)
    f=next(r for r in metric_rows if r["model"]=="FinalWinner_3SeedEnsemble" and r["task"]=="volatility" and r["asset"]==asset)
    effects.append({"asset":asset,"comparator":model,"model_pinball":f["PinballLoss_tau_0.5"],
                    "reference_pinball":r["PinballLoss_tau_0.5"],
                    "relative_reduction_percent":100*(r["PinballLoss_tau_0.5"]-f["PinballLoss_tau_0.5"])/r["PinballLoss_tau_0.5"]})
write_csv("reference_relative_reductions.csv",effects)
report={"method":"Reanalysis of frozen predictions and SHAP attribution archive; no refitting or new SHAP sampling",
        "date":"2026-11-10", "shap_archive_sha256":hashlib.sha256(shap_path.read_bytes()).hexdigest(),
        "normalization":"absolute SHAP additivity residual divided by corresponding training target standard deviation",
        "residual_definition":"prediction minus background minus sum of all signed contributions",
        "local_indices":local_indices,"selected_outputs":[target_names[0],target_names[4]],
        "diagnostics":diag,"macro_metrics":macro,"reference_relative_reductions":effects,
        "limits":["Normalized residuals are descriptive, with no post-hoc acceptance threshold",
                  "No new Monte Carlo or alternative-background stability experiment was conducted",
                  "Macro means are descriptive and are not additional significance tests"]}
(OUT/"reporting_analysis.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"status":"PASS","asset_metric_rows":len(metric_rows),"diagnostic_rows":len(diag),
                  "local_examples":len(local_summary),"local_contribution_rows":len(local_terms),
                  "macro_metrics":macro,"relative_reductions":effects},ensure_ascii=False,indent=2))

import platform, sys, scipy
from datetime import datetime, timezone
_inputs = [raw_path, shap_path, ROOT / "main_project/results/final_test/final_test_y_true_raw_v4.npy",
    ROOT / "main_project/results/final_test/pred_final_ensemble_raw_v4.npy",
    ROOT / "post_review/reporting/inputs/training_scaler_values.json",
    ROOT / "post_review/reporting/inputs/naive_metrics.csv", ROOT / "post_review/reporting/inputs/learned_metrics.csv"]
_record={"status":"PORTABLE_FROZEN_REPORTING_REPLAY_COMPLETE", "timestamp_utc":datetime.now(timezone.utc).isoformat(),
    "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "source_sha256":hashlib.sha256((ROOT / "post_review/reporting/source/analyze_existing_outputs.py").read_bytes()).hexdigest(),
    "environment":{"python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__,"platform":platform.platform()},
    "inputs":[{"path":str(p.relative_to(ROOT)).replace("\\","/"),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"size_bytes":p.stat().st_size} for p in _inputs],
    "training_scaler_values":"Frozen scalar values extracted earlier without executing the scaler pickle; treated here as numeric inputs. No scaler payload is opened by this helper.",
    "no_new_training":True,"no_new_inference":True,"no_new_SHAP_sampling":True,
    "path_changes_record":"post_review/path_relocations.json"}
(OUT / "portable_run_record.json").write_text(json.dumps(_record,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
