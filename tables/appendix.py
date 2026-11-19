"""Ek D ve E çizelgeleri. Girdi yolları açılmış P0/H2 veya P1 arşivine göredir."""

import json
import math
from collections import Counter
from functools import partial
from pathlib import Path

from table_data import SUPERSCRIPTS, number, records, scientific


H2 = Path("h2_project/original_period/runs/original_test_v1")
REPORT = Path("post_review/reporting/reference")
HAC = Path("post_review/return_robustness/reference")
P1 = Path("thesis_rebuild_20261113")
RUN = P1 / "training/runs/p1_v2_mse_mae_20261113_full"
ASSETS = {"BIST100": "BIST100", "USDTRY": "USD/TRY", "EURTRY": "EUR/TRY", "GOLD": "Altın"}
P1_ASSETS = {**ASSETS, "GOLD": "Vadeli altın"}
TASKS = {"return": "Getiri", "volatility": "Volatilite"}
ARMS = {"NS_base": "NoSharing", "FS_same_width": "Tam paylaşım (aynı genişlik)",
        "FS_budget_control": "Tam paylaşım (parametre sayısı kontrolü)",
        "STL_pair": "Bağımsız görev modelleri (STL)"}
MODELS = {"ReturnZero": "Sıfır getiri", "ReturnPersistence": "Getiri sürekliliği",
          "VolPersistence": "Volatilite sürekliliği", "SingleTaskTransformer": "Tek görevli Transformer",
          "SingleTaskLSTM": "Tek görevli LSTM", "XGBoost": "XGBoost",
          "FinalWinner_3SeedEnsemble": "Nihai NoSharing topluluğu"}
SHORT = {**MODELS, "SingleTaskTransformer": "STT", "SingleTaskLSTM": "ST-LSTM"}


def probability(value):
    return "< 0,0001" if float(value) < 0.0001 else number(value, 4)


def d1(root):
    def key_of(row):
        return row["arm"], row["seed"], row["task"]

    epochs = {key_of(r): r["selected_epoch"] for r in records(root / H2 / "learning_curves_v1.csv")}
    losses = {key_of(r): r for r in records(root / H2 / "generalization_gaps_descriptive_v1.csv")}
    rows = [["Mimari", "Tohum", "Görev", "Seçilen dönem", "Eğitim", "Test", "Test eksi eğitim"]]
    for key in sorted(losses, key=lambda k: (list(ARMS).index(k[0]), int(k[1]), k[2])):
        arm, seed, task = key
        rec = losses[key]
        rows.append([ARMS[arm], seed, TASKS[task], epochs[key], number(rec["train_primary_loss"]),
                     number(rec["test_primary_loss"]), number(rec["test_minus_train_gap"])])
    return rows


def d2(root):
    rows = [["Kontrol", "Paylaşım modeli", "Görev", "DM–HLN", "Holm p", "Düşük kayıp"]]
    for r in records(root / H2 / "dm_holm_comparisons_v1.csv"):
        if r["analysis"] == "lag0_sensitivity":
            control = {"NS_base": "NoSharing", "STL_pair": "STL"}[r["control"]]
            model = ARMS[r["sharing_arm"]]
            winner = control if float(r["control_mean_loss"]) < float(r["sharing_mean_loss"]) else model
            rows.append([control, model, TASKS[r["task"]], number(r["dm_hln_t"]),
                         scientific(r["holm_adjusted_p"]), winner])
    return rows


def d3(root):
    report = json.loads((root / H2 / "analysis_report_v1.json").read_text(encoding="utf-8"))
    env = report["runtime_environment"]
    libs = env["packages"]
    replay = report["train_validation_replay"]
    if not (env["determinism"]["deterministic_algorithms"] and not env["determinism"]["amp_enabled"]
            and "float32" in report["precision"] and "float64 before pointwise subtraction" in report["precision"]):
        raise ValueError("H2 çıkarım veya kayıp hesabı kaydı beklenen protokolden farklı.")
    def power_of_ten(value):
        exponent = round(math.log10(value))
        if 10.0 ** exponent != value:
            raise ValueError(f"Onun kuvveti biçiminde olmayan tolerans: {value}")
        return "10" + str(exponent).translate(SUPERSCRIPTS)
    return [["Bileşen", "Doğrulanmış değer"], ["Python", env["python_version"]], ["GPU", env["gpu_name"]],
            ["PyTorch", libs["torch"]], ["NumPy ve pandas", f"{libs['numpy']} ve {libs['pandas']}"],
            ["SciPy ve scikit-learn", f"{libs['scipy']} ve {libs['scikit-learn']}"],
            ["CUDA çalışma zamanı", env["cuda_runtime"]],
            ["cuDNN ve NVIDIA sürücüsü", f"{env['cudnn_version']} ve {env['driver_version']}"],
            ["Model çıkarımı", "FP32; deterministik algoritmalar; AMP kapalı"],
            ["Görev kayıplarının hesabı", "Ham hedef ve tahmin FP64’e çevrildikten sonra noktasal kayıp ve ortalama"],
            ["Eğitim/doğrulama sayısal toleransı",
             f"Göreli {power_of_ten(replay['relative_tolerance'])} ve mutlak {power_of_ten(replay['absolute_tolerance'])}"]]


def e1(root):
    loss = {(r["model"], r["task"]): number(r["macro_loss"])
            for r in records(root / REPORT / "absolute_task_metrics.csv")}
    rows = [["Model veya görev referansı", "Getiri MAE", "Volatilite pinball"]]
    for ret, vol in (("FinalWinner_3SeedEnsemble", "FinalWinner_3SeedEnsemble"),
                     ("ReturnZero", "VolPersistence"), ("ReturnPersistence", None),
                     ("SingleTaskTransformer", "SingleTaskTransformer"),
                     ("SingleTaskLSTM", "SingleTaskLSTM"), ("XGBoost", "XGBoost")):
        label = MODELS[ret] if ret != "ReturnZero" else "Sıfır getiri / volatilite sürekliliği"
        rows.append([label, loss[ret, "return"], loss[vol, "volatility"] if vol else "—"])
    return rows


def asset_metrics(root, task):
    metric = "MAE" if task == "return" else "PinballLoss_tau_0.5"
    rows = [["Varlık", "Model", "MAE" if task == "return" else "Pinball (0,5)", "RMSE", "R²"]]
    data = {(r["asset"], r["model"]): r for r in records(root / REPORT / "absolute_asset_metrics.csv")
            if r["task"] == task}
    models = ["ReturnZero", "ReturnPersistence"] if task == "return" else ["VolPersistence"]
    models.extend(["SingleTaskTransformer", "SingleTaskLSTM", "XGBoost", "FinalWinner_3SeedEnsemble"])
    for asset in ASSETS:
        for model in models:
            r = data[asset, model]
            rows.append([ASSETS[asset], MODELS[model], number(r[metric]), number(r["RMSE"]), number(r["R2"], 4)])
    return rows


def e4(root):
    rows = [["Varlık", "Kıyaslama", "Nihai pinball", "Kıyaslama pinball", "Azalma (%)"]]
    for r in records(root / REPORT / "reference_relative_reductions.csv"):
        rows.append([ASSETS[r["asset"]], MODELS[r["comparator"]], number(r["model_pinball"]),
                     number(r["reference_pinball"]), number(r["relative_reduction_percent"], 2)])
    return rows


def target_label(target):
    asset, task = target.split("_")
    return asset + {"NextRet": " getiri", "NextVol": " vol."}[task]


def e5(root):
    rows = [["Çıktı", "Eğitim σ", "Ort. |artık|", "En büyük |artık|", "Ort. |artık| / σ", "En büyük |artık| / σ"]]
    for r in records(root / REPORT / "shap_output_diagnostics.csv"):
        rows.append([target_label(r["target"]), number(r["train_target_sd"]), number(r["mean_abs_residual"]),
                     number(r["max_abs_residual"]), number(r["mean_abs_residual_per_train_sd"], 4),
                     number(r["max_abs_residual_per_train_sd"], 4)])
    return rows


def local_contributions(root, target):
    rows = [["Bileşen", "Konum", "Girdi tarihi", "İşaretli katkı"]]
    top = [r for r in records(root / REPORT / "shap_local_top5.csv")
           if r["test_index_zero_based"] == "0" and r["target"] == target]
    for r in sorted(top, key=lambda r: int(r["rank"])):
        rows.append([r["feature"], r["lag"], r["feature_date"], number(r["signed_contribution"], 8)])
    summary, = [r for r in records(root / REPORT / "shap_local_summary.csv")
                if r["test_index_zero_based"] == "0" and r["target"] == target]
    rows.extend([["Diğer 75 katkının toplamı", "—", "—", number(summary["other75_contributions"], 8)],
                 ["Toplamsallık artığı", "—", "—", number(summary["residual_prediction_minus_reconstruction"], 8)]])
    return rows


def e8(root):
    rows = [["Konum ve hedef tarihi", "Çıktı", "Tahmin", "Arka plan", "Katkılar toplamı", "Artık"]]
    target_names = {"BIST100_NextRet": "Getiri", "BIST100_NextVol": "Volatilite"}
    for r in records(root / REPORT / "shap_local_summary.csv"):
        rows.append([f"{r['test_index_zero_based']} / {r['target_date']}", target_names[r["target"]],
                     number(r["prediction"], 7), number(r["background"], 7), number(r["all_contributions"], 7),
                     number(r["residual_prediction_minus_reconstruction"], 7)])
    return rows


def e9(root):
    keys = ["bandwidth", "n_comparisons", "final_better", "baseline_better", "not_significant", "changed_vs_L0"]
    rows = [["Bartlett L", "Karşılaştırma", "Nihai üstün", "Kıyaslama üstün", "Anlamlı fark yok", "L0’dan değişen"]]
    return rows + [[r[k] for k in keys] for r in records(root / HAC / "decision_counts.csv")]


def e10(root):
    lags = (0, 3, 6, 10, 19)
    rows = [["Varlık", "Kıyaslama", "ΔL"] + [f"p L{i}" for i in lags]]
    for r in records(root / HAC / "pvalue_matrix.csv"):
        rows.append([ASSETS[r["asset"]], SHORT[r["baseline_model"]], number(r["mean_loss_differential"])]
                    + [probability(r[f"holm_p_L{i}"]) for i in lags])
    return rows


def e11(root):
    rows = [["Varlık", "Kıyaslama", "ACF 1", "En büyük |ACF|", "LB5 pH", "LB10 pH", "LB20 pH"]]
    for r in records(root / HAC / "dependence_diagnostics.csv"):
        rows.append([ASSETS[r["asset"]], SHORT[r["baseline_model"]], number(r["acf1"], 3),
                     number(r["max_abs_acf_1_20"], 3)] + [probability(r[f"lb{i}_holm_p_60"]) for i in (5, 10, 20)])
    return rows


def e12(root):
    """Score fixed P0 forecasts when both return prices or all 21 Vol20 prices are observed."""
    import numpy as np

    project = root / "main_project"
    prices = records(project / "data/raw/raw_prices.csv")
    positions = {r["Date"]: i for i, r in enumerate(prices)}
    dates = np.load(project / "data/sequences/baseline/lb10/target_realization_dates_test.npy", allow_pickle=False)
    indices = [positions[str(d)[:10]] for d in dates]
    final = project / "results/final_test"
    naive = project / "results/baselines/naive"
    truth = np.load(final / "final_test_y_true_raw_v4.npy", allow_pickle=False)
    forecast = np.load(final / "pred_final_ensemble_raw_v4.npy", allow_pickle=False)
    reference = np.column_stack([np.load(naive / name, allow_pickle=False)
                                 for name in ("pred_return_zero_raw_v4.npy", "pred_vol_persistence_raw_v4.npy")])
    if min(indices) < 20 or truth.shape != (len(indices), 8) or truth.shape != forecast.shape or truth.shape != reference.shape:
        raise ValueError("P0 hedef tarihleri ve sekiz sütunlu tahmin dizileri eşleşmiyor.")
    rows = [["Hedef", "Kalan n", "Kapsam (%)", "Nihai kayıp", "Naif kayıp", "Filtreli oran", "Tüm testte oran"]]
    for j, asset in enumerate(list(ASSETS) * 2):
        width = 2 if j < 4 else 21
        mask = [all(r[asset] and math.isfinite(float(r[asset])) and float(r[asset]) > 0
                    for r in prices[i - width + 1:i + 1]) for i in indices]
        factor = 1 if j < 4 else 0.5
        loss = factor * np.abs(truth[:, j] - forecast[:, j])
        ref_loss = factor * np.abs(truth[:, j] - reference[:, j])
        avg, ref_avg = loss[mask].mean(), ref_loss[mask].mean()
        n = sum(mask)
        rows.append([asset + (" getiri" if j < 4 else " vol."), str(n), number(100 * n / len(dates), 2),
                     number(avg), number(ref_avg), number(avg / ref_avg, 4), number(loss.mean() / ref_loss.mean(), 4)])
    return rows


def e13(root):
    models = {"GARCH_1_1_StudentsT_ZeroMean": "GARCH", "GJR_GARCH_1_1_StudentsT_ZeroMean": "GJR-GARCH"}
    rows = [["Varlık", "Kıyaslama", "Nihai pinball", "Kıyaslama pinball", "n"]]
    for r in records(root / "main_project/results/statistical_comparison/09_dm_harvey_holm_results_v4.csv"):
        if r["baseline_model"] in models and r["task"] == "volatility" and r["n"] == "584":
            rows.append([ASSETS[r["asset"]], models[r["baseline_model"]], number(r["mean_final_loss"]),
                         number(r["mean_baseline_loss"]), r["n"]])
    return rows


def e14(root):
    folder = root / "advisor_followup_20261112/fill_audit"
    runs = Counter((r["asset"], int(r["n_consecutive_union_rows"])) for r in records(folder / "fill_runs.csv"))
    rows = [["Varlık", "Eksik / doldurulan", "Oran (%)", "Blok", "1", "2", "3", "4"]]
    for r in records(folder / "asset_summary.csv"):
        a = r["asset"]
        rows.append([P1_ASSETS[a], f"{r['raw_missing_rows']} / {r['forward_filled_rows']}",
                     number(r["raw_missing_pct_of_union_rows"], 3), r["missing_run_count"]]
                    + [str(runs[a, i]) for i in range(1, 5)])
    return rows


def e15(root):
    masks = {(r["asset"], r["task"]): r for r in records(root / P1 / "scoring/scoring_mask_counts.csv")
             if r["split"] == "test"}
    rows = [["Varlık", "Ana kapsam", "Hedef: getiri", "Hedef: Vol20", "Girdi ve hedef: getiri / Vol20"]]
    for asset, name in P1_ASSETS.items():
        ret, vol = masks[asset, "return"], masks[asset, "vol20"]
        if ret["primary_all_eligible_count"] != vol["primary_all_eligible_count"]:
            raise ValueError("Ana kapsam iki görevde farklı.")
        strict = "strict_all_model_inputs_plus_task_target_count"
        rows.append([name, ret["primary_all_eligible_count"], ret["secondary_raw_target_support_count"],
                     vol["secondary_raw_target_support_count"], f"{ret[strict]} / {vol[strict]}"])
    return rows


def e16(root):
    metrics = {(r["asset"], r["task"], r["model"]): r for r in records(root / RUN / "comparison/metrics_by_asset.csv")
               if r["split"] == "test" and r["population"] == "primary"}
    rows = [["Görev", "Varlık", "MSE kolu", "MAE kolu", "Naif"]]
    for task, name in TASKS.items():
        key = "MAE" if task == "return" else "Pinball_tau05"
        for asset, label in P1_ASSETS.items():
            rows.append([name, label] + [number(metrics[asset, task, arm][key])
                                         for arm in ("mse_ensemble", "mae_ensemble", "matched_naive")])
    return rows


def e17(root):
    data = records(root / RUN / "shap_stability_20261113/stability/task_factor_summaries.csv")
    stats = {(r["arm"], r["task"], r["factor"]): r["median"] for r in data
             if r["model_scope"] == "individual_seed" and r["metric"] == "spearman_midrank"}
    rows = [["Eğitim kolu", "Görev", "Arka plan", "Model tohumu", "MC tohumu"]]
    for arm in ("mse", "mae"):
        for task, name in TASKS.items():
            rows.append([arm.upper(), name] + [number(stats[arm, task, factor], 4)
                                               for factor in ("background", "model_seed", "integration_MC")])
    return rows


TABLES = {
    "D.1": d1, "D.2": d2, "D.3": d3,
    "E.1": e1, "E.2": partial(asset_metrics, task="return"), "E.3": partial(asset_metrics, task="volatility"),
    "E.4": e4, "E.5": e5, "E.6": partial(local_contributions, target="BIST100_NextRet"),
    "E.7": partial(local_contributions, target="BIST100_NextVol"), "E.8": e8,
    "E.9": e9, "E.10": e10, "E.11": e11, "E.12": e12, "E.13": e13,
    "E.14": e14, "E.15": e15, "E.16": e16, "E.17": e17,
}
