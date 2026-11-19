"""Print-table exports for thesis Chapters 4 and 5, from the frozen result records."""
import json
from collections import Counter
from functools import partial
from pathlib import Path
from statistics import mean, stdev

from table_data import number, records, scientific


MAIN = Path("main_project/results")
DM = MAIN / "statistical_comparison/09_dm_harvey_holm_results_v4.csv"
HAC = MAIN / "robustness/bartlett_dm_hln_v1"
H2 = Path("h2_project/original_period/runs/original_test_v1")
SUBPERIOD = Path("thesis_rebuild_20261113/training/runs/p1_v2_mse_mae_20261113_full/subperiods_20261113")
TASK = {"return": "Getiri", "volatility": "Volatilite"}
ASSET = {"BIST100": "BIST100", "USDTRY": "USD/TRY", "EURTRY": "EUR/TRY", "GOLD": "Altın"}
MODEL = {
    "ReturnZero": "Sıfır getiri", "ReturnPersistence": "Getiri sürekliliği",
    "VolPersistence": "Volatilite sürekliliği", "SingleTaskLSTM": "Tek görevli LSTM",
    "SingleTaskTransformer": "Tek görevli Transformer", "XGBoost": "XGBoost",
    "GARCH_1_1_StudentsT_ZeroMean": "GARCH(1,1)", "GJR_GARCH_1_1_StudentsT_ZeroMean": "GJR-GARCH(1,1)",
}
ARM = {"NS_base": "NoSharing", "FS_same_width": "Tam paylaşım (aynı genişlik)",
       "FS_budget_control": "Tam paylaşım (parametre sayısı kontrolü)", "STL_pair": "Bağımsız görev modelleri (STL)"}
DECISION = ("SIGNIFICANT_FINAL_BETTER", "SIGNIFICANT_BASELINE_BETTER", "NOT_SIGNIFICANT")
DECISION_LABEL = ("Nihai model anlamlı biçimde üstün", "Kıyaslama modeli anlamlı biçimde üstün", "Anlamlı fark bulunmadı")


def dm_table(root, family):
    columns = ["Varlık", "Kıyaslama modeli", "ΔL", "DM–HLN", "Holm p", "Karar"]
    panels = family != "garch_family"
    result = [["Görev", *columns] if panels else columns]
    winner = {"learned": "Tek görevli", "naive": "Naif model", "garch_family": "GARCH ailesi"}[family]
    decisions = dict(zip(DECISION, ("Nihai model", winner, "Anlamlı fark yok")))
    short = {"SingleTaskLSTM": "ST-LSTM", "SingleTaskTransformer": "STT"}
    for row in records(root / DM):
        if row["baseline_family"] != family:
            continue
        model = row["baseline_model"]
        p = float(row["holm_adjusted_p"])
        cells = [ASSET[row["asset"]], short.get(model, MODEL[model]), number(row["mean_loss_differential"]),
                 number(row["dm_hln_statistic"], 3), "< 0,001" if p < .001 else number(p, 4),
                 decisions[row["interpretation"]]]
        result.append([TASK[row["task"]], *cells] if panels else cells)
    return result


def shap_summary(root):
    folder = root / MAIN / "shap_analysis"
    global_rows = records(folder / "10_shap_global_importance_v4.csv")
    recent = {r["output_name"]: r for r in records(folder / "10_shap_temporal_importance_v4.csv")
              if r["lag_name"] == "t"}
    result = [["Görev", "Varlık", "Baskın özellik", "Payı", "Volatilite göstergeleri",
               "Getiri göstergeleri", "En güncel adım (t)"]]
    for target, time_row in sorted(recent.items(), key=lambda item: int(item[1]["output_index"])):
        features = [r for r in global_rows if r["output_name"] == target]
        best = max(features, key=lambda r: float(r["normalized_share_within_output"]))
        asset, kind = best["feature_name"].split("_")
        label = ASSET[asset] + " " + {"LogRet": "getirisi", "Vol20": "volatilitesi"}[kind]
        shares = [float(best["normalized_share_within_output"])]
        for suffix in ("_Vol20", "_LogRet"):
            shares.append(sum(float(r["normalized_share_within_output"]) for r in features
                              if r["feature_name"].endswith(suffix)))
        shares.append(float(time_row["normalized_share_within_output"]))
        result.append([TASK[time_row["task"]], ASSET[target.split("_")[0]], label,
                       *[number(share * 100, 2) for share in shares]])
    return result


def seed_validation(root):
    folder = root / MAIN / "multiseed"
    winner = json.loads((folder / "multiseed_winner_config_v4.json").read_text(encoding="utf-8"))
    runs = [r for r in records(folder / "multiseed_runs_v4.csv") if r["config_id"] == winner["winner"]["config_id"]]
    runs.sort(key=lambda r: int(r["seed"]))
    keys = ("avg_return_ratio", "avg_vol_ratio", "validation_score")
    result = [["Rastgelelik tohumu", "Ortalama getiri hata oranı", "Ortalama volatilite hata oranı",
               "Birleşik doğrulama skoru", "Seçilen dönem"]]
    for row in runs:
        result.append([row["seed"], *[number(row[k]) for k in keys], row["best_epoch"]])
    for name, statistic in (("Ortalama", mean), ("Standart sapma", stdev)):
        result.append([name, *[number(statistic(float(r[k]) for r in runs)) for k in keys], "-"])
    return result


def decision_counts(rows, field):
    counts = Counter(r[field] for r in rows)
    if counts.keys() - set(DECISION):
        raise ValueError(f"Unknown decision in {field}: {set(counts) - set(DECISION)}")
    return [str(len(rows)), *[str(counts[d]) for d in DECISION]]


def task_decisions(root):
    rows = records(root / DM)
    result = [["Görev", "Toplam", *DECISION_LABEL]]
    for task, turkish in TASK.items():
        result.append([turkish, *decision_counts([r for r in rows if r["task"] == task], "interpretation")])
    result.append(["Toplam", *decision_counts(rows, "interpretation")])
    return result


def hac_decisions(root):
    rows = records(root / HAC / "bartlett_dm_hln_results_v1.csv")
    return [["Analiz", "Toplam", *DECISION_LABEL],
            ["Ana DM–HLN analizi (L = 0)", *decision_counts(rows, "stage09_decision")],
            ["Bartlett-HAC sağlamlık analizi (L = 19)", *decision_counts(rows, "bartlett_decision")]]


def hac_changes(root):
    result = [["Varlık", "Kıyaslama modeli", "Ana analiz kararı",
               "Holm p (Bartlett-HAC, L = 19)", "Sağlamlık analizi kararı"]]
    rows = records(root / HAC / "bartlett_dm_hln_changed_decisions_v1.csv")
    for r in rows:
        result.append([ASSET[r["asset"]], MODEL[r["baseline_model"]], r["stage09_decision_tr"],
                       number(r["holm_adjusted_p"], 4), r["bartlett_decision_tr"]])
    return result


def h2_loss(root):
    rows = {(r["arm"], r["task"]): r for r in records(root / H2 / "primary_losses_seed_summary_v1.csv")
            if r["asset"] == "four_asset_mean"}
    result = [["Mimari", "Getiri MAE", "Volatilite pinball"]]
    for arm, name in ARM.items():
        cells = [name]
        for task in TASK:
            row = rows[arm, task]
            cells.append(number(row["seed_mean_primary_loss"]) + " ± " + number(row["seed_sample_sd_primary_loss"]))
        result.append(cells)
    return result


def h2_gaps(root):
    rows = records(root / H2 / "generalization_gaps_seed_summary_v1.csv")
    result = [["Mimari", "Görev", "Eğitim", "Doğrulama", "Test", "Test eksi eğitim"]]
    for r in rows:
        result.append([ARM[r["arm"]], TASK[r["task"]], number(r["train_primary_loss_seed_mean"]),
                       number(r["validation_primary_loss_seed_mean"]), number(r["test_primary_loss_seed_mean"]),
                       number(r["test_minus_train_gap_seed_mean"])])
    return result


def h2_comparisons(root):
    result = [["Kontrol", "Paylaşım modeli", "Görev", "Ortalama kayıp farkı", "Holm p", "Düşük kayıp"]]
    for r in records(root / H2 / "dm_holm_comparisons_v1.csv"):
        if r["analysis"] != "fixed_original_period_supplement":
            continue
        control = "STL" if r["control"] == "STL_pair" else ARM[r["control"]]
        sharing = ARM[r["sharing_arm"]]
        lower = control if float(r["control_mean_loss"]) < float(r["sharing_mean_loss"]) else sharing
        result.append([control, sharing, TASK[r["task"]], number(r["mean_control_minus_sharing_loss"]),
                       scientific(r["holm_adjusted_p"]), lower])
    return result


def p1_subperiods(root):
    rows = {(r["period"], r["task"], r["model"]): r for r in records(root / SUBPERIOD / "metrics_equal_asset_macro.csv")
            if r["population"] == "primary"}
    periods = {"full": "Tam dönem", "2022_partial": "2022 (kısmi)", "2023": "2023", "2024": "2024"}
    result = [["Dönem", "Görev", "n", "MSE kolu", "MAE kolu", "Naif", "Fark (%)"]]
    for period, name in periods.items():
        for task, metric in (("return", "MAE"), ("volatility", "Pinball_tau05")):
            group = [rows[period, task, model] for model in ("mse_ensemble", "mae_ensemble", "matched_naive")]
            counts = {part.split(":")[1] for r in group for part in r["n_by_asset"].split(";")}
            if len(counts) != 1:
                raise ValueError(f"Primary target counts differ for {period}, {task}")
            mse, mae, naive = [float(r[metric]) for r in group]
            difference = 100 * (mae - mse) / mse
            result.append([name, TASK[task], counts.pop(), number(mse), number(mae), number(naive),
                           number(difference, 2).replace("-", "−")])
    return result


def relative_losses(root):
    # The printed table selects these comparisons in this order from the 42-test family.
    selected = [
        ("return", "BIST100", "ReturnPersistence"), ("return", "GOLD", "ReturnPersistence"),
        ("volatility", "USDTRY", "XGBoost"), ("return", "EURTRY", "ReturnPersistence"),
        ("volatility", "EURTRY", "SingleTaskTransformer"), ("volatility", "USDTRY", "VolPersistence"),
        ("volatility", "BIST100", "GJR_GARCH_1_1_StudentsT_ZeroMean"),
        ("volatility", "BIST100", "GARCH_1_1_StudentsT_ZeroMean"), ("volatility", "USDTRY", "SingleTaskLSTM"),
        ("volatility", "GOLD", "GARCH_1_1_StudentsT_ZeroMean"), ("volatility", "GOLD", "GJR_GARCH_1_1_StudentsT_ZeroMean"),
        ("return", "USDTRY", "SingleTaskTransformer"), ("return", "EURTRY", "SingleTaskTransformer"),
    ]
    dm = {(r["task"], r["asset"], r["baseline_model"]): r for r in records(root / DM)}
    final, = (r for r in records(root / MAIN / "final_test/final_test_summary_v4.csv")
              if r["model_label"] == "FinalWinner_3SeedEnsemble")
    macro = {"return": float(final["avg_return_mae"]), "volatility": float(final["avg_vol_pinball_tau_0.5"])}
    result = [["Görev", "Karşılaştırma", "ΔL", "Görev ortalaması kaybına oranı (%)", "Kıyaslamaya göre azalma (%)"]]
    for task, asset, model in selected:
        row = dm[task, asset, model]
        delta = float(row["mean_loss_differential"])
        signed = ("+" if delta >= 0 else "−") + number(abs(delta))
        result.append([TASK[task], ASSET[asset] + " · " + MODEL[model], signed,
                       number(100 * abs(delta) / macro[task], 1), number(row["relative_loss_reduction_pct"], 2).replace("-", "−")])
    return result


TABLES = {
    "4.1": partial(dm_table, family="learned"), "4.2": partial(dm_table, family="naive"),
    "4.3": partial(dm_table, family="garch_family"), "4.4": shap_summary,
    "4.5": seed_validation, "4.6": task_decisions, "4.7": hac_decisions, "4.8": hac_changes,
    "4.9": h2_loss, "4.10": h2_gaps, "4.11": h2_comparisons, "4.12": p1_subperiods,
    "5.1": relative_losses,
}
