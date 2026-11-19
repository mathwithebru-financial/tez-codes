"""Özgün dönem arşivinden yöntem ve istatistiksel karşılaştırma çizelgeleri."""
from datetime import date
from hashlib import sha256
import json
from statistics import mean

from table_data import number, records


ASSETS = {"BIST100": "BIST100", "USDTRY": "USD/TRY", "EURTRY": "EUR/TRY", "GOLD": "Altın"}
MODEL = {
    "SingleTaskTransformer": "Tek görevli Transformer",
    "SingleTaskLSTM": "Tek görevli LSTM",
    "XGBoost": "XGBoost",
    "ReturnZero": "Sıfır getiri",
    "ReturnPersistence": "Getiri sürekliliği",
    "VolPersistence": "Volatilite sürekliliği",
    "GARCH_1_1_StudentsT_ZeroMean": "GARCH(1,1)",
    "GJR_GARCH_1_1_StudentsT_ZeroMean": "GJR-GARCH(1,1)",
}
DM = "main_project/results/statistical_comparison/09_dm_harvey_holm_results_v4.csv"
BARTLETT = "main_project/results/robustness/bartlett_dm_hln_v1/bartlett_dm_hln_results_v1.csv"


def metadata(path):
    return json.loads(path.read_bytes())


def integer(value):
    return f"{int(value):,}".replace(",", ".")


def adjusted_p(value):
    return "< 0,001" if float(value) < 0.001 else number(value, 4)


def persistence(root):
    metrics = records(root / "main_project/results/baselines/naive/naive_baseline_metrics_long_v4.csv")
    by_asset = {r["asset"]: r for r in metrics if r["model_label"] == "VolPersistence"}
    grid = [["Varlık", "Vol20t+1 ~ Vol20t için R²", "Volatilite sürekliliği pinball kaybı"]]
    r2, loss = [], []
    for asset, name in ASSETS.items():
        r2.append(float(by_asset[asset]["R2"]))
        loss.append(float(by_asset[asset]["PinballLoss_tau_0.5"]))
        grid.append([name, number(r2[-1], 4), number(loss[-1], 6)])
    return grid + [["Ortalama", number(mean(r2), 4), number(mean(loss), 6)]]


def raw_assets(root):
    project = root / "main_project"
    definition = metadata(project / "config/schema_v4.json")["data"]
    rows = records(project / "data/raw/raw_prices.csv")
    market = {"BIST100": "Hisse senedi endeksi", "USDTRY": "Döviz kuru",
              "EURTRY": "Döviz kuru", "GOLD": "Emtia vadeli işlemi"}
    names = dict(ASSETS, GOLD="Altın vadeli işlemi")
    grid = [["Varlık", "Veri sembolü", "Temsil ettiği piyasa", "Ham eksik gözlem"]]
    for asset in definition["assets"]:
        missing = sum(r[asset].strip().lower() in {"", "na", "nan"} for r in rows)
        grid.append([names[asset], definition["tickers"][asset], market[asset], str(missing)])
    return grid


def splits(root):
    split = metadata(root / "main_project/data/processed/split_meta_v4.json")
    def period(row, prefix):
        return " – ".join(date.fromisoformat(row[f"{prefix}_{bound}"]).strftime("%d.%m.%Y")
                          for bound in ("start", "end"))
    grid = [["Küme", "Örnek sayısı", "Girdi tarihi", "Hedef gerçekleşme tarihi"]]
    for key, label in (("train", "Eğitim"), ("validation", "Doğrulama"), ("test", "Test")):
        row = split[key]
        grid.append([label, integer(row["n_rows"]), period(row, "anchor"),
                     period(row, "target_realization")])
    return grid


def model_sizes(root):
    sizes = metadata(root / "main_project/config/schema_v4.json")["model_sizes"]
    grid = [["Model büyüklüğü", "Model boyutu d", "Dikkat başlığı h",
             "Kodlayıcı katmanı Ne", "İleri besleme boyutu dff"]]
    for size, label in (("small", "Küçük"), ("medium", "Orta"), ("large", "Büyük")):
        grid.append([label] + [str(sizes[size][k]) for k in ("d_model", "n_head", "n_layers", "d_ff")])
    return grid


def selected_baselines(root):
    project = root / "main_project"
    learned = project / "results/baselines/learned"
    selected = records(learned / "learned_baseline_selection_v4.csv")
    counts = {(r["model_family"], r["task"], r["config_id"]): r["neural_parameter_count"]
              for r in records(learned / "learned_baseline_parameter_report_v4.csv")}
    seq = metadata(project / "data/sequences/baseline/lb10/sequence_meta.json")
    tasks = {"return": "Getiri", "volatility": "Volatilite"}
    grid = [["Model", "Görev", "Seçilen yapılandırma", "Eğitilebilir parametre"]]
    for row in selected:
        model, task = row["model_family"], row["task"]
        config = json.loads(row["config_json"])
        if model == "SingleTaskTransformer":
            spec = ("d={d_model}, h={n_head}, Ne={n_layers}, dff={d_ff}".format(**config)
                    + f", L={seq['lookback']}, {len(seq['feature_columns'])} öznitelik")
        elif model == "SingleTaskLSTM":
            spec = "{hidden_size} gizli birim, {num_layers} katman".format(**config)
        elif model == "XGBoost":
            spec = f"derinlik {config['max_depth']}, öğrenme oranı {number(config['learning_rate'], 2)}"
        else:
            raise ValueError(f"Bilinmeyen model: {model}")
        count = counts[model, task, row["config_id"]]
        grid.append([MODEL[model], tasks[task], spec, integer(float(count)) if count else "Uygulanamaz"])
    return grid


def runtime(root):
    project = root / "main_project"
    config = project / "config"
    garch = metadata(config / "08C_garch_protocol_lock_v4.json")["environment"]
    shap = metadata(config / "10_shap_protocol_lock_v4.json")["runtime"]
    addendum = metadata(config / "10_shap_protocol_addendum_v4a.json")["runtime_at_addendum_lock"]
    learned = metadata(project / "results/baselines/learned/learned_baseline_summary_v4.json")
    pins = dict(line.split("==") for line in (project / "requirements.txt").read_text(
        encoding="utf-8-sig").splitlines() if "==" in line and not line.startswith("#"))
    return [
        ["Bileşen", "Doğrulanmış bilgi"], ["Bulut çalışma ortamı", "Google Colab"],
        ["GPU", "NVIDIA " + shap["cuda_device_at_lock"]], ["Python", garch["python"]],
        ["PyTorch", shap["torch"]], ["NumPy", garch["numpy"]], ["pandas", garch["pandas"]],
        ["SciPy", garch["scipy"]], ["XGBoost", learned["xgboost"]["version"]],
        ["arch", garch["arch"]], ["statsmodels", pins["statsmodels"]], ["SHAP", shap["shap"]],
        ["scikit-learn", addendum["sklearn"]],
    ]


def h2_architectures(root):
    h2 = root / "h2_project"
    arms = metadata(h2 / "t4/protocol_locked_t4_v1.json")["arms"]
    plan = metadata(h2 / "stl_original_period/protocol_plan_v1.json")
    controls = plan["design"]["new_control"]
    models = {"NS_base": ("NoSharing", "Yok"), "FS_same_width": ("Tam paylaşım (aynı genişlik)", "Tam"),
              "FS_budget_control": ("Tam paylaşım (parametre sayısı kontrolü)", "Tam")}
    grid = [["Mimari", "Parametre paylaşımı", "İleri besleme genişliği",
             "Toplam eğitilebilir parametre sayısı"]]
    for arm in arms:
        name, sharing = models[arm["id"]]
        grid.append([name, sharing, str(arm["d_ff"]), integer(arm["expected_parameters_analytical"])])
    # Independent task branches use the NoSharing branch architecture and width.
    branch = next(arm for arm in arms if arm["id"] == "NS_base")
    grid.append(["Bağımsız görev modelleri (STL)", "Yok; görevler ayrı eğitilir",
                 str(branch["d_ff"]), integer(controls["parameters_pair"])])
    return grid


def identities(root):
    project = root / "main_project"
    files = [
        ("Bütünlüğü doğrulanmış ham veri (raw_prices.csv)", "data/raw/raw_prices.csv"),
        ("GARCH protokol kaydı", "config/08C_garch_protocol_lock_v4.json"),
        ("GARCH yakınsama kurtarma protokolü", "config/08C_R_garch_rescue_protocol_lock_v4.json"),
        ("DM–HLN / Holm protokol kaydı", "config/09_dm_harvey_holm_protocol_lock_v4.json"),
        ("SHAP protokol kaydı", "config/10_shap_protocol_lock_v4.json"),
        ("SHAP protokol eki", "config/10_shap_protocol_addendum_v4a.json"),
        ("Bartlett-HAC sağlamlık kontrolü protokol kaydı", "config/11_bartlett_dm_protocol_lock_v4.json"),
    ]
    return [["Dosya", "SHA-256 özeti"]] + [[label, sha256((project / path).read_bytes()).hexdigest()]
                                        for label, path in files]


def dm_table(root, task):
    rows = [r for r in records(root / DM) if r["task"] == task]
    asset_order = list(ASSETS)
    order = list(MODEL)
    family_order = ["learned", "naive", "garch_family"]
    def order_key(row):
        model, asset, family = row["baseline_model"], row["asset"], row["baseline_family"]
        ranks = (asset_order.index(asset), order.index(model))
        return (family_order.index(family), *(ranks[::-1] if family == "garch_family" else ranks))
    grid = [["Varlık", "Kıyaslama modeli", "ΔL", "DM–HLN", "Holm p", "Karar"]]
    winners = {"SIGNIFICANT_FINAL_BETTER": "Nihai model", "NOT_SIGNIFICANT": "Anlamlı fark yok"}
    other = {"learned": "Tek görevli", "naive": "Naif model", "garch_family": "GARCH ailesi"}
    for r in sorted(rows, key=order_key):
        decision = r["interpretation"]
        label = other[r["baseline_family"]] if decision == "SIGNIFICANT_BASELINE_BETTER" else winners[decision]
        delta = f"{float(r['mean_loss_differential']):+.6f}".replace(".", ",").replace("-", "−")
        grid.append([ASSETS[r["asset"]], MODEL[r["baseline_model"]], delta,
                     number(r["dm_hln_statistic"], 3).replace("-", "−"), adjusted_p(r["holm_adjusted_p"]), label])
    return grid


def bartlett(root):
    rows = records(root / BARTLETT)
    main = {r["comparison_id"]: r for r in records(root / DM)}
    codes = {"SIGNIFICANT_FINAL_BETTER": "Nihai model", "SIGNIFICANT_BASELINE_BETTER": "Kıyaslama",
             "NOT_SIGNIFICANT": "Fark yok"}
    short = dict(MODEL, SingleTaskTransformer="STT", SingleTaskLSTM="ST-LSTM")
    ordered_models = ["VolPersistence", "SingleTaskTransformer", "SingleTaskLSTM", "XGBoost",
                      "GARCH_1_1_StudentsT_ZeroMean", "GJR_GARCH_1_1_StudentsT_ZeroMean"]
    grid = [["Varlık", "Kıyaslama modeli", "DM* ana", "DM* Bartlett", "ω/γ₀", "Holm p (Bartlett)",
             "Karar (ana)", "Karar (Bartlett)"]]
    for r in sorted(rows, key=lambda x: (list(ASSETS).index(x["asset"]), ordered_models.index(x["baseline_model"]))):
        reference = main[r["comparison_id"]]
        ratio = float(r["bartlett_long_run_variance"]) / float(r["loss_differential_gamma0"])
        grid.append([ASSETS[r["asset"]], short[r["baseline_model"]], number(reference["dm_hln_statistic"], 3),
                     number(r["dm_hln_statistic"], 3), number(ratio, 2), adjusted_p(r["holm_adjusted_p"]),
                     codes[reference["interpretation"]], codes[r["bartlett_decision"]]])
    return grid


TABLES = {
    "2.1": persistence, "3.1": raw_assets, "3.2": splits, "3.4": model_sizes,
    "3.5": selected_baselines, "3.6": runtime, "3.7": h2_architectures, "A.1": identities,
    "B.1": lambda root: dm_table(root, "return"), "B.2": lambda root: dm_table(root, "volatility"),
    "B.3": bartlett,
}
