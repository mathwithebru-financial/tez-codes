# ==========================================================
# 06_best_model_multiseed_v4.py
#
# RESMÎ 06 MULTISEED SAĞLAMLAŞTIRMA AŞAMASI
#
# Kilitli protokol:
#   - Aday kümesi: Top-10 DISTINCT
#   - Aday sayısı: 10
#   - Seeds: [123, 777, 2026]
#   - MIN_EPOCHS = 45
#   - PATIENCE = 15
#   - MAX_EPOCHS = 100
#   - Toplam: 10 × 3 = 30 run
#   - Seçim: en düşük mean ValidationScore
#   - Tie-break: daha düşük seedler-arası sample std (ddof=1)
#   - Test dizileri YÜKLENMEZ ve test metriği HESAPLANMAZ
#
# 05_grid_search_v4.py ile metodolojik süreklilik:
#   - Aynı model sınıfları ve task-head yapıları
#   - Aynı MSE + PinballLoss(tau=0.5)
#   - Aynı v4 shared-parameter-only equal-weight PCGrad varyantı
#   - Aynı raw-scale ValidationScore
#   - Aynı StandardScaler inverse-transform doğrulamaları
#   - Aynı gerçek CPU-clone best-checkpoint mantığı
#
# Resume:
#   - (config_id, seed) bazında
#   - Yalnızca status == success olan run atlanır
#
# ÖNEMLİ:
#   - Bu script 07 test aşamasını açmaz.
#   - 06 yalnızca validation üzerinden config-level multiseed seçim yapar.
#   - Kazanan config'in üç seed checkpoint'i korunur; test checkpoint/ensemble
#     politikası 07 başlamadan önce ayrıca kilitlenmelidir.
# ==========================================================

import os
import json
import pickle
import random
import copy
import warnings
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ==========================================================
# 1. YOLLAR
# ==========================================================

BASE_DIR = "/content/drive/MyDrive/tez_transformer_v4_repro"

CONFIG_DIR = os.path.join(BASE_DIR, "config")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SEQUENCE_DIR = os.path.join(BASE_DIR, "data", "sequences")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

GRID_RESULTS_DIR = os.path.join(BASE_DIR, "results", "grid_search")
GRID_RANKED_CSV = os.path.join(GRID_RESULTS_DIR, "grid_results_ranked_v4.csv")

RESULTS_DIR = os.path.join(BASE_DIR, "results", "multiseed")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
HISTORY_DIR = os.path.join(RESULTS_DIR, "histories")
MODEL_DIR = os.path.join(BASE_DIR, "models", "multiseed")

for path in [RESULTS_DIR, METRICS_DIR, HISTORY_DIR, MODEL_DIR]:
    os.makedirs(path, exist_ok=True)

CANDIDATES_CSV = os.path.join(RESULTS_DIR, "multiseed_candidates_distinct_v4.csv")
RUNS_CSV = os.path.join(RESULTS_DIR, "multiseed_runs_v4.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "multiseed_summary_v4.csv")
RANKED_SUMMARY_CSV = os.path.join(RESULTS_DIR, "multiseed_summary_ranked_v4.csv")
PROGRESS_JSON = os.path.join(RESULTS_DIR, "multiseed_progress_v4.json")
SUMMARY_JSON = os.path.join(RESULTS_DIR, "multiseed_summary_v4.json")
WINNER_JSON = os.path.join(RESULTS_DIR, "multiseed_winner_config_v4.json")

CODE_MANIFEST_PATH = os.path.join(CONFIG_DIR, "code_manifest_v4.csv")
GRID_SCRIPT_PATH = os.path.join(SCRIPTS_DIR, "05_grid_search_v4.py")


# ==========================================================
# 2. KİLİTLİ 06 PROTOKOLÜ
# ==========================================================

SEEDS = [123, 777, 2026]
EXPECTED_CANDIDATE_COUNT = 10
EXPECTED_TOTAL_RUNS = 30
EXPECTED_DISTINCT_SOURCE_RANKS = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12]

BATCH_SIZE = 64

MAX_EPOCHS = 100
MIN_EPOCHS_BEFORE_STOP = 45
PATIENCE = 15

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
DROPOUT = 0.10
TAU = 0.5

# Tie-break tanımı: sample standard deviation (n-1, ddof=1)
STD_DDOF = 1


# ==========================================================
# 3. SEED + DEVICE
# ==========================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 90)
print("06 — v4 RESMÎ TOP-10 DISTINCT × 3 NEW-SEED MULTISEED")
print("=" * 90)
print("[DEVICE]", DEVICE)

if DEVICE.type != "cuda":
    raise RuntimeError(
        "GPU aktif değil. Colab > Çalışma zamanı türünü değiştir > T4 GPU seç."
    )

print("[GPU]", torch.cuda.get_device_name(0))
print("[CANDIDATES] Top-10 distinct")
print("[SEEDS]", SEEDS)
print("[MIN_EPOCHS]", MIN_EPOCHS_BEFORE_STOP)
print("[PATIENCE]", PATIENCE)
print("[MAX_EPOCHS]", MAX_EPOCHS)
print("[EXPECTED RUNS]", EXPECTED_TOTAL_RUNS)
print("[SELECTION] lowest mean ValidationScore; tie-break lower sample std")
print("[TEST ACCESS] NONE")


# ==========================================================
# 4. SCHEMA + DENOMINATOR + PROVENANCE PREFLIGHT
# ==========================================================

SCHEMA_PATH = os.path.join(CONFIG_DIR, "schema_v4.json")
DENOMINATOR_PATH = os.path.join(
    PROCESSED_DIR,
    "selection_baseline_denominators_v4.json"
)

for path in [SCHEMA_PATH, DENOMINATOR_PATH, GRID_RANKED_CSV, CODE_MANIFEST_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Gerekli dosya bulunamadı:\n{path}")

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    schema = json.load(f)

with open(DENOMINATOR_PATH, "r", encoding="utf-8") as f:
    denominators = json.load(f)

ASSET_ORDER = schema["data"]["assets"]
TARGET_NAMES = schema["targets"]["definition"]

if ASSET_ORDER != ["BIST100", "USDTRY", "EURTRY", "GOLD"]:
    raise RuntimeError(f"Asset sırası beklenenden farklı: {ASSET_ORDER}")

if len(TARGET_NAMES) != 8:
    raise RuntimeError(f"Target sayısı 8 değil: {len(TARGET_NAMES)}")


def sha256_file(path, chunk_size=1024 * 1024):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


manifest_df = pd.read_csv(CODE_MANIFEST_PATH)
manifest_row = manifest_df.loc[
    manifest_df["script_name"] == "05_grid_search_v4.py"
]

if len(manifest_row) != 1:
    raise RuntimeError(
        "code_manifest_v4.csv içinde 05_grid_search_v4.py için tek kayıt yok."
    )

if not os.path.exists(GRID_SCRIPT_PATH):
    raise FileNotFoundError(f"Grid script bulunamadı:\n{GRID_SCRIPT_PATH}")

manifest_grid_sha = str(manifest_row.iloc[0]["sha256"])
current_grid_sha = sha256_file(GRID_SCRIPT_PATH)

if current_grid_sha != manifest_grid_sha:
    raise RuntimeError(
        "05_grid_search_v4.py mevcut SHA-256 değeri manifest ile eşleşmiyor.\n"
        f"Manifest: {manifest_grid_sha}\n"
        f"Current : {current_grid_sha}"
    )

print("\n[PROVENANCE PREFLIGHT]")
print("05_grid_search_v4.py SHA-256 manifest eşleşmesi: TRUE")
print("SHA-256:", current_grid_sha)


# ==========================================================
# 5. YARDIMCI FONKSİYONLAR
# ==========================================================

def clone_state_to_cpu(module: nn.Module):
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def cleanup_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_name(config_id: str) -> str:
    return (
        config_id
        .replace("=", "-")
        .replace("__", "_")
        .replace("/", "-")
    )


# ==========================================================
# 6. METRİKLER + VALIDATIONSCORE
# ==========================================================

def mae_np(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse_np(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_np(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return float("nan")

    return float(1.0 - ss_res / ss_tot)


def pinball_np(y_true, y_pred, tau=0.5):
    diff = y_true - y_pred
    loss = np.maximum(tau * diff, (tau - 1.0) * diff)
    return float(np.mean(loss))


def compute_raw_metrics(y_true_raw, y_pred_raw):
    rows = []

    for i, asset in enumerate(ASSET_ORDER):
        true = y_true_raw[:, i]
        pred = y_pred_raw[:, i]

        rows.append({
            "task": "return",
            "asset": asset,
            "MAE": mae_np(true, pred),
            "RMSE": rmse_np(true, pred),
            "R2": r2_np(true, pred),
            "PinballLoss_tau_0.5": np.nan
        })

    for i, asset in enumerate(ASSET_ORDER):
        col = 4 + i
        true = y_true_raw[:, col]
        pred = y_pred_raw[:, col]

        rows.append({
            "task": "volatility",
            "asset": asset,
            "MAE": mae_np(true, pred),
            "RMSE": rmse_np(true, pred),
            "R2": r2_np(true, pred),
            "PinballLoss_tau_0.5": pinball_np(true, pred, tau=TAU)
        })

    return pd.DataFrame(rows)


def compute_validation_score(metrics_df):
    return_ratios = []
    vol_ratios = []
    asset_scores = {}

    for asset in ASSET_ORDER:
        model_return_mae = float(
            metrics_df.loc[
                (metrics_df["task"] == "return")
                & (metrics_df["asset"] == asset),
                "MAE"
            ].iloc[0]
        )

        denom_return_mae = float(
            denominators["return_denominator"][asset]["value"]
        )

        model_vol_pinball = float(
            metrics_df.loc[
                (metrics_df["task"] == "volatility")
                & (metrics_df["asset"] == asset),
                "PinballLoss_tau_0.5"
            ].iloc[0]
        )

        denom_vol_pinball = float(
            denominators["volatility_denominator"][asset]["value"]
        )

        if denom_return_mae <= 0 or denom_vol_pinball <= 0:
            raise RuntimeError(f"{asset} denominator pozitif değil.")

        return_ratio = model_return_mae / denom_return_mae
        vol_ratio = model_vol_pinball / denom_vol_pinball

        return_ratios.append(return_ratio)
        vol_ratios.append(vol_ratio)

        asset_scores[asset] = {
            "return_ratio": float(return_ratio),
            "vol_ratio": float(vol_ratio)
        }

    avg_return_ratio = float(np.mean(return_ratios))
    avg_vol_ratio = float(np.mean(vol_ratios))
    validation_score = float(
        0.5 * avg_return_ratio + 0.5 * avg_vol_ratio
    )
    catastrophic_max_ratio = float(
        max(max(return_ratios), max(vol_ratios))
    )

    return {
        "validation_score": validation_score,
        "avg_return_ratio": avg_return_ratio,
        "avg_vol_ratio": avg_vol_ratio,
        "catastrophic_max_ratio": catastrophic_max_ratio,
        "asset_scores": asset_scores,
        "lower_is_better": True
    }


# ==========================================================
# 7. LOSS FONKSİYONLARI
# ==========================================================

mse_loss = nn.MSELoss()


def pinball_loss_torch(y_pred, y_true, tau=0.5):
    diff = y_true - y_pred
    loss = torch.maximum(tau * diff, (tau - 1.0) * diff)
    return loss.mean()


def split_task_losses(y_pred, y_true):
    ret_loss = mse_loss(y_pred[:, :4], y_true[:, :4])
    vol_loss = pinball_loss_torch(
        y_pred[:, 4:],
        y_true[:, 4:],
        tau=TAU
    )
    return ret_loss, vol_loss


class UncertaintyWeightingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(self, ret_loss, vol_loss):
        loss_ret = (
            torch.exp(-self.log_vars[0]) * ret_loss
            + self.log_vars[0]
        )
        loss_vol = (
            torch.exp(-self.log_vars[1]) * vol_loss
            + self.log_vars[1]
        )
        return loss_ret + loss_vol


def fixed_lambda_loss(ret_loss, vol_loss, loss_name):
    lambda_map = {
        "FixedLambda_0.3": 0.3,
        "FixedLambda_0.5": 0.5,
        "FixedLambda_0.7": 0.7
    }

    if loss_name not in lambda_map:
        raise ValueError(f"Geçersiz FixedLambda loss: {loss_name}")

    lambda_ret = lambda_map[loss_name]
    return lambda_ret * ret_loss + (1.0 - lambda_ret) * vol_loss


# ==========================================================
# 8. TRANSFORMER YAPITAŞLARI
# ==========================================================

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, n_layers, dropout):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=n_layers
        )

    def forward(self, x):
        return self.encoder(x)


def make_head(d_model, dropout):
    return nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model, 4)
    )


# ==========================================================
# 9. MİMARİ 1 — FULL SHARING
# ==========================================================

class FullSharingMTL(nn.Module):
    def __init__(
        self,
        n_features,
        lookback,
        d_model,
        n_head,
        n_layers,
        d_ff,
        dropout
    ):
        super().__init__()

        self.input_projection = nn.Linear(n_features, d_model)
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, lookback, d_model)
        )
        self.encoder = TransformerBlock(
            d_model,
            n_head,
            d_ff,
            n_layers,
            dropout
        )
        self.norm = nn.LayerNorm(d_model)
        self.return_head = make_head(d_model, dropout)
        self.vol_head = make_head(d_model, dropout)

    def forward(self, x):
        h = self.input_projection(x)
        h = h + self.positional_embedding[:, :h.size(1), :]
        h = self.encoder(h)
        h = self.norm(h[:, -1, :])

        ret = self.return_head(h)
        vol = self.vol_head(h)
        return torch.cat([ret, vol], dim=1)

    def pcgrad_groups(self):
        shared = (
            list(self.input_projection.parameters())
            + [self.positional_embedding]
            + list(self.encoder.parameters())
            + list(self.norm.parameters())
        )

        return {
            "shared": shared,
            "return_specific": list(self.return_head.parameters()),
            "vol_specific": list(self.vol_head.parameters())
        }


# ==========================================================
# 10. MİMARİ 2 — PARTIAL SHARING
# ==========================================================

class PartialSharingMTL(nn.Module):
    def __init__(
        self,
        n_features,
        lookback,
        d_model,
        n_head,
        n_layers,
        d_ff,
        dropout
    ):
        super().__init__()

        shared_layers = max(1, n_layers // 2)
        task_layers = max(1, n_layers - shared_layers)

        self.input_projection = nn.Linear(n_features, d_model)
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, lookback, d_model)
        )

        self.shared_encoder = TransformerBlock(
            d_model,
            n_head,
            d_ff,
            shared_layers,
            dropout
        )
        self.return_encoder = TransformerBlock(
            d_model,
            n_head,
            d_ff,
            task_layers,
            dropout
        )
        self.vol_encoder = TransformerBlock(
            d_model,
            n_head,
            d_ff,
            task_layers,
            dropout
        )

        self.return_norm = nn.LayerNorm(d_model)
        self.vol_norm = nn.LayerNorm(d_model)

        # v4: Full/Partial/NoSharing görev başlıkları aynı MLP yapısını kullanır.
        self.return_head = make_head(d_model, dropout)
        self.vol_head = make_head(d_model, dropout)

    def forward(self, x):
        h = self.input_projection(x)
        h = h + self.positional_embedding[:, :h.size(1), :]

        shared = self.shared_encoder(h)

        h_ret = self.return_encoder(shared)
        h_vol = self.vol_encoder(shared)

        h_ret = self.return_norm(h_ret[:, -1, :])
        h_vol = self.vol_norm(h_vol[:, -1, :])

        ret = self.return_head(h_ret)
        vol = self.vol_head(h_vol)
        return torch.cat([ret, vol], dim=1)

    def pcgrad_groups(self):
        shared = (
            list(self.input_projection.parameters())
            + [self.positional_embedding]
            + list(self.shared_encoder.parameters())
        )

        ret = (
            list(self.return_encoder.parameters())
            + list(self.return_norm.parameters())
            + list(self.return_head.parameters())
        )

        vol = (
            list(self.vol_encoder.parameters())
            + list(self.vol_norm.parameters())
            + list(self.vol_head.parameters())
        )

        return {
            "shared": shared,
            "return_specific": ret,
            "vol_specific": vol
        }


# ==========================================================
# 11. MİMARİ 3 — HIERARCHICAL
# ==========================================================

class HierarchicalMTL(nn.Module):
    def __init__(
        self,
        n_features,
        lookback,
        d_model,
        n_head,
        n_layers,
        d_ff,
        dropout
    ):
        super().__init__()

        self.input_projection = nn.Linear(n_features, d_model)
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, lookback, d_model)
        )

        self.encoder = TransformerBlock(
            d_model,
            n_head,
            d_ff,
            n_layers,
            dropout
        )
        self.norm = nn.LayerNorm(d_model)

        self.return_hidden = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.return_head = nn.Linear(d_model, 4)

        self.vol_hidden = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.vol_head = nn.Linear(d_model, 4)

    def forward(self, x):
        h = self.input_projection(x)
        h = h + self.positional_embedding[:, :h.size(1), :]
        h = self.encoder(h)

        base = self.norm(h[:, -1, :])

        ret_hidden = self.return_hidden(base)
        ret = self.return_head(ret_hidden)

        vol_input = torch.cat([base, ret_hidden], dim=1)
        vol_hidden = self.vol_hidden(vol_input)
        vol = self.vol_head(vol_hidden)

        return torch.cat([ret, vol], dim=1)

    def pcgrad_groups(self):
        # return_hidden hem return hem volatility yolunda kullanıldığı için shared'dır.
        shared = (
            list(self.input_projection.parameters())
            + [self.positional_embedding]
            + list(self.encoder.parameters())
            + list(self.norm.parameters())
            + list(self.return_hidden.parameters())
        )

        return {
            "shared": shared,
            "return_specific": list(self.return_head.parameters()),
            "vol_specific": (
                list(self.vol_hidden.parameters())
                + list(self.vol_head.parameters())
            )
        }


# ==========================================================
# 12. MİMARİ 4 — NO SHARING
# ==========================================================

class NoSharing(nn.Module):
    def __init__(
        self,
        n_features,
        lookback,
        d_model,
        n_head,
        n_layers,
        d_ff,
        dropout
    ):
        super().__init__()

        # Return branch
        self.ret_projection = nn.Linear(n_features, d_model)
        self.ret_positional = nn.Parameter(
            torch.zeros(1, lookback, d_model)
        )
        self.ret_encoder = TransformerBlock(
            d_model,
            n_head,
            d_ff,
            n_layers,
            dropout
        )
        self.ret_norm = nn.LayerNorm(d_model)
        self.return_head = make_head(d_model, dropout)

        # Volatility branch
        self.vol_projection = nn.Linear(n_features, d_model)
        self.vol_positional = nn.Parameter(
            torch.zeros(1, lookback, d_model)
        )
        self.vol_encoder = TransformerBlock(
            d_model,
            n_head,
            d_ff,
            n_layers,
            dropout
        )
        self.vol_norm = nn.LayerNorm(d_model)
        self.vol_head = make_head(d_model, dropout)

    def forward(self, x):
        h_ret = self.ret_projection(x)
        h_ret = h_ret + self.ret_positional[:, :h_ret.size(1), :]
        h_ret = self.ret_encoder(h_ret)
        h_ret = self.ret_norm(h_ret[:, -1, :])

        h_vol = self.vol_projection(x)
        h_vol = h_vol + self.vol_positional[:, :h_vol.size(1), :]
        h_vol = self.vol_encoder(h_vol)
        h_vol = self.vol_norm(h_vol[:, -1, :])

        ret = self.return_head(h_ret)
        vol = self.vol_head(h_vol)
        return torch.cat([ret, vol], dim=1)

    def pcgrad_groups(self):
        ret = (
            list(self.ret_projection.parameters())
            + [self.ret_positional]
            + list(self.ret_encoder.parameters())
            + list(self.ret_norm.parameters())
            + list(self.return_head.parameters())
        )

        vol = (
            list(self.vol_projection.parameters())
            + [self.vol_positional]
            + list(self.vol_encoder.parameters())
            + list(self.vol_norm.parameters())
            + list(self.vol_head.parameters())
        )

        return {
            "shared": [],
            "return_specific": ret,
            "vol_specific": vol
        }


# ==========================================================
# 13. MODEL FACTORY + PCGRAD GROUP AUDIT
# ==========================================================

def build_model(architecture, n_features, lookback, size_name):
    size_cfg = schema["model_sizes"][size_name]

    kwargs = {
        "n_features": n_features,
        "lookback": lookback,
        "d_model": int(size_cfg["d_model"]),
        "n_head": int(size_cfg["n_head"]),
        "n_layers": int(size_cfg["n_layers"]),
        "d_ff": int(size_cfg["d_ff"]),
        "dropout": DROPOUT
    }

    if architecture == "FullSharingMTL":
        return FullSharingMTL(**kwargs)

    if architecture == "PartialSharingMTL":
        return PartialSharingMTL(**kwargs)

    if architecture == "HierarchicalMTL":
        return HierarchicalMTL(**kwargs)

    if architecture == "NoSharing":
        return NoSharing(**kwargs)

    raise ValueError(f"Bilinmeyen mimari: {architecture}")


def validate_pcgrad_groups(model):
    groups = model.pcgrad_groups()

    required = {"shared", "return_specific", "vol_specific"}
    if set(groups.keys()) != required:
        raise RuntimeError(
            f"PCGrad group anahtarları yanlış: {groups.keys()}"
        )

    shared_ids = {id(p) for p in groups["shared"]}
    ret_ids = {id(p) for p in groups["return_specific"]}
    vol_ids = {id(p) for p in groups["vol_specific"]}

    if shared_ids & ret_ids:
        raise RuntimeError("shared ve return_specific parametreleri çakışıyor.")

    if shared_ids & vol_ids:
        raise RuntimeError("shared ve vol_specific parametreleri çakışıyor.")

    if ret_ids & vol_ids:
        raise RuntimeError("return_specific ve vol_specific parametreleri çakışıyor.")

    model_ids = {
        id(p)
        for p in model.parameters()
        if p.requires_grad
    }

    grouped_ids = shared_ids | ret_ids | vol_ids

    if grouped_ids != model_ids:
        missing = len(model_ids - grouped_ids)
        extra = len(grouped_ids - model_ids)
        raise RuntimeError(
            "PCGrad group kapsamı model parametreleriyle tam eşleşmiyor. "
            f"missing={missing}, extra={extra}"
        )

    return {
        "shared_param_count": int(
            sum(p.numel() for p in groups["shared"])
        ),
        "return_specific_param_count": int(
            sum(p.numel() for p in groups["return_specific"])
        ),
        "vol_specific_param_count": int(
            sum(p.numel() for p in groups["vol_specific"])
        )
    }


# ==========================================================
# 14. SHARED-PARAMETER-ONLY PCGRAD
# ==========================================================

def grads_or_zeros(loss, params, retain_graph):
    if not params:
        return []

    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        allow_unused=True
    )

    return [
        torch.zeros_like(parameter)
        if gradient is None
        else gradient
        for parameter, gradient in zip(params, grads)
    ]


def flatten_grads(grads):
    if not grads:
        return None

    return torch.cat([g.reshape(-1) for g in grads])


def assign_flat_grad(params, flat_grad):
    offset = 0

    for parameter in params:
        n = parameter.numel()
        parameter.grad = (
            flat_grad[offset:offset + n]
            .view_as(parameter)
            .detach()
            .clone()
        )
        offset += n

    if offset != flat_grad.numel():
        raise RuntimeError(
            "Flat gradient ile parametre boyutları uyuşmuyor."
        )


def pcgrad_backward_equal_weight(model, ret_loss, vol_loss):
    groups = model.pcgrad_groups()

    shared = groups["shared"]
    ret_params = groups["return_specific"]
    vol_params = groups["vol_specific"]

    model.zero_grad(set_to_none=True)

    conflict = False
    shared_dot = None

    if shared:
        g_ret = flatten_grads(
            grads_or_zeros(ret_loss, shared, retain_graph=True)
        )
        g_vol = flatten_grads(
            grads_or_zeros(vol_loss, shared, retain_graph=True)
        )

        dot = torch.dot(g_ret, g_vol)
        shared_dot = float(dot.detach().cpu().item())

        ret_norm_sq = torch.dot(g_ret, g_ret)
        vol_norm_sq = torch.dot(g_vol, g_vol)
        eps = 1e-12

        if float(dot.detach().cpu().item()) < 0.0:
            conflict = True

            g_ret_proj = (
                g_ret
                - dot / (vol_norm_sq + eps) * g_vol
            )
            g_vol_proj = (
                g_vol
                - dot / (ret_norm_sq + eps) * g_ret
            )
        else:
            g_ret_proj = g_ret
            g_vol_proj = g_vol

        assign_flat_grad(
            shared,
            0.5 * (g_ret_proj + g_vol_proj)
        )

    ret_grads = grads_or_zeros(
        ret_loss,
        ret_params,
        retain_graph=True
    )
    vol_grads = grads_or_zeros(
        vol_loss,
        vol_params,
        retain_graph=False
    )

    for parameter, gradient in zip(ret_params, ret_grads):
        parameter.grad = (0.5 * gradient).detach().clone()

    for parameter, gradient in zip(vol_params, vol_grads):
        parameter.grad = (0.5 * gradient).detach().clone()

    return {
        "conflict": conflict,
        "shared_dot": shared_dot,
        "shared_param_count": int(sum(p.numel() for p in shared))
    }


# ==========================================================
# 15. DATA LOADING — TEST YOK
# ==========================================================

DATA_CACHE = {}


def load_sequence_data(feature_set, lookback):
    key = (feature_set, int(lookback))

    if key in DATA_CACHE:
        return DATA_CACHE[key]

    seq_dir = os.path.join(
        SEQUENCE_DIR,
        feature_set,
        f"lb{lookback}"
    )
    scaler_path = os.path.join(
        SEQUENCE_DIR,
        feature_set,
        "scalers.pkl"
    )

    paths = {
        "X_train": os.path.join(seq_dir, "X_train.npy"),
        "y_train": os.path.join(seq_dir, "y_train.npy"),
        "X_val": os.path.join(seq_dir, "X_val.npy"),
        "y_val": os.path.join(seq_dir, "y_val.npy"),
        "y_val_raw": os.path.join(seq_dir, "y_val_raw.npy")
    }

    for path in list(paths.values()) + [scaler_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Gerekli dosya bulunamadı:\n{path}")

    arrays = {
        name: np.load(path)
        for name, path in paths.items()
    }

    with open(scaler_path, "rb") as f:
        scaler_obj = pickle.load(f)

    y_scaler = scaler_obj["y_scaler"]

    inverse_check = y_scaler.inverse_transform(arrays["y_val"])
    max_inverse_diff = float(
        np.max(np.abs(inverse_check - arrays["y_val_raw"]))
    )

    if max_inverse_diff > 1e-5:
        raise RuntimeError(
            f"Inverse-scale kontrolü geçmedi: {max_inverse_diff}"
        )

    if len(arrays["X_val"]) != 584:
        raise RuntimeError(
            f"Validation örnek sayısı 584 değil: {len(arrays['X_val'])}"
        )

    payload = (
        arrays["X_train"],
        arrays["y_train"],
        arrays["X_val"],
        arrays["y_val"],
        arrays["y_val_raw"],
        y_scaler,
        max_inverse_diff
    )

    DATA_CACHE[key] = payload
    return payload


def make_loaders(X_train, y_train, X_val, y_val, seed):
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )

    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        generator=generator
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False
    )

    return train_loader, val_loader


# ==========================================================
# 16. TRAIN / EVAL
# ==========================================================

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    optimizer_params,
    loss_strategy,
    uncertainty_loss_module=None
):
    model.train()

    total_loss_sum = 0.0
    ret_loss_sum = 0.0
    vol_loss_sum = 0.0
    n_obs = 0

    pcgrad_conflicts = 0
    pcgrad_batches = 0
    pcgrad_dots = []

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        y_pred = model(X_batch)

        if y_pred.shape[1] != 8:
            raise RuntimeError(
                f"Model output boyutu 8 değil: {y_pred.shape}"
            )

        ret_loss, vol_loss = split_task_losses(y_pred, y_batch)

        if loss_strategy.startswith("FixedLambda"):
            loss = fixed_lambda_loss(
                ret_loss,
                vol_loss,
                loss_strategy
            )
            loss.backward()

        elif loss_strategy == "UncertaintyWeighting":
            if uncertainty_loss_module is None:
                raise RuntimeError(
                    "UncertaintyWeighting modülü oluşturulmamış."
                )

            loss = uncertainty_loss_module(ret_loss, vol_loss)
            loss.backward()

        elif loss_strategy == "PCGrad":
            loss = 0.5 * ret_loss + 0.5 * vol_loss
            info = pcgrad_backward_equal_weight(
                model,
                ret_loss,
                vol_loss
            )

            pcgrad_batches += 1
            pcgrad_conflicts += int(info["conflict"])

            if info["shared_dot"] is not None:
                pcgrad_dots.append(info["shared_dot"])

        else:
            raise ValueError(
                f"Bilinmeyen loss stratejisi: {loss_strategy}"
            )

        if not torch.isfinite(loss):
            raise RuntimeError("Training loss finite değil.")

        torch.nn.utils.clip_grad_norm_(
            optimizer_params,
            GRAD_CLIP
        )

        optimizer.step()

        batch_size = X_batch.size(0)
        total_loss_sum += loss.item() * batch_size
        ret_loss_sum += ret_loss.item() * batch_size
        vol_loss_sum += vol_loss.item() * batch_size
        n_obs += batch_size

    return {
        "loss": total_loss_sum / n_obs,
        "return_loss": ret_loss_sum / n_obs,
        "vol_loss": vol_loss_sum / n_obs,
        "pcgrad_conflict_batches": pcgrad_conflicts,
        "pcgrad_total_batches": pcgrad_batches,
        "pcgrad_conflict_rate": (
            pcgrad_conflicts / pcgrad_batches
            if pcgrad_batches
            else 0.0
        ),
        "pcgrad_mean_shared_dot": (
            float(np.mean(pcgrad_dots))
            if pcgrad_dots
            else None
        )
    }


@torch.no_grad()
def evaluate_model(model, val_loader, y_scaler, y_val_raw):
    model.eval()

    ret_loss_sum = 0.0
    vol_loss_sum = 0.0
    n_obs = 0
    preds_scaled = []
    true_scaled = []

    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        y_pred = model(X_batch)
        ret_loss, vol_loss = split_task_losses(y_pred, y_batch)

        batch_size = X_batch.size(0)
        ret_loss_sum += ret_loss.item() * batch_size
        vol_loss_sum += vol_loss.item() * batch_size
        n_obs += batch_size

        preds_scaled.append(y_pred.detach().cpu().numpy())
        true_scaled.append(y_batch.detach().cpu().numpy())

    preds_scaled = np.vstack(preds_scaled)
    true_scaled = np.vstack(true_scaled)

    true_raw_check = y_scaler.inverse_transform(true_scaled)
    true_diff = float(
        np.max(np.abs(true_raw_check - y_val_raw))
    )

    if true_diff > 1e-5:
        raise RuntimeError(
            f"Validation true inverse-scale uyuşmuyor: {true_diff}"
        )

    preds_raw = y_scaler.inverse_transform(preds_scaled)

    if not np.isfinite(preds_raw).all():
        raise RuntimeError("Validation prediction içinde NaN/Inf var.")

    metrics_df = compute_raw_metrics(y_val_raw, preds_raw)
    score_obj = compute_validation_score(metrics_df)

    return {
        "val_equal_weight_proxy_loss": (
            0.5 * (ret_loss_sum / n_obs)
            + 0.5 * (vol_loss_sum / n_obs)
        ),
        "val_return_loss": ret_loss_sum / n_obs,
        "val_vol_loss": vol_loss_sum / n_obs,
        "metrics_df": metrics_df,
        "score_obj": score_obj,
        "max_true_inverse_diff": true_diff
    }




# ==========================================================
# 17. TOP-10 DISTINCT ADAY KÜMESİNİ PROGRAMATİK ÜRET
# ==========================================================

ranked_df = pd.read_csv(GRID_RANKED_CSV)

required_ranked_cols = {
    "rank", "config_id", "architecture", "loss_strategy",
    "lookback", "size", "feature_set", "status",
    "validation_score", "test_arrays_loaded", "test_metrics_computed"
}
missing_ranked_cols = required_ranked_cols - set(ranked_df.columns)
if missing_ranked_cols:
    raise RuntimeError(
        f"grid_results_ranked_v4.csv eksik kolonlar: {sorted(missing_ranked_cols)}"
    )

if len(ranked_df) != 480:
    raise RuntimeError(f"Ranked grid 480 satır değil: {len(ranked_df)}")

if ranked_df["config_id"].nunique() != 480:
    raise RuntimeError(
        f"Ranked grid unique config sayısı 480 değil: {ranked_df['config_id'].nunique()}"
    )

if not (ranked_df["status"] == "success").all():
    raise RuntimeError("Ranked grid içinde success olmayan config var.")

# CSV bool kolonları bazen string gelebilir; güvenli bool normalize.
def bool_series_is_all_false(series):
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin(["false", "0", "nan", "none", ""]).all()

if not bool_series_is_all_false(ranked_df["test_arrays_loaded"]):
    raise RuntimeError("05 ranked sonuçlarında test_arrays_loaded=True bulundu.")

if not bool_series_is_all_false(ranked_df["test_metrics_computed"]):
    raise RuntimeError("05 ranked sonuçlarında test_metrics_computed=True bulundu.")

ranked_df = ranked_df.sort_values("rank").reset_index(drop=True)


def canonical_equivalence_key(row):
    architecture = str(row["architecture"])
    loss_strategy = str(row["loss_strategy"])

    if (
        architecture == "NoSharing"
        and loss_strategy in {"FixedLambda_0.5", "PCGrad"}
    ):
        loss_key = "NoSharing_FL0.5_EQ_PCGrad"
    else:
        loss_key = loss_strategy

    return (
        architecture,
        loss_key,
        int(row["lookback"]),
        str(row["size"]),
        str(row["feature_set"])
    )


candidate_rows = []
seen_keys = set()

for _, row in ranked_df.iterrows():
    key = canonical_equivalence_key(row)

    if key in seen_keys:
        continue

    seen_keys.add(key)
    candidate_rows.append(row.copy())

    if len(candidate_rows) == EXPECTED_CANDIDATE_COUNT:
        break

if len(candidate_rows) != EXPECTED_CANDIDATE_COUNT:
    raise RuntimeError(
        f"Top-10 distinct üretilemedi: {len(candidate_rows)} aday bulundu."
    )

candidates_df = pd.DataFrame(candidate_rows).copy()
candidates_df.insert(
    0,
    "candidate_position",
    np.arange(1, EXPECTED_CANDIDATE_COUNT + 1)
)
candidates_df = candidates_df.rename(columns={"rank": "source_rank"})
candidates_df["canonical_equivalence_key"] = candidates_df.apply(
    lambda row: repr(canonical_equivalence_key(row)),
    axis=1
)

actual_source_ranks = candidates_df["source_rank"].astype(int).tolist()

if actual_source_ranks != EXPECTED_DISTINCT_SOURCE_RANKS:
    raise RuntimeError(
        "Top-10 distinct source rank'leri MASTER v5 ile uyuşmuyor.\n"
        f"Beklenen: {EXPECTED_DISTINCT_SOURCE_RANKS}\n"
        f"Gerçek   : {actual_source_ranks}"
    )

if candidates_df["config_id"].nunique() != EXPECTED_CANDIDATE_COUNT:
    raise RuntimeError("Distinct adaylarda duplicate config_id var.")

candidates_df.to_csv(CANDIDATES_CSV, index=False)

print("\n" + "=" * 90)
print("TOP-10 DISTINCT ADAY KÜMESİ — DOĞRULANDI")
print("=" * 90)
print(candidates_df[[
    "candidate_position", "source_rank", "architecture", "loss_strategy",
    "lookback", "size", "feature_set", "validation_score"
]].to_string(index=False))
print("\nSource ranks:", actual_source_ranks)
print("Test access   : NONE")


# ==========================================================
# 18. RESULT SCHEMA + RESUME
# ==========================================================

RATIO_COLUMNS = []
for asset in ASSET_ORDER:
    RATIO_COLUMNS.append(f"{asset}_return_ratio")
    RATIO_COLUMNS.append(f"{asset}_vol_ratio")

RUN_RESULT_COLUMNS = [
    "candidate_position",
    "source_rank",
    "config_id",
    "architecture",
    "loss_strategy",
    "lookback",
    "size",
    "feature_set",
    "seed",
    "max_epochs",
    "min_epochs_before_stop",
    "patience",
    "best_epoch",
    "epochs_ran",
    "validation_score",
    "avg_return_ratio",
    "avg_vol_ratio",
    "catastrophic_max_ratio",
    "parameter_count",
    "loss_parameter_count",
    "total_trainable_parameter_count",
    "shared_param_count",
    "return_specific_param_count",
    "vol_specific_param_count",
    "pcgrad_conflict_batches_total",
    "pcgrad_batches_total",
    "pcgrad_conflict_rate_total",
    "uw_log_var_return",
    "uw_log_var_volatility",
    "uw_weight_return",
    "uw_weight_volatility",
    "max_inverse_diff",
    "elapsed_seconds",
    "status",
    "error_message",
    "metrics_file",
    "history_file",
    "checkpoint_file",
    "test_arrays_loaded",
    "test_metrics_computed"
] + RATIO_COLUMNS

if os.path.exists(RUNS_CSV):
    existing_runs = pd.read_csv(RUNS_CSV)
    missing_cols = [
        col for col in RUN_RESULT_COLUMNS
        if col not in existing_runs.columns
    ]
    if missing_cols:
        raise RuntimeError(
            "Mevcut multiseed_runs_v4.csv schema ile uyuşmuyor. "
            f"Eksik kolonlar: {missing_cols}"
        )
else:
    existing_runs = pd.DataFrame(columns=RUN_RESULT_COLUMNS)

completed_pairs = set(
    zip(
        existing_runs.loc[
            existing_runs["status"] == "success",
            "config_id"
        ].astype(str),
        existing_runs.loc[
            existing_runs["status"] == "success",
            "seed"
        ].astype(int)
    )
)

print("\n[RESUME]")
print("Başarılı tamamlanmış (config_id, seed) çifti:", len(completed_pairs))


def append_run_result(row):
    row_full = {
        col: row.get(col, np.nan)
        for col in RUN_RESULT_COLUMNS
    }

    row_df = pd.DataFrame([row_full], columns=RUN_RESULT_COLUMNS)

    if os.path.exists(RUNS_CSV):
        row_df.to_csv(
            RUNS_CSV,
            mode="a",
            header=False,
            index=False
        )
    else:
        row_df.to_csv(RUNS_CSV, index=False)


def save_progress(current_run_idx, current_config_id, current_seed):
    progress = {
        "updated_at": datetime.now().isoformat(),
        "current_run_idx": int(current_run_idx),
        "expected_total_runs": EXPECTED_TOTAL_RUNS,
        "completed_success_pairs": int(len(completed_pairs)),
        "current_config_id": current_config_id,
        "current_seed": int(current_seed),
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "seeds": SEEDS,
        "min_epochs_before_stop": MIN_EPOCHS_BEFORE_STOP,
        "patience": PATIENCE,
        "max_epochs": MAX_EPOCHS,
        "test_arrays_loaded": False,
        "test_metrics_computed": False
    }

    with open(PROGRESS_JSON, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


# ==========================================================
# 19. 30-RUN MULTISEED LOOP
# ==========================================================

run_plan = []
for _, candidate in candidates_df.iterrows():
    for seed in SEEDS:
        run_plan.append((candidate.to_dict(), int(seed)))

if len(run_plan) != EXPECTED_TOTAL_RUNS:
    raise RuntimeError(
        f"Run plan 30 değil: {len(run_plan)}"
    )

for run_idx, (cfg, seed) in enumerate(run_plan, start=1):
    config_id = str(cfg["config_id"])
    pair_key = (config_id, int(seed))

    if pair_key in completed_pairs:
        print(
            f"[SKIP] ({run_idx}/{EXPECTED_TOTAL_RUNS}) "
            f"Zaten success: seed={seed} | {config_id}"
        )
        continue

    print("\n" + "=" * 90)
    print(f"[RUN {run_idx}/{EXPECTED_TOTAL_RUNS}] seed={seed}")
    print(config_id)
    print("=" * 90)

    start_time = datetime.now()

    model = None
    optimizer = None
    uncertainty_loss_module = None
    train_loader = None
    val_loader = None

    try:
        set_seed(seed)

        (
            X_train,
            y_train,
            X_val,
            y_val,
            y_val_raw,
            y_scaler,
            max_inverse_diff
        ) = load_sequence_data(
            feature_set=str(cfg["feature_set"]),
            lookback=int(cfg["lookback"])
        )

        train_loader, val_loader = make_loaders(
            X_train,
            y_train,
            X_val,
            y_val,
            seed=seed
        )

        model = build_model(
            architecture=str(cfg["architecture"]),
            n_features=X_train.shape[2],
            lookback=int(cfg["lookback"]),
            size_name=str(cfg["size"])
        ).to(DEVICE)

        group_info = validate_pcgrad_groups(model)

        parameter_count = int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )

        model_params = list(model.parameters())
        optimizer_params = list(model_params)
        loss_parameter_count = 0

        if str(cfg["loss_strategy"]) == "UncertaintyWeighting":
            uncertainty_loss_module = UncertaintyWeightingLoss().to(DEVICE)
            optimizer_params = (
                optimizer_params
                + list(uncertainty_loss_module.parameters())
            )
            loss_parameter_count = int(
                sum(p.numel() for p in uncertainty_loss_module.parameters())
            )

        total_trainable_parameter_count = int(
            parameter_count + loss_parameter_count
        )

        optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        best_cfg_score = np.inf
        best_epoch = 0
        no_improve_count = 0
        best_epoch_payload = None
        history_rows = []

        total_pcgrad_conflicts = 0
        total_pcgrad_batches = 0
        epochs_ran = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            epochs_ran = epoch

            train_info = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                optimizer_params=optimizer_params,
                loss_strategy=str(cfg["loss_strategy"]),
                uncertainty_loss_module=uncertainty_loss_module
            )

            val_info = evaluate_model(
                model=model,
                val_loader=val_loader,
                y_scaler=y_scaler,
                y_val_raw=y_val_raw
            )

            val_score = float(
                val_info["score_obj"]["validation_score"]
            )

            total_pcgrad_conflicts += int(
                train_info["pcgrad_conflict_batches"]
            )
            total_pcgrad_batches += int(
                train_info["pcgrad_total_batches"]
            )

            uw_log_var_return = np.nan
            uw_log_var_volatility = np.nan
            uw_weight_return = np.nan
            uw_weight_volatility = np.nan

            if uncertainty_loss_module is not None:
                log_vars = (
                    uncertainty_loss_module
                    .log_vars
                    .detach()
                    .cpu()
                    .numpy()
                )

                uw_log_var_return = float(log_vars[0])
                uw_log_var_volatility = float(log_vars[1])
                uw_weight_return = float(np.exp(-log_vars[0]))
                uw_weight_volatility = float(np.exp(-log_vars[1]))

            history_rows.append({
                "candidate_position": int(cfg["candidate_position"]),
                "source_rank": int(cfg["source_rank"]),
                "config_id": config_id,
                "seed": int(seed),
                "epoch": int(epoch),
                "train_loss": train_info["loss"],
                "train_return_loss": train_info["return_loss"],
                "train_vol_loss": train_info["vol_loss"],
                "val_equal_weight_proxy_loss": val_info[
                    "val_equal_weight_proxy_loss"
                ],
                "val_return_loss": val_info["val_return_loss"],
                "val_vol_loss": val_info["val_vol_loss"],
                "validation_score": val_score,
                "avg_return_ratio": val_info["score_obj"][
                    "avg_return_ratio"
                ],
                "avg_vol_ratio": val_info["score_obj"][
                    "avg_vol_ratio"
                ],
                "pcgrad_conflict_batches": train_info[
                    "pcgrad_conflict_batches"
                ],
                "pcgrad_total_batches": train_info[
                    "pcgrad_total_batches"
                ],
                "pcgrad_conflict_rate": train_info[
                    "pcgrad_conflict_rate"
                ],
                "pcgrad_mean_shared_dot": train_info[
                    "pcgrad_mean_shared_dot"
                ],
                "uw_log_var_return": uw_log_var_return,
                "uw_log_var_volatility": uw_log_var_volatility,
                "uw_weight_return": uw_weight_return,
                "uw_weight_volatility": uw_weight_volatility
            })

            print(
                f"Epoch {epoch:03d} | "
                f"train={train_info['loss']:.5f} | "
                f"score={val_score:.5f} | "
                f"ret={val_info['score_obj']['avg_return_ratio']:.4f} | "
                f"vol={val_info['score_obj']['avg_vol_ratio']:.4f}"
            )

            if val_score < best_cfg_score:
                best_cfg_score = val_score
                best_epoch = epoch
                no_improve_count = 0

                best_epoch_payload = {
                    "model_state_dict": clone_state_to_cpu(model),
                    "uncertainty_state_dict": (
                        clone_state_to_cpu(uncertainty_loss_module)
                        if uncertainty_loss_module is not None
                        else None
                    ),
                    "config": copy.deepcopy(cfg),
                    "seed": int(seed),
                    "epoch": int(epoch),
                    "validation_score": val_score,
                    "score_obj": copy.deepcopy(val_info["score_obj"]),
                    "metrics_df": val_info["metrics_df"].copy(),
                    "uw_log_var_return": uw_log_var_return,
                    "uw_log_var_volatility": uw_log_var_volatility,
                    "uw_weight_return": uw_weight_return,
                    "uw_weight_volatility": uw_weight_volatility
                }
            else:
                no_improve_count += 1

            if (
                epoch >= MIN_EPOCHS_BEFORE_STOP
                and no_improve_count >= PATIENCE
            ):
                print(
                    f"[EARLY STOP] epoch={epoch}, "
                    f"best_epoch={best_epoch}, "
                    f"{PATIENCE} epoch iyileşme yok."
                )
                break

        if best_epoch_payload is None:
            raise RuntimeError("Best checkpoint oluşmadı.")

        elapsed_seconds = (
            datetime.now() - start_time
        ).total_seconds()

        score_obj = best_epoch_payload["score_obj"]
        metrics_df = best_epoch_payload["metrics_df"]

        config_safe = safe_name(config_id)
        run_tag = f"cand{int(cfg['candidate_position']):02d}_seed{seed}"

        metrics_path = os.path.join(
            METRICS_DIR,
            f"metrics_{run_tag}_{config_safe}.csv"
        )
        history_path = os.path.join(
            HISTORY_DIR,
            f"history_{run_tag}_{config_safe}.csv"
        )
        checkpoint_path = os.path.join(
            MODEL_DIR,
            f"checkpoint_{run_tag}_{config_safe}.pt"
        )

        metrics_df.to_csv(metrics_path, index=False)
        pd.DataFrame(history_rows).to_csv(history_path, index=False)

        torch.save({
            "model_state_dict": best_epoch_payload["model_state_dict"],
            "uncertainty_state_dict": best_epoch_payload[
                "uncertainty_state_dict"
            ],
            "config": copy.deepcopy(cfg),
            "seed": int(seed),
            "epoch": int(best_epoch),
            "validation_score": float(score_obj["validation_score"]),
            "score_obj": copy.deepcopy(score_obj),
            "target_names": TARGET_NAMES,
            "asset_order": ASSET_ORDER,
            "parameter_count": parameter_count,
            "loss_parameter_count": loss_parameter_count,
            "total_trainable_parameter_count": total_trainable_parameter_count,
            "pcgrad_group_counts": group_info,
            "protocol": {
                "candidate_set": "Top-10 distinct",
                "seeds": SEEDS,
                "min_epochs_before_stop": MIN_EPOCHS_BEFORE_STOP,
                "patience": PATIENCE,
                "max_epochs": MAX_EPOCHS,
                "selection": "lowest mean ValidationScore; tie-break lower sample std",
                "std_ddof": STD_DDOF
            },
            "test_arrays_loaded": False,
            "test_metrics_computed": False
        }, checkpoint_path)

        row = {
            "candidate_position": int(cfg["candidate_position"]),
            "source_rank": int(cfg["source_rank"]),
            "config_id": config_id,
            "architecture": str(cfg["architecture"]),
            "loss_strategy": str(cfg["loss_strategy"]),
            "lookback": int(cfg["lookback"]),
            "size": str(cfg["size"]),
            "feature_set": str(cfg["feature_set"]),
            "seed": int(seed),
            "max_epochs": MAX_EPOCHS,
            "min_epochs_before_stop": MIN_EPOCHS_BEFORE_STOP,
            "patience": PATIENCE,
            "best_epoch": int(best_epoch),
            "epochs_ran": int(epochs_ran),
            "validation_score": float(score_obj["validation_score"]),
            "avg_return_ratio": float(score_obj["avg_return_ratio"]),
            "avg_vol_ratio": float(score_obj["avg_vol_ratio"]),
            "catastrophic_max_ratio": float(
                score_obj["catastrophic_max_ratio"]
            ),
            "parameter_count": parameter_count,
            "loss_parameter_count": loss_parameter_count,
            "total_trainable_parameter_count": total_trainable_parameter_count,
            "shared_param_count": group_info["shared_param_count"],
            "return_specific_param_count": group_info[
                "return_specific_param_count"
            ],
            "vol_specific_param_count": group_info[
                "vol_specific_param_count"
            ],
            "pcgrad_conflict_batches_total": total_pcgrad_conflicts,
            "pcgrad_batches_total": total_pcgrad_batches,
            "pcgrad_conflict_rate_total": (
                total_pcgrad_conflicts / total_pcgrad_batches
                if total_pcgrad_batches
                else 0.0
            ),
            "uw_log_var_return": best_epoch_payload[
                "uw_log_var_return"
            ],
            "uw_log_var_volatility": best_epoch_payload[
                "uw_log_var_volatility"
            ],
            "uw_weight_return": best_epoch_payload[
                "uw_weight_return"
            ],
            "uw_weight_volatility": best_epoch_payload[
                "uw_weight_volatility"
            ],
            "max_inverse_diff": max_inverse_diff,
            "elapsed_seconds": elapsed_seconds,
            "status": "success",
            "error_message": "",
            "metrics_file": metrics_path,
            "history_file": history_path,
            "checkpoint_file": checkpoint_path,
            "test_arrays_loaded": False,
            "test_metrics_computed": False
        }

        for asset in ASSET_ORDER:
            row[f"{asset}_return_ratio"] = score_obj[
                "asset_scores"
            ][asset]["return_ratio"]
            row[f"{asset}_vol_ratio"] = score_obj[
                "asset_scores"
            ][asset]["vol_ratio"]

        append_run_result(row)
        completed_pairs.add(pair_key)

        save_progress(
            current_run_idx=run_idx,
            current_config_id=config_id,
            current_seed=seed
        )

        print(
            f"[SUCCESS] seed={seed} | best_epoch={best_epoch} | "
            f"best_score={best_cfg_score:.6f} | "
            f"elapsed={elapsed_seconds:.1f}s"
        )

    except Exception as error:
        elapsed_seconds = (
            datetime.now() - start_time
        ).total_seconds()

        error_row = {
            "candidate_position": int(cfg["candidate_position"]),
            "source_rank": int(cfg["source_rank"]),
            "config_id": config_id,
            "architecture": str(cfg["architecture"]),
            "loss_strategy": str(cfg["loss_strategy"]),
            "lookback": int(cfg["lookback"]),
            "size": str(cfg["size"]),
            "feature_set": str(cfg["feature_set"]),
            "seed": int(seed),
            "max_epochs": MAX_EPOCHS,
            "min_epochs_before_stop": MIN_EPOCHS_BEFORE_STOP,
            "patience": PATIENCE,
            "elapsed_seconds": elapsed_seconds,
            "status": "error",
            "error_message": repr(error),
            "test_arrays_loaded": False,
            "test_metrics_computed": False
        }

        append_run_result(error_row)

        print("[ERROR]", config_id, "| seed=", seed)
        print(repr(error))

        save_progress(
            current_run_idx=run_idx,
            current_config_id=config_id,
            current_seed=seed
        )

    finally:
        try:
            del model
            del optimizer
            del uncertainty_loss_module
            del train_loader
            del val_loader
        except Exception:
            pass

        cleanup_cuda()


# ==========================================================
# 20. FINAL 30/30 INTEGRITY AUDIT + CONFIG-LEVEL SUMMARY
# ==========================================================

if not os.path.exists(RUNS_CSV):
    raise RuntimeError("multiseed_runs_v4.csv oluşmadı.")

runs_all = pd.read_csv(RUNS_CSV)

success_runs = runs_all.loc[
    runs_all["status"] == "success"
].copy()

# Aynı pair için resume/yeniden deneme olmuşsa son success kaydını tut.
success_runs = success_runs.drop_duplicates(
    subset=["config_id", "seed"],
    keep="last"
)

candidate_ids = set(candidates_df["config_id"].astype(str))
success_runs = success_runs.loc[
    success_runs["config_id"].astype(str).isin(candidate_ids)
].copy()

if len(success_runs) != EXPECTED_TOTAL_RUNS:
    raise RuntimeError(
        f"06 henüz 30/30 success değil: {len(success_runs)}/{EXPECTED_TOTAL_RUNS}. "
        "Aynı script yeniden çalıştırıldığında yalnızca success olmayan pair'ler koşar."
    )

if success_runs[["config_id", "seed"]].drop_duplicates().shape[0] != EXPECTED_TOTAL_RUNS:
    raise RuntimeError("30 success run içinde duplicate (config_id, seed) var.")

for config_id, group in success_runs.groupby("config_id"):
    actual_seeds = sorted(group["seed"].astype(int).tolist())
    if actual_seeds != sorted(SEEDS):
        raise RuntimeError(
            f"Seed set uyuşmuyor: {config_id}\n"
            f"Beklenen: {sorted(SEEDS)}\n"
            f"Gerçek   : {actual_seeds}"
        )

if not bool_series_is_all_false(success_runs["test_arrays_loaded"]):
    raise RuntimeError("06 success runlarında test_arrays_loaded=True bulundu.")

if not bool_series_is_all_false(success_runs["test_metrics_computed"]):
    raise RuntimeError("06 success runlarında test_metrics_computed=True bulundu.")

summary_rows = []

for _, candidate in candidates_df.iterrows():
    config_id = str(candidate["config_id"])
    group = success_runs.loc[
        success_runs["config_id"].astype(str) == config_id
    ].copy()

    if len(group) != len(SEEDS):
        raise RuntimeError(
            f"Config için 3 success seed yok: {config_id} -> {len(group)}"
        )

    row = {
        "candidate_position": int(candidate["candidate_position"]),
        "source_rank": int(candidate["source_rank"]),
        "config_id": config_id,
        "architecture": str(candidate["architecture"]),
        "loss_strategy": str(candidate["loss_strategy"]),
        "lookback": int(candidate["lookback"]),
        "size": str(candidate["size"]),
        "feature_set": str(candidate["feature_set"]),
        "n_seeds": int(len(group)),
        "seeds": repr(sorted(group["seed"].astype(int).tolist())),
        "mean_validation_score": float(group["validation_score"].mean()),
        "std_validation_score_sample": float(
            group["validation_score"].std(ddof=STD_DDOF)
        ),
        "min_validation_score": float(group["validation_score"].min()),
        "max_validation_score": float(group["validation_score"].max()),
        "mean_avg_return_ratio": float(group["avg_return_ratio"].mean()),
        "std_avg_return_ratio_sample": float(
            group["avg_return_ratio"].std(ddof=STD_DDOF)
        ),
        "mean_avg_vol_ratio": float(group["avg_vol_ratio"].mean()),
        "std_avg_vol_ratio_sample": float(
            group["avg_vol_ratio"].std(ddof=STD_DDOF)
        ),
        "mean_catastrophic_max_ratio": float(
            group["catastrophic_max_ratio"].mean()
        ),
        "min_best_epoch": int(group["best_epoch"].min()),
        "median_best_epoch": float(group["best_epoch"].median()),
        "max_best_epoch": int(group["best_epoch"].max()),
        "mean_epochs_ran": float(group["epochs_ran"].mean()),
        "parameter_count": int(group["parameter_count"].iloc[0]),
        "total_trainable_parameter_count": int(
            group["total_trainable_parameter_count"].iloc[0]
        )
    }

    for asset in ASSET_ORDER:
        row[f"mean_{asset}_return_ratio"] = float(
            group[f"{asset}_return_ratio"].mean()
        )
        row[f"std_{asset}_return_ratio_sample"] = float(
            group[f"{asset}_return_ratio"].std(ddof=STD_DDOF)
        )
        row[f"mean_{asset}_vol_ratio"] = float(
            group[f"{asset}_vol_ratio"].mean()
        )
        row[f"std_{asset}_vol_ratio_sample"] = float(
            group[f"{asset}_vol_ratio"].std(ddof=STD_DDOF)
        )

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_CSV, index=False)

ranked_summary_df = summary_df.sort_values(
    [
        "mean_validation_score",
        "std_validation_score_sample",
        "candidate_position"
    ],
    ascending=[True, True, True]
).reset_index(drop=True)

ranked_summary_df.insert(
    0,
    "multiseed_rank",
    np.arange(1, len(ranked_summary_df) + 1)
)

ranked_summary_df.to_csv(RANKED_SUMMARY_CSV, index=False)

winner = ranked_summary_df.iloc[0]
winner_config_id = str(winner["config_id"])
winner_runs = success_runs.loc[
    success_runs["config_id"].astype(str) == winner_config_id
].sort_values("seed")

winner_payload = {
    "project_version": "v4_repro",
    "created_at": datetime.now().isoformat(),
    "script": "06_best_model_multiseed_v4.py",
    "candidate_set": "Top-10 distinct",
    "candidate_count": EXPECTED_CANDIDATE_COUNT,
    "source_ranks": EXPECTED_DISTINCT_SOURCE_RANKS,
    "seeds": SEEDS,
    "min_epochs_before_stop": MIN_EPOCHS_BEFORE_STOP,
    "patience": PATIENCE,
    "max_epochs": MAX_EPOCHS,
    "selection_rule": "lowest mean ValidationScore",
    "tie_break": "lower seed-to-seed sample std of ValidationScore",
    "std_ddof": STD_DDOF,
    "winner": {
        "multiseed_rank": int(winner["multiseed_rank"]),
        "candidate_position": int(winner["candidate_position"]),
        "source_rank": int(winner["source_rank"]),
        "config_id": winner_config_id,
        "architecture": str(winner["architecture"]),
        "loss_strategy": str(winner["loss_strategy"]),
        "lookback": int(winner["lookback"]),
        "size": str(winner["size"]),
        "feature_set": str(winner["feature_set"]),
        "mean_validation_score": float(winner["mean_validation_score"]),
        "std_validation_score_sample": float(
            winner["std_validation_score_sample"]
        ),
        "mean_avg_return_ratio": float(winner["mean_avg_return_ratio"]),
        "mean_avg_vol_ratio": float(winner["mean_avg_vol_ratio"]),
        "mean_catastrophic_max_ratio": float(
            winner["mean_catastrophic_max_ratio"]
        ),
        "seed_checkpoints": [
            {
                "seed": int(row["seed"]),
                "validation_score": float(row["validation_score"]),
                "best_epoch": int(row["best_epoch"]),
                "checkpoint_file": str(row["checkpoint_file"])
            }
            for _, row in winner_runs.iterrows()
        ]
    },
    "test_arrays_loaded": False,
    "test_metrics_computed": False,
    "note": (
        "06 selects the winning configuration at config level across three new seeds. "
        "All three winner-seed checkpoints are retained. The 07 test checkpoint/ensemble "
        "policy must be locked before any test evaluation."
    )
}

with open(WINNER_JSON, "w", encoding="utf-8") as f:
    json.dump(winner_payload, f, ensure_ascii=False, indent=2)

summary_payload = {
    "project_version": "v4_repro",
    "created_at": datetime.now().isoformat(),
    "script": "06_best_model_multiseed_v4.py",
    "purpose": "validation-only Top-10-distinct multiseed robustness and selection",
    "grid_script_sha256_verified": True,
    "grid_script_sha256": current_grid_sha,
    "candidate_set": "Top-10 distinct",
    "candidate_count": EXPECTED_CANDIDATE_COUNT,
    "source_ranks": EXPECTED_DISTINCT_SOURCE_RANKS,
    "seeds": SEEDS,
    "expected_total_runs": EXPECTED_TOTAL_RUNS,
    "successful_unique_runs": int(len(success_runs)),
    "all_30_success": bool(len(success_runs) == EXPECTED_TOTAL_RUNS),
    "min_epochs_before_stop": MIN_EPOCHS_BEFORE_STOP,
    "patience": PATIENCE,
    "max_epochs": MAX_EPOCHS,
    "selection_rule": "lowest mean ValidationScore",
    "tie_break": "lower seed-to-seed sample std of ValidationScore",
    "std_ddof": STD_DDOF,
    "test_arrays_loaded": False,
    "test_metrics_computed": False,
    "candidates_csv": CANDIDATES_CSV,
    "runs_csv": RUNS_CSV,
    "summary_csv": SUMMARY_CSV,
    "ranked_summary_csv": RANKED_SUMMARY_CSV,
    "winner_json": WINNER_JSON
}

with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
    json.dump(summary_payload, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 90)
print("06_best_model_multiseed_v4.py TAMAMLANDI")
print("=" * 90)
print("Başarılı unique run:", len(success_runs), "/", EXPECTED_TOTAL_RUNS)
print("Candidate count      :", len(ranked_summary_df))
print("Seeds                :", SEEDS)
print("Test arrays loaded   : False")
print("Test metrics computed: False")

print("\nMULTISEED RANKING:")
print(ranked_summary_df[[
    "multiseed_rank",
    "source_rank",
    "architecture",
    "loss_strategy",
    "lookback",
    "size",
    "feature_set",
    "mean_validation_score",
    "std_validation_score_sample",
    "mean_avg_return_ratio",
    "mean_avg_vol_ratio"
]].to_string(index=False))

print("\nWINNER CONFIG:")
print("config_id :", winner_config_id)
print("mean score:", float(winner["mean_validation_score"]))
print("std score :", float(winner["std_validation_score_sample"]))
print("source rank:", int(winner["source_rank"]))

print("\nKURAL KONTROLÜ:")
print("[OK] 05 grid script SHA-256 manifest ile eşleşti.")
print("[OK] Top-10 distinct programatik üretildi.")
print("[OK] Source ranks tam olarak [1,2,4,5,6,7,8,10,11,12].")
print("[OK] Seeds tam olarak [123,777,2026].")
print("[OK] 30/30 unique (config_id, seed) success doğrulandı.")
print("[OK] MIN_EPOCHS=45, PATIENCE=15, MAX_EPOCHS=100 uygulandı.")
print("[OK] Config-level seçim mean ValidationScore ile yapıldı.")
print("[OK] Tie-break sample std (ddof=1) ile tanımlandı.")
print("[OK] Resume yalnızca status == success pair'leri atlıyor.")
print("[OK] Her run için gerçek CPU-clone best checkpoint saklandı.")
print("[OK] Test dizileri yüklenmedi.")
print("[OK] Test metriği hesaplanmadı.")
print("=" * 90)
