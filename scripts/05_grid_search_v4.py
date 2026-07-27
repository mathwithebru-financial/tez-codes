# ==========================================================
# 05_grid_search_v4.py
#
# RESMÎ ANA GRID:
#   4 mimari × 5 loss × 4 lookback × 3 boyut × 2 feature set = 480 config
#   Maksimum 50 epoch
#   İlk 20 epoch içinde early stopping yok
#   20. epoch sonrası patience=10
#   Seed=42
#   Model seçimi yalnızca validation ile
#   Test dizileri YÜKLENMEZ ve test metriği HESAPLANMAZ
#
# v4 metodolojik düzeltmeleri:
#   - Frozen raw veri hattı
#   - Düzeltilmiş RSI14
#   - Target-realization-aware split
#   - StandardScaler yalnızca train'e fit
#   - Shared-parameter-only PCGrad
#   - Best checkpoint gerçek CPU clone
#   - Resume yalnızca status == success configleri atlar
# ==========================================================

import os
import json
import pickle
import random
import itertools
import copy
import warnings
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
RESULTS_DIR = os.path.join(BASE_DIR, "results", "grid_search")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
HISTORY_DIR = os.path.join(RESULTS_DIR, "histories")
MODEL_DIR = os.path.join(BASE_DIR, "models", "grid_search")

for path in [RESULTS_DIR, METRICS_DIR, HISTORY_DIR, MODEL_DIR]:
    os.makedirs(path, exist_ok=True)

RESULTS_CSV = os.path.join(RESULTS_DIR, "grid_results_v4.csv")
RANKED_CSV = os.path.join(RESULTS_DIR, "grid_results_ranked_v4.csv")
TOP10_CSV = os.path.join(RESULTS_DIR, "grid_top10_v4.csv")
PROGRESS_JSON = os.path.join(RESULTS_DIR, "grid_progress_v4.json")
PREFLIGHT_CSV = os.path.join(RESULTS_DIR, "grid_preflight_audit_v4.csv")
SUMMARY_JSON = os.path.join(RESULTS_DIR, "grid_summary_v4.json")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "grid_best_validation_model_v4.pt")


# ==========================================================
# 2. RESMÎ GRID AYARLARI
# ==========================================================

SEED = 42
BATCH_SIZE = 64

MAX_EPOCHS = 50
MIN_EPOCHS_BEFORE_STOP = 20
PATIENCE = 10

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
DROPOUT = 0.10
TAU = 0.5

# Debug için None yerine küçük sayı verilebilir.
# Resmî koşuda None kalmalı.
MAX_CONFIGS = None


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


set_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("05 — v4 RESMÎ 480-CONFIG ANA GRID")
print("=" * 80)
print("[DEVICE]", DEVICE)

if DEVICE.type != "cuda":
    raise RuntimeError(
        "GPU aktif değil. Colab > Çalışma zamanı türünü değiştir > T4 GPU seç."
    )

print("[GPU]", torch.cuda.get_device_name(0))


# ==========================================================
# 4. SCHEMA + DENOMINATOR
# ==========================================================

SCHEMA_PATH = os.path.join(CONFIG_DIR, "schema_v4.json")
DENOMINATOR_PATH = os.path.join(
    PROCESSED_DIR,
    "selection_baseline_denominators_v4.json"
)

for path in [SCHEMA_PATH, DENOMINATOR_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Gerekli dosya bulunamadı:\n{path}")

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    schema = json.load(f)

with open(DENOMINATOR_PATH, "r", encoding="utf-8") as f:
    denominators = json.load(f)

ASSET_ORDER = schema["data"]["assets"]
TARGET_NAMES = schema["targets"]["definition"]
LOOKBACKS = list(schema["sequence"]["lookbacks"])
ARCHITECTURES = list(schema["models"])
LOSS_STRATEGIES = list(schema["loss_strategies"])
MODEL_SIZES = list(schema["model_sizes"].keys())
FEATURE_SETS = ["baseline", "full"]

EXPECTED_ARCHITECTURES = [
    "FullSharingMTL",
    "PartialSharingMTL",
    "HierarchicalMTL",
    "NoSharing"
]

EXPECTED_LOSSES = [
    "FixedLambda_0.3",
    "FixedLambda_0.5",
    "FixedLambda_0.7",
    "UncertaintyWeighting",
    "PCGrad"
]

if ASSET_ORDER != ["BIST100", "USDTRY", "EURTRY", "GOLD"]:
    raise RuntimeError(f"Asset sırası beklenenden farklı: {ASSET_ORDER}")

if len(TARGET_NAMES) != 8:
    raise RuntimeError(f"Target sayısı 8 değil: {len(TARGET_NAMES)}")

if ARCHITECTURES != EXPECTED_ARCHITECTURES:
    raise RuntimeError(
        "Mimari listesi schema_v4 ile beklenen resmî sırada değil.\n"
        f"Beklenen: {EXPECTED_ARCHITECTURES}\n"
        f"Gerçek   : {ARCHITECTURES}"
    )

if LOSS_STRATEGIES != EXPECTED_LOSSES:
    raise RuntimeError(
        "Loss listesi schema_v4 ile beklenen resmî sırada değil.\n"
        f"Beklenen: {EXPECTED_LOSSES}\n"
        f"Gerçek   : {LOSS_STRATEGIES}"
    )

EXPECTED_TOTAL_CONFIGS = (
    len(ARCHITECTURES)
    * len(LOSS_STRATEGIES)
    * len(LOOKBACKS)
    * len(MODEL_SIZES)
    * len(FEATURE_SETS)
)

if EXPECTED_TOTAL_CONFIGS != 480:
    raise RuntimeError(
        f"Resmî grid 480 config değil: {EXPECTED_TOTAL_CONFIGS}"
    )

print("\n[RESMÎ GRID]")
print("Architectures :", ARCHITECTURES)
print("Losses        :", LOSS_STRATEGIES)
print("Lookbacks     :", LOOKBACKS)
print("Model sizes   :", MODEL_SIZES)
print("Feature sets  :", FEATURE_SETS)
print("Total configs :", EXPECTED_TOTAL_CONFIGS)
print("MAX_EPOCHS    :", MAX_EPOCHS)
print("MIN_EPOCHS    :", MIN_EPOCHS_BEFORE_STOP)
print("PATIENCE      :", PATIENCE)
print("Seed          :", SEED)
print("Test access   : NONE")


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


def make_loaders(X_train, y_train, X_val, y_val):
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )

    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

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
# 17. TEK-BATCH PREFLIGHT — 4 MİMARİ × 5 LOSS
# ==========================================================

def run_preflight():
    print("\n" + "=" * 80)
    print("PREFLIGHT AUDIT — 4 MİMARİ × 5 LOSS")
    print("=" * 80)

    (
        X_train,
        y_train,
        _,
        _,
        _,
        _,
        _
    ) = load_sequence_data("baseline", 10)

    batch_n = min(BATCH_SIZE, len(X_train))

    X_batch = torch.tensor(
        X_train[:batch_n],
        dtype=torch.float32,
        device=DEVICE
    )
    y_batch = torch.tensor(
        y_train[:batch_n],
        dtype=torch.float32,
        device=DEVICE
    )

    rows = []

    for architecture in ARCHITECTURES:
        for loss_strategy in LOSS_STRATEGIES:
            set_seed(SEED)

            model = None
            optimizer = None
            uncertainty_module = None

            try:
                model = build_model(
                    architecture=architecture,
                    n_features=X_train.shape[2],
                    lookback=10,
                    size_name="small"
                ).to(DEVICE)

                group_info = validate_pcgrad_groups(model)

                optimizer_params = list(model.parameters())

                if loss_strategy == "UncertaintyWeighting":
                    uncertainty_module = UncertaintyWeightingLoss().to(DEVICE)
                    optimizer_params = (
                        optimizer_params
                        + list(uncertainty_module.parameters())
                    )

                optimizer = torch.optim.AdamW(
                    optimizer_params,
                    lr=LEARNING_RATE,
                    weight_decay=WEIGHT_DECAY
                )

                optimizer.zero_grad(set_to_none=True)
                pred = model(X_batch)

                if pred.shape != (batch_n, 8):
                    raise RuntimeError(
                        f"Preflight output shape yanlış: {pred.shape}"
                    )

                ret_loss, vol_loss = split_task_losses(pred, y_batch)
                conflict = False
                shared_dot = None

                if loss_strategy.startswith("FixedLambda"):
                    loss = fixed_lambda_loss(
                        ret_loss,
                        vol_loss,
                        loss_strategy
                    )
                    loss.backward()

                elif loss_strategy == "UncertaintyWeighting":
                    loss = uncertainty_module(ret_loss, vol_loss)
                    loss.backward()

                elif loss_strategy == "PCGrad":
                    loss = 0.5 * ret_loss + 0.5 * vol_loss
                    info = pcgrad_backward_equal_weight(
                        model,
                        ret_loss,
                        vol_loss
                    )
                    conflict = bool(info["conflict"])
                    shared_dot = info["shared_dot"]

                else:
                    raise ValueError(loss_strategy)

                if not torch.isfinite(loss):
                    raise RuntimeError("Preflight loss finite değil.")

                torch.nn.utils.clip_grad_norm_(
                    optimizer_params,
                    GRAD_CLIP
                )
                optimizer.step()

                rows.append({
                    "architecture": architecture,
                    "loss_strategy": loss_strategy,
                    "status": "success",
                    "parameter_count": int(
                        sum(p.numel() for p in model.parameters())
                    ),
                    "shared_param_count": group_info["shared_param_count"],
                    "return_specific_param_count": group_info[
                        "return_specific_param_count"
                    ],
                    "vol_specific_param_count": group_info[
                        "vol_specific_param_count"
                    ],
                    "pcgrad_conflict": conflict,
                    "pcgrad_shared_dot": shared_dot,
                    "output_shape": str(tuple(pred.shape)),
                    "error": ""
                })

                print(
                    f"[OK] {architecture:20s} | "
                    f"{loss_strategy:22s} | "
                    f"shared={group_info['shared_param_count']}"
                )

            except Exception as error:
                rows.append({
                    "architecture": architecture,
                    "loss_strategy": loss_strategy,
                    "status": "error",
                    "parameter_count": np.nan,
                    "shared_param_count": np.nan,
                    "return_specific_param_count": np.nan,
                    "vol_specific_param_count": np.nan,
                    "pcgrad_conflict": False,
                    "pcgrad_shared_dot": np.nan,
                    "output_shape": "",
                    "error": repr(error)
                })

                print(
                    f"[ERROR] {architecture} | {loss_strategy} | {repr(error)}"
                )

            finally:
                try:
                    del model
                    del optimizer
                    del uncertainty_module
                except Exception:
                    pass
                cleanup_cuda()

    preflight_df = pd.DataFrame(rows)
    preflight_df.to_csv(PREFLIGHT_CSV, index=False)

    if len(preflight_df) != 20:
        raise RuntimeError(
            f"Preflight 20 kombinasyon üretmedi: {len(preflight_df)}"
        )

    if not (preflight_df["status"] == "success").all():
        failed = preflight_df.loc[
            preflight_df["status"] != "success",
            ["architecture", "loss_strategy", "error"]
        ]
        raise RuntimeError(
            "Preflight tamamlanmadı. 480-grid başlamayacak.\n"
            + failed.to_string(index=False)
        )

    print("\n✅ PREFLIGHT 20/20 SUCCESS. Ana grid başlayabilir.")
    return preflight_df


preflight_df = run_preflight()


# ==========================================================
# 18. 480 CONFIG GRID OLUŞTUR
# ==========================================================

grid = []

for (
    architecture,
    loss_strategy,
    lookback,
    size_name,
    feature_set
) in itertools.product(
    ARCHITECTURES,
    LOSS_STRATEGIES,
    LOOKBACKS,
    MODEL_SIZES,
    FEATURE_SETS
):
    config_id = (
        f"arch={architecture}"
        f"__loss={loss_strategy}"
        f"__lb={lookback}"
        f"__size={size_name}"
        f"__feat={feature_set}"
    )

    grid.append({
        "config_id": config_id,
        "architecture": architecture,
        "loss_strategy": loss_strategy,
        "lookback": int(lookback),
        "size": size_name,
        "feature_set": feature_set
    })

if MAX_CONFIGS is not None:
    grid = grid[:int(MAX_CONFIGS)]

print("\n[GRID]")
print("Toplam config:", len(grid))

if MAX_CONFIGS is None and len(grid) != 480:
    raise RuntimeError(f"Resmî grid 480 değil: {len(grid)}")


# ==========================================================
# 19. RESULT SCHEMA + RESUME
# ==========================================================

RATIO_COLUMNS = []
for asset in ASSET_ORDER:
    RATIO_COLUMNS.append(f"{asset}_return_ratio")
    RATIO_COLUMNS.append(f"{asset}_vol_ratio")

RESULT_COLUMNS = [
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
    "test_arrays_loaded",
    "test_metrics_computed"
] + RATIO_COLUMNS


if os.path.exists(RESULTS_CSV):
    existing_results = pd.read_csv(RESULTS_CSV)

    missing_cols = [
        col
        for col in RESULT_COLUMNS
        if col not in existing_results.columns
    ]

    if missing_cols:
        raise RuntimeError(
            "Mevcut grid_results_v4.csv schema ile uyuşmuyor. "
            f"Eksik kolonlar: {missing_cols}"
        )
else:
    existing_results = pd.DataFrame(columns=RESULT_COLUMNS)


completed_config_ids = set(
    existing_results.loc[
        existing_results["status"] == "success",
        "config_id"
    ].astype(str).tolist()
)

print("Tamamlanan başarılı config:", len(completed_config_ids))


def append_result(row):
    row_full = {
        col: row.get(col, np.nan)
        for col in RESULT_COLUMNS
    }

    row_df = pd.DataFrame([row_full], columns=RESULT_COLUMNS)

    if os.path.exists(RESULTS_CSV):
        row_df.to_csv(
            RESULTS_CSV,
            mode="a",
            header=False,
            index=False
        )
    else:
        row_df.to_csv(
            RESULTS_CSV,
            index=False
        )


def save_progress(current_idx, total, best_score, best_config_id):
    progress = {
        "updated_at": datetime.now().isoformat(),
        "current_idx": int(current_idx),
        "total": int(total),
        "completed_success": int(len(completed_config_ids)),
        "best_score": (
            None if not np.isfinite(best_score) else float(best_score)
        ),
        "best_config_id": best_config_id,
        "results_csv": RESULTS_CSV,
        "test_arrays_loaded": False,
        "test_metrics_computed": False
    }

    with open(PROGRESS_JSON, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


# ==========================================================
# 20. BAŞLANGIÇ BEST
# ==========================================================

best_score = np.inf
best_config_id = None

if not existing_results.empty:
    success_existing = existing_results[
        existing_results["status"] == "success"
    ].copy()

    success_existing = success_existing.drop_duplicates(
        subset=["config_id"],
        keep="last"
    )

    if len(success_existing) > 0:
        best_row = success_existing.sort_values(
            "validation_score",
            ascending=True
        ).iloc[0]

        best_score = float(best_row["validation_score"])
        best_config_id = str(best_row["config_id"])

print("\n[BAŞLANGIÇ BEST]")
print("best_score:", best_score)
print("best_config_id:", best_config_id)

if best_config_id is not None and not os.path.exists(BEST_MODEL_PATH):
    print(
        "⚠️ Uyarı: Mevcut success sonuçları var ancak global best checkpoint "
        "dosyası bulunamadı. Yeni bir config mevcut best'i geçerse yeniden oluşur."
    )


# ==========================================================
# 21. ANA GRID LOOP
# ==========================================================

for idx, cfg in enumerate(grid, start=1):
    config_id = cfg["config_id"]

    if config_id in completed_config_ids:
        print(f"[SKIP] ({idx}/{len(grid)}) Zaten success: {config_id}")
        continue

    print("\n" + "=" * 80)
    print(f"[CONFIG {idx}/{len(grid)}]")
    print(config_id)
    print("=" * 80)

    start_time = datetime.now()

    model = None
    optimizer = None
    uncertainty_loss_module = None
    train_loader = None
    val_loader = None

    try:
        set_seed(SEED)

        (
            X_train,
            y_train,
            X_val,
            y_val,
            y_val_raw,
            y_scaler,
            max_inverse_diff
        ) = load_sequence_data(
            feature_set=cfg["feature_set"],
            lookback=cfg["lookback"]
        )

        train_loader, val_loader = make_loaders(
            X_train,
            y_train,
            X_val,
            y_val
        )

        model = build_model(
            architecture=cfg["architecture"],
            n_features=X_train.shape[2],
            lookback=cfg["lookback"],
            size_name=cfg["size"]
        ).to(DEVICE)

        group_info = validate_pcgrad_groups(model)
        parameter_count = int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )

        model_params = list(model.parameters())
        optimizer_params = list(model_params)
        loss_parameter_count = 0

        if cfg["loss_strategy"] == "UncertaintyWeighting":
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
                loss_strategy=cfg["loss_strategy"],
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
                "config_id": config_id,
                "epoch": epoch,
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
                f"Epoch {epoch:02d} | "
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

        metrics_path = os.path.join(
            METRICS_DIR,
            f"metrics_{config_safe}.csv"
        )
        history_path = os.path.join(
            HISTORY_DIR,
            f"history_{config_safe}.csv"
        )

        metrics_df.to_csv(metrics_path, index=False)
        pd.DataFrame(history_rows).to_csv(history_path, index=False)

        row = {
            "config_id": config_id,
            "architecture": cfg["architecture"],
            "loss_strategy": cfg["loss_strategy"],
            "lookback": cfg["lookback"],
            "size": cfg["size"],
            "feature_set": cfg["feature_set"],
            "seed": SEED,
            "max_epochs": MAX_EPOCHS,
            "min_epochs_before_stop": MIN_EPOCHS_BEFORE_STOP,
            "patience": PATIENCE,
            "best_epoch": best_epoch,
            "epochs_ran": epochs_ran,
            "validation_score": score_obj["validation_score"],
            "avg_return_ratio": score_obj["avg_return_ratio"],
            "avg_vol_ratio": score_obj["avg_vol_ratio"],
            "catastrophic_max_ratio": score_obj[
                "catastrophic_max_ratio"
            ],
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

        append_result(row)

        if score_obj["validation_score"] < best_score:
            best_score = float(score_obj["validation_score"])
            best_config_id = config_id

            torch.save({
                "model_state_dict": best_epoch_payload[
                    "model_state_dict"
                ],
                "uncertainty_state_dict": best_epoch_payload[
                    "uncertainty_state_dict"
                ],
                "config": copy.deepcopy(cfg),
                "epoch": best_epoch,
                "validation_score": best_score,
                "score_obj": score_obj,
                "target_names": TARGET_NAMES,
                "asset_order": ASSET_ORDER,
                "parameter_count": parameter_count,
                "loss_parameter_count": loss_parameter_count,
                "total_trainable_parameter_count": total_trainable_parameter_count,
                "pcgrad_group_counts": group_info,
                "test_arrays_loaded": False,
                "test_metrics_computed": False
            }, BEST_MODEL_PATH)

            print("[BEST UPDATE] Yeni en iyi validation modeli kaydedildi.")
            print("Best score:", best_score)
            print("Best config:", best_config_id)

        completed_config_ids.add(config_id)

        save_progress(
            current_idx=idx,
            total=len(grid),
            best_score=best_score,
            best_config_id=best_config_id
        )

        print(
            f"[SUCCESS] best_epoch={best_epoch} | "
            f"best_score={best_cfg_score:.6f} | "
            f"elapsed={elapsed_seconds:.1f}s"
        )

    except Exception as error:
        elapsed_seconds = (
            datetime.now() - start_time
        ).total_seconds()

        error_row = {
            "config_id": config_id,
            "architecture": cfg["architecture"],
            "loss_strategy": cfg["loss_strategy"],
            "lookback": cfg["lookback"],
            "size": cfg["size"],
            "feature_set": cfg["feature_set"],
            "seed": SEED,
            "max_epochs": MAX_EPOCHS,
            "min_epochs_before_stop": MIN_EPOCHS_BEFORE_STOP,
            "patience": PATIENCE,
            "elapsed_seconds": elapsed_seconds,
            "status": "error",
            "error_message": repr(error),
            "test_arrays_loaded": False,
            "test_metrics_computed": False
        }

        append_result(error_row)

        print("[ERROR]", config_id)
        print(repr(error))

        save_progress(
            current_idx=idx,
            total=len(grid),
            best_score=best_score,
            best_config_id=best_config_id
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
# 22. FINAL RANKING + TOP10 + SUMMARY
# ==========================================================

if not os.path.exists(RESULTS_CSV):
    raise RuntimeError("Grid sonuç dosyası oluşmadı.")

results_df = pd.read_csv(RESULTS_CSV)

success_df = results_df[
    results_df["status"] == "success"
].copy()

success_df = success_df.drop_duplicates(
    subset=["config_id"],
    keep="last"
)

success_df = success_df.sort_values(
    ["validation_score", "config_id"],
    ascending=[True, True]
).reset_index(drop=True)

success_df.insert(
    0,
    "rank",
    np.arange(1, len(success_df) + 1)
)

success_df.to_csv(RANKED_CSV, index=False)
success_df.head(10).to_csv(TOP10_CSV, index=False)

summary = {
    "project_version": "v4_repro",
    "created_at": datetime.now().isoformat(),
    "script": "05_grid_search_v4.py",
    "purpose": "official validation-only 480-config main grid",
    "expected_total_configs": 480,
    "actual_grid_configs": int(len(grid)),
    "unique_success_configs": int(len(success_df)),
    "all_480_success": bool(len(success_df) == 480),
    "seed": SEED,
    "max_epochs": MAX_EPOCHS,
    "min_epochs_before_stop": MIN_EPOCHS_BEFORE_STOP,
    "patience": PATIENCE,
    "selection_rule": (
        "ValidationScore = 0.5 * AvgReturnRatio + 0.5 * AvgVolRatio"
    ),
    "pcgrad_method": (
        "Equal-weight shared-parameter-only PCGrad. Projection is applied "
        "only on parameters used by both tasks when gradient dot product is negative. "
        "Task-specific parameters retain their own task gradient scaled by 0.5. "
        "NoSharing therefore reduces exactly to FixedLambda_0.5."
    ),
    "head_policy": (
        "FullSharingMTL, PartialSharingMTL and NoSharing use the same two-layer "
        "task-head template. HierarchicalMTL keeps its hierarchy-specific hidden path."
    ),
    "test_arrays_loaded": False,
    "test_metrics_computed": False,
    "preflight_combinations": int(len(preflight_df)),
    "preflight_all_success": bool(
        (preflight_df["status"] == "success").all()
    ),
    "results_csv": RESULTS_CSV,
    "ranked_csv": RANKED_CSV,
    "top10_csv": TOP10_CSV,
    "best_model_path": BEST_MODEL_PATH
}

if len(success_df) > 0:
    best_row = success_df.iloc[0]
    summary["best"] = {
        "config_id": str(best_row["config_id"]),
        "validation_score": float(best_row["validation_score"]),
        "avg_return_ratio": float(best_row["avg_return_ratio"]),
        "avg_vol_ratio": float(best_row["avg_vol_ratio"]),
        "catastrophic_max_ratio": float(
            best_row["catastrophic_max_ratio"]
        ),
        "best_epoch": int(best_row["best_epoch"])
    }

with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 80)
print("05_grid_search_v4.py TAMAMLANDI")
print("=" * 80)
print("Başarılı unique config:", len(success_df), "/ 480")
print("Toplam kayıt:", len(results_df))

if len(success_df) > 0:
    print("\nEn iyi config:")
    print(
        success_df.iloc[0][[
            "config_id",
            "validation_score",
            "avg_return_ratio",
            "avg_vol_ratio",
            "catastrophic_max_ratio",
            "best_epoch"
        ]]
    )

print("\nDosyalar:")
print(" -", RESULTS_CSV)
print(" -", RANKED_CSV)
print(" -", TOP10_CSV)
print(" -", PROGRESS_JSON)
print(" -", PREFLIGHT_CSV)
print(" -", SUMMARY_JSON)
print(" -", BEST_MODEL_PATH)

print("\nKURAL KONTROLÜ:")
print("✅ Preflight 20/20 kombinasyonu doğruladı.")
print("✅ 4 mimari kullanıldı.")
print("✅ 5 loss stratejisi kullanıldı.")
print("✅ 4 lookback kullanıldı.")
print("✅ 3 model boyutu kullanıldı.")
print("✅ 2 feature set kullanıldı.")
print("✅ Shared-parameter-only PCGrad kullanıldı.")
print("✅ Best checkpoint gerçek CPU clone ile saklandı.")
print("✅ Resume yalnızca status == success configleri atlıyor.")
print("✅ Model seçimi yalnızca validation ile yapıldı.")
print("✅ Test dizileri yüklenmedi.")
print("✅ Test metriği hesaplanmadı.")

if len(success_df) != 480:
    raise RuntimeError(
        f"Ana grid henüz 480/480 success değil: {len(success_df)}/480. "
        "Aynı script yeniden çalıştırıldığında yalnızca success olmayan configler koşar."
    )

print("=" * 80)
