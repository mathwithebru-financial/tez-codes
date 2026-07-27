# ==========================================================
# 07_final_test_evaluation_v4.py
#
# RESMÎ NİHAİ TEST DEĞERLENDİRMESİ
#
# Kilitli giriş:
#   Winner config:
#     NoSharing + FixedLambda_0.7 + lb10 + small + baseline
#
#   Winner seeds:
#     [123, 777, 2026]
#
#   Ana final tahmin politikası:
#     Üç kilitli winner-seed checkpoint tahmininin
#     HAM ÖLÇEKTE aritmetik ortalaması (3-seed ensemble)
#
#   Ek raporlama:
#     Her seed checkpoint'i ayrıca test edilir ve raporlanır.
#
# Kritik kurallar:
#   - 07 içinde model seçimi YOKTUR.
#   - Test sonucuna bakarak mimari/loss/lookback/size/feature değişmez.
#   - Test, yalnızca 06 ile kilitlenmiş winner için açılır.
#   - 06 winner JSON değiştirilmez.
#   - 06 checkpointleri değiştirilmez.
#   - 07 yeni eğitim yapmaz.
#   - 07 baseline karşılaştırması yapmaz; bu 08'in görevidir.
#   - 07 istatistiksel model karşılaştırması yapmaz; bu 09'un görevidir.
#
# Çıktılar:
#   results/final_test/
#     final_test_metrics_long_v4.csv
#     final_test_summary_v4.csv
#     final_test_summary_v4.json
#     final_test_y_true_raw_v4.npy
#     pred_final_seed123_raw_v4.npy
#     pred_final_seed777_raw_v4.npy
#     pred_final_seed2026_raw_v4.npy
#     pred_final_ensemble_raw_v4.npy
#
# ==========================================================

import os
import json
import pickle
import random
import hashlib
import warnings
from datetime import datetime, timezone

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

MULTISEED_RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "multiseed"
)

WINNER_JSON = os.path.join(
    MULTISEED_RESULTS_DIR,
    "multiseed_winner_config_v4.json"
)

CODE_MANIFEST_PATH = os.path.join(
    CONFIG_DIR,
    "code_manifest_v4.csv"
)

SCRIPT_06_PATH = os.path.join(
    SCRIPTS_DIR,
    "06_best_model_multiseed_v4.py"
)

SCHEMA_PATH = os.path.join(
    CONFIG_DIR,
    "schema_v4.json"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "final_test"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

METRICS_LONG_CSV = os.path.join(
    RESULTS_DIR,
    "final_test_metrics_long_v4.csv"
)

SUMMARY_CSV = os.path.join(
    RESULTS_DIR,
    "final_test_summary_v4.csv"
)

SUMMARY_JSON = os.path.join(
    RESULTS_DIR,
    "final_test_summary_v4.json"
)

Y_TRUE_RAW_PATH = os.path.join(
    RESULTS_DIR,
    "final_test_y_true_raw_v4.npy"
)

ENSEMBLE_PRED_PATH = os.path.join(
    RESULTS_DIR,
    "pred_final_ensemble_raw_v4.npy"
)


# ==========================================================
# 2. 07 ÖNCESİ KİLİTLİ KARARLAR
# ==========================================================

EXPECTED_WINNER = {
    "architecture": "NoSharing",
    "loss_strategy": "FixedLambda_0.7",
    "lookback": 10,
    "size": "small",
    "feature_set": "baseline",
}

EXPECTED_WINNER_CONFIG_ID = (
    "arch=NoSharing"
    "__loss=FixedLambda_0.7"
    "__lb=10"
    "__size=small"
    "__feat=baseline"
)

EXPECTED_SEEDS = [123, 777, 2026]

EXPECTED_06_SHA256 = (
    "35de2ee398699003dfef6be36b70c112f"
    "b2c0d1b1e9577cbf64bef58877e16d8"
)

PRIMARY_TEST_POLICY = (
    "arithmetic_mean_of_three_locked_winner_seed_predictions_in_raw_scale"
)

BATCH_SIZE = 64
DROPOUT = 0.10
TAU = 0.5

EXPECTED_TEST_SAMPLES = 584

TEST_ACCESS_STARTED = False
TEST_METRICS_COMPUTED = False


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


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 100)
print("07 — v4 RESMÎ NİHAİ TEST DEĞERLENDİRMESİ")
print("=" * 100)
print("[DEVICE]", DEVICE)

if DEVICE.type == "cuda":
    print("[GPU]", torch.cuda.get_device_name(0))
else:
    print("[INFO] GPU yok; evaluation CPU üzerinde çalışacak.")

print("[PRIMARY TEST POLICY]")
print("  3 locked winner-seed checkpoint prediction arithmetic mean")
print("  averaging scale: RAW")
print("[INDIVIDUAL SEEDS]", EXPECTED_SEEDS)
print("[MODEL SELECTION INSIDE 07] NONE")
print("[TEST ACCESS BEFORE PREFLIGHT] NONE")


# ==========================================================
# 4. YARDIMCI FONKSİYONLAR
# ==========================================================

def sha256_file(path, chunk_size=1024 * 1024):
    sha = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            sha.update(chunk)

    return sha.hexdigest()


def clone_json_safe(value):
    if isinstance(value, dict):
        return {
            str(k): clone_json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            clone_json_safe(v)
            for v in value
        ]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    return value


def normalize_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if value is None:
        return False

    text = str(value).strip().lower()

    if text in {"true", "1", "yes"}:
        return True

    if text in {"false", "0", "no", ""}:
        return False

    raise ValueError(
        f"Boolean değere çevrilemeyen kayıt: {value!r}"
    )


def load_checkpoint_cpu(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu"
        )


def mae_np(y_true, y_pred):
    return float(
        np.mean(
            np.abs(y_true - y_pred)
        )
    )


def rmse_np(y_true, y_pred):
    return float(
        np.sqrt(
            np.mean(
                (y_true - y_pred) ** 2
            )
        )
    )


def r2_np(y_true, y_pred):
    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    if ss_tot == 0:
        return float("nan")

    return float(
        1.0 - ss_res / ss_tot
    )


def pinball_np(y_true, y_pred, tau=0.5):
    diff = y_true - y_pred

    loss = np.maximum(
        tau * diff,
        (tau - 1.0) * diff
    )

    return float(
        np.mean(loss)
    )


# ==========================================================
# 5. PREFLIGHT — TEST AÇILMADAN ÖNCE
# ==========================================================

required_preflight_paths = [
    WINNER_JSON,
    CODE_MANIFEST_PATH,
    SCRIPT_06_PATH,
    SCHEMA_PATH,
]

missing = [
    path
    for path in required_preflight_paths
    if not os.path.exists(path)
]

if missing:
    raise FileNotFoundError(
        "07 preflight için gerekli dosyalar eksik:\n"
        + "\n".join(missing)
    )


# ----------------------------------------------------------
# 5.1 06 SCRIPT PROVENANCE
# ----------------------------------------------------------

actual_06_sha = sha256_file(
    SCRIPT_06_PATH
)

if actual_06_sha != EXPECTED_06_SHA256:
    raise RuntimeError(
        "06 script SHA-256 beklenen resmî sürümle eşleşmiyor.\n"
        f"Beklenen: {EXPECTED_06_SHA256}\n"
        f"Gerçek  : {actual_06_sha}"
    )

manifest_df = pd.read_csv(
    CODE_MANIFEST_PATH
)

manifest_match = manifest_df[
    (
        manifest_df["script_name"]
        == "06_best_model_multiseed_v4.py"
    )
    &
    (
        manifest_df["sha256"]
        == actual_06_sha
    )
]

if len(manifest_match) < 1:
    raise RuntimeError(
        "06 script mevcut hash ile code manifest içinde bulunamadı."
    )


# ----------------------------------------------------------
# 5.2 SCHEMA
# ----------------------------------------------------------

with open(
    SCHEMA_PATH,
    "r",
    encoding="utf-8"
) as f:
    schema = json.load(f)

ASSET_ORDER = list(
    schema["data"]["assets"]
)

TARGET_NAMES = list(
    schema["targets"]["definition"]
)

EXPECTED_ASSET_ORDER = [
    "BIST100",
    "USDTRY",
    "EURTRY",
    "GOLD",
]

if ASSET_ORDER != EXPECTED_ASSET_ORDER:
    raise RuntimeError(
        f"Asset sırası yanlış: {ASSET_ORDER}"
    )

if len(TARGET_NAMES) != 8:
    raise RuntimeError(
        f"Target sayısı 8 değil: {len(TARGET_NAMES)}"
    )


# ----------------------------------------------------------
# 5.3 WINNER JSON
# ----------------------------------------------------------

with open(
    WINNER_JSON,
    "r",
    encoding="utf-8"
) as f:
    winner_payload = json.load(f)

if normalize_bool(
    winner_payload.get(
        "test_arrays_loaded",
        False
    )
):
    raise RuntimeError(
        "06 winner JSON test_arrays_loaded=True."
    )

if normalize_bool(
    winner_payload.get(
        "test_metrics_computed",
        False
    )
):
    raise RuntimeError(
        "06 winner JSON test_metrics_computed=True."
    )

winner = winner_payload["winner"]

if str(
    winner["config_id"]
) != EXPECTED_WINNER_CONFIG_ID:
    raise RuntimeError(
        "Winner config_id kilitli değerle uyuşmuyor.\n"
        f"Beklenen: {EXPECTED_WINNER_CONFIG_ID}\n"
        f"Gerçek  : {winner['config_id']}"
    )

for key, expected in EXPECTED_WINNER.items():
    actual = winner[key]

    if str(actual) != str(expected):
        raise RuntimeError(
            f"Winner {key} uyuşmuyor.\n"
            f"Beklenen: {expected}\n"
            f"Gerçek  : {actual}"
        )

winner_seed_checkpoints = list(
    winner["seed_checkpoints"]
)

actual_seeds = sorted(
    int(item["seed"])
    for item in winner_seed_checkpoints
)

if actual_seeds != EXPECTED_SEEDS:
    raise RuntimeError(
        "Winner checkpoint seed set yanlış.\n"
        f"Beklenen: {EXPECTED_SEEDS}\n"
        f"Gerçek  : {actual_seeds}"
    )

if len(
    winner_seed_checkpoints
) != 3:
    raise RuntimeError(
        "Winner JSON exact 3 checkpoint içermiyor."
    )


# ----------------------------------------------------------
# 5.4 CHECKPOINT PREFLIGHT
# ----------------------------------------------------------

checkpoint_records = []

for item in sorted(
    winner_seed_checkpoints,
    key=lambda x: int(x["seed"])
):
    seed = int(
        item["seed"]
    )

    checkpoint_path = str(
        item["checkpoint_file"]
    )

    if not os.path.exists(
        checkpoint_path
    ):
        raise FileNotFoundError(
            f"Winner checkpoint bulunamadı:\n{checkpoint_path}"
        )

    checkpoint = load_checkpoint_cpu(
        checkpoint_path
    )

    if str(
        checkpoint["config"]["config_id"]
    ) != EXPECTED_WINNER_CONFIG_ID:
        raise RuntimeError(
            f"Checkpoint config_id yanlış: seed={seed}"
        )

    if int(
        checkpoint["seed"]
    ) != seed:
        raise RuntimeError(
            f"Checkpoint seed metadata yanlış: seed={seed}"
        )

    if int(
        checkpoint["epoch"]
    ) != int(
        item["best_epoch"]
    ):
        raise RuntimeError(
            f"Checkpoint best_epoch yanlış: seed={seed}"
        )

    if not np.isclose(
        float(checkpoint["validation_score"]),
        float(item["validation_score"]),
        rtol=0.0,
        atol=1e-12
    ):
        raise RuntimeError(
            f"Checkpoint validation score yanlış: seed={seed}"
        )

    if normalize_bool(
        checkpoint.get(
            "test_arrays_loaded",
            False
        )
    ):
        raise RuntimeError(
            f"Checkpoint test_arrays_loaded=True: seed={seed}"
        )

    if normalize_bool(
        checkpoint.get(
            "test_metrics_computed",
            False
        )
    ):
        raise RuntimeError(
            f"Checkpoint test_metrics_computed=True: seed={seed}"
        )

    checkpoint_records.append({
        "seed": seed,
        "validation_score": float(
            checkpoint["validation_score"]
        ),
        "best_epoch": int(
            checkpoint["epoch"]
        ),
        "checkpoint_file": checkpoint_path,
        "checkpoint_sha256": sha256_file(
            checkpoint_path
        ),
    })


print("\n" + "-" * 100)
print("PREFLIGHT — TÜM KONTROLLER GEÇTİ")
print("-" * 100)
print("06 SHA-256 manifest eşleşmesi       : TRUE")
print("06 winner config kilitli değerle aynı: TRUE")
print("Winner seeds                        :", actual_seeds)
print("Winner checkpoint count             : 3")
print("Model selection inside 07           : NONE")
print("Test access                         : HENÜZ YOK")


# ==========================================================
# 6. TRANSFORMER YAPITAŞLARI
#    05/06 ile aynı mimari tanımları
# ==========================================================

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_head,
        d_ff,
        n_layers,
        dropout
    ):
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


def make_head(
    d_model,
    dropout
):
    return nn.Sequential(
        nn.Linear(
            d_model,
            d_model
        ),
        nn.GELU(),
        nn.Dropout(
            dropout
        ),
        nn.Linear(
            d_model,
            4
        )
    )


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

        self.input_projection = nn.Linear(
            n_features,
            d_model
        )

        self.positional_embedding = nn.Parameter(
            torch.zeros(
                1,
                lookback,
                d_model
            )
        )

        self.encoder = TransformerBlock(
            d_model,
            n_head,
            d_ff,
            n_layers,
            dropout
        )

        self.norm = nn.LayerNorm(
            d_model
        )

        self.return_head = make_head(
            d_model,
            dropout
        )

        self.vol_head = make_head(
            d_model,
            dropout
        )

    def forward(self, x):
        h = self.input_projection(x)

        h = (
            h
            + self.positional_embedding[
                :,
                :h.size(1),
                :
            ]
        )

        h = self.encoder(h)

        h = self.norm(
            h[:, -1, :]
        )

        ret = self.return_head(h)

        vol = self.vol_head(h)

        return torch.cat(
            [ret, vol],
            dim=1
        )


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

        shared_layers = max(
            1,
            n_layers // 2
        )

        task_layers = max(
            1,
            n_layers - shared_layers
        )

        self.input_projection = nn.Linear(
            n_features,
            d_model
        )

        self.positional_embedding = nn.Parameter(
            torch.zeros(
                1,
                lookback,
                d_model
            )
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

        self.return_norm = nn.LayerNorm(
            d_model
        )

        self.vol_norm = nn.LayerNorm(
            d_model
        )

        self.return_head = make_head(
            d_model,
            dropout
        )

        self.vol_head = make_head(
            d_model,
            dropout
        )

    def forward(self, x):
        h = self.input_projection(x)

        h = (
            h
            + self.positional_embedding[
                :,
                :h.size(1),
                :
            ]
        )

        shared = self.shared_encoder(
            h
        )

        h_ret = self.return_encoder(
            shared
        )

        h_vol = self.vol_encoder(
            shared
        )

        h_ret = self.return_norm(
            h_ret[:, -1, :]
        )

        h_vol = self.vol_norm(
            h_vol[:, -1, :]
        )

        ret = self.return_head(
            h_ret
        )

        vol = self.vol_head(
            h_vol
        )

        return torch.cat(
            [ret, vol],
            dim=1
        )


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

        self.input_projection = nn.Linear(
            n_features,
            d_model
        )

        self.positional_embedding = nn.Parameter(
            torch.zeros(
                1,
                lookback,
                d_model
            )
        )

        self.encoder = TransformerBlock(
            d_model,
            n_head,
            d_ff,
            n_layers,
            dropout
        )

        self.norm = nn.LayerNorm(
            d_model
        )

        self.return_hidden = nn.Sequential(
            nn.Linear(
                d_model,
                d_model
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            )
        )

        self.return_head = nn.Linear(
            d_model,
            4
        )

        self.vol_hidden = nn.Sequential(
            nn.Linear(
                d_model * 2,
                d_model
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            )
        )

        self.vol_head = nn.Linear(
            d_model,
            4
        )

    def forward(self, x):
        h = self.input_projection(x)

        h = (
            h
            + self.positional_embedding[
                :,
                :h.size(1),
                :
            ]
        )

        h = self.encoder(h)

        base = self.norm(
            h[:, -1, :]
        )

        ret_hidden = self.return_hidden(
            base
        )

        ret = self.return_head(
            ret_hidden
        )

        vol_input = torch.cat(
            [base, ret_hidden],
            dim=1
        )

        vol_hidden = self.vol_hidden(
            vol_input
        )

        vol = self.vol_head(
            vol_hidden
        )

        return torch.cat(
            [ret, vol],
            dim=1
        )


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

        self.ret_projection = nn.Linear(
            n_features,
            d_model
        )

        self.ret_positional = nn.Parameter(
            torch.zeros(
                1,
                lookback,
                d_model
            )
        )

        self.ret_encoder = TransformerBlock(
            d_model,
            n_head,
            d_ff,
            n_layers,
            dropout
        )

        self.ret_norm = nn.LayerNorm(
            d_model
        )

        self.return_head = make_head(
            d_model,
            dropout
        )

        self.vol_projection = nn.Linear(
            n_features,
            d_model
        )

        self.vol_positional = nn.Parameter(
            torch.zeros(
                1,
                lookback,
                d_model
            )
        )

        self.vol_encoder = TransformerBlock(
            d_model,
            n_head,
            d_ff,
            n_layers,
            dropout
        )

        self.vol_norm = nn.LayerNorm(
            d_model
        )

        self.vol_head = make_head(
            d_model,
            dropout
        )

    def forward(self, x):
        h_ret = self.ret_projection(
            x
        )

        h_ret = (
            h_ret
            + self.ret_positional[
                :,
                :h_ret.size(1),
                :
            ]
        )

        h_ret = self.ret_encoder(
            h_ret
        )

        h_ret = self.ret_norm(
            h_ret[:, -1, :]
        )

        h_vol = self.vol_projection(
            x
        )

        h_vol = (
            h_vol
            + self.vol_positional[
                :,
                :h_vol.size(1),
                :
            ]
        )

        h_vol = self.vol_encoder(
            h_vol
        )

        h_vol = self.vol_norm(
            h_vol[:, -1, :]
        )

        ret = self.return_head(
            h_ret
        )

        vol = self.vol_head(
            h_vol
        )

        return torch.cat(
            [ret, vol],
            dim=1
        )


def build_model(
    architecture,
    n_features,
    lookback,
    size_name
):
    size_cfg = schema[
        "model_sizes"
    ][size_name]

    kwargs = {
        "n_features": n_features,
        "lookback": lookback,
        "d_model": int(
            size_cfg["d_model"]
        ),
        "n_head": int(
            size_cfg["n_head"]
        ),
        "n_layers": int(
            size_cfg["n_layers"]
        ),
        "d_ff": int(
            size_cfg["d_ff"]
        ),
        "dropout": DROPOUT
    }

    if architecture == "FullSharingMTL":
        return FullSharingMTL(
            **kwargs
        )

    if architecture == "PartialSharingMTL":
        return PartialSharingMTL(
            **kwargs
        )

    if architecture == "HierarchicalMTL":
        return HierarchicalMTL(
            **kwargs
        )

    if architecture == "NoSharing":
        return NoSharing(
            **kwargs
        )

    raise ValueError(
        f"Bilinmeyen mimari: {architecture}"
    )


# ==========================================================
# 7. TESTİ İLK KEZ RESMÎ OLARAK AÇ
# ==========================================================

feature_set = EXPECTED_WINNER[
    "feature_set"
]

lookback = int(
    EXPECTED_WINNER[
        "lookback"
    ]
)

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

test_paths = {
    "X_test": os.path.join(
        seq_dir,
        "X_test.npy"
    ),
    "y_test": os.path.join(
        seq_dir,
        "y_test.npy"
    ),
    "y_test_raw": os.path.join(
        seq_dir,
        "y_test_raw.npy"
    ),
}

for path in list(
    test_paths.values()
) + [scaler_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"07 test dosyası bulunamadı:\n{path}"
        )


TEST_ACCESS_STARTED = True
test_access_started_at = datetime.now(
    timezone.utc
).isoformat()

print("\n" + "=" * 100)
print("FIRST OFFICIAL TEST ACCESS IN v4 PIPELINE")
print("=" * 100)
print("[TEST ACCESS STARTED]", test_access_started_at)
print("[LOCKED WINNER]", EXPECTED_WINNER_CONFIG_ID)
print("[NO MODEL SELECTION] TRUE")
print("[NO HYPERPARAMETER CHANGE] TRUE")


X_test = np.load(
    test_paths["X_test"]
)

y_test = np.load(
    test_paths["y_test"]
)

y_test_raw = np.load(
    test_paths["y_test_raw"]
)

with open(
    scaler_path,
    "rb"
) as f:
    scaler_obj = pickle.load(f)

y_scaler = scaler_obj[
    "y_scaler"
]


# ==========================================================
# 8. TEST VERİ BÜTÜNLÜĞÜ
# ==========================================================

if len(X_test) != EXPECTED_TEST_SAMPLES:
    raise RuntimeError(
        "Test örnek sayısı beklenenden farklı.\n"
        f"Beklenen: {EXPECTED_TEST_SAMPLES}\n"
        f"Gerçek  : {len(X_test)}"
    )

if len(y_test) != EXPECTED_TEST_SAMPLES:
    raise RuntimeError(
        f"y_test örnek sayısı 584 değil: {len(y_test)}"
    )

if len(y_test_raw) != EXPECTED_TEST_SAMPLES:
    raise RuntimeError(
        f"y_test_raw örnek sayısı 584 değil: {len(y_test_raw)}"
    )

if X_test.ndim != 3:
    raise RuntimeError(
        f"X_test 3D değil: {X_test.shape}"
    )

if X_test.shape[1] != lookback:
    raise RuntimeError(
        f"X_test lookback yanlış: {X_test.shape}"
    )

if y_test.shape[1] != 8:
    raise RuntimeError(
        f"y_test target boyutu 8 değil: {y_test.shape}"
    )

if y_test_raw.shape[1] != 8:
    raise RuntimeError(
        f"y_test_raw target boyutu 8 değil: {y_test_raw.shape}"
    )

if not np.isfinite(
    X_test
).all():
    raise RuntimeError(
        "X_test içinde NaN/Inf var."
    )

if not np.isfinite(
    y_test
).all():
    raise RuntimeError(
        "y_test içinde NaN/Inf var."
    )

if not np.isfinite(
    y_test_raw
).all():
    raise RuntimeError(
        "y_test_raw içinde NaN/Inf var."
    )

inverse_check = y_scaler.inverse_transform(
    y_test
)

max_inverse_diff = float(
    np.max(
        np.abs(
            inverse_check
            - y_test_raw
        )
    )
)

if max_inverse_diff > 1e-5:
    raise RuntimeError(
        "Test inverse-scale kontrolü geçmedi.\n"
        f"max diff = {max_inverse_diff}"
    )


print("\n[TEST DATA AUDIT]")
print("X_test shape     :", X_test.shape)
print("y_test shape     :", y_test.shape)
print("y_test_raw shape :", y_test_raw.shape)
print("Inverse max diff :", max_inverse_diff)
print("Test sample count:", len(X_test))


# ==========================================================
# 9. TEST LOADER
# ==========================================================

test_ds = TensorDataset(
    torch.tensor(
        X_test,
        dtype=torch.float32
    ),
    torch.tensor(
        y_test,
        dtype=torch.float32
    )
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False
)


# ==========================================================
# 10. EVALUATION
# ==========================================================

@torch.no_grad()
def predict_scaled(
    model,
    loader
):
    model.eval()

    preds = []
    trues = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(
            DEVICE
        )

        y_pred = model(
            X_batch
        )

        if y_pred.shape[1] != 8:
            raise RuntimeError(
                "Model output target boyutu 8 değil."
            )

        preds.append(
            y_pred.detach().cpu().numpy()
        )

        trues.append(
            y_batch.numpy()
        )

    preds = np.vstack(
        preds
    )

    trues = np.vstack(
        trues
    )

    return preds, trues


def compute_raw_metrics(
    y_true_raw,
    y_pred_raw,
    model_label,
    seed_value=None,
    primary_prediction=False
):
    rows = []

    for i, asset in enumerate(
        ASSET_ORDER
    ):
        true = y_true_raw[
            :,
            i
        ]

        pred = y_pred_raw[
            :,
            i
        ]

        rows.append({
            "model_label": model_label,
            "seed": seed_value,
            "primary_prediction": primary_prediction,
            "task": "return",
            "asset": asset,
            "MAE": mae_np(
                true,
                pred
            ),
            "RMSE": rmse_np(
                true,
                pred
            ),
            "R2": r2_np(
                true,
                pred
            ),
            "PinballLoss_tau_0.5": np.nan,
        })

    for i, asset in enumerate(
        ASSET_ORDER
    ):
        col = 4 + i

        true = y_true_raw[
            :,
            col
        ]

        pred = y_pred_raw[
            :,
            col
        ]

        rows.append({
            "model_label": model_label,
            "seed": seed_value,
            "primary_prediction": primary_prediction,
            "task": "volatility",
            "asset": asset,
            "MAE": mae_np(
                true,
                pred
            ),
            "RMSE": rmse_np(
                true,
                pred
            ),
            "R2": r2_np(
                true,
                pred
            ),
            "PinballLoss_tau_0.5": pinball_np(
                true,
                pred,
                tau=TAU
            ),
        })

    return pd.DataFrame(
        rows
    )


def summarize_metrics(
    metrics_df,
    model_label,
    seed_value,
    primary_prediction
):
    ret = metrics_df[
        metrics_df["task"]
        == "return"
    ]

    vol = metrics_df[
        metrics_df["task"]
        == "volatility"
    ]

    return {
        "model_label": model_label,
        "seed": seed_value,
        "primary_prediction": primary_prediction,
        "avg_return_mae": float(
            ret["MAE"].mean()
        ),
        "avg_return_rmse": float(
            ret["RMSE"].mean()
        ),
        "avg_return_r2": float(
            ret["R2"].mean()
        ),
        "avg_vol_mae": float(
            vol["MAE"].mean()
        ),
        "avg_vol_rmse": float(
            vol["RMSE"].mean()
        ),
        "avg_vol_r2": float(
            vol["R2"].mean()
        ),
        "avg_vol_pinball_tau_0.5": float(
            vol[
                "PinballLoss_tau_0.5"
            ].mean()
        ),
    }


# ==========================================================
# 11. 3 WINNER-SEED CHECKPOINT TESTİ
# ==========================================================

all_metrics = []
summary_rows = []
seed_predictions_raw = {}
true_scaled_reference = None

print("\n" + "=" * 100)
print("3 WINNER-SEED CHECKPOINT TEST DEĞERLENDİRMESİ")
print("=" * 100)

for item in sorted(
    winner_seed_checkpoints,
    key=lambda x: int(x["seed"])
):
    seed = int(
        item["seed"]
    )

    checkpoint_path = str(
        item["checkpoint_file"]
    )

    print("\n" + "-" * 100)
    print(f"[SEED {seed}]")
    print(checkpoint_path)

    set_seed(seed)

    checkpoint = load_checkpoint_cpu(
        checkpoint_path
    )

    model = build_model(
        architecture=EXPECTED_WINNER[
            "architecture"
        ],
        n_features=X_test.shape[2],
        lookback=lookback,
        size_name=EXPECTED_WINNER[
            "size"
        ]
    ).to(
        DEVICE
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True
    )

    preds_scaled, true_scaled = predict_scaled(
        model,
        test_loader
    )

    if true_scaled_reference is None:
        true_scaled_reference = true_scaled.copy()

    else:
        max_true_scaled_diff = float(
            np.max(
                np.abs(
                    true_scaled
                    - true_scaled_reference
                )
            )
        )

        if max_true_scaled_diff != 0.0:
            raise RuntimeError(
                "Seed evaluation sırasında true_scaled değişti."
            )

    true_raw_check = y_scaler.inverse_transform(
        true_scaled
    )

    true_raw_diff = float(
        np.max(
            np.abs(
                true_raw_check
                - y_test_raw
            )
        )
    )

    if true_raw_diff > 1e-5:
        raise RuntimeError(
            f"Seed {seed} true raw inverse check geçmedi."
        )

    preds_raw = y_scaler.inverse_transform(
        preds_scaled
    )

    if not np.isfinite(
        preds_raw
    ).all():
        raise RuntimeError(
            f"Seed {seed} prediction içinde NaN/Inf var."
        )

    seed_predictions_raw[
        seed
    ] = preds_raw.copy()

    seed_pred_path = os.path.join(
        RESULTS_DIR,
        f"pred_final_seed{seed}_raw_v4.npy"
    )

    np.save(
        seed_pred_path,
        preds_raw
    )

    metrics_df = compute_raw_metrics(
        y_true_raw=y_test_raw,
        y_pred_raw=preds_raw,
        model_label=f"FinalWinner_Seed{seed}",
        seed_value=seed,
        primary_prediction=False
    )

    all_metrics.append(
        metrics_df
    )

    summary_rows.append(
        summarize_metrics(
            metrics_df,
            model_label=f"FinalWinner_Seed{seed}",
            seed_value=seed,
            primary_prediction=False
        )
    )

    ret_mae = metrics_df.loc[
        metrics_df["task"]
        == "return",
        "MAE"
    ].mean()

    vol_pb = metrics_df.loc[
        metrics_df["task"]
        == "volatility",
        "PinballLoss_tau_0.5"
    ].mean()

    print(
        f"Avg Return MAE     : {ret_mae:.10f}"
    )

    print(
        f"Avg Vol Pinball    : {vol_pb:.10f}"
    )

    print(
        f"Prediction saved   : {seed_pred_path}"
    )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==========================================================
# 12. ANA FINAL TAHMİN — 3-SEED ENSEMBLE
# ==========================================================

if sorted(
    seed_predictions_raw.keys()
) != EXPECTED_SEEDS:
    raise RuntimeError(
        "Ensemble öncesi exact 3 winner-seed prediction yok."
    )

ensemble_pred_raw = np.mean(
    np.stack(
        [
            seed_predictions_raw[
                seed
            ]
            for seed in EXPECTED_SEEDS
        ],
        axis=0
    ),
    axis=0
)

if ensemble_pred_raw.shape != y_test_raw.shape:
    raise RuntimeError(
        "Ensemble prediction shape y_test_raw ile uyuşmuyor."
    )

if not np.isfinite(
    ensemble_pred_raw
).all():
    raise RuntimeError(
        "Ensemble prediction içinde NaN/Inf var."
    )

ensemble_metrics_df = compute_raw_metrics(
    y_true_raw=y_test_raw,
    y_pred_raw=ensemble_pred_raw,
    model_label="FinalWinner_3SeedEnsemble",
    seed_value=np.nan,
    primary_prediction=True
)

all_metrics.append(
    ensemble_metrics_df
)

summary_rows.append(
    summarize_metrics(
        ensemble_metrics_df,
        model_label="FinalWinner_3SeedEnsemble",
        seed_value=np.nan,
        primary_prediction=True
    )
)


# ==========================================================
# 13. TEST METRİKLERİ TAMAMLANDI
# ==========================================================

TEST_METRICS_COMPUTED = True

test_metrics_computed_at = datetime.now(
    timezone.utc
).isoformat()


# ==========================================================
# 14. ÇIKTILARI KAYDET
# ==========================================================

metrics_long_df = pd.concat(
    all_metrics,
    ignore_index=True
)

summary_df = pd.DataFrame(
    summary_rows
)

np.save(
    Y_TRUE_RAW_PATH,
    y_test_raw
)

np.save(
    ENSEMBLE_PRED_PATH,
    ensemble_pred_raw
)

metrics_long_df.to_csv(
    METRICS_LONG_CSV,
    index=False
)

summary_df.to_csv(
    SUMMARY_CSV,
    index=False
)


# ==========================================================
# 15. SON KONTROLLER
# ==========================================================

if len(metrics_long_df) != 32:
    raise RuntimeError(
        "Final metrics long dosyası 32 satır değil.\n"
        "Beklenen: 4 prediction set × 8 asset-task = 32"
    )

if len(summary_df) != 4:
    raise RuntimeError(
        "Final summary exact 4 prediction set içermiyor."
    )

if (
    summary_df[
        "primary_prediction"
    ].sum()
    != 1
):
    raise RuntimeError(
        "Exact bir adet primary prediction olmalı."
    )

primary_row = summary_df.loc[
    summary_df[
        "primary_prediction"
    ]
].iloc[0]

if str(
    primary_row[
        "model_label"
    ]
) != "FinalWinner_3SeedEnsemble":
    raise RuntimeError(
        "Primary prediction ensemble değil."
    )


# ==========================================================
# 16. JSON PROVENANCE + METODOLOJİK KAYIT
# ==========================================================

summary_payload_07 = {
    "project_version": "v4_repro",
    "created_at": datetime.now(
        timezone.utc
    ).isoformat(),
    "script": "07_final_test_evaluation_v4.py",
    "purpose": (
        "Final out-of-sample test evaluation of the single configuration "
        "locked by 06 multiseed validation selection."
    ),
    "model_selection_inside_07": False,
    "hyperparameter_change_inside_07": False,
    "retraining_inside_07": False,
    "baseline_comparison_inside_07": False,
    "statistical_model_comparison_inside_07": False,
    "primary_test_policy": PRIMARY_TEST_POLICY,
    "primary_prediction": "FinalWinner_3SeedEnsemble",
    "individual_seed_predictions_reported": True,
    "expected_seeds": EXPECTED_SEEDS,
    "winner_config_id": EXPECTED_WINNER_CONFIG_ID,
    "winner": clone_json_safe(
        winner
    ),
    "winner_checkpoint_audit": checkpoint_records,
    "schema_asset_order": ASSET_ORDER,
    "target_names": TARGET_NAMES,
    "test_sample_count": int(
        len(y_test_raw)
    ),
    "test_feature_shape": list(
        X_test.shape
    ),
    "test_target_shape": list(
        y_test_raw.shape
    ),
    "max_test_inverse_scale_diff": max_inverse_diff,
    "first_official_test_access_in_v4_script_log": True,
    "test_access_started": TEST_ACCESS_STARTED,
    "test_access_started_at_utc": test_access_started_at,
    "test_metrics_computed": TEST_METRICS_COMPUTED,
    "test_metrics_computed_at_utc": test_metrics_computed_at,
    "06_script_sha256_verified": True,
    "06_script_sha256": actual_06_sha,
    "06_winner_json_path": WINNER_JSON,
    "outputs": {
        "metrics_long_csv": METRICS_LONG_CSV,
        "summary_csv": SUMMARY_CSV,
        "summary_json": SUMMARY_JSON,
        "y_true_raw_npy": Y_TRUE_RAW_PATH,
        "ensemble_prediction_raw_npy": ENSEMBLE_PRED_PATH,
        "seed_prediction_raw_npy": {
            str(seed): os.path.join(
                RESULTS_DIR,
                f"pred_final_seed{seed}_raw_v4.npy"
            )
            for seed in EXPECTED_SEEDS
        }
    },
    "primary_metrics": {
        key: (
            None
            if pd.isna(value)
            else float(value)
        )
        for key, value in primary_row.items()
        if key not in {
            "model_label",
            "primary_prediction"
        }
    },
    "notes": [
        (
            "The 07 stage does not perform model selection. "
            "The configuration was locked before test access by 06 multiseed validation."
        ),
        (
            "The primary final prediction is the arithmetic mean, in raw target scale, "
            "of the three locked winner-seed checkpoint predictions."
        ),
        (
            "Individual seed test metrics are retained as robustness information, "
            "not as alternative post-hoc model selection candidates."
        ),
        (
            "No baseline superiority claim is made in 07. "
            "Baseline comparison is reserved for stage 08."
        ),
        (
            "No Diebold-Mariano or multiple-comparison inference is performed in 07. "
            "Statistical comparison is reserved for stage 09."
        )
    ]
}

with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary_payload_07,
        f,
        ensure_ascii=False,
        indent=2
    )


# ==========================================================
# 17. FINAL ÇIKTI
# ==========================================================

print("\n" + "=" * 100)
print("07_final_test_evaluation_v4.py TAMAMLANDI")
print("=" * 100)

print("\nLOCKED WINNER:")
print(
    EXPECTED_WINNER_CONFIG_ID
)

print("\nPRIMARY TEST POLICY:")
print(
    PRIMARY_TEST_POLICY
)

print("\nINDIVIDUAL + ENSEMBLE SUMMARY:")
print(
    summary_df.to_string(
        index=False
    )
)

print("\nPRIMARY FINAL TEST METRICS:")
print(
    ensemble_metrics_df[
        [
            "task",
            "asset",
            "MAE",
            "RMSE",
            "R2",
            "PinballLoss_tau_0.5",
        ]
    ].to_string(
        index=False
    )
)

print("\nRULE CHECK:")
print("✅ 06 winner config değiştirilmedi.")
print("✅ 06 winner'ın exact 3 seed checkpoint'i kullanıldı.")
print("✅ 07 içinde model seçimi yapılmadı.")
print("✅ 07 içinde hyperparameter değişmedi.")
print("✅ 07 içinde yeniden eğitim yapılmadı.")
print("✅ Ana final tahmin 3-seed ensemble olarak üretildi.")
print("✅ Her seed ayrıca raporlandı.")
print("✅ Test seti yalnızca kilitli final model için açıldı.")
print("✅ 07 baseline üstünlüğü iddiası üretmedi.")
print("✅ 07 istatistiksel karşılaştırma yapmadı.")
print("✅ 08 ve 09 için raw tahmin dizileri kaydedildi.")

print("\nOUTPUTS:")
print(" -", METRICS_LONG_CSV)
print(" -", SUMMARY_CSV)
print(" -", SUMMARY_JSON)
print(" -", Y_TRUE_RAW_PATH)
print(" -", ENSEMBLE_PRED_PATH)

for seed in EXPECTED_SEEDS:
    print(
        " -",
        os.path.join(
            RESULTS_DIR,
            f"pred_final_seed{seed}_raw_v4.npy"
        )
    )

print("\nTEST ACCESS STATUS:")
print("Test access started :", TEST_ACCESS_STARTED)
print("Test metrics computed:", TEST_METRICS_COMPUTED)

print("=" * 100)
