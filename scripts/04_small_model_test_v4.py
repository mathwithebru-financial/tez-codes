
# ==========================================================
# 04_small_model_test_v4.py
#
# AMAÇ:
# - Küçük bir FullSharingMTL Transformer ile smoke test yapmak
# - Veri -> DataLoader -> Model -> Loss -> Backprop -> Validation
#   -> inverse scaling -> ValidationScore hattını doğrulamak
#
# ÖNEMLİ:
# - Grid search YOK.
# - Final model seçimi YOK.
# - Test seti YÜKLENMEZ ve test metriği HESAPLANMAZ.
# ==========================================================

import os
import json
import pickle
import random
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
except ImportError as e:
    raise ImportError(
        "PyTorch bulunamadı. Colab runtime'ında PyTorch kurulumu kontrol edilmeli."
    ) from e


# ==========================================================
# 1. YOLLAR
# ==========================================================

BASE_DIR = "/content/drive/MyDrive/tez_transformer_v4_repro"

CONFIG_DIR = os.path.join(
    BASE_DIR,
    "config"
)

SEQUENCE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "sequences"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "small_model_test"
)

for path in [
    MODEL_DIR,
    RESULTS_DIR
]:
    os.makedirs(
        path,
        exist_ok=True
    )


# ==========================================================
# 2. SEED VE DEVICE
# ==========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("[DEVICE]", device)
print("[SEED]", SEED)


# ==========================================================
# 3. ŞEMA VE DENOMINATOR DOSYALARINI OKU
# ==========================================================

schema_path = os.path.join(
    CONFIG_DIR,
    "schema_v4.json"
)

denominator_path = os.path.join(
    PROCESSED_DIR,
    "selection_baseline_denominators_v4.json"
)

split_meta_path = os.path.join(
    PROCESSED_DIR,
    "split_meta_v4.json"
)

for path in [
    schema_path,
    denominator_path,
    split_meta_path
]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Gerekli dosya bulunamadı:\n{path}"
        )


with open(
    schema_path,
    "r",
    encoding="utf-8"
) as f:

    schema = json.load(f)


with open(
    denominator_path,
    "r",
    encoding="utf-8"
) as f:

    denominators = json.load(f)


with open(
    split_meta_path,
    "r",
    encoding="utf-8"
) as f:

    split_meta = json.load(f)


ASSET_ORDER = schema[
    "data"
][
    "assets"
]


TARGET_NAMES = schema[
    "targets"
][
    "definition"
]


if ASSET_ORDER != [
    "BIST100",
    "USDTRY",
    "EURTRY",
    "GOLD"
]:

    raise RuntimeError(
        f"Asset sırası beklenenden farklı: {ASSET_ORDER}"
    )


if len(TARGET_NAMES) != 8:

    raise RuntimeError(
        f"Target sayısı 8 değil: {len(TARGET_NAMES)}"
    )


print(
    "\n[OK] Schema, split meta ve denominator dosyaları okundu."
)

print(
    "Asset order:",
    ASSET_ORDER
)

print(
    "Target order:",
    TARGET_NAMES
)


# ==========================================================
# 4. SMOKE TEST KONFİGÜRASYONU
# ==========================================================

FEATURE_SET = "baseline"
LOOKBACK = 10

D_MODEL = 32
N_HEAD = 4
N_LAYERS = 2
D_FF = 128
DROPOUT = 0.10

LOSS_STRATEGY = "FixedLambda_0.5"
LAMBDA_RETURN = 0.5
TAU = 0.5

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0


print("\n" + "=" * 80)
print("SMOKE TEST KONFİGÜRASYONU")
print("=" * 80)

print(
    "Feature set :",
    FEATURE_SET
)

print(
    "Lookback    :",
    LOOKBACK
)

print(
    "Architecture:",
    "FullSharingMTL"
)

print(
    "Size        :",
    "small"
)

print(
    "Loss        :",
    LOSS_STRATEGY
)

print(
    "d_model     :",
    D_MODEL
)

print(
    "n_head      :",
    N_HEAD
)

print(
    "n_layers    :",
    N_LAYERS
)

print(
    "d_ff        :",
    D_FF
)

print(
    "epochs      :",
    EPOCHS
)

print(
    "batch_size  :",
    BATCH_SIZE
)


# ==========================================================
# 5. SADECE TRAIN VE VALIDATION DOSYALARINI YÜKLE
# ==========================================================

seq_dir = os.path.join(
    SEQUENCE_DIR,
    FEATURE_SET,
    f"lb{LOOKBACK}"
)

scaler_path = os.path.join(
    SEQUENCE_DIR,
    FEATURE_SET,
    "scalers.pkl"
)


required_files = [

    os.path.join(
        seq_dir,
        "X_train.npy"
    ),

    os.path.join(
        seq_dir,
        "y_train.npy"
    ),

    os.path.join(
        seq_dir,
        "y_train_raw.npy"
    ),

    os.path.join(
        seq_dir,
        "X_val.npy"
    ),

    os.path.join(
        seq_dir,
        "y_val.npy"
    ),

    os.path.join(
        seq_dir,
        "y_val_raw.npy"
    ),

    os.path.join(
        seq_dir,
        "anchor_dates_val.npy"
    ),

    os.path.join(
        seq_dir,
        "target_realization_dates_val.npy"
    ),

    scaler_path
]


for file_path in required_files:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Gerekli dosya bulunamadı:\n{file_path}"
        )


X_train = np.load(
    os.path.join(
        seq_dir,
        "X_train.npy"
    )
)


y_train = np.load(
    os.path.join(
        seq_dir,
        "y_train.npy"
    )
)


y_train_raw = np.load(
    os.path.join(
        seq_dir,
        "y_train_raw.npy"
    )
)


X_val = np.load(
    os.path.join(
        seq_dir,
        "X_val.npy"
    )
)


y_val = np.load(
    os.path.join(
        seq_dir,
        "y_val.npy"
    )
)


y_val_raw = np.load(
    os.path.join(
        seq_dir,
        "y_val_raw.npy"
    )
)


anchor_dates_val = np.load(
    os.path.join(
        seq_dir,
        "anchor_dates_val.npy"
    )
)


target_dates_val = np.load(
    os.path.join(
        seq_dir,
        "target_realization_dates_val.npy"
    )
)


with open(
    scaler_path,
    "rb"
) as f:

    scaler_obj = pickle.load(f)


y_scaler = scaler_obj[
    "y_scaler"
]


print("\n" + "=" * 80)
print("VERİ SHAPE")
print("=" * 80)

print(
    "X_train   :",
    X_train.shape
)

print(
    "y_train   :",
    y_train.shape
)

print(
    "X_val     :",
    X_val.shape
)

print(
    "y_val     :",
    y_val.shape
)

print(
    "y_val_raw :",
    y_val_raw.shape
)


if (
    X_train.ndim != 3
    or
    X_val.ndim != 3
):

    raise ValueError(
        "X dizileri 3 boyutlu olmalı: "
        "(n, lookback, n_features)"
    )


if (
    y_train.ndim != 2
    or
    y_val.ndim != 2
):

    raise ValueError(
        "y dizileri 2 boyutlu olmalı: "
        "(n, n_targets)"
    )


if (
    X_train.shape[1] != LOOKBACK
    or
    X_val.shape[1] != LOOKBACK
):

    raise ValueError(
        "Lookback boyutu uyuşmuyor."
    )


if (
    y_train.shape[1] != 8
    or
    y_val.shape[1] != 8
):

    raise ValueError(
        "Target boyutu 8 olmalı."
    )


if len(X_val) != 584:

    raise ValueError(
        f"Validation örnek sayısı 584 değil: {len(X_val)}"
    )


if (
    len(anchor_dates_val) != len(X_val)
    or
    len(target_dates_val) != len(X_val)
):

    raise ValueError(
        "Validation tarih dizileri X_val ile hizalı değil."
    )


# ==========================================================
# 6. INVERSE-SCALING KONTROLÜ
# ==========================================================

y_val_inverse_check = (
    y_scaler.inverse_transform(
        y_val
    )
)


max_inverse_diff = float(
    np.max(
        np.abs(
            y_val_inverse_check
            -
            y_val_raw
        )
    )
)


if max_inverse_diff > 1e-5:

    raise RuntimeError(
        "y_val_raw ile "
        "y_scaler.inverse_transform(y_val) uyuşmuyor. "
        f"Maksimum fark: {max_inverse_diff}"
    )


val_target_dt = pd.to_datetime(
    target_dates_val
)


expected_val_target_start = pd.Timestamp(
    split_meta[
        "validation"
    ][
        "target_realization_start"
    ]
)


expected_val_target_end = pd.Timestamp(
    split_meta[
        "validation"
    ][
        "target_realization_end"
    ]
)


if (
    val_target_dt.min()
    !=
    expected_val_target_start
):

    raise RuntimeError(
        "Validation target başlangıç tarihi "
        "split meta ile uyuşmuyor."
    )


if (
    val_target_dt.max()
    !=
    expected_val_target_end
):

    raise RuntimeError(
        "Validation target bitiş tarihi "
        "split meta ile uyuşmuyor."
    )


N_FEATURES = X_train.shape[2]
N_TARGETS = y_train.shape[1]


print(
    "\n[OK] Veri şekilleri ve inverse-scaling kontrolü geçti."
)

print(
    "N_FEATURES:",
    N_FEATURES
)

print(
    "N_TARGETS :",
    N_TARGETS
)

print(
    "Max inverse diff:",
    max_inverse_diff
)

print(
    "Validation target realization:",
    val_target_dt.min().date(),
    "→",
    val_target_dt.max().date()
)


# ==========================================================
# 7. DATALOADER
# ==========================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32
)

X_val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_val_tensor = torch.tensor(
    y_val,
    dtype=torch.float32
)


train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)


val_dataset = TensorDataset(
    X_val_tensor,
    y_val_tensor
)


loader_generator = torch.Generator()

loader_generator.manual_seed(
    SEED
)


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    drop_last=False,

    generator=loader_generator
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    drop_last=False
)


print(
    "[OK] DataLoader hazır."
)


# ==========================================================
# 8. MODEL: FullSharingMTL
# ==========================================================

class FullSharingMTL(nn.Module):

    """
    Tam paylaşım mimarisi:

    - ortak input projection
    - ortak positional embedding
    - ortak Transformer encoder
    - ortak latent representation
    - ayrı return head
    - ayrı volatility head

    Çıktı:
        ilk 4 = NextRet
        son 4 = NextVol
    """

    def __init__(

        self,

        n_features,

        lookback,

        d_model=32,

        n_head=4,

        n_layers=2,

        d_ff=128,

        dropout=0.10
    ):

        super().__init__()


        self.n_features = (
            n_features
        )

        self.lookback = (
            lookback
        )

        self.d_model = (
            d_model
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


        encoder_layer = nn.TransformerEncoderLayer(

            d_model=d_model,

            nhead=n_head,

            dim_feedforward=d_ff,

            dropout=dropout,

            activation="gelu",

            batch_first=True
        )


        self.encoder = nn.TransformerEncoder(

            encoder_layer,

            num_layers=n_layers
        )


        self.norm = nn.LayerNorm(
            d_model
        )


        self.return_head = nn.Sequential(

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


        self.vol_head = nn.Sequential(

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


    def forward(
        self,
        x
    ):

        h = self.input_projection(
            x
        )


        h = (
            h
            +
            self.positional_embedding[
                :,
                :h.size(1),
                :
            ]
        )


        h = self.encoder(
            h
        )


        h_last = self.norm(
            h[
                :,
                -1,
                :
            ]
        )


        ret_out = self.return_head(
            h_last
        )


        vol_out = self.vol_head(
            h_last
        )


        return torch.cat(

            [
                ret_out,
                vol_out
            ],

            dim=1
        )


model = FullSharingMTL(

    n_features=N_FEATURES,

    lookback=LOOKBACK,

    d_model=D_MODEL,

    n_head=N_HEAD,

    n_layers=N_LAYERS,

    d_ff=D_FF,

    dropout=DROPOUT

).to(device)


parameter_count = int(

    sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad
    )
)


print(
    "\n[MODEL]"
)

print(
    model
)

print(
    "\nTrainable parameter count:",
    parameter_count
)


# ==========================================================
# 9. LOSS VE OPTIMIZER
# ==========================================================

mse_loss = nn.MSELoss()


def pinball_loss_torch(

    y_pred,

    y_true,

    tau=0.5
):

    diff = (
        y_true
        -
        y_pred
    )


    loss = torch.maximum(

        tau * diff,

        (tau - 1.0) * diff
    )


    return loss.mean()


def multi_task_loss(

    y_pred,

    y_true
):

    pred_ret = (
        y_pred[
            :,
            :4
        ]
    )


    true_ret = (
        y_true[
            :,
            :4
        ]
    )


    pred_vol = (
        y_pred[
            :,
            4:
        ]
    )


    true_vol = (
        y_true[
            :,
            4:
        ]
    )


    return_loss = mse_loss(

        pred_ret,

        true_ret
    )


    vol_loss = pinball_loss_torch(

        pred_vol,

        true_vol,

        tau=TAU
    )


    total_loss = (

        LAMBDA_RETURN
        *
        return_loss

        +

        (
            1.0
            -
            LAMBDA_RETURN
        )
        *
        vol_loss
    )


    return (
        total_loss,
        return_loss,
        vol_loss
    )


optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)


# ==========================================================
# 10. TRAIN / VALIDATION FONKSİYONLARI
# ==========================================================

def train_one_epoch(

    model,

    loader,

    optimizer
):

    model.train()


    total_loss_sum = 0.0

    return_loss_sum = 0.0

    vol_loss_sum = 0.0

    n_obs = 0


    for (
        X_batch,
        y_batch
    ) in loader:


        X_batch = X_batch.to(
            device
        )


        y_batch = y_batch.to(
            device
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        y_pred = model(
            X_batch
        )


        (
            loss,
            return_loss,
            vol_loss

        ) = multi_task_loss(

            y_pred,

            y_batch
        )


        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                "Training loss finite değil."
            )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=GRAD_CLIP
        )


        optimizer.step()


        batch_size = X_batch.size(
            0
        )


        total_loss_sum += (
            loss.item()
            *
            batch_size
        )


        return_loss_sum += (
            return_loss.item()
            *
            batch_size
        )


        vol_loss_sum += (
            vol_loss.item()
            *
            batch_size
        )


        n_obs += (
            batch_size
        )


    return {

        "loss":
            total_loss_sum
            /
            n_obs,

        "return_loss":
            return_loss_sum
            /
            n_obs,

        "vol_loss":
            vol_loss_sum
            /
            n_obs
    }


@torch.no_grad()
def evaluate_scaled_loss(

    model,

    loader
):

    model.eval()


    total_loss_sum = 0.0

    return_loss_sum = 0.0

    vol_loss_sum = 0.0

    n_obs = 0


    preds = []

    trues = []


    for (
        X_batch,
        y_batch
    ) in loader:


        X_batch = X_batch.to(
            device
        )


        y_batch = y_batch.to(
            device
        )


        y_pred = model(
            X_batch
        )


        (
            loss,
            return_loss,
            vol_loss

        ) = multi_task_loss(

            y_pred,

            y_batch
        )


        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                "Validation loss finite değil."
            )


        batch_size = X_batch.size(
            0
        )


        total_loss_sum += (
            loss.item()
            *
            batch_size
        )


        return_loss_sum += (
            return_loss.item()
            *
            batch_size
        )


        vol_loss_sum += (
            vol_loss.item()
            *
            batch_size
        )


        n_obs += (
            batch_size
        )


        preds.append(

            y_pred
            .detach()
            .cpu()
            .numpy()
        )


        trues.append(

            y_batch
            .detach()
            .cpu()
            .numpy()
        )


    return {

        "loss":
            total_loss_sum
            /
            n_obs,

        "return_loss":
            return_loss_sum
            /
            n_obs,

        "vol_loss":
            vol_loss_sum
            /
            n_obs,

        "preds_scaled":
            np.vstack(
                preds
            ),

        "trues_scaled":
            np.vstack(
                trues
            )
    }


# ==========================================================
# 11. RAW METRİKLER
# ==========================================================

def mae_np(
    y_true,
    y_pred
):

    return float(

        np.mean(

            np.abs(
                y_true
                -
                y_pred
            )
        )
    )


def rmse_np(
    y_true,
    y_pred
):

    return float(

        np.sqrt(

            np.mean(

                (
                    y_true
                    -
                    y_pred
                )
                ** 2
            )
        )
    )


def r2_np(
    y_true,
    y_pred
):

    ss_res = np.sum(

        (
            y_true
            -
            y_pred
        )
        ** 2
    )


    ss_tot = np.sum(

        (
            y_true
            -
            np.mean(
                y_true
            )
        )
        ** 2
    )


    if ss_tot == 0:

        return float(
            "nan"
        )


    return float(

        1.0
        -
        ss_res
        /
        ss_tot
    )


def pinball_np(

    y_true,

    y_pred,

    tau=0.5
):

    diff = (
        y_true
        -
        y_pred
    )


    loss = np.maximum(

        tau
        *
        diff,

        (
            tau
            -
            1.0
        )
        *
        diff
    )


    return float(

        np.mean(
            loss
        )
    )


def compute_validation_raw_metrics(

    y_true_raw,

    y_pred_raw
):

    rows = []


    for (
        i,
        asset
    ) in enumerate(
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


        rows.append(

            {

                "task":
                    "return",

                "asset":
                    asset,

                "MAE":
                    mae_np(
                        true,
                        pred
                    ),

                "RMSE":
                    rmse_np(
                        true,
                        pred
                    ),

                "R2":
                    r2_np(
                        true,
                        pred
                    ),

                "PinballLoss_tau_0.5":
                    np.nan
            }
        )


    for (
        i,
        asset
    ) in enumerate(
        ASSET_ORDER
    ):


        col = (
            4
            +
            i
        )


        true = y_true_raw[
            :,
            col
        ]


        pred = y_pred_raw[
            :,
            col
        ]


        rows.append(

            {

                "task":
                    "volatility",

                "asset":
                    asset,

                "MAE":
                    mae_np(
                        true,
                        pred
                    ),

                "RMSE":
                    rmse_np(
                        true,
                        pred
                    ),

                "R2":
                    r2_np(
                        true,
                        pred
                    ),

                "PinballLoss_tau_0.5":
                    pinball_np(

                        true,

                        pred,

                        tau=TAU
                    )
            }
        )


    return pd.DataFrame(
        rows
    )


# ==========================================================
# 12. VALIDATIONSCORE
# ==========================================================

def compute_validation_score(

    metrics_df,

    denominators
):

    return_ratios = []

    vol_ratios = []


    return_ratio_map = {}

    vol_ratio_map = {}


    for asset in ASSET_ORDER:


        model_return_mae = float(

            metrics_df.loc[

                (
                    metrics_df[
                        "task"
                    ]
                    ==
                    "return"
                )

                &

                (
                    metrics_df[
                        "asset"
                    ]
                    ==
                    asset
                ),

                "MAE"

            ].iloc[0]
        )


        denom_return_mae = float(

            denominators[

                "return_denominator"

            ][

                asset

            ][

                "value"

            ]
        )


        model_vol_pinball = float(

            metrics_df.loc[

                (
                    metrics_df[
                        "task"
                    ]
                    ==
                    "volatility"
                )

                &

                (
                    metrics_df[
                        "asset"
                    ]
                    ==
                    asset
                ),

                "PinballLoss_tau_0.5"

            ].iloc[0]
        )


        denom_vol_pinball = float(

            denominators[

                "volatility_denominator"

            ][

                asset

            ][

                "value"

            ]
        )


        if (
            denom_return_mae <= 0
            or
            denom_vol_pinball <= 0
        ):

            raise RuntimeError(

                f"{asset} denominator pozitif değil."
            )


        return_ratio = (

            model_return_mae

            /

            denom_return_mae
        )


        vol_ratio = (

            model_vol_pinball

            /

            denom_vol_pinball
        )


        return_ratios.append(
            return_ratio
        )


        vol_ratios.append(
            vol_ratio
        )


        return_ratio_map[
            asset
        ] = float(
            return_ratio
        )


        vol_ratio_map[
            asset
        ] = float(
            vol_ratio
        )


    avg_return_ratio = float(

        np.mean(
            return_ratios
        )
    )


    avg_vol_ratio = float(

        np.mean(
            vol_ratios
        )
    )


    validation_score = float(

        0.5
        *
        avg_return_ratio

        +

        0.5
        *
        avg_vol_ratio
    )


    return {

        "return_ratios":
            return_ratio_map,

        "vol_ratios":
            vol_ratio_map,

        "avg_return_ratio":
            avg_return_ratio,

        "avg_vol_ratio":
            avg_vol_ratio,

        "validation_score":
            validation_score,

        "lower_is_better":
            True
    }


# ==========================================================
# 13. EĞİTİM
# ==========================================================

print("\n" + "=" * 80)
print("EĞİTİM BAŞLIYOR — SMOKE TEST")
print("=" * 80)


history = []


for epoch in range(
    1,
    EPOCHS + 1
):


    train_metrics = train_one_epoch(

        model=model,

        loader=train_loader,

        optimizer=optimizer
    )


    val_scaled_metrics = evaluate_scaled_loss(

        model=model,

        loader=val_loader
    )


    row = {

        "epoch":
            epoch,

        "train_loss":
            train_metrics[
                "loss"
            ],

        "train_return_loss":
            train_metrics[
                "return_loss"
            ],

        "train_vol_loss":
            train_metrics[
                "vol_loss"
            ],

        "val_loss":
            val_scaled_metrics[
                "loss"
            ],

        "val_return_loss":
            val_scaled_metrics[
                "return_loss"
            ],

        "val_vol_loss":
            val_scaled_metrics[
                "vol_loss"
            ]
    }


    history.append(
        row
    )


    print(

        f"Epoch {epoch:02d} | "

        f"train_loss="
        f"{row['train_loss']:.6f} | "

        f"val_loss="
        f"{row['val_loss']:.6f} | "

        f"val_ret="
        f"{row['val_return_loss']:.6f} | "

        f"val_vol="
        f"{row['val_vol_loss']:.6f}"
    )


# ==========================================================
# 14. VALIDATION RAW METRİKLER VE SKOR
# ==========================================================

val_scaled_metrics = evaluate_scaled_loss(

    model,

    val_loader
)


y_val_pred_scaled = (
    val_scaled_metrics[
        "preds_scaled"
    ]
)


y_val_pred_raw = (
    y_scaler.inverse_transform(
        y_val_pred_scaled
    )
)


if not np.isfinite(
    y_val_pred_raw
).all():

    raise RuntimeError(
        "Validation raw prediction içinde NaN/Inf var."
    )


validation_raw_metrics_df = compute_validation_raw_metrics(

    y_true_raw=y_val_raw,

    y_pred_raw=y_val_pred_raw
)


validation_score_obj = compute_validation_score(

    metrics_df=
        validation_raw_metrics_df,

    denominators=
        denominators
)


print("\n" + "=" * 80)
print("VALIDATION RAW METRİKLER — SMOKE TEST")
print("=" * 80)

print(

    validation_raw_metrics_df.to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("BASELINE-NORMALIZE VALIDATION SCORE — SMOKE TEST")
print("=" * 80)

print(

    json.dumps(

        validation_score_obj,

        ensure_ascii=False,

        indent=2
    )
)


# ==========================================================
# 15. DOSYALARI KAYDET
# ==========================================================

history_df = pd.DataFrame(
    history
)


history_path = os.path.join(

    RESULTS_DIR,

    "small_model_training_history_v4.csv"
)


metrics_path = os.path.join(

    RESULTS_DIR,

    "small_model_validation_metrics_v4.csv"
)


score_path = os.path.join(

    RESULTS_DIR,

    "small_model_validation_score_v4.json"
)


summary_path = os.path.join(

    RESULTS_DIR,

    "small_model_test_summary_v4.json"
)


model_path = os.path.join(

    MODEL_DIR,

    "small_model_test_fullsharing_baseline_lb10_v4.pt"
)


history_df.to_csv(

    history_path,

    index=False
)


validation_raw_metrics_df.to_csv(

    metrics_path,

    index=False
)


with open(

    score_path,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        validation_score_obj,

        f,

        ensure_ascii=False,

        indent=2
    )


torch.save(

    {

        "model_state_dict":
            model.state_dict(),

        "config": {

            "project_version":
                "v4_repro",

            "feature_set":
                FEATURE_SET,

            "lookback":
                LOOKBACK,

            "architecture":
                "FullSharingMTL",

            "size":
                "small",

            "d_model":
                D_MODEL,

            "n_head":
                N_HEAD,

            "n_layers":
                N_LAYERS,

            "d_ff":
                D_FF,

            "dropout":
                DROPOUT,

            "loss_strategy":
                LOSS_STRATEGY,

            "lambda_return":
                LAMBDA_RETURN,

            "tau":
                TAU,

            "epochs":
                EPOCHS,

            "seed":
                SEED
        },

        "target_names":
            TARGET_NAMES,

        "asset_order":
            ASSET_ORDER,

        "trainable_parameter_count":
            parameter_count,

        "purpose":
            "smoke_test_only"
    },

    model_path
)


summary = {

    "created_at":
        datetime.now().isoformat(),

    "script":
        "04_small_model_test_v4.py",

    "project_version":
        "v4_repro",

    "status":
        "success",

    "purpose":
        (
            "Smoke test only: verifies "
            "data -> model -> loss -> backprop -> "
            "validation -> inverse scaling -> "
            "ValidationScore pipeline."
        ),

    "device":
        str(device),

    "seed":
        SEED,

    "feature_set":
        FEATURE_SET,

    "lookback":
        LOOKBACK,

    "architecture":
        "FullSharingMTL",

    "size":
        "small",

    "loss_strategy":
        LOSS_STRATEGY,

    "epochs":
        EPOCHS,

    "trainable_parameter_count":
        parameter_count,

    "validation_rows":
        int(
            len(
                X_val
            )
        ),

    "validation_anchor_start":
        str(
            pd.to_datetime(
                anchor_dates_val
            ).min().date()
        ),

    "validation_anchor_end":
        str(
            pd.to_datetime(
                anchor_dates_val
            ).max().date()
        ),

    "validation_target_realization_start":
        str(
            val_target_dt
            .min()
            .date()
        ),

    "validation_target_realization_end":
        str(
            val_target_dt
            .max()
            .date()
        ),

    "max_inverse_scaling_diff":
        max_inverse_diff,

    "validation_score":
        validation_score_obj,

    "test_arrays_loaded":
        False,

    "test_metrics_computed":
        False,

    "final_model_selection_performed":
        False,

    "files_created": {

        "training_history":
            history_path,

        "validation_metrics":
            metrics_path,

        "validation_score":
            score_path,

        "model_checkpoint":
            model_path
    },

    "important_notes": [

        "Bu dosyada test dizileri yüklenmemiştir.",

        "Bu dosyada test metriği hesaplanmamıştır.",

        "Bu dosyada final model seçimi yapılmamıştır.",

        "Bu dosya yalnızca smoke testtir.",

        "ValidationScore yalnızca validation denominator'ları ile hesaplanmıştır."
    ]
}


with open(

    summary_path,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        summary,

        f,

        ensure_ascii=False,

        indent=2
    )


# ==========================================================
# 16. SON KONTROLLER VE ÇIKTI
# ==========================================================

if len(
    validation_raw_metrics_df
) != 8:

    raise RuntimeError(
        "Beklenen 8 validation metric satırı üretilemedi."
    )


if not np.isfinite(
    validation_score_obj[
        "validation_score"
    ]
):

    raise RuntimeError(
        "ValidationScore finite değil."
    )


print("\n" + "=" * 80)
print("04_small_model_test_v4.py BAŞARIYLA TAMAMLANDI")
print("=" * 80)


print("\nÜretilen dosyalar:")

print(
    " -",
    history_path
)

print(
    " -",
    metrics_path
)

print(
    " -",
    score_path
)

print(
    " -",
    summary_path
)

print(
    " -",
    model_path
)


print("\nKURAL KONTROLÜ:")

print(
    "✅ Train ve validation dizileri yüklendi."
)

print(
    "✅ Test dizileri yüklenmedi."
)

print(
    "✅ 5 epoch smoke test tamamlandı."
)

print(
    "✅ Forward pass tamamlandı."
)

print(
    "✅ Return MSE hesaplandı."
)

print(
    "✅ Volatility PinballLoss(tau=0.5) hesaplandı."
)

print(
    "✅ Backpropagation tamamlandı."
)

print(
    "✅ Validation prediction inverse-scale edildi."
)

print(
    "✅ ValidationScore hesaplandı."
)

print(
    "✅ Test metriği hesaplanmadı."
)

print(
    "✅ Final model seçimi yapılmadı."
)

print("=" * 80)
