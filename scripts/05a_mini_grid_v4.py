
import os
import json
import pickle
import random
import copy
import itertools
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)


# ==========================================================
# 1. YOLLAR
# ==========================================================

BASE_DIR = "/content/drive/MyDrive/tez_transformer_v4_repro"

if not os.path.exists(BASE_DIR):

    raise FileNotFoundError(
        f"Proje klasörü yok:\n{BASE_DIR}\n"
        "Drive mount edildi mi?"
    )


CONFIG_DIR = os.path.join(
    BASE_DIR,
    "config"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

SEQUENCE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "sequences"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "mini_grid"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "mini_grid"
)


os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ==========================================================
# 2. MINI-GRID SABİTLERİ
# ==========================================================

SEED = 42

EPOCHS = 5

BATCH_SIZE = 64

LR = 1e-3

WEIGHT_DECAY = 1e-4

GRAD_CLIP = 1.0

TAU = 0.5

LAMBDA_RETURN = 0.5


FEATURE_SET = "baseline"


LOOKBACKS = [
    10,
    30
]


ARCHITECTURES = [
    "FullSharingMTL",
    "NoSharing"
]


LOSS_STRATEGIES = [
    "FixedLambda_0.5",
    "PCGrad"
]


# medium model

D_MODEL = 64

N_HEAD = 4

N_LAYERS = 2

D_FF = 256

DROPOUT = 0.10


TOTAL_CONFIGS = (
    len(LOOKBACKS)
    *
    len(ARCHITECTURES)
    *
    len(LOSS_STRATEGIES)
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 80)

print(
    "05a — v4 MINI-GRID AUDIT"
)

print("=" * 80)

print(
    "[DEVICE]",
    DEVICE
)


if torch.cuda.is_available():

    print(
        "[GPU]",
        torch.cuda.get_device_name(0)
    )

else:

    print(
        "⚠️ GPU aktif değil; CPU kullanılacak."
    )


print(
    "[TOTAL CONFIGS]",
    TOTAL_CONFIGS
)

print(
    "[PURPOSE] Audit only — final model selection değildir."
)


# ==========================================================
# 3. ŞEMA VE DENOMINATOR
# ==========================================================

schema_path = os.path.join(
    CONFIG_DIR,
    "schema_v4.json"
)


denom_path = os.path.join(
    PROCESSED_DIR,
    "selection_baseline_denominators_v4.json"
)


for path in [
    schema_path,
    denom_path
]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Gerekli dosya yok:\n{path}"
        )


with open(
    schema_path,
    "r",
    encoding="utf-8"
) as f:

    schema = json.load(f)


with open(
    denom_path,
    "r",
    encoding="utf-8"
) as f:

    denominators = json.load(f)


ASSETS = schema[
    "data"
][
    "assets"
]


TARGETS = schema[
    "targets"
][
    "definition"
]


if ASSETS != [
    "BIST100",
    "USDTRY",
    "EURTRY",
    "GOLD"
]:

    raise RuntimeError(
        f"Asset sırası beklenenden farklı: {ASSETS}"
    )


if len(TARGETS) != 8:

    raise RuntimeError(
        f"Target sayısı 8 değil: {len(TARGETS)}"
    )


# ==========================================================
# 4. YARDIMCI FONKSİYONLAR
# ==========================================================

def set_seed(
    seed=42
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )


    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


    if hasattr(
        torch.backends,
        "cudnn"
    ):

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False



def clone_state_to_cpu(
    model
):

    return {

        key:
            value
            .detach()
            .cpu()
            .clone()

        for (
            key,
            value
        )

        in model.state_dict().items()
    }



def pinball_torch(
    pred,
    true,
    tau=0.5
):

    diff = (
        true
        -
        pred
    )


    return torch.maximum(

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

    ).mean()



def pinball_np(
    true,
    pred,
    tau=0.5
):

    diff = (
        true
        -
        pred
    )


    return float(

        np.maximum(

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

        ).mean()
    )



def score_validation(
    y_true_raw,
    y_pred_raw
):

    rows = []

    ret_ratios = {}

    vol_ratios = {}


    # ------------------------------------------------------
    # RETURN
    # ------------------------------------------------------

    for (
        i,
        asset
    ) in enumerate(
        ASSETS
    ):

        mae = float(

            np.mean(

                np.abs(

                    y_true_raw[
                        :,
                        i
                    ]

                    -

                    y_pred_raw[
                        :,
                        i
                    ]
                )
            )
        )


        denom = float(

            denominators[

                "return_denominator"

            ][

                asset

            ][

                "value"

            ]
        )


        if denom <= 0:

            raise RuntimeError(

                f"{asset} return denominator pozitif değil."
            )


        ratio = (
            mae
            /
            denom
        )


        ret_ratios[
            asset
        ] = float(
            ratio
        )


        rows.append(

            {

                "task":
                    "return",

                "asset":
                    asset,

                "MAE":
                    mae,

                "PinballLoss_tau_0.5":
                    np.nan,

                "baseline_ratio":
                    ratio
            }
        )


    # ------------------------------------------------------
    # VOLATILITY
    # ------------------------------------------------------

    for (
        i,
        asset
    ) in enumerate(
        ASSETS
    ):

        col = (
            4
            +
            i
        )


        mae = float(

            np.mean(

                np.abs(

                    y_true_raw[
                        :,
                        col
                    ]

                    -

                    y_pred_raw[
                        :,
                        col
                    ]
                )
            )
        )


        pb = pinball_np(

            y_true_raw[
                :,
                col
            ],

            y_pred_raw[
                :,
                col
            ],

            TAU
        )


        denom = float(

            denominators[

                "volatility_denominator"

            ][

                asset

            ][

                "value"

            ]
        )


        if denom <= 0:

            raise RuntimeError(

                f"{asset} volatility denominator pozitif değil."
            )


        ratio = (
            pb
            /
            denom
        )


        vol_ratios[
            asset
        ] = float(
            ratio
        )


        rows.append(

            {

                "task":
                    "volatility",

                "asset":
                    asset,

                "MAE":
                    mae,

                "PinballLoss_tau_0.5":
                    pb,

                "baseline_ratio":
                    ratio
            }
        )


    avg_ret = float(

        np.mean(

            list(
                ret_ratios.values()
            )
        )
    )


    avg_vol = float(

        np.mean(

            list(
                vol_ratios.values()
            )
        )
    )


    score = float(

        0.5
        *
        avg_ret

        +

        0.5
        *
        avg_vol
    )


    return (

        pd.DataFrame(
            rows
        ),

        {

            "return_ratios":
                ret_ratios,

            "vol_ratios":
                vol_ratios,

            "avg_return_ratio":
                avg_ret,

            "avg_vol_ratio":
                avg_vol,

            "validation_score":
                score,

            "lower_is_better":
                True
        }
    )


# ==========================================================
# 5. ORTAK ENCODER / HEAD ÜRETİCİLERİ
# ==========================================================

def make_encoder(

    d_model,

    n_head,

    n_layers,

    d_ff,

    dropout
):

    layer = nn.TransformerEncoderLayer(

        d_model=d_model,

        nhead=n_head,

        dim_feedforward=d_ff,

        dropout=dropout,

        activation="gelu",

        batch_first=True
    )


    return nn.TransformerEncoder(

        layer,

        num_layers=n_layers
    )



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


# ==========================================================
# 6. FULL SHARING
# ==========================================================

class FullSharingMTL(
    nn.Module
):

    def __init__(

        self,

        n_features,

        lookback
    ):

        super().__init__()


        self.input_projection = nn.Linear(

            n_features,

            D_MODEL
        )


        self.positional_embedding = nn.Parameter(

            torch.zeros(

                1,

                lookback,

                D_MODEL
            )
        )


        self.encoder = make_encoder(

            D_MODEL,

            N_HEAD,

            N_LAYERS,

            D_FF,

            DROPOUT
        )


        self.norm = nn.LayerNorm(

            D_MODEL
        )


        self.return_head = make_head(

            D_MODEL,

            DROPOUT
        )


        self.vol_head = make_head(

            D_MODEL,

            DROPOUT
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


        h = self.norm(

            h[
                :,
                -1,
                :
            ]
        )


        return torch.cat(

            [

                self.return_head(
                    h
                ),

                self.vol_head(
                    h
                )
            ],

            dim=1
        )


    def pcgrad_groups(
        self
    ):

        shared = (

            list(
                self.input_projection.parameters()
            )

            +

            [
                self.positional_embedding
            ]

            +

            list(
                self.encoder.parameters()
            )

            +

            list(
                self.norm.parameters()
            )
        )


        return {

            "shared":
                shared,

            "return_specific":
                list(
                    self.return_head.parameters()
                ),

            "vol_specific":
                list(
                    self.vol_head.parameters()
                )
        }


# ==========================================================
# 7. NO SHARING
# ==========================================================

class NoSharing(
    nn.Module
):

    def __init__(

        self,

        n_features,

        lookback
    ):

        super().__init__()


        # RETURN BRANCH

        self.ret_projection = nn.Linear(

            n_features,

            D_MODEL
        )


        self.ret_positional = nn.Parameter(

            torch.zeros(

                1,

                lookback,

                D_MODEL
            )
        )


        self.ret_encoder = make_encoder(

            D_MODEL,

            N_HEAD,

            N_LAYERS,

            D_FF,

            DROPOUT
        )


        self.ret_norm = nn.LayerNorm(

            D_MODEL
        )


        self.return_head = make_head(

            D_MODEL,

            DROPOUT
        )


        # VOLATILITY BRANCH

        self.vol_projection = nn.Linear(

            n_features,

            D_MODEL
        )


        self.vol_positional = nn.Parameter(

            torch.zeros(

                1,

                lookback,

                D_MODEL
            )
        )


        self.vol_encoder = make_encoder(

            D_MODEL,

            N_HEAD,

            N_LAYERS,

            D_FF,

            DROPOUT
        )


        self.vol_norm = nn.LayerNorm(

            D_MODEL
        )


        self.vol_head = make_head(

            D_MODEL,

            DROPOUT
        )


    def forward(
        self,
        x
    ):

        hr = self.ret_projection(
            x
        )


        hr = (

            hr

            +

            self.ret_positional[

                :,

                :hr.size(1),

                :
            ]
        )


        hr = self.ret_encoder(
            hr
        )


        hr = self.ret_norm(

            hr[
                :,
                -1,
                :
            ]
        )


        hv = self.vol_projection(
            x
        )


        hv = (

            hv

            +

            self.vol_positional[

                :,

                :hv.size(1),

                :
            ]
        )


        hv = self.vol_encoder(
            hv
        )


        hv = self.vol_norm(

            hv[
                :,
                -1,
                :
            ]
        )


        return torch.cat(

            [

                self.return_head(
                    hr
                ),

                self.vol_head(
                    hv
                )
            ],

            dim=1
        )


    def pcgrad_groups(
        self
    ):

        ret = (

            list(
                self.ret_projection.parameters()
            )

            +

            [
                self.ret_positional
            ]

            +

            list(
                self.ret_encoder.parameters()
            )

            +

            list(
                self.ret_norm.parameters()
            )

            +

            list(
                self.return_head.parameters()
            )
        )


        vol = (

            list(
                self.vol_projection.parameters()
            )

            +

            [
                self.vol_positional
            ]

            +

            list(
                self.vol_encoder.parameters()
            )

            +

            list(
                self.vol_norm.parameters()
            )

            +

            list(
                self.vol_head.parameters()
            )
        )


        return {

            "shared":
                [],

            "return_specific":
                ret,

            "vol_specific":
                vol
        }


# ==========================================================
# 8. MODEL FACTORY
# ==========================================================

def build_model(

    architecture,

    n_features,

    lookback
):

    if architecture == "FullSharingMTL":

        return FullSharingMTL(

            n_features,

            lookback
        )


    if architecture == "NoSharing":

        return NoSharing(

            n_features,

            lookback
        )


    raise ValueError(

        f"Bilinmeyen mimari: {architecture}"
    )


# ==========================================================
# 9. LOSS
# ==========================================================

def task_losses(

    pred,

    true
):

    ret_loss = nn.functional.mse_loss(

        pred[
            :,
            :4
        ],

        true[
            :,
            :4
        ]
    )


    vol_loss = pinball_torch(

        pred[
            :,
            4:
        ],

        true[
            :,
            4:
        ],

        TAU
    )


    total = (

        LAMBDA_RETURN
        *
        ret_loss

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

        total,

        ret_loss,

        vol_loss
    )


# ==========================================================
# 10. PCGRAD — SHARED PARAMETER ONLY
# ==========================================================

def grads_or_zeros(

    loss,

    params,

    retain_graph
):

    if not params:

        return []


    grads = torch.autograd.grad(

        loss,

        params,

        retain_graph=retain_graph,

        allow_unused=True
    )


    return [

        torch.zeros_like(
            parameter
        )

        if gradient is None

        else gradient

        for (
            parameter,
            gradient
        )

        in zip(
            params,
            grads
        )
    ]



def flatten_grads(
    grads
):

    if not grads:

        return None


    return torch.cat(

        [

            gradient.reshape(
                -1
            )

            for gradient in grads
        ]
    )



def assign_flat_grad(

    params,

    flat_grad
):

    offset = 0


    for parameter in params:

        n = parameter.numel()


        parameter.grad = (

            flat_grad[

                offset
                :
                offset + n

            ]

            .view_as(
                parameter
            )

            .detach()

            .clone()
        )


        offset += n


    if offset != flat_grad.numel():

        raise RuntimeError(

            "Flat gradient ile parameter boyutları uyuşmuyor."
        )



def pcgrad_backward_equal_weight(

    model,

    ret_loss,

    vol_loss
):

    groups = model.pcgrad_groups()


    shared = groups[
        "shared"
    ]


    ret_params = groups[
        "return_specific"
    ]


    vol_params = groups[
        "vol_specific"
    ]


    model.zero_grad(
        set_to_none=True
    )


    conflict = False

    shared_dot = None


    # ------------------------------------------------------
    # SHARED PARAMETRELER
    # ------------------------------------------------------

    if shared:

        g_ret = flatten_grads(

            grads_or_zeros(

                ret_loss,

                shared,

                retain_graph=True
            )
        )


        g_vol = flatten_grads(

            grads_or_zeros(

                vol_loss,

                shared,

                retain_graph=True
            )
        )


        dot = torch.dot(

            g_ret,

            g_vol
        )


        shared_dot = float(

            dot
            .detach()
            .cpu()
            .item()
        )


        ret_norm_sq = torch.dot(

            g_ret,

            g_ret
        )


        vol_norm_sq = torch.dot(

            g_vol,

            g_vol
        )


        eps = torch.finfo(

            g_ret.dtype

        ).eps


        if dot < 0:

            conflict = True


            g_ret_proj = (

                g_ret

                -

                dot
                /
                (
                    vol_norm_sq
                    +
                    eps
                )

                *
                g_vol
            )


            g_vol_proj = (

                g_vol

                -

                dot
                /
                (
                    ret_norm_sq
                    +
                    eps
                )

                *
                g_ret
            )


        else:

            g_ret_proj = g_ret

            g_vol_proj = g_vol


        assign_flat_grad(

            shared,

            0.5
            *
            (
                g_ret_proj
                +
                g_vol_proj
            )
        )


    # ------------------------------------------------------
    # TASK-SPECIFIC PARAMETRELER
    # ------------------------------------------------------

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


    for (
        parameter,
        gradient
    ) in zip(

        ret_params,

        ret_grads
    ):

        parameter.grad = (

            0.5
            *
            gradient

        ).detach().clone()


    for (
        parameter,
        gradient
    ) in zip(

        vol_params,

        vol_grads
    ):

        parameter.grad = (

            0.5
            *
            gradient

        ).detach().clone()


    return {

        "conflict":
            conflict,

        "shared_dot":
            shared_dot,

        "shared_param_count":
            int(

                sum(

                    parameter.numel()

                    for parameter in shared
                )
            )
    }


# ==========================================================
# 11. DATALOADER
# ==========================================================

def make_loaders(

    X_train,

    y_train,

    X_val,

    y_val
):

    train_ds = TensorDataset(

        torch.tensor(

            X_train,

            dtype=torch.float32
        ),

        torch.tensor(

            y_train,

            dtype=torch.float32
        )
    )


    val_ds = TensorDataset(

        torch.tensor(

            X_val,

            dtype=torch.float32
        ),

        torch.tensor(

            y_val,

            dtype=torch.float32
        )
    )


    generator = torch.Generator()


    generator.manual_seed(

        SEED
    )


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


    return (

        train_loader,

        val_loader
    )


# ==========================================================
# 12. TRAIN ONE EPOCH
# ==========================================================

def train_one_epoch(

    model,

    loader,

    optimizer,

    loss_strategy
):

    model.train()


    sums = {

        "total":
            0.0,

        "ret":
            0.0,

        "vol":
            0.0
    }


    n_obs = 0

    conflicts = 0

    pc_batches = 0

    dots = []


    for (
        X_batch,
        y_batch
    ) in loader:


        X_batch = X_batch.to(

            DEVICE
        )


        y_batch = y_batch.to(

            DEVICE
        )


        optimizer.zero_grad(

            set_to_none=True
        )


        pred = model(

            X_batch
        )


        (
            total,

            ret_loss,

            vol_loss

        ) = task_losses(

            pred,

            y_batch
        )


        if not torch.isfinite(

            total
        ):

            raise RuntimeError(

                "Training loss finite değil."
            )


        if loss_strategy == "FixedLambda_0.5":

            total.backward()


        elif loss_strategy == "PCGrad":

            info = pcgrad_backward_equal_weight(

                model,

                ret_loss,

                vol_loss
            )


            pc_batches += 1


            conflicts += int(

                info[
                    "conflict"
                ]
            )


            if info[
                "shared_dot"
            ] is not None:

                dots.append(

                    info[
                        "shared_dot"
                    ]
                )


        else:

            raise ValueError(

                f"Bilinmeyen loss strategy: {loss_strategy}"
            )


        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            GRAD_CLIP
        )


        optimizer.step()


        batch_size = X_batch.size(
            0
        )


        sums[
            "total"
        ] += (

            total.item()

            *
            batch_size
        )


        sums[
            "ret"
        ] += (

            ret_loss.item()

            *
            batch_size
        )


        sums[
            "vol"
        ] += (

            vol_loss.item()

            *
            batch_size
        )


        n_obs += batch_size


    return {

        "loss":
            sums[
                "total"
            ]
            /
            n_obs,

        "return_loss":
            sums[
                "ret"
            ]
            /
            n_obs,

        "vol_loss":
            sums[
                "vol"
            ]
            /
            n_obs,

        "pcgrad_conflict_batches":
            conflicts,

        "pcgrad_total_batches":
            pc_batches,

        "pcgrad_conflict_rate":
            (

                conflicts
                /
                pc_batches

                if pc_batches

                else 0.0
            ),

        "pcgrad_mean_shared_dot":
            (

                float(
                    np.mean(
                        dots
                    )
                )

                if dots

                else None
            )
    }


# ==========================================================
# 13. VALIDATION PREDICTION
# ==========================================================

@torch.no_grad()

def predict_scaled(

    model,

    loader
):

    model.eval()


    preds = []

    trues = []


    for (
        X_batch,
        y_batch
    ) in loader:


        pred = model(

            X_batch.to(
                DEVICE
            )
        )


        preds.append(

            pred
            .detach()
            .cpu()
            .numpy()
        )


        trues.append(

            y_batch.numpy()
        )


    return (

        np.vstack(
            preds
        ),

        np.vstack(
            trues
        )
    )


# ==========================================================
# 14. LOOKBACK VERİSİNİ YÜKLE
# TEST DİZİLERİ YÜKLENMEZ.
# ==========================================================

def load_data(

    lookback
):

    lb_dir = os.path.join(

        SEQUENCE_DIR,

        FEATURE_SET,

        f"lb{lookback}"
    )


    scaler_path = os.path.join(

        SEQUENCE_DIR,

        FEATURE_SET,

        "scalers.pkl"
    )


    files = {

        "X_train":
            os.path.join(
                lb_dir,
                "X_train.npy"
            ),

        "y_train":
            os.path.join(
                lb_dir,
                "y_train.npy"
            ),

        "X_val":
            os.path.join(
                lb_dir,
                "X_val.npy"
            ),

        "y_val":
            os.path.join(
                lb_dir,
                "y_val.npy"
            ),

        "y_val_raw":
            os.path.join(
                lb_dir,
                "y_val_raw.npy"
            )
    }


    for path in (

        list(
            files.values()
        )

        +

        [
            scaler_path
        ]
    ):

        if not os.path.exists(
            path
        ):

            raise FileNotFoundError(

                f"Gerekli dosya yok:\n{path}"
            )


    arrays = {

        key:
            np.load(
                path
            )

        for (
            key,
            path
        ) in files.items()
    }


    with open(

        scaler_path,

        "rb"

    ) as f:

        y_scaler = pickle.load(
            f
        )[
            "y_scaler"
        ]


    inverse_check = (

        y_scaler.inverse_transform(

            arrays[
                "y_val"
            ]
        )
    )


    max_diff = float(

        np.max(

            np.abs(

                inverse_check

                -

                arrays[
                    "y_val_raw"
                ]
            )
        )
    )


    if max_diff > 1e-5:

        raise RuntimeError(

            f"Inverse-scale kontrolü geçmedi: {max_diff}"
        )


    return (

        arrays[
            "X_train"
        ],

        arrays[
            "y_train"
        ],

        arrays[
            "X_val"
        ],

        arrays[
            "y_val"
        ],

        arrays[
            "y_val_raw"
        ],

        y_scaler,

        max_diff
    )


# ==========================================================
# 15. ÇIKTI DOSYALARI
# ==========================================================

results_path = os.path.join(

    RESULTS_DIR,

    "mini_grid_results_v4.csv"
)


ranked_path = os.path.join(

    RESULTS_DIR,

    "mini_grid_results_ranked_v4.csv"
)


history_path = os.path.join(

    RESULTS_DIR,

    "mini_grid_history_v4.csv"
)


pcgrad_audit_path = os.path.join(

    RESULTS_DIR,

    "mini_grid_pcgrad_audit_v4.csv"
)


summary_path = os.path.join(

    RESULTS_DIR,

    "mini_grid_summary_v4.json"
)


# ==========================================================
# 16. RESUME
# ==========================================================

if os.path.exists(
    results_path
):

    existing = pd.read_csv(

        results_path
    )

else:

    existing = pd.DataFrame()


success_keys = set()


if not existing.empty:

    done = existing[

        existing[
            "status"
        ]

        ==

        "success"
    ]


    success_keys = {

        (

            str(
                row[
                    "architecture"
                ]
            ),

            str(
                row[
                    "loss_strategy"
                ]
            ),

            int(
                row[
                    "lookback"
                ]
            )
        )

        for (
            _,
            row
        ) in done.iterrows()
    }


# ==========================================================
# 17. 8 KONFİGÜRASYON
# ==========================================================

configs = []


for (

    config_id,

    (
        architecture,
        loss_strategy,
        lookback
    )

) in enumerate(

    itertools.product(

        ARCHITECTURES,

        LOSS_STRATEGIES,

        LOOKBACKS
    ),

    start=1
):

    configs.append(

        {

            "config_id":
                config_id,

            "architecture":
                architecture,

            "loss_strategy":
                loss_strategy,

            "lookback":
                lookback
        }
    )


new_results = []

new_history = []


# ==========================================================
# 18. MINI-GRID
# ==========================================================

for config in configs:


    config_id = config[
        "config_id"
    ]


    architecture = config[
        "architecture"
    ]


    loss_strategy = config[
        "loss_strategy"
    ]


    lookback = config[
        "lookback"
    ]


    key = (

        architecture,

        loss_strategy,

        lookback
    )


    print(
        "\n"
        +
        "=" * 80
    )


    print(

        f"CONFIG {config_id}/{TOTAL_CONFIGS} | "

        f"{architecture} | "

        f"{loss_strategy} | "

        f"lb={lookback}"
    )


    print(
        "=" * 80
    )


    if key in success_keys:

        print(

            "[SKIP] Daha önce success."
        )

        continue


    try:

        set_seed(
            SEED
        )


        (

            X_train,

            y_train,

            X_val,

            y_val,

            y_val_raw,

            y_scaler,

            max_diff

        ) = load_data(

            lookback
        )


        (

            train_loader,

            val_loader

        ) = make_loaders(

            X_train,

            y_train,

            X_val,

            y_val
        )


        model = build_model(

            architecture,

            X_train.shape[
                2
            ],

            lookback

        ).to(
            DEVICE
        )


        parameter_count = int(

            sum(

                parameter.numel()

                for parameter in model.parameters()

                if parameter.requires_grad
            )
        )


        shared_count = int(

            sum(

                parameter.numel()

                for parameter in (

                    model
                    .pcgrad_groups()
                    [
                        "shared"
                    ]
                )
            )
        )


        optimizer = torch.optim.AdamW(

            model.parameters(),

            lr=LR,

            weight_decay=WEIGHT_DECAY
        )


        best_score = float(
            "inf"
        )


        best_epoch = None

        best_score_obj = None

        best_metrics = None

        best_state = None


        total_conflicts = 0

        total_pc_batches = 0


        # --------------------------------------------------
        # 5 EPOCH
        # --------------------------------------------------

        for epoch in range(

            1,

            EPOCHS
            +
            1
        ):


            train_info = train_one_epoch(

                model,

                train_loader,

                optimizer,

                loss_strategy
            )


            (

                pred_scaled,

                true_scaled

            ) = predict_scaled(

                model,

                val_loader
            )


            pred_raw = (

                y_scaler.inverse_transform(

                    pred_scaled
                )
            )


            true_raw_check = (

                y_scaler.inverse_transform(

                    true_scaled
                )
            )


            true_diff = float(

                np.max(

                    np.abs(

                        true_raw_check

                        -

                        y_val_raw
                    )
                )
            )


            if true_diff > 1e-5:

                raise RuntimeError(

                    "Validation true inverse-scale uyuşmuyor: "
                    f"{true_diff}"
                )


            (

                metrics_df,

                score_obj

            ) = score_validation(

                y_val_raw,

                pred_raw
            )


            score = score_obj[

                "validation_score"
            ]


            total_conflicts += (

                train_info[

                    "pcgrad_conflict_batches"
                ]
            )


            total_pc_batches += (

                train_info[

                    "pcgrad_total_batches"
                ]
            )


            new_history.append(

                {

                    "config_id":
                        config_id,

                    "architecture":
                        architecture,

                    "loss_strategy":
                        loss_strategy,

                    "lookback":
                        lookback,

                    "epoch":
                        epoch,

                    "train_loss":
                        train_info[
                            "loss"
                        ],

                    "train_return_loss":
                        train_info[
                            "return_loss"
                        ],

                    "train_vol_loss":
                        train_info[
                            "vol_loss"
                        ],

                    "validation_score":
                        score,

                    "avg_return_ratio":
                        score_obj[
                            "avg_return_ratio"
                        ],

                    "avg_vol_ratio":
                        score_obj[
                            "avg_vol_ratio"
                        ],

                    "pcgrad_conflict_batches":
                        train_info[
                            "pcgrad_conflict_batches"
                        ],

                    "pcgrad_total_batches":
                        train_info[
                            "pcgrad_total_batches"
                        ],

                    "pcgrad_conflict_rate":
                        train_info[
                            "pcgrad_conflict_rate"
                        ],

                    "pcgrad_mean_shared_dot":
                        train_info[
                            "pcgrad_mean_shared_dot"
                        ]
                }
            )


            print(

                f"Epoch {epoch:02d} | "

                f"train="
                f"{train_info['loss']:.6f} | "

                f"score="
                f"{score:.6f} | "

                f"ret="
                f"{score_obj['avg_return_ratio']:.6f} | "

                f"vol="
                f"{score_obj['avg_vol_ratio']:.6f}"
            )


            if score < best_score:


                best_score = float(
                    score
                )


                best_epoch = int(
                    epoch
                )


                best_score_obj = copy.deepcopy(

                    score_obj
                )


                best_metrics = metrics_df.copy()


                best_state = clone_state_to_cpu(

                    model
                )


        if best_state is None:

            raise RuntimeError(

                "Best checkpoint oluşmadı."
            )


        # --------------------------------------------------
        # CHECKPOINT
        # --------------------------------------------------

        model_path = os.path.join(

            MODEL_DIR,

            (
                f"mini_cfg{config_id:02d}_"
                f"{architecture}_"
                f"{loss_strategy}_"
                f"lb{lookback}_v4.pt"
            )
        )


        torch.save(

            {

                "model_state_dict":
                    best_state,

                "config": {

                    "config_id":
                        config_id,

                    "architecture":
                        architecture,

                    "loss_strategy":
                        loss_strategy,

                    "lookback":
                        lookback,

                    "feature_set":
                        FEATURE_SET,

                    "size":
                        "medium",

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

                    "epochs":
                        EPOCHS,

                    "seed":
                        SEED,

                    "purpose":
                        "mini_grid_audit_only"
                },

                "best_epoch":
                    best_epoch,

                "best_validation_score":
                    best_score,

                "parameter_count":
                    parameter_count,

                "shared_param_count":
                    shared_count,

                "test_arrays_loaded":
                    False,

                "test_metrics_computed":
                    False
            },

            model_path
        )


        # --------------------------------------------------
        # BEST METRICS
        # --------------------------------------------------

        metrics_path = os.path.join(

            RESULTS_DIR,

            f"mini_cfg{config_id:02d}_best_metrics_v4.csv"
        )


        best_metrics.to_csv(

            metrics_path,

            index=False
        )


        # --------------------------------------------------
        # RESULT ROW
        # --------------------------------------------------

        row = {

            "config_id":
                config_id,

            "status":
                "success",

            "architecture":
                architecture,

            "loss_strategy":
                loss_strategy,

            "lookback":
                lookback,

            "feature_set":
                FEATURE_SET,

            "size":
                "medium",

            "seed":
                SEED,

            "epochs":
                EPOCHS,

            "best_epoch":
                best_epoch,

            "validation_score":
                best_score_obj[
                    "validation_score"
                ],

            "avg_return_ratio":
                best_score_obj[
                    "avg_return_ratio"
                ],

            "avg_vol_ratio":
                best_score_obj[
                    "avg_vol_ratio"
                ],

            "parameter_count":
                parameter_count,

            "shared_param_count":
                shared_count,

            "pcgrad_conflict_batches_total":
                total_conflicts,

            "pcgrad_batches_total":
                total_pc_batches,

            "pcgrad_conflict_rate_total":
                (

                    total_conflicts
                    /
                    total_pc_batches

                    if total_pc_batches

                    else 0.0
                ),

            "max_inverse_diff":
                max_diff,

            "model_checkpoint":
                model_path,

            "metrics_file":
                metrics_path,

            "test_arrays_loaded":
                False,

            "test_metrics_computed":
                False
        }


        for asset in ASSETS:


            row[

                f"return_ratio_{asset}"

            ] = best_score_obj[

                "return_ratios"

            ][

                asset

            ]


            row[

                f"vol_ratio_{asset}"

            ] = best_score_obj[

                "vol_ratios"

            ][

                asset

            ]


        new_results.append(
            row
        )


        pd.concat(

            [

                existing,

                pd.DataFrame(
                    new_results
                )
            ],

            ignore_index=True

        ).to_csv(

            results_path,

            index=False
        )


        print(

            f"[SUCCESS] best_epoch={best_epoch} | "
            f"best_score={best_score:.6f}"
        )


        del model

        del optimizer


        if torch.cuda.is_available():

            torch.cuda.empty_cache()


    except Exception as error:


        new_results.append(

            {

                "config_id":
                    config_id,

                "status":
                    "error",

                "architecture":
                    architecture,

                "loss_strategy":
                    loss_strategy,

                "lookback":
                    lookback,

                "feature_set":
                    FEATURE_SET,

                "size":
                    "medium",

                "seed":
                    SEED,

                "epochs":
                    EPOCHS,

                "error":
                    repr(
                        error
                    ),

                "test_arrays_loaded":
                    False,

                "test_metrics_computed":
                    False
            }
        )


        pd.concat(

            [

                existing,

                pd.DataFrame(
                    new_results
                )
            ],

            ignore_index=True

        ).to_csv(

            results_path,

            index=False
        )


        print(

            "[ERROR]",

            repr(
                error
            )
        )


        if torch.cuda.is_available():

            torch.cuda.empty_cache()


        raise


# ==========================================================
# 19. HISTORY KAYDET
# ==========================================================

if new_history:


    new_history_df = pd.DataFrame(

        new_history
    )


    if os.path.exists(
        history_path
    ):

        old_history = pd.read_csv(

            history_path
        )


        history_df = pd.concat(

            [

                old_history,

                new_history_df
            ],

            ignore_index=True
        )


    else:

        history_df = new_history_df


    history_df.to_csv(

        history_path,

        index=False
    )


# ==========================================================
# 20. RANKING
# ==========================================================

final_results = pd.read_csv(

    results_path
)


success = final_results[

    final_results[
        "status"
    ]

    ==

    "success"

].copy()


success = (

    success

    .sort_values(

        [

            "validation_score",

            "config_id"
        ],

        ascending=[

            True,

            True
        ]
    )

    .reset_index(
        drop=True
    )
)


success[
    "rank"
] = np.arange(

    1,

    len(success)
    +
    1
)


success.to_csv(

    ranked_path,

    index=False
)


# ==========================================================
# 21. PCGRAD EŞDEĞERLİK AUDIT
# ==========================================================

audit_rows = []


for lookback in LOOKBACKS:


    rows = success[

        (

            success[
                "architecture"
            ]

            ==

            "NoSharing"
        )

        &

        (

            success[
                "lookback"
            ]

            ==

            lookback
        )
    ]


    fixed_row = rows[

        rows[
            "loss_strategy"
        ]

        ==

        "FixedLambda_0.5"
    ]


    pcgrad_row = rows[

        rows[
            "loss_strategy"
        ]

        ==

        "PCGrad"
    ]


    if (

        len(
            fixed_row
        )
        ==
        1

        and

        len(
            pcgrad_row
        )
        ==
        1
    ):


        fixed_score = float(

            fixed_row
            .iloc[0]
            [
                "validation_score"
            ]
        )


        pcgrad_score = float(

            pcgrad_row
            .iloc[0]
            [
                "validation_score"
            ]
        )


        difference = abs(

            fixed_score

            -

            pcgrad_score
        )


        audit_rows.append(

            {

                "architecture":
                    "NoSharing",

                "lookback":
                    lookback,

                "fixedlambda_score":
                    fixed_score,

                "pcgrad_score":
                    pcgrad_score,

                "absolute_score_difference":
                    difference,

                "numerically_equal_tol_1e_10":
                    bool(

                        difference
                        <=
                        1e-10
                    ),

                "interpretation":
                    (
                        "No shared parameters: "
                        "equal-weight PCGrad should reduce "
                        "to FixedLambda_0.5."
                    )
            }
        )


audit_df = pd.DataFrame(

    audit_rows
)


audit_df.to_csv(

    pcgrad_audit_path,

    index=False
)


# ==========================================================
# 22. SUMMARY
# ==========================================================

summary = {

    "project_version":
        "v4_repro",

    "created_at":
        datetime.now().isoformat(),

    "script":
        "05a_mini_grid_v4.py",

    "purpose":
        (
            "Audit only: FullSharingMTL/NoSharing, "
            "FixedLambda_0.5/PCGrad, lookback 10/30, "
            "checkpointing, resume, validation score ve "
            "NoSharing eşdeğerlik davranışını doğrular."
        ),

    "feature_set":
        FEATURE_SET,

    "size":
        "medium",

    "seed":
        SEED,

    "epochs":
        EPOCHS,

    "total_configs_expected":
        TOTAL_CONFIGS,

    "success_configs":
        int(
            len(
                success
            )
        ),

    "all_success":
        bool(

            len(
                success
            )

            ==

            TOTAL_CONFIGS
        ),

    "test_arrays_loaded":
        False,

    "test_metrics_computed":
        False,

    "pcgrad_method":
        (
            "Equal-weight shared-parameter PCGrad: "
            "projeksiyon yalnızca shared parametrelerde, "
            "iki görev gradienti çatıştığında uygulanır. "
            "Task-specific parametreler kendi görev gradientini "
            "0.5 katsayısıyla alır. NoSharing durumunda PCGrad, "
            "FixedLambda_0.5'e indirgenir."
        )
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
# 23. SON ÇIKTI
# ==========================================================

print(
    "\n"
    +
    "=" * 80
)


print(
    "MINI-GRID RANKING"
)


print(
    "=" * 80
)


display_columns = [

    "rank",

    "config_id",

    "architecture",

    "loss_strategy",

    "lookback",

    "best_epoch",

    "validation_score",

    "avg_return_ratio",

    "avg_vol_ratio",

    "parameter_count",

    "shared_param_count",

    "pcgrad_conflict_rate_total"
]


print(

    success[

        display_columns

    ].to_string(

        index=False
    )
)


print(
    "\n"
    +
    "=" * 80
)


print(
    "PCGRAD AUDIT — NoSharing vs FixedLambda_0.5"
)


print(
    "=" * 80
)


if len(
    audit_df
) > 0:

    print(

        audit_df.to_string(

            index=False
        )
    )


else:

    print(

        "Karşılaştırılabilir çift yok."
    )


if len(
    success
) != TOTAL_CONFIGS:

    raise RuntimeError(

        f"Mini-grid tamamlanmadı: "
        f"{len(success)}/{TOTAL_CONFIGS} success"
    )


print(
    "\n"
    +
    "=" * 80
)


print(
    "05a_mini_grid_v4.py BAŞARIYLA TAMAMLANDI"
)


print(
    "=" * 80
)


print(
    "✅ 8/8 config success."
)

print(
    "✅ FullSharingMTL çalıştı."
)

print(
    "✅ NoSharing çalıştı."
)

print(
    "✅ FixedLambda_0.5 çalıştı."
)

print(
    "✅ Shared-parameter PCGrad çalıştı."
)

print(
    "✅ Best checkpoint gerçek CPU clone ile saklandı."
)

print(
    "✅ Resume yalnızca success configleri atlıyor."
)

print(
    "✅ ValidationScore yalnızca validation ile hesaplandı."
)

print(
    "✅ Test dizileri yüklenmedi."
)

print(
    "✅ Test metriği hesaplanmadı."
)

print(
    "✅ NoSharing PCGrad eşdeğerliği ayrıca audit edildi."
)

print(
    "=" * 80
)
