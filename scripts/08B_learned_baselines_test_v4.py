# ==========================================================
# 08B_learned_baselines_test_v4.py
# v4_repro — RESMÎ LEARNED BASELINES AŞAMASI
#
# KİLİTLİ PROTOKOL
# - SingleTaskTransformer: fixed, final NoSharing branch-matched
# - LSTM grid: hidden {32,64,128} × layers {1,2}
# - XGBoost grid: max_depth {3,4,6} × lr {0.03,0.05}
# - Seeds: [123,777,2026] for all stochastic learned baselines
# - Neural budget: MAX=100, MIN=45, PATIENCE=15, batch=64,
#   AdamW, lr=1e-3, wd=1e-4, grad_clip=1.0, dropout=0.10
# - Return neural loss=MSE; Vol neural loss=Pinball(tau=0.5)
# - XGB return=reg:squarederror; vol=reg:quantileerror(alpha=0.5)
# - XGB n_estimators=1000 upper bound; early_stop=30 on VAL only
# - Config selection: lowest 3-seed mean task-specific VAL score
# - Test never used for tuning/checkpoint/early-stop/config selection
# - Primary prediction: selected config's 3-seed RAW-scale mean
# - 08B is NOT a shared-representation MTL ablation.
# ==========================================================

from __future__ import annotations

import copy
import hashlib
import json
import pickle
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from torch.utils.data import DataLoader, TensorDataset


# ==========================================================
# 1. PATHS + LOCKED CONSTANTS
# ==========================================================

BASE_DIR = Path('/content/drive/MyDrive/tez_transformer_v4_repro')
SCRIPTS_DIR = BASE_DIR / 'scripts'
CONFIG_DIR = BASE_DIR / 'config'
SEQ_DIR = BASE_DIR / 'data' / 'sequences' / 'baseline' / 'lb10'
SCALERS_PATH = BASE_DIR / 'data' / 'sequences' / 'baseline' / 'scalers.pkl'
FINAL_TEST_DIR = BASE_DIR / 'results' / 'final_test'
NAIVE_DIR = BASE_DIR / 'results' / 'baselines' / 'naive'
OUT_DIR = BASE_DIR / 'results' / 'baselines' / 'learned'

CKPT_DIR = OUT_DIR / 'checkpoints' / 'neural'
XGB_DIR = OUT_DIR / 'models' / 'xgboost'
HIST_DIR = OUT_DIR / 'histories'
META_DIR = OUT_DIR / 'run_meta'
PRED_DIR = OUT_DIR / 'predictions'
for d in [OUT_DIR, CKPT_DIR, XGB_DIR, HIST_DIR, META_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GRID_CSV = OUT_DIR / 'learned_baseline_grid_results_v4.csv'
GRID_SUMMARY_CSV = OUT_DIR / 'learned_baseline_grid_summary_v4.csv'
SELECTION_CSV = OUT_DIR / 'learned_baseline_selection_v4.csv'
SELECTION_LOCK_JSON = OUT_DIR / 'learned_baseline_selection_lock_v4.json'
METRICS_CSV = OUT_DIR / 'learned_baseline_metrics_long_v4.csv'
COMPARISON_CSV = OUT_DIR / 'learned_baseline_comparison_v4.csv'
PARAM_CSV = OUT_DIR / 'learned_baseline_parameter_report_v4.csv'
LOSS_NPZ = OUT_DIR / 'learned_baseline_loss_series_v4.npz'
SUMMARY_JSON = OUT_DIR / 'learned_baseline_summary_v4.json'
PROGRESS_JSON = OUT_DIR / 'learned_baseline_progress_v4.json'
PROTOCOL_JSON = OUT_DIR / 'learned_baseline_protocol_lock_v4.json'

PROJECT_VERSION = 'v4_repro'
FEATURE_SET = 'baseline'
LOOKBACK = 10
SEEDS = [123, 777, 2026]
ASSETS = ['BIST100', 'USDTRY', 'EURTRY', 'GOLD']
TARGET_ORDER = [
    'BIST100_NextRet', 'USDTRY_NextRet', 'EURTRY_NextRet', 'GOLD_NextRet',
    'BIST100_NextVol', 'USDTRY_NextVol', 'EURTRY_NextVol', 'GOLD_NextVol'
]
TAU = 0.5
EXPECTED_TEST_N = 584

BATCH_SIZE = 64
MAX_EPOCHS = 100
MIN_EPOCHS = 45
PATIENCE = 15
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
DROPOUT = 0.10

D_MODEL = 32
N_HEAD = 4
N_LAYERS = 2
D_FF = 128

LSTM_HIDDEN = [32, 64, 128]
LSTM_LAYERS = [1, 2]

XGB_DEPTHS = [3, 4, 6]
XGB_LRS = [0.03, 0.05]
XGB_N_ESTIMATORS = 1000
XGB_EARLY_STOP = 30
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE = 0.8
XGB_TREE_METHOD = 'hist'

EXPECTED_HASHES = {
    '05_grid_search_v4.py': '5d250d9d727cef15e6411cd027aad6089bf62b2cbf4b2c13a8c0f28ff7191a78',
    '06_best_model_multiseed_v4.py': '35de2ee398699003dfef6be36b70c112fb2c0d1b1e9577cbf64bef58877e16d8',
    '07_final_test_evaluation_v4.py': '8b0e3cf2edb9508b4fddd402ddcdbf8c4d2acd6080ffe6fe1876ad818306cd74',
    '08A_naive_baselines_test_v4.py': '95a9658e97f57eaa1a9bb63ec29d8159432f06438bd57ff54fc8ab43013487e8',
}

STATE = {'test_arrays_loaded': False, 'test_metrics_computed': False}


# ==========================================================
# 2. HELPERS
# ==========================================================

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def json_default(x: Any) -> Any:
    if isinstance(x, Path): return str(x)
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.floating): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    raise TypeError(type(x))


def dump_json(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=json_default)
    tmp.replace(path)


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def cpu_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_progress(stage: str, **extra: Any) -> None:
    payload = {
        'project_version': PROJECT_VERSION,
        'updated_at_utc': now_utc(),
        'stage': stage,
        'test_arrays_loaded': STATE['test_arrays_loaded'],
        'test_metrics_computed': STATE['test_metrics_computed'],
        'final_model_changed': False,
        'final_model_retrained': False,
        **extra,
    }
    dump_json(payload, PROGRESS_JSON)


def task_cols(task: str) -> np.ndarray:
    if task == 'return': return np.arange(0, 4)
    if task == 'volatility': return np.arange(4, 8)
    raise ValueError(task)


def mae(y, p): return float(np.mean(np.abs(y - p)))
def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))

def r2(y, p):
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return float('nan') if ss_tot <= 0 else float(1 - ss_res / ss_tot)


def pinball(y, p, tau=0.5):
    d = y - p
    return float(np.mean(np.maximum(tau * d, (tau - 1.0) * d)))


def pinball_series(y, p, tau=0.5):
    d = y - p
    return np.maximum(tau * d, (tau - 1.0) * d)


# ==========================================================
# 3. PROVENANCE PREFLIGHT
# ==========================================================

print('=' * 110)
print('08B — v4 RESMÎ LEARNED BASELINES')
print('=' * 110)

for name, expected in EXPECTED_HASHES.items():
    p = SCRIPTS_DIR / name
    require(p)
    actual = sha256_file(p)
    if actual != expected:
        raise RuntimeError(f'SHA mismatch: {name}\nexpected={expected}\nactual={actual}')
    print(f'[PROVENANCE] {name}: SHA MATCH ✅')

manifest_path = CONFIG_DIR / 'code_manifest_v4.csv'
require(manifest_path)
manifest = pd.read_csv(manifest_path).astype(str)
for name, expected in EXPECTED_HASHES.items():
    name_rows = manifest.apply(lambda c: c.str.contains(name, regex=False, na=False)).any(axis=1)
    rows = manifest.loc[name_rows]
    if rows.empty:
        raise RuntimeError(f'Manifest missing: {name}')
    has_hash = rows.apply(lambda c: c.str.contains(expected, regex=False, na=False)).any(axis=1).any()
    if not has_hash:
        raise RuntimeError(f'Manifest hash mismatch: {name}')
print('[PROVENANCE] Manifest cross-check PASS ✅')


# ==========================================================
# 4. LOCK PROTOCOL BEFORE TRAINING
# ==========================================================

protocol = {
    'status': 'LOCKED_BEFORE_08B_RESULTS',
    'created_at_utc': now_utc(),
    'feature_set': FEATURE_SET,
    'lookback': LOOKBACK,
    'target_order': TARGET_ORDER,
    'seeds': SEEDS,
    'single_task_transformer': {
        'fixed': True, 'grid': False, 'branch_matched_to_final_nosharing': True,
        'd_model': D_MODEL, 'n_head': N_HEAD, 'n_layers': N_LAYERS,
        'd_ff': D_FF, 'dropout': DROPOUT,
        'position_encoding': 'learned_positional_embedding',
        'sequence_representation': 'last_timestep',
    },
    'lstm_grid': {
        'hidden_size': LSTM_HIDDEN, 'num_layers': LSTM_LAYERS,
        'config_count': 6, 'bidirectional': False,
        'sequence_representation': 'last_timestep',
        'recurrent_dropout_rule': {'1_layer': 0.0, '2_layers': DROPOUT},
        'head': 'Linear(h,h)->GELU->Dropout(0.10)->Linear(h,4)',
    },
    'xgboost_grid': {
        'max_depth': XGB_DEPTHS, 'learning_rate': XGB_LRS, 'config_count': 6,
        'n_estimators_upper_bound': XGB_N_ESTIMATORS,
        'early_stopping_rounds': XGB_EARLY_STOP,
        'early_stopping_set': 'validation',
        'subsample': XGB_SUBSAMPLE, 'colsample_bytree': XGB_COLSAMPLE,
        'tree_method': XGB_TREE_METHOD, 'input': '(10,8)->80_flatten',
        'return_objective': 'reg:squarederror',
        'volatility_objective': 'reg:quantileerror', 'quantile_alpha': TAU,
    },
    'neural_budget': {
        'max_epochs': MAX_EPOCHS, 'min_epochs': MIN_EPOCHS,
        'patience': PATIENCE, 'batch_size': BATCH_SIZE,
        'optimizer': 'AdamW', 'learning_rate': LR,
        'weight_decay': WEIGHT_DECAY, 'grad_clip': GRAD_CLIP,
    },
    'selection': {
        'return': 'min 3-seed mean AvgReturnRatio on validation',
        'volatility': 'min 3-seed mean AvgVolRatio on validation',
        'tie_break_1': 'lower sample std ddof=1',
        'tie_break_2': 'config_id lexical order',
        'test_used': False,
    },
    'primary_prediction': '3-seed arithmetic ensemble in raw target scale',
    'scaler_policy': 'load frozen scalers.pkl; no refit',
    'interpretation_boundary': (
        '08B is not a shared-representation MTL ablation; no causal claim that '
        'sharing helps/hurts or that negative transfer is proven.'
    ),
}
dump_json(protocol, PROTOCOL_JSON)
print(f'[LOCK] {PROTOCOL_JSON}')


# ==========================================================
# 5. LOAD TRAIN/VAL ONLY + FROZEN SCALER
# ==========================================================

meta_path = SEQ_DIR / 'sequence_meta.json'
require(meta_path)
meta = json.loads(meta_path.read_text(encoding='utf-8'))
assert meta['project_version'] == PROJECT_VERSION
assert meta['feature_set'] == FEATURE_SET
assert int(meta['lookback']) == LOOKBACK
assert meta['target_columns'] == TARGET_ORDER

X_train = np.load(SEQ_DIR / 'X_train.npy')
X_val = np.load(SEQ_DIR / 'X_val.npy')
y_train = np.load(SEQ_DIR / 'y_train.npy')
y_val = np.load(SEQ_DIR / 'y_val.npy')
y_train_raw = np.load(SEQ_DIR / 'y_train_raw.npy')
y_val_raw = np.load(SEQ_DIR / 'y_val_raw.npy')

assert X_train.shape == (2714, 10, 8)
assert X_val.shape == (584, 10, 8)
assert y_train.shape == y_train_raw.shape == (2714, 8)
assert y_val.shape == y_val_raw.shape == (584, 8)
for name, arr in [('X_train', X_train), ('X_val', X_val), ('y_train', y_train),
                  ('y_val', y_val), ('y_train_raw', y_train_raw), ('y_val_raw', y_val_raw)]:
    if not np.isfinite(arr).all(): raise RuntimeError(f'{name}: NaN/Inf')
print('[DATA] Train/validation loaded; test NOT loaded ✅')

require(SCALERS_PATH)
with SCALERS_PATH.open('rb') as f:
    scalers = pickle.load(f)


def find_target_scaler(obj: Any):
    seen = set()
    def walk(x):
        if id(x) in seen: return None
        seen.add(id(x))
        if hasattr(x, 'mean_') and hasattr(x, 'scale_'):
            m, s = np.asarray(x.mean_), np.asarray(x.scale_)
            if m.shape == (8,) and s.shape == (8,): return x
        if isinstance(x, dict):
            for k in ['y_scaler', 'target_scaler', 'targets', 'y']:
                if k in x:
                    z = walk(x[k])
                    if z is not None: return z
            for v in x.values():
                z = walk(v)
                if z is not None: return z
        if isinstance(x, (list, tuple)):
            for v in x:
                z = walk(v)
                if z is not None: return z
        return None
    return walk(obj)


y_scaler = find_target_scaler(scalers)
if y_scaler is None:
    raise RuntimeError('8-dimensional frozen target scaler not found.')
Y_MEAN = np.asarray(y_scaler.mean_, dtype=np.float64)
Y_SCALE = np.asarray(y_scaler.scale_, dtype=np.float64)
if np.any(Y_SCALE <= 0): raise RuntimeError('Invalid target scaler scale_.')
val_rebuilt = y_val.astype(np.float64) * Y_SCALE + Y_MEAN
val_inverse_diff = float(np.max(np.abs(val_rebuilt - y_val_raw.astype(np.float64))))
if val_inverse_diff > 1e-5:
    raise RuntimeError(f'Frozen scaler validation inverse diff too large: {val_inverse_diff}')
print(f'[SCALER] Frozen scaler loaded; NO REFIT. max_diff={val_inverse_diff:.3e} ✅')


def inverse_task(pred_scaled: np.ndarray, task: str) -> np.ndarray:
    cols = task_cols(task)
    if pred_scaled.shape[1] != 4: raise ValueError(pred_scaled.shape)
    return pred_scaled.astype(np.float64) * Y_SCALE[cols] + Y_MEAN[cols]


# Validation denominators from train/val only.
VAL_RETURN_ZERO = {a: mae(y_val_raw[:, i], np.zeros(len(y_val_raw))) for i, a in enumerate(ASSETS)}
val_vol_persist = np.empty((len(y_val_raw), 4), dtype=np.float64)
val_vol_persist[0] = y_train_raw[-1, 4:]
val_vol_persist[1:] = y_val_raw[:-1, 4:]
VAL_VOL_PERSIST = {
    a: pinball(y_val_raw[:, 4 + i], val_vol_persist[:, i], TAU)
    for i, a in enumerate(ASSETS)
}
if any(v <= 0 for v in VAL_RETURN_ZERO.values()) or any(v <= 0 for v in VAL_VOL_PERSIST.values()):
    raise RuntimeError('Non-positive validation denominator.')


def task_score(task: str, y_true_raw_task: np.ndarray, pred_raw_task: np.ndarray):
    ratios = {}
    if task == 'return':
        for i, a in enumerate(ASSETS):
            ratios[a] = mae(y_true_raw_task[:, i], pred_raw_task[:, i]) / VAL_RETURN_ZERO[a]
    else:
        for i, a in enumerate(ASSETS):
            ratios[a] = pinball(y_true_raw_task[:, i], pred_raw_task[:, i], TAU) / VAL_VOL_PERSIST[a]
    return float(np.mean(list(ratios.values()))), ratios


# Date semantics check using train/val only.
val_anchor = np.load(SEQ_DIR / 'anchor_dates_val.npy', allow_pickle=False)
val_target_dates = np.load(SEQ_DIR / 'target_realization_dates_val.npy', allow_pickle=False)
train_target_dates = np.load(SEQ_DIR / 'target_realization_dates_train.npy', allow_pickle=False)
assert len(val_anchor) == len(val_target_dates) == 584
assert np.all(val_target_dates > val_anchor)
assert train_target_dates[-1] == val_anchor[0]
print('[VAL DATES] chronology/persistence boundary PASS ✅')


# ==========================================================
# 6. MODELS + LOSSES
# ==========================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('[DEVICE]', DEVICE)
if torch.cuda.is_available(): print('[GPU]', torch.cuda.get_device_name(0))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, n_layers, dropout):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
    def forward(self, x): return self.encoder(x)


def make_head(d, dropout):
    return nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 4))


class SingleTaskTransformer(nn.Module):
    def __init__(self, n_features=8, lookback=10):
        super().__init__()
        self.input_projection = nn.Linear(n_features, D_MODEL)
        self.positional_embedding = nn.Parameter(torch.zeros(1, lookback, D_MODEL))
        self.encoder = TransformerBlock(D_MODEL, N_HEAD, D_FF, N_LAYERS, DROPOUT)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = make_head(D_MODEL, DROPOUT)
    def forward(self, x):
        h = self.input_projection(x)
        h = h + self.positional_embedding[:, :h.size(1), :]
        h = self.encoder(h)
        h = self.norm(h[:, -1, :])
        return self.head(h)


class SingleTaskLSTM(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        rec_dropout = 0.0 if num_layers == 1 else DROPOUT
        self.lstm = nn.LSTM(
            input_size=8, hidden_size=hidden_size, num_layers=num_layers,
            dropout=rec_dropout, batch_first=True, bidirectional=False
        )
        self.head = make_head(hidden_size, DROPOUT)
    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h[:, -1, :])


def build_neural(family: str, cfg: Dict[str, Any]):
    if family == 'SingleTaskTransformer': return SingleTaskTransformer()
    if family == 'SingleTaskLSTM': return SingleTaskLSTM(int(cfg['hidden_size']), int(cfg['num_layers']))
    raise ValueError(family)


MSE = nn.MSELoss()
def pinball_torch(pred, true, tau=0.5):
    d = true - pred
    return torch.maximum(tau * d, (tau - 1.0) * d).mean()

def task_loss(task, pred, true):
    return MSE(pred, true) if task == 'return' else pinball_torch(pred, true, TAU)


def make_loaders(task: str, seed: int):
    cols = task_cols(task)
    tr = TensorDataset(torch.from_numpy(X_train.astype(np.float32, copy=False)),
                       torch.from_numpy(y_train[:, cols].astype(np.float32, copy=False)))
    va = TensorDataset(torch.from_numpy(X_val.astype(np.float32, copy=False)),
                       torch.from_numpy(y_val[:, cols].astype(np.float32, copy=False)))
    g = torch.Generator(); g.manual_seed(seed)
    return (
        DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, generator=g),
        DataLoader(va, batch_size=BATCH_SIZE, shuffle=False, drop_last=False),
    )


@torch.no_grad()
def predict_neural_scaled(model: nn.Module, X: np.ndarray):
    ds = TensorDataset(torch.from_numpy(X.astype(np.float32, copy=False)))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    model.eval(); out = []
    for (xb,) in dl:
        p = model(xb.to(DEVICE)).detach().cpu().numpy()
        out.append(p)
    p = np.concatenate(out, axis=0)
    if p.shape != (len(X), 4) or not np.isfinite(p).all(): raise RuntimeError(f'Bad neural pred: {p.shape}')
    return p


# ==========================================================
# 7. NEURAL RUN + RESUME
# ==========================================================

def train_neural_run(family: str, task: str, config_id: str, cfg: Dict[str, Any], seed: int):
    run_id = f'{family}__{task}__{config_id}__seed{seed}'
    ckpt = CKPT_DIR / f'{run_id}.pt'
    hist = HIST_DIR / f'{run_id}_history.csv'
    meta_file = META_DIR / f'{run_id}.json'
    if ckpt.exists() and hist.exists() and meta_file.exists():
        m = json.loads(meta_file.read_text(encoding='utf-8'))
        if m.get('status') == 'success':
            print('[RESUME]', run_id)
            return m['result_row']

    print('\n' + '=' * 110 + f'\n[NEURAL RUN] {run_id}\n' + '=' * 110)
    set_seed(seed)
    model = build_neural(family, cfg).to(DEVICE)
    pcount = count_params(model)
    train_loader, _ = make_loaders(task, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    cols = task_cols(task)
    y_val_task_raw = y_val_raw[:, cols]

    best_score, best_epoch, best_state, best_ratios = float('inf'), 0, None, None
    no_improve, epochs_ran = 0, 0
    history = []
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        epochs_ran = epoch
        model.train(); loss_sum = 0.0; n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = task_loss(task, pred, yb)
            if not torch.isfinite(loss): raise RuntimeError(f'Non-finite loss: {run_id}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            bn = xb.size(0); loss_sum += float(loss.item()) * bn; n += bn

        vp_scaled = predict_neural_scaled(model, X_val)
        vp_raw = inverse_task(vp_scaled, task)
        score, ratios = task_score(task, y_val_task_raw, vp_raw)
        history.append({
            'model_family': family, 'task': task, 'config_id': config_id, 'seed': seed,
            'epoch': epoch, 'train_loss': loss_sum / n,
            'validation_task_score': score,
            **{f'ratio_{a}': ratios[a] for a in ASSETS},
        })
        if score < best_score:
            best_score, best_epoch = float(score), epoch
            best_state, best_ratios = cpu_state(model), copy.deepcopy(ratios)
            no_improve = 0
        else:
            no_improve += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == best_epoch:
            print(f'[{run_id}] epoch={epoch:3d} val={score:.8f} best={best_score:.8f}@{best_epoch}')
        if epoch >= MIN_EPOCHS and no_improve >= PATIENCE:
            print(f'[EARLY STOP] {run_id} at epoch={epoch}')
            break

    if best_state is None: raise RuntimeError(f'No best state: {run_id}')
    torch.save({
        'project_version': PROJECT_VERSION, 'stage': '08B', 'model_family': family,
        'task': task, 'config_id': config_id, 'config': cfg, 'seed': seed,
        'model_state_dict': best_state, 'parameter_count': pcount,
        'best_epoch': best_epoch, 'epochs_ran': epochs_ran,
        'best_validation_task_score': best_score,
        'test_arrays_loaded_during_training': False,
        'test_used_for_checkpoint_selection': False,
    }, ckpt)
    pd.DataFrame(history).to_csv(hist, index=False)
    row = {
        'status': 'success', 'algorithm_group': 'neural', 'model_family': family,
        'task': task, 'config_id': config_id,
        'config_json': json.dumps(cfg, sort_keys=True), 'seed': seed,
        'validation_task_score': best_score,
        'validation_score_name': 'AvgReturnRatio' if task == 'return' else 'AvgVolRatio',
        'best_epoch': best_epoch, 'epochs_ran': epochs_ran,
        'parameter_count': pcount, 'elapsed_seconds': time.time() - t0,
        'test_arrays_loaded_during_training': False, 'test_used_for_selection': False,
        **{f'ratio_{a}': best_ratios[a] for a in ASSETS},
    }
    dump_json({'status': 'success', 'run_id': run_id, 'result_row': row}, meta_file)
    return row


def load_neural_predict(family, task, config_id, cfg, seed, X):
    run_id = f'{family}__{task}__{config_id}__seed{seed}'
    ckpt = CKPT_DIR / f'{run_id}.pt'; require(ckpt)
    payload = torch.load(ckpt, map_location='cpu', weights_only=False)
    assert payload['task'] == task and payload['config_id'] == config_id and int(payload['seed']) == seed
    model = build_neural(family, cfg)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    return inverse_task(predict_neural_scaled(model.to(DEVICE), X), task)


# ==========================================================
# 8. XGBOOST RUN + RESUME
# ==========================================================

# Fiilî objective smoke test.
rng = np.random.default_rng(42)
Xd, yd = rng.normal(size=(50, 5)), rng.normal(size=50)
for kw in [
    dict(objective='reg:squarederror'),
    dict(objective='reg:quantileerror', quantile_alpha=TAU),
]:
    m = xgb.XGBRegressor(n_estimators=5, max_depth=2, learning_rate=0.1,
                         random_state=42, n_jobs=1, tree_method='hist', verbosity=0, **kw)
    m.fit(Xd, yd)
    if not np.isfinite(m.predict(Xd)).all(): raise RuntimeError(f'XGB smoke fail: {kw}')
print('[XGBOOST] objectives work ✅')

X_train_flat = X_train.reshape(len(X_train), -1)
X_val_flat = X_val.reshape(len(X_val), -1)
assert X_train_flat.shape == (2714, 80) and X_val_flat.shape == (584, 80)


def make_xgb(task: str, cfg: Dict[str, Any], seed: int):
    common = dict(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=int(cfg['max_depth']),
        learning_rate=float(cfg['learning_rate']),
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE,
        tree_method=XGB_TREE_METHOD,
        random_state=seed,
        early_stopping_rounds=XGB_EARLY_STOP,
        n_jobs=-1, verbosity=0, validate_parameters=True, device='cpu',
    )
    if task == 'return': return xgb.XGBRegressor(objective='reg:squarederror', **common)
    return xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=TAU, **common)


def xgb_predict_best(model, X):
    bi = getattr(model, 'best_iteration', None)
    p = model.predict(X, iteration_range=(0, int(bi) + 1)) if bi is not None else model.predict(X)
    p = np.asarray(p).reshape(-1)
    if not np.isfinite(p).all(): raise RuntimeError('Non-finite XGB pred')
    return p


def xgb_paths(task, config_id, seed):
    run_id = f'XGBoost__{task}__{config_id}__seed{seed}'
    return META_DIR / f'{run_id}.json', [XGB_DIR / f'{run_id}__{a}.json' for a in ASSETS]


def train_xgb_run(task: str, config_id: str, cfg: Dict[str, Any], seed: int):
    run_id = f'XGBoost__{task}__{config_id}__seed{seed}'
    meta_file, model_files = xgb_paths(task, config_id, seed)
    if meta_file.exists() and all(p.exists() for p in model_files):
        m = json.loads(meta_file.read_text(encoding='utf-8'))
        if m.get('status') == 'success':
            print('[RESUME]', run_id)
            return m['result_row']

    print('\n' + '=' * 110 + f'\n[XGB RUN] {run_id}\n' + '=' * 110)
    cols = task_cols(task)
    pred_val_scaled = np.empty((584, 4), dtype=np.float64)
    best_iter, rounds = [], []
    t0 = time.time()

    for i, col in enumerate(cols):
        model = make_xgb(task, cfg, seed)
        model.fit(X_train_flat, y_train[:, col], eval_set=[(X_val_flat, y_val[:, col])], verbose=False)
        pred_val_scaled[:, i] = xgb_predict_best(model, X_val_flat)
        model.save_model(model_files[i])
        bi = getattr(model, 'best_iteration', None)
        if bi is None: bi = model.get_booster().num_boosted_rounds() - 1
        best_iter.append(int(bi)); rounds.append(int(model.get_booster().num_boosted_rounds()))
        print(f'[{run_id}] {ASSETS[i]} best_iter={best_iter[-1]} rounds={rounds[-1]}')

    pred_val_raw = inverse_task(pred_val_scaled, task)
    score, ratios = task_score(task, y_val_raw[:, cols], pred_val_raw)
    row = {
        'status': 'success', 'algorithm_group': 'xgboost', 'model_family': 'XGBoost',
        'task': task, 'config_id': config_id, 'config_json': json.dumps(cfg, sort_keys=True),
        'seed': seed, 'validation_task_score': score,
        'validation_score_name': 'AvgReturnRatio' if task == 'return' else 'AvgVolRatio',
        'best_epoch': np.nan, 'epochs_ran': np.nan, 'parameter_count': np.nan,
        'elapsed_seconds': time.time() - t0,
        'avg_best_iteration': float(np.mean(best_iter)),
        'avg_boosted_rounds': float(np.mean(rounds)),
        'test_arrays_loaded_during_training': False, 'test_used_for_selection': False,
        **{f'ratio_{a}': ratios[a] for a in ASSETS},
    }
    dump_json({
        'status': 'success', 'run_id': run_id, 'task': task, 'config_id': config_id,
        'config': cfg, 'seed': seed, 'model_paths': [str(p) for p in model_files],
        'best_iterations': best_iter, 'boosted_rounds': rounds,
        'result_row': row, 'test_arrays_loaded_during_training': False,
        'test_used_for_selection': False,
    }, meta_file)
    return row


def load_xgb_predict(task, config_id, cfg, seed, X_flat):
    meta_file, model_files = xgb_paths(task, config_id, seed)
    require(meta_file)
    pred_scaled = np.empty((len(X_flat), 4), dtype=np.float64)
    for i, p in enumerate(model_files):
        require(p)
        model = make_xgb(task, cfg, seed)
        model.load_model(p)
        pred_scaled[:, i] = xgb_predict_best(model, X_flat)
    return inverse_task(pred_scaled, task)


# ==========================================================
# 9. LOCKED CONFIG GRIDS + VALIDATION RUNS
# ==========================================================

TRANSFORMER_CFG = [{
    'config_id': 'transformer_fixed_branchmatched',
    'config': {'d_model': D_MODEL, 'n_head': N_HEAD, 'n_layers': N_LAYERS, 'd_ff': D_FF, 'dropout': DROPOUT},
}]
LSTM_CFGS = [
    {'config_id': f'lstm_h{h}_l{l}', 'config': {
        'hidden_size': h, 'num_layers': l,
        'recurrent_dropout': 0.0 if l == 1 else DROPOUT,
        'head_dropout': DROPOUT, 'bidirectional': False,
        'sequence_representation': 'last_timestep',
    }}
    for h in LSTM_HIDDEN for l in LSTM_LAYERS
]
XGB_CFGS = [
    {'config_id': f"xgb_d{d}_lr{str(lr).replace('.', 'p')}", 'config': {
        'max_depth': d, 'learning_rate': lr, 'n_estimators': XGB_N_ESTIMATORS,
        'early_stopping_rounds': XGB_EARLY_STOP, 'subsample': XGB_SUBSAMPLE,
        'colsample_bytree': XGB_COLSAMPLE, 'tree_method': XGB_TREE_METHOD,
    }}
    for d in XGB_DEPTHS for lr in XGB_LRS
]
assert len(LSTM_CFGS) == len(XGB_CFGS) == 6

rows = []; done = 0
save_progress('validation_grid_started', completed_run_count=done)

for obj in TRANSFORMER_CFG:
    for task in ['return', 'volatility']:
        for seed in SEEDS:
            save_progress('transformer_validation', family='SingleTaskTransformer', task=task,
                          config_id=obj['config_id'], seed=seed, completed_run_count=done)
            rows.append(train_neural_run('SingleTaskTransformer', task, obj['config_id'], obj['config'], seed)); done += 1

for obj in LSTM_CFGS:
    for task in ['return', 'volatility']:
        for seed in SEEDS:
            save_progress('lstm_validation_grid', family='SingleTaskLSTM', task=task,
                          config_id=obj['config_id'], seed=seed, completed_run_count=done)
            rows.append(train_neural_run('SingleTaskLSTM', task, obj['config_id'], obj['config'], seed)); done += 1

for obj in XGB_CFGS:
    for task in ['return', 'volatility']:
        for seed in SEEDS:
            save_progress('xgboost_validation_grid', family='XGBoost', task=task,
                          config_id=obj['config_id'], seed=seed, completed_run_count=done)
            rows.append(train_xgb_run(task, obj['config_id'], obj['config'], seed)); done += 1

runs = pd.DataFrame(rows)
expected_rows = 6 + 36 + 36
assert len(runs) == expected_rows and (runs['status'] == 'success').all()
runs.to_csv(GRID_CSV, index=False)
print(f'[VALIDATION GRID] {len(runs)}/{expected_rows} success ✅')


# ==========================================================
# 10. 3-SEED MEAN CONFIG SELECTION — VAL ONLY
# ==========================================================

def summarize_family_task(family: str, task: str):
    sub = runs[(runs.model_family == family) & (runs.task == task)].copy()
    out = []
    for config_id, g in sub.groupby('config_id', sort=True):
        found = sorted(g.seed.astype(int).unique().tolist())
        if found != SEEDS: raise RuntimeError(f'Seed mismatch: {family}/{task}/{config_id}: {found}')
        s = g.validation_task_score.astype(float).to_numpy()
        cfg_jsons = g.config_json.unique().tolist()
        if len(cfg_jsons) != 1: raise RuntimeError(f'Config inconsistency: {config_id}')
        out.append({
            'model_family': family, 'task': task, 'config_id': config_id,
            'config_json': cfg_jsons[0], 'seed_count': len(s), 'seeds': json.dumps(SEEDS),
            'mean_validation_task_score': float(np.mean(s)),
            'std_validation_task_score': float(np.std(s, ddof=1)),
            'min_validation_task_score': float(np.min(s)),
            'max_validation_task_score': float(np.max(s)),
            'selection_rule': 'lowest 3-seed mean; tie lower sample std; then config_id',
            'test_used_for_selection': False,
        })
    df = pd.DataFrame(out).sort_values(
        ['mean_validation_task_score', 'std_validation_task_score', 'config_id']
    ).reset_index(drop=True)
    df['rank_within_model_task'] = np.arange(1, len(df) + 1)
    return df

summaries = []
for family in ['SingleTaskTransformer', 'SingleTaskLSTM', 'XGBoost']:
    for task in ['return', 'volatility']:
        summaries.append(summarize_family_task(family, task))
grid_summary = pd.concat(summaries, ignore_index=True)
grid_summary.to_csv(GRID_SUMMARY_CSV, index=False)

selection_rows = []
for family in ['SingleTaskTransformer', 'SingleTaskLSTM', 'XGBoost']:
    for task in ['return', 'volatility']:
        sub = grid_summary[(grid_summary.model_family == family) & (grid_summary.task == task)].sort_values(
            ['mean_validation_task_score', 'std_validation_task_score', 'config_id']
        )
        selection_rows.append(sub.iloc[0].to_dict())
selection = pd.DataFrame(selection_rows)
assert len(selection) == 6
selection.to_csv(SELECTION_CSV, index=False)


def selected(family: str, task: str):
    sub = selection[(selection.model_family == family) & (selection.task == task)]
    if len(sub) != 1: raise RuntimeError(f'Non-unique selection: {family}/{task}')
    row = sub.iloc[0].to_dict(); row['config'] = json.loads(row['config_json']); return row

selection_lock = {
    'locked_at_utc': now_utc(),
    'selection_completed_before_test_load': True,
    'test_arrays_loaded_during_selection': False,
    'test_used_for_selection': False,
    'seeds': SEEDS,
    'selected_configs': {
        f"{r['model_family']}__{r['task']}": {
            'config_id': r['config_id'], 'config': json.loads(r['config_json']),
            'mean_validation_task_score': float(r['mean_validation_task_score']),
            'std_validation_task_score': float(r['std_validation_task_score']),
        } for _, r in selection.iterrows()
    },
    'all_candidates_path': str(GRID_CSV),
    'all_candidate_summary_path': str(GRID_SUMMARY_CSV),
}
dump_json(selection_lock, SELECTION_LOCK_JSON)
print('\n[SELECTION LOCKED BEFORE TEST LOAD] ✅')
print(selection[['model_family', 'task', 'config_id', 'mean_validation_task_score', 'std_validation_task_score']].to_string(index=False))


# ==========================================================
# 11. ONLY NOW LOAD TEST + ALIGNMENT AUDIT
# ==========================================================

X_test = np.load(SEQ_DIR / 'X_test.npy')
y_test = np.load(SEQ_DIR / 'y_test.npy')
y_test_raw = np.load(SEQ_DIR / 'y_test_raw.npy')
STATE['test_arrays_loaded'] = True
assert X_test.shape == (584, 10, 8) and y_test.shape == y_test_raw.shape == (584, 8)

test_rebuilt = y_test.astype(np.float64) * Y_SCALE + Y_MEAN
test_inverse_diff = float(np.max(np.abs(test_rebuilt - y_test_raw.astype(np.float64))))
if test_inverse_diff > 1e-5: raise RuntimeError(f'Test inverse diff too large: {test_inverse_diff}')

final_truth = np.load(FINAL_TEST_DIR / 'final_test_y_true_raw_v4.npy')
final_pred = np.load(FINAL_TEST_DIR / 'pred_final_ensemble_raw_v4.npy')
assert final_truth.shape == final_pred.shape == (584, 8)
truth_diff = float(np.max(np.abs(final_truth.astype(np.float64) - y_test_raw.astype(np.float64))))
if truth_diff != 0.0: raise RuntimeError(f'07 truth != 08B truth: {truth_diff}')

anchor_test = np.load(SEQ_DIR / 'anchor_dates_test.npy', allow_pickle=False)
target_test = np.load(SEQ_DIR / 'target_realization_dates_test.npy', allow_pickle=False)
assert len(anchor_test) == len(target_test) == 584
checks = {
    'test_sample_count': 584,
    'truth_max_diff_07_vs_08B': truth_diff,
    'scaler_test_inverse_max_diff': test_inverse_diff,
    'all_584_target_after_anchor': bool(np.all(target_test > anchor_test)),
    'test_internal_persistence_alignment_583_of_583': bool(np.all(anchor_test[1:] == target_test[:-1])),
    'validation_test_boundary_alignment': bool(val_target_dates[-1] == anchor_test[0]),
    'anchor_monotonic': bool(np.all(anchor_test[1:] > anchor_test[:-1])),
    'target_monotonic': bool(np.all(target_test[1:] > target_test[:-1])),
}
if not all(v for k, v in checks.items() if isinstance(v, bool)):
    raise RuntimeError(f'Date alignment failed: {checks}')
print('[TEST ALIGNMENT] truth/date checks PASS ✅')


# ==========================================================
# 12. SELECTED TEST PREDICTIONS + RAW-SCALE 3-SEED ENSEMBLES
# ==========================================================

X_test_flat = X_test.reshape(584, 80)


def neural_family_predictions(family: str):
    rs, vs = selected(family, 'return'), selected(family, 'volatility')
    out = {}
    for seed in SEEDS:
        pr = load_neural_predict(family, 'return', rs['config_id'], rs['config'], seed, X_test)
        pv = load_neural_predict(family, 'volatility', vs['config_id'], vs['config'], seed, X_test)
        p = np.concatenate([pr, pv], axis=1)
        if p.shape != (584, 8): raise RuntimeError(p.shape)
        out[seed] = p
    return out


def xgb_family_predictions():
    rs, vs = selected('XGBoost', 'return'), selected('XGBoost', 'volatility')
    out = {}
    for seed in SEEDS:
        pr = load_xgb_predict('return', rs['config_id'], rs['config'], seed, X_test_flat)
        pv = load_xgb_predict('volatility', vs['config_id'], vs['config'], seed, X_test_flat)
        p = np.concatenate([pr, pv], axis=1)
        if p.shape != (584, 8): raise RuntimeError(p.shape)
        out[seed] = p
    return out

family_seed = {
    'SingleTaskTransformer': neural_family_predictions('SingleTaskTransformer'),
    'SingleTaskLSTM': neural_family_predictions('SingleTaskLSTM'),
    'XGBoost': xgb_family_predictions(),
}
family_ensemble = {}
for family, seed_preds in family_seed.items():
    assert sorted(seed_preds) == SEEDS
    for seed, p in seed_preds.items():
        np.save(PRED_DIR / f'pred_{family.lower()}_seed{seed}_raw_v4.npy', p)
    ens = np.mean(np.stack([seed_preds[s] for s in SEEDS], axis=0), axis=0)
    exact = (seed_preds[123] + seed_preds[777] + seed_preds[2026]) / 3.0
    if float(np.max(np.abs(ens - exact))) != 0.0: raise RuntimeError(f'Ensemble mismatch: {family}')
    family_ensemble[family] = ens
    np.save(PRED_DIR / f'pred_{family.lower()}_ensemble_raw_v4.npy', ens)
print('[PREDICTIONS] 3 families × 3 seeds + ensembles saved ✅')


# ==========================================================
# 13. METRICS + COMPARISONS
# ==========================================================

def metrics_for(y_true, pred, model_name, prediction_type, seed=None):
    out = []
    for i, a in enumerate(ASSETS):
        out.append({'model': model_name, 'prediction_type': prediction_type, 'seed': seed,
                    'task': 'return', 'asset': a, 'target_index': i,
                    'MAE': mae(y_true[:, i], pred[:, i]),
                    'RMSE': rmse(y_true[:, i], pred[:, i]),
                    'R2': r2(y_true[:, i], pred[:, i]),
                    'PinballLoss_tau_0.5': np.nan})
    for i, a in enumerate(ASSETS):
        c = 4 + i
        out.append({'model': model_name, 'prediction_type': prediction_type, 'seed': seed,
                    'task': 'volatility', 'asset': a, 'target_index': c,
                    'MAE': mae(y_true[:, c], pred[:, c]),
                    'RMSE': rmse(y_true[:, c], pred[:, c]),
                    'R2': r2(y_true[:, c], pred[:, c]),
                    'PinballLoss_tau_0.5': pinball(y_true[:, c], pred[:, c], TAU)})
    return pd.DataFrame(out)

metric_frames = []
for family, seed_preds in family_seed.items():
    for seed in SEEDS:
        metric_frames.append(metrics_for(y_test_raw, seed_preds[seed], family, 'seed', seed))
    metric_frames.append(metrics_for(y_test_raw, family_ensemble[family], family, 'ensemble_primary'))
metric_frames.append(metrics_for(y_test_raw, final_pred, 'FinalWinner_3SeedEnsemble', 'locked_final_reference'))
metrics_df = pd.concat(metric_frames, ignore_index=True)
metrics_df.to_csv(METRICS_CSV, index=False)
STATE['test_metrics_computed'] = True

pred_return_zero = np.load(NAIVE_DIR / 'pred_return_zero_raw_v4.npy')
pred_vol_persist = np.load(NAIVE_DIR / 'pred_vol_persistence_raw_v4.npy')
assert pred_return_zero.shape == pred_vol_persist.shape == (584, 4)

comp = []
for family, p in family_ensemble.items():
    for i, a in enumerate(ASSETS):
        y = y_test_raw[:, i]; fe = mae(y, final_pred[:, i]); le = mae(y, p[:, i]); ne = mae(y, pred_return_zero[:, i])
        comp += [
            {'comparison_type': 'final_vs_learned', 'task': 'return', 'asset': a, 'primary_metric': 'MAE',
             'reference_model': family, 'reference_error': le, 'candidate_model': 'FinalWinner_3SeedEnsemble',
             'candidate_error': fe, 'candidate_to_reference_ratio': fe / le, 'candidate_beats_reference': fe < le},
            {'comparison_type': 'learned_vs_strong_naive', 'task': 'return', 'asset': a, 'primary_metric': 'MAE',
             'reference_model': 'ReturnZero', 'reference_error': ne, 'candidate_model': family,
             'candidate_error': le, 'candidate_to_reference_ratio': le / ne, 'candidate_beats_reference': le < ne},
        ]
    for i, a in enumerate(ASSETS):
        c = 4 + i; y = y_test_raw[:, c]
        fe = pinball(y, final_pred[:, c], TAU); le = pinball(y, p[:, c], TAU); ne = pinball(y, pred_vol_persist[:, i], TAU)
        comp += [
            {'comparison_type': 'final_vs_learned', 'task': 'volatility', 'asset': a,
             'primary_metric': 'PinballLoss_tau_0.5', 'reference_model': family, 'reference_error': le,
             'candidate_model': 'FinalWinner_3SeedEnsemble', 'candidate_error': fe,
             'candidate_to_reference_ratio': fe / le, 'candidate_beats_reference': fe < le},
            {'comparison_type': 'learned_vs_strong_naive', 'task': 'volatility', 'asset': a,
             'primary_metric': 'PinballLoss_tau_0.5', 'reference_model': 'VolPersistence',
             'reference_error': ne, 'candidate_model': family, 'candidate_error': le,
             'candidate_to_reference_ratio': le / ne, 'candidate_beats_reference': le < ne},
        ]
pd.DataFrame(comp).to_csv(COMPARISON_CSV, index=False)


# ==========================================================
# 14. PARAMETER / COMPLEXITY REPORT
# ==========================================================

param_rows = []
transformer_params = count_params(SingleTaskTransformer())
param_rows += [
    {'model_family': 'SingleTaskTransformer', 'task': 'return', 'config_id': TRANSFORMER_CFG[0]['config_id'],
     'neural_parameter_count': transformer_params, 'complexity_note': 'exact final NoSharing branch counterpart'},
    {'model_family': 'SingleTaskTransformer', 'task': 'volatility', 'config_id': TRANSFORMER_CFG[0]['config_id'],
     'neural_parameter_count': transformer_params, 'complexity_note': 'exact final NoSharing branch counterpart'},
    {'model_family': 'FinalNoSharing', 'task': 'total', 'config_id': 'locked_final_nosharing_small_lb10_baseline',
     'neural_parameter_count': 2 * transformer_params,
     'complexity_note': 'two disjoint branch-matched Transformer branches'},
]
for obj in LSTM_CFGS:
    pc = count_params(SingleTaskLSTM(obj['config']['hidden_size'], obj['config']['num_layers']))
    for task in ['return', 'volatility']:
        param_rows.append({'model_family': 'SingleTaskLSTM', 'task': task, 'config_id': obj['config_id'],
                           'neural_parameter_count': pc, 'complexity_note': 'candidate config'})
for task in ['return', 'volatility']:
    s = selected('XGBoost', task)
    g = runs[(runs.model_family == 'XGBoost') & (runs.task == task) & (runs.config_id == s['config_id'])]
    param_rows.append({'model_family': 'XGBoost', 'task': task, 'config_id': s['config_id'],
                       'neural_parameter_count': np.nan,
                       'complexity_note': 'tree ensemble; neural parameter count not directly comparable',
                       'xgb_mean_avg_best_iteration_across_seeds': float(g.avg_best_iteration.mean()),
                       'xgb_mean_avg_boosted_rounds_across_seeds': float(g.avg_boosted_rounds.mean())})
pd.DataFrame(param_rows).to_csv(PARAM_CSV, index=False)


# ==========================================================
# 15. LOSS SERIES FOR 09
# ==========================================================

loss_payload = {}
for family, p in family_ensemble.items():
    for i, a in enumerate(ASSETS):
        loss_payload[f'{family}__return__{a}'] = np.abs(y_test_raw[:, i] - p[:, i]).astype(np.float64)
    for i, a in enumerate(ASSETS):
        c = 4 + i
        loss_payload[f'{family}__volatility__{a}'] = pinball_series(y_test_raw[:, c], p[:, c], TAU).astype(np.float64)
assert len(loss_payload) == 24
for k, v in loss_payload.items():
    if v.shape != (584,) or not np.isfinite(v).all(): raise RuntimeError(f'Bad loss series: {k}')
np.savez_compressed(LOSS_NPZ, **loss_payload)


# ==========================================================
# 16. SUMMARY
# ==========================================================

primary = metrics_df[metrics_df.prediction_type == 'ensemble_primary']
perf = {}
for family in family_ensemble:
    s = primary[primary.model == family]; r = s[s.task == 'return']; v = s[s.task == 'volatility']
    perf[family] = {
        'avg_return_MAE': float(r.MAE.mean()),
        'avg_return_RMSE': float(r.RMSE.mean()),
        'avg_return_R2': float(r.R2.mean()),
        'avg_volatility_MAE': float(v.MAE.mean()),
        'avg_volatility_RMSE': float(v.RMSE.mean()),
        'avg_volatility_R2': float(v.R2.mean()),
        'avg_volatility_PinballLoss_tau_0.5': float(v['PinballLoss_tau_0.5'].mean()),
    }

summary = {
    'project_version': PROJECT_VERSION,
    'stage': '08B_learned_baselines',
    'completed_at_utc': now_utc(),
    'protocol_lock_path': str(PROTOCOL_JSON),
    'selection_lock_path': str(SELECTION_LOCK_JSON),
    'selection_completed_before_test_load': True,
    'validation_only_model_selection': True,
    'test_used_for_hyperparameter_selection': False,
    'test_used_for_checkpoint_selection': False,
    'test_used_for_early_stopping': False,
    'all_candidate_configs_reported': True,
    'final_model_changed': False,
    'final_model_retrained': False,
    'seed_policy': SEEDS,
    'primary_prediction_policy': '3-seed arithmetic ensemble in raw target scale',
    'selected_configs': selection_lock['selected_configs'],
    'alignment_audit': checks,
    'scaler': {'path': str(SCALERS_PATH), 'refit_inside_08B': False,
               'validation_inverse_max_diff': val_inverse_diff,
               'test_inverse_max_diff': test_inverse_diff},
    'xgboost': {
        'version': xgb.__version__, 'return_objective': 'reg:squarederror',
        'volatility_objective': 'reg:quantileerror', 'quantile_alpha': TAU,
        'tree_method': XGB_TREE_METHOD,
        'estimator_level_early_stopping': 'each asset estimator uses validation objective',
        'config_level_selection': '3-seed mean task-level normalized validation score across four assets',
    },
    'capacity': {
        'single_task_transformer_parameter_count': transformer_params,
        'reconstructed_final_nosharing_total_parameter_count': 2 * transformer_params,
        'warning': 'parameter-count differences are reported; performance differences are not attributed solely to task sharing',
    },
    'test_performance_summary': perf,
    'outputs': {
        'grid_results_csv': str(GRID_CSV), 'grid_summary_csv': str(GRID_SUMMARY_CSV),
        'selection_csv': str(SELECTION_CSV), 'metrics_csv': str(METRICS_CSV),
        'comparison_csv': str(COMPARISON_CSV), 'parameter_report_csv': str(PARAM_CSV),
        'loss_series_npz': str(LOSS_NPZ), 'prediction_dir': str(PRED_DIR),
    },
    'interpretation_boundary': (
        '08B is not a shared-representation MTL ablation because final model is NoSharing. '
        'Do not claim task sharing helps/hurts or negative transfer is proven.'
    ),
}
dump_json(summary, SUMMARY_JSON)
save_progress('completed', completed_run_count=done)

print('\n' + '=' * 110)
print('08B — TAMAMLANDI')
print('=' * 110)
print('\nSELECTED CONFIGS')
print(selection[['model_family', 'task', 'config_id', 'mean_validation_task_score', 'std_validation_task_score']].to_string(index=False))
print('\nPRIMARY TEST SUMMARY')
for family, vals in perf.items():
    print('\n' + family)
    for k, v in vals.items(): print(f'  {k}: {v:.12f}')
print('\nRULE CHECK')
print('Validation-only model selection      : True ✅')
print('Test used for hyperparameter tuning  : False ✅')
print('Test used for checkpoint selection   : False ✅')
print('Test used for early stopping         : False ✅')
print('Final model changed                  : False ✅')
print('Final model retrained                : False ✅')
print('Frozen scaler re-fit                 : False ✅')
print('3-seed raw-scale ensemble policy     : True ✅')
print('All candidate configs reported       : True ✅')
print('Estimator early-stop != config select: True ✅')
print('\nSON HÜKÜM: 08B outputs require 08B_FINAL_AUDIT before becoming final thesis findings.')
print('=' * 110)
