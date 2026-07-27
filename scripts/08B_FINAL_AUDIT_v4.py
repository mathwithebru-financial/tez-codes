# ==========================================================
# 08B_FINAL_AUDIT_v4.py
#
# v4_repro — 08B LEARNED BASELINES BAĞIMSIZ SON AUDİTİ
#
# AMAÇ
# ----------------------------------------------------------
# 08B learned-baseline aşamasının resmî çıktılarından bağımsız olarak:
# 1) kod/provenance zincirini,
# 2) 78/78 validation-run bütünlüğünü,
# 3) 3-seed mean config-selection kuralını,
# 4) selected checkpoint/model -> stored seed prediction eşleşmesini,
# 5) 3-seed raw-scale ensemble yeniden üretimini,
# 6) 07-locked final reference ile 08B truth/final-metric eşleşmesini,
# 7) 584 gözlem + raw scale + target/date hizalamasını,
# 8) learned/final/naive metric ve comparison tablolarını,
# 9) loss-series ve kapasite raporunu,
# 10) final-vs-learned görev-ortalaması yüzde farklarını
# bağımsız biçimde doğrular.
#
# SINIR
# ----------------------------------------------------------
# - Bu audit hiçbir modeli yeniden eğitmez.
# - 08B resmî output dosyalarını değiştirmez.
# - Test setini model seçimi için kullanmaz.
# - 08B, shared-representation MTL ablasyonu değildir.
# - Audit PASS, model başarısı/üstünlüğü anlamına gelmez.
# ==========================================================

from __future__ import annotations

import ast
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from torch.utils.data import DataLoader, TensorDataset


# ==========================================================
# 1. PATHS + LOCKS
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

GRID_CSV = OUT_DIR / 'learned_baseline_grid_results_v4.csv'
GRID_SUMMARY_CSV = OUT_DIR / 'learned_baseline_grid_summary_v4.csv'
SELECTION_CSV = OUT_DIR / 'learned_baseline_selection_v4.csv'
SELECTION_LOCK_JSON = OUT_DIR / 'learned_baseline_selection_lock_v4.json'
METRICS_CSV = OUT_DIR / 'learned_baseline_metrics_long_v4.csv'
COMPARISON_CSV = OUT_DIR / 'learned_baseline_comparison_v4.csv'
PARAM_CSV = OUT_DIR / 'learned_baseline_parameter_report_v4.csv'
LOSS_NPZ = OUT_DIR / 'learned_baseline_loss_series_v4.npz'
SUMMARY_JSON = OUT_DIR / 'learned_baseline_summary_v4.json'
PROTOCOL_JSON = OUT_DIR / 'learned_baseline_protocol_lock_v4.json'

AUDIT_DIR = OUT_DIR / 'audit_08B'
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_CHECKS_CSV = AUDIT_DIR / '08B_FINAL_AUDIT_checks_v4.csv'
AUDIT_TASK_COMPARISON_CSV = AUDIT_DIR / '08B_FINAL_AUDIT_task_average_comparison_v4.csv'
AUDIT_RESULT_JSON = AUDIT_DIR / '08B_FINAL_AUDIT_result_v4.json'

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

D_MODEL = 32
N_HEAD = 4
N_LAYERS = 2
D_FF = 128
DROPOUT = 0.10
BATCH_SIZE = 64

EXPECTED_HASHES = {
    '05_grid_search_v4.py': '5d250d9d727cef15e6411cd027aad6089bf62b2cbf4b2c13a8c0f28ff7191a78',
    '06_best_model_multiseed_v4.py': '35de2ee398699003dfef6be36b70c112fb2c0d1b1e9577cbf64bef58877e16d8',
    '07_final_test_evaluation_v4.py': '8b0e3cf2edb9508b4fddd402ddcdbf8c4d2acd6080ffe6fe1876ad818306cd74',
    '08A_naive_baselines_test_v4.py': '95a9658e97f57eaa1a9bb63ec29d8159432f06438bd57ff54fc8ab43013487e8',
    '08B_learned_baselines_test_v4.py': 'df4fa2b29a6b522fc410f693a30520e045c144a15f7c7368af5eb6a1f0f12566',
}

EXPECTED_SELECTIONS = {
    ('SingleTaskTransformer', 'return'): 'transformer_fixed_branchmatched',
    ('SingleTaskTransformer', 'volatility'): 'transformer_fixed_branchmatched',
    ('SingleTaskLSTM', 'return'): 'lstm_h64_l2',
    ('SingleTaskLSTM', 'volatility'): 'lstm_h128_l1',
    ('XGBoost', 'return'): 'xgb_d3_lr0p03',
    ('XGBoost', 'volatility'): 'xgb_d3_lr0p05',
}

# Locked 07 final averages; these are checked against the 07 prediction array,
# not used to create any 08B model selection.
EXPECTED_FINAL_AVG_RETURN_MAE = 0.0075620993156917
EXPECTED_FINAL_AVG_VOL_PINBALL = 0.0050393493147567

# 08B transcript expectations; raw predictions remain the primary audit source.
EXPECTED_08B_PRIMARY = {
    'SingleTaskTransformer': {
        'avg_return_MAE': 0.007252313671,
        'avg_volatility_PinballLoss_tau_0.5': 0.005016331033,
    },
    'SingleTaskLSTM': {
        'avg_return_MAE': 0.006901154128,
        'avg_volatility_PinballLoss_tau_0.5': 0.003490170566,
    },
    'XGBoost': {
        'avg_return_MAE': 0.006955989809,
        'avg_volatility_PinballLoss_tau_0.5': 0.005049587302,
    },
}

NUM_TOL = 1e-10
PRED_REBUILD_TOL = 1e-5
XGB_REBUILD_TOL = 1e-8
SCALER_TOL = 1e-5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================================
# 2. GENERIC HELPERS
# ==========================================================

CHECK_ROWS: List[Dict[str, Any]] = []


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f'Gerekli dosya yok: {path}')


def load_json(path: Path) -> Any:
    require(path)
    return json.loads(path.read_text(encoding='utf-8'))


def json_default(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


def dump_json(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=json_default)
    tmp.replace(path)


def record_check(
    section: str,
    check_name: str,
    passed: bool,
    observed: Any = '',
    expected: Any = '',
    note: str = '',
) -> None:
    row = {
        'section': section,
        'check_name': check_name,
        'passed': bool(passed),
        'observed': str(observed),
        'expected': str(expected),
        'note': note,
    }
    CHECK_ROWS.append(row)
    mark = 'PASS ✅' if passed else 'FAIL ❌'
    print(f'[{section}] {check_name}: {mark}')
    if not passed:
        raise RuntimeError(
            f'Audit check failed: {section} / {check_name}\n'
            f'Observed: {observed}\nExpected: {expected}\n{note}'
        )


def assert_close(
    section: str,
    check_name: str,
    observed: float,
    expected: float,
    tol: float = NUM_TOL,
    note: str = '',
) -> None:
    diff = abs(float(observed) - float(expected))
    record_check(
        section,
        check_name,
        diff <= tol,
        observed=f'{observed:.16g}; diff={diff:.3e}',
        expected=f'{expected:.16g}; tol={tol:.3e}',
        note=note,
    )


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float('inf')
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def r2(y: np.ndarray, p: np.ndarray) -> float:
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return float('nan') if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)


def pinball(y: np.ndarray, p: np.ndarray, tau: float = 0.5) -> float:
    d = y - p
    return float(np.mean(np.maximum(tau * d, (tau - 1.0) * d)))


def pinball_series(y: np.ndarray, p: np.ndarray, tau: float = 0.5) -> np.ndarray:
    d = y - p
    return np.maximum(tau * d, (tau - 1.0) * d)


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {'true', '1', 'yes'}:
            return True
        if v in {'false', '0', 'no', ''}:
            return False
    if pd.isna(value):
        return False
    return bool(value)


# ==========================================================
# 3. START
# ==========================================================

print('=' * 118)
print('08B FINAL AUDIT — BAĞIMSIZ YENİDEN ÜRETİM + HİZALAMA + PROVENANCE')
print('=' * 118)
print('[DEVICE]', DEVICE)


# ==========================================================
# 4. PROVENANCE + MANIFEST
# ==========================================================

for script_name, expected_hash in EXPECTED_HASHES.items():
    path = SCRIPTS_DIR / script_name
    require(path)
    actual = sha256_file(path)
    record_check(
        'A-PROVENANCE',
        f'{script_name} SHA-256',
        actual == expected_hash,
        observed=actual,
        expected=expected_hash,
    )

manifest_path = CONFIG_DIR / 'code_manifest_v4.csv'
require(manifest_path)
manifest = pd.read_csv(manifest_path)
record_check(
    'A-PROVENANCE',
    'Manifest required columns',
    {'script_name', 'sha256'}.issubset(manifest.columns),
    observed=manifest.columns.tolist(),
    expected=['script_name', 'sha256'],
)

for script_name, expected_hash in EXPECTED_HASHES.items():
    mask = (
        manifest['script_name'].astype(str).str.strip().eq(script_name)
        & manifest['sha256'].astype(str).str.strip().eq(expected_hash)
    )
    record_check(
        'A-PROVENANCE',
        f'Manifest exact row: {script_name}',
        bool(mask.any()),
        observed=int(mask.sum()),
        expected='>=1 exact script+hash row',
    )


# ==========================================================
# 5. SOURCE-ORDER + LOCK FILES
# ==========================================================

script_08b = SCRIPTS_DIR / '08B_learned_baselines_test_v4.py'
source = script_08b.read_text(encoding='utf-8')
tree = ast.parse(source, filename=str(script_08b))

protocol_dump_line = None
selection_dump_line = None
x_test_load_line = None
first_training_call_line = None

training_call_lines: List[int] = []

for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id in {'train_neural_run', 'train_xgb_run'}:
                training_call_lines.append(node.lineno)
            if node.func.id == 'dump_json' and len(node.args) >= 2:
                a0, a1 = node.args[0], node.args[1]
                if isinstance(a0, ast.Name) and isinstance(a1, ast.Name):
                    if a0.id == 'protocol' and a1.id == 'PROTOCOL_JSON':
                        protocol_dump_line = node.lineno
                    if a0.id == 'selection_lock' and a1.id == 'SELECTION_LOCK_JSON':
                        selection_dump_line = node.lineno
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id == 'X_test':
                x_test_load_line = node.lineno

first_training_call_line = min(training_call_lines) if training_call_lines else None

record_check(
    'B-SOURCE-ORDER',
    'Protocol lock before first training call',
    protocol_dump_line is not None and first_training_call_line is not None
    and protocol_dump_line < first_training_call_line,
    observed=f'protocol={protocol_dump_line}, first_train={first_training_call_line}',
    expected='protocol_line < first_training_line',
)
record_check(
    'B-SOURCE-ORDER',
    'Selection lock before X_test load',
    selection_dump_line is not None and x_test_load_line is not None
    and selection_dump_line < x_test_load_line,
    observed=f'selection_lock={selection_dump_line}, X_test={x_test_load_line}',
    expected='selection_lock_line < X_test_load_line',
)

protocol = load_json(PROTOCOL_JSON)
selection_lock = load_json(SELECTION_LOCK_JSON)
summary = load_json(SUMMARY_JSON)

record_check(
    'B-LOCKS',
    'Protocol status locked before 08B results',
    protocol.get('status') == 'LOCKED_BEFORE_08B_RESULTS',
    observed=protocol.get('status'),
    expected='LOCKED_BEFORE_08B_RESULTS',
)
record_check(
    'B-LOCKS',
    'Selection completed before test load flag',
    selection_lock.get('selection_completed_before_test_load') is True,
    observed=selection_lock.get('selection_completed_before_test_load'),
    expected=True,
)
record_check(
    'B-LOCKS',
    'Test not used for selection flag',
    selection_lock.get('test_used_for_selection') is False,
    observed=selection_lock.get('test_used_for_selection'),
    expected=False,
)


# ==========================================================
# 6. DATA + TARGET ORDER + FROZEN SCALER
# ==========================================================

sequence_meta_path = SEQ_DIR / 'sequence_meta.json'
sequence_meta = load_json(sequence_meta_path)
record_check(
    'C-DATA',
    'Sequence project version',
    sequence_meta.get('project_version') == PROJECT_VERSION,
    observed=sequence_meta.get('project_version'),
    expected=PROJECT_VERSION,
)
record_check(
    'C-DATA',
    'Feature set',
    sequence_meta.get('feature_set') == FEATURE_SET,
    observed=sequence_meta.get('feature_set'),
    expected=FEATURE_SET,
)
record_check(
    'C-DATA',
    'Lookback',
    int(sequence_meta.get('lookback')) == LOOKBACK,
    observed=sequence_meta.get('lookback'),
    expected=LOOKBACK,
)
record_check(
    'C-DATA',
    'Locked target order',
    sequence_meta.get('target_columns') == TARGET_ORDER,
    observed=sequence_meta.get('target_columns'),
    expected=TARGET_ORDER,
)

X_train = np.load(SEQ_DIR / 'X_train.npy')
X_val = np.load(SEQ_DIR / 'X_val.npy')
X_test = np.load(SEQ_DIR / 'X_test.npy')
y_train = np.load(SEQ_DIR / 'y_train.npy')
y_val = np.load(SEQ_DIR / 'y_val.npy')
y_test = np.load(SEQ_DIR / 'y_test.npy')
y_train_raw = np.load(SEQ_DIR / 'y_train_raw.npy')
y_val_raw = np.load(SEQ_DIR / 'y_val_raw.npy')
y_test_raw = np.load(SEQ_DIR / 'y_test_raw.npy')

expected_shapes = {
    'X_train': (2714, 10, 8), 'X_val': (584, 10, 8), 'X_test': (584, 10, 8),
    'y_train': (2714, 8), 'y_val': (584, 8), 'y_test': (584, 8),
    'y_train_raw': (2714, 8), 'y_val_raw': (584, 8), 'y_test_raw': (584, 8),
}
actual_arrays = {
    'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
    'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
    'y_train_raw': y_train_raw, 'y_val_raw': y_val_raw, 'y_test_raw': y_test_raw,
}
for name, expected_shape in expected_shapes.items():
    arr = actual_arrays[name]
    record_check(
        'C-DATA',
        f'{name} shape',
        arr.shape == expected_shape,
        observed=arr.shape,
        expected=expected_shape,
    )
    record_check(
        'C-DATA',
        f'{name} finite',
        bool(np.isfinite(arr).all()),
        observed=bool(np.isfinite(arr).all()),
        expected=True,
    )

require(SCALERS_PATH)
with SCALERS_PATH.open('rb') as f:
    scalers_obj = pickle.load(f)


def find_target_scaler(obj: Any) -> Any:
    visited = set()

    def walk(x: Any):
        oid = id(x)
        if oid in visited:
            return None
        visited.add(oid)
        if hasattr(x, 'mean_') and hasattr(x, 'scale_'):
            mean_ = np.asarray(getattr(x, 'mean_'))
            scale_ = np.asarray(getattr(x, 'scale_'))
            if mean_.shape == (8,) and scale_.shape == (8,):
                return x
        if isinstance(x, dict):
            for key in ['y_scaler', 'target_scaler', 'targets', 'y']:
                if key in x:
                    found = walk(x[key])
                    if found is not None:
                        return found
            for value in x.values():
                found = walk(value)
                if found is not None:
                    return found
        if isinstance(x, (list, tuple)):
            for value in x:
                found = walk(value)
                if found is not None:
                    return found
        return None

    return walk(obj)


y_scaler = find_target_scaler(scalers_obj)
record_check(
    'C-SCALER',
    '8-dimensional frozen target scaler found',
    y_scaler is not None,
    observed=type(y_scaler).__name__ if y_scaler is not None else None,
    expected='fitted 8-dimensional scaler',
)

Y_MEAN = np.asarray(y_scaler.mean_, dtype=np.float64)
Y_SCALE = np.asarray(y_scaler.scale_, dtype=np.float64)

val_inverse = y_val.astype(np.float64) * Y_SCALE.reshape(1, -1) + Y_MEAN.reshape(1, -1)
test_inverse = y_test.astype(np.float64) * Y_SCALE.reshape(1, -1) + Y_MEAN.reshape(1, -1)
val_inverse_diff = max_abs_diff(val_inverse, y_val_raw)
test_inverse_diff = max_abs_diff(test_inverse, y_test_raw)

record_check(
    'C-SCALER',
    'Validation inverse-transform consistency',
    val_inverse_diff <= SCALER_TOL,
    observed=val_inverse_diff,
    expected=f'<= {SCALER_TOL}',
)
record_check(
    'C-SCALER',
    'Test inverse-transform consistency',
    test_inverse_diff <= SCALER_TOL,
    observed=test_inverse_diff,
    expected=f'<= {SCALER_TOL}',
)


# ==========================================================
# 7. VALIDATION DENOMINATORS + TASK SCORE
# ==========================================================

val_return_zero_mae = {
    a: mae(y_val_raw[:, i], np.zeros(len(y_val_raw)))
    for i, a in enumerate(ASSETS)
}
val_vol_persist_pred = np.empty((len(y_val_raw), 4), dtype=np.float64)
val_vol_persist_pred[0] = y_train_raw[-1, 4:]
val_vol_persist_pred[1:] = y_val_raw[:-1, 4:]
val_vol_persist_pinball = {
    a: pinball(y_val_raw[:, 4 + i], val_vol_persist_pred[:, i], TAU)
    for i, a in enumerate(ASSETS)
}


def task_cols(task: str) -> np.ndarray:
    if task == 'return':
        return np.arange(0, 4)
    if task == 'volatility':
        return np.arange(4, 8)
    raise ValueError(task)


def inverse_task(pred_scaled: np.ndarray, task: str) -> np.ndarray:
    cols = task_cols(task)
    return pred_scaled.astype(np.float64) * Y_SCALE[cols].reshape(1, -1) + Y_MEAN[cols].reshape(1, -1)


def task_score(task: str, y_true_raw_task: np.ndarray, pred_raw_task: np.ndarray) -> Tuple[float, Dict[str, float]]:
    ratios: Dict[str, float] = {}
    if task == 'return':
        for i, a in enumerate(ASSETS):
            ratios[a] = mae(y_true_raw_task[:, i], pred_raw_task[:, i]) / val_return_zero_mae[a]
    elif task == 'volatility':
        for i, a in enumerate(ASSETS):
            ratios[a] = pinball(y_true_raw_task[:, i], pred_raw_task[:, i], TAU) / val_vol_persist_pinball[a]
    else:
        raise ValueError(task)
    return float(np.mean(list(ratios.values()))), ratios


# ==========================================================
# 8. EXPECTED RUN INVENTORY
# ==========================================================

TRANSFORMER_CFG = [{
    'config_id': 'transformer_fixed_branchmatched',
    'config': {'d_model': 32, 'n_head': 4, 'n_layers': 2, 'd_ff': 128, 'dropout': 0.10},
}]
LSTM_CFGS = [
    {'config_id': f'lstm_h{h}_l{l}', 'config': {
        'hidden_size': h,
        'num_layers': l,
        'recurrent_dropout': 0.0 if l == 1 else 0.10,
        'head_dropout': 0.10,
        'bidirectional': False,
        'sequence_representation': 'last_timestep',
    }}
    for h in [32, 64, 128] for l in [1, 2]
]
XGB_CFGS = [
    {'config_id': f"xgb_d{d}_lr{str(lr).replace('.', 'p')}", 'config': {
        'max_depth': d,
        'learning_rate': lr,
        'n_estimators': 1000,
        'early_stopping_rounds': 30,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',
    }}
    for d in [3, 4, 6] for lr in [0.03, 0.05]
]

EXPECTED_RUNS: List[Dict[str, Any]] = []
for obj in TRANSFORMER_CFG:
    for task in ['return', 'volatility']:
        for seed in SEEDS:
            EXPECTED_RUNS.append({'family': 'SingleTaskTransformer', 'task': task, 'seed': seed, **obj})
for obj in LSTM_CFGS:
    for task in ['return', 'volatility']:
        for seed in SEEDS:
            EXPECTED_RUNS.append({'family': 'SingleTaskLSTM', 'task': task, 'seed': seed, **obj})
for obj in XGB_CFGS:
    for task in ['return', 'volatility']:
        for seed in SEEDS:
            EXPECTED_RUNS.append({'family': 'XGBoost', 'task': task, 'seed': seed, **obj})

record_check(
    'D-RUN-INVENTORY',
    'Expected run count',
    len(EXPECTED_RUNS) == 78,
    observed=len(EXPECTED_RUNS),
    expected=78,
)


# ==========================================================
# 9. GRID CSV + RUN-META + NEURAL HISTORY INDEPENDENT CHECK
# ==========================================================

require(GRID_CSV)
grid_stored = pd.read_csv(GRID_CSV)
record_check(
    'D-RUN-INVENTORY',
    'Stored grid row count',
    len(grid_stored) == 78,
    observed=len(grid_stored),
    expected=78,
)
record_check(
    'D-RUN-INVENTORY',
    'Stored grid all success',
    bool((grid_stored['status'] == 'success').all()),
    observed=int((grid_stored['status'] == 'success').sum()),
    expected=78,
)

independent_run_rows: List[Dict[str, Any]] = []
neural_history_score_diffs: List[float] = []
neural_history_epoch_mismatches = 0
xgb_validation_score_diffs: List[float] = []

X_train_flat = X_train.reshape(len(X_train), -1)
X_val_flat = X_val.reshape(len(X_val), -1)
record_check('D-XGB', 'X_train flatten shape', X_train_flat.shape == (2714, 80), X_train_flat.shape, (2714, 80))
record_check('D-XGB', 'X_val flatten shape', X_val_flat.shape == (584, 80), X_val_flat.shape, (584, 80))


def make_xgb(task: str, cfg: Dict[str, Any], seed: int) -> xgb.XGBRegressor:
    common = dict(
        n_estimators=1000,
        max_depth=int(cfg['max_depth']),
        learning_rate=float(cfg['learning_rate']),
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        random_state=int(seed),
        early_stopping_rounds=30,
        n_jobs=-1,
        verbosity=0,
        validate_parameters=True,
        device='cpu',
    )
    if task == 'return':
        return xgb.XGBRegressor(objective='reg:squarederror', **common)
    return xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=TAU, **common)


def xgb_predict_best(model: xgb.XGBRegressor, X: np.ndarray) -> np.ndarray:
    bi = getattr(model, 'best_iteration', None)
    pred = model.predict(X, iteration_range=(0, int(bi) + 1)) if bi is not None else model.predict(X)
    pred = np.asarray(pred).reshape(-1)
    if not np.isfinite(pred).all():
        raise RuntimeError('Non-finite XGBoost prediction')
    return pred


for spec in EXPECTED_RUNS:
    family = spec['family']
    task = spec['task']
    config_id = spec['config_id']
    cfg = spec['config']
    seed = int(spec['seed'])
    run_id = f'{family}__{task}__{config_id}__seed{seed}'
    meta_file = META_DIR / f'{run_id}.json'
    require(meta_file)
    meta = load_json(meta_file)
    record_check(
        'D-RUN-META',
        f'{run_id} status',
        meta.get('status') == 'success',
        observed=meta.get('status'),
        expected='success',
    )
    row = dict(meta['result_row'])
    independent_run_rows.append(row)

    # Compare independent run-meta record to stored 78-row grid table.
    stored_match = grid_stored[
        (grid_stored['model_family'] == family)
        & (grid_stored['task'] == task)
        & (grid_stored['config_id'] == config_id)
        & (grid_stored['seed'].astype(int) == seed)
    ]
    record_check(
        'D-GRID-CROSSCHECK',
        f'{run_id} unique grid row',
        len(stored_match) == 1,
        observed=len(stored_match),
        expected=1,
    )
    stored_row = stored_match.iloc[0]
    assert_close(
        'D-GRID-CROSSCHECK',
        f'{run_id} score grid vs run_meta',
        float(stored_row['validation_task_score']),
        float(row['validation_task_score']),
        tol=NUM_TOL,
    )

    if family in {'SingleTaskTransformer', 'SingleTaskLSTM'}:
        ckpt = CKPT_DIR / f'{run_id}.pt'
        hist = HIST_DIR / f'{run_id}_history.csv'
        require(ckpt)
        require(hist)
        hist_df = pd.read_csv(hist)
        min_score = float(hist_df['validation_task_score'].min())
        best_idx = hist_df['validation_task_score'].astype(float).idxmin()
        best_epoch_from_history = int(hist_df.loc[best_idx, 'epoch'])
        score_diff = abs(min_score - float(row['validation_task_score']))
        neural_history_score_diffs.append(score_diff)
        if best_epoch_from_history != int(row['best_epoch']):
            neural_history_epoch_mismatches += 1
        record_check(
            'D-NEURAL-HISTORY',
            f'{run_id} best score from history',
            score_diff <= NUM_TOL,
            observed=f'{min_score}; diff={score_diff:.3e}',
            expected=float(row['validation_task_score']),
        )
        record_check(
            'D-NEURAL-HISTORY',
            f'{run_id} best epoch from history',
            best_epoch_from_history == int(row['best_epoch']),
            observed=best_epoch_from_history,
            expected=int(row['best_epoch']),
        )
    else:
        model_files = [XGB_DIR / f'{run_id}__{a}.json' for a in ASSETS]
        for p in model_files:
            require(p)
        pred_scaled = np.empty((584, 4), dtype=np.float64)
        for i, model_path in enumerate(model_files):
            model = make_xgb(task, cfg, seed)
            model.load_model(model_path)
            pred_scaled[:, i] = xgb_predict_best(model, X_val_flat)
        pred_raw = inverse_task(pred_scaled, task)
        cols = task_cols(task)
        independent_score, _ = task_score(task, y_val_raw[:, cols], pred_raw)
        score_diff = abs(independent_score - float(row['validation_task_score']))
        xgb_validation_score_diffs.append(score_diff)
        record_check(
            'D-XGB-VAL-REBUILD',
            f'{run_id} validation score rebuilt from saved models',
            score_diff <= XGB_REBUILD_TOL,
            observed=f'{independent_score}; diff={score_diff:.3e}',
            expected=float(row['validation_task_score']),
        )

record_check(
    'D-RUN-INVENTORY',
    'Independent run-meta count',
    len(independent_run_rows) == 78,
    observed=len(independent_run_rows),
    expected=78,
)

independent_runs = pd.DataFrame(independent_run_rows)


# ==========================================================
# 10. INDEPENDENT GRID SUMMARY + CONFIG SELECTION
# ==========================================================

summary_rows: List[Dict[str, Any]] = []
for family in ['SingleTaskTransformer', 'SingleTaskLSTM', 'XGBoost']:
    for task in ['return', 'volatility']:
        sub = independent_runs[
            (independent_runs['model_family'] == family)
            & (independent_runs['task'] == task)
        ].copy()
        for config_id, g in sub.groupby('config_id', sort=True):
            found_seeds = sorted(g['seed'].astype(int).unique().tolist())
            record_check(
                'E-CONFIG-SUMMARY',
                f'{family}/{task}/{config_id} seed set',
                found_seeds == SEEDS,
                observed=found_seeds,
                expected=SEEDS,
            )
            scores = g['validation_task_score'].astype(float).to_numpy()
            summary_rows.append({
                'model_family': family,
                'task': task,
                'config_id': config_id,
                'config_json': g['config_json'].iloc[0],
                'seed_count': len(scores),
                'seeds': json.dumps(SEEDS),
                'mean_validation_task_score': float(np.mean(scores)),
                'std_validation_task_score': float(np.std(scores, ddof=1)),
                'min_validation_task_score': float(np.min(scores)),
                'max_validation_task_score': float(np.max(scores)),
            })

independent_summary = pd.DataFrame(summary_rows)
independent_summary = independent_summary.sort_values(
    ['model_family', 'task', 'mean_validation_task_score', 'std_validation_task_score', 'config_id']
).reset_index(drop=True)
independent_summary['rank_within_model_task'] = independent_summary.groupby(
    ['model_family', 'task'], sort=False
).cumcount() + 1

require(GRID_SUMMARY_CSV)
stored_summary = pd.read_csv(GRID_SUMMARY_CSV)
record_check(
    'E-CONFIG-SUMMARY',
    'Grid-summary row count',
    len(independent_summary) == len(stored_summary),
    observed=len(independent_summary),
    expected=len(stored_summary),
)

max_summary_mean_diff = 0.0
max_summary_std_diff = 0.0
for _, row in independent_summary.iterrows():
    m = stored_summary[
        (stored_summary['model_family'] == row['model_family'])
        & (stored_summary['task'] == row['task'])
        & (stored_summary['config_id'] == row['config_id'])
    ]
    record_check(
        'E-CONFIG-SUMMARY',
        f"Stored summary unique: {row['model_family']}/{row['task']}/{row['config_id']}",
        len(m) == 1,
        observed=len(m),
        expected=1,
    )
    sr = m.iloc[0]
    max_summary_mean_diff = max(
        max_summary_mean_diff,
        abs(float(sr['mean_validation_task_score']) - float(row['mean_validation_task_score']))
    )
    max_summary_std_diff = max(
        max_summary_std_diff,
        abs(float(sr['std_validation_task_score']) - float(row['std_validation_task_score']))
    )

record_check(
    'E-CONFIG-SUMMARY',
    'Max mean-score diff independent vs stored summary',
    max_summary_mean_diff <= NUM_TOL,
    observed=max_summary_mean_diff,
    expected=f'<= {NUM_TOL}',
)
record_check(
    'E-CONFIG-SUMMARY',
    'Max sample-std diff independent vs stored summary',
    max_summary_std_diff <= NUM_TOL,
    observed=max_summary_std_diff,
    expected=f'<= {NUM_TOL}',
)

independent_selection_rows: List[Dict[str, Any]] = []
for family in ['SingleTaskTransformer', 'SingleTaskLSTM', 'XGBoost']:
    for task in ['return', 'volatility']:
        sub = independent_summary[
            (independent_summary['model_family'] == family)
            & (independent_summary['task'] == task)
        ].sort_values(
            ['mean_validation_task_score', 'std_validation_task_score', 'config_id'],
            ascending=[True, True, True],
        )
        winner = sub.iloc[0].to_dict()
        independent_selection_rows.append(winner)
        expected_config = EXPECTED_SELECTIONS[(family, task)]
        record_check(
            'E-SELECTION',
            f'{family}/{task} selected config',
            winner['config_id'] == expected_config,
            observed=winner['config_id'],
            expected=expected_config,
        )

independent_selection = pd.DataFrame(independent_selection_rows)
require(SELECTION_CSV)
stored_selection = pd.read_csv(SELECTION_CSV)
record_check(
    'E-SELECTION',
    'Stored selection row count',
    len(stored_selection) == 6,
    observed=len(stored_selection),
    expected=6,
)

for _, row in independent_selection.iterrows():
    family, task, config_id = row['model_family'], row['task'], row['config_id']
    m = stored_selection[
        (stored_selection['model_family'] == family)
        & (stored_selection['task'] == task)
    ]
    record_check(
        'E-SELECTION',
        f'{family}/{task} stored unique selection',
        len(m) == 1,
        observed=len(m),
        expected=1,
    )
    record_check(
        'E-SELECTION',
        f'{family}/{task} independent == stored config',
        m.iloc[0]['config_id'] == config_id,
        observed=m.iloc[0]['config_id'],
        expected=config_id,
    )
    lock_key = f'{family}__{task}'
    lock_cfg = selection_lock['selected_configs'][lock_key]['config_id']
    record_check(
        'E-SELECTION',
        f'{family}/{task} independent == selection lock config',
        lock_cfg == config_id,
        observed=lock_cfg,
        expected=config_id,
    )


# ==========================================================
# 11. TEST ALIGNMENT + 07 LOCKED FINAL REFERENCE
# ==========================================================

final_truth = np.load(FINAL_TEST_DIR / 'final_test_y_true_raw_v4.npy')
final_pred = np.load(FINAL_TEST_DIR / 'pred_final_ensemble_raw_v4.npy')
record_check('F-TEST', '07 final truth shape', final_truth.shape == (584, 8), final_truth.shape, (584, 8))
record_check('F-TEST', '07 final prediction shape', final_pred.shape == (584, 8), final_pred.shape, (584, 8))
truth_diff = max_abs_diff(final_truth, y_test_raw)
record_check(
    'F-TEST',
    '07 final truth == 08B sequence y_test_raw exactly',
    truth_diff == 0.0,
    observed=truth_diff,
    expected=0.0,
)

anchor_val = np.load(SEQ_DIR / 'anchor_dates_val.npy', allow_pickle=False)
target_val = np.load(SEQ_DIR / 'target_realization_dates_val.npy', allow_pickle=False)
anchor_test = np.load(SEQ_DIR / 'anchor_dates_test.npy', allow_pickle=False)
target_test = np.load(SEQ_DIR / 'target_realization_dates_test.npy', allow_pickle=False)

record_check('F-DATES', 'Test anchor count', len(anchor_test) == 584, len(anchor_test), 584)
record_check('F-DATES', 'Test target realization count', len(target_test) == 584, len(target_test), 584)
record_check(
    'F-DATES',
    '584/584 target realization after anchor',
    bool(np.all(target_test > anchor_test)),
    observed=int(np.sum(target_test > anchor_test)),
    expected=584,
)
record_check(
    'F-DATES',
    '583/583 internal persistence date alignment',
    bool(np.all(anchor_test[1:] == target_test[:-1])),
    observed=int(np.sum(anchor_test[1:] == target_test[:-1])),
    expected=583,
)
record_check(
    'F-DATES',
    'Validation target end == first test anchor',
    bool(target_val[-1] == anchor_test[0]),
    observed=f'{target_val[-1]} vs {anchor_test[0]}',
    expected='equal',
)
record_check(
    'F-DATES',
    'Test anchor dates strictly monotonic',
    bool(np.all(anchor_test[1:] > anchor_test[:-1])),
    observed=bool(np.all(anchor_test[1:] > anchor_test[:-1])),
    expected=True,
)
record_check(
    'F-DATES',
    'Test target-realization dates strictly monotonic',
    bool(np.all(target_test[1:] > target_test[:-1])),
    observed=bool(np.all(target_test[1:] > target_test[:-1])),
    expected=True,
)


# ==========================================================
# 12. MODEL DEFINITIONS FOR SELECTED-CHECKPOINT REBUILD
# ==========================================================

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, n_layers: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def make_head(d: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d, d),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d, 4),
    )


class SingleTaskTransformer(nn.Module):
    def __init__(self, n_features: int = 8, lookback: int = 10):
        super().__init__()
        self.input_projection = nn.Linear(n_features, D_MODEL)
        self.positional_embedding = nn.Parameter(torch.zeros(1, lookback, D_MODEL))
        self.encoder = TransformerBlock(D_MODEL, N_HEAD, D_FF, N_LAYERS, DROPOUT)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = make_head(D_MODEL, DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
            input_size=8,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=rec_dropout,
            batch_first=True,
            bidirectional=False,
        )
        self.head = make_head(hidden_size, DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        return self.head(h[:, -1, :])


def build_neural(family: str, cfg: Dict[str, Any]) -> nn.Module:
    if family == 'SingleTaskTransformer':
        return SingleTaskTransformer()
    if family == 'SingleTaskLSTM':
        return SingleTaskLSTM(int(cfg['hidden_size']), int(cfg['num_layers']))
    raise ValueError(family)


@torch.no_grad()
def predict_neural_scaled(model: nn.Module, X: np.ndarray) -> np.ndarray:
    ds = TensorDataset(torch.from_numpy(X.astype(np.float32, copy=False)))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    out: List[np.ndarray] = []
    model.eval()
    for (xb,) in dl:
        out.append(model(xb.to(DEVICE)).detach().cpu().numpy())
    pred = np.concatenate(out, axis=0)
    if pred.shape != (len(X), 4) or not np.isfinite(pred).all():
        raise RuntimeError(f'Bad neural prediction: {pred.shape}')
    return pred


def selection_record(family: str, task: str) -> Dict[str, Any]:
    m = independent_selection[
        (independent_selection['model_family'] == family)
        & (independent_selection['task'] == task)
    ]
    if len(m) != 1:
        raise RuntimeError(f'Non-unique selection: {family}/{task}')
    row = m.iloc[0].to_dict()
    row['config'] = json.loads(row['config_json'])
    return row


def rebuild_neural_seed_prediction(family: str, seed: int) -> np.ndarray:
    pieces = []
    for task in ['return', 'volatility']:
        sel = selection_record(family, task)
        run_id = f"{family}__{task}__{sel['config_id']}__seed{seed}"
        ckpt_path = CKPT_DIR / f'{run_id}.pt'
        require(ckpt_path)
        payload = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        record_check(
            'G-SELECTED-CHECKPOINT',
            f'{run_id} checkpoint task/config/seed metadata',
            payload.get('task') == task
            and payload.get('config_id') == sel['config_id']
            and int(payload.get('seed')) == int(seed),
            observed=(payload.get('task'), payload.get('config_id'), payload.get('seed')),
            expected=(task, sel['config_id'], seed),
        )
        model = build_neural(family, sel['config'])
        model.load_state_dict(payload['model_state_dict'], strict=True)
        model = model.to(DEVICE)
        pred_scaled = predict_neural_scaled(model, X_test)
        pieces.append(inverse_task(pred_scaled, task))
    return np.concatenate(pieces, axis=1)


def rebuild_xgb_seed_prediction(seed: int) -> np.ndarray:
    pieces = []
    for task in ['return', 'volatility']:
        sel = selection_record('XGBoost', task)
        cfg = sel['config']
        run_id = f"XGBoost__{task}__{sel['config_id']}__seed{seed}"
        pred_scaled = np.empty((584, 4), dtype=np.float64)
        for i, a in enumerate(ASSETS):
            model_path = XGB_DIR / f'{run_id}__{a}.json'
            require(model_path)
            model = make_xgb(task, cfg, seed)
            model.load_model(model_path)
            pred_scaled[:, i] = xgb_predict_best(model, X_test.reshape(584, 80))
        pieces.append(inverse_task(pred_scaled, task))
    return np.concatenate(pieces, axis=1)


# ==========================================================
# 13. STORED SEED PREDICTIONS + SELECTED-CHECKPOINT REBUILD
# ==========================================================

FAMILIES = ['SingleTaskTransformer', 'SingleTaskLSTM', 'XGBoost']
family_seed_preds: Dict[str, Dict[int, np.ndarray]] = {}
family_ensembles: Dict[str, np.ndarray] = {}
selected_rebuild_diffs: Dict[str, Dict[int, float]] = {}
ensemble_rebuild_diffs: Dict[str, float] = {}

for family in FAMILIES:
    seed_map: Dict[int, np.ndarray] = {}
    selected_rebuild_diffs[family] = {}
    for seed in SEEDS:
        seed_path = PRED_DIR / f'pred_{family.lower()}_seed{seed}_raw_v4.npy'
        require(seed_path)
        stored_seed = np.load(seed_path)
        record_check(
            'G-PREDICTIONS',
            f'{family} seed {seed} stored shape',
            stored_seed.shape == (584, 8),
            observed=stored_seed.shape,
            expected=(584, 8),
        )
        record_check(
            'G-PREDICTIONS',
            f'{family} seed {seed} finite',
            bool(np.isfinite(stored_seed).all()),
            observed=bool(np.isfinite(stored_seed).all()),
            expected=True,
        )
        if family in {'SingleTaskTransformer', 'SingleTaskLSTM'}:
            rebuilt = rebuild_neural_seed_prediction(family, seed)
            tol = PRED_REBUILD_TOL
        else:
            rebuilt = rebuild_xgb_seed_prediction(seed)
            tol = XGB_REBUILD_TOL
        diff = max_abs_diff(stored_seed, rebuilt)
        selected_rebuild_diffs[family][seed] = diff
        record_check(
            'G-SELECTED-REBUILD',
            f'{family} seed {seed} selected model -> stored raw prediction',
            diff <= tol,
            observed=diff,
            expected=f'<= {tol}',
            note='No retraining; prediction is rebuilt from the selected saved checkpoint/model only.',
        )
        seed_map[seed] = stored_seed.astype(np.float64)

    ensemble_path = PRED_DIR / f'pred_{family.lower()}_ensemble_raw_v4.npy'
    require(ensemble_path)
    stored_ensemble = np.load(ensemble_path).astype(np.float64)
    exact_ensemble = (
        seed_map[123] + seed_map[777] + seed_map[2026]
    ) / 3.0
    ens_diff = max_abs_diff(stored_ensemble, exact_ensemble)
    ensemble_rebuild_diffs[family] = ens_diff
    record_check(
        'G-ENSEMBLE',
        f'{family} 3-seed raw-scale ensemble exact reconstruction',
        ens_diff == 0.0,
        observed=ens_diff,
        expected=0.0,
    )
    family_seed_preds[family] = seed_map
    family_ensembles[family] = stored_ensemble


# ==========================================================
# 14. INDEPENDENT METRICS REBUILD
# ==========================================================


def metrics_for(y_true: np.ndarray, pred: np.ndarray, model_name: str, prediction_type: str, seed: Any = np.nan) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for i, a in enumerate(ASSETS):
        rows.append({
            'model': model_name,
            'prediction_type': prediction_type,
            'seed': seed,
            'task': 'return',
            'asset': a,
            'target_index': i,
            'MAE': mae(y_true[:, i], pred[:, i]),
            'RMSE': rmse(y_true[:, i], pred[:, i]),
            'R2': r2(y_true[:, i], pred[:, i]),
            'PinballLoss_tau_0.5': np.nan,
        })
    for i, a in enumerate(ASSETS):
        c = 4 + i
        rows.append({
            'model': model_name,
            'prediction_type': prediction_type,
            'seed': seed,
            'task': 'volatility',
            'asset': a,
            'target_index': c,
            'MAE': mae(y_true[:, c], pred[:, c]),
            'RMSE': rmse(y_true[:, c], pred[:, c]),
            'R2': r2(y_true[:, c], pred[:, c]),
            'PinballLoss_tau_0.5': pinball(y_true[:, c], pred[:, c], TAU),
        })
    return pd.DataFrame(rows)


independent_metric_frames: List[pd.DataFrame] = []
for family in FAMILIES:
    for seed in SEEDS:
        independent_metric_frames.append(
            metrics_for(y_test_raw, family_seed_preds[family][seed], family, 'seed', seed)
        )
    independent_metric_frames.append(
        metrics_for(y_test_raw, family_ensembles[family], family, 'ensemble_primary')
    )
independent_metric_frames.append(
    metrics_for(y_test_raw, final_pred, 'FinalWinner_3SeedEnsemble', 'locked_final_reference')
)
independent_metrics = pd.concat(independent_metric_frames, ignore_index=True)

require(METRICS_CSV)
stored_metrics = pd.read_csv(METRICS_CSV)
record_check(
    'H-METRICS',
    'Metric row count',
    len(independent_metrics) == len(stored_metrics) == 104,
    observed=f'independent={len(independent_metrics)}, stored={len(stored_metrics)}',
    expected=104,
)

metric_key_cols = ['model', 'prediction_type', 'task', 'asset', 'target_index']
max_metric_diffs = {'MAE': 0.0, 'RMSE': 0.0, 'R2': 0.0, 'PinballLoss_tau_0.5': 0.0}

for _, ir in independent_metrics.iterrows():
    m = stored_metrics.copy()
    for col in metric_key_cols:
        m = m[m[col].astype(str) == str(ir[col])]
    if ir['prediction_type'] == 'seed':
        m = m[pd.to_numeric(m['seed'], errors='coerce').eq(float(ir['seed']))]
    else:
        # Ensemble/final rows have empty/NaN seed.
        m = m[pd.to_numeric(m['seed'], errors='coerce').isna()]
    record_check(
        'H-METRICS',
        f"Stored metric unique: {ir['model']}/{ir['prediction_type']}/{ir['task']}/{ir['asset']}",
        len(m) == 1,
        observed=len(m),
        expected=1,
    )
    sr = m.iloc[0]
    for col in max_metric_diffs:
        iv = ir[col]
        sv = sr[col]
        if pd.isna(iv) and pd.isna(sv):
            continue
        diff = abs(float(iv) - float(sv))
        max_metric_diffs[col] = max(max_metric_diffs[col], diff)

for col, diff in max_metric_diffs.items():
    record_check(
        'H-METRICS',
        f'Max {col} diff independent vs stored',
        diff <= NUM_TOL,
        observed=diff,
        expected=f'<= {NUM_TOL}',
    )

# Final averages from 07-locked prediction array.
final_metric_rows = metrics_for(
    y_test_raw,
    final_pred,
    'FinalWinner_3SeedEnsemble',
    'locked_final_reference',
)
final_avg_ret_mae = float(final_metric_rows[final_metric_rows['task'] == 'return']['MAE'].mean())
final_avg_vol_pinball = float(
    final_metric_rows[final_metric_rows['task'] == 'volatility']['PinballLoss_tau_0.5'].mean()
)
assert_close(
    'H-FINAL-REFERENCE',
    '07-locked final Avg Return MAE',
    final_avg_ret_mae,
    EXPECTED_FINAL_AVG_RETURN_MAE,
    tol=NUM_TOL,
)
assert_close(
    'H-FINAL-REFERENCE',
    '07-locked final Avg Vol Pinball',
    final_avg_vol_pinball,
    EXPECTED_FINAL_AVG_VOL_PINBALL,
    tol=NUM_TOL,
)

# 08B final-reference rows must reproduce metrics from 07 locked array.
stored_final = stored_metrics[
    (stored_metrics['model'] == 'FinalWinner_3SeedEnsemble')
    & (stored_metrics['prediction_type'] == 'locked_final_reference')
]
record_check(
    'H-FINAL-REFERENCE',
    '08B metrics contains 8 locked-final reference rows',
    len(stored_final) == 8,
    observed=len(stored_final),
    expected=8,
)

# Learned primary averages and transcript expectations.
primary_perf: Dict[str, Dict[str, float]] = {}
for family in FAMILIES:
    m = independent_metrics[
        (independent_metrics['model'] == family)
        & (independent_metrics['prediction_type'] == 'ensemble_primary')
    ]
    ret = m[m['task'] == 'return']
    vol = m[m['task'] == 'volatility']
    perf = {
        'avg_return_MAE': float(ret['MAE'].mean()),
        'avg_return_RMSE': float(ret['RMSE'].mean()),
        'avg_return_R2': float(ret['R2'].mean()),
        'avg_volatility_MAE': float(vol['MAE'].mean()),
        'avg_volatility_RMSE': float(vol['RMSE'].mean()),
        'avg_volatility_R2': float(vol['R2'].mean()),
        'avg_volatility_PinballLoss_tau_0.5': float(vol['PinballLoss_tau_0.5'].mean()),
    }
    primary_perf[family] = perf
    for key, expected in EXPECTED_08B_PRIMARY[family].items():
        assert_close(
            'H-PRIMARY-EXPECTED',
            f'{family} {key}',
            perf[key],
            expected,
            tol=5e-12,
        )

# Summary JSON must match independent primary metrics.
summary_perf = summary['test_performance_summary']
for family in FAMILIES:
    for key, value in primary_perf[family].items():
        assert_close(
            'H-SUMMARY-JSON',
            f'{family} {key}',
            float(summary_perf[family][key]),
            float(value),
            tol=NUM_TOL,
        )


# ==========================================================
# 15. INDEPENDENT COMPARISON CSV REBUILD
# ==========================================================

pred_return_zero = np.load(NAIVE_DIR / 'pred_return_zero_raw_v4.npy')
pred_vol_persist = np.load(NAIVE_DIR / 'pred_vol_persistence_raw_v4.npy')
record_check('I-NAIVE', 'ReturnZero prediction shape', pred_return_zero.shape == (584, 4), pred_return_zero.shape, (584, 4))
record_check('I-NAIVE', 'VolPersistence prediction shape', pred_vol_persist.shape == (584, 4), pred_vol_persist.shape, (584, 4))

comparison_rows: List[Dict[str, Any]] = []
for family, pred in family_ensembles.items():
    for i, a in enumerate(ASSETS):
        y = y_test_raw[:, i]
        fe = mae(y, final_pred[:, i])
        le = mae(y, pred[:, i])
        ne = mae(y, pred_return_zero[:, i])
        comparison_rows += [
            {
                'comparison_type': 'final_vs_learned',
                'task': 'return',
                'asset': a,
                'primary_metric': 'MAE',
                'reference_model': family,
                'reference_error': le,
                'candidate_model': 'FinalWinner_3SeedEnsemble',
                'candidate_error': fe,
                'candidate_to_reference_ratio': fe / le,
                'candidate_beats_reference': fe < le,
            },
            {
                'comparison_type': 'learned_vs_strong_naive',
                'task': 'return',
                'asset': a,
                'primary_metric': 'MAE',
                'reference_model': 'ReturnZero',
                'reference_error': ne,
                'candidate_model': family,
                'candidate_error': le,
                'candidate_to_reference_ratio': le / ne,
                'candidate_beats_reference': le < ne,
            },
        ]
    for i, a in enumerate(ASSETS):
        c = 4 + i
        y = y_test_raw[:, c]
        fe = pinball(y, final_pred[:, c], TAU)
        le = pinball(y, pred[:, c], TAU)
        ne = pinball(y, pred_vol_persist[:, i], TAU)
        comparison_rows += [
            {
                'comparison_type': 'final_vs_learned',
                'task': 'volatility',
                'asset': a,
                'primary_metric': 'PinballLoss_tau_0.5',
                'reference_model': family,
                'reference_error': le,
                'candidate_model': 'FinalWinner_3SeedEnsemble',
                'candidate_error': fe,
                'candidate_to_reference_ratio': fe / le,
                'candidate_beats_reference': fe < le,
            },
            {
                'comparison_type': 'learned_vs_strong_naive',
                'task': 'volatility',
                'asset': a,
                'primary_metric': 'PinballLoss_tau_0.5',
                'reference_model': 'VolPersistence',
                'reference_error': ne,
                'candidate_model': family,
                'candidate_error': le,
                'candidate_to_reference_ratio': le / ne,
                'candidate_beats_reference': le < ne,
            },
        ]

independent_comparison = pd.DataFrame(comparison_rows)
require(COMPARISON_CSV)
stored_comparison = pd.read_csv(COMPARISON_CSV)
record_check(
    'I-COMPARISON',
    'Comparison row count',
    len(independent_comparison) == len(stored_comparison) == 48,
    observed=f'independent={len(independent_comparison)}, stored={len(stored_comparison)}',
    expected=48,
)

comparison_keys = ['comparison_type', 'task', 'asset', 'primary_metric', 'reference_model', 'candidate_model']
max_comparison_numeric_diff = 0.0
comparison_bool_mismatches = 0
for _, ir in independent_comparison.iterrows():
    m = stored_comparison.copy()
    for col in comparison_keys:
        m = m[m[col].astype(str) == str(ir[col])]
    record_check(
        'I-COMPARISON',
        f"Stored comparison unique: {ir['comparison_type']}/{ir['task']}/{ir['asset']}/{ir['reference_model']}",
        len(m) == 1,
        observed=len(m),
        expected=1,
    )
    sr = m.iloc[0]
    for col in ['reference_error', 'candidate_error', 'candidate_to_reference_ratio']:
        max_comparison_numeric_diff = max(
            max_comparison_numeric_diff,
            abs(float(ir[col]) - float(sr[col]))
        )
    if as_bool(ir['candidate_beats_reference']) != as_bool(sr['candidate_beats_reference']):
        comparison_bool_mismatches += 1

record_check(
    'I-COMPARISON',
    'Max numeric diff independent vs stored comparison',
    max_comparison_numeric_diff <= NUM_TOL,
    observed=max_comparison_numeric_diff,
    expected=f'<= {NUM_TOL}',
)
record_check(
    'I-COMPARISON',
    'Boolean comparison mismatches',
    comparison_bool_mismatches == 0,
    observed=comparison_bool_mismatches,
    expected=0,
)


# ==========================================================
# 16. LOSS SERIES FOR 09 REBUILD
# ==========================================================

require(LOSS_NPZ)
stored_loss = np.load(LOSS_NPZ)
expected_loss: Dict[str, np.ndarray] = {}
for family, pred in family_ensembles.items():
    for i, a in enumerate(ASSETS):
        expected_loss[f'{family}__return__{a}'] = np.abs(y_test_raw[:, i] - pred[:, i]).astype(np.float64)
    for i, a in enumerate(ASSETS):
        c = 4 + i
        expected_loss[f'{family}__volatility__{a}'] = pinball_series(
            y_test_raw[:, c], pred[:, c], TAU
        ).astype(np.float64)

record_check(
    'J-LOSS-SERIES',
    'Loss-series key count',
    len(stored_loss.files) == len(expected_loss) == 24,
    observed=f'stored={len(stored_loss.files)}, expected={len(expected_loss)}',
    expected=24,
)
record_check(
    'J-LOSS-SERIES',
    'Loss-series key set',
    set(stored_loss.files) == set(expected_loss),
    observed=sorted(stored_loss.files),
    expected=sorted(expected_loss),
)
max_loss_diff = 0.0
for key, arr in expected_loss.items():
    diff = max_abs_diff(stored_loss[key], arr)
    max_loss_diff = max(max_loss_diff, diff)
record_check(
    'J-LOSS-SERIES',
    'Max loss-series diff',
    max_loss_diff == 0.0,
    observed=max_loss_diff,
    expected=0.0,
)


# ==========================================================
# 17. PARAMETER / CAPACITY REPORT
# ==========================================================

require(PARAM_CSV)
stored_params = pd.read_csv(PARAM_CSV)
transformer_params = count_params(SingleTaskTransformer())
final_nosharing_total = 2 * transformer_params
lstm_return_sel = selection_record('SingleTaskLSTM', 'return')
lstm_vol_sel = selection_record('SingleTaskLSTM', 'volatility')
lstm_return_params = count_params(SingleTaskLSTM(
    int(lstm_return_sel['config']['hidden_size']), int(lstm_return_sel['config']['num_layers'])
))
lstm_vol_params = count_params(SingleTaskLSTM(
    int(lstm_vol_sel['config']['hidden_size']), int(lstm_vol_sel['config']['num_layers'])
))

capacity_rebuilt = {
    'SingleTaskTransformer_return': transformer_params,
    'SingleTaskTransformer_volatility': transformer_params,
    'FinalNoSharing_total_reconstructed': final_nosharing_total,
    'SelectedLSTM_return': lstm_return_params,
    'SelectedLSTM_volatility': lstm_vol_params,
    'SelectedLSTM_total_if_both_models_used': lstm_return_params + lstm_vol_params,
}

# Cross-check key rows in stored parameter report.
def get_param_row(family: str, task: str, config_id: str) -> pd.Series:
    m = stored_params[
        (stored_params['model_family'] == family)
        & (stored_params['task'] == task)
        & (stored_params['config_id'] == config_id)
    ]
    record_check(
        'K-CAPACITY',
        f'Unique parameter row {family}/{task}/{config_id}',
        len(m) == 1,
        observed=len(m),
        expected=1,
    )
    return m.iloc[0]

tr_row = get_param_row('SingleTaskTransformer', 'return', 'transformer_fixed_branchmatched')
assert_close('K-CAPACITY', 'SingleTaskTransformer parameter count', float(tr_row['neural_parameter_count']), transformer_params, tol=0.0)
fn_row = get_param_row('FinalNoSharing', 'total', 'locked_final_nosharing_small_lb10_baseline')
assert_close('K-CAPACITY', 'FinalNoSharing reconstructed total parameter count', float(fn_row['neural_parameter_count']), final_nosharing_total, tol=0.0)
lr_row = get_param_row('SingleTaskLSTM', 'return', lstm_return_sel['config_id'])
assert_close('K-CAPACITY', 'Selected LSTM return parameter count', float(lr_row['neural_parameter_count']), lstm_return_params, tol=0.0)
lv_row = get_param_row('SingleTaskLSTM', 'volatility', lstm_vol_sel['config_id'])
assert_close('K-CAPACITY', 'Selected LSTM volatility parameter count', float(lv_row['neural_parameter_count']), lstm_vol_params, tol=0.0)


# ==========================================================
# 18. TASK-AVERAGE FINAL VS LEARNED PERCENT DIFFERENCES
# ==========================================================

comparison_task_rows: List[Dict[str, Any]] = []
for family in FAMILIES:
    learned_ret = primary_perf[family]['avg_return_MAE']
    learned_vol = primary_perf[family]['avg_volatility_PinballLoss_tau_0.5']

    ret_percent_lower_vs_final = (final_avg_ret_mae - learned_ret) / final_avg_ret_mae * 100.0
    vol_percent_lower_vs_final = (final_avg_vol_pinball - learned_vol) / final_avg_vol_pinball * 100.0

    comparison_task_rows.append({
        'model': family,
        'task': 'return',
        'metric': 'Avg Return MAE',
        'final_error': final_avg_ret_mae,
        'learned_error': learned_ret,
        'percent_lower_error_vs_final': ret_percent_lower_vs_final,
        'interpretation': 'positive means learned baseline lower error; negative means higher error',
    })
    comparison_task_rows.append({
        'model': family,
        'task': 'volatility',
        'metric': 'Avg Vol PinballLoss_tau_0.5',
        'final_error': final_avg_vol_pinball,
        'learned_error': learned_vol,
        'percent_lower_error_vs_final': vol_percent_lower_vs_final,
        'interpretation': 'positive means learned baseline lower error; negative means higher error',
    })

comparison_task_df = pd.DataFrame(comparison_task_rows)
comparison_task_df.to_csv(AUDIT_TASK_COMPARISON_CSV, index=False)

# Explicit correction: 44% is not the LSTM volatility difference.
lstm_vol_percent = float(
    comparison_task_df[
        (comparison_task_df['model'] == 'SingleTaskLSTM')
        & (comparison_task_df['task'] == 'volatility')
    ]['percent_lower_error_vs_final'].iloc[0]
)
record_check(
    'L-PERCENT',
    'SingleTaskLSTM volatility percent-lower-error vs final',
    abs(lstm_vol_percent - 30.74164246205851) <= 1e-8,
    observed=lstm_vol_percent,
    expected='30.74164246205851% (not 44%)',
)


# ==========================================================
# 19. SUMMARY / POLICY FLAGS
# ==========================================================

policy_expectations = {
    'selection_completed_before_test_load': True,
    'validation_only_model_selection': True,
    'test_used_for_hyperparameter_selection': False,
    'test_used_for_checkpoint_selection': False,
    'test_used_for_early_stopping': False,
    'all_candidate_configs_reported': True,
    'final_model_changed': False,
    'final_model_retrained': False,
}
for key, expected in policy_expectations.items():
    record_check(
        'M-POLICY',
        key,
        summary.get(key) is expected,
        observed=summary.get(key),
        expected=expected,
    )
record_check(
    'M-POLICY',
    'Seed policy',
    summary.get('seed_policy') == SEEDS,
    observed=summary.get('seed_policy'),
    expected=SEEDS,
)
record_check(
    'M-POLICY',
    'Interpretation boundary mentions not shared-representation MTL ablation',
    'not a shared-representation MTL ablation' in summary.get('interpretation_boundary', ''),
    observed=summary.get('interpretation_boundary'),
    expected='explicit interpretation boundary',
)


# ==========================================================
# 20. FINAL AUDIT OUTPUTS
# ==========================================================

checks_df = pd.DataFrame(CHECK_ROWS)
checks_df.to_csv(AUDIT_CHECKS_CSV, index=False)

all_pass = bool(checks_df['passed'].all())

result = {
    'project_version': PROJECT_VERSION,
    'stage': '08B_FINAL_AUDIT',
    'completed_at_utc': now_utc(),
    'audit_passed': all_pass,
    'audit_is_model_success_claim': False,
    'audit_scope': (
        'provenance, 78-run validation integrity, 3-seed selection, selected saved-model prediction rebuild, '
        'ensemble reconstruction, 07-locked final reference, 584-row raw-scale/date alignment, metrics, comparisons, '
        'loss-series, capacity, and task-average percentage differences'
    ),
    'official_08B_script_sha256': EXPECTED_HASHES['08B_learned_baselines_test_v4.py'],
    'run_integrity': {
        'expected_runs': 78,
        'independent_run_meta_rows': len(independent_run_rows),
        'stored_grid_rows': len(grid_stored),
        'neural_history_max_score_diff': max(neural_history_score_diffs) if neural_history_score_diffs else None,
        'neural_history_best_epoch_mismatches': neural_history_epoch_mismatches,
        'xgboost_saved_model_validation_max_score_diff': max(xgb_validation_score_diffs) if xgb_validation_score_diffs else None,
    },
    'selection': {
        f'{family}__{task}': EXPECTED_SELECTIONS[(family, task)]
        for family, task in EXPECTED_SELECTIONS
    },
    'test_alignment': {
        'sample_count': EXPECTED_TEST_N,
        'target_count': 8,
        'truth_max_diff_07_vs_sequence': truth_diff,
        'validation_inverse_max_diff': val_inverse_diff,
        'test_inverse_max_diff': test_inverse_diff,
        'all_584_target_realization_after_anchor': bool(np.all(target_test > anchor_test)),
        'internal_persistence_alignment_583_of_583': bool(np.all(anchor_test[1:] == target_test[:-1])),
        'validation_test_boundary_alignment': bool(target_val[-1] == anchor_test[0]),
    },
    'prediction_rebuild': {
        'selected_saved_model_vs_stored_seed_prediction_max_diffs': selected_rebuild_diffs,
        'stored_seed_mean_vs_stored_ensemble_exact_diffs': ensemble_rebuild_diffs,
    },
    'metrics': {
        'final_locked_reference': {
            'avg_return_MAE': final_avg_ret_mae,
            'avg_volatility_PinballLoss_tau_0.5': final_avg_vol_pinball,
        },
        'learned_primary': primary_perf,
        'max_metric_diffs_independent_vs_stored': max_metric_diffs,
    },
    'capacity': capacity_rebuilt,
    'task_average_final_vs_learned': comparison_task_df.to_dict(orient='records'),
    'interpretation_guardrail': (
        '08B compares final NoSharing Transformer with fully separate single-task training and learned baselines. '
        'It is not a controlled causal test of shared representation, does not prove negative transfer, and DM/Holm analysis '
        'is still required before statistical-significance claims.'
    ),
    'outputs': {
        'checks_csv': str(AUDIT_CHECKS_CSV),
        'task_average_comparison_csv': str(AUDIT_TASK_COMPARISON_CSV),
        'audit_result_json': str(AUDIT_RESULT_JSON),
    },
}

dump_json(result, AUDIT_RESULT_JSON)

print('\n' + '=' * 118)
print('08B FINAL AUDIT — SON HÜKÜM')
print('=' * 118)
print(f'Total checks                         : {len(checks_df)}')
print(f'Passed checks                        : {int(checks_df["passed"].sum())}/{len(checks_df)}')
print(f'78/78 validation runs                : {len(independent_run_rows) == 78}')
print(f'07 truth == sequence y_test_raw      : max diff {truth_diff:.3e}')
print('Selected configs                    :')
for (family, task), config_id in EXPECTED_SELECTIONS.items():
    print(f'  {family:24s} {task:10s} -> {config_id}')
print('\nPRIMARY TASK-AVERAGE TEST METRICS')
print(f'  Final NoSharing Avg Return MAE      : {final_avg_ret_mae:.12f}')
print(f'  Final NoSharing Avg Vol Pinball     : {final_avg_vol_pinball:.12f}')
for family in FAMILIES:
    print(f'  {family:24s} Return MAE={primary_perf[family]["avg_return_MAE"]:.12f} | '
          f'Vol Pinball={primary_perf[family]["avg_volatility_PinballLoss_tau_0.5"]:.12f}')
print('\nTASK-AVERAGE FINAL VS LEARNED PERCENT DIFFERENCES')
for _, r in comparison_task_df.iterrows():
    direction = 'lower error' if r['percent_lower_error_vs_final'] >= 0 else 'higher error'
    print(f"  {r['model']:24s} {r['task']:10s}: {abs(r['percent_lower_error_vs_final']):.6f}% {direction} vs final")
print('\nCAPACITY')
for k, v in capacity_rebuilt.items():
    print(f'  {k}: {v}')
print('\nRULE GUARDRAIL')
print('  Audit PASS != model success.')
print('  08B != shared-representation causal ablation.')
print('  Negative transfer is not proven.')
print('  Statistical significance still awaits 09 DM + Harvey + Holm-Bonferroni.')
print('\nOUTPUTS')
print(' ', AUDIT_CHECKS_CSV)
print(' ', AUDIT_TASK_COMPARISON_CSV)
print(' ', AUDIT_RESULT_JSON)
print('\nAUDIT PASSED:', all_pass)
print('=' * 118)

if not all_pass:
    raise RuntimeError('08B FINAL AUDIT FAILED')
