"""Typed, fixed P1V2 plans. Importing does not load data or run a model."""
from dataclasses import dataclass,asdict

SEEDS=(123,777,2026)
ASSETS=('BIST100','USDTRY','EURTRY','GOLD')
FEATURE_ORDER=tuple(c for a in ASSETS for c in (a+'_LogRet',a+'_Vol20'))
OUTPUT_ORDER=tuple(a+'_NextRet' for a in ASSETS)+tuple(a+'_NextVol' for a in ASSETS)
PROTOCOL_VERSION='P1_V2_STRICT_INFORMATION_BOUNDARY'

@dataclass(frozen=True)
class ArmConfig:
    return_loss:str
    architecture:str='NoSharing'
    size:str='small'
    lookback:int=10
    input_dim:int=8
    output_dim:int=8
    d_model:int=32
    n_head:int=4
    n_layers:int=2
    d_ff:int=128
    dropout:float=0.10
    return_weight:float=0.7
    volatility_weight:float=0.3
    volatility_pinball_tau:float=0.5
    optimizer:str='AdamW'
    learning_rate:float=1e-3
    weight_decay:float=1e-4
    global_gradient_clip:float=1.0
    batch_size:int=64
    max_epochs:int=100
    min_epochs_before_stop:int=45
    patience:int=15
    scheduler:str='none'
    selection_population:str='ALL_PRIMARY_STRICT_F_K_SELECTION'
    validation_rows:int=583
    prediction_inverse_dtype:str='float64'

    def __post_init__(self):
        if self.return_loss not in ('mse','mae'):raise ValueError('Invalid return loss arm')

def fixed_arms():
    configs=[ArmConfig('mse'),ArmConfig('mae')]
    a,b=(asdict(c) for c in configs)
    differences=[k for k in a if a[k]!=b[k]]
    if differences!=['return_loss']:raise ValueError('Arms differ outside the declared loss factor')
    return configs

def should_stop(epoch,no_improve):
    return epoch>=45 and no_improve>=15

def is_improvement(candidate,best):
    return candidate<best
