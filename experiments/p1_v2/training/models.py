"""Selected v4 NoSharing definition copied as TEXT; no archived script import.

TransformerBlock, make_head and NoSharing are AST-compared to original06 in
the synthetic verification. Importing this module does not construct a model.
"""
import torch
from torch import nn

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

def build_selected_model():
    return NoSharing(n_features=8,lookback=10,d_model=32,n_head=4,n_layers=2,d_ff=128,dropout=0.10)

def parameter_counts(model):
    groups=model.pcgrad_groups()
    ids=[id(p) for v in groups.values() for p in v]
    if len(ids)!=len(set(ids)) or set(ids)!={id(p) for p in model.parameters()}:
        raise ValueError('Parameter groups overlap or do not cover the model')
    counts={k:sum(p.numel() for p in v) for k,v in groups.items()}
    counts['total']=sum(p.numel() for p in model.parameters())
    if counts!={'shared':0,'return_specific':27268,'vol_specific':27268,'total':54536}:
        raise ValueError('Selected model parameter counts differ: '+str(counts))
    return counts
