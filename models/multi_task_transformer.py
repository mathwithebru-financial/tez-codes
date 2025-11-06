import torch
import torch.nn as nn

class MultiTaskTransformer(nn.Module):
    def __init__(self, d_in=8, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head_return = nn.Linear(d_model, 1)
        self.head_vol = nn.Linear(d_model, 1)

    def forward(self, x):  # x: (B, T, d_in)
        h = self.input_proj(x)
        h = self.encoder(h)           # (B, T, d_model)
        h = h.transpose(1, 2)         # (B, d_model, T)
        z = self.pool(h).squeeze(-1)  # (B, d_model)
        return self.head_return(z), self.head_vol(z)
