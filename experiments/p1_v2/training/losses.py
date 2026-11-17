"""Only the return training-loss factor changes between the two fixed arms."""
import torch
from torch import nn

def pinball_loss_torch(y_pred,y_true,tau=0.5):
    diff=y_true-y_pred
    return torch.maximum(tau*diff,(tau-1.0)*diff).mean()

def task_losses(y_pred,y_true,return_loss):
    if y_pred.ndim!=2 or y_pred.shape!=y_true.shape or y_pred.shape[1]!=8:
        raise ValueError('Both tensors must have shape[B,8], returns first, volatility last')
    if return_loss=='mse':
        ret=nn.functional.mse_loss(y_pred[:,:4],y_true[:,:4],reduction='mean')
    elif return_loss=='mae':
        ret=nn.functional.l1_loss(y_pred[:,:4],y_true[:,:4],reduction='mean')
    else:raise ValueError('Only mse and mae return arms are supported')
    vol=pinball_loss_torch(y_pred[:,4:],y_true[:,4:],tau=0.5)
    total=0.7*ret+0.3*vol
    return total,ret,vol
