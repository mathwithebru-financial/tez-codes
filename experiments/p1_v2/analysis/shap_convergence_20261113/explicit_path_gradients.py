"""Fixed-model input gradients on explicit, nested Expected Gradients paths.

New follow-up implementation; no edits to the completed V3 estimator. Imports
only NumPy until explicitly called with an already frozen eval model.
"""
import numpy as np


def path_stream(n, seed, skip=0, background_n=64):
    rng = np.random.RandomState(seed)
    for _ in range(skip):
        rng.choice(background_n); rng.uniform()
    slots = np.empty(n, dtype=np.int64)
    alpha = np.empty(n, dtype=np.float64)
    for j in range(n):
        slots[j] = rng.choice(background_n); alpha[j] = rng.uniform()
    return slots, alpha


def explicit_contributions(model, x, background, slots, alpha, torch, batch=1024):
    """Return all F64 draw contributions [S,10,8,8], never parameter grads."""
    if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError('Frozen eval model with parameter gradients disabled required')
    x, bg = np.asarray(x), np.asarray(background)
    if x.shape != (10,8) or bg.shape != (64,10,8) or x.dtype != np.float32 or bg.dtype != np.float32:
        raise ValueError('Expected float32 x[10,8], training background[64,10,8]')
    if slots.dtype != np.int64 or alpha.dtype != np.float64 or slots.shape != alpha.shape or slots.ndim != 1:
        raise ValueError('Explicit int64 slots and float64 alpha vectors required')
    if not (np.isfinite(x).all() and np.isfinite(bg).all() and np.isfinite(alpha).all()):
        raise ValueError('Nonfinite input')
    if np.any(slots < 0) or np.any(slots >= 64) or np.any(alpha < 0) or np.any(alpha >= 1):
        raise ValueError('Invalid interpolation path')
    xg=torch.from_numpy(x).to('cuda'); bgg=torch.from_numpy(bg).to('cuda')
    out=np.empty((len(slots),10,8,8), dtype=np.float64)
    with torch.enable_grad(), torch.autocast(device_type='cuda', enabled=False):
        for start in range(0,len(slots),batch):
            end=min(start+batch,len(slots))
            bb=bgg[torch.from_numpy(slots[start:end]).to('cuda')]
            a=torch.from_numpy(alpha[start:end].astype(np.float32)).to('cuda')
            b=torch.from_numpy((1.-alpha[start:end]).astype(np.float32)).to('cuda')
            xx=(a[:,None,None]*xg+b[:,None,None]*bb).detach().requires_grad_(True)
            delta=(xg-bb).detach().cpu().numpy().astype(np.float64)
            predictions=model(xx)
            if predictions.shape != (end-start,8): raise ValueError('Wrong output axes')
            for j in range(8):
                grad=torch.autograd.grad(predictions[:,j].sum(), xx,
                    retain_graph=j < 7, create_graph=False, allow_unused=False)[0]
                np.multiply(grad.detach().cpu().numpy(),delta,out=out[start:end,:,:,j],dtype=np.float64)
            del xx, predictions, bb, delta, a, b, grad
    if not np.isfinite(out).all() or any(p.grad is not None or p.requires_grad for p in model.parameters()):
        raise ValueError('Gradient/output invariant failed')
    return out
