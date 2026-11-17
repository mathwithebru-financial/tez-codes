"""Vectorized SHAP0.52-equivalent Expected Gradients for the fixed eval model.

Import and synthetic verification use NumPy only. Torch is imported lazily by
the execution callable. The caller supplies the reviewed numerical adapter and
gradient backend context; this module never changes either or loads weights.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time
import numpy as np

SHAP052_GRADIENT_SOURCE_SHA256 = 'ffe68905cf1078d97c2bc1c4d70cef655f70a8839ae6d73a0ec853a94f775198'


def common_paths(n, nsamples, rseed, background_n=64):
    """Private MT19937, exactly choice-then-uniform for each sample/draw."""
    if min(n, nsamples, background_n) < 1:
        raise ValueError('Positive sample/draw/background counts required')
    rng = np.random.RandomState(rseed)
    slots = np.empty((n, nsamples), dtype=np.int64)
    alpha = np.empty((n, nsamples), dtype=np.float64)
    for i in range(n):
        for k in range(nsamples):
            slots[i, k] = rng.choice(background_n)
            alpha[i, k] = rng.uniform()
    return slots, alpha


def _inputs(X, background):
    x, bg = np.asarray(X), np.asarray(background)
    if x.dtype != np.float32 or bg.dtype != np.float32:
        raise ValueError('Actual X/background inputs must already be float32')
    if x.ndim != 3 or bg.ndim != 3 or x.shape[1:] != (10, 8) or bg.shape[1:] != (10, 8) or min(len(x), len(bg)) < 1:
        raise ValueError('Expected X[N,10,8] and background[B,10,8]')
    if not np.isfinite(x).all() or not np.isfinite(bg).all():
        raise ValueError('Finite input values required')
    return np.ascontiguousarray(x), np.ascontiguousarray(bg)


def construct_numpy_paths(X, background, slots, alpha):
    """Pure reference: independent F32(t), F32(1-t); delta F32 then F64."""
    x, bg = _inputs(X, background)
    slots, alpha = np.asarray(slots), np.asarray(alpha)
    if slots.ndim != 2 or slots.shape != alpha.shape or slots.shape[0] != len(x) or slots.dtype != np.int64 or alpha.dtype != np.float64:
        raise ValueError('Path shapes/dtypes must be slots int64 and alpha float64 [N,S]')
    if np.any(slots < 0) or np.any(slots >= len(bg)) or not np.isfinite(alpha).all() or np.any(alpha < 0) or np.any(alpha >= 1):
        raise ValueError('Invalid background slot or interpolation alpha')
    selected = bg[slots]
    x_expanded = x[:, None, :, :]
    a = alpha.astype(np.float32)[:, :, None, None]
    b = (1.-alpha).astype(np.float32)[:, :, None, None]
    left = np.multiply(a, x_expanded, dtype=np.float32)
    right = np.multiply(b, selected, dtype=np.float32)
    interpolation = np.add(left, right, dtype=np.float32)
    delta = np.subtract(x_expanded, selected, dtype=np.float32).astype(np.float64)
    return interpolation, delta


def aggregate_contributions(contributions, n, nsamples):
    """Use the library's per-explained-row/output F64 mean and population var."""
    c = np.asarray(contributions)
    if c.dtype != np.float64 or c.shape != (n*nsamples, 10, 8, 8) or not np.isfinite(c).all():
        raise ValueError('Finite float64 contributions [N*S,10,8,8] required')
    phi = np.empty((n, 10, 8, 8), dtype=np.float64)
    diagnostic = np.empty_like(phi)
    for i in range(n):
        for output in range(8):
            # Match SHAP's contiguous [S,lag,feature] reduction independently.
            samples = np.ascontiguousarray(c[i*nsamples:(i+1)*nsamples, :, :, output])
            phi[i, :, :, output] = samples.mean(axis=0, dtype=np.float64)
            diagnostic[i, :, :, output] = samples.var(axis=0, dtype=np.float64, ddof=0)/math.sqrt(nsamples)
    return phi, diagnostic


def vectorized_expected_gradients(model, X, background, *, rseed, nsamples=256,
                                 batch_size=1024, device='cuda', batch_independent=True):
    """Execute input VJPs only. Caller owns GPU authorization/backend/adapter.

    The fixed model must be eval(), float32, parameter-frozen and independent
    across batch rows. Exactly eight separate output VJPs reuse each forward;
    outputs are NEVER summed together. All sample contributions/reductions are
    NumPy float64, matching SHAP's float64 samples_delta allocation.
    """
    if sys.flags.optimize != 0:
        raise ValueError('Run without -O')
    if not batch_independent:
        raise ValueError('This VJP construction requires independent batch rows')
    if batch_size not in (256, 1024) or not isinstance(nsamples, int) or nsamples < 1:
        raise ValueError('Only declared batch sizes256/1024 and positive nsamples are supported')
    x, bg = _inputs(X, background)
    slots, alpha = common_paths(len(x), nsamples, rseed, len(bg))
    # Execution is deliberately lazy: importing this module never imports Torch.
    import torch
    target = torch.device(device)
    if target.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('Requested CUDA device is unavailable')
        if target.index is None:
            target = torch.device('cuda', torch.cuda.current_device())
    if model.training:
        raise ValueError('Model must already be in eval mode')
    parameters = list(model.parameters())
    if any(p.requires_grad or p.grad is not None for p in parameters):
        raise ValueError('All parameter gradients must be disabled and None')
    if any(p.device != target or p.dtype != torch.float32 for p in parameters):
        raise ValueError('The provided model must already be float32 on the requested device')
    started = time.perf_counter()
    n, s, total = len(x), nsamples, len(x)*nsamples
    x_gpu = torch.from_numpy(x).to(target)
    bg_gpu = torch.from_numpy(bg).to(target)
    slots_gpu = torch.from_numpy(slots.reshape(-1)).to(target)
    # Do not derive coefficient b as 1 - float32(a): SHAP rounds separately.
    a_gpu = torch.from_numpy(alpha.reshape(-1).astype(np.float32)).to(target)
    b_gpu = torch.from_numpy((1.-alpha).reshape(-1).astype(np.float32)).to(target)
    sample_rows = torch.from_numpy(np.repeat(np.arange(n, dtype=np.int64), s)).to(target)
    contributions = np.empty((total, 10, 8, 8), dtype=np.float64)
    forward_batches = 0
    with torch.enable_grad(), torch.autocast(device_type=target.type, enabled=False):
        for start in range(0, total, batch_size):
            stop = min(start+batch_size, total)
            xx = x_gpu[sample_rows[start:stop]]
            bb = bg_gpu[slots_gpu[start:stop]]
            left = a_gpu[start:stop, None, None]*xx
            right = b_gpu[start:stop, None, None]*bb
            inputs = (left+right).detach().requires_grad_(True)
            # Subtraction takes place in F32 before the host F64 promotion.
            delta = (xx-bb).detach().cpu().numpy().astype(np.float64)
            outputs = model(inputs)
            if outputs.shape != (stop-start, 8) or outputs.dtype != torch.float32:
                raise ValueError('Expected model output float32[B,8]')
            if not np.isfinite(outputs.detach().cpu().numpy()).all():
                raise ValueError('Nonfinite interpolated model output')
            for output in range(8):
                gradient = torch.autograd.grad(outputs[:, output].sum(), inputs,
                                               retain_graph=output < 7, create_graph=False,
                                               allow_unused=False)[0]
                grad_np = gradient.detach().cpu().numpy()
                if grad_np.dtype != np.float32 or not np.isfinite(grad_np).all():
                    raise ValueError('Nonfinite or unexpected input-gradient dtype')
                # F32 gradient is promoted BEFORE the F64 product, as NumPy does
                # for SHAP's grad(F32)*samples_delta(F64). No GPU F32 averaging.
                np.multiply(grad_np, delta, out=contributions[start:stop, :, :, output], dtype=np.float64)
                del gradient, grad_np
            forward_batches += 1
            del inputs, outputs, xx, bb, left, right, delta
    if any(p.requires_grad or p.grad is not None for p in parameters):
        raise ValueError('Parameter-gradient invariant changed')
    phi, diagnostic = aggregate_contributions(contributions, n, s)
    elapsed = time.perf_counter()-started
    return {'phi_scaled': phi, 'library_variance_diagnostic_scaled': diagnostic,
            'sampling_metadata': {'rseed': int(rseed), 'nsamples': s, 'background_n': len(bg),
                                  'explained_n': n, 'rng': 'NumPy RandomState MT19937 choice-then-uniform',
                                  'background_slots_sha256': hashlib.sha256(slots.tobytes()).hexdigest(),
                                  'alpha_float64_sha256': hashlib.sha256(alpha.tobytes()).hexdigest(),
                                  'coefficient_rule': 'F32(alpha64) and F32(1-alpha64) independently'},
            'execution_metadata': {'batch_size': batch_size, 'forward_batches': forward_batches,
                                   'input_output_vjp_calls': forward_batches*8, 'elapsed_seconds': elapsed,
                                   'host_contribution_bytes': contributions.nbytes,
                                   'batch_independence_required': True, 'parameter_gradients_computed': False,
                                   'optimizer_or_training': False, 'adapter_or_backend_changed_by_engine': False,
                                   'library_variance_diagnostic_is_SE_or_CI': False,
                                   'reference_shap052_gradient_source_sha256': SHAP052_GRADIENT_SOURCE_SHA256}}


def verify_synthetic():
    if sys.flags.optimize != 0:
        raise ValueError('Synthetic assertions require Python without -O')
    n, s, b = 3, 17, 5
    slots, alpha = common_paths(n, s, 42, b)
    global_state = np.random.get_state()
    try:
        np.random.seed(42)
        reference_slots = np.empty_like(slots); reference_alpha = np.empty_like(alpha)
        for i in range(n):
            for k in range(s):
                reference_slots[i, k] = np.random.choice(b)
                reference_alpha[i, k] = np.random.uniform()
    finally:
        np.random.set_state(global_state)
    assert np.array_equal(slots, reference_slots) and np.array_equal(alpha, reference_alpha)
    rng = np.random.RandomState(9)
    x = rng.normal(size=(n, 10, 8)).astype(np.float32)
    bg = rng.normal(size=(b, 10, 8)).astype(np.float32)
    interpolated, delta = construct_numpy_paths(x, bg, slots, alpha)
    for i in range(n):
        for k in range(s):
            aa, zz = np.float32(alpha[i, k]), np.float32(1.-alpha[i, k])
            scalar = np.add(np.multiply(aa, x[i], dtype=np.float32), np.multiply(zz, bg[slots[i, k]], dtype=np.float32), dtype=np.float32)
            assert np.array_equal(interpolated[i, k], scalar)
            assert np.array_equal(delta[i, k], np.subtract(x[i], bg[slots[i, k]], dtype=np.float32).astype(np.float64))
    edge_alpha = np.asarray([[1.-2.**-26]], dtype=np.float64)
    edge, _ = construct_numpy_paths(np.zeros((1, 10, 8), np.float32), np.ones((1, 10, 8), np.float32), np.zeros((1, 1), np.int64), edge_alpha)
    assert np.all(edge == np.float32(2.**-26)) and np.float32(1.)-np.float32(edge_alpha[0, 0]) == 0
    _, large_delta = construct_numpy_paths(np.full((1, 10, 8), 1e8, np.float32), np.ones((1, 10, 8), np.float32), np.zeros((1, 1), np.int64), np.full((1, 1), .5, np.float64))
    assert np.all(large_delta == 1e8) and 1e8-1. != large_delta[0, 0, 0, 0]
    weights = rng.normal(size=(10, 8, 8)).astype(np.float32)
    c = (delta[:, :, :, :, None]*weights.astype(np.float64)[None, None]).reshape(n*s, 10, 8, 8)
    phi, var = aggregate_contributions(c, n, s)
    for i in range(n):
        for output in range(8):
            samples = weights[:, :, output].astype(np.float64)[None]*delta[i]
            assert np.array_equal(phi[i, :, :, output], samples.mean(axis=0))
            assert np.array_equal(var[i, :, :, output], samples.var(axis=0)/np.sqrt(s))
    all_background_delta = x.astype(np.float64)-bg.astype(np.float64).mean(axis=0)
    theoretical = all_background_delta[:, :, :, None]*weights.astype(np.float64)[None]
    assert not np.allclose(phi, theoretical, rtol=1e-6, atol=1e-8), 'Finite MC uses sampled slots, not an exact full-background expectation'
    assert 'torch' not in sys.modules and 'shap' not in sys.modules
    result = {'status': 'PASS_PURE_VECTOR_CONSTRUCTION_AND_AGGREGATION_SYNTHETIC',
              'common_paths_match_SHAP_global_MT19937_draw_order': True,
              'separately_rounded_float32_coefficients_verified': True,
              'delta_subtracted_float32_then_promoted_float64': True,
              'per_sample_output_F64_mean_population_var_match_library': True,
              'finite_MC_sampled_background_not_exact_full_background': True,
              'Torch_SHAP_model_or_GPU_execution': False,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path = Path(__file__).with_name('vectorized_expected_gradients_synthetic_verification.json')
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-synthetic', action='store_true')
    args = parser.parse_args()
    if args.verify_synthetic:
        verify_synthetic()
    else:
        print(json.dumps({'status': 'API_ONLY_NO_EXECUTION', 'callable': 'vectorized_expected_gradients',
                          'execution_requires_caller_authorization': True}))
