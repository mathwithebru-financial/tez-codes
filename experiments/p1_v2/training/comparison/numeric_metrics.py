"""Pure float64 metrics; no files, torch, fitting, or model execution."""
import numpy as np


def errors_and_losses(truth, prediction):
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError('Finite, matching truth and prediction arrays are required')
    error = truth-prediction
    absolute = np.abs(error)
    return error, absolute, error*error, .5*absolute


def metrics(truth, prediction, *, task):
    if task not in ('return', 'volatility'):
        raise ValueError('Unknown task')
    truth = np.asarray(truth, dtype=np.float64)
    error, absolute, squared, pinball = errors_and_losses(truth, prediction)
    if truth.ndim != 1 or not len(truth):
        raise ValueError('A nonempty one-dimensional population is required')
    mse = float(np.mean(squared, dtype=np.float64))
    result = {'n': len(truth), 'MAE': float(np.mean(absolute, dtype=np.float64)),
              'MSE': mse, 'RMSE': float(np.sqrt(mse)),
              'Pinball_tau05': float(np.mean(pinball, dtype=np.float64)) if task == 'volatility' else None,
              'R2': None, 'R2_status': 'UNDEFINED_INSUFFICIENT_OBSERVATIONS'}
    if len(truth) >= 2:
        centered = truth-np.mean(truth, dtype=np.float64)
        denominator = float(np.sum(centered*centered, dtype=np.float64))
        result['R2_status'] = 'DEFINED' if denominator > 0 else 'UNDEFINED_CONSTANT_TRUTH'
        result['R2'] = 1-float(np.sum(squared, dtype=np.float64))/denominator if denominator > 0 else None
    return result


def raw_seed_ensemble(predictions):
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != 3 or values.shape[2] != 8 or not np.isfinite(values).all():
        raise ValueError('Exactly three finite [N,8] raw prediction arrays are required')
    return np.mean(values, axis=0, dtype=np.float64)


def descriptive_difference(mse_prediction, mae_prediction, truth, *, task):
    """Positive difference means a larger MSE-arm evaluation loss, not a p-value."""
    _, mse_abs, _, mse_pin = errors_and_losses(truth, mse_prediction)
    _, mae_abs, _, mae_pin = errors_and_losses(truth, mae_prediction)
    difference = (mse_abs-mae_abs) if task == 'return' else (mse_pin-mae_pin)
    mse_loss = float(np.mean(mse_abs if task == 'return' else mse_pin, dtype=np.float64))
    mae_loss = float(np.mean(mae_abs if task == 'return' else mae_pin, dtype=np.float64))
    return difference, {'n': len(difference), 'MSE_arm_mean_evaluation_loss': mse_loss,
        'MAE_arm_mean_evaluation_loss': mae_loss,
        'mean_MSE_minus_MAE_loss': float(np.mean(difference, dtype=np.float64)),
        'relative_change_MAE_minus_MSE_percent': 100*(mae_loss-mse_loss)/mse_loss if mse_loss > 0 else None,
        'relative_change_status': 'DEFINED' if mse_loss > 0 else 'UNDEFINED_ZERO_MSE_ARM_EVALUATION_LOSS',
        'fraction_MSE_lower': float(np.mean(difference < 0)),
        'fraction_equal': float(np.mean(difference == 0)),
        'fraction_MAE_lower': float(np.mean(difference > 0)),
        'inferential_test_performed': False}
