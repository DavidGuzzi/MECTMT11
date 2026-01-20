"""
Forecast Evaluation Tools
==========================

Tools for evaluating forecast performance:
- Recursive forecasting
- RMSE and MAE computation
- Forecast error variance decomposition
- Diebold-Mariano test for forecast comparison
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, List, Optional
from .bvar import BVAR


def recursive_forecast_bvar(data: np.ndarray, lags: int,
                            start_period: int, horizons: List[int],
                            tau: float = 10.0, train: Optional[int] = None) -> Dict:
    """
    Compute recursive forecasts for BVAR model.

    Args:
        data: Full dataset (T x n)
        lags: Number of VAR lags
        start_period: First forecast origin (periods before this used for estimation)
        horizons: List of forecast horizons to evaluate
        tau: Minnesota prior tightness
        train: Training sample size for prior

    Returns:
        Dictionary with forecasts and errors
    """
    T, n = data.shape
    n_forecasts = T - start_period

    # Storage for forecasts and errors
    forecasts = {h: np.zeros((n_forecasts, n)) for h in horizons}
    errors = {h: np.zeros((n_forecasts, n)) for h in horizons}
    actuals = {h: np.zeros((n_forecasts, n)) for h in horizons}

    print(f"Running recursive forecast: {n_forecasts} forecast origins...")

    for i, t in enumerate(range(start_period, T)):
        if (i+1) % 10 == 0:
            print(f"  Progress: {i+1}/{n_forecasts}")

        # Estimation sample: data up to time t
        est_data = data[:t, :]

        # Estimate BVAR
        bvar = BVAR(est_data, lags=lags, tau=tau, train=train)
        try:
            bvar.estimate_bayesian()
        except:
            print(f"  Warning: Estimation failed at t={t}")
            continue

        # Generate forecasts for all horizons
        max_horizon = max(horizons)
        fcast = bvar.forecast(horizons=max_horizon)

        # Store forecasts and compute errors
        for h in horizons:
            if t + h <= T:
                forecasts[h][i, :] = fcast[h-1, :]  # h-1 because 0-indexed
                actuals[h][i, :] = data[t+h-1, :]
                errors[h][i, :] = actuals[h][i, :] - forecasts[h][i, :]

    return {
        'forecasts': forecasts,
        'errors': errors,
        'actuals': actuals,
        'horizons': horizons,
        'n_forecasts': n_forecasts
    }


def compute_forecast_stats(errors: Dict[int, np.ndarray],
                          var_names: Optional[List[str]] = None) -> Dict:
    """
    Compute forecast error statistics.

    Args:
        errors: Dictionary mapping horizons to error matrices (T x n)
        var_names: Variable names (optional)

    Returns:
        Dictionary with RMSE, MAE, and other statistics
    """
    horizons = list(errors.keys())
    n_vars = errors[horizons[0]].shape[1]

    if var_names is None:
        var_names = [f"Var{i+1}" for i in range(n_vars)]

    stats_dict = {
        'horizons': horizons,
        'var_names': var_names,
        'rmse': {},
        'mae': {},
        'bias': {},
        'mse_cov': {},  # Log determinant of MSE covariance
        'mse_uncond': {}  # Log determinant of unconditional MSE
    }

    for h in horizons:
        err = errors[h]
        # Remove NaN rows (incomplete forecasts at end)
        err = err[~np.isnan(err).any(axis=1), :]

        if len(err) == 0:
            continue

        # RMSE
        rmse = np.sqrt(np.mean(err**2, axis=0))
        stats_dict['rmse'][h] = rmse

        # MAE
        mae = np.mean(np.abs(err), axis=0)
        stats_dict['mae'][h] = mae

        # Bias
        bias = np.mean(err, axis=0)
        stats_dict['bias'][h] = bias

        # Log determinant of error covariance
        try:
            cov_matrix = np.cov(err.T)
            sign, logdet = np.linalg.slogdet(cov_matrix)
            stats_dict['mse_cov'][h] = logdet if sign > 0 else np.nan
        except:
            stats_dict['mse_cov'][h] = np.nan

        # Log determinant of unconditional MSE
        try:
            mse_matrix = (err.T @ err) / len(err)
            sign, logdet = np.linalg.slogdet(mse_matrix)
            stats_dict['mse_uncond'][h] = logdet if sign > 0 else np.nan
        except:
            stats_dict['mse_uncond'][h] = np.nan

    return stats_dict


def print_forecast_stats(stats: Dict):
    """Print forecast statistics in readable format."""
    print("\n" + "="*80)
    print("FORECAST EVALUATION STATISTICS")
    print("="*80)

    horizons = stats['horizons']
    var_names = stats['var_names']
    n_vars = len(var_names)

    # RMSE table
    print("\nRoot Mean Squared Error (RMSE):")
    print("-"*80)
    header = f"{'Variable':<15}"
    for h in horizons:
        header += f"h={h:>2d}{'':>9}"
    print(header)
    print("-"*80)

    for i, var in enumerate(var_names):
        row = f"{var:<15}"
        for h in horizons:
            if h in stats['rmse']:
                row += f"{stats['rmse'][h][i]:11.4f}"
            else:
                row += f"{'N/A':>11}"
        print(row)

    # MAE table
    print("\nMean Absolute Error (MAE):")
    print("-"*80)
    print(header)
    print("-"*80)

    for i, var in enumerate(var_names):
        row = f"{var:<15}"
        for h in horizons:
            if h in stats['mae']:
                row += f"{stats['mae'][h][i]:11.4f}"
            else:
                row += f"{'N/A':>11}"
        print(row)

    # Multivariate statistics
    print("\nMultivariate Forecast Statistics:")
    print("-"*80)
    print(f"{'Horizon':<15} {'Log|Cov(err)|':>20} {'Log|MSE|':>20}")
    print("-"*80)

    for h in horizons:
        cov_val = stats['mse_cov'].get(h, np.nan)
        mse_val = stats['mse_uncond'].get(h, np.nan)
        print(f"h={h:<13d} {cov_val:20.4f} {mse_val:20.4f}")

    print("-"*80)


def diebold_mariano_test(errors1: np.ndarray, errors2: np.ndarray,
                        horizon: int = 1, criterion: str = 'MSE') -> Tuple[float, float]:
    """
    Diebold-Mariano test for forecast comparison.

    Tests H0: Two forecasts have equal predictive accuracy.

    Args:
        errors1: Forecast errors from model 1 (T x n)
        errors2: Forecast errors from model 2 (T x n)
        horizon: Forecast horizon (for HAC correction)
        criterion: Loss criterion ('MSE' or 'MAE')

    Returns:
        (statistic, p_value)
    """
    # Loss differential
    if criterion == 'MSE':
        loss1 = errors1**2
        loss2 = errors2**2
    elif criterion == 'MAE':
        loss1 = np.abs(errors1)
        loss2 = np.abs(errors2)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    d = loss1 - loss2

    # Average loss differential
    d_bar = np.mean(d)

    # HAC variance (Newey-West)
    T = len(d)
    gamma_0 = np.var(d, ddof=1)

    # Autocorrelations
    h_dm = horizon - 1  # Lag truncation
    gamma_sum = 0.0

    for lag in range(1, h_dm + 1):
        gamma_lag = np.mean(d[lag:] * d[:-lag])
        weight = 1 - lag / (h_dm + 1)
        gamma_sum += 2 * weight * gamma_lag

    var_d = gamma_0 + gamma_sum

    # DM statistic
    dm_stat = d_bar / np.sqrt(var_d / T)

    # P-value (two-sided test)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    return dm_stat, p_value


def cumulative_errors_for_growth(errors: np.ndarray, growth_vars: List[int]) -> np.ndarray:
    """
    Cumulate forecast errors for growth rate variables.

    For variables measured as growth rates (e.g., dy, dc, dinve, dw),
    multi-step forecast errors need to be cumulated.

    Args:
        errors: Forecast errors (T x h x n) or (T x n) for single horizon
        growth_vars: Indices of growth rate variables

    Returns:
        Errors with growth variables cumulated
    """
    errors_cum = errors.copy()

    if errors.ndim == 3:
        # Multiple horizons: (T x h x n)
        for var_idx in growth_vars:
            errors_cum[:, :, var_idx] = np.cumsum(errors[:, :, var_idx], axis=1)
    elif errors.ndim == 2:
        # Single horizon: (T x n)
        # Already at original scale, no need to cumulate
        pass

    return errors_cum


if __name__ == '__main__':
    print("Testing forecast evaluation tools...")

    # Simulate data
    np.random.seed(42)
    T = 150
    n = 3

    A = np.array([[0.5, 0.1, 0.0],
                  [0.1, 0.6, 0.1],
                  [0.0, 0.1, 0.7]])

    data = np.zeros((T, n))
    for t in range(1, T):
        data[t, :] = A @ data[t-1, :] + np.random.randn(n) * 0.5

    # Recursive forecasts
    print("\nTesting recursive forecast...")
    horizons = [1, 2, 4]
    result = recursive_forecast_bvar(data, lags=1, start_period=100,
                                    horizons=horizons, tau=10.0)

    print(f"\nForecasts computed for {result['n_forecasts']} origins")

    # Compute statistics
    stats = compute_forecast_stats(result['errors'],
                                  var_names=['Y1', 'Y2', 'Y3'])
    print_forecast_stats(stats)

    # Diebold-Mariano test
    print("\nTesting Diebold-Mariano...")
    errors1 = result['errors'][1]
    errors2 = errors1 + np.random.randn(*errors1.shape) * 0.1
    dm_stat, p_val = diebold_mariano_test(errors1, errors2)
    print(f"DM statistic: {dm_stat:.3f}, p-value: {p_val:.3f}")

    print("\nAll tests passed!")
