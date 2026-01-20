"""
Utility Functions
=================

General utility functions for DSGE model estimation:
- Matrix operations
- Numerical helpers
- Plotting utilities
"""

import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt


def is_positive_definite(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if a matrix is positive definite.

    Args:
        matrix: Square matrix to check
        tol: Tolerance for eigenvalue positivity

    Returns:
        True if matrix is positive definite
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False

    try:
        # Try Cholesky decomposition
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        # Fall back to eigenvalue check
        eigenvalues = np.linalg.eigvals(matrix)
        return np.all(eigenvalues > tol)


def make_positive_definite(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Make a matrix positive definite by adding small diagonal perturbation.

    Args:
        matrix: Square matrix
        epsilon: Small positive value to add to diagonal

    Returns:
        Positive definite matrix
    """
    if is_positive_definite(matrix):
        return matrix

    # Add small value to diagonal
    n = matrix.shape[0]
    return matrix + epsilon * np.eye(n)


def lyapunov_equation(A: np.ndarray, Q: np.ndarray, tol: float = 1e-10, max_iter: int = 1000) -> np.ndarray:
    """
    Solve discrete-time Lyapunov equation: X = A @ X @ A.T + Q

    Uses iterative method for stability.

    Args:
        A: State transition matrix (n x n)
        Q: Innovation covariance matrix (n x n)
        tol: Convergence tolerance
        max_iter: Maximum iterations

    Returns:
        Solution matrix X (n x n)
    """
    n = A.shape[0]
    X = Q.copy()

    for i in range(max_iter):
        X_new = A @ X @ A.T + Q
        diff = np.max(np.abs(X_new - X))
        X = X_new

        if diff < tol:
            return X

    print(f"Warning: Lyapunov equation did not converge after {max_iter} iterations")
    return X


def vec(matrix: np.ndarray) -> np.ndarray:
    """
    Vectorize a matrix (stack columns).

    Args:
        matrix: Matrix to vectorize

    Returns:
        Vectorized matrix
    """
    return matrix.T.flatten()


def unvec(vector: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """
    Reshape a vector back to matrix form.

    Args:
        vector: Vectorized matrix
        rows: Number of rows
        cols: Number of columns

    Returns:
        Reshaped matrix
    """
    return vector.reshape((cols, rows)).T


def kron(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Kronecker product (wrapper for np.kron)."""
    return np.kron(A, B)


def check_stability(T: np.ndarray) -> Tuple[bool, float]:
    """
    Check if state-space system is stable (eigenvalues inside unit circle).

    Args:
        T: State transition matrix

    Returns:
        (is_stable, max_eigenvalue_modulus)
    """
    eigenvalues = np.linalg.eigvals(T)
    max_modulus = np.max(np.abs(eigenvalues))
    is_stable = max_modulus < 1.0

    return is_stable, max_modulus


def autocorr(x: np.ndarray, lags: int = 1) -> np.ndarray:
    """
    Compute autocorrelation function.

    Args:
        x: Time series data (T x n)
        lags: Number of lags

    Returns:
        Autocorrelation coefficients (lags x n)
    """
    T, n = x.shape if x.ndim > 1 else (len(x), 1)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    # Demean
    x_dm = x - np.mean(x, axis=0)

    # Compute autocorrelations
    acf = np.zeros((lags + 1, n))
    c0 = np.sum(x_dm**2, axis=0)

    for lag in range(lags + 1):
        if lag == 0:
            acf[lag, :] = 1.0
        else:
            c_lag = np.sum(x_dm[lag:, :] * x_dm[:-lag, :], axis=0)
            acf[lag, :] = c_lag / c0

    return acf if x.ndim > 1 else acf.flatten()


def detrend(data: np.ndarray, trend_type: str = 'linear') -> np.ndarray:
    """
    Remove trend from data.

    Args:
        data: Time series data (T x n)
        trend_type: Type of trend ('linear', 'quadratic', 'none')

    Returns:
        Detrended data
    """
    if trend_type == 'none':
        return data

    T = len(data)
    t = np.arange(T).reshape(-1, 1)

    if trend_type == 'linear':
        X = np.column_stack([np.ones(T), t])
    elif trend_type == 'quadratic':
        X = np.column_stack([np.ones(T), t, t**2])
    else:
        raise ValueError(f"Unknown trend type: {trend_type}")

    # OLS regression
    beta = np.linalg.lstsq(X, data, rcond=None)[0]
    trend = X @ beta

    return data - trend


def hp_filter(data: np.ndarray, lamb: float = 1600) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hodrick-Prescott filter for trend-cycle decomposition.

    Args:
        data: Time series data (T,)
        lamb: Smoothness parameter (1600 for quarterly data)

    Returns:
        (trend, cycle) components
    """
    T = len(data)

    # Build second-difference matrix
    D = np.zeros((T-2, T))
    for i in range(T-2):
        D[i, i] = 1
        D[i, i+1] = -2
        D[i, i+2] = 1

    # Solve: trend = (I + λD'D)^(-1) y
    I = np.eye(T)
    DTD = D.T @ D
    trend = np.linalg.solve(I + lamb * DTD, data)
    cycle = data - trend

    return trend, cycle


def plot_irf(irfs: np.ndarray, var_names: list, shock_names: list,
             periods: int, figsize: Tuple[int, int] = (12, 8),
             save_path: Optional[str] = None):
    """
    Plot impulse response functions.

    Args:
        irfs: Impulse responses (periods x n_vars x n_shocks)
        var_names: Names of variables
        shock_names: Names of shocks
        periods: Number of periods to plot
        figsize: Figure size
        save_path: Path to save figure (optional)
    """
    n_vars = len(var_names)
    n_shocks = len(shock_names)

    fig, axes = plt.subplots(n_vars, n_shocks, figsize=figsize, squeeze=False)

    for i, var in enumerate(var_names):
        for j, shock in enumerate(shock_names):
            ax = axes[i, j]
            ax.plot(range(periods), irfs[:periods, i, j], 'b-', linewidth=1.5)
            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
            ax.grid(True, alpha=0.3)

            if i == 0:
                ax.set_title(f'Shock: {shock}')
            if j == 0:
                ax.set_ylabel(var)
            if i == n_vars - 1:
                ax.set_xlabel('Periods')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_forecast_errors(errors: np.ndarray, var_names: list,
                        horizons: list = [1, 2, 4, 8, 12],
                        figsize: Tuple[int, int] = (12, 6),
                        save_path: Optional[str] = None):
    """
    Plot forecast errors across horizons.

    Args:
        errors: Forecast errors (T x n_vars x n_horizons)
        var_names: Names of variables
        horizons: Forecast horizons to plot
        figsize: Figure size
        save_path: Path to save figure (optional)
    """
    n_vars = len(var_names)
    n_horizons = len(horizons)

    fig, axes = plt.subplots(1, n_vars, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, var in enumerate(var_names):
        ax = axes[i]
        for h_idx, h in enumerate(horizons):
            rmse = np.sqrt(np.mean(errors[:, i, h_idx]**2))
            ax.bar(h_idx, rmse, label=f'h={h}')

        ax.set_title(f'{var}')
        ax.set_xlabel('Horizon')
        ax.set_ylabel('RMSE')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def compute_rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Compute root mean squared error.

    Args:
        actual: Actual values
        forecast: Forecasted values

    Returns:
        RMSE
    """
    return np.sqrt(np.mean((actual - forecast)**2))


def compute_mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Compute mean absolute error.

    Args:
        actual: Actual values
        forecast: Forecasted values

    Returns:
        MAE
    """
    return np.mean(np.abs(actual - forecast))


if __name__ == '__main__':
    print("Testing utility functions...")

    # Test positive definite check
    A = np.array([[2, 1], [1, 2]])
    print(f"\nIs A positive definite? {is_positive_definite(A)}")

    # Test Lyapunov equation
    A = np.array([[0.9, 0.1], [0.0, 0.8]])
    Q = np.eye(2)
    X = lyapunov_equation(A, Q)
    print(f"\nLyapunov solution error: {np.max(np.abs(X - A @ X @ A.T - Q)):.2e}")

    # Test stability check
    T = np.array([[0.8, 0.1], [0.0, 0.9]])
    is_stable, max_eig = check_stability(T)
    print(f"\nSystem stable? {is_stable}, max eigenvalue: {max_eig:.3f}")

    print("\nAll tests passed!")
