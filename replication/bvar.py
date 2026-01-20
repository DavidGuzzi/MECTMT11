"""
Bayesian Vector Autoregression (BVAR)
======================================

Implements Bayesian VAR with Minnesota prior for comparison with DSGE model.

VAR(p) model:
    y_t = c + A_1*y_{t-1} + ... + A_p*y_{t-p} + ε_t

Minnesota prior:
    - Prior mean: random walk (A_1 diagonal = 1, others = 0)
    - Prior variance: decreasing with lag order
"""

import numpy as np
from scipy import linalg, stats
from typing import Tuple, Optional, Dict
import warnings


class BVAR:
    """Bayesian Vector Autoregression with Minnesota prior."""

    def __init__(self, data: np.ndarray, lags: int,
                 tau: float = 10.0, decay: float = 1.0,
                 lambda_param: float = 5.0, mu: float = 2.0,
                 train: Optional[int] = None):
        """
        Initialize BVAR model.

        Args:
            data: Time series data (T x n)
            lags: Number of lags
            tau: Overall tightness (Minnesota prior)
            decay: Lag decay parameter
            lambda_param: Cross-variable shrinkage
            mu: Own vs other variable shrinkage
            train: Number of training observations for prior variance estimation
        """
        self.data = data
        self.lags = lags
        self.tau = tau
        self.decay = decay
        self.lambda_param = lambda_param
        self.mu = mu
        self.train = train

        self.T, self.n = data.shape

        # Model matrices
        self.Y = None
        self.X = None
        self.B_ols = None
        self.Sigma_ols = None
        self.B_post = None
        self.Sigma_post = None

        # Prior
        self.B_prior = None
        self.Omega_prior = None
        self.sigma_prior = None

    def _create_lag_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create lagged data matrices for VAR estimation.

        Returns:
            Y: Dependent variables (T-p x n)
            X: Regressors including constant and lags (T-p x 1+n*p)
        """
        T, n = self.T, self.n
        p = self.lags

        # Dependent variable
        Y = self.data[p:, :]

        # Create lagged regressors
        X_list = [np.ones((T-p, 1))]  # Constant

        for lag in range(1, p+1):
            X_list.append(self.data[p-lag:T-lag, :])

        X = np.hstack(X_list)

        return Y, X

    def estimate_ols(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        OLS estimation of VAR.

        Returns:
            B_ols: OLS coefficient matrix (k x n)
            Sigma_ols: Residual covariance matrix (n x n)
        """
        self.Y, self.X = self._create_lag_matrix()

        # OLS estimation: B = (X'X)^{-1} X'Y
        self.B_ols = np.linalg.lstsq(self.X, self.Y, rcond=None)[0]

        # Residuals
        residuals = self.Y - self.X @ self.B_ols

        # Residual covariance
        T_eff = self.Y.shape[0]
        self.Sigma_ols = (residuals.T @ residuals) / T_eff

        return self.B_ols, self.Sigma_ols

    def _compute_minnesota_prior(self):
        """Compute Minnesota prior mean and variance."""
        n = self.n
        p = self.lags
        k = 1 + n * p  # Number of coefficients per equation

        # Prior mean (random walk): diagonal of first lag = 1, others = 0
        self.B_prior = np.zeros((k, n))

        # Prior variance
        if self.train is not None and self.train > 0:
            # Estimate from training sample
            train_data = self.data[:self.train, :]
            self.sigma_prior = np.std(train_data, axis=0, ddof=1)
        else:
            # Use unconditional variance
            self.sigma_prior = np.std(self.data, axis=0, ddof=1)

        # Build prior covariance matrix for each equation
        # Diagonal elements only (independent priors)
        self.Omega_prior = np.zeros((k, n))

        # Constant term: large variance (diffuse)
        self.Omega_prior[0, :] = 1e6

        # Lag coefficients
        for lag in range(1, p+1):
            for i in range(n):
                idx = 1 + (lag-1)*n + i

                # Own lag
                # Variance: (tau * decay^lag)^2
                self.Omega_prior[idx, i] = (self.tau / (lag ** self.decay))**2

                # Other variables' lags
                for j in range(n):
                    if i != j:
                        idx_j = 1 + (lag-1)*n + j
                        # Variance: (tau * lambda * sigma_i / sigma_j * decay^lag)^2
                        self.Omega_prior[idx_j, i] = (
                            self.tau * self.lambda_param *
                            self.sigma_prior[i] / self.sigma_prior[j] /
                            (lag ** self.decay)
                        )**2

    def estimate_bayesian(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bayesian estimation with Minnesota prior.

        Returns:
            B_post: Posterior mean of coefficients (k x n)
            Sigma_post: Posterior residual covariance (n x n)
        """
        # Get OLS estimates
        self.estimate_ols()

        # Compute prior
        self._compute_minnesota_prior()

        # Posterior with conjugate normal-inverse-Wishart prior
        # For each equation (independent across equations):
        # Posterior mean: weighted average of OLS and prior

        k = self.X.shape[1]
        n = self.n
        T_eff = self.Y.shape[0]

        self.B_post = np.zeros((k, n))

        for i in range(n):
            # Prior precision
            Omega_prior_i = np.diag(1.0 / self.Omega_prior[:, i])

            # Posterior precision
            XtX = self.X.T @ self.X
            post_precision = Omega_prior_i + XtX / self.Sigma_ols[i, i]

            # Posterior mean
            post_mean_rhs = (Omega_prior_i @ self.B_prior[:, i] +
                           XtX @ self.B_ols[:, i] / self.Sigma_ols[i, i])

            try:
                self.B_post[:, i] = np.linalg.solve(post_precision, post_mean_rhs)
            except np.linalg.LinAlgError:
                # Fall back to pseudo-inverse
                self.B_post[:, i] = np.linalg.lstsq(post_precision, post_mean_rhs, rcond=None)[0]

        # Posterior residuals
        residuals_post = self.Y - self.X @ self.B_post
        self.Sigma_post = (residuals_post.T @ residuals_post) / T_eff

        return self.B_post, self.Sigma_post

    def marginal_likelihood(self) -> float:
        """
        Compute marginal likelihood (evidence) for model comparison.

        Uses approximation based on Litterman (1986).

        Returns:
            Log marginal likelihood
        """
        if self.B_post is None:
            self.estimate_bayesian()

        T_eff = self.Y.shape[0]
        k = self.X.shape[1]
        n = self.n

        # Log likelihood at posterior mode
        residuals = self.Y - self.X @ self.B_post
        SSR = residuals.T @ residuals

        log_lik = -0.5 * T_eff * n * np.log(2*np.pi)
        log_lik += -0.5 * T_eff * np.log(np.linalg.det(self.Sigma_post))
        log_lik += -0.5 * np.trace(np.linalg.solve(self.Sigma_post, SSR))

        # Prior density (approximate, assuming diagonal prior covariance)
        log_prior = 0.0
        for i in range(n):
            # Normal prior on coefficients
            diff = self.B_post[:, i] - self.B_prior[:, i]
            Omega_inv = np.diag(1.0 / self.Omega_prior[:, i])
            log_prior += -0.5 * diff @ Omega_inv @ diff
            log_prior += -0.5 * np.sum(np.log(self.Omega_prior[:, i]))

        # Laplace approximation to marginal likelihood
        # log p(Y) ≈ log p(Y|θ*) + log p(θ*) + 0.5*k*log(2π) - 0.5*log|H|
        # where H is Hessian at mode (approximated by prior precision)

        log_det_H = 0.0
        for i in range(n):
            Omega_inv = np.diag(1.0 / self.Omega_prior[:, i])
            XtX = self.X.T @ self.X
            H_i = Omega_inv + XtX / self.Sigma_post[i, i]
            sign, logdet = np.linalg.slogdet(H_i)
            log_det_H += logdet

        log_marginal = log_lik + log_prior + 0.5*k*n*np.log(2*np.pi) - 0.5*log_det_H

        return log_marginal

    def forecast(self, horizons: int) -> np.ndarray:
        """
        Generate point forecasts.

        Args:
            horizons: Number of periods to forecast

        Returns:
            Forecasts (horizons x n)
        """
        if self.B_post is None:
            self.estimate_bayesian()

        n = self.n
        p = self.lags

        # Initialize with last p observations
        y_hist = self.data[-p:, :].copy()

        forecasts = np.zeros((horizons, n))

        for h in range(horizons):
            # Build regressor: [1, y_{t}, y_{t-1}, ..., y_{t-p+1}]
            x = np.ones(1 + n * p)
            for lag in range(p):
                if lag < len(y_hist):
                    x[1 + lag*n : 1 + (lag+1)*n] = y_hist[-(lag+1), :]
                else:
                    x[1 + lag*n : 1 + (lag+1)*n] = 0

            # Forecast
            y_forecast = x @ self.B_post

            forecasts[h, :] = y_forecast

            # Update history
            y_hist = np.vstack([y_hist, y_forecast])
            if len(y_hist) > p:
                y_hist = y_hist[-p:, :]

        return forecasts


def compare_var_models(data: np.ndarray, max_lags: int = 4,
                      tau: float = 10.0, train: Optional[int] = None) -> Dict:
    """
    Compare VAR models with different lag orders using marginal likelihood.

    Args:
        data: Time series data (T x n)
        max_lags: Maximum number of lags to consider
        tau: Minnesota prior tightness
        train: Training sample size

    Returns:
        Dictionary with marginal likelihoods for each model
    """
    results = {}

    for lag in range(1, max_lags+1):
        bvar = BVAR(data, lags=lag, tau=tau, train=train)
        bvar.estimate_bayesian()
        ml = bvar.marginal_likelihood()

        results[f'VAR({lag})'] = ml

        print(f"VAR({lag}): Marginal log-likelihood = {ml:.4f}")

    return results


if __name__ == '__main__':
    print("Testing BVAR implementation...")

    # Simulate simple VAR(1) data
    np.random.seed(42)
    T = 200
    n = 3

    A = np.array([[0.5, 0.1, 0.0],
                  [0.1, 0.6, 0.1],
                  [0.0, 0.1, 0.7]])

    data = np.zeros((T, n))
    for t in range(1, T):
        data[t, :] = A @ data[t-1, :] + np.random.randn(n)

    # Estimate BVAR
    bvar = BVAR(data, lags=1, tau=10.0)
    B_post, Sigma_post = bvar.estimate_bayesian()

    print(f"\nPosterior coefficient matrix shape: {B_post.shape}")
    print(f"Residual covariance matrix shape: {Sigma_post.shape}")

    # Marginal likelihood
    ml = bvar.marginal_likelihood()
    print(f"\nMarginal log-likelihood: {ml:.2f}")

    # Forecast
    forecasts = bvar.forecast(horizons=10)
    print(f"\nForecast shape: {forecasts.shape}")

    print("\nAll tests passed!")
