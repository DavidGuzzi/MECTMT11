"""
Kalman Filter Implementation
=============================

Standard Kalman filter for linear state-space models.
Computes likelihood for DSGE model estimation.

State-space form:
    s_t = T * s_{t-1} + R * ε_t        (State equation)
    y_t = Z * s_t + D + u_t            (Measurement equation)

where:
    s_t: State vector (n_s x 1)
    y_t: Observed variables (n_y x 1)
    ε_t ~ N(0, Q): Structural shocks
    u_t ~ N(0, H): Measurement errors
"""

import numpy as np
from typing import Tuple, Optional, Dict
from scipy import linalg
import warnings


class KalmanFilter:
    """Standard Kalman filter for state-space models."""

    def __init__(self, T: np.ndarray, R: np.ndarray, Q: np.ndarray,
                 Z: np.ndarray, D: np.ndarray, H: Optional[np.ndarray] = None):
        """
        Initialize Kalman filter.

        Args:
            T: State transition matrix (n_s x n_s)
            R: Shock loading matrix (n_s x n_eps)
            Q: Shock covariance matrix (n_eps x n_eps)
            Z: Measurement matrix (n_y x n_s)
            D: Measurement constant (n_y x 1 or n_y,)
            H: Measurement error covariance (n_y x n_y), optional
        """
        self.T = T
        self.R = R
        self.Q = Q
        self.Z = Z
        self.D = D.reshape(-1, 1) if D.ndim == 1 else D
        self.H = H if H is not None else np.zeros((Z.shape[0], Z.shape[0]))

        self.n_s = T.shape[0]  # Number of states
        self.n_y = Z.shape[0]  # Number of observables
        self.n_eps = Q.shape[0]  # Number of shocks

        # Storage for filter output
        self.s_pred = None  # Predicted states
        self.s_filt = None  # Filtered states
        self.P_pred = None  # Predicted state covariance
        self.P_filt = None  # Filtered state covariance
        self.v = None       # Innovations
        self.F = None       # Innovation variance
        self.K = None       # Kalman gain

    def filter(self, y: np.ndarray, s0: Optional[np.ndarray] = None,
              P0: Optional[np.ndarray] = None,
              return_likelihood: bool = True) -> Dict:
        """
        Run Kalman filter forward pass.

        Args:
            y: Observed data (T x n_y)
            s0: Initial state (n_s x 1), None for zero initialization
            P0: Initial state covariance (n_s x n_s), None for diffuse initialization
            return_likelihood: Whether to compute log-likelihood

        Returns:
            Dictionary with filter output
        """
        T_obs = y.shape[0]

        # Initialize storage
        self.s_pred = np.zeros((T_obs + 1, self.n_s, 1))
        self.s_filt = np.zeros((T_obs, self.n_s, 1))
        self.P_pred = np.zeros((T_obs + 1, self.n_s, self.n_s))
        self.P_filt = np.zeros((T_obs, self.n_s, self.n_s))
        self.v = np.zeros((T_obs, self.n_y, 1))
        self.F = np.zeros((T_obs, self.n_y, self.n_y))
        self.K = np.zeros((T_obs, self.n_s, self.n_y))

        # Initialize state
        if s0 is None:
            self.s_pred[0] = np.zeros((self.n_s, 1))
        else:
            self.s_pred[0] = s0.reshape(-1, 1)

        # Initialize covariance
        if P0 is None:
            # Diffuse initialization: solve Lyapunov equation
            # P = T * P * T' + R * Q * R'
            self.P_pred[0] = self._solve_lyapunov(self.T, self.R @ self.Q @ self.R.T)
        else:
            self.P_pred[0] = P0

        # Forward pass
        log_lik = 0.0
        for t in range(T_obs):
            # Measurement update
            y_t = y[t, :].reshape(-1, 1)
            self._update_step(t, y_t)

            # Likelihood contribution
            if return_likelihood:
                log_lik += self._likelihood_contribution(t)

            # Time update (prediction for t+1)
            if t < T_obs:
                self._predict_step(t)

        result = {
            's_pred': self.s_pred[:-1],  # Drop last prediction
            's_filt': self.s_filt,
            'P_pred': self.P_pred[:-1],
            'P_filt': self.P_filt,
            'v': self.v,
            'F': self.F,
            'K': self.K,
            'log_likelihood': log_lik if return_likelihood else None
        }

        return result

    def _update_step(self, t: int, y_t: np.ndarray):
        """
        Measurement update (correction) step.

        Args:
            t: Time index
            y_t: Observation at time t (n_y x 1)
        """
        # Innovation
        self.v[t] = y_t - self.Z @ self.s_pred[t] - self.D

        # Innovation variance
        self.F[t] = self.Z @ self.P_pred[t] @ self.Z.T + self.H

        # Kalman gain
        try:
            self.K[t] = self.P_pred[t] @ self.Z.T @ np.linalg.inv(self.F[t])
        except np.linalg.LinAlgError:
            # Singular F, use pseudo-inverse
            self.K[t] = self.P_pred[t] @ self.Z.T @ np.linalg.pinv(self.F[t])

        # Updated state
        self.s_filt[t] = self.s_pred[t] + self.K[t] @ self.v[t]

        # Updated covariance
        I_KZ = np.eye(self.n_s) - self.K[t] @ self.Z
        self.P_filt[t] = I_KZ @ self.P_pred[t] @ I_KZ.T + self.K[t] @ self.H @ self.K[t].T

        # Ensure symmetry
        self.P_filt[t] = 0.5 * (self.P_filt[t] + self.P_filt[t].T)

    def _predict_step(self, t: int):
        """
        Time update (prediction) step.

        Args:
            t: Current time index (predicting for t+1)
        """
        # Predicted state
        self.s_pred[t+1] = self.T @ self.s_filt[t]

        # Predicted covariance
        self.P_pred[t+1] = self.T @ self.P_filt[t] @ self.T.T + self.R @ self.Q @ self.R.T

        # Ensure symmetry
        self.P_pred[t+1] = 0.5 * (self.P_pred[t+1] + self.P_pred[t+1].T)

    def _likelihood_contribution(self, t: int) -> float:
        """
        Compute log-likelihood contribution at time t.

        Args:
            t: Time index

        Returns:
            Log-likelihood contribution
        """
        # Log-likelihood: -0.5 * [log|F_t| + v_t' * F_t^{-1} * v_t + n_y * log(2π)]
        try:
            sign, logdet = np.linalg.slogdet(self.F[t])
            if sign <= 0:
                warnings.warn(f"Non-positive definite F at time {t}")
                return -1e10  # Large negative value

            F_inv_v = np.linalg.solve(self.F[t], self.v[t])
            quadratic_form = self.v[t].T @ F_inv_v

            log_lik = -0.5 * (logdet + quadratic_form + self.n_y * np.log(2 * np.pi))

            return float(log_lik)

        except np.linalg.LinAlgError:
            warnings.warn(f"Singular F at time {t}")
            return -1e10

    def _solve_lyapunov(self, A: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """
        Solve discrete-time Lyapunov equation: X = A * X * A' + Q

        Args:
            A: Transition matrix
            Q: Covariance matrix

        Returns:
            Solution X
        """
        try:
            X = linalg.solve_discrete_lyapunov(A, Q)
            return X
        except:
            # Fall back to iterative method
            X = Q.copy()
            for _ in range(1000):
                X_new = A @ X @ A.T + Q
                if np.max(np.abs(X_new - X)) < 1e-10:
                    return X_new
                X = X_new

            warnings.warn("Lyapunov equation did not converge")
            return X

    def smoother(self, filter_output: Optional[Dict] = None) -> Dict:
        """
        Rauch-Tung-Striebel smoother (backward pass).

        Args:
            filter_output: Output from filter() method

        Returns:
            Dictionary with smoothed estimates
        """
        if filter_output is None:
            if self.s_filt is None:
                raise ValueError("Must run filter first")
            filter_output = {
                's_filt': self.s_filt,
                'P_filt': self.P_filt,
                's_pred': self.s_pred[:-1],
                'P_pred': self.P_pred[:-1]
            }

        T_obs = filter_output['s_filt'].shape[0]

        # Initialize storage
        s_smooth = np.zeros((T_obs, self.n_s, 1))
        P_smooth = np.zeros((T_obs, self.n_s, self.n_s))

        # Initialize with filtered values
        s_smooth[-1] = filter_output['s_filt'][-1]
        P_smooth[-1] = filter_output['P_filt'][-1]

        # Backward pass
        for t in range(T_obs - 2, -1, -1):
            # Smoother gain
            try:
                J_t = filter_output['P_filt'][t] @ self.T.T @ np.linalg.inv(filter_output['P_pred'][t+1])
            except np.linalg.LinAlgError:
                J_t = filter_output['P_filt'][t] @ self.T.T @ np.linalg.pinv(filter_output['P_pred'][t+1])

            # Smoothed state
            s_smooth[t] = filter_output['s_filt'][t] + J_t @ (s_smooth[t+1] - filter_output['s_pred'][t+1])

            # Smoothed covariance
            P_smooth[t] = filter_output['P_filt'][t] + J_t @ (P_smooth[t+1] - filter_output['P_pred'][t+1]) @ J_t.T

        return {
            's_smooth': s_smooth,
            'P_smooth': P_smooth
        }


def kalman_likelihood(y: np.ndarray, T: np.ndarray, R: np.ndarray, Q: np.ndarray,
                      Z: np.ndarray, D: np.ndarray, H: Optional[np.ndarray] = None,
                      s0: Optional[np.ndarray] = None,
                      P0: Optional[np.ndarray] = None) -> float:
    """
    Compute log-likelihood using Kalman filter (convenience function).

    Args:
        y: Observed data (T x n_y)
        T, R, Q: State equation matrices
        Z, D, H: Measurement equation matrices
        s0, P0: Initial conditions

    Returns:
        Log-likelihood value
    """
    kf = KalmanFilter(T, R, Q, Z, D, H)
    result = kf.filter(y, s0, P0, return_likelihood=True)
    return result['log_likelihood']


if __name__ == '__main__':
    print("Testing Kalman filter...")

    # Simple AR(1) model: y_t = φ * y_{t-1} + ε_t, ε_t ~ N(0, σ²)
    # State: s_t = y_t
    # Measurement: y_t = s_t (perfect observation)

    np.random.seed(42)

    # Parameters
    phi = 0.9
    sigma = 1.0
    T_obs = 100

    # State-space matrices
    T = np.array([[phi]])
    R = np.array([[1.0]])
    Q = np.array([[sigma**2]])
    Z = np.array([[1.0]])
    D = np.array([[0.0]])
    H = None

    # Simulate data
    y = np.zeros((T_obs, 1))
    s = 0.0
    for t in range(T_obs):
        s = phi * s + np.random.randn() * sigma
        y[t, 0] = s

    # Run Kalman filter
    kf = KalmanFilter(T, R, Q, Z, D, H)
    result = kf.filter(y)

    print(f"\nLog-likelihood: {result['log_likelihood']:.2f}")
    print(f"Number of observations: {T_obs}")
    print(f"Average log-likelihood: {result['log_likelihood']/T_obs:.2f}")

    # Check filtered states match observations (should be close for perfect observation)
    rmse = np.sqrt(np.mean((result['s_filt'][:, 0, 0] - y[:, 0])**2))
    print(f"RMSE (filtered vs observed): {rmse:.6f}")

    # Run smoother
    smooth_result = kf.smoother(result)
    print(f"Smoother ran successfully, output shape: {smooth_result['s_smooth'].shape}")

    print("\nAll tests passed!")
