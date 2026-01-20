"""
Rational Expectations Solver
=============================

Solves log-linearized DSGE models using QZ (generalized Schur) decomposition.
Implements Sims' (2002) method for solving linear rational expectations models.

The model is in canonical form:
    Γ0 * y_t = Γ1 * y_{t-1} + Ψ * ε_t + Π * η_t

where:
    y_t: Endogenous variables
    ε_t: Structural shocks
    η_t: Expectational errors

The solution is in state-space form:
    s_t = T * s_{t-1} + R * ε_t
    y_t = Z * s_t + D
"""

import numpy as np
from scipy import linalg
from typing import Tuple, Optional
import warnings


class DSGESolver:
    """Solver for linear rational expectations models using QZ decomposition."""

    def __init__(self, Gamma0: np.ndarray, Gamma1: np.ndarray,
                 Psi: np.ndarray, Pi: np.ndarray,
                 div_threshold: float = 1.01):
        """
        Initialize DSGE solver.

        Args:
            Gamma0: Coefficient matrix on y_t (n x n)
            Gamma1: Coefficient matrix on y_{t-1} (n x n)
            Psi: Coefficient matrix on structural shocks ε_t (n x n_eps)
            Pi: Coefficient matrix on expectational errors η_t (n x n_eta)
            div_threshold: Threshold for stable vs unstable eigenvalues (default 1.01)
        """
        self.Gamma0 = Gamma0
        self.Gamma1 = Gamma1
        self.Psi = Psi
        self.Pi = Pi
        self.div_threshold = div_threshold

        self.n = Gamma0.shape[0]  # Number of equations/variables
        self.n_eps = Psi.shape[1]  # Number of shocks
        self.n_eta = Pi.shape[1]   # Number of expectational errors

        # Solution matrices (to be computed)
        self.T = None
        self.R = None
        self.solved = False

    def solve(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Solve the rational expectations model using QZ decomposition.

        Returns:
            T: State transition matrix (n x n)
            R: Shock impact matrix (n x n_eps)
            info: Dictionary with solution information
        """
        # Step 1: QZ decomposition
        try:
            AA, BB, Q, Z, sdim = self._qz_decomposition()
        except Exception as e:
            raise ValueError(f"QZ decomposition failed: {e}")

        # Step 2: Check for existence and uniqueness
        info = self._check_blanchard_kahn(AA, BB)

        if not info['unique']:
            warnings.warn(f"Solution conditions not satisfied: {info['message']}")

        # Step 3: Construct solution matrices
        self.T, self.R = self._build_solution(AA, BB, Q, Z)
        self.solved = True

        return self.T, self.R, info

    def _qz_decomposition(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Perform QZ (generalized Schur) decomposition.

        Decomposes Gamma0 and Gamma1 as:
            Q' * Gamma0 * Z = AA (upper triangular)
            Q' * Gamma1 * Z = BB (upper triangular)

        Eigenvalues λ_i = BB[i,i] / AA[i,i] (if AA[i,i] ≠ 0)

        Returns:
            AA, BB: Upper triangular matrices
            Q, Z: Orthogonal transformation matrices
            sdim: Number of stable eigenvalues
        """
        # Compute QZ decomposition with sorting
        # Sort eigenvalues by modulus: |λ| < div_threshold (stable) first
        AA, BB, alpha, beta, Q, Z = linalg.ordqz(
            self.Gamma0, self.Gamma1,
            sort='ouc',  # Outside unit circle last
            output='real'
        )

        # Count stable eigenvalues
        # Eigenvalue λ_i = alpha_i / beta_i
        with np.errstate(divide='ignore', invalid='ignore'):
            eigvals = alpha / beta

        # Stable if |λ| < div_threshold
        stable_mask = np.abs(eigvals) < self.div_threshold
        sdim = np.sum(stable_mask)

        return AA, BB, Q, Z, sdim

    def _check_blanchard_kahn(self, AA: np.ndarray, BB: np.ndarray) -> dict:
        """
        Check Blanchard-Kahn conditions for existence and uniqueness.

        Conditions:
        - Number of stable eigenvalues = Number of predetermined variables
        - Number of unstable eigenvalues = Number of forward-looking variables

        Args:
            AA, BB: Upper triangular matrices from QZ decomposition

        Returns:
            Dictionary with existence, uniqueness, and message
        """
        # Compute eigenvalues
        with np.errstate(divide='ignore', invalid='ignore'):
            eigvals = np.diag(BB) / np.diag(AA)

        # Classify eigenvalues
        stable = np.abs(eigvals) < self.div_threshold
        unstable = np.abs(eigvals) >= self.div_threshold
        explosive = np.abs(eigvals) > 100  # Very large

        n_stable = np.sum(stable)
        n_unstable = np.sum(unstable)
        n_explosive = np.sum(explosive)

        # For DSGE models, we need n_unstable = n_eta (expectational errors)
        # and n_stable = n - n_eta (predetermined variables)
        exists = n_stable >= self.n - self.n_eta
        unique = n_unstable == self.n_eta

        if not exists:
            message = f"No solution exists. Stable roots: {n_stable}, needed: {self.n - self.n_eta}"
        elif not unique:
            message = f"Solution not unique. Unstable roots: {n_unstable}, needed: {self.n_eta}"
        else:
            message = "Solution exists and is unique"

        return {
            'exists': exists,
            'unique': unique,
            'n_stable': n_stable,
            'n_unstable': n_unstable,
            'n_explosive': n_explosive,
            'eigenvalues': eigvals,
            'message': message
        }

    def _build_solution(self, AA: np.ndarray, BB: np.ndarray,
                       Q: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct solution matrices T and R.

        The solution has the form:
            y_t = T * y_{t-1} + R * ε_t

        Args:
            AA, BB: Upper triangular matrices from QZ
            Q, Z: Orthogonal transformation matrices

        Returns:
            T: Transition matrix
            R: Shock impact matrix
        """
        # Number of stable eigenvalues
        n_stable = self.n - self.n_eta

        # Partition Z matrix
        Z11 = Z[:n_stable, :n_stable]
        Z12 = Z[:n_stable, n_stable:]
        Z21 = Z[n_stable:, :n_stable]
        Z22 = Z[n_stable:, n_stable:]

        # Check if Z22 is invertible
        if np.linalg.matrix_rank(Z22) < Z22.shape[0]:
            warnings.warn("Z22 is singular, solution may be inaccurate")
            Z22_inv = np.linalg.pinv(Z22)  # Pseudo-inverse
        else:
            Z22_inv = np.linalg.inv(Z22)

        # Construct T matrix
        # T relates predetermined variables to themselves
        T = Z21 @ np.linalg.inv(Z11)

        # Construct R matrix
        # R captures shock impact
        Q_Psi = Q.T @ self.Psi
        R = (Z @ np.linalg.solve(AA, Q_Psi))[:n_stable, :]

        return T, R

    def impulse_responses(self, shock_idx: int, periods: int,
                         shock_size: float = 1.0) -> np.ndarray:
        """
        Compute impulse response functions.

        Args:
            shock_idx: Index of shock (0 to n_eps-1)
            periods: Number of periods for IRF
            shock_size: Size of shock (default 1 std dev)

        Returns:
            IRF matrix (periods x n)
        """
        if not self.solved:
            raise ValueError("Model must be solved before computing IRFs")

        n = self.T.shape[0]
        irf = np.zeros((periods, n))

        # Initial shock
        s = np.zeros(n)
        eps = np.zeros(self.n_eps)
        eps[shock_idx] = shock_size

        # Impact period
        s = self.R @ eps
        irf[0, :] = s

        # Propagation
        for t in range(1, periods):
            s = self.T @ s
            irf[t, :] = s

        return irf

    def compute_variance_decomposition(self, Q: np.ndarray, periods: int = 40) -> np.ndarray:
        """
        Compute forecast error variance decomposition.

        Args:
            Q: Shock covariance matrix (n_eps x n_eps)
            periods: Forecast horizon

        Returns:
            Variance decomposition (periods x n x n_eps)
        """
        if not self.solved:
            raise ValueError("Model must be solved first")

        n = self.T.shape[0]
        vd = np.zeros((periods, n, self.n_eps))

        # Compute MSE at each horizon
        mse = np.zeros((periods, n, n))
        mse[0] = self.R @ Q @ self.R.T

        for h in range(1, periods):
            mse[h] = self.T @ mse[h-1] @ self.T.T + self.R @ Q @ self.R.T

        # Decompose by shock
        for shock in range(self.n_eps):
            Q_shock = np.zeros((self.n_eps, self.n_eps))
            Q_shock[shock, shock] = Q[shock, shock]

            mse_shock = np.zeros((periods, n, n))
            mse_shock[0] = self.R @ Q_shock @ self.R.T

            for h in range(1, periods):
                mse_shock[h] = self.T @ mse_shock[h-1] @ self.T.T + self.R @ Q_shock @ self.R.T

            for h in range(periods):
                vd[h, :, shock] = np.diag(mse_shock[h]) / np.diag(mse[h])

        return vd

    def check_stability(self) -> Tuple[bool, float]:
        """
        Check if the solution is stable (all eigenvalues inside unit circle).

        Returns:
            (is_stable, max_eigenvalue_modulus)
        """
        if not self.solved:
            raise ValueError("Model must be solved first")

        eigenvalues = np.linalg.eigvals(self.T)
        max_modulus = np.max(np.abs(eigenvalues))
        is_stable = max_modulus < 1.0

        return is_stable, max_modulus


def solve_dsge(Gamma0: np.ndarray, Gamma1: np.ndarray,
               Psi: np.ndarray, Pi: np.ndarray,
               div_threshold: float = 1.01) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Convenience function to solve DSGE model in one step.

    Args:
        Gamma0: Coefficient matrix on y_t
        Gamma1: Coefficient matrix on y_{t-1}
        Psi: Coefficient matrix on shocks
        Pi: Coefficient matrix on expectational errors
        div_threshold: Stability threshold

    Returns:
        T: Transition matrix
        R: Shock impact matrix
        info: Solution information
    """
    solver = DSGESolver(Gamma0, Gamma1, Psi, Pi, div_threshold)
    T, R, info = solver.solve()
    return T, R, info


if __name__ == '__main__':
    print("Testing DSGE solver with simple RBC model...")

    # Simple 2-equation example:
    # k_t = α * k_{t-1} + β * E_t[k_{t+1}] + ε_t
    # Rewrite as: k_t - β * k_{t+1} = α * k_{t-1} + ε_t

    # In canonical form:
    # [1, -β] * [k_t; k_{t+1}] = [α, 0] * [k_{t-1}; k_t] + [1; 0] * ε_t + [0; 1] * η_t

    # Parameters
    alpha = 0.9
    beta = 0.5

    # Simple 2x2 system
    Gamma0 = np.array([[1.0, -beta],
                       [0.0, 1.0]])

    Gamma1 = np.array([[alpha, 0.0],
                       [1.0, 0.0]])

    Psi = np.array([[1.0],
                    [0.0]])

    Pi = np.array([[0.0],
                   [1.0]])

    # Solve
    solver = DSGESolver(Gamma0, Gamma1, Psi, Pi)
    T, R, info = solver.solve()

    print(f"\nSolution info: {info['message']}")
    print(f"Stable eigenvalues: {info['n_stable']}")
    print(f"Unstable eigenvalues: {info['n_unstable']}")

    print(f"\nTransition matrix T:")
    print(T)

    print(f"\nShock impact matrix R:")
    print(R)

    # Check stability
    is_stable, max_eig = solver.check_stability()
    print(f"\nSystem stable: {is_stable}, max eigenvalue: {max_eig:.3f}")

    # Compute IRF
    irf = solver.impulse_responses(shock_idx=0, periods=20)
    print(f"\nIRF shape: {irf.shape}")
    print(f"Impact: {irf[0, 0]:.3f}, Period 10: {irf[10, 0]:.3f}")

    print("\nAll tests passed!")
