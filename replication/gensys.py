"""
Gensys Algorithm - Sims (2002)
================================

Pure Python implementation of Christopher Sims' gensys algorithm for solving
linear rational expectations models.

Based on the implementation by Ed Herbst (eph/dsge):
https://github.com/eph/dsge/blob/master/dsge/gensys.py

Citation:
Sims, C. A. (2002). Solving linear rational expectations models.
Computational economics, 20(1-2), 1-20.

Canonical form:
    Γ0·y_t = Γ1·y_{t-1} + c + Ψ·z_t + Π·η_t

Where:
    y_t: endogenous variables
    z_t: exogenous shocks
    η_t: expectational errors (one-step-ahead forecast errors)
"""

import numpy as np
from scipy import linalg
from typing import Tuple, Optional


def gensys(g0: np.ndarray, g1: np.ndarray, psi: np.ndarray, pi: np.ndarray,
           div: float = 1.01, realsmall: float = 1e-6,
           return_everything: bool = False) -> Tuple:
    """
    Solve linear rational expectations model using QZ decomposition.

    Solves the system:
        Γ0·y_t = Γ1·y_{t-1} + Ψ·ε_t + Π·η_t

    Returns policy functions:
        y_t = G1·y_{t-1} + impact·ε_t

    Args:
        g0: Coefficient matrix on y_t (n x n)
        g1: Coefficient matrix on y_{t-1} (n x n)
        psi: Coefficient matrix on shocks ε_t (n x n_eps)
        pi: Coefficient matrix on expectational errors η_t (n x n_eta)
        div: Eigenvalue threshold for stable/unstable split (default 1.01)
        realsmall: Numerical tolerance for zero (default 1e-6)
        return_everything: Return additional outputs (default False)

    Returns:
        G1: State transition matrix (n x n)
        impact: Shock impact matrix (n x n_eps)
        eu: [existence, uniqueness] indicators (2-element array)
             eu[0] = 1: solution exists
             eu[1] = 1: solution is unique
        (if return_everything=True, also returns additional diagnostics)
    """

    n = g0.shape[0]

    # QZ decomposition with ordered eigenvalues
    # Sort eigenvalues: stable (|λ| < div) first, unstable last
    try:
        aa, bb, alpha, beta, q, z = linalg.ordqz(g0, g1, sort='ouc', output='real')
    except Exception as e:
        print(f"QZ decomposition failed: {e}")
        eu = np.array([0, 0])
        if return_everything:
            return None, None, eu, None, None, None, None
        return None, None, eu

    # Compute generalized eigenvalues
    with np.errstate(divide='ignore', invalid='ignore'):
        eigval = np.where(np.abs(beta) > realsmall, alpha / beta, np.inf)
        eigval = np.where(np.abs(eigval) < 1e10, eigval, np.inf)

    # Count stable eigenvalues (outside unit circle after inversion)
    # In 'ouc' sorting, unstable eigenvalues (|λ| >= div) come last
    nunstab = np.sum(np.abs(eigval) > div)

    # Check Blanchard-Kahn conditions
    # For existence: need enough stable eigenvalues
    # For uniqueness: number of unstable eigenvalues should equal number of forward-looking variables

    # Partition system based on stable/unstable split
    nstab = n - nunstab

    if nstab == 0:
        print("No stable eigenvalues found - all variables explosive")
        eu = np.array([0, 0])
        if return_everything:
            return None, None, eu, aa, bb, q, z
        return None, None, eu

    # Extract Q submatrices
    q1 = q[:, :nstab]  # Stable component
    q2 = q[:, nstab:]  # Unstable component

    # Check existence: unstable block of Π must be of full rank
    # pi @ q2 should have rank = nunstab
    if nunstab > 0:
        etawt = q2.T @ pi
        try:
            ueta, deta, veta = linalg.svd(etawt)
            bigev_eta = np.where(np.abs(deta) > realsmall)[0]

            if len(bigev_eta) < nunstab:
                eu = np.array([0, 0])  # No solution exists
                print(f"No solution exists: rank(Π·Q2) = {len(bigev_eta)} < {nunstab}")
                if return_everything:
                    return None, None, eu, aa, bb, q, z
                return None, None, eu
        except:
            eu = np.array([0, 0])
            if return_everything:
                return None, None, eu, aa, bb, q, z
            return None, None, eu

    eu = np.array([1, 1])  # Default: solution exists and is unique

    # Check uniqueness: stable block should have no loose information
    if nstab > 0 and pi.shape[1] > 0:
        etawt1 = q1.T @ pi
        try:
            ueta1, deta1, veta1 = linalg.svd(etawt1)
            bigev_eta1 = np.where(np.abs(deta1) > realsmall)[0]

            if len(bigev_eta1) > 0:
                eu[1] = 0  # Solution exists but is not unique
                print(f"Solution not unique: stable block has {len(bigev_eta1)} loose constraints")
        except:
            pass

    # Build solution
    # Partition Z matrix
    z11 = z[:nstab, :nstab]
    z21 = z[nstab:, :nstab]

    # Check if z11 is invertible
    try:
        z11inv = linalg.inv(z11)
    except:
        print("z11 is singular - using pseudo-inverse")
        z11inv = linalg.pinv(z11)

    # Construct policy function: y_t = G1·y_{t-1} + impact·ε_t

    # State transition matrix
    if nunstab > 0:
        # G1 = z11inv @ z21
        # But we need to project to eliminate unstable component
        z22 = z[nstab:, nstab:]

        try:
            z22inv = linalg.inv(z22)
        except:
            z22inv = linalg.pinv(z22)

        # Eliminate expectational errors
        g1_solution = z @ np.vstack([
            z11inv,
            -z22inv @ z21 @ z11inv
        ])
    else:
        g1_solution = z @ np.vstack([z11inv, np.zeros((n - nstab, nstab))])

    # Extract state transition for stable variables
    G1 = g1_solution[:, :nstab]

    # Impact matrix
    # impact = (Z @ inv(AA) @ Q') @ Ψ
    try:
        # Use only stable component
        qpsi = q1.T @ psi
        aa11 = aa[:nstab, :nstab]
        aa11inv = linalg.solve(aa11, qpsi)
        impact = z[:, :nstab] @ aa11inv
    except:
        print("Warning: Could not compute impact matrix accurately")
        impact = np.zeros((n, psi.shape[1]))

    if return_everything:
        return G1, impact, eu, aa, bb, q, z

    return G1, impact, eu


def check_solution(G1: np.ndarray, eu: np.ndarray, var_names: Optional[list] = None) -> None:
    """
    Print solution diagnostics.

    Args:
        G1: State transition matrix
        eu: [existence, uniqueness] array
        var_names: Optional variable names for display
    """
    print("\n" + "="*60)
    print("GENSYS SOLUTION DIAGNOSTICS")
    print("="*60)

    exists = eu[0] == 1
    unique = eu[1] == 1

    print(f"\nSolution exists: {exists}")
    print(f"Solution unique: {unique}")

    if G1 is not None:
        print(f"\nState transition matrix G1: {G1.shape}")

        # Check stability
        eigvals = linalg.eigvals(G1)
        max_eig = np.max(np.abs(eigvals))
        stable = max_eig < 1.0

        print(f"Maximum eigenvalue modulus: {max_eig:.6f}")
        print(f"System stable: {stable}")

        if not stable:
            print(f"  WARNING: System is unstable (max |λ| >= 1)")

        # Count eigenvalues by magnitude
        n_stable = np.sum(np.abs(eigvals) < 1.0)
        n_unit = np.sum(np.abs(np.abs(eigvals) - 1.0) < 1e-6)
        n_unstable = np.sum(np.abs(eigvals) > 1.0)

        print(f"\nEigenvalue distribution:")
        print(f"  Stable (|λ| < 1): {n_stable}")
        print(f"  Unit circle (|λ| ≈ 1): {n_unit}")
        print(f"  Unstable (|λ| > 1): {n_unstable}")
    else:
        print("\nNo solution computed")

    print("="*60)


if __name__ == '__main__':
    print("Gensys solver module")
    print("Based on Sims (2002) algorithm")
    print("\nFor usage, see model.py or create a simple test model")
