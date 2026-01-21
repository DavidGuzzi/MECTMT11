"""
Diagnostic Script for Eigenvalue Analysis
==========================================

This script examines the eigenvalues from the QZ decomposition to understand
why all eigenvalues are stable when we expect 12 unstable ones.
"""

import sys
sys.path.insert(0, r'C:\Users\HP\OneDrive\Escritorio\David Guzzi\Github\MECTMT11')

import numpy as np
from replication.model import SmetsWoutersModel

print("="*70)
print("EIGENVALUE DIAGNOSTIC")
print("="*70)

# Create model
model = SmetsWoutersModel()

# Build matrices
print("\n1. Building matrices...")
Gamma0, Gamma1, Psi, Pi = model.build_matrices()

print(f"   Gamma0 shape: {Gamma0.shape}")
print(f"   Gamma1 shape: {Gamma1.shape}")
print(f"   Psi shape: {Psi.shape}")
print(f"   Pi shape: {Pi.shape}")

# Check matrix properties
print("\n2. Matrix properties:")
print(f"   Gamma0 rank: {np.linalg.matrix_rank(Gamma0)}")
print(f"   Gamma1 rank: {np.linalg.matrix_rank(Gamma1)}")
print(f"   Gamma0 condition number: {np.linalg.cond(Gamma0):.2e}")
print(f"   Gamma1 condition number: {np.linalg.cond(Gamma1):.2e}")

# Check for zero rows/columns
zero_rows_g0 = np.where(np.all(np.abs(Gamma0) < 1e-10, axis=1))[0]
zero_cols_g0 = np.where(np.all(np.abs(Gamma0) < 1e-10, axis=0))[0]
zero_rows_g1 = np.where(np.all(np.abs(Gamma1) < 1e-10, axis=1))[0]
zero_cols_g1 = np.where(np.all(np.abs(Gamma1) < 1e-10, axis=0))[0]

if len(zero_rows_g0) > 0:
    print(f"\n   WARNING: Gamma0 has {len(zero_rows_g0)} zero rows: {zero_rows_g0}")
    for idx in zero_rows_g0:
        print(f"      Equation {idx}: {model.var_names[idx] if idx < len(model.var_names) else 'N/A'}")

if len(zero_cols_g0) > 0:
    print(f"\n   WARNING: Gamma0 has {len(zero_cols_g0)} zero columns: {zero_cols_g0}")
    for idx in zero_cols_g0:
        print(f"      Variable {idx}: {model.var_names[idx] if idx < len(model.var_names) else 'N/A'}")

# Perform QZ decomposition manually
print("\n3. Performing QZ decomposition...")
try:
    from scipy.linalg import qz
    AA, BB, Q, Z = qz(Gamma0, Gamma1, output='complex')

    # Compute eigenvalues
    eigvals = np.diag(AA) / (np.diag(BB) + 1e-20)
    eigvals_abs = np.abs(eigvals)

    print(f"   QZ decomposition successful")
    print(f"   AA diagonal: min={np.min(np.abs(np.diag(AA))):.2e}, max={np.max(np.abs(np.diag(AA))):.2e}")
    print(f"   BB diagonal: min={np.min(np.abs(np.diag(BB))):.2e}, max={np.max(np.abs(np.diag(BB))):.2e}")

    # Categorize eigenvalues
    stable = eigvals_abs < 1.0
    unstable = (eigvals_abs >= 1.0) & (eigvals_abs < 100)
    explosive = eigvals_abs >= 100

    n_stable = np.sum(stable)
    n_unstable = np.sum(unstable)
    n_explosive = np.sum(explosive)

    print(f"\n4. Eigenvalue classification:")
    print(f"   Stable (|eig| < 1): {n_stable} (need {40 - 12} = 28)")
    print(f"   Unstable (1 <= |eig| < 100): {n_unstable} (need 12)")
    print(f"   Explosive (|eig| >= 100): {n_explosive}")

    # Print eigenvalue distribution
    print(f"\n5. Eigenvalue magnitudes:")
    sorted_idx = np.argsort(eigvals_abs)
    for i in range(len(eigvals)):
        idx = sorted_idx[i]
        category = "STABLE" if stable[idx] else ("UNSTABLE" if unstable[idx] else "EXPLOSIVE")
        print(f"   eig_{i+1:2d}: {eigvals_abs[idx]:12.6f}  [{category:9s}]  (real={eigvals[idx].real:8.4f}, imag={eigvals[idx].imag:8.4f})")

    # Check for near-zero or infinite eigenvalues
    near_zero = eigvals_abs < 1e-6
    near_inf = eigvals_abs > 1e6

    if np.sum(near_zero) > 0:
        print(f"\n   WARNING: {np.sum(near_zero)} near-zero eigenvalues detected")
    if np.sum(near_inf) > 0:
        print(f"   WARNING: {np.sum(near_inf)} near-infinite eigenvalues detected")

    # Expected forward-looking variables
    forward_vars = ['invef', 'rkf', 'pkf', 'cf', 'labf',
                    'inve', 'rk', 'pk', 'c', 'lab', 'pinf', 'w']

    print(f"\n6. Expected forward-looking variables ({len(forward_vars)}):")
    for v in forward_vars:
        if v in model.var_names:
            idx = model.var_names.index(v)
            print(f"   - {v:10s} (index {idx})")

    # Check Pi matrix
    print(f"\n7. Pi matrix properties:")
    print(f"   Non-zero entries: {np.sum(np.abs(Pi) > 1e-10)}")
    print(f"   Max absolute value: {np.max(np.abs(Pi)):.6f}")
    print(f"   Rows with non-zero entries: {np.sum(np.any(np.abs(Pi) > 1e-10, axis=1))}")
    print(f"   Columns with non-zero entries: {np.sum(np.any(np.abs(Pi) > 1e-10, axis=0))}")

    # Check which equations have expectational errors
    eqs_with_exp = np.where(np.any(np.abs(Pi) > 1e-10, axis=1))[0]
    print(f"\n8. Equations with expectational errors ({len(eqs_with_exp)}):")
    for eq_idx in eqs_with_exp:
        var_name = model.var_names[eq_idx] if eq_idx < len(model.var_names) else f"Eq {eq_idx}"
        n_exp_errors = np.sum(np.abs(Pi[eq_idx, :]) > 1e-10)
        print(f"   Equation {eq_idx:2d} ({var_name:10s}): {n_exp_errors} exp. errors")

except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)
