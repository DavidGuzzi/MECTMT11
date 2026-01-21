"""
Test gensys solver directly with SW model matrices
"""

import sys
sys.path.insert(0, r'C:\Users\HP\OneDrive\Escritorio\David Guzzi\Github\MECTMT11')

import numpy as np
from replication.model import SmetsWoutersModel
from replication.gensys import gensys
from scipy import linalg

print("="*70)
print("DIRECT GENSYS TEST WITH SMETS-WOUTERS MATRICES")
print("="*70)

# Create model and build matrices
model = SmetsWoutersModel()
Gamma0, Gamma1, Psi, Pi = model.build_matrices()

print("\n1. Matrix dimensions:")
print(f"   Gamma0: {Gamma0.shape}")
print(f"   Gamma1: {Gamma1.shape}")
print(f"   Psi: {Psi.shape}")
print(f"   Pi: {Pi.shape}")

print("\n2. Matrix properties:")
print(f"   Gamma0 rank: {np.linalg.matrix_rank(Gamma0)}")
print(f"   Gamma1 rank: {np.linalg.matrix_rank(Gamma1)}")
print(f"   Pi rank: {np.linalg.matrix_rank(Pi)}")

# Check for issues
print("\n3. Checking for potential issues:")
det_g0 = np.linalg.det(Gamma0)
print(f"   det(Gamma0): {det_g0:.2e}")
if abs(det_g0) < 1e-10:
    print("   WARNING: Gamma0 is nearly singular!")

# Try QZ decomposition manually
print("\n4. Attempting QZ decomposition...")
try:
    aa, bb, alpha, beta, q, z = linalg.ordqz(Gamma0, Gamma1, sort='ouc', output='real')
    print("   QZ decomposition successful")

    # Compute eigenvalues
    with np.errstate(divide='ignore', invalid='ignore'):
        eigval = alpha / beta
        eigval = np.where(np.abs(eigval) < 1e10, eigval, np.inf)

    eigval_abs = np.abs(eigval)

    # Classify
    stable = eigval_abs < 1.01
    unstable = (eigval_abs >= 1.01) & (eigval_abs < 100)
    explosive = eigval_abs >= 100

    n_stable = np.sum(stable)
    n_unstable = np.sum(unstable)
    n_explosive = np.sum(explosive)

    print(f"   Stable eigenvalues: {n_stable}")
    print(f"   Unstable eigenvalues: {n_unstable}")
    print(f"   Explosive eigenvalues: {n_explosive}")

    # Show first few eigenvalues
    sorted_idx = np.argsort(eigval_abs)
    print(f"\n   First 15 eigenvalues (sorted by magnitude):")
    for i in range(min(15, len(eigval))):
        idx = sorted_idx[i]
        cat = "STABLE" if stable[idx] else ("UNSTABLE" if unstable[idx] else "EXPLOSIVE")
        print(f"      λ_{i+1:2d}: {eigval_abs[idx]:12.6f} [{cat:9s}]")

except Exception as e:
    print(f"   QZ decomposition failed: {e}")
    import traceback
    traceback.print_exc()

# Now try gensys
print("\n5. Calling gensys solver...")
try:
    G1, impact, eu = gensys(Gamma0, Gamma1, Psi, Pi, div=1.01, realsmall=1e-6)

    print(f"\n   Existence: {eu[0]}")
    print(f"   Uniqueness: {eu[1]}")

    if G1 is not None:
        print(f"   G1 shape: {G1.shape}")
        print(f"   impact shape: {impact.shape}")

        # Check G1 eigenvalues
        g1_eig = linalg.eigvals(G1)
        max_g1_eig = np.max(np.abs(g1_eig))
        print(f"   Max |λ(G1)|: {max_g1_eig:.6f}")
        print(f"   Stable: {max_g1_eig < 1.0}")
    else:
        print("   G1 is None - no solution found")

except Exception as e:
    print(f"   Gensys failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
