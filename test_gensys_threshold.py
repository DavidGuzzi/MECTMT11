"""
Test different thresholds for gensys
"""

import sys
sys.path.insert(0, r'C:\Users\HP\OneDrive\Escritorio\David Guzzi\Github\MECTMT11')

import numpy as np
from replication.model import SmetsWoutersModel
from replication.gensys import gensys

print("="*70)
print("TESTING DIFFERENT THRESHOLDS FOR GENSYS")
print("="*70)

model = SmetsWoutersModel()
Gamma0, Gamma1, Psi, Pi = model.build_matrices()

print(f"\nPi matrix rank: {np.linalg.matrix_rank(Pi)}")
print(f"Pi matrix shape: {Pi.shape}")
print(f"Expected forward-looking variables: {Pi.shape[1]}")

# Try different thresholds
thresholds = [1.001, 1.01, 1.05, 1.1, 1.5, 2.0, 10.0, 100.0]

for div in thresholds:
    print(f"\n{'='*70}")
    print(f"Testing with div = {div}")
    print(f"{'='*70}")

    try:
        G1, impact, eu = gensys(Gamma0, Gamma1, Psi, Pi, div=div, realsmall=1e-6)

        print(f"  Existence: {eu[0]}")
        print(f"  Uniqueness: {eu[1]}")

        if G1 is not None:
            print(f"  G1 shape: {G1.shape}")
            from scipy import linalg
            g1_eig = linalg.eigvals(G1)
            max_g1_eig = np.max(np.abs(g1_eig))
            print(f"  Max |λ(G1)|: {max_g1_eig:.6f}")
            print(f"  Stable: {max_g1_eig < 1.0}")

            if eu[0] == 1 and eu[1] == 1:
                print(f"  ✓ SOLUTION FOUND!")
                break
        else:
            print(f"  ✗ No solution")

    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "="*70)
