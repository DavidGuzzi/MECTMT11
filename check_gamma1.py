"""Check which rows/columns of Gamma1 are zero"""

import sys
sys.path.insert(0, r'C:\Users\HP\OneDrive\Escritorio\David Guzzi\Github\MECTMT11')

import numpy as np
from replication.model import SmetsWoutersModel

model = SmetsWoutersModel()
Gamma0, Gamma1, Psi, Pi = model.build_matrices()

print("Checking Gamma1 matrix for zero rows/columns:")
print("=" * 60)

# Check rows
zero_rows = []
for i in range(40):
    row_norm = np.linalg.norm(Gamma1[i, :])
    if row_norm < 1e-10:
        var_name = model.var_names[i]
        print(f"Row {i:2d} ({var_name:10s}): ZERO (norm={row_norm:.2e})")
        zero_rows.append(i)

print(f"\nTotal zero rows: {len(zero_rows)}")

# Check columns
zero_cols = []
for j in range(40):
    col_norm = np.linalg.norm(Gamma1[:, j])
    if col_norm < 1e-10:
        var_name = model.var_names[j]
        print(f"Col {j:2d} ({var_name:10s}): ZERO (norm={col_norm:.2e})")
        zero_cols.append(j)

print(f"\nTotal zero columns: {len(zero_cols)}")

# Check which variables appear in Gamma1 (as lags)
print("\n" + "=" * 60)
print("Variables with lagged terms (non-zero columns in Gamma1):")
print("=" * 60)

for j in range(40):
    col_norm = np.linalg.norm(Gamma1[:, j])
    if col_norm > 1e-10:
        var_name = model.var_names[j]
        n_eqs = np.sum(np.abs(Gamma1[:, j]) > 1e-10)
        print(f"  {var_name:10s} (col {j:2d}): appears in {n_eqs} equations")
