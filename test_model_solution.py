"""
Test Smets-Wouters Model Solution
==================================

Quick test to verify that the model solves correctly with default parameters.
"""

import sys
sys.path.insert(0, r'C:\Users\HP\OneDrive\Escritorio\David Guzzi\Github\MECTMT11')

from replication.model import SmetsWoutersModel
import numpy as np

print("="*60)
print("TESTING SMETS-WOUTERS MODEL SOLUTION")
print("="*60)

# Create model instance
print("\n1. Creating model...")
model = SmetsWoutersModel()
print(f"   Variables: {len(model.var_names)}")
print(f"   Shocks: {len(model.shock_names)}")
print(f"   Parameters: {len(model.params)}")

# Test model solution
print("\n2. Solving model...")
try:
    info = model.solve()
    print(f"   Status: {info['message']}")
    print(f"   Solution exists: {info['exists']}")
    print(f"   Solution unique: {info['unique']}")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check stability
print("\n3. Checking stability...")
try:
    is_stable, max_eig = model.check_stability()
    print(f"   Stable: {is_stable}")
    print(f"   Max eigenvalue modulus: {max_eig:.6f}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test IRF computation
print("\n4. Testing IRF computation...")
try:
    irf_tech = model.impulse_responses(shock_idx=0, periods=20)  # Technology shock
    print(f"   IRF shape: {irf_tech.shape}")
    print(f"   Impact on output (yf): {irf_tech[0, model.var_names.index('yf')]:.6f}")
    print(f"   Period 10 on output: {irf_tech[10, model.var_names.index('yf')]:.6f}")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

# Display state-space matrices dimensions
print("\n5. State-space representation:")
print(f"   T matrix: {model.T.shape}")
print(f"   R matrix: {model.R.shape}")
print(f"   Z matrix: {model.Z.shape if model.Z is not None else 'None'}")
print(f"   Q matrix: {model.Q.shape if model.Q is not None else 'None'}")

# Display some key parameters
print("\n6. Key parameters:")
key_params = ['calfa', 'csigma', 'chabb', 'cprobp', 'cprobw',
              'crpi', 'crr', 'crhoa', 'crhob']
for param in key_params:
    if param in model.params:
        print(f"   {param}: {model.params[param]:.4f}")

print("\n" + "="*60)
print("TEST COMPLETED SUCCESSFULLY!" if info['unique'] else "TEST COMPLETED WITH WARNINGS")
print("="*60)
