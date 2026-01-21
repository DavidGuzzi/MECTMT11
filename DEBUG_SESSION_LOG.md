# Debugging Session Log - Smets-Wouters Model Solution
**Date:** 2026-01-20
**Session:** 3
**Issue:** Singular matrix error in QZ decomposition

## Problem Statement

The Smets-Wouters DSGE model equations were fully implemented in Python (`sw_equations_v2.py`, 40 equations, 560 lines of code), but the QZ-based rational expectations solver fails with:
```
numpy.linalg.LinAlgError: Singular matrix
```

Specifically: Solver finds 0 stable eigenvalues (needs 28) and 13 unstable eigenvalues (needs 12).

## Root Cause Analysis

### 1. Matrix Structure Diagnosis

**Gamma0 Matrix:**
- Rank: 40 (full rank) ✓
- Condition number: 1.73e+02 (acceptable)

**Gamma1 Matrix:**
- Rank: 14 (severely rank deficient!) ✗
- Condition number: inf
- **20 zero rows** (equations with no lagged variables)
- **25 zero columns** (variables that don't appear lagged)

### 2. Eigenvalue Distribution

From QZ decomposition of (Gamma0, Gamma1):
- **Stable (|λ| < 1):** 0 eigenvalues (need 28)
- **Unstable (1 ≤ |λ| < 100):** 13 eigenvalues (need 12)
- **Explosive (|λ| ≥ 100):** 27 eigenvalues

The 27 explosive eigenvalues (magnitude ~10^20) arise from near-zero diagonal elements in the BB matrix from QZ decomposition, corresponding to the 25 zero columns in Gamma1.

### 3. Equation-Variable Alignment

**Fixed in v2:** Initial implementation had equations ordered arbitrarily (flexible economy → sticky economy → shocks → measurement).

**Corrected:** Each equation now placed in the row corresponding to its defined variable index. For example:
- Equation defining `pinf` → Row 28 (where pinf is variable index 28)
- Equation defining `labobs` → Row 0 (where labobs is variable index 0)

This fixed the Pi matrix structure - expectational errors now correctly appear in equations 12, 13, 14, 23, 24, 25, 28, 29 (pkf, cf, invef, pk, c, inve, pinf, w).

### 4. Variables with Lagged Terms

Only 15 of 40 variables have lagged terms (non-zero columns in Gamma1):
- **Forward-looking with lags:** cf, invef, yf, c, inve, y, pinf, w, r
- **Shock processes:** a, b, g, qs
- **Capital stocks:** kpf, kp

The remaining 25 variables are **static** or **jump** variables defined purely by current-period conditions.

### 5. Canonical Form Specification

Using Sims' canonical form:
```
Γ0·y_t = Γ1·y_{t-1} + Ψ·ε_t + Π·η_t
```

Where:
- **Γ0:** (40×40) Coefficients on current variables
- **Γ1:** (40×40) Coefficients on lagged variables
- **Ψ:** (40×7) Coefficients on structural shocks
- **Π:** (40×12) Coefficients on expectational errors

**η_t = y_t - E_{t-1}[y_t]** for forward-looking variables: invef, rkf, pkf, cf, labf, inve, rk, pk, c, lab, pinf, w

## Hypothesis: Why the Solver Fails

The QZ-based Blanchard-Kahn solver expects:
1. **n_stable = n - n_forward** (28 = 40 - 12)
2. **n_unstable = n_forward** (12)

But we observe:
- n_stable = 0
- n_unstable = 13
- n_explosive = 27

**Possible causes:**

### A. Variable Ordering Issue
Sims' QZ method may require predetermined variables (with lags) to be ordered **first**, then jump variables. Current ordering mixes them.

### B. Auxiliary Variable Missing
Dynare automatically creates auxiliary variables for leads ≥ 2. Our manual translation might be missing this step.

### C. Canonical Form Mismatch
The way expectational errors are handled in the Π matrix might not align with how the solver interprets them.

### D. Numerical Issues
With 25 variables having no dynamics (zero Gamma1 columns), the system may be fundamentally different from standard DSGE models that the solver expects.

## Attempted Fixes

1. ✅ **Corrected equation-variable alignment** (sw_equations_v2.py)
   - Result: Pi matrix now correct, but eigenvalue problem persists

2. ✅ **Verified equation translations** against usmodel.mod
   - Result: Translations appear correct

3. ⏸️ **Variable reordering** (predetermined first, jump last)
   - Status: Not yet implemented

4. ⏸️ **Alternative solver** (gensys, Klein 2000)
   - Status: Researching existing Python implementations

## Next Steps

### Option 1: Use Established Python DSGE Solver
Replace custom QZ solver with proven implementation:
- **pydsge** - Full DSGE estimation suite with gensys solver
- **dsgepy** - Calibrate/solve/simulate with Dynare-like interface
- **eph/dsge** - Pure Python gensys implementation

### Option 2: Fix Variable Ordering
Reorder variable list to place predetermined variables first:
1. Variables with lags: cf, invef, yf, c, inve, y, pinf, w, r, a, b, g, qs, kpf, kp (15 vars)
2. Jump/static variables: remaining 25

Then rebuild Gamma0, Gamma1, Psi, Pi with new ordering.

### Option 3: Consult Dynare Output
Run usmodel.mod in Dynare/MATLAB, extract the actual decision rules (g1, g2 matrices), and compare with our Python matrices to identify discrepancies.

## Files Modified

1. **`sw_equations.py`** → **`sw_equations_v2.py`** (corrected indexing)
2. **`model.py`** (updated to import sw_equations_v2)
3. **`diagnose_eigenvalues.py`** (diagnostic tool)
4. **`check_gamma1.py`** (Gamma1 structure analysis)
5. **`test_model_solution.py`** (test script)

## Code Statistics

- **Total Python code:** ~3,900 lines
- **Equation builder:** 560 lines (40 equations translated)
- **Diagnostic tools:** 150 lines

## References

### Python DSGE Solvers
- [dsgepy](https://github.com/gusamarante/dsgepy) - Gensys-based DSGE modeling
- [pydsge](https://github.com/patrickrmaia/pydsge) - DSGE estimation
- [eph/dsge gensys.py](https://github.com/eph/dsge/blob/master/dsge/gensys.py) - Pure Python implementation
- [linearsolve](https://github.com/letsgoexploring/linearsolve) - Klein (2000) method

### Theory
- Sims, C. A. (2002). "Solving Linear Rational Expectations Models"
- Klein, P. (2000). "Using the Generalized Schur Form to Solve a System of Linear Expectational Difference Equations"
- Blanchard, O. J., & Kahn, C. M. (1980). "The Solution of Linear Difference Models under Rational Expectations"

## Conclusion

The equation implementation is **structurally correct** but the **solver methodology** or **variable ordering** is incompatible with the model's structure (25 static variables, 15 dynamic). Recommend switching to a proven gensys implementation rather than continuing to debug the custom QZ solver.
