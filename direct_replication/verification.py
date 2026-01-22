"""
Verification Module

Tests to verify that replication results match the original paper.
Implements the verification criteria from memory.md.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


class ReplicationVerification:
    """
    Verify replication results against Smets & Wouters (2007) paper.

    This class implements the verification criteria documented in memory.md:
    1. State-space matrices (T, R, Z) match within tolerance
    2. Log-likelihood at mode matches within 1e-4
    3. Parameter estimates match within 1%
    4. IRFs visually match (RMSE < 5% of peak)
    5. BVAR marginal likelihoods match within 1%
    6. Forecast RMSE match within 1%
    """

    def __init__(self):
        """Initialize verification with reference values from paper."""
        # Reference log-likelihood (approximate from paper)
        self.reference_loglik = -894.82

        # Reference parameter values from Table 1 of SW (2007)
        self.reference_params = {
            # Estimated parameters (posterior mode)
            'csigma': 1.39,      # Risk aversion
            'cfc': 1.61,         # Fixed cost
            'constelab': 0.69,   # Labor supply
            'constepinf': 0.78,  # Steady-state inflation
            'ctrend': 0.43,      # Trend growth
            'constebeta': 0.16,  # Discount factor
            'calfa': 0.19,       # Capital share
            'crhoa': 0.96,       # AR(1) technology shock
            'crhob': 0.18,       # AR(1) preference shock
            'crhog': 0.97,       # AR(1) spending shock
            'crhoqs': 0.71,      # AR(1) investment shock
            'crhoms': 0.15,      # AR(1) monetary shock
            'crhopinf': 0.90,    # AR(1) price markup shock
            'crhow': 0.97,       # AR(1) wage markup shock
            'crpi': 2.04,        # Taylor rule inflation
            'crr': 0.81,         # Taylor rule smoothing
            'cry': 0.08,         # Taylor rule output gap
            'crdy': 0.22,        # Taylor rule output growth
            'czcap': 0.54,       # Capacity utilization
            'cgy': 0.52,         # Govt spending/GDP
            'csadjcost': 5.48,   # Investment adjustment cost
            'chabb': 0.71,       # Habit formation
            'cprobw': 0.73,      # Wage stickiness (Calvo)
            'cprobp': 0.65,      # Price stickiness (Calvo)
            'csigl': 1.92,       # Labor supply elasticity
            'cindw': 0.59,       # Wage indexation
            'cindp': 0.22,       # Price indexation
        }

        # Reference BVAR marginal likelihoods (approximate from paper Table 2)
        self.reference_bvar_mlik = {
            'BVAR(1)': -1050.0,  # Approximate values
            'BVAR(2)': -1020.0,
            'BVAR(3)': -1030.0,
            'BVAR(4)': -1045.0,
        }

    def verify_likelihood(self, ll_computed: float,
                         ll_reference: Optional[float] = None,
                         tol: float = 1e-4) -> bool:
        """
        Verify log-likelihood at posterior mode.

        Args:
            ll_computed: Computed log-likelihood
            ll_reference: Reference value (default: from paper)
            tol: Tolerance (default: 1e-4)

        Returns:
            True if verification passes

        Notes:
            Criterion 2 from memory.md
        """
        if ll_reference is None:
            ll_reference = self.reference_loglik

        diff = abs(ll_computed - ll_reference)
        passed = diff < tol

        print(f"\nLikelihood Verification:")
        print(f"  Computed: {ll_computed:.6f}")
        print(f"  Reference: {ll_reference:.6f}")
        print(f"  Difference: {diff:.6f}")
        print(f"  Tolerance: {tol:.6f}")
        print(f"  Status: {'PASS ✓' if passed else 'FAIL ✗'}")

        return passed

    def verify_parameters(self, params_computed: Dict[str, float],
                         params_reference: Optional[Dict[str, float]] = None,
                         tol_pct: float = 0.01) -> Dict[str, bool]:
        """
        Verify parameter estimates against paper values.

        Args:
            params_computed: Dictionary of computed parameters
            params_reference: Dictionary of reference values (default: from paper)
            tol_pct: Tolerance as percentage (default: 1%)

        Returns:
            Dictionary with verification status for each parameter

        Notes:
            Criterion 3 from memory.md
        """
        if params_reference is None:
            params_reference = self.reference_params

        results = {}
        print(f"\nParameter Verification (tolerance: {tol_pct*100}%):")
        print("-" * 80)
        print(f"{'Parameter':<15} {'Computed':>12} {'Reference':>12} {'Diff %':>10} {'Status':>8}")
        print("-" * 80)

        for param, ref_val in params_reference.items():
            if param in params_computed:
                comp_val = params_computed[param]
                diff_pct = abs(comp_val - ref_val) / abs(ref_val)
                passed = diff_pct < tol_pct

                results[param] = passed

                status = '✓' if passed else '✗'
                print(f"{param:<15} {comp_val:>12.4f} {ref_val:>12.4f} "
                      f"{diff_pct*100:>9.2f}% {status:>8}")
            else:
                results[param] = False
                print(f"{param:<15} {'N/A':>12} {ref_val:>12.4f} {'N/A':>10} {'✗':>8}")

        print("-" * 80)
        n_passed = sum(results.values())
        n_total = len(results)
        print(f"Passed: {n_passed}/{n_total} ({n_passed/n_total*100:.1f}%)")

        return results

    def verify_state_space(self, T_computed: np.ndarray,
                          T_reference: np.ndarray,
                          tol: float = 1e-4) -> bool:
        """
        Verify state-space matrices match.

        Args:
            T_computed: Computed state transition matrix
            T_reference: Reference state transition matrix
            tol: Tolerance for Frobenius norm

        Returns:
            True if verification passes

        Notes:
            Criterion 1 from memory.md
        """
        # Compute Frobenius norm of difference
        diff = T_computed - T_reference
        frob_norm = np.linalg.norm(diff, ord='fro')

        passed = frob_norm < tol

        print(f"\nState-Space Matrix Verification:")
        print(f"  Shape: {T_computed.shape}")
        print(f"  Frobenius norm of difference: {frob_norm:.6e}")
        print(f"  Tolerance: {tol:.6e}")
        print(f"  Status: {'PASS ✓' if passed else 'FAIL ✗'}")

        return passed

    def verify_irfs(self, irfs_computed: pd.DataFrame,
                   irfs_reference: pd.DataFrame,
                   tol: float = 0.05) -> Tuple[bool, pd.DataFrame]:
        """
        Verify IRFs match paper figures.

        Args:
            irfs_computed: Computed IRFs (variable, shock, period, value)
            irfs_reference: Reference IRFs
            tol: Tolerance as fraction of peak response (default: 5%)

        Returns:
            Tuple of (passed, detailed_results)

        Notes:
            Criterion 4 from memory.md
        """
        # Merge on variable, shock, period
        merged = irfs_computed.merge(
            irfs_reference,
            on=['variable', 'shock', 'period'],
            suffixes=('_comp', '_ref')
        )

        # Compute RMSE for each variable-shock pair
        results = []
        for (var, shock), group in merged.groupby(['variable', 'shock']):
            rmse = np.sqrt(np.mean((group['value_comp'] - group['value_ref'])**2))
            peak = np.max(np.abs(group['value_ref']))
            rel_rmse = rmse / peak if peak > 0 else np.inf

            passed = rel_rmse < tol

            results.append({
                'variable': var,
                'shock': shock,
                'rmse': rmse,
                'peak': peak,
                'rel_rmse': rel_rmse,
                'passed': passed
            })

        results_df = pd.DataFrame(results)

        print(f"\nIRF Verification (tolerance: {tol*100}% of peak):")
        print("-" * 80)
        print(results_df.to_string(index=False))
        print("-" * 80)

        n_passed = results_df['passed'].sum()
        n_total = len(results_df)
        print(f"Passed: {n_passed}/{n_total} ({n_passed/n_total*100:.1f}%)")

        all_passed = results_df['passed'].all()

        return all_passed, results_df

    def verify_bvar_marginal_lik(self, ml_computed: Dict[str, float],
                                ml_reference: Optional[Dict[str, float]] = None,
                                tol_pct: float = 0.01) -> Dict[str, bool]:
        """
        Verify BVAR marginal likelihoods.

        Args:
            ml_computed: Dictionary of computed marginal likelihoods
            ml_reference: Dictionary of reference values (default: from paper)
            tol_pct: Tolerance as percentage (default: 1%)

        Returns:
            Dictionary with verification status for each BVAR

        Notes:
            Criterion 5 from memory.md
        """
        if ml_reference is None:
            ml_reference = self.reference_bvar_mlik

        results = {}
        print(f"\nBVAR Marginal Likelihood Verification (tolerance: {tol_pct*100}%):")
        print("-" * 70)
        print(f"{'Model':<12} {'Computed':>15} {'Reference':>15} {'Diff %':>10} {'Status':>8}")
        print("-" * 70)

        for model, ref_val in ml_reference.items():
            if model in ml_computed:
                comp_val = ml_computed[model]
                diff_pct = abs(comp_val - ref_val) / abs(ref_val)
                passed = diff_pct < tol_pct

                results[model] = passed

                status = '✓' if passed else '✗'
                print(f"{model:<12} {comp_val:>15.4f} {ref_val:>15.4f} "
                      f"{diff_pct*100:>9.2f}% {status:>8}")
            else:
                results[model] = False
                print(f"{model:<12} {'N/A':>15} {ref_val:>15.4f} {'N/A':>10} {'✗':>8}")

        print("-" * 70)
        n_passed = sum(results.values())
        n_total = len(results)
        print(f"Passed: {n_passed}/{n_total} ({n_passed/n_total*100:.1f}%)")

        return results

    def verify_forecast_rmse(self, rmse_computed: pd.DataFrame,
                            rmse_reference: pd.DataFrame,
                            tol_pct: float = 0.01) -> Tuple[bool, pd.DataFrame]:
        """
        Verify forecast RMSE match paper Table 4.

        Args:
            rmse_computed: Computed RMSE (variables × horizons)
            rmse_reference: Reference RMSE from paper
            tol_pct: Tolerance as percentage (default: 1%)

        Returns:
            Tuple of (passed, detailed_results)

        Notes:
            Criterion 6 from memory.md
        """
        # Compare each cell
        results = []

        for var in rmse_computed.index:
            for col in rmse_computed.columns:
                if var in rmse_reference.index and col in rmse_reference.columns:
                    comp_val = rmse_computed.loc[var, col]
                    ref_val = rmse_reference.loc[var, col]

                    diff_pct = abs(comp_val - ref_val) / abs(ref_val)
                    passed = diff_pct < tol_pct

                    results.append({
                        'variable': var,
                        'horizon': col,
                        'computed': comp_val,
                        'reference': ref_val,
                        'diff_pct': diff_pct * 100,
                        'passed': passed
                    })

        results_df = pd.DataFrame(results)

        print(f"\nForecast RMSE Verification (tolerance: {tol_pct*100}%):")
        print("-" * 80)
        print(results_df.to_string(index=False))
        print("-" * 80)

        n_passed = results_df['passed'].sum()
        n_total = len(results_df)
        print(f"Passed: {n_passed}/{n_total} ({n_passed/n_total*100:.1f}%)")

        all_passed = results_df['passed'].all()

        return all_passed, results_df

    def run_all_tests(self, results: Dict) -> Dict[str, bool]:
        """
        Run all verification tests.

        Args:
            results: Dictionary with all computed results

        Returns:
            Dictionary with test results

        Expected keys in results:
            - 'likelihood': float
            - 'parameters': Dict[str, float]
            - 'T_matrix': np.ndarray
            - 'irfs': pd.DataFrame
            - 'bvar_mlik': Dict[str, float]
            - 'forecast_rmse': pd.DataFrame
        """
        test_results = {}

        print("\n" + "="*80)
        print("RUNNING ALL VERIFICATION TESTS")
        print("="*80)

        # Test 1: Likelihood
        if 'likelihood' in results:
            test_results['likelihood'] = self.verify_likelihood(results['likelihood'])

        # Test 2: Parameters
        if 'parameters' in results:
            param_results = self.verify_parameters(results['parameters'])
            test_results['parameters'] = all(param_results.values())

        # Test 3: State-space (needs reference)
        if 'T_matrix' in results and 'T_reference' in results:
            test_results['state_space'] = self.verify_state_space(
                results['T_matrix'],
                results['T_reference']
            )

        # Test 4: IRFs (needs reference)
        if 'irfs' in results and 'irfs_reference' in results:
            passed, _ = self.verify_irfs(
                results['irfs'],
                results['irfs_reference']
            )
            test_results['irfs'] = passed

        # Test 5: BVAR marginal likelihoods
        if 'bvar_mlik' in results:
            bvar_results = self.verify_bvar_marginal_lik(results['bvar_mlik'])
            test_results['bvar_mlik'] = all(bvar_results.values())

        # Test 6: Forecast RMSE (needs reference)
        if 'forecast_rmse' in results and 'forecast_rmse_reference' in results:
            passed, _ = self.verify_forecast_rmse(
                results['forecast_rmse'],
                results['forecast_rmse_reference']
            )
            test_results['forecast_rmse'] = passed

        # Summary
        print("\n" + "="*80)
        print("VERIFICATION SUMMARY")
        print("="*80)
        for test, passed in test_results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{test:<25}: {status}")
        print("="*80)

        n_passed = sum(test_results.values())
        n_total = len(test_results)
        print(f"\nOverall: {n_passed}/{n_total} tests passed ({n_passed/n_total*100:.1f}%)")

        return test_results
