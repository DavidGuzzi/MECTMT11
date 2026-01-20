"""
Bayesian Estimation for DSGE Models
====================================

Implements Bayesian posterior mode-finding for DSGE models.

Posterior ∝ Prior × Likelihood
- Prior: Specified distributions on parameters
- Likelihood: Computed via Kalman filter
"""

import numpy as np
from scipy import optimize
from typing import Dict, List, Tuple, Optional, Callable
import warnings
from .model import DSGEModel
from .priors import Prior


class BayesianEstimator:
    """Bayesian estimator for DSGE models."""

    def __init__(self, model: DSGEModel, priors: Dict[str, Prior],
                 data: np.ndarray, param_names: List[str]):
        """
        Initialize Bayesian estimator.

        Args:
            model: DSGE model object
            priors: Dictionary mapping parameter names to Prior objects
            data: Observed data (T x n_obs)
            param_names: List of parameter names to estimate
        """
        self.model = model
        self.priors = priors
        self.data = data
        self.param_names = param_names

        # Extract bounds from priors
        self.bounds = [(priors[p].lower, priors[p].upper) for p in param_names]

        # Results storage
        self.mode = None
        self.hessian = None
        self.vcov = None
        self.log_posterior_mode = None

    def log_prior(self, params: np.ndarray) -> float:
        """
        Compute log prior density.

        Args:
            params: Parameter vector

        Returns:
            Log prior density
        """
        log_p = 0.0

        for i, pname in enumerate(self.param_names):
            if pname in self.priors:
                log_p += self.priors[pname].log_pdf(params[i])
            else:
                warnings.warn(f"No prior specified for {pname}")

        return log_p

    def log_likelihood(self, params: np.ndarray) -> float:
        """
        Compute log likelihood via Kalman filter.

        Args:
            params: Parameter vector

        Returns:
            Log likelihood
        """
        # Update model parameters
        param_dict = {name: val for name, val in zip(self.param_names, params)}
        self.model.set_parameters(param_dict)

        try:
            # Solve model
            self.model.solve()

            # Compute likelihood
            log_lik = self.model.log_likelihood(self.data)

            return log_lik

        except Exception as e:
            # Return large negative value if model fails to solve
            warnings.warn(f"Model solution failed: {e}")
            return -1e10

    def log_posterior(self, params: np.ndarray) -> float:
        """
        Compute log posterior density.

        Args:
            params: Parameter vector

        Returns:
            Log posterior density
        """
        log_prior = self.log_prior(params)

        # If prior is -inf, don't evaluate likelihood
        if not np.isfinite(log_prior):
            return -1e10

        log_lik = self.log_likelihood(params)

        return log_prior + log_lik

    def neg_log_posterior(self, params: np.ndarray) -> float:
        """Negative log posterior for minimization."""
        return -self.log_posterior(params)

    def find_mode(self, initial_params: Optional[np.ndarray] = None,
                 method: str = 'L-BFGS-B',
                 options: Optional[Dict] = None) -> Dict:
        """
        Find posterior mode using numerical optimization.

        Args:
            initial_params: Initial parameter values (uses priors if None)
            method: Optimization method ('L-BFGS-B', 'TNC', 'SLSQP')
            options: Options for scipy.optimize.minimize

        Returns:
            Dictionary with estimation results
        """
        # Initialize parameters
        if initial_params is None:
            # Draw from priors or use prior means
            initial_params = np.array([
                self.priors[p].mean if hasattr(self.priors[p], 'mean')
                else 0.5 * (self.priors[p].lower + self.priors[p].upper)
                for p in self.param_names
            ])

        # Default options
        if options is None:
            options = {
                'maxiter': 1000,
                'disp': True,
                'ftol': 1e-6
            }

        print(f"Starting optimization with {method}...")
        print(f"Initial log posterior: {self.log_posterior(initial_params):.2f}")

        # Optimize
        result = optimize.minimize(
            self.neg_log_posterior,
            initial_params,
            method=method,
            bounds=self.bounds,
            options=options
        )

        # Store results
        self.mode = result.x
        self.log_posterior_mode = -result.fun

        # Compute Hessian at mode (numerical)
        try:
            self.hessian = self._numerical_hessian(self.mode)
            self.vcov = np.linalg.inv(self.hessian)
        except:
            warnings.warn("Failed to compute Hessian/variance-covariance matrix")
            self.hessian = None
            self.vcov = None

        # Create results dictionary
        results = {
            'mode': self.mode,
            'log_posterior': self.log_posterior_mode,
            'log_likelihood': self.log_likelihood(self.mode),
            'log_prior': self.log_prior(self.mode),
            'param_names': self.param_names,
            'hessian': self.hessian,
            'vcov': self.vcov,
            'std_errors': np.sqrt(np.diag(self.vcov)) if self.vcov is not None else None,
            'success': result.success,
            'message': result.message,
            'n_iterations': result.nit,
        }

        return results

    def _numerical_hessian(self, params: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """
        Compute numerical Hessian using finite differences.

        Args:
            params: Parameter vector
            eps: Step size for finite differences

        Returns:
            Hessian matrix
        """
        n = len(params)
        hessian = np.zeros((n, n))

        f0 = self.neg_log_posterior(params)

        for i in range(n):
            params_i_plus = params.copy()
            params_i_plus[i] += eps
            f_i_plus = self.neg_log_posterior(params_i_plus)

            params_i_minus = params.copy()
            params_i_minus[i] -= eps
            f_i_minus = self.neg_log_posterior(params_i_minus)

            hessian[i, i] = (f_i_plus - 2*f0 + f_i_minus) / eps**2

            for j in range(i+1, n):
                params_ij_plus = params.copy()
                params_ij_plus[i] += eps
                params_ij_plus[j] += eps
                f_ij_plus = self.neg_log_posterior(params_ij_plus)

                params_ij_minus = params.copy()
                params_ij_minus[i] -= eps
                params_ij_minus[j] -= eps
                f_ij_minus = self.neg_log_posterior(params_ij_minus)

                hessian[i, j] = (f_ij_plus - f_i_plus - f_i_minus + f_ij_minus) / (4 * eps**2)
                hessian[j, i] = hessian[i, j]

        return hessian

    def print_results(self, results: Dict):
        """Print estimation results in readable format."""
        print("\n" + "="*60)
        print("BAYESIAN ESTIMATION RESULTS")
        print("="*60)

        print(f"\nLog Posterior: {results['log_posterior']:.4f}")
        print(f"Log Likelihood: {results['log_likelihood']:.4f}")
        print(f"Log Prior: {results['log_prior']:.4f}")
        print(f"\nOptimization: {results['message']}")
        print(f"Iterations: {results['n_iterations']}")

        print(f"\nPosterior Mode Estimates:")
        print("-" * 60)
        print(f"{'Parameter':<15} {'Mode':>12} {'Std Error':>12} {'t-stat':>12}")
        print("-" * 60)

        for i, name in enumerate(results['param_names']):
            mode = results['mode'][i]
            if results['std_errors'] is not None:
                se = results['std_errors'][i]
                t_stat = mode / se if se > 0 else np.nan
                print(f"{name:<15} {mode:12.4f} {se:12.4f} {t_stat:12.2f}")
            else:
                print(f"{name:<15} {mode:12.4f} {'N/A':>12} {'N/A':>12}")

        print("-" * 60)


def estimate_dsge(model: DSGEModel, data: np.ndarray,
                 priors: Dict[str, Prior], param_names: List[str],
                 initial_params: Optional[np.ndarray] = None,
                 method: str = 'L-BFGS-B') -> Dict:
    """
    Convenience function to estimate DSGE model.

    Args:
        model: DSGE model object
        data: Observed data (T x n_obs)
        priors: Prior specifications
        param_names: Parameters to estimate
        initial_params: Initial parameter values
        method: Optimization method

    Returns:
        Estimation results dictionary
    """
    estimator = BayesianEstimator(model, priors, data, param_names)
    results = estimator.find_mode(initial_params, method)
    estimator.print_results(results)

    return results


if __name__ == '__main__':
    print("Testing Bayesian estimation framework...")

    # This is a placeholder test
    # Full testing requires a properly specified model
    print("\nEstimation framework created successfully.")
    print("Full testing requires complete model specification.")
    print("See replication.ipynb for usage examples.")
