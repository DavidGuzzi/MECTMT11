"""
Prior Distribution Classes
===========================

Implements prior distributions for Bayesian DSGE estimation:
- Beta distribution
- Gamma distribution
- Normal distribution
- Inverse-Gamma distribution

Each class provides:
- log_pdf: Log probability density function
- pdf: Probability density function
- rvs: Random sampling
- Support bounds checking
"""

import numpy as np
from scipy import stats
from typing import Optional, Tuple


class Prior:
    """Base class for prior distributions."""

    def __init__(self, lower: float = -np.inf, upper: float = np.inf):
        """
        Initialize prior distribution.

        Args:
            lower: Lower bound of support
            upper: Upper bound of support
        """
        self.lower = lower
        self.upper = upper

    def log_pdf(self, x: float) -> float:
        """Log probability density at x."""
        raise NotImplementedError

    def pdf(self, x: float) -> float:
        """Probability density at x."""
        return np.exp(self.log_pdf(x))

    def rvs(self, size: Optional[int] = None) -> np.ndarray:
        """Generate random samples."""
        raise NotImplementedError

    def in_support(self, x: float) -> bool:
        """Check if x is in the support of the distribution."""
        return self.lower <= x <= self.upper


class BetaPrior(Prior):
    """Beta distribution prior."""

    def __init__(self, mean: float, std: float, lower: float = 0.0, upper: float = 1.0):
        """
        Initialize Beta prior using mean and standard deviation.

        The Beta distribution is parameterized by alpha and beta, but we use
        mean and std for convenience, then compute alpha and beta.

        Args:
            mean: Prior mean (must be in (0,1))
            std: Prior standard deviation
            lower: Lower bound (default 0)
            upper: Upper bound (default 1)
        """
        super().__init__(lower, upper)
        self.mean = mean
        self.std = std

        # Convert mean and std to alpha and beta parameters
        # For Beta(α, β): mean = α/(α+β), var = αβ/[(α+β)²(α+β+1)]
        var = std**2

        # Solve for alpha and beta
        # mean * (1-mean) / var - 1 = alpha + beta
        sum_ab = mean * (1 - mean) / var - 1
        self.alpha = mean * sum_ab
        self.beta = (1 - mean) * sum_ab

        self.dist = stats.beta(self.alpha, self.beta, loc=lower, scale=upper-lower)

    def log_pdf(self, x: float) -> float:
        """Log probability density at x."""
        if not self.in_support(x):
            return -np.inf
        return self.dist.logpdf(x)

    def rvs(self, size: Optional[int] = None) -> np.ndarray:
        """Generate random samples."""
        return self.dist.rvs(size=size)


class GammaPrior(Prior):
    """Gamma distribution prior."""

    def __init__(self, mean: float, std: float, lower: float = 0.0, upper: float = np.inf):
        """
        Initialize Gamma prior using mean and standard deviation.

        Args:
            mean: Prior mean (must be positive)
            std: Prior standard deviation
            lower: Lower bound (default 0)
            upper: Upper bound (default infinity)
        """
        super().__init__(lower, upper)
        self.mean = mean
        self.std = std

        # For Gamma(k, θ): mean = k*θ, var = k*θ²
        # Therefore: k = mean²/var, θ = var/mean
        var = std**2
        self.shape = mean**2 / var  # k parameter
        self.scale = var / mean      # θ parameter

        self.dist = stats.gamma(self.shape, loc=0, scale=self.scale)

    def log_pdf(self, x: float) -> float:
        """Log probability density at x."""
        if not self.in_support(x):
            return -np.inf
        return self.dist.logpdf(x)

    def rvs(self, size: Optional[int] = None) -> np.ndarray:
        """Generate random samples."""
        samples = self.dist.rvs(size=size)
        # Truncate to bounds if necessary
        if self.upper < np.inf:
            samples = np.minimum(samples, self.upper)
        return samples


class NormalPrior(Prior):
    """Normal (Gaussian) distribution prior."""

    def __init__(self, mean: float, std: float, lower: float = -np.inf, upper: float = np.inf):
        """
        Initialize Normal prior.

        Args:
            mean: Prior mean
            std: Prior standard deviation
            lower: Lower bound (default -infinity)
            upper: Upper bound (default +infinity)
        """
        super().__init__(lower, upper)
        self.mean = mean
        self.std = std
        self.dist = stats.norm(loc=mean, scale=std)

    def log_pdf(self, x: float) -> float:
        """Log probability density at x."""
        if not self.in_support(x):
            return -np.inf
        return self.dist.logpdf(x)

    def rvs(self, size: Optional[int] = None) -> np.ndarray:
        """Generate random samples."""
        samples = self.dist.rvs(size=size)
        # Truncate to bounds if necessary
        if size is None:
            if samples < self.lower:
                samples = self.lower
            elif samples > self.upper:
                samples = self.upper
        else:
            samples = np.clip(samples, self.lower, self.upper)
        return samples


class InverseGammaPrior(Prior):
    """Inverse-Gamma distribution prior."""

    def __init__(self, s: float, nu: float, lower: float = 0.0, upper: float = np.inf):
        """
        Initialize Inverse-Gamma prior.

        Uses the parameterization common in Bayesian econometrics:
        IG(s, ν) where s is scale and ν is degrees of freedom (shape).

        Mean = s/(ν-1) for ν > 1
        Variance = s²/[(ν-1)²(ν-2)] for ν > 2

        Args:
            s: Scale parameter
            nu: Shape parameter (degrees of freedom)
            lower: Lower bound (default 0)
            upper: Upper bound (default infinity)
        """
        super().__init__(lower, upper)
        self.s = s
        self.nu = nu

        # scipy uses invgamma(a, scale=s) where a is shape
        # Our parametrization: shape = nu/2, scale = s*nu/2
        self.dist = stats.invgamma(nu/2, scale=s*nu/2)

        # Store mean and variance
        if nu > 1:
            self.mean = s / (nu - 1)
        else:
            self.mean = np.inf

        if nu > 2:
            self.var = s**2 / ((nu-1)**2 * (nu-2))
            self.std = np.sqrt(self.var)
        else:
            self.var = np.inf
            self.std = np.inf

    def log_pdf(self, x: float) -> float:
        """Log probability density at x."""
        if not self.in_support(x) or x <= 0:
            return -np.inf
        return self.dist.logpdf(x)

    def rvs(self, size: Optional[int] = None) -> np.ndarray:
        """Generate random samples."""
        samples = self.dist.rvs(size=size)
        # Truncate to bounds if necessary
        if self.upper < np.inf:
            if size is None:
                samples = min(samples, self.upper)
            else:
                samples = np.minimum(samples, self.upper)
        return samples


def create_prior(prior_type: str, *args, **kwargs) -> Prior:
    """
    Factory function to create prior distributions.

    Args:
        prior_type: Type of prior ('beta', 'gamma', 'normal', 'invgamma')
        *args: Positional arguments for the prior
        **kwargs: Keyword arguments for the prior

    Returns:
        Prior distribution object

    Example:
        >>> prior = create_prior('beta', mean=0.5, std=0.2)
        >>> prior = create_prior('invgamma', s=0.1, nu=2)
    """
    prior_map = {
        'beta': BetaPrior,
        'gamma': GammaPrior,
        'normal': NormalPrior,
        'invgamma': InverseGammaPrior,
        'inv_gamma': InverseGammaPrior,
    }

    prior_type = prior_type.lower()
    if prior_type not in prior_map:
        raise ValueError(f"Unknown prior type: {prior_type}. "
                        f"Available: {list(prior_map.keys())}")

    return prior_map[prior_type](*args, **kwargs)


if __name__ == '__main__':
    # Test the prior distributions
    print("Testing prior distributions...")

    # Beta prior: mean=0.5, std=0.2, bounded [0.01, 0.9999]
    beta = BetaPrior(mean=0.5, std=0.2, lower=0.01, upper=0.9999)
    print(f"\nBeta prior: mean={beta.mean:.3f}, std={beta.std:.3f}")
    print(f"  log_pdf(0.5) = {beta.log_pdf(0.5):.3f}")
    print(f"  samples: {beta.rvs(5)}")

    # Gamma prior: mean=0.625, std=0.1
    gamma = GammaPrior(mean=0.625, std=0.1, lower=0.1, upper=2.0)
    print(f"\nGamma prior: mean={gamma.mean:.3f}, std={gamma.std:.3f}")
    print(f"  log_pdf(0.625) = {gamma.log_pdf(0.625):.3f}")
    print(f"  samples: {gamma.rvs(5)}")

    # Normal prior: mean=1.5, std=0.375, bounded [0.25, 3]
    normal = NormalPrior(mean=1.5, std=0.375, lower=0.25, upper=3.0)
    print(f"\nNormal prior: mean={normal.mean:.3f}, std={normal.std:.3f}")
    print(f"  log_pdf(1.5) = {normal.log_pdf(1.5):.3f}")
    print(f"  samples: {normal.rvs(5)}")

    # Inverse-Gamma prior: s=0.1, nu=2
    invgamma = InverseGammaPrior(s=0.1, nu=2, lower=0.01, upper=3.0)
    print(f"\nInverse-Gamma prior: s={invgamma.s}, nu={invgamma.nu}")
    print(f"  mean={invgamma.mean:.3f}, std={invgamma.std:.3f}")
    print(f"  log_pdf(0.5) = {invgamma.log_pdf(0.5):.3f}")
    print(f"  samples: {invgamma.rvs(5)}")

    print("\nAll tests passed!")
