"""
DSGE Model Replication Package
===============================

Python replication of Smets & Wouters (2007) "Shocks and Frictions in US Business Cycles:
A Bayesian DSGE Approach" for application to Argentine macroeconomic data.

Modules:
    - priors: Prior distribution classes
    - utils: General utilities
    - data_loader: Data I/O functions
    - model: DSGE model specification
    - solver: Rational expectations solver (QZ decomposition)
    - kalman: Kalman filter implementation
    - estimation: Bayesian parameter estimation
    - bvar: Bayesian VAR with Minnesota prior
    - forecast: Forecast evaluation tools
"""

__version__ = '0.1.0'
__author__ = 'David Guzzi'

from . import priors
from . import utils
from . import data_loader

__all__ = ['priors', 'utils', 'data_loader']
