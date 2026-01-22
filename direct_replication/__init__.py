"""
Direct Replication Module for Smets & Wouters (2007)

This module provides Python interfaces to call MATLAB/Octave functions
via oct2py, enabling direct replication of the original paper's results.

Requires:
    - GNU Octave (v6.x or higher)
    - Dynare (v5.x or higher)
    - Sims VARtools (varprior.m, rfvar3.m, matrictint.m)

See setup_instructions.md for installation details.
"""

from .dynare_interface import DynareInterface
from .sims_bvar_interface import SimsBVAR
from .results_extractor import (
    extract_state_space_matrices,
    extract_parameter_estimates,
    convert_dr_ordering
)
from .verification import ReplicationVerification

__all__ = [
    'DynareInterface',
    'SimsBVAR',
    'extract_state_space_matrices',
    'extract_parameter_estimates',
    'convert_dr_ordering',
    'ReplicationVerification'
]

__version__ = '0.1.0'
