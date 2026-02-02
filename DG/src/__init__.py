"""
Argentina DSGE Replication Module

Replicacion del modelo Smets & Wouters (2007) con datos de Argentina.
Periodo: 2004Q2 - 2025Q3 (86 observaciones trimestrales)
"""

from .dynare_interface import DynareInterface
from .results_extractor import (
    extract_state_space_matrices,
    extract_parameter_estimates,
    get_model_info
)

__all__ = [
    'DynareInterface',
    'extract_state_space_matrices',
    'extract_parameter_estimates',
    'get_model_info'
]
