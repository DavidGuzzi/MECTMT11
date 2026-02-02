"""
Results Extractor Module

Functions for extracting and converting results from Dynare output structures.
Handles ordering conversions and formatting for Python analysis.
"""

import numpy as np
import pandas as pd
from oct2py import Oct2Py
from typing import Dict, Tuple, List


def extract_state_space_matrices(oc: Oct2Py) -> Dict[str, np.ndarray]:
    """
    Extract state-space representation from Dynare output.

    Args:
        oc: Oct2Py instance with loaded Dynare results

    Returns:
        Dictionary with keys:
            - 'T': State transition matrix (n_states x n_states)
            - 'R': Shock impact matrix (n_states x n_shocks)
            - 'Z': Measurement matrix (n_obs x n_states)
            - 'C': Constant vector (n_states,)
            - 'D': Measurement constant (n_obs,)
            - 'steady_state': Steady state values
            - 'state_var_names': List of state variable names
            - 'obs_var_names': List of observable variable names
            - 'shock_names': List of shock names

    Notes:
        - Dynare stores results in oo_.dr structure
        - ghx: State transition matrix (DR ordering)
        - ghu: Shock impact matrix (DR ordering)
        - Use convert_dr_ordering() to get declaration order
    """
    try:
        # Extract core matrices
        T = oc.pull('oo_.dr.ghx')
        R = oc.pull('oo_.dr.ghu')

        # Extract steady state
        steady_state = oc.pull('oo_.dr.ys')

        # Get variable names
        state_var_names = oc.pull('M_.endo_names')
        shock_names = oc.pull('M_.exo_names')

        # Try to get observables
        try:
            obs_var_names = oc.pull('options_.varobs')
        except:
            obs_var_names = None

        # Try to build measurement matrix Z
        Z = None
        if obs_var_names is not None:
            n_obs = len(obs_var_names)
            n_states = len(state_var_names)
            Z = np.zeros((n_obs, n_states))

            # Map observables to states
            for i, obs_var in enumerate(obs_var_names):
                # Find index in state variables
                try:
                    idx = list(state_var_names).index(obs_var)
                    Z[i, idx] = 1.0
                except ValueError:
                    # Observable not in state (might be transformed)
                    pass

        # Constants (usually zero in linearized models)
        C = np.zeros(T.shape[0])
        D = np.zeros(len(obs_var_names)) if obs_var_names else None

        result = {
            'T': T,
            'R': R,
            'Z': Z,
            'C': C,
            'D': D,
            'steady_state': steady_state,
            'state_var_names': state_var_names,
            'obs_var_names': obs_var_names,
            'shock_names': shock_names
        }

        return result

    except Exception as e:
        raise RuntimeError(f"Error extracting state-space matrices: {e}")


def extract_parameter_estimates(oc: Oct2Py) -> pd.DataFrame:
    """
    Extract parameter estimates and standard errors from Dynare estimation.

    Args:
        oc: Oct2Py instance with loaded Dynare estimation results

    Returns:
        DataFrame with columns:
            - parameter: Parameter name
            - mode: Posterior mode value
            - std_error: Standard error at mode
            - prior_mean: Prior mean
            - prior_std: Prior standard deviation
            - prior_type: Prior distribution type

    Notes:
        Information is extracted from:
        - M_.params: Parameter values
        - M_.param_names: Parameter names
        - bayestopt_: Prior information
        - oo_.posterior: Posterior results
    """
    try:
        # Get parameter names and values
        param_names = oc.pull('M_.param_names')
        param_values = oc.pull('M_.params')

        # Initialize DataFrame
        df = pd.DataFrame({
            'parameter': param_names,
            'mode': param_values
        })

        # Try to extract prior information
        try:
            prior_means = oc.pull('bayestopt_.p2')
            prior_stds = oc.pull('bayestopt_.p3')
            prior_types = oc.pull('bayestopt_.pshape')

            df['prior_mean'] = prior_means
            df['prior_std'] = prior_stds
            df['prior_type_code'] = prior_types

            # Map prior type codes to names
            prior_type_map = {
                1: 'beta',
                2: 'gamma',
                3: 'normal',
                4: 'inv_gamma',
                5: 'uniform',
                6: 'inv_gamma2'
            }
            df['prior_type'] = df['prior_type_code'].map(prior_type_map)

        except:
            pass

        # Try to extract standard errors from Hessian
        try:
            hess_inv = oc.pull('oo_.posterior.optimization.mode.hess_inv')
            std_errors = np.sqrt(np.diag(hess_inv))
            df['std_error'] = std_errors
        except:
            pass

        # Try to extract t-statistics
        if 'mode' in df.columns and 'std_error' in df.columns:
            df['t_stat'] = df['mode'] / df['std_error']

        return df

    except Exception as e:
        raise RuntimeError(f"Error extracting parameter estimates: {e}")


def convert_dr_ordering_to_declaration(oc: Oct2Py, matrix: np.ndarray,
                                       axis: int = 0) -> np.ndarray:
    """
    Convert matrix from Dynare's DR ordering to declaration ordering.

    Args:
        oc: Oct2Py instance with loaded Dynare results
        matrix: Matrix to convert (can be vector or 2D matrix)
        axis: Axis to reorder (0 for rows, 1 for columns)

    Returns:
        Reordered matrix

    Notes:
        Dynare's DR ordering groups variables as:
        1. Static variables
        2. Predetermined variables
        3. Mixed variables
        4. Forward-looking variables

        Declaration ordering is the order variables appear in the .mod file.

        Use oo_.dr.order_var for mapping.
    """
    try:
        # Get ordering vector
        order_var = oc.pull('oo_.dr.order_var').astype(int)

        # Adjust for 0-based indexing
        order_var = order_var - 1

        # Create inverse permutation
        inv_order = np.argsort(order_var)

        # Apply reordering
        if matrix.ndim == 1:
            return matrix[inv_order]
        elif matrix.ndim == 2:
            if axis == 0:
                return matrix[inv_order, :]
            else:
                return matrix[:, inv_order]
        else:
            raise ValueError(f"Matrix dimension {matrix.ndim} not supported")

    except Exception as e:
        raise RuntimeError(f"Error converting DR ordering: {e}")


def extract_forecast_results(oc: Oct2Py, horizon: int) -> Dict[str, np.ndarray]:
    """
    Extract forecast results from Dynare.

    Args:
        oc: Oct2Py instance with loaded Dynare forecast results
        horizon: Forecast horizon

    Returns:
        Dictionary with:
            - 'forecasts': Forecast values (n_vars x horizon)
            - 'forecast_vars': Variable names
            - 'forecast_periods': Period labels
    """
    try:
        # Get forecast from oo_.forecast
        forecast_vars = oc.pull('M_.endo_names')

        # Extract forecasts
        forecasts = []
        for var in forecast_vars:
            try:
                var_forecast = oc.pull(f'oo_.forecast.Mean.{var}')
                forecasts.append(var_forecast[:horizon])
            except:
                # Variable not forecasted
                forecasts.append(np.full(horizon, np.nan))

        forecasts = np.array(forecasts)

        result = {
            'forecasts': forecasts,
            'forecast_vars': forecast_vars,
            'forecast_periods': np.arange(1, horizon + 1)
        }

        return result

    except Exception as e:
        raise RuntimeError(f"Error extracting forecast results: {e}")


def extract_shock_decomposition(oc: Oct2Py) -> pd.DataFrame:
    """
    Extract historical shock decomposition from Dynare.

    Args:
        oc: Oct2Py instance with loaded Dynare results

    Returns:
        DataFrame with columns: variable, shock, period, contribution

    Notes:
        Requires that shock_decomposition option was used in Dynare.
    """
    try:
        # Get shock decomposition
        shock_decomp = oc.pull('oo_.shock_decomposition')

        var_names = oc.pull('M_.endo_names')
        shock_names = oc.pull('M_.exo_names')

        # shock_decomp is (n_vars x n_shocks x n_periods)
        n_vars, n_shocks, n_periods = shock_decomp.shape

        # Convert to long format
        data = []
        for i, var in enumerate(var_names):
            for j, shock in enumerate(shock_names):
                for t in range(n_periods):
                    data.append({
                        'variable': var,
                        'shock': shock,
                        'period': t + 1,
                        'contribution': shock_decomp[i, j, t]
                    })

        return pd.DataFrame(data)

    except Exception as e:
        raise RuntimeError(f"Error extracting shock decomposition: {e}")


def extract_smoother_results(oc: Oct2Py) -> Dict[str, np.ndarray]:
    """
    Extract Kalman smoother results from Dynare.

    Args:
        oc: Oct2Py instance with loaded Dynare smoother results

    Returns:
        Dictionary with:
            - 'smoothed_states': Smoothed state estimates
            - 'smoothed_shocks': Smoothed shock estimates
            - 'smoothed_vars': Variable names
            - 'shock_names': Shock names
    """
    try:
        # Get smoothed variables
        smoothed_vars = oc.pull('oo_.SmoothedVariables')
        smoothed_shocks = oc.pull('oo_.SmoothedShocks')

        var_names = oc.pull('M_.endo_names')
        shock_names = oc.pull('M_.exo_names')

        # Extract values
        smoothed_states = []
        for var in var_names:
            try:
                vals = oc.pull(f'oo_.SmoothedVariables.{var}')
                smoothed_states.append(vals)
            except:
                smoothed_states.append(None)

        smoothed_shock_vals = []
        for shock in shock_names:
            try:
                vals = oc.pull(f'oo_.SmoothedShocks.{shock}')
                smoothed_shock_vals.append(vals)
            except:
                smoothed_shock_vals.append(None)

        result = {
            'smoothed_states': np.array(smoothed_states),
            'smoothed_shocks': np.array(smoothed_shock_vals),
            'smoothed_vars': var_names,
            'shock_names': shock_names
        }

        return result

    except Exception as e:
        raise RuntimeError(f"Error extracting smoother results: {e}")


def get_model_info(oc: Oct2Py) -> Dict:
    """
    Extract general model information from Dynare.

    Args:
        oc: Oct2Py instance with loaded Dynare results

    Returns:
        Dictionary with model metadata
    """
    try:
        info = {}

        # Basic counts
        info['n_endo'] = int(oc.pull('M_.endo_nbr'))
        info['n_exo'] = int(oc.pull('M_.exo_nbr'))
        info['n_params'] = int(oc.pull('M_.param_nbr'))

        # Variable names
        info['endo_names'] = oc.pull('M_.endo_names')
        info['exo_names'] = oc.pull('M_.exo_names')
        info['param_names'] = oc.pull('M_.param_names')

        # Try to get observable info
        try:
            info['obs_names'] = oc.pull('options_.varobs')
            info['n_obs'] = len(info['obs_names'])
        except:
            info['obs_names'] = None
            info['n_obs'] = 0

        # Model name
        try:
            info['model_name'] = oc.pull('M_.fname')
        except:
            info['model_name'] = 'unknown'

        return info

    except Exception as e:
        raise RuntimeError(f"Error extracting model info: {e}")
