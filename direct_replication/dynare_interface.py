"""
Dynare Interface Module

Provides a Python wrapper around Dynare using oct2py for calling
Dynare models and extracting results.
"""

import numpy as np
import pandas as pd
from oct2py import Oct2Py
from typing import Dict, Tuple, Optional, List
import os


class DynareInterface:
    """
    Wrapper for calling Dynare via oct2py.

    This class manages the Octave session, adds necessary paths,
    runs Dynare models, and extracts results from Dynare's output
    structures (oo_, M_, options_).

    Attributes:
        oc: Oct2Py instance
        dynare_path: Path to Dynare's matlab folder
        model_path: Path to folder containing .mod files
        model_name: Name of the loaded model
    """

    def __init__(self, dynare_path: str, model_path: str):
        """
        Initialize the Dynare interface.

        Args:
            dynare_path: Path to Dynare's matlab folder (e.g., 'C:/dynare/6.2/matlab')
            model_path: Path to folder containing .mod files
        """
        self.oc = Oct2Py()
        self.dynare_path = dynare_path
        self.model_path = model_path
        self.model_name = None

        # Add paths to Octave
        self.oc.addpath(self.dynare_path)
        self.oc.addpath(self.model_path)

        # Verify Dynare is accessible
        dynare_exists = self.oc.eval('exist("dynare")', nout=1)
        if dynare_exists != 2:
            raise RuntimeError(
                f"Dynare not found. Verify path: {self.dynare_path}"
            )

    def run_model(self, mod_file: str, options: Optional[Dict] = None) -> None:
        """
        Run Dynare on a .mod file.

        Args:
            mod_file: Name of .mod file (without path, e.g., 'usmodel.mod')
            options: Dictionary of Dynare options (e.g., {'nograph': True})

        Example:
            >>> di = DynareInterface(dynare_path, model_path)
            >>> di.run_model('usmodel.mod')
        """
        # Extract model name (without .mod extension)
        if mod_file.endswith('.mod'):
            self.model_name = mod_file[:-4]
        else:
            self.model_name = mod_file
            mod_file = f"{mod_file}.mod"

        # Verify .mod file exists
        full_path = os.path.join(self.model_path, mod_file)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model file not found: {full_path}")

        # Build Dynare command
        cmd = f"dynare {self.model_name}"

        # Add options
        if options:
            for key, value in options.items():
                if value is True:
                    cmd += f" {key}"
                elif value is not False:
                    cmd += f" {key}={value}"
        else:
            # Default: no graph output (cleaner for notebook use)
            cmd += " nograph"

        # Run Dynare
        print(f"Running: {cmd}")
        try:
            self.oc.eval(cmd, verbose=True)
            print(f"Dynare completed successfully for model: {self.model_name}")
        except Exception as e:
            print(f"Error running Dynare: {e}")
            raise

    def get_state_space(self) -> Dict[str, np.ndarray]:
        """
        Extract state-space representation from Dynare.

        Returns:
            Dictionary with keys:
                - 'T': State transition matrix (ghx)
                - 'R': Shock impact matrix (ghu)
                - 'steady_state': Steady state values (ys)
                - 'state_vars': Names of state variables
                - 'obs_vars': Names of observable variables

        Notes:
            - T matrix: oo_.dr.ghx (n_states × n_states)
            - R matrix: oo_.dr.ghu (n_states × n_shocks)
            - These are in DR ordering (decision rule ordering)
        """
        if self.model_name is None:
            raise RuntimeError("No model loaded. Run run_model() first.")

        # Extract matrices from oo_.dr
        T = self.oc.pull('oo_.dr.ghx')
        R = self.oc.pull('oo_.dr.ghu')
        steady_state = self.oc.pull('oo_.dr.ys')

        # Get variable names
        state_var_list = self.oc.pull('M_.endo_names')

        result = {
            'T': T,
            'R': R,
            'steady_state': steady_state,
            'state_vars': state_var_list
        }

        return result

    def get_irfs(self, periods: int = 20, variables: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Extract impulse response functions from Dynare.

        Args:
            periods: Number of periods (default: 20)
            variables: List of variable names to extract (default: all)

        Returns:
            DataFrame with columns: variable, shock, period, value

        Notes:
            IRFs are stored in oo_.irfs as a structure with fields
            like 'dy_ea' (response of dy to shock ea).
        """
        if self.model_name is None:
            raise RuntimeError("No model loaded. Run run_model() first.")

        # Get all IRF field names
        irf_names = self.oc.eval('fieldnames(oo_.irfs)', nout=1)

        # Parse IRF data
        irf_data = []
        for name in irf_names:
            # IRF names are in format: variable_shock
            parts = name.split('_')
            if len(parts) >= 2:
                shock = parts[-1]
                variable = '_'.join(parts[:-1])

                # Filter if needed
                if variables is not None and variable not in variables:
                    continue

                # Extract values
                values = self.oc.pull(f'oo_.irfs.{name}')

                # Add to data (limit to requested periods)
                for t, val in enumerate(values[:periods]):
                    irf_data.append({
                        'variable': variable,
                        'shock': shock,
                        'period': t,
                        'value': val
                    })

        return pd.DataFrame(irf_data)

    def get_parameters(self) -> pd.DataFrame:
        """
        Extract estimated parameters from Dynare.

        Returns:
            DataFrame with columns: parameter, value, prior_mean, prior_std

        Notes:
            - M_.params contains parameter values
            - M_.param_names contains parameter names
            - bayestopt_.p2 contains prior means (if available)
        """
        if self.model_name is None:
            raise RuntimeError("No model loaded. Run run_model() first.")

        # Get parameter names and values
        param_names = self.oc.pull('M_.param_names')
        param_values = self.oc.pull('M_.params')

        # Create DataFrame
        df = pd.DataFrame({
            'parameter': param_names,
            'value': param_values
        })

        # Try to get prior information if available
        try:
            prior_means = self.oc.pull('bayestopt_.p2')
            prior_stds = self.oc.pull('bayestopt_.p3')
            df['prior_mean'] = prior_means
            df['prior_std'] = prior_stds
        except:
            # Priors not available (not an estimation run)
            pass

        return df

    def get_likelihood(self) -> float:
        """
        Extract log-likelihood at posterior mode.

        Returns:
            Log-likelihood value

        Notes:
            This is only available after estimation.
            The value is stored in oo_.posterior.optimization.mode.f
        """
        if self.model_name is None:
            raise RuntimeError("No model loaded. Run run_model() first.")

        try:
            # Try new structure (Dynare 5+)
            loglik = self.oc.pull('oo_.posterior.optimization.mode.f')
        except:
            try:
                # Try old structure
                loglik = self.oc.pull('oo_.MarginalDensity.ModifiedHarmonicMean')
            except:
                raise RuntimeError(
                    "Log-likelihood not found. Model may not have been estimated."
                )

        return float(loglik)

    def get_variance_decomposition(self) -> pd.DataFrame:
        """
        Extract variance decomposition from Dynare.

        Returns:
            DataFrame with variance decomposition for each variable
        """
        if self.model_name is None:
            raise RuntimeError("No model loaded. Run run_model() first.")

        # Get variance decomposition
        var_decomp = self.oc.pull('oo_.variance_decomposition')
        var_names = self.oc.pull('M_.endo_names')
        shock_names = self.oc.pull('M_.exo_names')

        # Convert to DataFrame
        data = []
        for i, var in enumerate(var_names):
            for j, shock in enumerate(shock_names):
                data.append({
                    'variable': var,
                    'shock': shock,
                    'variance_share': var_decomp[i, j]
                })

        return pd.DataFrame(data)

    def get_moments(self) -> Dict[str, pd.DataFrame]:
        """
        Extract theoretical moments from Dynare.

        Returns:
            Dictionary with:
                - 'mean': Mean values
                - 'std': Standard deviations
                - 'variance': Variances
                - 'correlations': Correlation matrix
        """
        if self.model_name is None:
            raise RuntimeError("No model loaded. Run run_model() first.")

        var_names = self.oc.pull('M_.endo_names')

        result = {}

        # Try to extract available moments
        try:
            mean = self.oc.pull('oo_.mean')
            result['mean'] = pd.Series(mean, index=var_names)
        except:
            pass

        try:
            std = self.oc.pull('oo_.var')
            result['std'] = pd.Series(np.sqrt(np.diag(std)), index=var_names)
        except:
            pass

        try:
            corr = self.oc.pull('oo_.autocorr')
            result['correlations'] = pd.DataFrame(corr, index=var_names, columns=var_names)
        except:
            pass

        return result

    def close(self):
        """Close the Octave session."""
        if hasattr(self, 'oc'):
            self.oc.exit()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
