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

        # CRITICAL: Close Octave FIRST to release file locks
        print("\n" + "="*60)
        print("Step 1: Closing Octave session to release file locks...")
        print("="*60)
        try:
            self.oc.exit()
        except:
            pass

        # Wait for Windows to release file handles
        import time
        print("Waiting for Windows to release file handles...")
        time.sleep(2.0)

        # Clean up temporary and model directories AFTER closing Octave
        print("\n" + "="*60)
        print("Step 2: Cleaning up directories...")
        print("="*60)
        self._cleanup_directories(model_name=self.model_name)

        # CRITICAL: Save current directory
        original_dir = os.getcwd()

        try:
            # Create fresh Octave session
            print("\n" + "="*60)
            print("Step 3: Starting fresh Octave session...")
            print("="*60)
            self.oc = Oct2Py()
            self.oc.addpath(self.dynare_path)
            self.oc.addpath(self.model_path)
            print("Octave session ready")

            # Change to model directory (REQUIRED for Dynare)
            os.chdir(self.model_path)
            self.oc.eval(f"cd('{self.model_path}')", verbose=False)

            # Run Dynare
            print("\n" + "="*60)
            print("Step 4: Running Dynare estimation...")
            print("="*60)
            print(f"Command: {cmd}")
            print("(This may take several minutes...)\n")
            self.oc.eval(cmd, verbose=True)
            print(f"Dynare completed successfully for model: {self.model_name}")

        except Exception as e:
            print(f"Error running Dynare: {e}")
            raise
        finally:
            # CRITICAL: Always restore original directory
            os.chdir(original_dir)
            self.oc.eval(f"cd('{original_dir}')", verbose=False)
            print(f"Restored working directory to: {original_dir}")

    def _cleanup_directories(self, model_name: Optional[str] = None):
        """
        Clean up temporary and model directories (Windows file locking workaround).

        Removes:
        1. Temporary directories (10-char alphanumeric names)
        2. Model output directory (if model_name provided)

        Uses 3-tier strategy to handle Windows file locks:
        - Strategy 1: shutil.rmtree with retries (5 attempts, 1s delay)
        - Strategy 2: Windows 'rmdir /S /Q' command
        - Strategy 3: Rename directory (allows Dynare to create fresh one)

        Args:
            model_name: Optional model name (e.g., 'usmodel'). If provided,
                       cleans the <model_name>/ directory containing bytecode files.
        """
        import shutil
        from pathlib import Path
        import time
        import subprocess

        model_dir = Path(self.model_path)

        print("Searching for directories to clean up...")

        # Find temporary directories (10-character alphanumeric names)
        dirs_to_clean = [item for item in model_dir.iterdir()
                         if item.is_dir() and len(item.name) == 10 and item.name.isalnum()]

        # If model_name provided, add model directory
        if model_name:
            model_output_dir = model_dir / model_name
            if model_output_dir.exists() and model_output_dir.is_dir():
                dirs_to_clean.append(model_output_dir)
                print(f"Adding model directory to cleanup: {model_name}/")

        if not dirs_to_clean:
            print("No directories found to clean up.")
            return

        print(f"Found {len(dirs_to_clean)} directory(ies): {[d.name for d in dirs_to_clean]}")

        for item in dirs_to_clean:
            # Strategy 1: Normal shutil.rmtree with retries
            success = False
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    shutil.rmtree(item)
                    print(f"Cleaned up directory: {item.name}")
                    success = True
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        print(f"  Retry {attempt + 1}/{max_retries} for {item.name}...")
                        time.sleep(1.0)  # Wait longer between retries
                    else:
                        print(f"  Failed to remove {item.name} with shutil, trying Windows command...")

            # Strategy 2: Use Windows rmdir command as fallback
            if not success:
                try:
                    subprocess.run(['cmd', '/c', 'rmdir', '/S', '/Q', str(item)],
                                   capture_output=True, timeout=5)
                    if not item.exists():
                        print(f"Cleaned up directory with rmdir: {item.name}")
                        success = True
                except Exception:
                    pass

            # Strategy 3: Rename directory (so Dynare can create new one)
            if not success:
                try:
                    new_name = item.parent / f"_old_{item.name}"
                    item.rename(new_name)
                    print(f"Renamed locked directory: {item.name} -> {new_name.name}")
                except Exception:
                    print(f"Could not remove or rename {item.name} (will try to continue anyway)")

        # Final wait to ensure Windows releases all handles
        time.sleep(1.0)

    def get_state_space(self) -> Dict[str, np.ndarray]:
        """
        Extract state-space representation from Dynare.

        Returns:
            Dictionary with keys:
                - 'T': State transition matrix (ghx)
                - 'R': Shock impact matrix (ghu)
                - 'steady_state': Steady state values (ys)
                - 'state_vars': Names of state variables
                - 'obs_vars': Names of observable variables (None if not defined)
                - 'shock_names': Names of shocks
                - 'Z': Measurement matrix (n_obs x n_states, None if obs_vars not defined)

        Notes:
            - T matrix: oo_.dr.ghx (n_states x n_states)
            - R matrix: oo_.dr.ghu (n_states x n_shocks)
            - Z matrix: Constructed by mapping observable variables to state variables
              (Z[i,j] = 1.0 if obs_vars[i] == state_vars[j], else 0.0)
            - shock_names: Extracted from M_.exo_names
            - obs_vars: Extracted from options_.varobs (if varobs declared in .mod file)
            - These are in DR ordering (decision rule ordering)
        """
        if self.model_name is None:
            raise RuntimeError("No model loaded. Run run_model() first.")

        # Extract matrices - use eval() for struct fields
        try:
            T = self.oc.pull('oo_.dr.ghx')
            if not isinstance(T, np.ndarray):
                T = self.oc.eval('oo_.dr.ghx', nout=1)
        except:
            T = self.oc.eval('oo_.dr.ghx', nout=1)

        try:
            R = self.oc.pull('oo_.dr.ghu')
            if not isinstance(R, np.ndarray):
                R = self.oc.eval('oo_.dr.ghu', nout=1)
        except:
            R = self.oc.eval('oo_.dr.ghu', nout=1)

        steady_state = self.oc.pull('oo_.dr.ys')

        # Get variable names - Cell array conversion
        n_endo = int(self.oc.eval('M_.endo_nbr', nout=1))
        state_var_list = []
        for i in range(n_endo):
            var_name = self.oc.eval(f'deblank(M_.endo_names{{{i+1}}})', nout=1)
            state_var_list.append(str(var_name).strip())

        # Get shock names - Cell array conversion
        n_shocks = int(self.oc.eval('M_.exo_nbr', nout=1))
        shock_names = []
        for i in range(n_shocks):
            shock_name = self.oc.eval(f'deblank(M_.exo_names{{{i+1}}})', nout=1)
            shock_names.append(str(shock_name).strip())

        # Get observable variables (if available) - Cell array conversion
        obs_vars = None
        try:
            n_obs = int(self.oc.eval('length(options_.varobs)', nout=1))
            if n_obs > 0:
                obs_vars = []
                for i in range(n_obs):
                    obs_name = self.oc.eval(f'deblank(options_.varobs{{{i+1}}})', nout=1)
                    obs_vars.append(str(obs_name).strip())
        except:
            # Observable variables not defined in model
            pass

        # Build measurement matrix Z
        Z = None
        if obs_vars is not None:
            n_obs_vars = len(obs_vars)
            n_states = T.shape[0]
            Z = np.zeros((n_obs_vars, n_states))

            # Map observables to states
            for i, obs_var in enumerate(obs_vars):
                try:
                    idx = state_var_list.index(obs_var)
                    Z[i, idx] = 1.0
                except ValueError:
                    # Observable not in state (e.g., growth variables)
                    pass

        result = {
            'T': T,
            'R': R,
            'steady_state': steady_state,
            'state_vars': state_var_list,
            'obs_vars': obs_vars,
            'shock_names': shock_names,
            'Z': Z
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

        # Check if IRFs exist
        has_irfs = self.oc.eval('isfield(oo_, "irfs")', nout=1)
        if not has_irfs:
            raise RuntimeError("IRFs not found. Run with 'stoch_simul' to generate IRFs.")

        # Get IRF field names - Cell array conversion
        n_irfs = int(self.oc.eval('length(fieldnames(oo_.irfs))', nout=1))
        if n_irfs == 0:
            raise RuntimeError("No IRFs available.")

        irf_names = []
        for i in range(n_irfs):
            name = self.oc.eval(f'deblank(fieldnames(oo_.irfs){{{i+1}}})', nout=1)
            irf_names.append(str(name).strip())

        # Parse IRF data
        irf_data = []
        for name in irf_names:
            if '_' not in name:
                continue

            # Use rsplit to handle variable names with underscores
            parts = name.rsplit('_', 1)
            if len(parts) != 2:
                continue

            variable = parts[0]
            shock = parts[1]

            if variables is not None and variable not in variables:
                continue

            try:
                values = self.oc.eval(f'oo_.irfs.{name}', nout=1)
                if values.ndim == 0:
                    values = np.array([values])
                elif values.ndim > 1:
                    values = values.flatten()

                n_periods = min(len(values), periods)
                for t in range(n_periods):
                    irf_data.append({
                        'variable': variable,
                        'shock': shock,
                        'period': t,
                        'value': float(values[t])
                    })
            except Exception as e:
                print(f"Warning: Could not extract IRF '{name}': {e}")
                continue

        if not irf_data:
            raise RuntimeError("No IRF data extracted.")

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

        # Get parameter values
        param_values = self.oc.eval('M_.params', nout=1)
        if param_values.ndim > 1:
            param_values = param_values.flatten()

        n_params = len(param_values)

        # Get parameter names - Cell array needs special handling
        param_names = []
        for i in range(n_params):
            name = self.oc.eval(f'deblank(M_.param_names{{{i+1}}})', nout=1)
            param_names.append(str(name).strip())

        # Create DataFrame
        df = pd.DataFrame({
            'parameter': param_names,
            'value': param_values
        })

        # Try to get prior information if available
        try:
            exists = self.oc.eval('exist("bayestopt_", "var")', nout=1)
            if exists == 1:
                prior_means = self.oc.eval('bayestopt_.p2', nout=1)
                prior_stds = self.oc.eval('bayestopt_.p3', nout=1)
                if prior_means.ndim > 1:
                    prior_means = prior_means.flatten()
                if prior_stds.ndim > 1:
                    prior_stds = prior_stds.flatten()
                if len(prior_means) == len(df):
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

        # Strategy 1: Modern Dynare structure
        try:
            exists = self.oc.eval(
                'isfield(oo_, "posterior") && '
                'isfield(oo_.posterior, "optimization") && '
                'isfield(oo_.posterior.optimization, "mode") && '
                'isfield(oo_.posterior.optimization.mode, "f")',
                nout=1
            )
            if exists:
                loglik = self.oc.eval('oo_.posterior.optimization.mode.f', nout=1)
                return float(loglik)
        except:
            pass

        # Strategy 2: Laplace approximation (most reliable)
        try:
            exists = self.oc.eval(
                'isfield(oo_, "MarginalDensity") && '
                'isfield(oo_.MarginalDensity, "LaplaceApproximation")',
                nout=1
            )
            if exists:
                loglik = self.oc.eval('oo_.MarginalDensity.LaplaceApproximation', nout=1)
                return float(loglik)
        except:
            pass

        # Strategy 3: ModifiedHarmonicMean
        try:
            exists = self.oc.eval(
                'isfield(oo_, "MarginalDensity") && '
                'isfield(oo_.MarginalDensity, "ModifiedHarmonicMean")',
                nout=1
            )
            if exists:
                loglik = self.oc.eval('oo_.MarginalDensity.ModifiedHarmonicMean', nout=1)
                return float(loglik)
        except:
            pass

        raise RuntimeError(
            "Log-likelihood not found. Model may not have been estimated."
        )

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
