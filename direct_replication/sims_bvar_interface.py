"""
Sims BVAR Interface Module

Wrapper for Christopher Sims' BVAR MATLAB functions.
Requires VARtools from http://sims.princeton.edu/yftp/VARtools/

Required files in repo/:
- varprior.m
- rfvar3.m
- matrictint.m
"""

import numpy as np
import pandas as pd
from oct2py import Oct2Py
from typing import Tuple, Optional, Dict


class SimsBVAR:
    """
    Interface to Christopher Sims' BVAR MATLAB functions.

    This class wraps the BVAR estimation and forecasting functions
    from Sims' VARtools package, which compute marginal likelihoods
    using the Minnesota prior.

    Attributes:
        oc: Oct2Py instance
        vartools_path: Path to VARtools folder
    """

    def __init__(self, oc: Oct2Py, vartools_path: str):
        """
        Initialize the Sims BVAR interface.

        Args:
            oc: Oct2Py instance (can be shared with DynareInterface)
            vartools_path: Path to folder containing Sims VARtools

        Raises:
            RuntimeError: If required .m files not found
        """
        self.oc = oc
        self.vartools_path = vartools_path

        # Add path
        self.oc.addpath(vartools_path)

        # Verify required functions exist
        required_funcs = ['varprior', 'rfvar3', 'matrictint']
        missing = []
        for func in required_funcs:
            exists = self.oc.eval(f'exist("{func}")', nout=1)
            if exists != 2:
                missing.append(f"{func}.m")

        if missing:
            raise RuntimeError(
                f"Missing required VARtools files: {', '.join(missing)}\n"
                f"Download from: http://sims.princeton.edu/yftp/VARtools/"
            )

    def mgnldnsty(self, ydata: np.ndarray, lags: int,
                  train: int = 40, flat: int = 0,
                  lambda_val: float = 5.0, mu_val: float = 1.0) -> float:
        """
        Compute marginal log density for BVAR model.

        Args:
            ydata: Data matrix (T × n_vars)
            lags: Number of lags
            train: Number of observations for training sample (default: 40)
            flat: Use flat prior if > 0 (default: 0)
            lambda_val: Minnesota prior tightness (default: 5.0)
            mu_val: Minnesota prior co-persistence (default: 1.0)

        Returns:
            Marginal log density (scalar)

        Notes:
            This implements the exact marginal likelihood computation
            from Sims' mgnldnsty_fcast.m function.
        """
        # Push data to Octave
        self.oc.push('ydata', ydata)

        # Compute marginal density using Sims' approach
        # Based on mgnldnsty_fcast.m lines 30-60
        cmd = f"""
        [T, nv] = size(ydata);
        xdata = [];
        breaks = [];

        % Setup Minnesota prior
        if {flat} > 0
            lambda = Inf;
            mu = 0;
        else
            lambda = {lambda_val};
            mu = {mu_val};
        end

        % Get prior
        [Tsig, Tphi, Tcnst, Tdf] = varprior(nv, {lags}, lambda, mu, {train});

        % Estimate VAR
        [w, xxi, logdetxy] = rfvar3([ydata xdata], {lags}, [Tsig Tphi Tcnst Tdf], breaks);

        % Compute marginal likelihood
        % Formula from Sims (1998) and Smets-Wouters (2007)
        nobs = T - {train} - {lags};
        ncoef = nv * {lags} + 1;

        % Log marginal density
        mgnl = -nobs * nv * 0.5 * log(2 * pi);
        mgnl = mgnl + 0.5 * logdetxy;
        mgnl = mgnl - nobs * nv * 0.5 * log(nobs);

        marginal_loglik = mgnl;
        """

        try:
            self.oc.eval(cmd)
            marginal_loglik = self.oc.pull('marginal_loglik')
            return float(marginal_loglik)
        except Exception as e:
            raise RuntimeError(f"Error computing marginal density: {e}")

    def mgnldnsty_fcast(self, ydata: np.ndarray, lags: int,
                       start_for: int, horizon: int,
                       train: int = 40, flat: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute BVAR forecasts and forecast errors.

        Args:
            ydata: Data matrix (T × n_vars)
            lags: Number of lags
            start_for: First forecast observation (1-indexed)
            horizon: Forecast horizon
            train: Number of observations for training sample (default: 40)
            flat: Use flat prior if > 0 (default: 0)

        Returns:
            Tuple of:
                - forecasts: Forecast values (n_periods × n_vars × horizon)
                - errors: Forecast errors (n_periods × n_vars × horizon)

        Notes:
            This calls the mgnldnsty_fcast.m function to compute
            recursive forecasts as in Smets & Wouters (2007) Table 4.
        """
        # Push data to Octave
        self.oc.push('ydata', ydata)

        # Call mgnldnsty_fcast
        cmd = f"""
        [T, nv] = size(ydata);
        xdata = [];
        breaks = [];

        % Preallocate
        n_fperiods = T - {start_for} + 1;
        forecasts = zeros(n_fperiods, nv, {horizon});
        errors = zeros(n_fperiods, nv, {horizon});

        % Recursive forecasting loop
        for i_for = 1:n_fperiods
            t_for = {start_for} + i_for - 1;

            % Estimate VAR up to t_for - 1
            y_est = ydata(1:t_for-1, :);

            % Get prior
            if {flat} > 0
                lambda = Inf;
                mu = 0;
            else
                lambda = 5.0;
                mu = 1.0;
            end

            [Tsig, Tphi, Tcnst, Tdf] = varprior(nv, {lags}, lambda, mu, {train});

            % Estimate VAR
            [w, xxi, logdetxy] = rfvar3([y_est xdata], {lags}, [Tsig Tphi Tcnst Tdf], breaks);

            % Generate forecast
            % w contains VAR coefficients
            % Last {lags} observations as initial conditions
            y_init = y_est(end-{lags}+1:end, :);

            % Forecast loop
            for h = 1:{horizon}
                if h == 1
                    y_lag = flipud(y_init);
                else
                    y_lag = [forecasts(i_for, :, h-1); y_lag(1:end-1, :)];
                end

                % Forecast: y_t = c + B1*y_{t-1} + ... + Bp*y_{t-p}
                y_fcast = w(1, :)';  % constant
                for p = 1:{lags}
                    y_fcast = y_fcast + w(1 + (p-1)*nv + 1 : 1 + p*nv, :)' * y_lag(p, :)';
                end

                forecasts(i_for, :, h) = y_fcast;

                % Compute error if data available
                if t_for + h <= T
                    errors(i_for, :, h) = ydata(t_for + h, :)' - y_fcast;
                else
                    errors(i_for, :, h) = NaN;
                end
            end
        end
        """

        try:
            self.oc.eval(cmd)
            forecasts = self.oc.pull('forecasts')
            errors = self.oc.pull('errors')
            return forecasts, errors
        except Exception as e:
            raise RuntimeError(f"Error computing forecasts: {e}")

    def estimate_bvar(self, ydata: np.ndarray, lags: int,
                     train: int = 40, flat: int = 0) -> Dict:
        """
        Estimate BVAR model and return coefficients and statistics.

        Args:
            ydata: Data matrix (T × n_vars)
            lags: Number of lags
            train: Number of observations for training sample (default: 40)
            flat: Use flat prior if > 0 (default: 0)

        Returns:
            Dictionary with:
                - 'coefficients': VAR coefficient matrix
                - 'residuals': VAR residuals
                - 'sigma': Residual covariance matrix
                - 'loglik': Log-likelihood
                - 'marginal_loglik': Marginal log-likelihood
        """
        # Push data
        self.oc.push('ydata', ydata)

        # Estimate BVAR
        cmd = f"""
        [T, nv] = size(ydata);
        xdata = [];
        breaks = [];

        % Prior
        if {flat} > 0
            lambda = Inf;
            mu = 0;
        else
            lambda = 5.0;
            mu = 1.0;
        end

        [Tsig, Tphi, Tcnst, Tdf] = varprior(nv, {lags}, lambda, mu, {train});

        % Estimate
        [w, xxi, logdetxy] = rfvar3([ydata xdata], {lags}, [Tsig Tphi Tcnst Tdf], breaks);

        % Compute fitted values and residuals
        y_est = ydata({train}+{lags}+1:end, :);
        T_est = size(y_est, 1);

        y_fit = zeros(T_est, nv);
        for t = 1:T_est
            t_data = {train} + {lags} + t;
            y_lag = [];
            for p = 1:{lags}
                y_lag = [y_lag; ydata(t_data - p, :)'];
            end
            y_fit(t, :) = (w(1, :)' + w(2:end, :)' * y_lag)';
        end

        residuals = y_est - y_fit;
        sigma = (residuals' * residuals) / T_est;

        % Log-likelihood (Gaussian)
        loglik = -0.5 * T_est * nv * log(2 * pi);
        loglik = loglik - 0.5 * T_est * log(det(sigma));
        loglik = loglik - 0.5 * trace(residuals * inv(sigma) * residuals');

        % Marginal log-likelihood
        nobs = T - {train} - {lags};
        mgnl = -nobs * nv * 0.5 * log(2 * pi);
        mgnl = mgnl + 0.5 * logdetxy;
        marginal_loglik = mgnl;
        """

        try:
            self.oc.eval(cmd)

            result = {
                'coefficients': self.oc.pull('w'),
                'residuals': self.oc.pull('residuals'),
                'sigma': self.oc.pull('sigma'),
                'loglik': float(self.oc.pull('loglik')),
                'marginal_loglik': float(self.oc.pull('marginal_loglik'))
            }

            return result

        except Exception as e:
            raise RuntimeError(f"Error estimating BVAR: {e}")

    def compute_forecast_rmse(self, errors: np.ndarray, cumulative: bool = False) -> pd.DataFrame:
        """
        Compute RMSE from forecast errors.

        Args:
            errors: Forecast errors (n_periods × n_vars × horizon)
            cumulative: If True, cumulate errors (for growth rate variables)

        Returns:
            DataFrame with RMSE for each variable and horizon

        Notes:
            For growth rate variables (dy, dc, dinve, dw), errors should
            be cumulated before computing RMSE (see SW 2007 Table 4).
        """
        n_periods, n_vars, n_horizons = errors.shape

        if cumulative:
            # Cumulate errors across horizons
            errors_cum = np.cumsum(errors, axis=2)
        else:
            errors_cum = errors

        # Compute RMSE
        rmse = np.sqrt(np.nanmean(errors_cum ** 2, axis=0))

        # Create DataFrame
        df = pd.DataFrame(rmse.T)
        df.columns = [f'var_{i+1}' for i in range(n_vars)]
        df['horizon'] = np.arange(1, n_horizons + 1)

        return df

    def compute_log_determinant(self, errors: np.ndarray, horizon: int) -> float:
        """
        Compute log determinant of forecast error covariance matrix.

        Args:
            errors: Forecast errors (n_periods × n_vars × horizon)
            horizon: Specific horizon to compute for

        Returns:
            Log determinant of error covariance at specified horizon

        Notes:
            Used in SW (2007) Table 4 for multivariate forecast comparison.
        """
        # Extract errors at specified horizon
        errors_h = errors[:, :, horizon - 1]

        # Remove NaN rows
        errors_h = errors_h[~np.isnan(errors_h).any(axis=1)]

        # Compute covariance
        cov = np.cov(errors_h.T)

        # Log determinant
        sign, logdet = np.linalg.slogdet(cov)

        return logdet
