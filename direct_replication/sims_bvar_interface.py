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
                  lambda_val: float = 5.0, mu_val: float = 2.0,
                  tau: float = 10.0, d: float = 1.0) -> float:
        """
        Compute marginal log density for BVAR model.

        Args:
            ydata: Data matrix (T × n_vars)
            lags: Number of lags
            train: Number of observations for training sample (default: 40)
            flat: Use flat prior if > 0 (default: 0)
            lambda_val: Weight on co-persistence prior (default: 5.0)
            mu_val: Weight on own-persistence prior (default: 2.0)
            tau: Minnesota prior overall tightness (default: 10.0)
            d: Lag decay parameter (default: 1.0)

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

        % Setup prior weights for rfvar3
        if {flat} > 0
            lambda = Inf;
            mu = 0;
        else
            lambda = {lambda_val};
            mu = {mu_val};
        end

        % Compute prior variance from training sample
        if {train} > 0
            % Use AR(1) on training sample
            y_train = ydata(1:{train}, :);
            y_train_lag = y_train(1:end-1, :);
            y_train_dep = y_train(2:end, :);

            % OLS regression
            coef = (y_train_lag' * y_train_lag) \\ (y_train_lag' * y_train_dep);
            resid = y_train_dep - y_train_lag * coef;
            sig_prior = sqrt(diag(resid' * resid / ({train}-1)));
        else
            % Use sample std dev if no training sample
            sig_prior = std(ydata)';
        end

        % Create Minnesota and variance prior structures
        if {flat} > 0
            mnprior = [];
            vprior = [];
        else
            % Minnesota prior parameters
            mnprior.tight = {tau};
            mnprior.decay = {d};

            % Variance prior parameters
            vprior.sig = sig_prior';
            vprior.w = 1;
        end

        % Call varprior with correct signature
        nx = 0;  % No exogenous variables
        [ydum, xdum, pbreaks] = varprior(nv, nx, {lags}, mnprior, vprior);

        % Estimate VAR with dummy observations as prior
        var = rfvar3([ydata; ydum], {lags}, [xdata; xdum], [breaks; T; T + pbreaks], lambda, mu);

        % Extract results from var structure
        u = var.u;
        xxi = var.xxi;
        Tu = size(u, 1);

        % Compute marginal likelihood using matrictint
        loglik_full = matrictint(u' * u, xxi, Tu - {flat} * (nv + 1)) - {flat} * 0.5 * nv * (nv + 1) * log(2 * pi);

        % If training sample used, compute training likelihood and subtract
        if {train} > 0
            % Estimate on training sample only
            var_train = rfvar3([ydata(1:{train}, :); ydum], {lags}, [xdata; xdum], [breaks; {train}; {train} + pbreaks], lambda, mu);
            u_train = var_train.u;
            Tu_train = size(u_train, 1);

            loglik_train = matrictint(u_train' * u_train, var_train.xxi, Tu_train - {flat} * (nv + 1) / 2) - {flat} * 0.5 * nv * (nv + 1) * log(2 * pi);

            % Marginal likelihood is difference
            marginal_loglik = loglik_full - loglik_train;
        else
            marginal_loglik = loglik_full;
        end
        """

        try:
            self.oc.eval(cmd)
            marginal_loglik = self.oc.pull('marginal_loglik')
            return float(marginal_loglik)
        except Exception as e:
            raise RuntimeError(f"Error computing marginal density: {e}")

    def mgnldnsty_fcast(self, ydata: np.ndarray, lags: int,
                       start_for: int, horizon: int,
                       train: int = 40, flat: int = 0,
                       lambda_val: float = 5.0, mu_val: float = 2.0,
                       tau: float = 10.0, d: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute BVAR forecasts and forecast errors.

        Args:
            ydata: Data matrix (T × n_vars)
            lags: Number of lags
            start_for: First forecast observation (1-indexed)
            horizon: Forecast horizon
            train: Number of observations for training sample (default: 40)
            flat: Use flat prior if > 0 (default: 0)
            lambda_val: Weight on co-persistence prior (default: 5.0)
            mu_val: Weight on own-persistence prior (default: 2.0)
            tau: Minnesota prior overall tightness (default: 10.0)
            d: Lag decay parameter (default: 1.0)

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

        % Compute prior variance from training sample (once)
        if {train} > 0
            y_train = ydata(1:{train}, :);
            y_train_lag = y_train(1:end-1, :);
            y_train_dep = y_train(2:end, :);
            coef = (y_train_lag' * y_train_lag) \\ (y_train_lag' * y_train_dep);
            resid = y_train_dep - y_train_lag * coef;
            sig_prior = sqrt(diag(resid' * resid / ({train}-1)));
        else
            sig_prior = std(ydata)';
        end

        % Create prior structures
        if {flat} > 0
            mnprior = [];
            vprior = [];
            lambda = Inf;
            mu = 0;
        else
            mnprior.tight = {tau};
            mnprior.decay = {d};
            vprior.sig = sig_prior';
            vprior.w = 1;
            lambda = {lambda_val};
            mu = {mu_val};
        end

        % Get dummy observations (same for all forecasts)
        nx = 0;
        [ydum, xdum, pbreaks] = varprior(nv, nx, {lags}, mnprior, vprior);

        % Recursive forecasting loop
        for i_for = 1:n_fperiods
            t_for = {start_for} + i_for - 1;

            % Estimate VAR up to t_for - 1
            y_est = ydata(1:t_for-1, :);

            % Estimate VAR with dummy observations
            var = rfvar3([y_est; ydum], {lags}, [xdata; xdum], [breaks; t_for - 1; t_for - 1 + pbreaks], lambda, mu);

            % Extract coefficients from var structure
            By = var.By;  % Shape: (equations, variables, lags)

            % Generate forecasts
            y_init = y_est(end - {lags} + 1:end, :);

            % Forecast loop
            for h = 1:{horizon}
                % Build lagged variables vector
                if h == 1
                    y_lags = flipud(y_init);  % Most recent first
                else
                    % Use previous forecasts
                    y_lags = [forecasts(i_for, :, h-1); y_lags(1:end-1, :)];
                end

                % Forecast using By coefficients: By(eq, var, lag)
                y_fcast = zeros(1, nv);
                for eq = 1:nv
                    for lag = 1:{lags}
                        for var = 1:nv
                            y_fcast(eq) = y_fcast(eq) + By(eq, var, lag) * y_lags(lag, var);
                        end
                    end
                end

                forecasts(i_for, :, h) = y_fcast;

                % Compute error if data available
                if t_for + h <= T
                    errors(i_for, :, h) = ydata(t_for + h, :) - y_fcast;
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
                     train: int = 40, flat: int = 0,
                     lambda_val: float = 5.0, mu_val: float = 2.0,
                     tau: float = 10.0, d: float = 1.0) -> Dict:
        """
        Estimate BVAR model and return coefficients and statistics.

        Args:
            ydata: Data matrix (T × n_vars)
            lags: Number of lags
            train: Number of observations for training sample (default: 40)
            flat: Use flat prior if > 0 (default: 0)
            lambda_val: Weight on co-persistence prior (default: 5.0)
            mu_val: Weight on own-persistence prior (default: 2.0)
            tau: Minnesota prior overall tightness (default: 10.0)
            d: Lag decay parameter (default: 1.0)

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

        % Compute prior variance from training sample
        if {train} > 0
            y_train = ydata(1:{train}, :);
            y_train_lag = y_train(1:end-1, :);
            y_train_dep = y_train(2:end, :);
            coef = (y_train_lag' * y_train_lag) \\ (y_train_lag' * y_train_dep);
            resid = y_train_dep - y_train_lag * coef;
            sig_prior = sqrt(diag(resid' * resid / ({train}-1)));
        else
            sig_prior = std(ydata)';
        end

        % Create prior structures
        if {flat} > 0
            mnprior = [];
            vprior = [];
            lambda = Inf;
            mu = 0;
        else
            mnprior.tight = {tau};
            mnprior.decay = {d};
            vprior.sig = sig_prior';
            vprior.w = 1;
            lambda = {lambda_val};
            mu = {mu_val};
        end

        % Get dummy observations
        nx = 0;
        [ydum, xdum, pbreaks] = varprior(nv, nx, {lags}, mnprior, vprior);

        % Estimate VAR with dummy observations
        var = rfvar3([ydata; ydum], {lags}, [xdata; xdum], [breaks; T; T + pbreaks], lambda, mu);

        % Extract results from var structure
        By = var.By;    % Coefficients: (equations, variables, lags)
        u = var.u;      % Residuals
        xxi = var.xxi;  % (X'X)^(-1)

        % Reshape By for output (equations, variables*lags)
        w_reshaped = zeros(nv, nv * {lags});
        for eq = 1:nv
            for lag = 1:{lags}
                w_reshaped(eq, (lag-1)*nv+1:lag*nv) = By(eq, :, lag);
            end
        end

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
            % Use w_reshaped for predictions
            y_fit(t, :) = (w_reshaped * y_lag)';
        end

        residuals = y_est - y_fit;
        sigma = (residuals' * residuals) / T_est;

        % Log-likelihood (Gaussian)
        loglik = -0.5 * T_est * nv * log(2 * pi);
        loglik = loglik - 0.5 * T_est * log(det(sigma));
        loglik = loglik - 0.5 * trace(residuals * inv(sigma) * residuals');

        % Compute marginal log-likelihood using matrictint
        Tu = size(u, 1);
        loglik_full = matrictint(u' * u, xxi, Tu - {flat} * (nv + 1)) - {flat} * 0.5 * nv * (nv + 1) * log(2 * pi);

        if {train} > 0
            var_train = rfvar3([ydata(1:{train}, :); ydum], {lags}, [xdata; xdum], [breaks; {train}; {train} + pbreaks], lambda, mu);
            u_train = var_train.u;
            Tu_train = size(u_train, 1);
            loglik_train = matrictint(u_train' * u_train, var_train.xxi, Tu_train - {flat} * (nv + 1) / 2) - {flat} * 0.5 * nv * (nv + 1) * log(2 * pi);
            marginal_loglik = loglik_full - loglik_train;
        else
            marginal_loglik = loglik_full;
        end
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
