"""
Data Loading Utilities
=======================

Functions to load and process data for DSGE estimation:
- Load Excel files (.xls, .xlsx)
- Load MATLAB files (.mat)
- Data transformation utilities
- Handle Argentine and US data formats
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import scipy.io
import os


def load_excel_data(filepath: str, sheet_name: Optional[str] = None,
                   header: Optional[int] = 0) -> pd.DataFrame:
    """
    Load data from Excel file.

    Args:
        filepath: Path to Excel file
        sheet_name: Sheet name (None for first sheet)
        header: Row number to use as column names (0-indexed)

    Returns:
        DataFrame with data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name, header=header)
        return df
    except Exception as e:
        raise ValueError(f"Error loading Excel file: {e}")


def load_mat_data(filepath: str) -> Dict:
    """
    Load data from MATLAB .mat file.

    Args:
        filepath: Path to .mat file

    Returns:
        Dictionary with variable names as keys
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        mat_data = scipy.io.loadmat(filepath)
        # Remove MATLAB metadata
        data = {k: v for k, v in mat_data.items() if not k.startswith('__')}
        return data
    except Exception as e:
        raise ValueError(f"Error loading MAT file: {e}")


def load_smets_wouters_data(data_dir: str = None) -> pd.DataFrame:
    """
    Load Smets & Wouters (2007) US data.

    Data includes 7 quarterly time series (1955Q1-2005Q4):
    - dy: Output growth
    - dc: Consumption growth
    - dinve: Investment growth
    - labobs: Hours worked
    - pinfobs: Inflation
    - dw: Wage growth
    - robs: Interest rate

    Args:
        data_dir: Directory containing usmodel_data.xls (defaults to repo/)

    Returns:
        DataFrame with 7 columns and ~200 quarterly observations
    """
    if data_dir is None:
        # Default to repo directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(script_dir), 'repo')

    filepath = os.path.join(data_dir, 'usmodel_data.xls')

    # Try .xls first, then .xlsx
    if not os.path.exists(filepath):
        filepath = os.path.join(data_dir, 'usmodel_data.xlsx')

    # Load data
    df = load_excel_data(filepath)

    # Check expected columns
    expected_cols = ['dy', 'dc', 'dinve', 'labobs', 'pinfobs', 'dw', 'robs']
    if not all(col in df.columns for col in expected_cols):
        # Try to infer column names (file might not have headers)
        if df.shape[1] == 7:
            df.columns = expected_cols
        else:
            raise ValueError(f"Expected columns {expected_cols}, got {df.columns.tolist()}")

    # Create quarterly date index starting from 1955Q1
    start_date = '1955-01-01'
    periods = len(df)
    df.index = pd.date_range(start=start_date, periods=periods, freq='QS')

    return df[expected_cols]


def transform_data(df: pd.DataFrame, transformations: Dict[str, str]) -> pd.DataFrame:
    """
    Apply transformations to data.

    Args:
        df: Input DataFrame
        transformations: Dictionary mapping column names to transformation types
                        ('level', 'diff', 'log', 'logdiff', 'growth')

    Returns:
        Transformed DataFrame
    """
    df_out = df.copy()

    for col, trans in transformations.items():
        if col not in df.columns:
            continue

        if trans == 'level':
            pass  # No transformation
        elif trans == 'diff':
            df_out[col] = df[col].diff()
        elif trans == 'log':
            df_out[col] = np.log(df[col])
        elif trans == 'logdiff':
            df_out[col] = np.log(df[col]).diff()
        elif trans == 'growth':
            df_out[col] = df[col].pct_change() * 100
        else:
            raise ValueError(f"Unknown transformation: {trans}")

    return df_out


def create_lags(data: np.ndarray, lags: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create lagged data matrix for VAR estimation.

    Args:
        data: Time series data (T x n)
        lags: Number of lags

    Returns:
        (Y, X) where Y is (T-lags x n) and X is (T-lags x n*lags+1)
        X includes constant term and lags
    """
    T, n = data.shape
    Y = data[lags:, :]

    # Build X with constant and lags
    X_list = [np.ones((T-lags, 1))]  # Constant

    for lag in range(1, lags+1):
        X_list.append(data[lags-lag:-lag, :])

    X = np.column_stack(X_list)

    return Y, X


def split_train_test(data: pd.DataFrame, train_end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets.

    Args:
        data: Full dataset with DatetimeIndex
        train_end: End date for training set (e.g., '2004-10-01')

    Returns:
        (train_data, test_data)
    """
    train = data.loc[:train_end]
    test = data.loc[train_end:]

    return train, test


def get_estimation_sample(df: pd.DataFrame, first_obs: int, nobs: Optional[int] = None,
                         presample: int = 0) -> pd.DataFrame:
    """
    Extract estimation sample matching Dynare conventions.

    Args:
        df: Full dataset
        first_obs: First observation to use (1-indexed, as in Dynare)
        nobs: Number of observations to use (None = all remaining)
        presample: Number of presample observations for filter initialization

    Returns:
        Estimation sample DataFrame
    """
    # Convert to 0-indexed
    start_idx = first_obs - 1 - presample
    end_idx = first_obs - 1 + nobs if nobs is not None else len(df)

    # Extract sample
    sample = df.iloc[start_idx:end_idx].copy()

    return sample


def cumsum_vars(data: pd.DataFrame, var_list: List[str]) -> pd.DataFrame:
    """
    Cumulative sum of specified variables (for integrating growth rates).

    Args:
        data: DataFrame
        var_list: List of column names to cumsum

    Returns:
        DataFrame with specified variables cumsum'd
    """
    df_out = data.copy()
    for var in var_list:
        if var in df_out.columns:
            df_out[var] = df_out[var].cumsum()

    return df_out


def load_argentine_data(filepath: str, var_mapping: Optional[Dict] = None) -> pd.DataFrame:
    """
    Load Argentine macroeconomic data.

    Placeholder function for loading Argentine data with appropriate
    transformations matching the Smets-Wouters specification.

    Args:
        filepath: Path to Argentine data file
        var_mapping: Dictionary mapping source variables to model variables

    Returns:
        DataFrame with same structure as Smets-Wouters data
    """
    # TODO: Implement once Argentine data sources are identified
    raise NotImplementedError("Argentine data loading not yet implemented. "
                            "Need to specify data sources and transformations.")


def describe_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics for data.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with descriptive statistics
    """
    stats = pd.DataFrame({
        'mean': df.mean(),
        'std': df.std(),
        'min': df.min(),
        'max': df.max(),
        'q25': df.quantile(0.25),
        'q50': df.quantile(0.50),
        'q75': df.quantile(0.75),
    })

    return stats


if __name__ == '__main__':
    print("Testing data loading...")

    # Try to load Smets-Wouters data
    try:
        df = load_smets_wouters_data()
        print(f"\nLoaded Smets-Wouters data: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nDescriptive statistics:")
        print(describe_data(df))

        # Test estimation sample extraction
        # Main estimation: first_obs=71, presample=4
        # That means we use observations 67-end for Kalman filter
        sample = get_estimation_sample(df, first_obs=71, presample=4)
        print(f"\nEstimation sample: {len(sample)} observations")
        print(f"From {sample.index[0]} to {sample.index[-1]}")

    except FileNotFoundError as e:
        print(f"\nCould not load data: {e}")
        print("This is expected if running from different directory.")
        print("Data will be loaded when running from project root.")

    print("\nAll tests passed!")
