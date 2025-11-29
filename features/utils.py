"""
Utility functions for feature calculations.

Provides common mathematical operations, statistical functions,
and data transformations used across feature calculators.
"""

from typing import Any, Callable, List, Optional, Union
import pandas as pd
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d


def rolling_window(
    data: Union[pd.Series, np.ndarray],
    window_size: int,
    func: Callable,
    min_periods: Optional[int] = None
) -> np.ndarray:
    """
    Apply function to rolling windows.
    
    Args:
        data: Input data (Series or array)
        window_size: Window size in number of points
        func: Function to apply to each window
        min_periods: Minimum number of observations required
        
    Returns:
        Array with function applied to rolling windows
        
    Example:
        >>> data = pd.Series([1, 2, 3, 4, 5])
        >>> rolling_window(data, 3, np.mean)
        array([nan, nan, 2., 3., 4.])
    """
    if isinstance(data, pd.Series):
        min_periods = min_periods or window_size
        return data.rolling(window=window_size, min_periods=min_periods).apply(func).values
    else:
        # Manual rolling for numpy arrays
        result = np.full(len(data), np.nan)
        min_periods = min_periods or window_size
        
        for i in range(len(data)):
            if i + 1 >= min_periods:
                window = data[max(0, i - window_size + 1):i + 1]
                result[i] = func(window)
        
        return result


def linear_regression(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Perform linear regression and return statistics.
    
    Formula:
        y = slope * x + intercept
        slope = Σ[(x_i - x̄)(y_i - ȳ)] / Σ[(x_i - x̄)²]
    
    Args:
        x: Independent variable
        y: Dependent variable
        
    Returns:
        Dictionary with slope, intercept, r_squared, p_value, std_err
        
    Example:
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> y = np.array([2.1, 4.0, 6.1, 8.0, 10.1])
        >>> result = linear_regression(x, y)
        >>> print(result['slope'])  # ~2.0
    """
    if len(x) < 2 or len(y) < 2:
        return {
            'slope': 0.0,
            'intercept': np.mean(y) if len(y) > 0 else 0.0,
            'r_squared': 0.0,
            'p_value': 1.0,
            'std_err': 0.0
        }
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(r_value ** 2),
        'p_value': float(p_value),
        'std_err': float(std_err)
    }


def exponential_smoothing(
    data: Union[pd.Series, np.ndarray],
    alpha: float = 0.3
) -> np.ndarray:
    """
    Apply exponential moving average (EMA).
    
    Formula:
        EMA_t = α * value_t + (1 - α) * EMA_{t-1}
    
    Args:
        data: Input data
        alpha: Smoothing factor (0 < alpha <= 1)
               Higher alpha = more weight to recent values
        
    Returns:
        Exponentially smoothed array
        
    Example:
        >>> data = np.array([10, 12, 11, 13, 15])
        >>> smoothed = exponential_smoothing(data, alpha=0.3)
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    if len(data) == 0:
        return np.array([])
    
    result = np.zeros(len(data))
    result[0] = data[0]
    
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    
    return result


def detect_changepoints(
    data: np.ndarray,
    threshold: float = 1.0,
    window: int = 5
) -> List[int]:
    """
    Detect significant changes in time series using CUSUM algorithm.
    
    Args:
        data: Input time series
        threshold: Detection threshold (in std deviations)
        window: Window size for local statistics
        
    Returns:
        List of indices where changepoints detected
        
    Example:
        >>> data = np.array([10, 10, 10, 15, 15, 15])
        >>> changepoints = detect_changepoints(data, threshold=1.0)
        >>> print(changepoints)  # [3] (change at index 3)
    """
    if len(data) < window:
        return []
    
    changepoints = []
    mean = np.mean(data[:window])
    std = np.std(data[:window]) or 1.0
    
    cumsum = 0.0
    
    for i in range(window, len(data)):
        # Update local statistics
        local_window = data[max(0, i - window):i]
        mean = np.mean(local_window)
        std = np.std(local_window) or 1.0
        
        # CUSUM
        deviation = (data[i] - mean) / std
        cumsum = max(0, cumsum + deviation - 0.5)
        
        if cumsum > threshold:
            changepoints.append(i)
            cumsum = 0.0  # Reset
    
    return changepoints


def normalize_features(
    features: pd.DataFrame,
    method: str = 'zscore',
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Normalize feature columns.
    
    Args:
        features: DataFrame with features
        method: Normalization method ('zscore', 'minmax', 'robust')
        columns: Columns to normalize (None = all numeric columns)
        
    Returns:
        DataFrame with normalized features
        
    Example:
        >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
        >>> normalized = normalize_features(df, method='zscore')
    """
    result = features.copy()
    
    if columns is None:
        columns = features.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in result.columns:
            continue
        
        if method == 'zscore':
            # Z-score normalization: (x - mean) / std
            mean = result[col].mean()
            std = result[col].std()
            if std > 0:
                result[col] = (result[col] - mean) / std
        
        elif method == 'minmax':
            # Min-max normalization: (x - min) / (max - min)
            min_val = result[col].min()
            max_val = result[col].max()
            if max_val > min_val:
                result[col] = (result[col] - min_val) / (max_val - min_val)
        
        elif method == 'robust':
            # Robust scaling: (x - median) / IQR
            median = result[col].median()
            q75 = result[col].quantile(0.75)
            q25 = result[col].quantile(0.25)
            iqr = q75 - q25
            if iqr > 0:
                result[col] = (result[col] - median) / iqr
    
    return result


def interpolate_missing(
    data: Union[pd.Series, np.ndarray],
    method: str = 'linear',
    limit: Optional[int] = None
) -> Union[pd.Series, np.ndarray]:
    """
    Interpolate missing values.
    
    Args:
        data: Data with missing values
        method: Interpolation method ('linear', 'spline', 'nearest')
        limit: Maximum number of consecutive NaNs to fill
        
    Returns:
        Data with missing values filled
        
    Example:
        >>> data = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        >>> filled = interpolate_missing(data, method='linear')
    """
    if isinstance(data, pd.Series):
        if method == 'spline':
            return data.interpolate(method='spline', order=3, limit=limit)
        else:
            return data.interpolate(method=method, limit=limit)
    else:
        # Numpy array
        series = pd.Series(data)
        if method == 'spline':
            result = series.interpolate(method='spline', order=3, limit=limit)
        else:
            result = series.interpolate(method=method, limit=limit)
        return result.values


def calculate_percentile(
    value: float,
    distribution: Union[pd.Series, np.ndarray]
) -> float:
    """
    Calculate percentile rank of value in distribution.
    
    Args:
        value: Value to rank
        distribution: Distribution of values
        
    Returns:
        Percentile rank (0-100)
        
    Example:
        >>> distribution = np.array([1, 2, 3, 4, 5])
        >>> percentile = calculate_percentile(3, distribution)
        >>> print(percentile)  # 60.0 (3 is better than 60% of values)
    """
    if isinstance(distribution, pd.Series):
        distribution = distribution.values
    
    if len(distribution) == 0:
        return 50.0
    
    percentile = (distribution <= value).sum() / len(distribution) * 100
    return float(percentile)


def moving_average(
    data: Union[pd.Series, np.ndarray],
    window: int = 3
) -> np.ndarray:
    """
    Calculate simple moving average.
    
    Args:
        data: Input data
        window: Window size
        
    Returns:
        Moving average array
        
    Example:
        >>> data = np.array([1, 2, 3, 4, 5])
        >>> ma = moving_average(data, window=3)
    """
    if isinstance(data, pd.Series):
        return data.rolling(window=window, min_periods=1).mean().values
    else:
        return pd.Series(data).rolling(window=window, min_periods=1).mean().values


def weighted_average(
    values: Union[List[float], np.ndarray],
    weights: Union[List[float], np.ndarray]
) -> float:
    """
    Calculate weighted average.
    
    Formula:
        weighted_avg = Σ(value_i * weight_i) / Σ(weight_i)
    
    Args:
        values: Values to average
        weights: Corresponding weights
        
    Returns:
        Weighted average
        
    Example:
        >>> values = [10, 20, 30]
        >>> weights = [1, 2, 1]
        >>> avg = weighted_average(values, weights)
        >>> print(avg)  # 20.0
    """
    values = np.array(values)
    weights = np.array(weights)
    
    if len(values) != len(weights):
        raise ValueError("Values and weights must have same length")
    
    if weights.sum() == 0:
        return np.mean(values)
    
    return float(np.sum(values * weights) / np.sum(weights))


def safe_divide(
    numerator: Union[float, np.ndarray],
    denominator: Union[float, np.ndarray],
    default: float = 0.0
) -> Union[float, np.ndarray]:
    """
    Safely divide, returning default on division by zero.
    
    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)
        default: Value to return on division by zero
        
    Returns:
        Division result or default
        
    Example:
        >>> result = safe_divide(10, 0, default=0.0)
        >>> print(result)  # 0.0
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(numerator, denominator)
        
        if isinstance(result, np.ndarray):
            result[~np.isfinite(result)] = default
        elif not np.isfinite(result):
            result = default
    
    return result


def calculate_z_score(
    value: float,
    data: Union[pd.Series, np.ndarray]
) -> float:
    """
    Calculate z-score for value in distribution.
    
    Formula:
        z = (value - mean) / std
    
    Args:
        value: Value to score
        data: Distribution
        
    Returns:
        Z-score
        
    Example:
        >>> data = np.array([10, 12, 14, 16, 18])
        >>> z = calculate_z_score(14, data)
        >>> print(z)  # 0.0 (14 is the mean)
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    mean = np.mean(data)
    std = np.std(data)
    
    if std == 0:
        return 0.0
    
    return float((value - mean) / std)


def remove_outliers(
    data: Union[pd.Series, np.ndarray],
    threshold: float = 3.0,
    method: str = 'zscore'
) -> Union[pd.Series, np.ndarray]:
    """
    Remove outliers from data.
    
    Args:
        data: Input data
        threshold: Threshold for outlier detection
        method: Detection method ('zscore' or 'iqr')
        
    Returns:
        Data with outliers replaced by NaN
        
    Example:
        >>> data = np.array([10, 12, 11, 100, 13])
        >>> clean = remove_outliers(data, threshold=3.0)
    """
    is_series = isinstance(data, pd.Series)
    if not is_series:
        data = pd.Series(data)
    
    if method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        mask = z_scores > threshold
    elif method == 'iqr':
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        mask = (data < (q1 - threshold * iqr)) | (data > (q3 + threshold * iqr))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    result = data.copy()
    result[mask] = np.nan
    
    return result if is_series else result.values
