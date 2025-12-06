"""
Data Validation Utilities for F1 Data Quality Checks

Provides validation functions for driver numbers, tire compounds, lap times, and data quality.
"""

from typing import List, Optional, Tuple
from enum import Enum
import re


class TireCompound(str, Enum):
    """Valid F1 tire compounds."""

    SOFT = "SOFT"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    INTERMEDIATE = "INTERMEDIATE"
    WET = "WET"
    # Pirelli compound codes
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"
    C4 = "C4"
    C5 = "C5"


class TrackCondition(str, Enum):
    """Track surface conditions."""

    DRY = "DRY"
    DAMP = "DAMP"
    WET = "WET"
    SOAKING = "SOAKING"


class SessionType(str, Enum):
    """F1 session types."""

    FP1 = "FP1"
    FP2 = "FP2"
    FP3 = "FP3"
    QUALIFYING = "QUALIFYING"
    SPRINT = "SPRINT"
    RACE = "RACE"


def validate_driver_number(driver_number: int) -> bool:
    """
    Validate F1 driver number.

    Args:
        driver_number: Driver racing number

    Returns:
        True if valid (1-99), False otherwise
    """
    return 1 <= driver_number <= 99


def validate_tire_compound(compound: str) -> bool:
    """
    Validate tire compound designation.

    Args:
        compound: Tire compound string

    Returns:
        True if valid compound, False otherwise
    """
    try:
        TireCompound(compound.upper())
        return True
    except (ValueError, AttributeError):
        return False


def validate_lap_number(lap_number: int, max_laps: int = 100) -> bool:
    """
    Validate lap number.

    Args:
        lap_number: Lap number to validate
        max_laps: Maximum expected laps (default 100 for safety)

    Returns:
        True if valid, False otherwise
    """
    return 1 <= lap_number <= max_laps


def validate_lap_time(lap_time_seconds: float) -> Tuple[bool, Optional[str]]:
    """
    Validate lap time is within realistic bounds.

    Args:
        lap_time_seconds: Lap time in seconds

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Realistic F1 lap times: 30 seconds (Monaco short) to 150 seconds (Spa long)
    MIN_LAP_TIME = 30.0
    MAX_LAP_TIME = 150.0

    if lap_time_seconds < MIN_LAP_TIME:
        return False, f"Lap time too fast: {lap_time_seconds}s (minimum {MIN_LAP_TIME}s)"

    if lap_time_seconds > MAX_LAP_TIME:
        return False, f"Lap time too slow: {lap_time_seconds}s (maximum {MAX_LAP_TIME}s)"

    return True, None


def validate_sector_times(
    sector_1: float, sector_2: float, sector_3: float, lap_time: Optional[float] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate sector times are realistic and sum correctly.

    Args:
        sector_1: Sector 1 time in seconds
        sector_2: Sector 2 time in seconds
        sector_3: Sector 3 time in seconds
        lap_time: Optional total lap time for consistency check

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Sector times should be positive
    if sector_1 <= 0 or sector_2 <= 0 or sector_3 <= 0:
        return False, "Sector times must be positive"

    # Sector times typically between 5 and 60 seconds
    for i, sector in enumerate([sector_1, sector_2, sector_3], 1):
        if sector < 5.0 or sector > 60.0:
            return False, f"Sector {i} time unrealistic: {sector}s"

    # Check sum equals lap time if provided
    if lap_time is not None:
        sector_sum = sector_1 + sector_2 + sector_3
        difference = abs(lap_time - sector_sum)
        if difference > 0.1:  # 100ms tolerance
            return False, f"Sector sum ({sector_sum:.3f}s) doesn't match lap time ({lap_time:.3f}s)"

    return True, None


def validate_position(position: int, max_cars: int = 20) -> bool:
    """
    Validate race position.

    Args:
        position: Position number
        max_cars: Maximum number of cars (default 20)

    Returns:
        True if valid, False otherwise
    """
    return 1 <= position <= max_cars


def validate_speed(speed_kmh: float) -> Tuple[bool, Optional[str]]:
    """
    Validate speed is within realistic F1 bounds.

    Args:
        speed_kmh: Speed in km/h

    Returns:
        Tuple of (is_valid, error_message)
    """
    MIN_SPEED = 0.0
    MAX_SPEED = 380.0  # F1 top speed record

    if speed_kmh < MIN_SPEED:
        return False, f"Speed cannot be negative: {speed_kmh} km/h"

    if speed_kmh > MAX_SPEED:
        return False, f"Speed exceeds maximum: {speed_kmh} km/h (max {MAX_SPEED} km/h)"

    return True, None


def validate_temperature(temp_celsius: float, sensor_type: str = "track") -> Tuple[bool, Optional[str]]:
    """
    Validate temperature reading.

    Args:
        temp_celsius: Temperature in Celsius
        sensor_type: Type of sensor ('track', 'air', 'tire', 'brake')

    Returns:
        Tuple of (is_valid, error_message)
    """
    ranges = {
        "track": (10.0, 60.0),
        "air": (0.0, 50.0),
        "tire": (50.0, 130.0),
        "brake": (100.0, 1200.0),
    }

    if sensor_type not in ranges:
        return False, f"Unknown sensor type: {sensor_type}"

    min_temp, max_temp = ranges[sensor_type]

    if temp_celsius < min_temp or temp_celsius > max_temp:
        return (
            False,
            f"{sensor_type.capitalize()} temperature out of range: {temp_celsius}°C "
            f"(valid: {min_temp}-{max_temp}°C)",
        )

    return True, None


def validate_percentage(value: float, field_name: str = "value") -> Tuple[bool, Optional[str]]:
    """
    Validate percentage value (0-100).

    Args:
        value: Percentage value
        field_name: Name of field for error message

    Returns:
        Tuple of (is_valid, error_message)
    """
    if value < 0.0 or value > 100.0:
        return False, f"{field_name} must be between 0 and 100, got {value}"

    return True, None


def detect_outliers(values: List[float], threshold: float = 3.0) -> List[int]:
    """
    Detect outliers using Z-score method.

    Args:
        values: List of numeric values
        threshold: Z-score threshold (default 3.0)

    Returns:
        List of indices of outlier values
    """
    if len(values) < 3:
        return []

    # Calculate mean and standard deviation
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std_dev = variance**0.5

    if std_dev == 0:
        return []

    # Find outliers
    outliers = []
    for i, value in enumerate(values):
        z_score = abs((value - mean) / std_dev)
        if z_score > threshold:
            outliers.append(i)

    return outliers


def validate_gap_consistency(gap_to_leader: float, interval_to_ahead: float, position: int) -> bool:
    """
    Validate gap measurements are consistent.

    Args:
        gap_to_leader: Gap to race leader in seconds
        interval_to_ahead: Interval to car ahead in seconds
        position: Current position

    Returns:
        True if consistent, False otherwise
    """
    # Leader should have zero gap
    if position == 1:
        return abs(gap_to_leader) < 0.01

    # Gap to leader should be >= interval to ahead
    if gap_to_leader < interval_to_ahead - 0.1:  # 100ms tolerance
        return False

    return True


def validate_numeric_range(
    value: float,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    field_name: str = "value"
) -> Tuple[bool, Optional[str]]:
    """
    Validate that a numeric value is within specified bounds.

    Args:
        value: The numeric value to validate
        min_value: Minimum allowed value (inclusive), None for no lower bound
        max_value: Maximum allowed value (inclusive), None for no upper bound
        field_name: Name of the field being validated (for error messages)

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if value is within bounds, False otherwise
        - error_message: None if valid, descriptive error message if invalid

    Examples:
        >>> validate_numeric_range(50, 0, 100, "temperature")
        (True, None)
        >>> validate_numeric_range(-10, 0, 100, "temperature")
        (False, "temperature must be >= 0 (got -10)")
    """
    if min_value is not None and value < min_value:
        return False, f"{field_name} must be >= {min_value} (got {value})"

    if max_value is not None and value > max_value:
        return False, f"{field_name} must be <= {max_value} (got {value})"

    return True, None


def validate_dataframe(
    df,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 1,
    check_nulls: bool = True,
    column_dtypes: Optional[dict] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate pandas DataFrame structure and content.

    Args:
        df: pandas DataFrame to validate
        required_columns: List of column names that must be present
        min_rows: Minimum number of rows required (default 1)
        check_nulls: If True, check for null values in required columns
        column_dtypes: Optional dict mapping column names to expected dtypes

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if DataFrame passes all checks, False otherwise
        - error_message: None if valid, descriptive error message if invalid

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> validate_dataframe(df, required_columns=['a', 'b'])
        (True, None)
        >>> validate_dataframe(df, required_columns=['a', 'c'])
        (False, "Missing required columns: ['c']")
    """
    import pandas as pd

    # Check if input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        return False, f"Expected pandas DataFrame, got {type(df).__name__}"

    # Check minimum rows
    if len(df) < min_rows:
        return False, f"DataFrame has {len(df)} rows, minimum required is {min_rows}"

    # Check for empty DataFrame
    if df.empty and min_rows > 0:
        return False, "DataFrame is empty"

    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            return False, f"Missing required columns: {sorted(list(missing_cols))}"

    # Check for null values in required columns
    if check_nulls and required_columns:
        for col in required_columns:
            if col in df.columns and df[col].isnull().any():
                null_count = df[col].isnull().sum()
                return False, f"Column '{col}' contains {null_count} null values"

    # Check column data types
    if column_dtypes:
        for col, expected_dtype in column_dtypes.items():
            if col not in df.columns:
                continue  # Already checked in required_columns

            actual_dtype = df[col].dtype
            # Handle string dtype comparison
            if expected_dtype == 'object' or expected_dtype == str:
                if actual_dtype != 'object':
                    return False, f"Column '{col}' has dtype {actual_dtype}, expected object/string"
            elif expected_dtype == 'numeric':
                if not pd.api.types.is_numeric_dtype(actual_dtype):
                    return False, f"Column '{col}' has dtype {actual_dtype}, expected numeric"
            else:
                if str(actual_dtype) != str(expected_dtype):
                    return False, f"Column '{col}' has dtype {actual_dtype}, expected {expected_dtype}"

    return True, None


# Example usage
if __name__ == "__main__":
    # Driver number validation
    print(f"Driver #1 valid: {validate_driver_number(1)}")
    print(f"Driver #100 valid: {validate_driver_number(100)}")

    # Tire compound validation
    print(f"SOFT compound valid: {validate_tire_compound('SOFT')}")
    print(f"SUPER_SOFT compound valid: {validate_tire_compound('SUPER_SOFT')}")

    # Lap time validation
    is_valid, error = validate_lap_time(83.456)
    print(f"Lap time 83.456s valid: {is_valid}")

    # Sector times validation
    is_valid, error = validate_sector_times(28.123, 30.234, 25.099, 83.456)
    print(f"Sector times valid: {is_valid}, Error: {error}")

    # Outlier detection
    lap_times = [83.1, 83.3, 83.2, 83.4, 95.8, 83.5]  # 95.8 is outlier
    outliers = detect_outliers(lap_times)
    print(f"Outliers at indices: {outliers}")
