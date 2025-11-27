"""
Time Conversion Utilities for F1-specific Time Formats

Handles lap time conversions, race time calculations, and session time tracking.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Union
import re


def lap_time_to_seconds(lap_time: str) -> float:
    """
    Convert lap time string to seconds.

    Args:
        lap_time: Lap time in format 'M:SS.mmm' or 'SS.mmm'
                  Examples: '1:23.456', '83.456'

    Returns:
        Lap time in seconds as float

    Raises:
        ValueError: If format is invalid
    """
    try:
        # Handle format with minutes: '1:23.456'
        if ":" in lap_time:
            parts = lap_time.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid lap time format: {lap_time}")

            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds

        # Handle format without minutes: '83.456'
        return float(lap_time)

    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid lap time format: {lap_time}") from e


def seconds_to_lap_time(seconds: float, include_minutes: bool = True) -> str:
    """
    Convert seconds to lap time string format.

    Args:
        seconds: Lap time in seconds
        include_minutes: Whether to include minutes in format

    Returns:
        Formatted lap time string ('M:SS.mmm' or 'SS.mmm')
    """
    if seconds < 0:
        raise ValueError("Lap time cannot be negative")

    if include_minutes:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}:{remaining_seconds:06.3f}"
    else:
        return f"{seconds:.3f}"


def milliseconds_to_formatted_string(milliseconds: int) -> str:
    """
    Convert milliseconds to formatted time string.

    Args:
        milliseconds: Time in milliseconds

    Returns:
        Formatted string 'M:SS.mmm' or 'H:MM:SS.mmm' for longer durations
    """
    total_seconds = milliseconds / 1000.0

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:06.3f}"
    elif minutes > 0:
        return f"{minutes}:{seconds:06.3f}"
    else:
        return f"{seconds:.3f}"


def calculate_race_time(start_time: datetime, current_time: datetime) -> str:
    """
    Calculate elapsed race time.

    Args:
        start_time: Race start timestamp
        current_time: Current timestamp

    Returns:
        Formatted elapsed time 'H:MM:SS'
    """
    elapsed = current_time - start_time
    total_seconds = int(elapsed.total_seconds())

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    return f"{hours}:{minutes:02d}:{seconds:02d}"


def parse_session_time(session_time: str) -> datetime:
    """
    Parse session time string to datetime object.

    Args:
        session_time: Time string in ISO 8601 format

    Returns:
        Parsed datetime object in UTC
    """
    # Handle various ISO 8601 formats
    formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(session_time, fmt)
            # Ensure UTC timezone
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue

    raise ValueError(f"Unable to parse session time: {session_time}")


def format_gap(gap_seconds: float) -> str:
    """
    Format gap time for display.

    Args:
        gap_seconds: Gap in seconds

    Returns:
        Formatted gap string (e.g., '+1.234s', '+12.345s', '+1 LAP')
    """
    if gap_seconds >= 100:
        laps = int(gap_seconds / 90)  # Approximate lap time
        return f"+{laps} LAP" if laps == 1 else f"+{laps} LAPS"
    else:
        return f"+{gap_seconds:.3f}s"


def calculate_sector_sum(sector_1: float, sector_2: float, sector_3: float) -> float:
    """
    Calculate total lap time from sector times.

    Args:
        sector_1: Sector 1 time in seconds
        sector_2: Sector 2 time in seconds
        sector_3: Sector 3 time in seconds

    Returns:
        Total lap time in seconds
    """
    return sector_1 + sector_2 + sector_3


def validate_sector_consistency(
    lap_time: float, sector_1: float, sector_2: float, sector_3: float, tolerance: float = 0.1
) -> bool:
    """
    Validate that sector times sum to lap time within tolerance.

    Args:
        lap_time: Total lap time in seconds
        sector_1: Sector 1 time in seconds
        sector_2: Sector 2 time in seconds
        sector_3: Sector 3 time in seconds
        tolerance: Acceptable difference in seconds

    Returns:
        True if consistent, False otherwise
    """
    sector_sum = calculate_sector_sum(sector_1, sector_2, sector_3)
    difference = abs(lap_time - sector_sum)
    return difference <= tolerance


def get_current_utc_timestamp() -> str:
    """
    Get current UTC timestamp in ISO 8601 format.

    Returns:
        ISO 8601 formatted timestamp string
    """
    return datetime.now(timezone.utc).isoformat()


# Example usage
if __name__ == "__main__":
    # Lap time conversions
    lap_time_str = "1:23.456"
    seconds = lap_time_to_seconds(lap_time_str)
    print(f"{lap_time_str} = {seconds} seconds")

    converted_back = seconds_to_lap_time(seconds)
    print(f"{seconds} seconds = {converted_back}")

    # Milliseconds conversion
    ms = 83456
    formatted = milliseconds_to_formatted_string(ms)
    print(f"{ms}ms = {formatted}")

    # Gap formatting
    gap = 12.345
    print(f"Gap: {format_gap(gap)}")

    # Sector validation
    is_valid = validate_sector_consistency(83.456, 28.123, 30.234, 25.099)
    print(f"Sector consistency: {is_valid}")
