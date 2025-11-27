"""
F1 Race Strategy Intelligence Engine - Core Application Package
"""

__version__ = "0.1.0"
__author__ = "F1 Strategy Team"

# Package-level imports for common utilities
from app.utils.logger import get_logger
from app.utils.time_utils import lap_time_to_seconds, seconds_to_lap_time
from app.utils.validators import validate_driver_number, validate_tire_compound

__all__ = [
    "get_logger",
    "lap_time_to_seconds",
    "seconds_to_lap_time",
    "validate_driver_number",
    "validate_tire_compound",
]
