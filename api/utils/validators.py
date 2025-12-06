"""Custom validators for API requests."""

from fastapi import HTTPException, status

# Valid F1 circuits (2024 season)
VALID_CIRCUITS = {
    "Bahrain", "Saudi Arabia", "Australia", "Japan", "China",
    "Miami", "Imola", "Monaco", "Canada", "Spain",
    "Austria", "Great Britain", "Hungary", "Belgium", "Netherlands",
    "Monza", "Azerbaijan", "Singapore", "USA", "Mexico",
    "Brazil", "Las Vegas", "Abu Dhabi", "Silverstone", "Baku"
}

# Valid tire compounds
VALID_TIRE_COMPOUNDS = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}

# Valid weather conditions
VALID_WEATHER_CONDITIONS = {"Dry", "Wet", "Mixed", "Damp"}


def validate_circuit_name(circuit: str) -> str:
    """
    Validate circuit name.
    
    Args:
        circuit: Circuit name to validate
        
    Returns:
        Validated circuit name
        
    Raises:
        HTTPException: If circuit is invalid
    """
    if circuit not in VALID_CIRCUITS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid circuit name. Must be one of: {', '.join(sorted(VALID_CIRCUITS))}"
        )
    return circuit


def validate_tire_compound(compound: str) -> str:
    """
    Validate tire compound.
    
    Args:
        compound: Tire compound to validate
        
    Returns:
        Validated compound (uppercase)
        
    Raises:
        HTTPException: If compound is invalid
    """
    compound_upper = compound.upper()
    if compound_upper not in VALID_TIRE_COMPOUNDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tire compound. Must be one of: {', '.join(VALID_TIRE_COMPOUNDS)}"
        )
    return compound_upper


def validate_weather_condition(weather: str) -> str:
    """
    Validate weather condition.
    
    Args:
        weather: Weather condition to validate
        
    Returns:
        Validated weather condition
        
    Raises:
        HTTPException: If weather condition is invalid
    """
    if weather not in VALID_WEATHER_CONDITIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid weather condition. Must be one of: {', '.join(VALID_WEATHER_CONDITIONS)}"
        )
    return weather


def validate_lap_number(lap: int, total_laps: int) -> None:
    """
    Validate lap number.
    
    Args:
        lap: Current lap number
        total_laps: Total laps in race
        
    Raises:
        HTTPException: If lap number is invalid
    """
    if lap < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Lap number must be at least 1"
        )
    
    if lap > total_laps:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Lap number ({lap}) cannot exceed total laps ({total_laps})"
        )
