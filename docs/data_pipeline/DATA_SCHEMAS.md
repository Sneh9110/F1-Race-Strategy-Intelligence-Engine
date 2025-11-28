# Data Schemas Reference

Complete reference for all data schemas in the F1 Race Strategy Intelligence Engine.

## Overview

All data schemas are defined using Pydantic for validation and type safety. Located in `data_pipeline/schemas/`.

## Timing Data

### TimingPoint

Individual timing measurement for a driver at a specific moment.

```python
{
    "driver": str,              # Driver identifier (e.g., "VER", "HAM")
    "timestamp": datetime,      # UTC timestamp
    "lap_data": LapData         # Lap-level data (see below)
}
```

### LapData

Complete lap information including sectors and position.

```python
{
    "lap_number": int,          # Lap number (1-based)
    "lap_time": float,          # Total lap time (seconds)
    "sector1_time": float,      # Sector 1 time (seconds)
    "sector2_time": float,      # Sector 2 time (seconds)
    "sector3_time": float,      # Sector 3 time (seconds)
    "position": int,            # Current position (1-20)
    "gap_to_leader": float,     # Gap to leader (seconds)
    "tire_compound": str,       # Tire type: "SOFT", "MEDIUM", "HARD", "INTER", "WET"
    "tire_age": int,            # Tire age (laps)
    "is_pit_lap": bool,         # True if lap includes pit stop
    "timestamp": datetime       # Lap completion time
}
```

**Validation Rules**:
- `lap_time`: 30-150 seconds (race pace range)
- `sector1_time + sector2_time + sector3_time ≈ lap_time` (±100ms tolerance)
- `position`: 1-20
- `tire_age`: ≥ 0
- `gap_to_leader`: ≥ 0 for all except P1 (0.0)

### SessionTiming

Complete timing data for a session.

```python
{
    "session_id": str,              # Unique session identifier
    "session_name": str,            # "FP1", "FP2", "FP3", "Q", "Sprint", "Race"
    "track_name": str,              # Circuit name
    "timing_points": List[TimingPoint],  # All timing data
    "timestamp": datetime           # Session start time
}
```

## Weather Data

### WeatherData

Single weather observation.

```python
{
    "timestamp": datetime,          # Observation time
    "air_temperature": float,       # Air temp (°C)
    "track_temperature": float,     # Track surface temp (°C)
    "humidity": float,              # Relative humidity (%)
    "pressure": float,              # Atmospheric pressure (hPa)
    "wind_speed": float,            # Wind speed (km/h)
    "wind_direction": int,          # Wind direction (degrees, 0-359)
    "rainfall": float,              # Rainfall rate (mm/h)
    "weather_condition": str        # "Clear", "Cloudy", "Overcast", "Rain", "Heavy Rain"
}
```

**Validation Rules**:
- `air_temperature`: 10-50°C
- `track_temperature`: ≥ `air_temperature` (typically +8 to +15°C)
- `humidity`: 0-100%
- `pressure`: 950-1050 hPa
- `wind_speed`: ≥ 0 km/h
- `wind_direction`: 0-359

### WeatherForecast

Weather forecast for future timepoint.

```python
{
    "forecast_time": datetime,              # Forecasted time
    "air_temperature": float,               # Predicted air temp (°C)
    "track_temperature": float,             # Predicted track temp (°C)
    "humidity": float,                      # Predicted humidity (%)
    "wind_speed": float,                    # Predicted wind speed (km/h)
    "precipitation_probability": float,     # Rain probability (%)
    "weather_condition": str                # Predicted condition
}
```

### WeatherSession

Complete weather data for a session.

```python
{
    "session_id": str,                          # Unique session identifier
    "track_name": str,                          # Circuit name
    "observations": List[WeatherData],          # Historical observations
    "forecasts": List[WeatherForecast],         # Future forecasts
    "timestamp": datetime                       # Session start time
}
```

## Telemetry Data

### TelemetryPoint

High-frequency car telemetry measurement (10-60 Hz).

```python
{
    "timestamp": datetime,              # Measurement time (microsecond precision)
    "lap_number": int,                  # Current lap
    "distance": float,                  # Distance around track (meters)
    "speed": float,                     # Speed (km/h)
    "throttle": float,                  # Throttle position (0-100%)
    "brake": float,                     # Brake pressure (0-100%)
    "gear": int,                        # Current gear (1-8, 0=neutral, -1=reverse)
    "rpm": int,                         # Engine RPM
    "drs": int,                         # DRS status (0=closed, 1=open)
    "engine_temp": float,               # Engine temperature (°C)
    "brake_temp": Dict[str, float],     # Brake temps by corner: {"FL": 300, "FR": 300, "RL": 250, "RR": 250}
    "tire_temp": Dict[str, float]       # Tire temps by corner: {"FL": 85, "FR": 85, "RL": 80, "RR": 80}
}
```

**Validation Rules**:
- `speed`: 0-380 km/h
- `throttle`, `brake`: 0-100%
- `throttle + brake < 150` (both can't be high simultaneously)
- `gear`: -1 to 8
- `rpm`: 0-15000
- `drs`: 0 or 1
- `engine_temp`: 70-120°C
- `brake_temp`: 0-1200°C
- `tire_temp`: 20-130°C

### TelemetrySession

Complete telemetry data for a session.

```python
{
    "session_id": str,                      # Unique session identifier
    "driver": str,                          # Driver identifier
    "track_name": str,                      # Circuit name
    "telemetry_points": List[TelemetryPoint],  # All telemetry data
    "timestamp": datetime                   # Session start time
}
```

**Note**: Telemetry sessions can contain 10,000-100,000+ points. Stored as compressed Parquet for efficiency.

## Historical Data

### HistoricalRace

Complete historical race data from FastF1.

```python
{
    "year": int,                        # Season year
    "round": int,                       # Round number (1-23)
    "race_name": str,                   # "Monaco Grand Prix"
    "circuit_name": str,                # "Circuit de Monaco"
    "date": datetime,                   # Race date
    "laps": List[LapData],              # All lap data
    "results": List[DriverResult],      # Final results
    "pit_stops": List[PitStop],         # All pit stops
    "weather": WeatherData,             # Race weather
    "compound_history": List[TireStint] # Tire strategy history
}
```

### HistoricalStrategy

Extracted strategy information.

```python
{
    "driver": str,                      # Driver identifier
    "race_id": str,                     # Race identifier
    "stints": List[TireStint],          # All tire stints
    "pit_stop_laps": List[int],         # Laps with pit stops
    "final_position": int,              # Final position
    "strategy_type": str                # "One-stop", "Two-stop", "Three-stop"
}
```

### TireStint

Single tire stint information.

```python
{
    "stint_number": int,                # Stint number (1-based)
    "compound": str,                    # Tire compound
    "start_lap": int,                   # First lap of stint
    "end_lap": int,                     # Last lap of stint
    "duration": int,                    # Stint length (laps)
    "average_lap_time": float,          # Average lap time (seconds)
    "degradation_rate": float           # Degradation (seconds per lap)
}
```

## Safety Car Data

### SafetyCarEvent

Safety car, VSC, or red flag event.

```python
{
    "event_type": str,                  # "SAFETY_CAR", "VIRTUAL_SAFETY_CAR", "RED_FLAG"
    "start_lap": int,                   # Starting lap
    "end_lap": int,                     # Ending lap
    "duration_laps": int,               # Duration (laps)
    "reason": str,                      # Event reason
    "incident": IncidentLog,            # Associated incident (optional)
    "timestamp": datetime               # Event start time
}
```

**Event Types**:
- `SAFETY_CAR`: Physical SC on track
- `VIRTUAL_SAFETY_CAR`: VSC (delta time enforcement)
- `RED_FLAG`: Session stopped

### IncidentLog

Incident causing safety car deployment.

```python
{
    "lap_number": int,                  # Lap of incident
    "sector": int,                      # Sector (1-3)
    "description": str,                 # Incident description
    "drivers_involved": List[str],      # Involved drivers
    "severity": str                     # "LOW", "MEDIUM", "HIGH"
}
```

## Storage Formats

### Parquet (Primary)

```python
# Default compression: Snappy
# Schema embedded in file
# Column-based storage for efficient queries
# Supports nested structures

storage_manager.save_raw(
    source="timing",
    data=df,
    format="parquet"
)
```

**File naming**: `data/raw/timing/2024-03-15/20240315_143022.parquet`

### JSON (Metadata)

```python
# Human-readable metadata
# Version manifests
# Configuration snapshots

{
    "version": "v1.0.0_20240315_143022",
    "timestamp": "2024-03-15T14:30:22Z",
    "source": "timing",
    "checksum": "a1b2c3...",
    "records": 1560,
    "size_bytes": 245000
}
```

### PostgreSQL Tables

#### timing_data

```sql
CREATE TABLE timing_data (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    driver VARCHAR(10) NOT NULL,
    lap_number INTEGER NOT NULL,
    lap_time REAL NOT NULL,
    sector1_time REAL,
    sector2_time REAL,
    sector3_time REAL,
    position INTEGER,
    gap_to_leader REAL,
    tire_compound VARCHAR(20),
    tire_age INTEGER,
    is_pit_lap BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMPTZ NOT NULL,
    ingestion_time TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(session_id, driver, lap_number)
);

CREATE INDEX idx_timing_session ON timing_data(session_id);
CREATE INDEX idx_timing_driver ON timing_data(driver);
CREATE INDEX idx_timing_timestamp ON timing_data(timestamp);
```

#### weather_data

```sql
CREATE TABLE weather_data (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    track_name VARCHAR(100) NOT NULL,
    air_temperature REAL NOT NULL,
    track_temperature REAL NOT NULL,
    humidity REAL,
    pressure REAL,
    wind_speed REAL,
    wind_direction INTEGER,
    rainfall REAL,
    weather_condition VARCHAR(50),
    timestamp TIMESTAMPTZ NOT NULL,
    ingestion_time TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_weather_session ON weather_data(session_id);
CREATE INDEX idx_weather_timestamp ON weather_data(timestamp);
```

## Validation Summary

| Field Type | Validation Method |
|------------|------------------|
| Lap times | Range (30-150s), Sector sum consistency |
| Temperatures | Range (10-60°C), Track > Air |
| Positions | Range (1-20), Uniqueness per lap |
| Tire age | Non-negative, Monotonic increase |
| Speeds | Range (0-380 km/h) |
| Throttle/Brake | Range (0-100%), Mutual exclusivity |
| Timestamps | No future dates, Temporal consistency |
| Statistical | Z-score anomaly detection (threshold=3.0) |

## QA Reports

### QAReport Structure

```python
{
    "passed": bool,                     # Overall pass/fail
    "total_records": int,               # Total records checked
    "valid_records": int,               # Valid records
    "failed_records": int,              # Failed records
    "anomalies_detected": int,          # Anomalies found
    "warnings": List[str],              # Warning messages
    "critical_failures": List[str],     # Critical failures
    "statistics": Dict[str, Any],       # Source-specific stats
    "timestamp": datetime               # Report generation time
}
```

### Statistics by Source

**Timing**:
```python
{
    "mean_lap_time": 72.5,
    "std_lap_time": 1.2,
    "fastest_lap": 70.8,
    "slowest_lap": 85.3,
    "completed_laps": 78
}
```

**Weather**:
```python
{
    "mean_air_temp": 24.5,
    "mean_track_temp": 35.2,
    "humidity_range": [60, 70],
    "total_rainfall": 0.0
}
```

**Telemetry**:
```python
{
    "samples": 45000,
    "duration": 75.0,
    "mean_speed": 180.5,
    "max_speed": 345.2,
    "throttle_time_percent": 65.3
}
```

## Usage Examples

### Load Timing Data

```python
# From Parquet
import pandas as pd

df = pd.read_parquet("data/raw/timing/2024-03-15/20240315_143022.parquet")

# From Database
from sqlalchemy import create_engine

engine = create_engine(DATABASE_URL)
df = pd.read_sql_query(
    "SELECT * FROM timing_data WHERE session_id = %s",
    engine,
    params=["MON_2024_R"]
)
```

### Validate Data

```python
from data_pipeline.schemas.timing_schema import SessionTiming

# Validate against schema
timing_data = SessionTiming(**raw_data)

# Custom validation
from app.utils.validators import validate_lap_time

is_valid = validate_lap_time(72.5, track="Monaco")
```

### Query Telemetry

```python
# Load telemetry for specific lap
telemetry_df = pd.read_parquet("data/raw/telemetry/2024-03-15/20240315_143022.parquet")

lap_5 = telemetry_df[telemetry_df['lap_number'] == 5]

# Plot speed trace
import matplotlib.pyplot as plt
plt.plot(lap_5['distance'], lap_5['speed'])
plt.xlabel('Distance (m)')
plt.ylabel('Speed (km/h)')
plt.title('Speed Trace - Lap 5')
plt.show()
```

## Schema Evolution

When schemas change:

1. **Backward Compatible Changes** (add optional fields):
   - No version bump needed
   - Existing data remains valid

2. **Breaking Changes** (remove/rename fields, change types):
   - Increment major version
   - Create migration script
   - Update QA engine validation

3. **Migration Process**:
```python
# Example migration
from data_pipeline.utils.versioning import DataVersionManager

version_mgr = DataVersionManager()

# Tag current version
version_mgr.create_version(
    source="timing",
    data_path="data/raw/timing/latest.parquet",
    version_type="major",
    description="Schema v2.0: Added tire_pressure field"
)

# Apply migration
# ... transform data ...

# Save new version
storage_manager.save_processed(source="timing", data=migrated_df)
```

## Reference

- Schema definitions: `data_pipeline/schemas/`
- Validators: `app/utils/validators.py`
- QA engine: `data_pipeline/base/qa_engine.py`
- Storage manager: `data_pipeline/base/storage_manager.py`
