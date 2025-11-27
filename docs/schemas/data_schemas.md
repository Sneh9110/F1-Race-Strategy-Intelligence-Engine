# Data Schemas Documentation

## Overview

This document provides comprehensive documentation for all data schemas used in the F1 Race Strategy Intelligence Engine. All schemas are defined using Pydantic for runtime validation and automatic JSON Schema generation.

## Schema Categories

1. **Timing Data**: Lap times, sector times, positions
2. **Weather Data**: Track conditions, forecasts
3. **Telemetry Data**: High-frequency car data
4. **Historical Data**: Past race results and strategies
5. **Safety Car Events**: Race interruptions and incidents

---

## 1. Timing Schemas (`data_pipeline/schemas/timing_schema.py`)

### TimingPoint

Real-time timing snapshot for a single driver at a specific moment.

**Fields**:
- `timestamp` (datetime): UTC timestamp
- `driver_number` (int): Racing number (1-99)
- `position` (int): Current race position
- `lap_number` (int): Current lap
- `lap_time` (float, optional): Lap time in seconds (30-150s valid range)
- `sector_1_time`, `sector_2_time`, `sector_3_time` (float, optional): Sector times in seconds
- `speed_trap` (float, optional): Speed trap km/h (0-380)
- `gap_to_leader` (float): Gap to P1 in seconds
- `interval_to_ahead` (float, optional): Gap to car ahead

**Validation Rules**:
- Leader (position 1) must have gap_to_leader = 0
- Sector times must be 5-60 seconds each
- Lap time must equal sector sum within 100ms tolerance

**Example**:
```json
{
  "timestamp": "2024-05-26T14:23:45.123Z",
  "driver_number": 1,
  "position": 1,
  "lap_number": 23,
  "lap_time": 83.456,
  "sector_1_time": 28.123,
  "sector_2_time": 30.234,
  "sector_3_time": 25.099,
  "speed_trap": 312.5,
  "gap_to_leader": 0.0,
  "interval_to_ahead": null
}
```

### LapData

Complete data for a single completed lap.

**Fields**:
- All timing fields from TimingPoint
- `tire_compound` (enum): SOFT/MEDIUM/HARD/INTERMEDIATE/WET
- `tire_age` (int): Laps on current tires
- `pit_in_time`, `pit_out_time` (float, optional): Pit lane timestamps
- `is_personal_best` (bool): Personal best lap flag
- `track_status` (enum): GREEN/YELLOW/RED/SAFETY_CAR/VIRTUAL_SAFETY_CAR

**Use Case**: Store complete lap history for analysis

### SessionTiming

Aggregated timing data for entire session.

**Fields**:
- `session_id` (str): Unique identifier (e.g., "2024_MONACO_RACE")
- `session_type` (str): FP1/FP2/FP3/QUALIFYING/SPRINT/RACE
- `track_name` (str): Circuit name
- `session_start_time` (datetime): Start timestamp
- `lap_data` (List[LapData]): All laps from all drivers
- `timestamp` (datetime): Data capture time

**Example**: See `data/sample/timing_sample.json`

---

## 2. Weather Schemas (`data_pipeline/schemas/weather_schema.py`)

### WeatherData

Real-time weather observation from track sensors.

**Fields**:
- `timestamp` (datetime): Observation time
- `track_temp_celsius` (float): Track surface temp (10-60°C)
- `air_temp_celsius` (float): Air temperature (0-50°C)
- `humidity_percent` (float): Relative humidity (0-100%)
- `wind_speed_kmh` (float): Wind speed (0-100 km/h)
- `wind_direction_degrees` (float): Wind direction (0-360°, 0=North)
- `rainfall_mm` (float): Rainfall amount (mm)
- `pressure_hpa` (float): Atmospheric pressure (950-1050 hPa)
- `track_condition` (enum): DRY/DAMP/WET/SOAKING

**Validation Rules**:
- Track temp typically > air temp
- Track cannot be DRY with significant rainfall (>1mm)
- Track cannot be SOAKING with 0mm rainfall

**Integration**: Updates every 5 minutes during sessions

**Example**: See `data/sample/weather_sample.json`

### WeatherForecast

Predictive weather data for future time windows.

**Fields**:
- `forecast_time` (datetime): Time being forecast
- `generated_at` (datetime): When forecast was created
- `predicted_track_temp_celsius`, `predicted_air_temp_celsius` (float): Temperature predictions
- `predicted_rainfall_mm` (float): Expected rainfall
- `rain_probability_percent` (float): Probability of rain (0-100%)
- `predicted_track_condition` (enum): Expected track state
- `confidence_score` (float): Forecast confidence (0-1)

**Use Case**: Strategic planning and risk assessment

---

## 3. Telemetry Schemas (`data_pipeline/schemas/telemetry_schema.py`)

### TelemetryPoint

High-frequency telemetry snapshot (10-60 Hz sampling).

**Fields**:
- `timestamp` (datetime): Capture time
- `driver_number` (int): Driver number
- `speed_kmh` (float): Speed (0-380 km/h)
- `throttle_percent`, `brake_percent` (float): Pedal inputs (0-100%)
- `gear` (int): Current gear (0-8)
- `rpm` (int): Engine RPM (0-15000)
- `drs_status` (enum): CLOSED/AVAILABLE/ACTIVE/DISABLED
- `tire_temp_fl`, `tire_temp_fr`, `tire_temp_rl`, `tire_temp_rr` (float): Tire temps (50-130°C)
- `brake_temp_fl`, `brake_temp_fr`, `brake_temp_rl`, `brake_temp_rr` (float): Brake temps (100-1200°C)
- `fuel_remaining_kg` (float, optional): Fuel load (0-110 kg)
- `position_x`, `position_y` (float, optional): Track position coordinates

**Validation Rules**:
- Throttle and brake cannot both be >50% simultaneously
- RPM must be reasonable for current gear and speed

**Example**: See `data/sample/telemetry_sample.json` (20 points showing corner sequence)

### TelemetryLapSummary

Aggregated telemetry summary for complete lap (for efficiency).

**Fields**: Max/avg values for speed, RPM, throttle, brake, temperatures, DRS activations, fuel consumed

**Use Case**: Quick lap analysis without processing full high-frequency data

---

## 4. Historical Schemas (`data_pipeline/schemas/historical_schema.py`)

### HistoricalRace

Complete race metadata and outcome.

**Fields**:
- `race_id` (str): Unique identifier
- `year` (int): Season year
- `round_number` (int): Championship round
- `track_name` (str): Circuit name
- `race_date` (date): Race date
- `winner_driver_number`, `winner_name` (str): Winner info
- `total_laps` (int): Race distance
- `race_duration_seconds` (float): Total race time
- `weather_conditions` (enum): DRY/MIXED/WET
- `safety_car_laps`, `vsc_laps`, `red_flag_laps` (List[int]): Disruption laps

**Example**: See `data/sample/historical_race_sample.json`

### HistoricalStint

Detailed stint performance for tire analysis.

**Fields**:
- `driver_number`, `driver_name` (str): Driver info
- `stint_number` (int): Stint sequence (1st, 2nd, etc.)
- `tire_compound` (enum): Tire type used
- `start_lap`, `end_lap` (int): Stint boundaries
- `lap_times` (List[float]): All lap times in stint
- `avg_pace` (float): Average lap time
- `degradation_rate` (float): Pace loss per lap (seconds/lap)
- `pit_stop_duration` (float, optional): Pit stop time

**Validation**: Lap times list length must match stint length (end_lap - start_lap + 1)

**Use Case**: Training tire degradation models

### HistoricalStrategy

Complete race strategy for one driver.

**Fields**:
- `race_id` (str): Race reference
- `driver_number`, `driver_name`, `team_name` (str): Driver/team info
- `final_position` (int): Finishing position
- `num_pit_stops` (int): Total stops
- `stints` (List[HistoricalStint]): All stint details
- `total_race_time` (float): Race duration
- `strategy_type` (str): Classification (e.g., "2-STOP")

**Validation**: num_pit_stops must equal len(stints) - 1

---

## 5. Safety Car Schemas (`data_pipeline/schemas/safety_car_schema.py`)

### SafetyCarEvent

Race control intervention event.

**Fields**:
- `event_id` (str): Unique identifier
- `event_type` (enum): SAFETY_CAR/VIRTUAL_SAFETY_CAR/RED_FLAG/YELLOW_FLAG/DOUBLE_YELLOW_FLAG
- `start_lap`, `end_lap` (int): Event duration
- `start_time`, `end_time` (datetime): Timestamps
- `reason` (str): Explanation
- `affected_sectors` (List[int]): Sectors impacted (1, 2, 3)
- `deployment_duration_seconds` (float): Total time

**Validation**: Duration must be realistic for lap count

**Example**: See `data/sample/safety_car_sample.json`

### IncidentLog

Detailed incident causing race control action.

**Fields**:
- `incident_id` (str): Unique identifier
- `lap_number` (int): When it occurred
- `turn_number` (int, optional): Corner number
- `sector_number` (int): Track sector
- `drivers_involved` (List[int]): Driver numbers
- `incident_type` (enum): COLLISION/SPIN/DEBRIS/MECHANICAL_FAILURE/OFF_TRACK/BARRIER_CONTACT/OTHER
- `severity` (enum): LOW/MEDIUM/HIGH/CRITICAL
- `marshalling_required` (bool): Marshal intervention needed
- `safety_car_triggered` (bool): Whether SC deployed
- `description` (str): Detailed explanation
- `investigation_notes` (str, optional): Stewards notes

**Use Case**: Train safety car probability models, analyze high-risk track areas

---

## Schema Relationships

```
SessionTiming
  ├─ LapData (many)
  │   └─ TimingPoint (embedded)
  └─ Related to WeatherSession (by session_id)

WeatherSession
  ├─ WeatherData (many observations)
  └─ WeatherForecast (many predictions)

TelemetrySession
  └─ TelemetryPoint (thousands, high frequency)

HistoricalRace
  └─ HistoricalStrategy (many, one per driver)
      └─ HistoricalStint (many, one per tire set)

SafetyCarSession
  ├─ SafetyCarEvent (many)
  └─ IncidentLog (many)
```

## Common Validation Patterns

### Time Ranges
- Lap times: 30-150 seconds (track-dependent)
- Sector times: 5-60 seconds
- Pit stop duration: 1.8-5.0 seconds (normal range)

### Temperature Ranges
- Track temperature: 10-60°C
- Air temperature: 0-50°C
- Tire temperature: 50-130°C
- Brake temperature: 100-1200°C

### Physical Constraints
- Speed: 0-380 km/h
- RPM: 0-15000
- Fuel: 0-110 kg
- Throttle/Brake: 0-100%

## Data Quality Checks

All schemas implement:
1. **Type validation**: Pydantic ensures correct types
2. **Range validation**: Field validators for realistic bounds
3. **Consistency checks**: Cross-field validation (e.g., sector sum = lap time)
4. **Enum validation**: Only allowed values accepted

## JSON Schema Export

Generate JSON Schema for any model:
```python
from data_pipeline.schemas.timing_schema import LapData
schema = LapData.model_json_schema()
```

## Integration with Pipeline

1. **Raw data arrives** (JSON, CSV, API response)
2. **Parse to Pydantic model** (validation happens automatically)
3. **Validation failure** → Log error, skip record
4. **Validation success** → Store in database
5. **Feature engineering** reads validated data

This ensures data quality throughout the pipeline.
