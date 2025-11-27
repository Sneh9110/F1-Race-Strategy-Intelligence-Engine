# Schema Examples and Use Cases

## Introduction

This guide provides practical examples of using the F1 data schemas in real-world scenarios. Each example includes context, sample data, and code snippets for common operations.

---

## 1. Processing Live Timing Data

### Scenario: Receiving timing data from FIA timing API

**Context**: During a race, timing data arrives every 100-200ms. We need to validate and store it.

**Sample Data** (from `data/sample/timing_sample.json`):
```json
{
  "timestamp": "2024-05-26T14:23:45.123Z",
  "driver_number": 1,
  "position": 1,
  "lap_number": 23,
  "lap_time": 73.456,
  "sector_1_time": 24.123,
  "sector_2_time": 25.234,
  "sector_3_time": 24.099,
  "speed_trap": 287.5,
  "gap_to_leader": 0.0,
  "interval_to_ahead": null
}
```

**Code**:
```python
from data_pipeline.schemas.timing_schema import TimingPoint
from datetime import datetime
import json

# Receive raw JSON from API
raw_data = json.loads(api_response)

try:
    # Validate and parse
    timing = TimingPoint(**raw_data)
    
    # Access validated data
    print(f"Driver {timing.driver_number} in P{timing.position}")
    print(f"Lap time: {timing.lap_time:.3f}s")
    
    # Store in database
    save_to_database(timing.model_dump())
    
except ValueError as e:
    logger.error(f"Invalid timing data: {e}")
    # Handle validation error (e.g., skip record, alert monitoring)
```

**Common Validation Errors**:
- Sector times don't sum to lap time → Check for missing data
- Gap to leader < 0 → Data corruption, discard
- Driver number out of range → Invalid driver ID

---

## 2. Analyzing Tire Degradation

### Scenario: Calculate degradation rate from historical stint data

**Sample Data** (from `data/sample/historical_race_sample.json`):
```json
{
  "driver_number": 1,
  "driver_name": "Max Verstappen",
  "stint_number": 1,
  "tire_compound": "MEDIUM",
  "start_lap": 1,
  "end_lap": 30,
  "lap_times": [88.234, 87.123, 87.045, 87.189, 87.401, ...],
  "avg_pace": 88.127,
  "degradation_rate": 0.031
}
```

**Code**:
```python
from data_pipeline.schemas.historical_schema import HistoricalStint
import numpy as np

# Load stint data
stint = HistoricalStint(**stint_data)

# Calculate degradation manually
laps = np.arange(len(stint.lap_times))
lap_times = np.array(stint.lap_times)

# Linear regression to find degradation slope
coefficients = np.polyfit(laps, lap_times, 1)
degradation_per_lap = coefficients[0]

print(f"{stint.tire_compound} stint: {degradation_per_lap:.3f}s/lap degradation")

# Compare with expected degradation from config
from config.tire_compounds import get_compound_specs
expected = get_compound_specs(stint.tire_compound)["degradation_rate_per_lap"]
print(f"Expected: {expected:.3f}s/lap")

# Flag unusual degradation
if abs(degradation_per_lap - expected) > 0.05:
    logger.warning(f"Unusual tire wear detected for {stint.driver_name}")
```

**Output**:
```
MEDIUM stint: 0.031s/lap degradation
Expected: 0.035s/lap
```

**Use Cases**:
- Predict remaining tire life
- Optimize pit stop window
- Compare team tire management

---

## 3. Weather-Based Strategy Adjustments

### Scenario: Adjust strategy when rain forecast changes

**Sample Data** (from `data/sample/weather_sample.json`):
```json
{
  "forecast_time": "2024-03-02T16:30:00Z",
  "generated_at": "2024-03-02T15:00:00Z",
  "predicted_track_temp_celsius": 42.0,
  "predicted_air_temp_celsius": 28.0,
  "predicted_rainfall_mm": 5.0,
  "rain_probability_percent": 70.0,
  "predicted_track_condition": "WET",
  "confidence_score": 0.75
}
```

**Code**:
```python
from data_pipeline.schemas.weather_schema import WeatherForecast

# Load forecast
forecast = WeatherForecast(**forecast_data)

# Decision logic
if forecast.rain_probability_percent > 60 and forecast.confidence_score > 0.7:
    print("⚠️ HIGH RAIN PROBABILITY - ADJUST STRATEGY")
    
    # Calculate pit stop window before rain
    current_lap = 15
    forecast_lap = estimate_lap_at_time(forecast.forecast_time)
    laps_until_rain = forecast_lap - current_lap
    
    print(f"Rain expected in {laps_until_rain} laps")
    
    if laps_until_rain < 5:
        print("🔴 URGENT: Box for intermediates NOW")
    elif laps_until_rain < 10:
        print("🟡 Prepare intermediate tires, box within 3 laps")
    else:
        print("🟢 Monitor, no immediate action")
        
    # Update strategy recommendation
    strategy_engine.update_rain_strategy(
        rain_lap=forecast_lap,
        probability=forecast.rain_probability_percent / 100,
        intensity=forecast.predicted_rainfall_mm
    )
```

**Output**:
```
⚠️ HIGH RAIN PROBABILITY - ADJUST STRATEGY
Rain expected in 8 laps
🟡 Prepare intermediate tires, box within 3 laps
```

---

## 4. Detecting Safety Car Opportunities

### Scenario: Identify optimal pit window during safety car

**Sample Data** (from `data/sample/safety_car_sample.json`):
```json
{
  "event_id": "2024_MONACO_SC_001",
  "event_type": "SAFETY_CAR",
  "start_lap": 12,
  "end_lap": 15,
  "start_time": "2024-05-26T14:35:12Z",
  "end_time": "2024-05-26T14:42:38Z",
  "reason": "Collision at Turn 4 - Sainz and Gasly",
  "affected_sectors": [1],
  "deployment_duration_seconds": 446.0
}
```

**Code**:
```python
from data_pipeline.schemas.safety_car_schema import SafetyCarEvent

# Safety car deployed
sc_event = SafetyCarEvent(**sc_data)

print(f"🟨 SAFETY CAR deployed on lap {sc_event.start_lap}")
print(f"Reason: {sc_event.reason}")

# Calculate pit stop advantage
normal_pit_loss = get_track_config("Monaco")["pit_loss_seconds"]  # ~16s
sc_pit_loss = 0.0  # Pit during SC = no time loss (field bunched)

time_saved = normal_pit_loss - sc_pit_loss
print(f"💰 Pit stop saves {time_saved:.1f} seconds during SC")

# Identify drivers who should pit
from strategy.decision_engine import evaluate_pit_opportunity

for driver in get_current_standings():
    tire_age = driver.current_tire_age
    tire_compound = driver.current_compound
    
    # Check if tires are due for change
    optimal_window = get_optimal_stint_length(tire_compound)
    
    if tire_age >= optimal_window - 5:
        print(f"✅ Driver {driver.number}: PIT NOW (tire age: {tire_age})")
    elif tire_age < optimal_window - 10:
        print(f"❌ Driver {driver.number}: STAY OUT (fresh tires, age: {tire_age})")
    else:
        print(f"⚠️ Driver {driver.number}: MARGINAL (consider track position)")
```

**Output**:
```
🟨 SAFETY CAR deployed on lap 12
Reason: Collision at Turn 4 - Sainz and Gasly
💰 Pit stop saves 16.0 seconds during SC
✅ Driver 1: PIT NOW (tire age: 23)
❌ Driver 11: STAY OUT (fresh tires, age: 7)
⚠️ Driver 16: MARGINAL (consider track position)
```

---

## 5. Telemetry Analysis for Lap Time Delta

### Scenario: Compare two laps to find time loss/gain

**Sample Data** (from `data/sample/telemetry_sample.json`):
```json
{
  "timestamp": "2024-07-07T14:15:23.000Z",
  "driver_number": 1,
  "speed_kmh": 312.5,
  "throttle_percent": 100.0,
  "brake_percent": 0.0,
  "gear": 8,
  "rpm": 11500,
  "drs_status": "ACTIVE"
}
```

**Code**:
```python
from data_pipeline.schemas.telemetry_schema import TelemetryPoint
import pandas as pd

# Load two laps of telemetry
lap_1_data = load_telemetry_for_lap(driver=1, lap=23)  # Reference lap
lap_2_data = load_telemetry_for_lap(driver=1, lap=24)  # Comparison lap

# Parse to models
lap_1_points = [TelemetryPoint(**p) for p in lap_1_data]
lap_2_points = [TelemetryPoint(**p) for p in lap_2_data]

# Convert to DataFrames for analysis
df1 = pd.DataFrame([p.model_dump() for p in lap_1_points])
df2 = pd.DataFrame([p.model_dump() for p in lap_2_points])

# Compare metrics
print("Speed Analysis:")
print(f"Lap 1 max speed: {df1['speed_kmh'].max():.1f} km/h")
print(f"Lap 2 max speed: {df2['speed_kmh'].max():.1f} km/h")
print(f"Delta: {df2['speed_kmh'].max() - df1['speed_kmh'].max():+.1f} km/h")

print("\nBraking Analysis:")
print(f"Lap 1 max brake: {df1['brake_percent'].max():.0f}%")
print(f"Lap 2 max brake: {df2['brake_percent'].max():.0f}%")

print("\nCorner Exit (Throttle application):")
throttle_on_1 = df1[df1['throttle_percent'] > 95].shape[0]
throttle_on_2 = df2[df2['throttle_percent'] > 95].shape[0]
print(f"Lap 1: Full throttle for {throttle_on_1} samples")
print(f"Lap 2: Full throttle for {throttle_on_2} samples")

# Tire temperature evolution
print("\nTire Temperature:")
print(f"Lap 1 avg FL: {df1['tire_temp_fl'].mean():.1f}°C")
print(f"Lap 2 avg FL: {df2['tire_temp_fl'].mean():.1f}°C")
```

**Output**:
```
Speed Analysis:
Lap 1 max speed: 312.5 km/h
Lap 2 max speed: 308.2 km/h
Delta: -4.3 km/h

Braking Analysis:
Lap 1 max brake: 100%
Lap 2 max brake: 100%

Corner Exit (Throttle application):
Lap 1: Full throttle for 124 samples
Lap 2: Full throttle for 118 samples

Tire Temperature:
Lap 1 avg FL: 98.3°C
Lap 2 avg FL: 102.7°C
```

**Interpretation**: Lap 2 is slower due to higher tire temps (overheating) and reduced time at full throttle (traction loss).

---

## 6. Complete Race Strategy Evaluation

### Scenario: Compare different strategies from historical data

**Sample Data** (from `data/sample/historical_race_sample.json`):
```json
{
  "race_id": "2023_ABU_DHABI_RACE",
  "driver_number": 1,
  "driver_name": "Max Verstappen",
  "team_name": "Red Bull Racing",
  "final_position": 1,
  "num_pit_stops": 1,
  "stints": [
    {
      "stint_number": 1,
      "tire_compound": "MEDIUM",
      "start_lap": 1,
      "end_lap": 30,
      "lap_times": [...],
      "avg_pace": 88.127,
      "degradation_rate": 0.031
    },
    {
      "stint_number": 2,
      "tire_compound": "HARD",
      "start_lap": 31,
      "end_lap": 58,
      "lap_times": [...],
      "avg_pace": 88.456,
      "degradation_rate": 0.019
    }
  ],
  "total_race_time": 5127.34,
  "strategy_type": "1-STOP"
}
```

**Code**:
```python
from data_pipeline.schemas.historical_schema import HistoricalStrategy

# Load multiple strategies
strategies = [
    HistoricalStrategy(**data) for data in load_race_strategies("2023_ABU_DHABI_RACE")
]

# Analyze each strategy
results = []
for strategy in strategies:
    # Calculate total pit time
    total_pit_time = sum(stint.pit_stop_duration or 0 for stint in strategy.stints)
    
    # Calculate average pace (weighted by stint length)
    total_laps = sum(stint.end_lap - stint.start_lap + 1 for stint in strategy.stints)
    weighted_pace = sum(
        stint.avg_pace * (stint.end_lap - stint.start_lap + 1) 
        for stint in strategy.stints
    ) / total_laps
    
    # Calculate tire degradation impact
    total_deg = sum(stint.degradation_rate for stint in strategy.stints)
    
    results.append({
        "driver": strategy.driver_name,
        "strategy": strategy.strategy_type,
        "position": strategy.final_position,
        "race_time": strategy.total_race_time,
        "num_stops": strategy.num_pit_stops,
        "avg_pace": weighted_pace,
        "total_deg": total_deg,
        "pit_time": total_pit_time
    })

# Sort by finishing position
results.sort(key=lambda x: x["position"])

# Display comparison
print("Strategy Comparison:")
print("-" * 80)
for r in results:
    print(f"P{r['position']}: {r['driver']:20s} | {r['strategy']:10s} | "
          f"Race: {r['race_time']:.2f}s | Pace: {r['avg_pace']:.3f}s | "
          f"Stops: {r['num_stops']}")

# Find optimal strategy
winner = results[0]
print(f"\n✅ Winning strategy: {winner['strategy']} (by {winner['driver']})")
print(f"Key factors: {winner['num_stops']} stop(s), avg pace {winner['avg_pace']:.3f}s")
```

**Output**:
```
Strategy Comparison:
--------------------------------------------------------------------------------
P1: Max Verstappen       | 1-STOP     | Race: 5127.34s | Pace: 88.265s | Stops: 1
P2: Sergio Perez         | 2-STOP     | Race: 5139.12s | Pace: 87.932s | Stops: 2
P3: Charles Leclerc      | 2-STOP     | Race: 5145.67s | Pace: 88.145s | Stops: 2
P8: Lando Norris         | 3-STOP     | Race: 5178.23s | Pace: 87.823s | Stops: 3

✅ Winning strategy: 1-STOP (by Max Verstappen)
Key factors: 1 stop(s), avg pace 88.265s
```

**Insight**: Despite having slightly slower average pace than Perez (88.265 vs 87.932), Verstappen's 1-stop strategy saved ~22 seconds in pit time, winning the race.

---

## 7. Real-Time Session Aggregation

### Scenario: Build live leaderboard from incoming timing data

**Code**:
```python
from data_pipeline.schemas.timing_schema import SessionTiming, LapData
from datetime import datetime
from collections import defaultdict

class LiveSessionTracker:
    def __init__(self, session_id: str, track_name: str):
        self.session_data = {
            "session_id": session_id,
            "session_type": "RACE",
            "track_name": track_name,
            "session_start_time": datetime.utcnow(),
            "lap_data": [],
            "timestamp": datetime.utcnow()
        }
        self.driver_current_laps = defaultdict(dict)
    
    def process_timing_point(self, timing_point: dict):
        """Process incoming timing point and accumulate lap data"""
        driver_num = timing_point["driver_number"]
        lap_num = timing_point["lap_number"]
        
        # Accumulate sector times
        if lap_num not in self.driver_current_laps[driver_num]:
            self.driver_current_laps[driver_num][lap_num] = {}
        
        self.driver_current_laps[driver_num][lap_num].update(timing_point)
        
        # Check if lap is complete (has all 3 sectors)
        lap = self.driver_current_laps[driver_num][lap_num]
        if all(k in lap for k in ["sector_1_time", "sector_2_time", "sector_3_time"]):
            # Lap complete, validate and store
            try:
                lap_data = LapData(**lap)
                self.session_data["lap_data"].append(lap_data)
                print(f"✓ Lap {lap_num} complete for driver {driver_num}: {lap_data.lap_time:.3f}s")
            except ValueError as e:
                print(f"✗ Invalid lap data: {e}")
    
    def get_session(self) -> SessionTiming:
        """Get validated SessionTiming object"""
        return SessionTiming(**self.session_data)
    
    def get_leaderboard(self):
        """Generate current leaderboard"""
        latest_laps = {}
        for lap in self.session_data["lap_data"]:
            driver = lap.driver_number
            if driver not in latest_laps or lap.lap_number > latest_laps[driver].lap_number:
                latest_laps[driver] = lap
        
        leaderboard = sorted(latest_laps.values(), key=lambda x: x.position)
        return leaderboard

# Usage
tracker = LiveSessionTracker("2024_MONACO_RACE", "Circuit de Monaco")

# Simulate incoming data
timing_updates = [
    {"driver_number": 1, "lap_number": 1, "position": 1, "sector_1_time": 24.123, ...},
    {"driver_number": 1, "lap_number": 1, "position": 1, "sector_2_time": 25.234, ...},
    {"driver_number": 1, "lap_number": 1, "position": 1, "sector_3_time": 24.099, ...},
]

for update in timing_updates:
    tracker.process_timing_point(update)

# Get current leaderboard
leaderboard = tracker.get_leaderboard()
for pos, lap in enumerate(leaderboard, 1):
    print(f"P{pos}: Driver {lap.driver_number} - Lap {lap.lap_number} - {lap.lap_time:.3f}s")
```

---

## Common Integration Patterns

### Pattern 1: API to Database
```python
# Receive from API → Validate → Store
raw_json = requests.get(api_url).json()
validated_model = TimingPoint(**raw_json)
db.insert(validated_model.model_dump())
```

### Pattern 2: Database to ML Model
```python
# Load from DB → Parse → Train
raw_records = db.query("SELECT * FROM historical_stints")
stints = [HistoricalStint(**r) for r in raw_records]
X, y = extract_features(stints)
model.fit(X, y)
```

### Pattern 3: Real-Time Enrichment
```python
# Timing + Weather → Decision
timing = TimingPoint(**timing_data)
weather = WeatherData(**weather_data)
decision = decision_engine.evaluate(timing, weather)
```

## Best Practices

1. **Always validate early**: Parse to Pydantic models as soon as data enters the system
2. **Handle validation errors gracefully**: Log errors, don't crash the pipeline
3. **Use model_dump()**: Convert to dict for database storage or JSON serialization
4. **Leverage model_json_schema()**: Generate API docs automatically
5. **Keep raw data**: Store original JSON alongside validated models for debugging

## Sample Data Files

All examples use data from:
- `data/sample/timing_sample.json`
- `data/sample/weather_sample.json`
- `data/sample/telemetry_sample.json`
- `data/sample/historical_race_sample.json`
- `data/sample/safety_car_sample.json`

These files contain realistic F1 data for testing and development.
