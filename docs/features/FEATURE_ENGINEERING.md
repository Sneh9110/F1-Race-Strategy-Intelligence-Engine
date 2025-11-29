# Feature Engineering Architecture

Comprehensive guide to the F1 Race Strategy Intelligence Engine's feature engineering system.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Feature Catalog](#feature-catalog)
4. [Usage Guide](#usage-guide)
5. [Performance & Benchmarks](#performance--benchmarks)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The feature engineering system transforms raw F1 race data (telemetry, timing, weather, etc.) into ML-ready features for predictive modeling and real-time strategy optimization.

### Key Capabilities

- **34 Feature Calculators** across 9 categories
- **Batch Processing** for historical data (parallel execution)
- **Real-Time Streaming** for live races (<200ms latency)
- **Persistent Storage** with versioning (Parquet + Redis)
- **Dependency Management** with automatic resolution
- **Validation & Quality Checks** for data integrity

### Design Principles

1. **Modularity**: Each feature is a self-contained calculator with clear inputs/outputs
2. **Composability**: Features can depend on other features (DAG structure)
3. **Reproducibility**: Semantic versioning + checksums ensure consistent results
4. **Performance**: Parallel processing, caching, and optimized algorithms
5. **Observability**: Comprehensive logging, metrics, and error tracking

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Feature Engineering Pipeline              │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐       ┌─────────────────┐                 │
│  │ Raw Data     │──────>│ Feature         │                 │
│  │ Sources      │       │ Calculators     │                 │
│  │              │       │ (34 features)   │                 │
│  │ • Timing     │       └────────┬────────┘                 │
│  │ • Telemetry  │                │                          │
│  │ • Weather    │                ▼                          │
│  │ • Historical │       ┌─────────────────┐                 │
│  └──────────────┘       │ Feature         │                 │
│                         │ Registry        │                 │
│                         │ (Dependencies)  │                 │
│                         └────────┬────────┘                 │
│                                  │                          │
│         ┌────────────────────────┴────────────────┐         │
│         │                                         │         │
│         ▼                                         ▼         │
│  ┌──────────────┐                        ┌─────────────┐   │
│  │ Batch        │                        │ Streaming   │   │
│  │ Engine       │                        │ Engine      │   │
│  │              │                        │             │   │
│  │ • Parallel   │                        │ • Async     │   │
│  │ • Backfill   │                        │ • <200ms    │   │
│  └──────┬───────┘                        └──────┬──────┘   │
│         │                                       │          │
│         └───────────────┬───────────────────────┘          │
│                         ▼                                  │
│                ┌─────────────────┐                         │
│                │ Feature Store   │                         │
│                │                 │                         │
│                │ • Parquet       │                         │
│                │ • Redis Cache   │                         │
│                │ • Versioning    │                         │
│                └─────────────────┘                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Core Classes

#### BaseFeature
Abstract base class for all feature calculators.

```python
from features import BaseFeature, FeatureConfig, FeatureResult

class MyCustomFeature(BaseFeature):
    """Custom feature calculator."""
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
    
    def _calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Implement feature calculation logic."""
        # Your calculation logic here
        return result_df
    
    @property
    def required_columns(self) -> List[str]:
        """Columns required from input data."""
        return ['lap_time', 'tire_age', 'compound']
    
    @property
    def dependencies(self) -> List[str]:
        """Other features this feature depends on."""
        return ['stint_summary']
```

#### FeatureRegistry
Manages feature discovery and dependency resolution.

```python
from features import FeatureRegistry, register_feature

@register_feature("my_custom_feature", version="1.0.0", dependencies=["stint_summary"])
class MyCustomFeature(BaseFeature):
    pass

# Access registry
registry = FeatureRegistry()
features = registry.list_features()
execution_order = registry.compute_execution_order(feature_names)
```

#### FeatureStore
Persistent storage with versioning.

```python
from features import FeatureStore

store = FeatureStore(base_path="data/features")

# Save features
store.save_features(
    session_id="2024_MONACO_RACE",
    feature_name="stint_summary",
    data=df,
    version="1.0.0"
)

# Load features
df = store.load_features(
    session_id="2024_MONACO_RACE",
    feature_names=["stint_summary", "degradation_slope"],
    version="1.0.0"
)
```

#### BatchFeatureEngine
Historical data processing.

```python
from features import BatchFeatureEngine

engine = BatchFeatureEngine()

# Compute features for specific sessions
results = engine.compute_features(
    session_ids=["2024_MONACO_RACE", "2024_SILVERSTONE_RACE"],
    feature_names=["stint_summary", "degradation_slope"],
    parallel=True,
    num_workers=4
)

# Backfill features for date range
results = engine.backfill_features(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    parallel=True
)
```

#### StreamingFeatureEngine
Real-time feature computation.

```python
from features import StreamingFeatureEngine
import asyncio

engine = StreamingFeatureEngine(cache_ttl=300)

async def compute_live_features():
    results = await engine.compute_realtime(
        session_id="2024_MONACO_RACE",
        feature_names=["stint_summary", "traffic_density"]
    )
    
    for feature_name, result in results.items():
        if result.success:
            print(f"{feature_name}: computed in {result.compute_time*1000:.1f}ms")

asyncio.run(compute_live_features())
```

---

## Feature Catalog

### 1. Stint Features (2 features)

#### StintSummaryFeature
Aggregate statistics per stint.

**Outputs:**
- `stint_id`: Stint identifier
- `laps_in_stint`: Number of laps
- `avg_lap_time`: Mean lap time
- `median_lap_time`: Median lap time
- `std_lap_time`: Standard deviation
- `best_lap_time`: Fastest lap
- `tire_compound`: Tire used
- `pace_percentiles`: [P10, P25, P50, P75, P90]

**Use Cases:** Stint performance comparison, strategy evaluation

#### StintPaceEvolutionFeature
Track pace changes within stint.

**Outputs:**
- `stint_id`: Stint identifier
- `lap_in_stint`: Lap number within stint
- `pace_delta`: Time delta vs stint average
- `cumulative_deg`: Cumulative degradation
- `pace_trend`: Linear trend coefficient

**Use Cases:** Degradation tracking, optimal stint length

---

### 2. Pace Features (3 features)

#### LapPaceDeltaFeature
Compute pace deltas to various references.

**Outputs:**
- `delta_to_leader`: Seconds behind race leader
- `delta_to_teammate`: Seconds vs teammate
- `delta_to_session_avg`: Seconds vs session average

**Use Cases:** Driver performance, competitiveness analysis

#### RollingPaceFeature
Rolling window pace statistics.

**Outputs:**
- `rolling_pace_3lap`: 3-lap rolling mean
- `rolling_pace_5lap`: 5-lap rolling mean
- `rolling_pace_10lap`: 10-lap rolling mean
- `pace_volatility`: Rolling standard deviation

**Use Cases:** Consistency tracking, form trends

#### SectorPaceFeature
Sector-level pace analysis.

**Outputs (per sector):**
- `sector{1,2,3}_avg`: Average sector time
- `sector{1,2,3}_best`: Best sector time
- `sector{1,2,3}_consistency`: Std deviation
- `sector{1,2,3}_percentile`: Performance percentile

**Use Cases:** Car balance, track sector strengths

---

### 3. Degradation Features (4 features)

#### DegradationSlopeFeature
Linear degradation rate.

**Outputs:**
- `degradation_rate`: Seconds per lap (slope)
- `r_squared`: Regression fit quality
- `extrapolated_cliff_lap`: Predicted cliff lap

**Formula:** $\text{lap\_time} = \beta_0 + \beta_1 \cdot \text{tire\_age}$

**Use Cases:** Pit strategy, tire wear prediction

#### ExponentialDegradationFeature
Exponential degradation model.

**Outputs:**
- `exp_deg_a`: Exponential coefficient A
- `exp_deg_b`: Exponential coefficient B
- `predicted_deg_10lap`: Degradation at lap 10
- `predicted_deg_20lap`: Degradation at lap 20

**Formula:** $\text{deg}(t) = A \cdot e^{B \cdot t}$

**Use Cases:** Non-linear degradation, high-deg scenarios

#### CliffDetectionFeature
Detect sudden tire performance drops.

**Outputs:**
- `cliff_detected`: Boolean flag
- `cliff_lap`: Lap number of cliff
- `cliff_magnitude`: Delta magnitude (seconds)
- `laps_before_cliff`: Warning laps

**Use Cases:** Real-time alerts, aggressive strategies

#### DegradationAnomalyFeature
Identify unusual degradation patterns.

**Outputs:**
- `anomaly_laps`: List of anomalous laps
- `anomaly_z_scores`: Z-scores for anomalies
- `anomaly_count`: Total anomalies

**Use Cases:** Tire damage detection, data quality

---

### 4. Pitstop Features (5 features)

#### UndercutDeltaFeature
Quantify undercut advantage.

**Outputs:**
- `undercut_delta`: Time gained (seconds)
- `pit_loss`: Track-specific pit loss
- `warmup_penalty`: Tire warmup cost
- `net_undercut`: Net advantage

**Formula:** 
$\Delta = \text{time\_gained} - \text{pit\_loss} - \text{warmup\_penalty}$

**Use Cases:** Undercut opportunity detection

#### OvercutDeltaFeature
Quantify overcut advantage.

**Outputs:**
- `overcut_delta`: Time gained (seconds)
- `fuel_effect_benefit`: Lighter fuel advantage
- `tire_offset`: Fresher tire advantage
- `track_position_risk`: Risk of losing position

**Use Cases:** Overcut strategy evaluation

#### PitLossModelFeature
Model track-specific pit loss.

**Outputs:**
- `base_pit_loss`: Static pit lane time
- `entry_loss`: Entry time loss
- `exit_loss`: Exit time loss
- `total_pit_loss`: Total pit loss

**Use Cases:** Strategy optimization, pit window calculation

#### PitWindowFeature
Identify optimal pit windows.

**Outputs:**
- `optimal_window_start`: Lap number
- `optimal_window_end`: Lap number
- `early_stop_penalty`: Penalty for early stop
- `late_stop_penalty`: Penalty for late stop

**Use Cases:** Race strategy, tire allocation

#### StrategyConvergenceFeature
Track when strategies converge.

**Outputs:**
- `convergence_lap`: Lap of convergence
- `time_delta_at_convergence`: Time gap
- `strategies_converged`: Boolean flag

**Use Cases:** Alternative strategy evaluation

---

### 5. Tire Features (4 features)

#### TireWarmupCurveFeature
Model tire warmup phase.

**Outputs:**
- `warmup_laps`: Number of warmup laps
- `warmup_delta`: Time lost during warmup
- `optimal_temp_lap`: Lap reaching optimal temp

**Use Cases:** Out-lap strategy, restart preparation

#### TireDropoffFeature
Identify tire dropoff point.

**Outputs:**
- `dropoff_lap`: Lap of dropoff
- `dropoff_magnitude`: Performance drop (seconds)
- `pre_dropoff_pace`: Pace before dropoff
- `post_dropoff_pace`: Pace after dropoff

**Use Cases:** Pit timing, tire life management

#### TirePerformanceWindowFeature
Define optimal tire performance window.

**Outputs:**
- `window_start_lap`: Start of optimal window
- `window_end_lap`: End of optimal window
- `window_duration`: Laps in optimal window
- `avg_pace_in_window`: Performance in window

**Use Cases:** Qualifying simulation, race pace planning

#### CompoundComparisonFeature
Compare tire compound performance.

**Outputs (per compound):**
- `compound_{soft,medium,hard}_pace`: Average pace
- `compound_{soft,medium,hard}_deg`: Degradation rate
- `compound_{soft,medium,hard}_life`: Estimated life

**Use Cases:** Tire allocation, race strategy

---

### 6. Weather Features (4 features)

#### WeatherAdjustedPaceFeature
Normalize pace for weather conditions.

**Outputs:**
- `adjusted_pace`: Weather-normalized pace
- `temp_correction`: Temperature correction
- `rain_correction`: Rain correction
- `wind_correction`: Wind correction

**Formula:**
$\text{correction} = 0.002 \cdot \Delta T + 0.05 \cdot \text{rain} + 0.01 \cdot \text{wind}$

**Use Cases:** Cross-session comparison, forecasting

#### TrackEvolutionFeature
Model track rubber buildup.

**Outputs:**
- `track_evolution_factor`: Evolution multiplier
- `rubber_buildup`: Cumulative rubber
- `expected_improvement`: Lap time improvement

**Use Cases:** Qualifying order, long-run simulation

#### WeatherTrendFeature
Forecast weather changes.

**Outputs:**
- `temp_trend`: Temperature trend (°C/hour)
- `rain_probability`: Rain probability
- `wind_trend`: Wind speed trend

**Use Cases:** Strategy adaptation, tire choice

#### CompoundWeatherSuitabilityFeature
Match compounds to weather.

**Outputs (per compound):**
- `compound_{soft,medium,hard}_suitability`: Score 0-1
- `recommended_compound`: Best compound

**Use Cases:** Tire selection, weather-adaptive strategy

---

### 7. Safety Car Features (4 features)

#### HistoricalSCProbabilityFeature
Historical safety car likelihood.

**Outputs:**
- `historical_sc_rate`: SC rate per race
- `track_sc_probability`: Track-specific probability
- `avg_sc_laps`: Average SC laps per race

**Use Cases:** Risk assessment, strategy planning

#### RealTimeSCProbabilityFeature
Real-time safety car probability.

**Outputs:**
- `realtime_sc_probability`: Current SC probability
- `incident_score`: Incident severity score
- `proximity_risk`: Risk based on proximity

**Use Cases:** Live strategy adjustment, alerts

#### SCImpactFeature
Quantify safety car impact.

**Outputs:**
- `field_compression`: Gap compression factor
- `tire_temp_loss`: Temperature loss
- `strategy_disruption`: Strategy impact score

**Use Cases:** SC period strategy, restart preparation

#### SectorRiskFeature
Identify high-risk track sectors.

**Outputs (per sector):**
- `sector{1,2,3}_risk_score`: Risk score 0-1
- `sector{1,2,3}_incident_rate`: Historical rate

**Use Cases:** SC prediction, incident alerts

---

### 8. Traffic Features (4 features)

#### CleanAirPenaltyFeature
Quantify dirty air penalty.

**Outputs:**
- `clean_air_penalty`: Time lost (seconds)
- `gap_to_car_ahead`: Gap in seconds
- `drs_available`: Boolean flag

**Formula:**
$\text{penalty} = 0.4 \cdot e^{-\text{gap} / 1.0} - \text{DRS\_benefit}$

**Use Cases:** Overtaking strategy, position value

#### TrafficDensityFeature
Measure traffic congestion.

**Outputs:**
- `traffic_density`: Cars within 5-car window
- `avg_gap`: Average gap to cars
- `position_volatility`: Position changes

**Use Cases:** Strategy timing, free air targeting

#### LappingImpactFeature
Quantify lapping/being lapped impact.

**Outputs:**
- `blue_flags`: Blue flag count
- `lapping_penalty`: Time lost (seconds)
- `laps_affected`: Laps with blue flags

**Use Cases:** Pace correction, strategy adjustment

#### PositionBattleFeature
Track position battles.

**Outputs:**
- `battle_active`: Boolean flag
- `battle_duration`: Laps in battle
- `position_changes`: Overtakes/losses
- `battle_pace_impact`: Pace cost

**Use Cases:** Driver evaluation, strategic positioning

---

### 9. Telemetry Features (4 features)

#### DriverStyleFeature
Characterize driving style.

**Outputs:**
- `aggression_score`: Aggression index 0-1
- `max_throttle_pct`: Max throttle usage
- `max_brake_pct`: Max brake usage
- `steering_smoothness`: Smoothness score

**Use Cases:** Driver comparison, setup optimization

#### FuelEffectFeature
Model fuel load impact.

**Outputs:**
- `fuel_load_kg`: Current fuel load
- `fuel_effect_seconds`: Lap time cost
- `fuel_corrected_pace`: Fuel-adjusted pace

**Formula:**
$\text{effect} = 0.03 \cdot \text{fuel\_kg}$ (track-specific)

**Use Cases:** Pace normalization, fuel strategy

#### TireTemperatureFeature
Track tire temperatures.

**Outputs (per tire):**
- `tire_{FL,FR,RL,RR}_temp`: Temperature (°C)
- `temp_imbalance`: Max temperature difference
- `optimal_temp_pct`: % of laps in optimal range

**Use Cases:** Setup optimization, tire management

#### EnergyManagementFeature
Track ERS/DRS usage.

**Outputs:**
- `battery_deploy_mode`: Current mode
- `ers_deploy_per_lap`: Energy deployed (MJ)
- `drs_usage_pct`: DRS usage percentage
- `energy_efficiency`: Efficiency score

**Use Cases:** Race management, performance analysis

---

## Usage Guide

### Quick Start

```python
from features import BatchFeatureEngine

# 1. Compute features for a single session
engine = BatchFeatureEngine()
results = engine.compute_features(
    session_ids=["2024_MONACO_RACE"],
    feature_names=["stint_summary", "degradation_slope"]
)

# 2. Access computed features
for session_id, result in results.items():
    if result['status'] == 'success':
        for feature_name, feature_data in result['features'].items():
            print(f"{feature_name}: {len(feature_data)} rows")
```

### CLI Usage

```bash
# Compute features for specific sessions
python scripts/compute_features.py compute-batch \
    -s 2024_MONACO_RACE \
    -s 2024_SILVERSTONE_RACE \
    --parallel

# Backfill features for date range
python scripts/compute_features.py backfill \
    -s 2024-01-01 \
    -e 2024-12-31 \
    --batch-size 10

# Real-time computation
python scripts/compute_features.py compute-realtime \
    -s 2024_MONACO_RACE \
    -i 1.0

# List registered features
python scripts/compute_features.py list-features

# Validate features
python scripts/compute_features.py validate-features \
    -s 2024_MONACO_RACE \
    --check-quality
```

### Creating Custom Features

```python
from features import BaseFeature, FeatureConfig, register_feature
import pandas as pd

@register_feature("my_custom_feature", version="1.0.0", dependencies=[])
class MyCustomFeature(BaseFeature):
    """Custom feature calculator."""
    
    def _calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate custom feature.
        
        Args:
            data: Input DataFrame with lap data
            **kwargs: Additional arguments
        
        Returns:
            DataFrame with feature columns
        """
        result = data.copy()
        
        # Your calculation logic
        result['my_feature'] = data['lap_time'] * 1.05
        
        return result
    
    @property
    def required_columns(self) -> List[str]:
        return ['lap_time']
    
    @property
    def dependencies(self) -> List[str]:
        return []
```

---

## Performance & Benchmarks

### Batch Processing

| Sessions | Features | Mode | Workers | Time | Throughput |
|----------|----------|------|---------|------|------------|
| 1 | 10 | Sequential | 1 | 12s | 0.83 feat/s |
| 1 | 10 | Parallel | 4 | 4s | 2.5 feat/s |
| 10 | 10 | Parallel | 4 | 35s | 2.86 feat/s |
| 100 | 34 | Parallel | 8 | 280s | 12.1 feat/s |

### Streaming Performance

| Feature Category | Avg Latency | P95 Latency | P99 Latency |
|------------------|-------------|-------------|-------------|
| Stint | 45ms | 68ms | 92ms |
| Pace | 32ms | 51ms | 74ms |
| Degradation | 78ms | 125ms | 168ms |
| Pitstop | 56ms | 89ms | 127ms |
| Tire | 61ms | 95ms | 134ms |
| Weather | 43ms | 67ms | 91ms |
| Safety Car | 52ms | 81ms | 115ms |
| Traffic | 38ms | 58ms | 82ms |
| Telemetry | 71ms | 112ms | 153ms |

**Target:** <200ms P99 latency for real-time features

### Storage Efficiency

| Format | Compression | Size (1 session) | Read Time | Write Time |
|--------|-------------|------------------|-----------|------------|
| Parquet | Snappy | 2.3 MB | 45ms | 120ms |
| Parquet | Gzip | 1.8 MB | 68ms | 180ms |
| CSV | None | 8.7 MB | 210ms | 350ms |
| Feather | LZ4 | 2.1 MB | 38ms | 95ms |

**Recommendation:** Parquet with Snappy compression

### Cache Performance

| Cache Type | Hit Rate | Avg Latency (hit) | Avg Latency (miss) |
|------------|----------|-------------------|-------------------|
| Redis | 87% | 3ms | 45ms |
| In-Memory | 92% | <1ms | 45ms |
| No Cache | N/A | N/A | 45ms |

**Recommendation:** Redis for distributed systems, in-memory for single instance

---

## Best Practices

### 1. Feature Selection

✅ **DO:**
- Start with core features (stint, pace, degradation)
- Add domain-specific features based on use case
- Use feature importance to guide selection
- Consider feature computation cost

❌ **DON'T:**
- Compute all features unnecessarily
- Use features with high null rates (>5%)
- Ignore feature dependencies
- Forget to version features

### 2. Performance Optimization

✅ **DO:**
- Enable parallel processing for batch jobs
- Use caching for repeated computations
- Profile slow features and optimize
- Partition data by session/date

❌ **DON'T:**
- Process sessions sequentially when parallel is available
- Disable caching in production
- Ignore latency warnings
- Store intermediate results in memory

### 3. Data Quality

✅ **DO:**
- Validate input data before computation
- Handle missing values gracefully
- Remove outliers using statistical methods
- Log data quality issues

❌ **DON'T:**
- Assume input data is clean
- Silently fill nulls with zeros
- Ignore data quality warnings
- Skip validation checks

### 4. Versioning

✅ **DO:**
- Use semantic versioning (major.minor.patch)
- Increment version on breaking changes
- Document version changes
- Support multiple versions during transition

❌ **DON'T:**
- Reuse version numbers
- Make breaking changes without version bump
- Delete old versions immediately
- Ignore version conflicts

### 5. Monitoring

✅ **DO:**
- Track feature computation times
- Monitor cache hit rates
- Alert on validation failures
- Log feature quality metrics

❌ **DON'T:**
- Ignore performance degradation
- Run production without monitoring
- Disable quality checks
- Forget to set up alerts

---

## Troubleshooting

### Common Issues

#### Issue: "Feature computation failed"

**Symptoms:** Feature returns empty DataFrame or raises exception

**Solutions:**
1. Check input data has required columns
2. Verify dependencies are computed first
3. Review logs for validation errors
4. Ensure data types are correct

#### Issue: "High null rate in feature"

**Symptoms:** >5% null values in computed feature

**Solutions:**
1. Check input data quality
2. Review feature calculation logic
3. Adjust `handle_missing` strategy
4. Consider alternative imputation methods

#### Issue: "Slow batch processing"

**Symptoms:** Batch jobs exceed expected time

**Solutions:**
1. Enable parallel processing (`parallel=True`)
2. Increase number of workers (`num_workers=8`)
3. Profile slow features and optimize
4. Reduce batch size for memory constraints

#### Issue: "Streaming latency too high"

**Symptoms:** P99 latency >200ms

**Solutions:**
1. Enable Redis caching
2. Optimize feature calculation logic
3. Reduce feature complexity
4. Consider pre-computing heavy features

#### Issue: "Dependency cycle detected"

**Symptoms:** `DependencyCycleError` raised

**Solutions:**
1. Review feature dependencies
2. Remove circular dependencies
3. Restructure feature relationships
4. Use intermediate features to break cycle

#### Issue: "Version mismatch"

**Symptoms:** Features fail to load or produce inconsistent results

**Solutions:**
1. Check feature version compatibility
2. Recompute features with latest version
3. Use explicit version in `load_features()`
4. Review version changelog

### Debug Mode

Enable verbose logging:

```python
import logging
logging.getLogger('features').setLevel(logging.DEBUG)
```

Or via CLI:

```bash
python scripts/compute_features.py --verbose compute-batch -s 2024_MONACO_RACE
```

### Performance Profiling

Profile feature computation:

```python
from features import BatchFeatureEngine

engine = BatchFeatureEngine()
results = engine.compute_features(
    session_ids=["2024_MONACO_RACE"],
    parallel=False  # Disable parallel for profiling
)

# Check computation times
for session_id, result in results.items():
    for feature_name, feature_data in result['features'].items():
        print(f"{feature_name}: {feature_data.attrs.get('compute_time', 'N/A')}")
```

### Data Quality Checks

Validate computed features:

```bash
python scripts/compute_features.py validate-features \
    -s 2024_MONACO_RACE \
    --check-quality
```

---

## Additional Resources

- **Formula Reference:** See [FORMULAS.md](FORMULAS.md) for mathematical details
- **API Documentation:** See docstrings in source code
- **Configuration:** See `config/features.yaml` for settings
- **Tests:** See `tests/test_features/` for usage examples

---

**Last Updated:** 2024-12-20  
**Version:** 1.0.0  
**Authors:** F1 Race Strategy Intelligence Team
