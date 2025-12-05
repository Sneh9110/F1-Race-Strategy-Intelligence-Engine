# Decision Engine Guide

## Overview

The Decision Engine is the strategic brain of the F1 Race Strategy Intelligence platform. It consumes simulation outputs, ML predictions, and real-time race data to generate actionable recommendations for:

- **Pit Timing:** When to pit NOW vs wait
- **Strategy Conversion:** Switch between 1-stop, 2-stop, 3-stop strategies
- **Offset Strategy:** Use different tire compounds vs rivals
- **Safety Car Decisions:** Pit under SC or stay out (time-critical)
- **Rain Strategy:** Switch to intermediates/wets
- **Pace Monitoring:** Detect pace drops, recommend adjustments
- **Undercut/Overcut:** Trigger tactical opportunities

## Quick Start

### Python API

```python
from decision_engine import DecisionEngine, DecisionInput, DecisionContext

# Initialize engine
engine = DecisionEngine(config_path="config/decision_engine.yaml")

# Create decision context
context = DecisionContext(
    session_id="2024_MONACO",
    lap_number=25,
    driver_number=44,
    track_name="Monaco",
    total_laps=78,
    current_position=3,
    tire_age=20,
    tire_compound="MEDIUM",
    fuel_load=85.0,
    stint_number=2,
    pit_stops_completed=1,
    gap_to_ahead=2.5,
    recent_lap_times=[90.5, 90.8, 91.2, 91.5, 91.8],
)

# Make decision
decision_input = DecisionInput(context=context)
output = engine.make_decision(decision_input)

# Access recommendations
for rec in output.recommendations:
    print(f"Action: {rec.action.value}")
    print(f"Confidence: {rec.confidence_score:.2%} ({rec.traffic_light.value})")
    print(f"Expected Gain: {rec.expected_gain_seconds:.1f}s")
    print(f"Reasoning: {rec.reasoning.primary_factors}")
```

### CLI

```bash
# Single decision
python scripts/run_decision_engine.py decide \
  --session-id 2024_MONACO \
  --lap-number 25 \
  --driver-number 44 \
  --track-name Monaco

# Batch decisions (all drivers)
python scripts/run_decision_engine.py decide-batch \
  --session-id 2024_MONACO \
  --lap-number 25

# Benchmark performance
python scripts/run_decision_engine.py benchmark --num-runs 1000

# List modules
python scripts/run_decision_engine.py list-modules
```

## Decision Modules

### 1. Pit Timing Decision (Priority: 9)

**When to use:** Every lap to assess pit timing

**Triggers:**
- **PIT_NOW:** Tire age > optimal stint length, degradation > 0.1s/lap, pace drop > 0.5s, tire cliff detected, undercut opportunity
- **STAY_OUT:** Just pitted (<5 laps), SC imminent (probability > 0.6), tire life > 10 laps, overcut opportunity

**Confidence:** High when multiple factors align (tire cliff + pit window + low SC risk)

**Example:**
```python
# Tire cliff scenario
context = DecisionContext(
    tire_age=28,
    tire_compound='SOFT',
    recent_lap_times=[90.5, 90.8, 91.2, 92.5, 94.8],  # Sudden 2s drop
    ...
)
# → Recommends PIT_NOW with HIGH confidence (GREEN light)
```

### 2. Safety Car Decision (Priority: 10)

**When to use:** When SC deployed or SC probability > 0.5

**Triggers:**
- **PIT_UNDER_SC:** SC just deployed, tire age > 15 laps, pit loss reduced (~50%), rivals staying out, not in podium
- **STAY_OUT_SC:** Just pitted (<5 laps), podium position, rivals pitting, tire life > 20 laps

**Latency:** <100ms (time-critical, fastest module)

**Example:**
```python
context = DecisionContext(
    safety_car_active=True,
    tire_age=18,
    current_position=5,
    ...
)
# → Recommends PIT_UNDER_SC with VERY_HIGH confidence (GREEN light)
# Pit loss reduced from ~22s to ~11s under SC
```

### 3. Strategy Conversion (Priority: 7)

**When to use:** Mid-race (laps 10-70% of race) to assess strategy changes

**Triggers:**
- **Switch to 2-stop:** Degradation >20% above expected, SC deployed, rivals on 2-stop gaining pace
- **Switch to 1-stop:** Degradation lower than expected, fuel-saving possible, track position critical

**Confidence:** High when strategy tree and Monte Carlo simulations agree (>90% alignment)

**Example:**
```python
decision_input = DecisionInput(
    context=context,
    simulation_context=SimulationContext(
        strategy_rankings=[
            {'strategy_id': '2-stop', 'expected_position': 2.5},
            {'strategy_id': '1-stop', 'expected_position': 4.2},
        ],
        ...
    ),
)
# → Recommends SWITCH_TO_TWO_STOP if 2+ position gain expected
```

### 4. Undercut/Overcut Decision (Priority: 8)

**When to use:** When rival within 3-5s gap

**Triggers:**
- **UNDERCUT_NOW:** Rival ahead within 3s, rival tire age >5 laps older, pit window open, pit loss < gap
- **OVERCUT_NOW:** Rival just pitted, tire life > 10 laps, can extend stint, gap < 5s after rival pit

**Timing:**
- Undercut: Pit 1-2 laps before rival's optimal pit lap
- Overcut: Extend 3-5 laps beyond rival's pit

**Example:**
```python
rival = RivalContext(
    rival_position=2,
    rival_tire_age=25,  # 5 laps older than us
    gap_to_rival=2.3,   # Within undercut range
    ...
)
# → Recommends UNDERCUT_NOW with HIGH confidence (AMBER/GREEN light)
```

### 5. Rain Strategy (Priority: 10)

**When to use:** When rain detected or track conditions changing

**Triggers:**
- **SWITCH_TO_INTERS:** Rain detected, lap times +2s slower, track temp dropping, rivals switching
- **SWITCH_TO_WETS:** Heavy rain, lap times +5s slower, standing water
- **SWITCH_TO_SLICKS:** Track drying, lap times on inters slower than slicks crossover

**Timing:** Critical - early switch gains positions, late switch loses time

**Example:**
```python
context = DecisionContext(
    recent_lap_times=[90.5, 92.1, 94.3],  # Increasing
    track_temp=28.0,  # Dropping
    weather_temp=20.0,
    ...
)
# → Recommends SWITCH_TO_INTERS with MEDIUM confidence (AMBER light)
```

### 6. Pace Monitoring (Priority: 5)

**When to use:** Every lap to monitor pace consistency

**Triggers:**
- **PACE_DROP:** Lap times >0.5s slower, degradation exceeds expected, losing positions
- **CONSERVATIVE_PACE:** Save tires, avoid cliff
- **AGGRESSIVE_PACE:** Push for undercut/overcut, gaining pace

**Example:**
```python
# Advisory only - lower priority
context = DecisionContext(
    recent_lap_times=[90.5, 90.7, 91.2, 91.8, 92.3],  # Degrading
    ...
)
# → Recommends CONSERVATIVE_PACE with MEDIUM confidence (AMBER light)
```

### 7. Offset Strategy (Priority: 6)

**When to use:** When rivals on different tire compounds

**Triggers:**
- **OFFSET:** Tire age delta > 5 laps, pace advantage > 0.3s/lap, within undercut range
- **MATCH_RIVAL:** Offset not working, rival strategy effective

**Example:**
```python
rival = RivalContext(
    rival_tire_compound='HARD',  # We're on SOFT
    rival_tire_age=10,           # Older tires
    gap_to_rival=2.8,            # Close
    ...
)
# → Recommends OFFSET_STRATEGY if pace delta favorable
```

## Configuration

Edit `config/decision_engine.yaml` to customize module behavior:

```yaml
modules:
  pit_timing:
    enabled: true
    priority: 9
    config:
      tire_age_threshold: 0.9  # 90% of optimal stint
      degradation_threshold: 0.1  # s/lap
      pace_drop_threshold: 0.5  # seconds
```

## Traffic Light System

Recommendations use traffic lights for quick visual assessment:

- **GREEN:** High confidence (≥0.8) AND low risk (≤0.3) → Execute with confidence
- **AMBER:** Medium confidence/risk → Consider carefully, evaluate alternatives
- **RED:** Low confidence (<0.5) OR high risk (>0.7) → High risk, avoid unless critical

**Note:** Safety car decisions never show RED (too time-critical).

## Performance

**Latency Targets:**
- Single decision: <200ms (p95)
- Safety car decision: <100ms (time-critical)
- Batch (20 drivers): <2s
- Cached decision: <50ms

**Optimization:**
- Redis caching (TTL 30s)
- Async parallel execution
- Circuit breakers for fault tolerance
- Pre-computed features

## Troubleshooting

**Low confidence recommendations:**
- Check if factors conflict (e.g., high tire age but SC imminent)
- Verify ML models are loaded (not using fallback heuristics)
- Ensure feature data is recent (<60s old)

**High latency:**
- Enable caching in config
- Reduce Monte Carlo runs (default 100)
- Disable non-critical modules temporarily
- Check Redis connection

**Incorrect recommendations:**
- Review decision logs for reasoning
- Audit post-race (compare expected vs actual)
- Adjust thresholds in config
- Retrain ML models if predictions off

## Best Practices

1. **Pre-race:** Run strategy tree exploration to identify optimal strategies
2. **During race:** Call decision engine every lap for all drivers
3. **Safety car:** Prioritize SC decision module (highest priority, fastest)
4. **Post-race:** Audit decisions to improve thresholds and models
5. **Testing:** Use CLI benchmark to verify <200ms latency

## Examples

### Complete Workflow

```python
from decision_engine import DecisionEngine, DecisionInput, DecisionContext
from decision_engine.explainer import DecisionExplainer

# Initialize
engine = DecisionEngine(config_path="config/decision_engine.yaml")

# Create context
context = DecisionContext(
    session_id="2024_MONACO",
    lap_number=25,
    driver_number=44,
    track_name="Monaco",
    total_laps=78,
    current_position=3,
    tire_age=20,
    tire_compound="MEDIUM",
    fuel_load=85.0,
    stint_number=2,
    pit_stops_completed=1,
    recent_lap_times=[90.5, 90.8, 91.2, 91.5, 91.8],
)

# Make decision
output = engine.make_decision(DecisionInput(context=context))

# Print results
for rec in output.recommendations:
    print(DecisionExplainer.generate_explanation_text(rec))

# Print comparison table
print(DecisionExplainer.generate_comparison_table(output.recommendations))

# Get stats
stats = engine.get_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
```

## Integration

- **Simulation Engine:** Provides strategy rankings, win probabilities
- **ML Models:** Provides predictions (degradation, lap time, SC, pit loss)
- **Feature Store:** Provides engineered features (pace, traffic, weather)
- **API Layer:** Exposes decision engine via REST endpoints
- **Dashboard:** Visualizes recommendations with traffic lights

For more details, see [Architecture Documentation](./ARCHITECTURE.md).
