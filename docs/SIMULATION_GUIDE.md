# Race Simulation Engine - User Guide

## Overview

The Race Simulation Engine is a production-grade Monte Carlo simulator that integrates all four ML models (Tire Degradation, Lap Time, Safety Car, Pit Stop Loss) to provide comprehensive race strategy analysis.

**Key Features:**
- **Single Simulation**: Deterministic race outcome for given strategy
- **Strategy Tree Exploration**: Discover optimal pit strategies
- **Monte Carlo Analysis**: Probabilistic outcome distribution (1000+ runs)
- **What-If Scenarios**: Test hypothetical race situations

**Performance Targets:**
- Single simulation: <500ms (70 laps, 20 drivers)
- Monte Carlo (1000 runs): <5s
- Strategy tree exploration: <10s

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from simulation import RaceSimulator; print('✓ Simulation engine ready')"
```

### Basic Usage

```python
from simulation import RaceSimulator, SimulationInput, RaceConfig, DriverState, StrategyOption, TireCompound, PaceTarget

# Create race configuration
race_config = RaceConfig(
    track_name="Monaco",
    total_laps=78,
    current_lap=1,
    weather_temp=25.0,
    track_temp=35.0,
    grid_positions=list(range(1, 21)),
    safety_car_active=False,
    vsc_active=False,
)

# Create driver states (simplified - single driver)
drivers = [
    DriverState(
        driver_number=1,
        current_position=1,
        tire_compound=TireCompound.SOFT,
        tire_age=0,
        fuel_load=110.0,
        gap_to_ahead=0.0,
        gap_to_behind=1.5,
        recent_lap_times=[89.0] * 5,
        num_pit_stops=0,
        current_stint=1,
        cumulative_race_time=0.0,
    )
]

# Define strategy
strategy = StrategyOption(
    pit_laps=[25],
    tire_sequence=[TireCompound.SOFT, TireCompound.MEDIUM],
    target_pace=PaceTarget.BALANCED,
)

# Create simulation input
sim_input = SimulationInput(
    race_config=race_config,
    drivers=drivers,
    strategy_to_evaluate=strategy,
    monte_carlo_runs=0,
)

# Run simulation
simulator = RaceSimulator()
output = simulator.simulate_race(sim_input)

# Access results
print(f"Final Position: {output.results[0].final_position}")
print(f"Race Time: {output.results[0].total_race_time:.2f}s")
print(f"Computation Time: {output.metadata['computation_time_ms']:.0f}ms")
```

---

## Simulation Modes

### 1. Single Deterministic Simulation

For testing specific strategies with known inputs:

```python
simulator = RaceSimulator()
output = simulator.simulate_race(sim_input)

# Analyze lap-by-lap results
for lap_result in output.results[0].laps[:5]:
    print(f"Lap {lap_result.lap_number}: {lap_result.lap_time:.2f}s")
```

### 2. Strategy Tree Exploration

Discover optimal strategies across pit stop counts:

```python
from simulation import StrategyTreeExplorer

explorer = StrategyTreeExplorer(
    simulator,
    max_workers=4,
    max_strategies=50,
)

# Explore strategies
rankings = explorer.explore_strategies(
    base_input=sim_input,
    available_compounds=[TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD],
    max_pit_stops=3,
)

# Get best strategy
best = rankings[0]
print(f"Best Strategy: Pit laps {best.strategy.pit_laps}")
print(f"Tire Sequence: {' → '.join(c.value for c in best.strategy.tire_sequence)}")
print(f"Expected Time: {best.expected_race_time:.2f}s")
print(f"Win Probability: {best.win_probability:.1%}")
```

**Undercut/Overcut Analysis:**

```python
# Analyze undercut window
undercut_results = explorer.analyze_undercut_window(
    base_input=sim_input,
    target_strategy=current_strategy,
    lap_range=(20, 30),
)

for lap, time_gained in undercut_results.items():
    print(f"Lap {lap}: {time_gained:+.2f}s")
```

### 3. Monte Carlo Analysis

Generate probabilistic outcome distributions:

```python
from simulation import MonteCarloSimulator, MonteCarloConfig

config = MonteCarloConfig(
    num_runs=1000,
    num_workers=8,
    tire_deg_noise_std=0.02,
    lap_time_noise_std=0.15,
)

mc_sim = MonteCarloSimulator(simulator, config)
result = mc_sim.run_monte_carlo(sim_input, target_driver=1)

# Statistics
stats = result["statistics"]
print(f"Win Probability: {stats['win_probability']:.1%}")
print(f"Podium Probability: {stats['podium_probability']:.1%}")
print(f"Mean Position: {stats['position']['mean']:.1f}")
print(f"P10-P90 Range: {stats['position']['p10']:.0f}-{stats['position']['p90']:.0f}")

# Position distribution
for pos, prob in stats["position_distribution"].items():
    if prob > 0.01:
        print(f"P{pos}: {prob:.1%}")
```

**Convergence Checking:**

```python
convergence = result["convergence"]
if convergence["converged"]:
    print(f"✓ Converged (variance: {convergence['variance']:.4f})")
else:
    print(f"✗ Not converged ({convergence['reason']})")
```

### 4. What-If Scenario Analysis

Test hypothetical race situations:

```python
from simulation import WhatIfEngine, ScenarioType

engine = WhatIfEngine(simulator)

# Analyze early safety car
scenario = engine.scenario_templates[ScenarioType.EARLY_SAFETY_CAR]
comparison = engine.analyze_scenario(sim_input, scenario, target_driver=1)

print(f"Scenario: {scenario.description}")
print(f"Position Change: {comparison.position_delta:+d}")
print(f"Time Delta: {comparison.time_delta:+.2f}s")
print(f"Win Probability Change: {comparison.win_probability_delta:+.1%}")

# Compare strategies under multiple scenarios
strategies = [strategy1, strategy2, strategy3]
scenarios = [
    engine.scenario_templates[ScenarioType.EARLY_SAFETY_CAR],
    engine.scenario_templates[ScenarioType.LATE_SAFETY_CAR],
]

matrix = engine.compare_strategies_under_scenarios(sim_input, strategies, scenarios)
best_idx = matrix["most_robust_strategy_index"]
print(f"Most robust strategy: {best_idx}")
```

**Available Scenarios:**
- `EARLY_SAFETY_CAR`: SC in first third
- `LATE_SAFETY_CAR`: SC in final third
- `DOUBLE_SAFETY_CAR`: Two SC periods
- `UNDERCUT_ATTEMPT`: Pit 2-3 laps earlier
- `OVERCUT_ATTEMPT`: Pit 2-3 laps later
- `TIRE_OFFSET`: Start on different compound
- `AGGRESSIVE_PACING`: Push 0.2s/lap faster
- `CONSERVATIVE_PACING`: Save tires, 0.3s/lap slower
- `RAIN_TRANSITION`: Mid-race rain
- `TIRE_FAILURE`: Unexpected extra stop

---

## CLI Interface

### Commands

**Create Input Template:**

```bash
python scripts/run_simulation.py create-input --track Monaco --total-laps 78 -o input.json
```

**Run Simulation:**

```bash
# Single run
python scripts/run_simulation.py simulate -i input.json --mode single -o output.json

# Monte Carlo
python scripts/run_simulation.py simulate -i input.json --mode monte_carlo --mc-runs 1000

# Strategy exploration
python scripts/run_simulation.py simulate -i input.json --mode strategy_tree
```

**Explore Strategies:**

```bash
python scripts/run_simulation.py explore-strategies -i input.json --max-stops 3 -o strategies.json
```

**What-If Analysis:**

```bash
# List scenarios
python scripts/run_simulation.py list-scenarios

# Analyze scenario
python scripts/run_simulation.py what-if -i input.json --scenario early_safety_car -o comparison.json
```

---

## Configuration

### simulation.yaml

Located in `config/simulation.yaml`:

```yaml
performance:
  max_latency_ms: 500
  enable_cache: true
  cache_ttl_seconds: 300

monte_carlo:
  default_runs: 1000
  max_runs: 10000
  num_workers: 8
  noise:
    tire_degradation_std: 0.02
    lap_time_std: 0.15
    pit_stop_std: 0.5

strategy_tree:
  max_workers: 4
  max_strategies: 50
  pruning_threshold_seconds: 5.0
```

### Track Configuration

Located in `config/tracks.yaml`:

```yaml
Monaco:
  base_lap_time_seconds: 72.5
  pit_loss_seconds: 18.0
  fuel_consumption_per_lap_kg: 1.3
  safety_car_probability: 0.25

Monza:
  base_lap_time_seconds: 82.0
  pit_loss_seconds: 22.5
  fuel_consumption_per_lap_kg: 1.8
  safety_car_probability: 0.10
```

---

## Best Practices

### 1. Strategy Exploration Workflow

```python
# Step 1: Explore strategies
explorer = StrategyTreeExplorer(simulator, max_strategies=50)
rankings = explorer.explore_strategies(sim_input, max_pit_stops=2)

# Step 2: Validate top 3 with Monte Carlo
top_strategies = [r.strategy for r in rankings[:3]]
mc_results = []

for strategy in top_strategies:
    sim_input.strategy_to_evaluate = strategy
    mc_result = mc_sim.run_monte_carlo(sim_input)
    mc_results.append(mc_result)

# Step 3: Test under scenarios
engine = WhatIfEngine(simulator)
matrix = engine.compare_strategies_under_scenarios(
    sim_input,
    top_strategies,
    [engine.scenario_templates[ScenarioType.EARLY_SAFETY_CAR]],
)
```

### 2. Performance Optimization

```python
# Use caching for repeated simulations
import redis
cache_client = redis.Redis(host='localhost', port=6379)
simulator = RaceSimulator(cache_client=cache_client)

# Adjust MC runs based on confidence needs
from simulation.monte_carlo import estimate_required_runs
required = estimate_required_runs(sim_input, target_confidence=0.95, margin_of_error=0.02)
config = MonteCarloConfig(num_runs=required)
```

### 3. Error Handling

```python
try:
    output = simulator.simulate_race(sim_input)
except Exception as e:
    logger.error(f"Simulation failed: {e}")
    # Fallback to simplified strategy
```

---

## Integration with ML Models

The simulator automatically loads and integrates all 4 ML models:

```python
# Models are loaded at initialization
simulator = RaceSimulator()

# Access predictors
deg_predictor = simulator.predictors["degradation"]
lap_time_predictor = simulator.predictors["lap_time"]
sc_predictor = simulator.predictors["safety_car"]
pit_predictor = simulator.predictors["pit_stop_loss"]

# Models are called internally during simulation
# with circuit breakers and fallbacks
```

---

## Troubleshooting

**Slow Simulations:**
- Reduce `max_strategies` in StrategyTreeExplorer
- Decrease `num_workers` if CPU constrained
- Enable Redis caching

**Inaccurate Predictions:**
- Check ML model versions in metadata
- Verify track configuration exists
- Validate input data ranges (tire age 0-50, fuel 0-110)

**Memory Issues:**
- Reduce `monte_carlo_runs`
- Lower `num_workers` in parallel operations
- Clear cache periodically

---

## API Reference

See docstrings in:
- `simulation/core.py` - RaceSimulator
- `simulation/strategy_tree.py` - StrategyTreeExplorer
- `simulation/monte_carlo.py` - MonteCarloSimulator
- `simulation/what_if.py` - WhatIfEngine
- `simulation/schemas.py` - All data models

---

## Support

For issues or questions:
1. Check logs in `logs/simulation.log`
2. Verify configuration in `config/simulation.yaml`
3. Review test examples in `tests/test_simulation/`
