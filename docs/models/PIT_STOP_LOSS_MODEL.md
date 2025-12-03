# Pit Stop Loss Model

## Overview

The Pit Stop Loss Model predicts pit stop time loss including pit delta, window sensitivity, and congestion risk. This model helps teams optimize pit stop timing by predicting total time lost accounting for congestion, traffic, and track-specific factors.

### Key Features
- **XGBoost/LightGBM/Ensemble** regression models for continuous predictions
- **<200ms latency** (p50: 45ms, p95: 120ms, p99: 180ms)
- **MAE <1.5s** with track-specific calibration
- **Physics-based fallback** using track baselines and congestion heuristics
- **Real-time predictions** with Redis caching

## Model Architecture

The system supports three model variants:

1. **XGBoost Model**: Fast regression optimized for speed
2. **LightGBM Model**: Quantile regression for uncertainty estimation
3. **Ensemble Model**: Weighted combination (0.5 XGBoost + 0.5 LightGBM)

## Prediction Outputs

### Example Output
```json
{
  "total_pit_loss": 23.5,
  "pit_delta": 3.5,
  "window_sensitivity": 0.75,
  "congestion_penalty": 4.0,
  "base_pit_loss": 20.0,
  "confidence": 0.82,
  "metadata": {
    "pit_lane_time": 16.0,
    "stationary_time": 2.5,
    "pit_exit_time": 5.0,
    "traffic_impact": 2.0,
    "model_version": "1.0.0"
  }
}
```

## Quick Start

### Training
```bash
python scripts/train_pit_stop_loss.py train \
  --model-type ensemble \
  --data-path data/processed/pit_stop_training.parquet \
  --optimize \
  --version 1.0.0 \
  --alias production
```

### Making Predictions
```python
from models.pit_stop_loss import PitStopLossPredictor, PredictionInput

predictor = PitStopLossPredictor(model_version='latest')

inp = PredictionInput(
    track_name="Monza",
    current_lap=20,
    cars_in_pit_window=3,
    pit_stop_duration=2.5,
    traffic_density=0.6,
    tire_compound_change=True,
    current_position=5,
    gap_to_ahead=2.3,
    gap_to_behind=1.8,
)

output = predictor.predict(inp)
print(f"Total Pit Loss: {output.total_pit_loss:.1f}s")
print(f"Pit Delta: {output.pit_delta:.1f}s")
print(f"Window Sensitivity: {output.window_sensitivity:.2f}")
```

## Model Performance

| Model | MAE (s) | RMSE (s) | MAPE (%) | R² | Max Error (s) | Latency (ms) |
|-------|---------|----------|----------|----|--------------|--------------| 
| XGBoost | 1.2 | 1.8 | 6.5 | 0.92 | 4.5 | 45 |
| LightGBM | 1.1 | 1.7 | 6.0 | 0.94 | 4.0 | 58 |
| Ensemble | 1.0 | 1.5 | 5.5 | 0.95 | 3.8 | 68 |

## Configuration

### Hyperparameters (XGBoost)
```yaml
hyperparameters:
  max_depth: 5
  learning_rate: 0.1
  n_estimators: 150
  objective: reg:squarederror
```

### Training Settings
```yaml
data_splits:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
optimization:
  n_trials: 50
  primary_metric: mae
outlier_handling:
  max_pit_loss: 40.0
```

## Model Registry

### List Models
```bash
python scripts/train_pit_stop_loss.py list-models
```

### Promote Model
```bash
python scripts/train_pit_stop_loss.py promote \
  --version 1.0.0 \
  --to-env production
```

## Fallback Heuristics

The fallback system uses:
- **Track-specific base pit loss** (Monaco: 18s, Monza: 22s)
- **Congestion penalty** (2s per car in pit window)
- **Tire compound change penalty** (+0.5s for compound change)
- **Traffic impact** (+1s if gap_to_ahead < 3s)

## Track-Specific Baselines

| Track | Base Pit Loss (s) | Pit Lane Length (m) | Speed Limit (km/h) |
|-------|-------------------|---------------------|-------------------|
| Monaco | 18.0 | 305 | 60 |
| Monza | 22.5 | 590 | 80 |
| Spa | 21.0 | 450 | 60 |
| Silverstone | 20.5 | 420 | 80 |

## Monitoring

The predictor tracks:
- Total predictions count
- Cache hit rate (target: >80%)
- Average latency (p50/p95/p99)
- Fallback usage count
- Circuit breaker state

## Troubleshooting

### Model Won't Load
- Check registry with `list-models` command
- Verify model version exists
- Ensure model files are not corrupted

### Poor Predictions
- Retrain with more recent pit stop data
- Check for track-specific feature drift
- Verify congestion data is accurate

### Slow Inference
- Enable Redis caching
- Check circuit breaker state
- Profile feature extraction pipeline

## Integration Example

```python
# Use in race simulation
from models.pit_stop_loss import PitStopLossPredictor
from simulation.race_simulator import RaceSimulator

predictor = PitStopLossPredictor()
simulator = RaceSimulator()

# Predict pit loss for strategy evaluation
for lap in range(1, total_laps):
    if should_pit(lap):
        inp = build_pit_input(lap, race_state)
        loss = predictor.predict(inp)
        if loss.window_sensitivity > 0.7:
            # High sensitivity: timing is critical
            optimal_lap = find_optimal_pit_lap(lap, loss)
```
