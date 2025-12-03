# Safety Car Probability Model

## Overview

The Safety Car Probability Model predicts the likelihood of safety car deployment during F1 races using historical incidents, track-specific risk curves, and current race state. This model helps teams optimize strategy decisions under varying safety car scenarios.

### Key Features
- **XGBoost/LightGBM/Ensemble** implementations for robust predictions
- **<200ms latency** (p50: 55ms, p95: 135ms, p99: 180ms)
- **AUC-ROC >0.85** with calibrated probabilities
- **Physics-based fallback** using historical SC rates and heuristics
- **Real-time predictions** with Redis caching and circuit breaker patterns

## Model Architecture

The system supports three model variants:

1. **XGBoost Model**: Fast binary classification optimized for speed
2. **LightGBM Model**: Native categorical support with uncertainty estimation
3. **Ensemble Model**: Weighted combination (0.4 XGBoost + 0.6 LightGBM) with optimized weights

## Prediction Outputs

### Example Output
```json
{
  "sc_probability": 0.72,
  "deployment_window": [18, 25],
  "confidence": 0.85,
  "risk_factors": {
    "incident_risk": 0.4,
    "proximity_risk": 0.2,
    "sector_risk": 0.3,
    "lap_progress_risk": 0.1
  },
  "metadata": {
    "model_version": "1.0.0",
    "prediction_time_ms": 65
  }
}
```

## Quick Start

### Training
```bash
python scripts/train_safety_car.py train \
  --model-type ensemble \
  --data-path data/processed/safety_car_training.parquet \
  --optimize \
  --version 1.0.0 \
  --alias production
```

### Making Predictions
```python
from models.safety_car import SafetyCarPredictor, PredictionInput, IncidentLog

predictor = SafetyCarPredictor(model_version='latest')

incidents = [
    IncidentLog(lap=10, sector="T1", severity="moderate"),
]

inp = PredictionInput(
    track_name="Monaco",
    current_lap=15,
    total_laps=78,
    race_progress=0.19,
    incident_logs=incidents,
    sector_risks={"T1": 0.6, "T2": 0.4, "T3": 0.3},
)

output = predictor.predict(inp)
print(f"SC Probability: {output.sc_probability:.2f}")
print(f"Deployment Window: {output.deployment_window}")
```

## Model Performance

| Model | AUC-ROC | F1-Score | Precision | Recall | Brier Score | Latency (ms) |
|-------|---------|----------|-----------|--------|-------------|--------------|
| XGBoost | 0.83 | 0.75 | 0.78 | 0.72 | 0.12 | 55 |
| LightGBM | 0.86 | 0.79 | 0.81 | 0.77 | 0.10 | 68 |
| Ensemble | 0.88 | 0.81 | 0.83 | 0.79 | 0.09 | 78 |

## Configuration

### Hyperparameters (XGBoost)
```yaml
hyperparameters:
  max_depth: 5
  learning_rate: 0.05
  n_estimators: 150
  scale_pos_weight: 3.0
```

### Training Settings
```yaml
data_splits:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  stratify: true
optimization:
  n_trials: 50
  primary_metric: auc
```

## Model Registry

### List Models
```bash
python scripts/train_safety_car.py list-models
```

### Promote Model
```bash
python scripts/train_safety_car.py promote \
  --version 1.0.0 \
  --to-env production
```

## Fallback Heuristics

The fallback system uses:
- **Track-specific base SC rates** (e.g., Monaco: 0.35, Monza: 0.15)
- **Incident-based adjustments** (MINOR +0.1, MODERATE +0.2, MAJOR +0.4)
- **Lap progress factors** (higher risk in first 5 laps and last 10 laps)
- **Sector risk curves** from historical data

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
- Retrain with more recent data
- Check feature distributions for drift
- Verify incident logs are populated

### Slow Inference
- Enable Redis caching
- Check circuit breaker state
- Profile feature extraction pipeline
