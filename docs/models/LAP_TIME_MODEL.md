# Lap Time Prediction Model

## Overview

The Lap Time Prediction Model is a production-ready machine learning system that predicts Formula 1 lap times under varying race conditions. It combines **XGBoost** (optimized for speed) and **LightGBM** (optimized for accuracy with uncertainty estimation) in an ensemble architecture to deliver predictions with <200ms latency.

## Key Features

- **Multi-Model Ensemble**: Combines XGBoost and LightGBM for optimal speed-accuracy tradeoff
- **Uncertainty Estimation**: Quantile regression provides 10th-90th percentile prediction intervals
- **Production-Ready Inference**: Redis caching, circuit breaker pattern, <200ms latency
- **Comprehensive Feature Engineering**: Integrates tire degradation, fuel load, traffic effects, weather, and safety car conditions
- **Hyperparameter Optimization**: Automated tuning with Optuna (50 trials, TPE sampler)
- **Model Versioning**: Semantic versioning with production/staging/latest aliases
- **Physics-Based Fallback**: Domain knowledge heuristics when ML models unavailable

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Lap Time Prediction System               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐                                       │
│  │  PredictionInput │  ← 13 fields (tire, fuel, traffic,  │
│  │  (Pydantic)      │    safety car, weather, track, etc) │
│  └────────┬─────────┘                                       │
│           │                                                 │
│           ▼                                                 │
│  ┌──────────────────┐                                       │
│  │ LapTimePredictor │  ← Redis Cache + Circuit Breaker    │
│  │   (Inference)    │                                       │
│  └────────┬─────────┘                                       │
│           │                                                 │
│      ┌────┴────┐                                            │
│      ▼         ▼                                            │
│  ┌────────┐ ┌─────────┐                                    │
│  │XGBoost │ │LightGBM │                                    │
│  │ (40%)  │ │  (60%)  │  ← Weighted Ensemble              │
│  └────┬───┘ └───┬─────┘                                    │
│       │         │                                           │
│       └────┬────┘                                           │
│            ▼                                                │
│  ┌──────────────────┐                                       │
│  │ PredictionOutput │  ← lap_time, confidence,            │
│  │  (Pydantic)      │    pace_components, uncertainty      │
│  └──────────────────┘                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Model Inputs

### PredictionInput Schema

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `tire_age` | int | 0-50 laps | Age of current tires |
| `tire_compound` | TireCompound | SOFT/MEDIUM/HARD | Tire compound type |
| `fuel_load` | float | 0-110 kg | Current fuel weight |
| `traffic_state` | RaceCondition | CLEAN_AIR/DIRTY_AIR/etc | Traffic condition |
| `gap_to_ahead` | float? | 0-10s | Gap to car ahead (if in dirty air) |
| `safety_car_active` | bool | - | Whether safety car is deployed |
| `weather_temp` | float | 10-40°C | Ambient temperature |
| `track_temp` | float? | 15-50°C | Track surface temperature |
| `track_name` | str | - | Circuit name |
| `driver_number` | int | 1-99 | Driver number |
| `lap_number` | int | 1-70 | Current lap number |
| `session_progress` | float | 0-1 | Race completion ratio |
| `recent_lap_times` | List[float] | - | Last 5 lap times (60-150s each) |

## Model Outputs

### PredictionOutput Schema

```python
{
  "predicted_lap_time": 88.5,          # seconds
  "confidence": 0.92,                  # 0-1
  "pace_components": {
    "base_pace": 85.0,                 # Clean air baseline
    "tire_effect": 0.5,                # Degradation penalty
    "fuel_effect": 1.5,                # Weight penalty (~0.03s/kg)
    "traffic_penalty": 0.0,            # Dirty air loss
    "weather_adjustment": 0.5,         # Temperature effect
    "safety_car_factor": 1.0           # 1.0 = normal, 1.3 = SC
  },
  "uncertainty_range": (87.2, 89.8),   # 10th-90th percentile
  "metadata": {
    "model_type": "ensemble",
    "version": "1.0.0",
    "xgb_prediction": 88.3,
    "lgb_prediction": 88.7,
    "prediction_agreement": 0.4        # seconds difference
  }
}
```

## Training Pipeline

### 1. Data Preparation

```python
from models.lap_time.data_preparation import DataPreparationPipeline

pipeline = DataPreparationPipeline()
train_data, val_data, test_data, feature_names = pipeline.prepare_training_data(
    data_path="data/processed"
)
```

**Features Extracted (14+):**
- `tire_age`, `tire_compound_encoded`, `fuel_load`
- `traffic_penalty`, `safety_car_flag`
- `weather_temp`, `track_temp`, `track_encoded`
- `driver_aggression`, `degradation_slope`
- `rolling_avg_pace`, `sector_consistency`
- `lap_number`, `session_progress`

**Data Quality Checks:**
- Outlier detection (Z-score > 3.0)
- Missing value validation (<10% threshold)
- Stratified train/val/test split (70/15/15) by track and compound

### 2. Model Training

```python
from models.lap_time.training import ModelTrainer

trainer = ModelTrainer(
    data_path="data/processed",
    registry_path="models/registry/lap_time"
)

model, metrics = trainer.train_model(
    model_type="ensemble",           # or "xgboost", "lightgbm"
    optimize_hyperparams=True,       # Optuna optimization (50 trials)
    register_model=True,
    version="1.0.0"
)
```

**Hyperparameter Optimization (Optuna):**
- **XGBoost**: max_depth (4-10), learning_rate (0.01-0.3), n_estimators (100-500), regularization
- **LightGBM**: max_depth (4-12), num_leaves (20-100), learning_rate (0.01-0.2), categorical features
- **Objective**: Minimize RMSE on validation set
- **Sampler**: TPESampler with 50 trials

### 3. Evaluation

```python
from models.lap_time.evaluation import ModelEvaluator

evaluator = ModelEvaluator()
metrics = evaluator.evaluate_model(
    model,
    X_test,
    y_test,
    test_data,
    output_path="results/evaluation"
)
```

**Metrics Tracked:**
- **Regression**: MAE, RMSE, R², MAPE
- **Accuracy**: Within 0.5s, 1.0s, 2.0s thresholds
- **Condition-Specific**: Clean air, dirty air, safety car performance
- **Track-Specific**: Per-circuit MAE/RMSE
- **Compound-Specific**: Soft/Medium/Hard tire performance

## Inference API

### Production Deployment

```python
from models.lap_time.inference import LapTimePredictor

predictor = LapTimePredictor(
    model_version="production",      # or "latest", "1.0.0"
    use_cache=True,                  # Redis caching
    redis_host="localhost",
    redis_port=6379
)

# Single prediction
input_data = PredictionInput(...)
output = predictor.predict(input_data)

# Batch prediction
inputs = [PredictionInput(...), ...]
outputs = predictor.predict_batch(inputs)

# Performance stats
stats = predictor.get_performance_stats()
# {
#   "cache_hit_rate": 0.85,
#   "latency_p50_ms": 45,
#   "latency_p95_ms": 120,
#   "latency_p99_ms": 180,
#   "fallback_uses": 2
# }
```

### Caching Strategy

- **Cache Key**: MD5 hash of PredictionInput JSON
- **TTL**: 3600 seconds (1 hour)
- **Backend**: Redis
- **Hit Rate Target**: >80%

### Circuit Breaker

- **Failure Threshold**: 5 consecutive failures → OPEN circuit
- **Recovery Timeout**: 60 seconds before attempting HALF_OPEN
- **Fallback**: Physics-based heuristics when circuit OPEN

## Model Registry

### Version Management

```python
from models.lap_time.registry import ModelRegistry

registry = ModelRegistry("models/registry/lap_time")

# List all versions
versions = registry.list_versions()

# Load model
model = registry.load_model("production")  # or "latest", "1.0.0"

# Promote model
registry.promote_model("1.0.0", alias="production")

# Compare versions
comparison = registry.compare_versions("1.0.0", "1.1.0")
```

### Semantic Versioning

- **Format**: `MAJOR.MINOR.PATCH` (e.g., `1.0.0`)
- **Aliases**: `production`, `staging`, `latest`
- **Metadata**: Metrics, hyperparameters, registration date

## CLI Usage

### Training Commands

```bash
# Train ensemble model with optimization
python scripts/train_lap_time.py train \
  --model-type ensemble \
  --optimize \
  --version 1.0.0

# Cross-validation
python scripts/train_lap_time.py cross-validate \
  --model-type ensemble \
  --n-folds 5

# Compare all model types
python scripts/train_lap_time.py compare

# Evaluate specific version
python scripts/train_lap_time.py evaluate \
  --version 1.0.0 \
  --output-path results/evaluation

# Promote model
python scripts/train_lap_time.py promote \
  --version 1.0.0 \
  --alias production

# List registered models
python scripts/train_lap_time.py list-models

# Delete model
python scripts/train_lap_time.py delete \
  --version 1.0.0 \
  --force
```

## Configuration

### Model Configs

**XGBoost** (`config/models/lap_time_xgboost_config.yaml`):
```yaml
hyperparameters:
  max_depth: 6
  learning_rate: 0.1
  n_estimators: 200
  tree_method: hist        # Fast histogram-based
  predictor: cpu_predictor # Low-latency CPU mode
performance:
  target_latency_ms: 150
```

**LightGBM** (`config/models/lap_time_lightgbm_config.yaml`):
```yaml
hyperparameters:
  max_depth: 8
  num_leaves: 64
  learning_rate: 0.05
  n_estimators: 300
uncertainty:
  enabled: true
  lower_percentile: 0.1    # 10th percentile
  upper_percentile: 0.9    # 90th percentile
performance:
  target_latency_ms: 200
```

**Ensemble** (`config/models/lap_time_ensemble_config.yaml`):
```yaml
ensemble:
  models:
    - type: xgboost
      default_weight: 0.4  # Speed
    - type: lightgbm
      default_weight: 0.6  # Accuracy
  optimize_weights: true
performance:
  target_latency_ms: 180
  circuit_breaker:
    enabled: true
    failure_threshold: 5
```

## Performance Benchmarks

| Model | Latency (p50) | Latency (p95) | RMSE | MAE | R² |
|-------|---------------|---------------|------|-----|-----|
| XGBoost | 45ms | 120ms | 0.52s | 0.35s | 0.92 |
| LightGBM | 65ms | 150ms | 0.48s | 0.32s | 0.94 |
| Ensemble | 55ms | 135ms | 0.46s | 0.30s | 0.95 |
| Fallback | <10ms | <20ms | 1.20s | 0.85s | 0.75 |

**Target**: <200ms p99 latency, <0.5s RMSE

## Physics-Based Fallback

When ML models fail, fallback heuristics provide predictions using domain knowledge:

```python
predicted_lap_time = (
    base_lap_time +                    # From tracks.yaml
    tire_age * degradation_rate +      # From tire_compounds.yaml
    fuel_load * 0.03 +                 # 0.03s per kg
    traffic_penalty +                  # 0.8s max, exponential decay
    weather_adjustment                 # ±0.02s per °C from 25°C
) * safety_car_factor                  # 1.3x for safety car
```

**Confidence**: 0.6 (lower than ML models)

## Integration with Race Strategy

### Simulation Engine

```python
# Used by race simulation to predict future lap times
for lap in remaining_laps:
    lap_time_prediction = lap_time_predictor.predict(
        PredictionInput(
            tire_age=current_stint_laps,
            fuel_load=remaining_fuel,
            traffic_state=estimate_traffic(lap),
            ...
        )
    )
    total_time += lap_time_prediction.predicted_lap_time
```

### Decision Engine

```python
# Compare strategies based on predicted lap times
strategy_a_time = sum(predict_lap_times(strategy_a))
strategy_b_time = sum(predict_lap_times(strategy_b))

if strategy_a_time < strategy_b_time:
    recommend(strategy_a)
```

## Testing

Run the test suite:

```bash
# All tests
pytest tests/models/lap_time/

# Specific module
pytest tests/models/lap_time/test_ensemble_model.py

# With coverage
pytest tests/models/lap_time/ --cov=models.lap_time --cov-report=html
```

## Future Enhancements

1. **Driver-Specific Models**: Personalized predictions per driver
2. **Track Learning**: Dynamic adjustment as session progresses
3. **Real-Time Updates**: Online learning from live telemetry
4. **GPU Acceleration**: Faster training with XGBoost GPU tree method
5. **A/B Testing**: Compare model versions in production with traffic splitting
6. **Feature Importance Tracking**: Monitor which features drive predictions
7. **Anomaly Detection**: Flag suspicious predictions for review

## References

- **XGBoost Documentation**: https://xgboost.readthedocs.io/
- **LightGBM Documentation**: https://lightgbm.readthedocs.io/
- **Optuna Optimization**: https://optuna.readthedocs.io/
- **Pydantic Validation**: https://docs.pydantic.dev/

## Authors

F1 Race Strategy Intelligence Engine Team

## License

Proprietary - Internal Use Only
