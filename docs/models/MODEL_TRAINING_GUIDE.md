# Tire Degradation Model Training Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Data Preparation](#data-preparation)
3. [Model Training](#model-training)
4. [Hyperparameter Optimization](#hyperparameter-optimization)
5. [Model Evaluation](#model-evaluation)
6. [Model Deployment](#model-deployment)
7. [Advanced Topics](#advanced-topics)

## Prerequisites

### Required Libraries

```bash
pip install xgboost lightgbm optuna scikit-learn pandas numpy pydantic redis circuitbreaker
```

### Data Requirements

- **Minimum Samples**: 1,000 stints for training
- **Required Features**:
  - Tire compound (SOFT/MEDIUM/HARD)
  - Tire age (laps on tire)
  - Weather temperature
  - Driver aggression metrics
  - Track characteristics
  - Lap time data

- **Target Variables**:
  - Degradation rate (s/lap)
  - Usable life (laps)
  - Cliff detection (boolean)

## Data Preparation

### 1. Collect Historical Stint Data

```python
from data_pipeline.schemas.historical_schema import HistoricalStint
from features.store import FeatureStore

# Load historical stints
feature_store = FeatureStore()
stints = feature_store.load_historical_stints(
    start_date='2023-01-01',
    end_date='2024-12-31'
)
```

### 2. Feature Engineering

The `DataPreparationPipeline` automatically:
- Extracts lap time statistics (mean, std, min, max)
- Computes pace evolution
- Calculates degradation slopes
- Detects cliff points
- Augments with FeatureStore features

```python
from models.tire_degradation.data_preparation import DataPreparationPipeline

pipeline = DataPreparationPipeline()
X_train, X_val, X_test, y_train, y_val, y_test = pipeline.prepare_training_data(
    stints_df,
    target_column='degradation_rate'
)
```

### 3. Data Quality Checks

```python
# Check for missing values
print(f"Missing values: {X_train.isnull().sum()}")

# Check target distribution
import matplotlib.pyplot as plt
plt.hist(y_train, bins=50)
plt.xlabel('Degradation Rate')
plt.ylabel('Count')
plt.show()

# Check class balance (for compounds)
print(X_train['tire_compound'].value_counts())
```

## Model Training

### Basic Training

```bash
python scripts/train_tire_degradation.py train \
    --model-type ensemble \
    --data-path data/historical_stints.parquet \
    --no-optimize \
    --version 1.0.0
```

### With Hyperparameter Optimization

```bash
python scripts/train_tire_degradation.py train \
    --model-type ensemble \
    --data-path data/historical_stints.parquet \
    --optimize \
    --n-trials 100 \
    --version 1.1.0 \
    --alias staging
```

### Programmatic Training

```python
from models.tire_degradation.training import ModelTrainer
from pathlib import Path

# Initialize trainer
trainer = ModelTrainer(
    model_type='ensemble',
    config_path=Path('config/models/training_config.yaml')
)

# Train model
model = trainer.train(
    data=stints_df,
    target_column='degradation_rate',
    optimize_hyperparams=True,
    n_trials=50
)
```

## Hyperparameter Optimization

### Optuna Configuration

Edit `config/models/training_config.yaml`:

```yaml
optimization:
  enabled: true
  engine: "optuna"
  n_trials: 100
  timeout: 7200  # 2 hours
  
  primary_metric: "rmse"
  secondary_metrics:
    - "mae"
    - "r2"
  
  pruner: "median"
  n_startup_trials: 10
```

### Search Spaces

#### XGBoost

```python
hyperparameters:
  max_depth: [3, 10]           # Tree depth
  learning_rate: [0.01, 0.3]   # Learning rate (log scale)
  n_estimators: [100, 500]     # Number of trees
  subsample: [0.6, 1.0]        # Row sampling
  colsample_bytree: [0.6, 1.0] # Column sampling
  gamma: [0.0, 0.5]            # Min split loss
  reg_alpha: [0.0, 1.0]        # L1 regularization
  reg_lambda: [0.0, 2.0]       # L2 regularization
```

#### LightGBM

```python
hyperparameters:
  max_depth: [3, 12]
  num_leaves: [20, 150]
  learning_rate: [0.01, 0.3]
  n_estimators: [100, 500]
  min_child_samples: [10, 50]
```

### Monitoring Optimization

```python
# Optuna provides real-time optimization monitoring
# View progress in console or use Optuna Dashboard:

import optuna
study = optuna.load_study(study_name='tire_degradation', storage='sqlite:///optuna.db')
print(f"Best trial: {study.best_trial.number}")
print(f"Best value: {study.best_value}")
print(f"Best params: {study.best_params}")

# Visualize optimization
from optuna.visualization import plot_optimization_history, plot_param_importances
plot_optimization_history(study).show()
plot_param_importances(study).show()
```

## Model Evaluation

### Comprehensive Evaluation

```bash
python scripts/train_tire_degradation.py evaluate \
    --version 1.1.0 \
    --data-path data/test_stints.parquet \
    --output reports/eval_1.1.0.json
```

### Evaluation Metrics

```python
from models.tire_degradation.evaluation import ModelEvaluator

evaluator = ModelEvaluator()
metrics = evaluator.evaluate(model, X_test, y_test, additional_data=test_df)

print(f"RMSE: {metrics['rmse']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
print(f"R²: {metrics['r2']:.4f}")
print(f"Usable Life MAE: {metrics['usable_life_mae']:.2f} laps")
print(f"Cliff Detection Accuracy: {metrics['cliff_detection_accuracy']:.2%}")
```

### Model Comparison

```bash
python scripts/train_tire_degradation.py compare \
    --model-type xgboost lightgbm ensemble \
    --data-path data/test_stints.parquet
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score
from models.tire_degradation.xgboost_model import XGBoostDegradationModel

model = XGBoostDegradationModel(config)
scores = cross_val_score(
    model.model,
    X_train,
    y_train,
    cv=5,
    scoring='neg_mean_squared_error'
)
print(f"CV RMSE: {np.sqrt(-scores.mean()):.4f} ± {np.sqrt(scores.std()):.4f}")
```

## Model Deployment

### 1. Register Model

```python
from models.tire_degradation.registry import ModelRegistry

registry = ModelRegistry()

# Register with version
registry.register_model(
    model=trained_model,
    version='1.1.0',
    metadata={
        'training_date': '2024-01-15',
        'training_samples': 50000,
        'test_rmse': 0.025
    },
    alias='staging'
)
```

### 2. Test in Staging

```python
from models.tire_degradation.inference import DegradationPredictor

# Load staging model
predictor = DegradationPredictor(model_version='staging')

# Test predictions
test_input = PredictionInput(...)
output = predictor.predict(test_input)

# Monitor performance
stats = predictor.get_stats()
print(f"Avg latency: {stats['avg_latency_ms']:.0f}ms")
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
```

### 3. Promote to Production

```bash
python scripts/train_tire_degradation.py promote \
    --version 1.1.0 \
    --from-env staging \
    --to-env production
```

### 4. A/B Testing

```python
# Split traffic between models
registry.set_alias('production_a', '1.0.0')  # 50% traffic
registry.set_alias('production_b', '1.1.0')  # 50% traffic

# Compare performance after 1 week
comparison = registry.compare_models('1.0.0', '1.1.0', metric='rmse')
print(f"Winner: {comparison['winner']}")
print(f"Improvement: {comparison['improvement_pct']:.2f}%")
```

## Advanced Topics

### Custom Feature Engineering

```python
from models.tire_degradation.data_preparation import DataPreparationPipeline

class CustomPipeline(DataPreparationPipeline):
    def augment_with_computed_features(self, df):
        """Add custom features."""
        df = super().augment_with_computed_features(df)
        
        # Add custom feature: tire age squared
        df['tire_age_squared'] = df['tire_age'] ** 2
        
        # Add interaction: compound * temperature
        df['compound_temp_interaction'] = (
            df['tire_compound_encoded'] * df['weather_temp']
        )
        
        return df
```

### Ensemble Weight Optimization

```python
from models.tire_degradation.ensemble_model import EnsembleDegradationModel

# Ensemble automatically optimizes weights on validation set
config = ModelConfig(
    version='1.0.0',
    hyperparameters={'weights': {'xgboost': 0.5, 'lightgbm': 0.5}}
)

ensemble = EnsembleDegradationModel(config)
ensemble.train(X_train, y_train, X_val, y_val)

# Check optimized weights
print(f"Optimized weights: {ensemble.weights}")
```

### Handling Data Drift

```python
# Monitor feature distributions over time
from scipy.stats import ks_2samp

def check_drift(train_df, production_df, feature):
    """Check for distribution drift using KS test."""
    stat, p_value = ks_2samp(
        train_df[feature],
        production_df[feature]
    )
    
    if p_value < 0.05:
        print(f"⚠️  Drift detected in {feature}: p={p_value:.4f}")
        return True
    return False

# Check all features
for feature in X_train.columns:
    check_drift(X_train, X_production, feature)
```

### Retraining Strategy

```python
# Automated retraining workflow
from datetime import datetime, timedelta

def should_retrain(last_train_date, drift_detected, performance_degraded):
    """Decide if model needs retraining."""
    days_since_train = (datetime.now() - last_train_date).days
    
    # Retrain if:
    # 1. More than 30 days since last training
    # 2. Data drift detected
    # 3. Performance degraded significantly
    
    return (
        days_since_train > 30 or
        drift_detected or
        performance_degraded
    )

if should_retrain(last_train_date, drift, perf_drop):
    # Trigger automated retraining
    trainer = ModelTrainer(model_type='ensemble')
    new_model = trainer.train(recent_data, optimize_hyperparams=True)
    registry.register_model(new_model, version='1.2.0', alias='staging')
```

## Best Practices

### ✅ Do

- Use stratified splits for compounds and tracks
- Enable hyperparameter optimization (Optuna)
- Monitor inference latency and cache hit rates
- Version models semantically
- Test in staging before production
- Keep training data fresh (<30 days old)
- Log training metrics and configurations

### ❌ Don't

- Train on data with missing lap times
- Use unbalanced datasets (oversample rare compounds)
- Deploy without staging validation
- Ignore data drift warnings
- Skip cross-validation
- Hardcode hyperparameters
- Mix different tire specification years

## Troubleshooting

### Problem: Poor Validation Performance

**Solutions:**
- Increase training data (target: 10k+ stints)
- Enable hyperparameter optimization
- Check for data leakage
- Review feature engineering
- Try ensemble model

### Problem: Overfitting

**Solutions:**
- Increase regularization (`reg_alpha`, `reg_lambda`)
- Reduce model complexity (`max_depth`, `num_leaves`)
- Use more training data
- Enable early stopping
- Cross-validate hyperparameters

### Problem: Slow Training

**Solutions:**
- Reduce `n_trials` for Optuna (start with 20)
- Use histogram-based tree method (`tree_method='hist'`)
- Parallelize with `n_jobs=-1`
- Sample data for hyperparameter search
- Use GPU acceleration (XGBoost/LightGBM)

## References

- [Tire Degradation Model Documentation](TIRE_DEGRADATION_MODEL.md)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
