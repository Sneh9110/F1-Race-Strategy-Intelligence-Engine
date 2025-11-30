"""
LightGBM-based tire degradation model.

Optimized for accuracy with native categorical support and uncertainty estimation.
"""

import joblib
import numpy as np
import lightgbm as lgb
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

from models.tire_degradation.base import (
    BaseDegradationModel,
    ModelConfig,
    PredictionInput,
    PredictionOutput
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LightGBMDegradationModel(BaseDegradationModel):
    """
    LightGBM implementation for tire degradation prediction.
    
    Features native categorical support and uncertainty estimation.
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize LightGBM model."""
        super().__init__(config)
        self.model = None
        self.feature_names = []
        self.categorical_features = []
        
        # Load default hyperparameters
        self.hyperparameters = {
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'num_leaves': 64,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.5,
            'objective': 'regression',
            'metric': ['rmse', 'mae'],
            'boosting_type': 'gbdt',
            'early_stopping_rounds': 30,
            'n_jobs': -1,
            'verbose': -1
        }
        self.hyperparameters.update(config.hyperparameters)
    
    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        categorical_features: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            categorical_features: List of categorical feature names
        
        Returns:
            Training metrics
        """
        logger.info("Training LightGBM degradation model")
        
        # Store feature info
        if hasattr(X_train, 'columns'):
            self.feature_names = list(X_train.columns)
        else:
            self.feature_names = [f'f{i}' for i in range(X_train.shape[1])]
        
        self.categorical_features = categorical_features or []
        
        # Create datasets
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=self.feature_names,
            categorical_feature=self.categorical_features
        )
        
        valid_data = None
        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(
                X_val,
                label=y_val,
                feature_name=self.feature_names,
                categorical_feature=self.categorical_features,
                reference=train_data
            )
        
        # Callbacks
        callbacks = []
        if valid_data:
            callbacks.append(lgb.early_stopping(
                self.hyperparameters.get('early_stopping_rounds', 30)
            ))
        
        # Train
        params = {k: v for k, v in self.hyperparameters.items()
                  if k not in ['early_stopping_rounds', 'n_estimators']}
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.hyperparameters.get('n_estimators', 300),
            valid_sets=[valid_data] if valid_data else None,
            callbacks=callbacks
        )
        
        # Update metadata
        self.metadata['trained'] = True
        self.metadata['feature_names'] = self.feature_names
        self.metadata['categorical_features'] = self.categorical_features
        self.metadata['n_features'] = len(self.feature_names)
        
        # Get metrics
        metrics = {
            'train_samples': len(X_train),
            'val_samples': len(X_val) if X_val is not None else 0,
            'best_iteration': self.model.best_iteration
        }
        
        if valid_data:
            metrics['val_rmse'] = self.model.best_score['valid_0']['rmse']
            metrics['val_mae'] = self.model.best_score['valid_0']['mae']
        
        self.metadata['performance_metrics'] = metrics
        
        logger.info(f"Training complete: {metrics}")
        return metrics
    
    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """
        Make prediction with uncertainty estimation.
        
        Args:
            input_data: Prediction input
        
        Returns:
            Prediction output with confidence intervals
        """
        self._validate_input(input_data)
        
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Extract features
        features = self._extract_features(input_data)
        X = self._features_to_array(features)
        
        # Predict with uncertainty
        deg_rate, uncertainty = self._predict_with_uncertainty(X)
        
        # Generate full prediction
        curve = self._predict_curve(input_data, deg_rate)
        usable_life = self._predict_usable_life_from_curve(curve, input_data.tire_compound)
        dropoff_lap = self._predict_dropoff_from_curve(curve)
        
        # Calculate confidence from uncertainty
        confidence = self._uncertainty_to_confidence(uncertainty)
        
        return PredictionOutput(
            degradation_curve=curve,
            usable_life=usable_life,
            dropoff_lap=dropoff_lap,
            confidence=confidence,
            degradation_rate=deg_rate,
            metadata={
                'model_type': 'lightgbm',
                'version': self.config.version,
                'uncertainty': uncertainty,
                'tire_age': input_data.tire_age,
                'tire_compound': input_data.tire_compound
            }
        )
    
    def _predict_with_uncertainty(self, X) -> tuple[float, float]:
        """
        Predict with uncertainty estimation.
        
        Uses leaf prediction variance as uncertainty metric.
        
        Args:
            X: Input features
        
        Returns:
            (prediction, uncertainty)
        """
        # Get prediction
        pred = float(self.model.predict(X)[0])
        
        # Estimate uncertainty from leaf values
        # Get predictions from each tree
        leaf_preds = []
        for tree_idx in range(self.model.num_trees()):
            tree_pred = self.model.predict(X, start_iteration=tree_idx, num_iteration=1)[0]
            leaf_preds.append(tree_pred)
        
        # Uncertainty = std of tree predictions
        uncertainty = float(np.std(leaf_preds))
        
        return pred, uncertainty
    
    def _uncertainty_to_confidence(self, uncertainty: float) -> float:
        """
        Convert uncertainty to confidence score.
        
        Args:
            uncertainty: Prediction uncertainty
        
        Returns:
            Confidence 0-1
        """
        # High uncertainty = low confidence
        # Sigmoid mapping: confidence = 1 / (1 + exp(k * uncertainty))
        k = 5.0  # Steepness
        confidence = 1.0 / (1.0 + np.exp(k * uncertainty))
        return float(np.clip(confidence, 0.0, 1.0))
    
    def predict_curve(
        self,
        input_data: PredictionInput,
        num_laps: int = 50
    ) -> List[float]:
        """Predict degradation curve."""
        features = self._extract_features(input_data)
        X = self._features_to_array(features)
        deg_rate, _ = self._predict_with_uncertainty(X)
        
        return self._predict_curve(input_data, deg_rate, num_laps)
    
    def _predict_curve(
        self,
        input_data: PredictionInput,
        base_rate: float,
        num_laps: int = 50
    ) -> List[float]:
        """Generate degradation curve."""
        curve = []
        current_age = input_data.tire_age
        
        # Compound-specific factors
        compound_factors = {
            'SOFT': 1.3,
            'MEDIUM': 1.0,
            'HARD': 0.75
        }
        factor = compound_factors.get(input_data.tire_compound, 1.0)
        
        # Weather adjustment
        temp_factor = 1.0
        if input_data.weather_temp > 30:
            temp_factor = 1.0 + (input_data.weather_temp - 30) * 0.01
        
        for lap in range(num_laps):
            # Quadratic + exponential degradation
            age = current_age + lap
            quad_term = 0.0001 * (age ** 2)
            exp_term = base_rate * np.exp(0.012 * factor * age)
            
            deg = exp_term + quad_term
            deg *= temp_factor
            
            # Clamp
            deg = np.clip(deg, 0.0, 5.0)
            curve.append(float(deg))
        
        return curve
    
    def predict_usable_life(self, input_data: PredictionInput) -> int:
        """Predict usable tire life."""
        curve = self.predict_curve(input_data)
        return self._predict_usable_life_from_curve(curve, input_data.tire_compound)
    
    def _predict_usable_life_from_curve(
        self,
        curve: List[float],
        compound: str
    ) -> int:
        """Calculate usable life from curve."""
        thresholds = {
            'SOFT': 1.5,
            'MEDIUM': 2.0,
            'HARD': 2.5
        }
        threshold = thresholds.get(compound, 2.0)
        
        for i, deg in enumerate(curve):
            if deg > threshold:
                return i + 1
        
        return len(curve)
    
    def predict_dropoff_lap(self, input_data: PredictionInput) -> Optional[int]:
        """Predict dropoff/cliff lap."""
        curve = self.predict_curve(input_data)
        return self._predict_dropoff_from_curve(curve)
    
    def _predict_dropoff_from_curve(self, curve: List[float]) -> Optional[int]:
        """Detect cliff in degradation curve."""
        if len(curve) < 5:
            return None
        
        # Look for sudden acceleration in degradation
        # Second derivative > threshold
        for i in range(2, len(curve)):
            if i >= len(curve) - 1:
                break
            
            # Acceleration = second derivative
            accel = (curve[i] - 2 * curve[i-1] + curve[i-2])
            if accel > 0.3:
                return i
        
        return None
    
    def _features_to_array(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert feature dict to array."""
        feature_vector = []
        
        for name in self.feature_names:
            if name in features:
                value = features[name]
                # LightGBM handles categoricals natively, but we encode them for consistency
                if isinstance(value, str):
                    value = hash(value) % 100
                feature_vector.append(float(value))
            else:
                feature_vector.append(0.0)
        
        return np.array([feature_vector])
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = path / 'lightgbm_model.txt'
        self.model.save_model(str(model_path))
        
        # Save metadata
        metadata_path = path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save config
        config_path = path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Path) -> None:
        """Load model from disk."""
        path = Path(path)
        
        # Load model
        model_path = path / 'lightgbm_model.txt'
        self.model = lgb.Booster(model_file=str(model_path))
        
        # Load metadata
        metadata_path = path / 'metadata.json'
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.feature_names = self.metadata.get('feature_names', [])
        self.categorical_features = self.metadata.get('categorical_features', [])
        
        logger.info(f"Model loaded from {path}")
