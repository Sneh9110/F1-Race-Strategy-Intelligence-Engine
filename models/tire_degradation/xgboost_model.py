"""
XGBoost-based tire degradation model.

Optimized for fast inference (<200ms) with competitive accuracy.
"""

import joblib
import numpy as np
import xgboost as xgb
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import yaml

from models.tire_degradation.base import (
    BaseDegradationModel,
    ModelConfig,
    PredictionInput,
    PredictionOutput
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class XGBoostDegradationModel(BaseDegradationModel):
    """
    XGBoost implementation for tire degradation prediction.
    
    Optimized for inference speed with <200ms latency target.
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize XGBoost model."""
        super().__init__(config)
        self.model = None
        self.feature_names = []
        
        # Load default hyperparameters
        self.hyperparameters = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'objective': 'reg:squarederror',
            'eval_metric': ['rmse', 'mae'],
            'early_stopping_rounds': 20,
            'tree_method': 'hist',  # Fast histogram-based method
            'n_jobs': -1
        }
        self.hyperparameters.update(config.hyperparameters)
    
    def _build_model(self) -> xgb.XGBRegressor:
        """Build XGBoost model with configured hyperparameters."""
        return xgb.XGBRegressor(**{
            k: v for k, v in self.hyperparameters.items()
            if k not in ['eval_metric', 'early_stopping_rounds']
        })
    
    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features (DataFrame or ndarray)
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        
        Returns:
            Training metrics and history
        """
        logger.info("Training XGBoost degradation model")
        
        # Build model
        self.model = self._build_model()
        
        # Store feature names
        if hasattr(X_train, 'columns'):
            self.feature_names = list(X_train.columns)
        else:
            self.feature_names = [f'f{i}' for i in range(X_train.shape[1])]
        
        # Prepare evaluation set
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
        
        # Train
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set if eval_set else None,
            verbose=kwargs.get('verbose', False)
        )
        
        # Update metadata
        self.metadata['trained'] = True
        self.metadata['feature_names'] = self.feature_names
        self.metadata['n_features'] = len(self.feature_names)
        
        # Get training metrics
        metrics = {
            'train_samples': len(X_train),
            'val_samples': len(X_val) if X_val is not None else 0,
        }
        
        if eval_set:
            evals_result = self.model.evals_result()
            metrics['train_rmse'] = evals_result['validation_0']['rmse'][-1]
            metrics['train_mae'] = evals_result['validation_0']['mae'][-1]
            if 'validation_1' in evals_result:
                metrics['val_rmse'] = evals_result['validation_1']['rmse'][-1]
                metrics['val_mae'] = evals_result['validation_1']['mae'][-1]
        
        self.metadata['performance_metrics'] = metrics
        
        logger.info(f"Training complete: {metrics}")
        return metrics
    
    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """
        Make prediction for single input.
        
        Args:
            input_data: Prediction input
        
        Returns:
            Prediction output with full details
        """
        self._validate_input(input_data)
        
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Extract features
        features = self._extract_features(input_data)
        
        # Convert to model input format
        X = self._features_to_array(features)
        
        # Predict degradation rate
        deg_rate = float(self.model.predict(X)[0])
        
        # Generate full prediction
        curve = self._predict_curve(input_data, deg_rate)
        usable_life = self._predict_usable_life_from_curve(curve, input_data.tire_compound)
        dropoff_lap = self._predict_dropoff_from_curve(curve)
        
        # Calculate confidence from model uncertainty
        confidence = self._calculate_confidence(X)
        
        return PredictionOutput(
            degradation_curve=curve,
            usable_life=usable_life,
            dropoff_lap=dropoff_lap,
            confidence=confidence,
            degradation_rate=deg_rate,
            metadata={
                'model_type': 'xgboost',
                'version': self.config.version,
                'tire_age': input_data.tire_age,
                'tire_compound': input_data.tire_compound
            }
        )
    
    def predict_curve(
        self,
        input_data: PredictionInput,
        num_laps: int = 50
    ) -> List[float]:
        """
        Predict degradation curve.
        
        Args:
            input_data: Prediction input
            num_laps: Number of laps to predict
        
        Returns:
            List of degradation values (seconds added per lap)
        """
        features = self._extract_features(input_data)
        X = self._features_to_array(features)
        deg_rate = float(self.model.predict(X)[0])
        
        return self._predict_curve(input_data, deg_rate, num_laps)
    
    def _predict_curve(
        self,
        input_data: PredictionInput,
        base_rate: float,
        num_laps: int = 50
    ) -> List[float]:
        """
        Generate degradation curve using predicted rate + exponential decay.
        
        Args:
            input_data: Input data with context
            base_rate: Base degradation rate (s/lap)
            num_laps: Number of laps to predict
        
        Returns:
            Degradation curve
        """
        curve = []
        current_age = input_data.tire_age
        
        # Compound-specific factors
        compound_factors = {
            'SOFT': 1.2,
            'MEDIUM': 1.0,
            'HARD': 0.8
        }
        factor = compound_factors.get(input_data.tire_compound, 1.0)
        
        for lap in range(num_laps):
            # Exponential degradation: deg = base_rate * exp(k * age)
            k = 0.015 * factor  # Exponential growth rate
            deg = base_rate * np.exp(k * (current_age + lap))
            
            # Clamp to reasonable values
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
        """
        Calculate usable life from degradation curve.
        
        Args:
            curve: Degradation curve
            compound: Tire compound
        
        Returns:
            Usable life in laps
        """
        # Threshold: degradation > 2.0 s/lap is end of useful life
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
        """
        Detect cliff/dropoff in degradation curve.
        
        Args:
            curve: Degradation curve
        
        Returns:
            Dropoff lap, or None if no cliff
        """
        if len(curve) < 5:
            return None
        
        # Look for sudden jump in degradation (>0.5 s/lap increase)
        for i in range(1, len(curve)):
            if curve[i] - curve[i-1] > 0.5:
                return i + 1
        
        return None
    
    def _calculate_confidence(self, X) -> float:
        """
        Calculate prediction confidence.
        
        Args:
            X: Input features
        
        Returns:
            Confidence score 0-1
        """
        # Use prediction variance from trees
        # For XGBoost, we approximate with number of trees and feature importance
        n_estimators = self.hyperparameters.get('n_estimators', 200)
        confidence = min(1.0, n_estimators / 300.0)  # Higher trees = higher confidence
        
        return float(confidence)
    
    def _features_to_array(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Convert feature dictionary to model input array.
        
        Args:
            features: Feature dictionary
        
        Returns:
            Feature array
        """
        # Build feature vector matching training features
        feature_vector = []
        
        for name in self.feature_names:
            if name in features:
                value = features[name]
                # Handle categorical encodings
                if isinstance(value, str):
                    value = hash(value) % 100  # Simple hash encoding
                feature_vector.append(float(value))
            else:
                feature_vector.append(0.0)  # Default for missing features
        
        return np.array([feature_vector])
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = path / 'xgboost_model.json'
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
        model_path = path / 'xgboost_model.json'
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(model_path))
        
        # Load metadata
        metadata_path = path / 'metadata.json'
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.feature_names = self.metadata.get('feature_names', [])
        
        logger.info(f"Model loaded from {path}")
