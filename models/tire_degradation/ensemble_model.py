"""
Ensemble tire degradation model combining multiple base models.

Combines XGBoost, LightGBM, and CatBoost with weighted voting.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

from models.tire_degradation.base import (
    BaseDegradationModel,
    ModelConfig,
    PredictionInput,
    PredictionOutput
)
from models.tire_degradation.xgboost_model import XGBoostDegradationModel
from models.tire_degradation.lightgbm_model import LightGBMDegradationModel
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EnsembleDegradationModel(BaseDegradationModel):
    """
    Ensemble model combining XGBoost and LightGBM.
    
    Uses weighted voting based on model confidence and historical performance.
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize ensemble model."""
        super().__init__(config)
        
        # Initialize base models
        self.xgboost_model = XGBoostDegradationModel(config)
        self.lightgbm_model = LightGBMDegradationModel(config)
        
        # Ensemble weights (can be learned)
        self.weights = config.hyperparameters.get('weights', {
            'xgboost': 0.5,
            'lightgbm': 0.5
        })
        
        # Normalization for weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        self.models = {
            'xgboost': self.xgboost_model,
            'lightgbm': self.lightgbm_model
        }
    
    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train all ensemble models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        
        Returns:
            Combined training metrics
        """
        logger.info("Training ensemble degradation model")
        
        all_metrics = {}
        
        # Train each model
        for name, model in self.models.items():
            logger.info(f"Training {name} model")
            metrics = model.train(X_train, y_train, X_val, y_val, **kwargs)
            all_metrics[name] = metrics
        
        # Update ensemble metadata
        self.metadata['trained'] = True
        self.metadata['ensemble_models'] = list(self.models.keys())
        self.metadata['weights'] = self.weights
        self.metadata['performance_metrics'] = all_metrics
        
        # Optionally: Learn optimal weights from validation performance
        if X_val is not None and y_val is not None:
            self._optimize_weights(X_val, y_val)
        
        logger.info(f"Ensemble training complete. Weights: {self.weights}")
        return all_metrics
    
    def _optimize_weights(self, X_val, y_val) -> None:
        """
        Optimize ensemble weights based on validation performance.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
        """
        logger.info("Optimizing ensemble weights")
        
        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            if hasattr(X_val, 'iloc'):
                preds = []
                for idx in range(len(X_val)):
                    # Create mock PredictionInput (simplified)
                    # In practice, you'd need to convert X_val rows properly
                    preds.append(model.model.predict(X_val.iloc[[idx]])[0])
                predictions[name] = np.array(preds)
            else:
                predictions[name] = model.model.predict(X_val)
        
        # Simple grid search for optimal weights
        best_weights = self.weights.copy()
        best_error = float('inf')
        
        # Try different weight combinations
        for w_xgb in np.linspace(0.2, 0.8, 7):
            w_lgb = 1.0 - w_xgb
            weights = {'xgboost': w_xgb, 'lightgbm': w_lgb}
            
            # Weighted average prediction
            ensemble_pred = sum(weights[name] * predictions[name] 
                              for name in predictions.keys())
            
            # Calculate error
            error = np.mean((ensemble_pred - y_val) ** 2)
            
            if error < best_error:
                best_error = error
                best_weights = weights
        
        self.weights = best_weights
        self.metadata['optimized_weights'] = True
        self.metadata['weights'] = self.weights
        
        logger.info(f"Optimized weights: {self.weights}, RMSE: {np.sqrt(best_error):.4f}")
    
    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """
        Make ensemble prediction.
        
        Args:
            input_data: Prediction input
        
        Returns:
            Weighted ensemble prediction
        """
        self._validate_input(input_data)
        
        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            try:
                pred = model.predict(input_data)
                predictions[name] = pred
            except Exception as e:
                logger.warning(f"Model {name} prediction failed: {e}")
                predictions[name] = None
        
        # Filter out failed predictions
        valid_preds = {k: v for k, v in predictions.items() if v is not None}
        
        if not valid_preds:
            raise ValueError("All ensemble models failed to predict")
        
        # Weighted averaging
        ensemble_output = self._combine_predictions(valid_preds)
        
        return ensemble_output
    
    def _combine_predictions(
        self,
        predictions: Dict[str, PredictionOutput]
    ) -> PredictionOutput:
        """
        Combine predictions using weighted voting.
        
        Args:
            predictions: Dictionary of model predictions
        
        Returns:
            Combined ensemble prediction
        """
        # Normalize weights for available models only
        available_weights = {name: self.weights[name] 
                           for name in predictions.keys()}
        total_weight = sum(available_weights.values())
        normalized_weights = {k: v / total_weight 
                            for k, v in available_weights.items()}
        
        # Weighted average for degradation rate
        avg_rate = sum(normalized_weights[name] * pred.degradation_rate
                      for name, pred in predictions.items())
        
        # Weighted average for usable life (round to int)
        avg_life = sum(normalized_weights[name] * pred.usable_life
                      for name, pred in predictions.items())
        avg_life = int(round(avg_life))
        
        # Weighted average for dropoff lap (with None handling)
        dropoff_laps = [pred.dropoff_lap for pred in predictions.values() 
                       if pred.dropoff_lap is not None]
        avg_dropoff = None
        if dropoff_laps:
            avg_dropoff = int(round(np.mean(dropoff_laps)))
        
        # Weighted average for confidence
        avg_confidence = sum(normalized_weights[name] * pred.confidence
                           for name, pred in predictions.items())
        
        # Average degradation curves (element-wise)
        curves = [pred.degradation_curve for pred in predictions.values()]
        max_len = max(len(c) for c in curves)
        
        # Pad curves to same length
        padded_curves = []
        for curve in curves:
            if len(curve) < max_len:
                curve = curve + [curve[-1]] * (max_len - len(curve))
            padded_curves.append(np.array(curve))
        
        # Weighted average
        ensemble_curve = np.zeros(max_len)
        for name, pred in predictions.items():
            idx = list(predictions.keys()).index(name)
            ensemble_curve += normalized_weights[name] * padded_curves[idx]
        
        return PredictionOutput(
            degradation_curve=ensemble_curve.tolist(),
            usable_life=avg_life,
            dropoff_lap=avg_dropoff,
            confidence=avg_confidence,
            degradation_rate=avg_rate,
            metadata={
                'model_type': 'ensemble',
                'version': self.config.version,
                'ensemble_models': list(predictions.keys()),
                'weights': normalized_weights,
                'individual_predictions': {
                    name: {
                        'degradation_rate': pred.degradation_rate,
                        'usable_life': pred.usable_life,
                        'dropoff_lap': pred.dropoff_lap,
                        'confidence': pred.confidence
                    }
                    for name, pred in predictions.items()
                }
            }
        )
    
    def predict_curve(
        self,
        input_data: PredictionInput,
        num_laps: int = 50
    ) -> List[float]:
        """Predict ensemble degradation curve."""
        pred = self.predict(input_data)
        return pred.degradation_curve[:num_laps]
    
    def predict_usable_life(self, input_data: PredictionInput) -> int:
        """Predict ensemble usable life."""
        pred = self.predict(input_data)
        return pred.usable_life
    
    def predict_dropoff_lap(self, input_data: PredictionInput) -> Optional[int]:
        """Predict ensemble dropoff lap."""
        pred = self.predict(input_data)
        return pred.dropoff_lap
    
    def save(self, path: Path) -> None:
        """Save ensemble model."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save each model
        for name, model in self.models.items():
            model_path = path / name
            model.save(model_path)
        
        # Save ensemble metadata
        metadata_path = path / 'ensemble_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save weights
        weights_path = path / 'ensemble_weights.json'
        with open(weights_path, 'w') as f:
            json.dump(self.weights, f, indent=2)
        
        # Save config
        config_path = path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        logger.info(f"Ensemble model saved to {path}")
    
    def load(self, path: Path) -> None:
        """Load ensemble model."""
        path = Path(path)
        
        # Load each model
        for name, model in self.models.items():
            model_path = path / name
            if model_path.exists():
                model.load(model_path)
        
        # Load ensemble metadata
        metadata_path = path / 'ensemble_metadata.json'
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load weights
        weights_path = path / 'ensemble_weights.json'
        with open(weights_path, 'r') as f:
            self.weights = json.load(f)
        
        logger.info(f"Ensemble model loaded from {path}")
