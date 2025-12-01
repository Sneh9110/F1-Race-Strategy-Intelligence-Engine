"""
LightGBM Lap Time Prediction Model

Optimized for accuracy with uncertainty estimation.
Uses gradient boosting with categorical features and quantile regression.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb
from pydantic import ValidationError

from .base import (
    BaseLapTimeModel,
    ModelConfig,
    PredictionInput,
    PredictionOutput,
    RaceCondition
)
from config.settings import Settings

logger = logging.getLogger(__name__)


class LightGBMLapTimeModel(BaseLapTimeModel):
    """
    LightGBM implementation of lap time prediction.
    
    Optimized for:
    - Accuracy: Higher precision with deeper trees
    - Uncertainty: Quantile regression for prediction intervals
    - Categorical features: Native support for tire compounds, tracks
    
    Attributes:
        model: LightGBM model (50th percentile)
        model_lower: LightGBM model (10th percentile)
        model_upper: LightGBM model (90th percentile)
        feature_names: Ordered list of feature names
        categorical_features: List of categorical feature names
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize LightGBM model.
        
        Args:
            config: Model configuration. If None, uses default settings.
        """
        super().__init__(config)
        self.model: Optional[lgb.Booster] = None
        self.model_lower: Optional[lgb.Booster] = None  # 10th percentile
        self.model_upper: Optional[lgb.Booster] = None  # 90th percentile
        self.feature_names: List[str] = []
        self.categorical_features: List[str] = []
        self.feature_importance_: Dict[str, float] = {}
        
        # LightGBM hyperparameters optimized for accuracy
        self.lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'num_leaves': 64,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.5,
            'boosting_type': 'gbdt',
            'verbosity': -1,
        }
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train LightGBM model with quantile regression for uncertainty.
        
        Args:
            X_train: Training features
            y_train: Training targets (lap times in seconds)
            X_val: Validation features
            y_val: Validation targets
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Training LightGBM model on {len(X_train)} samples")
        
        # Store feature names and identify categorical features
        self.feature_names = list(X_train.columns)
        self.categorical_features = [
            col for col in self.feature_names 
            if 'encoded' in col.lower() or col in ['tire_compound', 'track_name']
        ]
        
        # Override hyperparameters from kwargs
        params = {**self.lgb_params, **kwargs}
        n_estimators = params.pop('n_estimators', 300)
        
        # Create datasets
        train_data = lgb.Dataset(
            X_train, 
            label=y_train,
            feature_name=self.feature_names,
            categorical_feature=self.categorical_features
        )
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                reference=train_data,
                feature_name=self.feature_names,
                categorical_feature=self.categorical_features
            )
            valid_sets.append(val_data)
            valid_names.append('val')
        
        # Train median model (50th percentile)
        callbacks = [
            lgb.early_stopping(stopping_rounds=20),
            lgb.log_evaluation(period=0),
        ]
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        # Train lower bound model (10th percentile)
        params_lower = params.copy()
        params_lower['objective'] = 'quantile'
        params_lower['alpha'] = 0.1
        
        self.model_lower = lgb.train(
            params_lower,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        # Train upper bound model (90th percentile)
        params_upper = params.copy()
        params_upper['objective'] = 'quantile'
        params_upper['alpha'] = 0.9
        
        self.model_upper = lgb.train(
            params_upper,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        # Store feature importance (from median model)
        importance = self.model.feature_importance(importance_type='gain')
        total_gain = np.sum(importance)
        self.feature_importance_ = {
            name: float(imp / total_gain)
            for name, imp in zip(self.feature_names, importance)
        }
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
        train_mae = np.mean(np.abs(y_train - train_pred))
        
        metrics = {
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'best_iteration': self.model.best_iteration,
        }
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
            val_mae = np.mean(np.abs(y_val - val_pred))
            
            # Calculate uncertainty calibration
            val_lower = self.model_lower.predict(X_val)
            val_upper = self.model_upper.predict(X_val)
            coverage = np.mean((y_val >= val_lower) & (y_val <= val_upper))
            
            metrics.update({
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'uncertainty_coverage': coverage,  # Should be ~0.8 for 10-90th percentiles
            })
        
        logger.info(f"Training complete. Train RMSE: {train_rmse:.3f}s, MAE: {train_mae:.3f}s")
        if 'val_rmse' in metrics:
            logger.info(f"Validation RMSE: {metrics['val_rmse']:.3f}s, MAE: {metrics['val_mae']:.3f}s")
            logger.info(f"Uncertainty coverage: {metrics['uncertainty_coverage']:.2%}")
        
        return metrics
    
    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """
        Predict lap time with uncertainty estimation.
        
        Args:
            input_data: Prediction input with race conditions
            
        Returns:
            Prediction output with lap time, confidence, and uncertainty range
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() or load() first.")
        
        # Validate input
        self._validate_input(input_data)
        
        # Extract features
        features = self._extract_features(input_data)
        
        # Create feature vector in correct order
        X = pd.DataFrame([features])[self.feature_names]
        
        # Predict median, lower, and upper bounds
        prediction = self.model.predict(X)[0]
        lower_bound = self.model_lower.predict(X)[0] if self.model_lower else prediction - 1.0
        upper_bound = self.model_upper.predict(X)[0] if self.model_upper else prediction + 1.0
        
        # Decompose prediction into pace components
        pace_components = self._decompose_prediction(input_data, features, prediction)
        
        # Calculate confidence based on uncertainty range
        confidence = self._calculate_confidence(
            input_data, 
            features,
            uncertainty_range=(lower_bound, upper_bound)
        )
        
        return PredictionOutput(
            predicted_lap_time=float(prediction),
            confidence=confidence,
            pace_components=pace_components,
            uncertainty_range=(float(lower_bound), float(upper_bound)),
            metadata={
                'model_type': 'lightgbm',
                'version': self.config.version,
                'feature_count': len(self.feature_names),
                'uncertainty_width': float(upper_bound - lower_bound),
            }
        )
    
    def predict_batch(self, inputs: List[PredictionInput]) -> List[PredictionOutput]:
        """
        Predict lap times for multiple inputs with uncertainty.
        
        Args:
            inputs: List of prediction inputs
            
        Returns:
            List of prediction outputs
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() or load() first.")
        
        # Extract features for all inputs
        features_list = [self._extract_features(inp) for inp in inputs]
        X = pd.DataFrame(features_list)[self.feature_names]
        
        # Batch prediction
        predictions = self.model.predict(X)
        lower_bounds = self.model_lower.predict(X) if self.model_lower else predictions - 1.0
        upper_bounds = self.model_upper.predict(X) if self.model_upper else predictions + 1.0
        
        # Create outputs
        outputs = []
        for i, (input_data, pred, lower, upper) in enumerate(
            zip(inputs, predictions, lower_bounds, upper_bounds)
        ):
            pace_components = self._decompose_prediction(input_data, features_list[i], pred)
            confidence = self._calculate_confidence(
                input_data, 
                features_list[i],
                uncertainty_range=(lower, upper)
            )
            
            outputs.append(PredictionOutput(
                predicted_lap_time=float(pred),
                confidence=confidence,
                pace_components=pace_components,
                uncertainty_range=(float(lower), float(upper)),
                metadata={
                    'model_type': 'lightgbm',
                    'version': self.config.version,
                    'uncertainty_width': float(upper - lower),
                }
            ))
        
        return outputs
    
    def save(self, path: Path) -> None:
        """
        Save model to disk.
        
        Args:
            path: Directory to save model files
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        path.mkdir(parents=True, exist_ok=True)
        
        # Save LightGBM models
        self.model.save_model(str(path / "lightgbm_median.txt"))
        if self.model_lower:
            self.model_lower.save_model(str(path / "lightgbm_lower.txt"))
        if self.model_upper:
            self.model_upper.save_model(str(path / "lightgbm_upper.txt"))
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'feature_importance': self.feature_importance_,
            'config': self.config.__dict__,
            'lgb_params': self.lgb_params,
        }
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Path) -> None:
        """
        Load model from disk.
        
        Args:
            path: Directory containing model files
        """
        model_path = path / "lightgbm_median.txt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load LightGBM models
        self.model = lgb.Booster(model_file=str(model_path))
        
        lower_path = path / "lightgbm_lower.txt"
        if lower_path.exists():
            self.model_lower = lgb.Booster(model_file=str(lower_path))
        
        upper_path = path / "lightgbm_upper.txt"
        if upper_path.exists():
            self.model_upper = lgb.Booster(model_file=str(upper_path))
        
        # Load metadata
        with open(path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['feature_names']
        self.categorical_features = metadata['categorical_features']
        self.feature_importance_ = metadata['feature_importance']
        self.lgb_params = metadata['lgb_params']
        
        # Update config
        if 'config' in metadata:
            self.config = ModelConfig(**metadata['config'])
        
        logger.info(f"Model loaded from {path}")
    
    def get_metadata(self) -> Dict:
        """
        Get model metadata.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': 'lightgbm',
            'version': self.config.version,
            'trained': self.model is not None,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'feature_importance': self.feature_importance_,
            'hyperparameters': self.lgb_params,
            'uncertainty_estimation': self.model_lower is not None and self.model_upper is not None,
        }
    
    def _decompose_prediction(
        self,
        input_data: PredictionInput,
        features: Dict,
        total_prediction: float
    ) -> Dict[str, float]:
        """
        Decompose prediction into interpretable pace components.
        
        Args:
            input_data: Original input data
            features: Extracted features
            total_prediction: Total predicted lap time
            
        Returns:
            Dictionary of pace components
        """
        # Estimate base pace (clean air, optimal conditions)
        base_pace = total_prediction
        
        # Estimate tire effect (from tire age and degradation)
        tire_effect = features.get('tire_age', 0) * 0.05
        if 'degradation_slope' in features:
            tire_effect = features['degradation_slope'] * features.get('tire_age', 0)
        
        # Estimate fuel effect (from fuel load)
        fuel_effect = features.get('fuel_load', 0) * 0.03
        
        # Estimate traffic penalty
        traffic_penalty = 0.0
        if input_data.traffic_state == RaceCondition.DIRTY_AIR:
            traffic_penalty = features.get('traffic_penalty', 0.4)
        
        # Estimate safety car factor
        safety_car_factor = 1.0
        if input_data.safety_car_active:
            safety_car_factor = 1.3
        
        # Estimate weather adjustment
        weather_adjustment = 0.0
        if input_data.weather_temp:
            weather_adjustment = (input_data.weather_temp - 25) * 0.02
        
        # Calculate base pace by removing effects
        base_pace = total_prediction - tire_effect - fuel_effect - traffic_penalty - weather_adjustment
        base_pace = base_pace / safety_car_factor
        
        return {
            'base_pace': float(base_pace),
            'tire_effect': float(tire_effect),
            'fuel_effect': float(fuel_effect),
            'traffic_penalty': float(traffic_penalty),
            'weather_adjustment': float(weather_adjustment),
            'safety_car_factor': float(safety_car_factor),
        }
    
    def _calculate_confidence(
        self,
        input_data: PredictionInput,
        features: Dict,
        uncertainty_range: Optional[Tuple[float, float]] = None
    ) -> float:
        """
        Calculate prediction confidence based on uncertainty and input characteristics.
        
        Args:
            input_data: Original input data
            features: Extracted features
            uncertainty_range: (lower_bound, upper_bound) tuple
            
        Returns:
            Confidence score between 0 and 1
        """
        confidence = 1.0
        
        # Factor in uncertainty width (narrower = higher confidence)
        if uncertainty_range:
            lower, upper = uncertainty_range
            uncertainty_width = upper - lower
            # Typical lap time ~90s, if uncertainty is >5s, reduce confidence
            if uncertainty_width > 5.0:
                confidence *= 0.8
            elif uncertainty_width > 3.0:
                confidence *= 0.9
        
        # Reduce confidence for extreme tire ages
        if features.get('tire_age', 0) > 30:
            confidence *= 0.9
        elif features.get('tire_age', 0) > 40:
            confidence *= 0.8
        
        # Reduce confidence for rare conditions
        if input_data.safety_car_active:
            confidence *= 0.85
        
        # Reduce confidence for extreme fuel loads
        if features.get('fuel_load', 0) > 100:
            confidence *= 0.9
        
        # Reduce confidence if features are missing
        if input_data.track_temp is None:
            confidence *= 0.95
        if input_data.gap_to_ahead is None and input_data.traffic_state == RaceCondition.DIRTY_AIR:
            confidence *= 0.9
        
        return max(0.5, min(1.0, confidence))
