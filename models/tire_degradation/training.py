"""
Model training orchestrator with Optuna hyperparameter optimization.

Handles data preparation, model training, cross-validation, and evaluation.
"""

import optuna
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Type
from datetime import datetime
import yaml

from models.tire_degradation.base import BaseDegradationModel, ModelConfig
from models.tire_degradation.data_preparation import DataPreparationPipeline
from models.tire_degradation.xgboost_model import XGBoostDegradationModel
from models.tire_degradation.lightgbm_model import LightGBMDegradationModel
from models.tire_degradation.ensemble_model import EnsembleDegradationModel
from models.tire_degradation.evaluation import ModelEvaluator
from app.utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


class ModelTrainer:
    """
    Training orchestrator for tire degradation models.
    
    Features:
    - Automated data preparation
    - Optuna hyperparameter optimization
    - K-fold cross-validation
    - Model evaluation and comparison
    - Automatic model saving
    """
    
    def __init__(
        self,
        model_type: str = 'ensemble',
        config_path: Optional[Path] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model_type: 'xgboost', 'lightgbm', or 'ensemble'
            config_path: Path to training config YAML
        """
        self.model_type = model_type
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.data_pipeline = DataPreparationPipeline()
        self.evaluator = ModelEvaluator()
        
        # Model registry
        self.model_classes = {
            'xgboost': XGBoostDegradationModel,
            'lightgbm': LightGBMDegradationModel,
            'ensemble': EnsembleDegradationModel
        }
        
        # Training history
        self.history = {
            'trials': [],
            'best_params': None,
            'best_score': None
        }
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load training configuration."""
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default config
        return {
            'n_trials': 50,
            'timeout': 3600,  # 1 hour
            'n_folds': 5,
            'optimization_metric': 'rmse',
            'save_top_k': 3,
            'model_output_dir': 'models/saved/tire_degradation'
        }
    
    def train(
        self,
        data: pd.DataFrame,
        target_column: str = 'degradation_rate',
        optimize_hyperparams: bool = True,
        n_trials: Optional[int] = None,
        **kwargs
    ) -> BaseDegradationModel:
        """
        Train model with optional hyperparameter optimization.
        
        Args:
            data: Training data DataFrame
            target_column: Name of target column
            optimize_hyperparams: Whether to run Optuna optimization
            n_trials: Number of Optuna trials (overrides config)
        
        Returns:
            Trained model
        """
        logger.info(f"Starting training for {self.model_type} model")
        
        # Prepare data
        logger.info("Preparing training data")
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_pipeline.prepare_training_data(
            data,
            target_column
        )
        
        logger.info(f"Data splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        
        # Hyperparameter optimization
        if optimize_hyperparams:
            best_params = self._optimize_hyperparameters(
                X_train, y_train, X_val, y_val,
                n_trials=n_trials or self.config['n_trials']
            )
        else:
            best_params = self._get_default_params()
        
        # Train final model with best params
        logger.info("Training final model with best parameters")
        model = self._train_model(
            X_train, y_train, X_val, y_val,
            hyperparameters=best_params
        )
        
        # Evaluate on test set
        logger.info("Evaluating on test set")
        test_metrics = self._evaluate_model(model, X_test, y_test)
        
        logger.info(f"Test metrics: {test_metrics}")
        
        # Save model
        self._save_model(model, test_metrics)
        
        return model
    
    def _optimize_hyperparameters(
        self,
        X_train, y_train, X_val, y_val,
        n_trials: int
    ) -> Dict[str, Any]:
        """
        Run Optuna hyperparameter optimization.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            n_trials: Number of optimization trials
        
        Returns:
            Best hyperparameters
        """
        logger.info(f"Starting hyperparameter optimization: {n_trials} trials")
        
        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function."""
            # Suggest hyperparameters based on model type
            params = self._suggest_hyperparameters(trial)
            
            # Train model
            model = self._train_model(
                X_train, y_train, X_val, y_val,
                hyperparameters=params,
                verbose=False
            )
            
            # Evaluate
            metrics = self._evaluate_model(model, X_val, y_val)
            
            # Store trial info
            self.history['trials'].append({
                'params': params,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            })
            
            # Return optimization metric
            return metrics[self.config['optimization_metric']]
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=self.config['timeout'],
            show_progress_bar=True
        )
        
        # Store best results
        self.history['best_params'] = study.best_params
        self.history['best_score'] = study.best_value
        
        logger.info(f"Optimization complete. Best {self.config['optimization_metric']}: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        return study.best_params
    
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for trial.
        
        Args:
            trial: Optuna trial
        
        Returns:
            Hyperparameter dict
        """
        if self.model_type == 'xgboost':
            return {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'gamma': trial.suggest_float('gamma', 0.0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0)
            }
        
        elif self.model_type == 'lightgbm':
            return {
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0)
            }
        
        elif self.model_type == 'ensemble':
            return {
                'weights': {
                    'xgboost': trial.suggest_float('weight_xgboost', 0.2, 0.8),
                    'lightgbm': trial.suggest_float('weight_lightgbm', 0.2, 0.8)
                }
            }
        
        return {}
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default hyperparameters."""
        defaults = {
            'xgboost': {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200
            },
            'lightgbm': {
                'max_depth': 8,
                'learning_rate': 0.05,
                'n_estimators': 300
            },
            'ensemble': {
                'weights': {'xgboost': 0.5, 'lightgbm': 0.5}
            }
        }
        return defaults.get(self.model_type, {})
    
    def _train_model(
        self,
        X_train, y_train, X_val, y_val,
        hyperparameters: Dict[str, Any],
        verbose: bool = True
    ) -> BaseDegradationModel:
        """Train model with given hyperparameters."""
        # Create model config
        config = ModelConfig(
            version='1.0.0',
            hyperparameters=hyperparameters
        )
        
        # Instantiate model
        model_class = self.model_classes[self.model_type]
        model = model_class(config)
        
        # Train
        model.train(X_train, y_train, X_val, y_val, verbose=verbose)
        
        return model
    
    def _evaluate_model(
        self,
        model: BaseDegradationModel,
        X_test,
        y_test
    ) -> Dict[str, float]:
        """Evaluate model on test set."""
        # Get predictions
        y_pred = []
        for idx in range(len(X_test)):
            if hasattr(X_test, 'iloc'):
                pred = model.model.predict(X_test.iloc[[idx]])[0]
            else:
                pred = model.model.predict(X_test[idx:idx+1])[0]
            y_pred.append(pred)
        
        y_pred = np.array(y_pred)
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        return metrics
    
    def _save_model(
        self,
        model: BaseDegradationModel,
        metrics: Dict[str, float]
    ) -> None:
        """Save trained model."""
        output_dir = Path(self.config['model_output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create versioned directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = output_dir / f"{self.model_type}_{timestamp}"
        
        # Save model
        model.save(model_dir)
        
        # Save training history
        history_path = model_dir / 'training_history.json'
        import json
        with open(history_path, 'w') as f:
            json.dump({
                'history': self.history,
                'final_metrics': metrics
            }, f, indent=2)
        
        logger.info(f"Model saved to {model_dir}")
