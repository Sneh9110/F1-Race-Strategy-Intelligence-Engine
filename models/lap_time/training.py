"""
Model Training Pipeline with Hyperparameter Optimization

Orchestrates training of lap time prediction models with:
- Optuna hyperparameter optimization
- Cross-validation
- Model selection and persistence
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
from datetime import datetime

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold

from .base import BaseLapTimeModel, ModelConfig
from .xgboost_model import XGBoostLapTimeModel
from .lightgbm_model import LightGBMLapTimeModel
from .ensemble_model import EnsembleLapTimeModel
from .data_preparation import DataPreparationPipeline
from .registry import ModelRegistry
from .evaluation import ModelEvaluator
from config.settings import Settings

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Training pipeline for lap time prediction models.
    
    Features:
    - Optuna hyperparameter optimization (50 trials)
    - Cross-validation for robust evaluation
    - Automatic model selection
    - Versioned model persistence
    
    Attributes:
        data_pipeline: Data preparation pipeline
        registry: Model registry for versioning
        evaluator: Model evaluator for metrics
    """
    
    def __init__(
        self,
        data_path: Optional[Path] = None,
        registry_path: Optional[Path] = None
    ):
        """
        Initialize training pipeline.
        
        Args:
            data_path: Path to training data directory
            registry_path: Path to model registry directory
        """
        self.data_pipeline = DataPreparationPipeline()
        self.registry = ModelRegistry(registry_path)
        self.evaluator = ModelEvaluator()
        self.data_path = data_path or Path("data/processed")
        
        # Training configuration
        self.n_trials = 50
        self.n_folds = 5
        self.early_stopping_rounds = 20
        
    def train_model(
        self,
        model_type: str = "ensemble",
        optimize_hyperparams: bool = True,
        register_model: bool = True,
        version: Optional[str] = None
    ) -> Tuple[BaseLapTimeModel, Dict[str, float]]:
        """
        Train lap time prediction model.
        
        Args:
            model_type: Type of model ('xgboost', 'lightgbm', 'ensemble')
            optimize_hyperparams: Whether to run hyperparameter optimization
            register_model: Whether to register trained model
            version: Model version (auto-generated if None)
            
        Returns:
            Tuple of (trained_model, metrics)
        """
        logger.info(f"Starting training pipeline for {model_type} model")
        
        # Prepare data
        logger.info("Preparing training data...")
        train_data, val_data, test_data, feature_names = self.data_pipeline.prepare_training_data(
            self.data_path
        )
        
        X_train, y_train = train_data.drop('lap_time', axis=1), train_data['lap_time']
        X_val, y_val = val_data.drop('lap_time', axis=1), val_data['lap_time']
        X_test, y_test = test_data.drop('lap_time', axis=1), test_data['lap_time']
        
        logger.info(f"Training samples: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
        
        # Hyperparameter optimization
        if optimize_hyperparams:
            logger.info("Running hyperparameter optimization...")
            best_params = self._optimize_hyperparameters(
                model_type,
                X_train,
                y_train,
                X_val,
                y_val
            )
        else:
            best_params = {}
        
        # Train final model with best hyperparameters
        logger.info("Training final model...")
        model = self._create_model(model_type)
        training_metrics = model.train(
            X_train, y_train,
            X_val, y_val,
            **best_params
        )
        
        # Evaluate on test set
        logger.info("Evaluating model on test set...")
        test_metrics = self.evaluator.evaluate_model(
            model,
            X_test,
            y_test,
            test_data
        )
        
        # Combine metrics
        all_metrics = {
            **training_metrics,
            **{f'test_{k}': v for k, v in test_metrics.items()}
        }
        
        logger.info(f"Test RMSE: {test_metrics.get('rmse', 0):.3f}s, MAE: {test_metrics.get('mae', 0):.3f}s")
        logger.info(f"Test R²: {test_metrics.get('r2', 0):.3f}")
        
        # Register model
        if register_model:
            version = version or self._generate_version()
            model_info = self.registry.register_model(
                model,
                version=version,
                metrics=all_metrics,
                hyperparameters=best_params,
                model_type=model_type
            )
            logger.info(f"Model registered as version {version}")
        
        return model, all_metrics
    
    def cross_validate(
        self,
        model_type: str = "ensemble",
        hyperparams: Optional[Dict] = None
    ) -> Dict[str, List[float]]:
        """
        Perform cross-validation to assess model robustness.
        
        Args:
            model_type: Type of model to validate
            hyperparams: Hyperparameters to use (None for defaults)
            
        Returns:
            Dictionary of metrics across folds
        """
        logger.info(f"Starting {self.n_folds}-fold cross-validation for {model_type}")
        
        # Load full dataset
        train_data, _, _, _ = self.data_pipeline.prepare_training_data(self.data_path)
        X = train_data.drop('lap_time', axis=1)
        y = train_data['lap_time']
        
        # Setup k-fold
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        # Track metrics across folds
        fold_metrics = {
            'rmse': [],
            'mae': [],
            'r2': [],
        }
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
            logger.info(f"Training fold {fold}/{self.n_folds}")
            
            # Split data
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model = self._create_model(model_type)
            model.train(
                X_fold_train, y_fold_train,
                X_fold_val, y_fold_val,
                **(hyperparams or {})
            )
            
            # Evaluate
            val_data_fold = pd.concat([X_fold_val, y_fold_val], axis=1)
            metrics = self.evaluator.evaluate_model(
                model,
                X_fold_val,
                y_fold_val,
                val_data_fold
            )
            
            # Store metrics
            for key in fold_metrics:
                if key in metrics:
                    fold_metrics[key].append(metrics[key])
        
        # Calculate summary statistics
        summary = {}
        for key, values in fold_metrics.items():
            summary[f'{key}_mean'] = np.mean(values)
            summary[f'{key}_std'] = np.std(values)
        
        logger.info("Cross-validation complete:")
        logger.info(f"  RMSE: {summary['rmse_mean']:.3f} ± {summary['rmse_std']:.3f}")
        logger.info(f"  MAE: {summary['mae_mean']:.3f} ± {summary['mae_std']:.3f}")
        logger.info(f"  R²: {summary['r2_mean']:.3f} ± {summary['r2_std']:.3f}")
        
        return {**fold_metrics, **summary}
    
    def compare_models(self) -> pd.DataFrame:
        """
        Train and compare all model types.
        
        Returns:
            DataFrame with comparison metrics
        """
        logger.info("Comparing model types: xgboost, lightgbm, ensemble")
        
        model_types = ['xgboost', 'lightgbm', 'ensemble']
        results = []
        
        for model_type in model_types:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {model_type} model...")
            logger.info(f"{'='*60}")
            
            model, metrics = self.train_model(
                model_type=model_type,
                optimize_hyperparams=True,
                register_model=False
            )
            
            results.append({
                'model_type': model_type,
                **metrics
            })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)
        
        # Log comparison
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON")
        logger.info("="*60)
        logger.info(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def _create_model(self, model_type: str) -> BaseLapTimeModel:
        """
        Create model instance.
        
        Args:
            model_type: Type of model to create
            
        Returns:
            Model instance
        """
        config = ModelConfig(
            name=f"lap_time_{model_type}",
            version="1.0.0",
            model_type=model_type
        )
        
        if model_type == "xgboost":
            return XGBoostLapTimeModel(config)
        elif model_type == "lightgbm":
            return LightGBMLapTimeModel(config)
        elif model_type == "ensemble":
            return EnsembleLapTimeModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _optimize_hyperparameters(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            model_type: Type of model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Best hyperparameters
        """
        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function."""
            
            if model_type == "xgboost":
                params = {
                    'max_depth': trial.suggest_int('max_depth', 4, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0.0, 0.5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                }
            elif model_type == "lightgbm":
                params = {
                    'max_depth': trial.suggest_int('max_depth', 4, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                }
            else:  # ensemble
                params = {}
            
            # Train model
            model = self._create_model(model_type)
            model.train(X_train, y_train, X_val, y_val, **params)
            
            # Evaluate on validation set
            val_data = pd.concat([X_val, y_val], axis=1)
            metrics = self.evaluator.evaluate_model(model, X_val, y_val, val_data)
            
            # Return RMSE (to minimize)
            return metrics['rmse']
        
        # Create Optuna study
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        logger.info(f"Best trial value (RMSE): {study.best_value:.3f}s")
        logger.info(f"Best hyperparameters: {study.best_params}")
        
        return study.best_params
    
    def _generate_version(self) -> str:
        """
        Generate version string based on timestamp.
        
        Returns:
            Version string (e.g., "1.0.20240115_143022")
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"1.0.{timestamp}"
