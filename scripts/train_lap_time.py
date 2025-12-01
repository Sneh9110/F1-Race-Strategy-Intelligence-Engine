"""
Training CLI for Lap Time Prediction Models

Command-line interface for training, evaluating, and managing models.
"""

import logging
from pathlib import Path
from typing import Optional
import sys

import click
import pandas as pd

from models.lap_time.training import ModelTrainer
from models.lap_time.registry import ModelRegistry
from models.lap_time.evaluation import ModelEvaluator
from models.lap_time.data_preparation import DataPreparationPipeline
from config.settings import Settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Lap Time Prediction Model Training CLI"""
    pass


@cli.command()
@click.option(
    '--model-type',
    type=click.Choice(['xgboost', 'lightgbm', 'ensemble']),
    default='ensemble',
    help='Type of model to train'
)
@click.option(
    '--data-path',
    type=click.Path(exists=True),
    default='data/processed',
    help='Path to training data directory'
)
@click.option(
    '--optimize/--no-optimize',
    default=True,
    help='Whether to run hyperparameter optimization'
)
@click.option(
    '--register/--no-register',
    default=True,
    help='Whether to register trained model'
)
@click.option(
    '--version',
    type=str,
    default=None,
    help='Model version (auto-generated if not specified)'
)
def train(model_type: str, data_path: str, optimize: bool, register: bool, version: Optional[str]):
    """Train a lap time prediction model."""
    logger.info(f"Starting training for {model_type} model")
    
    try:
        trainer = ModelTrainer(
            data_path=Path(data_path),
            registry_path=Path("models/registry/lap_time")
        )
        
        model, metrics = trainer.train_model(
            model_type=model_type,
            optimize_hyperparams=optimize,
            register_model=register,
            version=version
        )
        
        # Display results
        click.echo("\n" + "="*60)
        click.echo("TRAINING RESULTS")
        click.echo("="*60)
        click.echo(f"Model Type: {model_type}")
        if version:
            click.echo(f"Version: {version}")
        click.echo(f"\nMetrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                click.echo(f"  {key}: {value:.4f}")
            else:
                click.echo(f"  {key}: {value}")
        
        click.echo("\n✅ Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        click.echo(f"❌ Training failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    '--model-type',
    type=click.Choice(['xgboost', 'lightgbm', 'ensemble']),
    default='ensemble',
    help='Type of model to cross-validate'
)
@click.option(
    '--data-path',
    type=click.Path(exists=True),
    default='data/processed',
    help='Path to training data directory'
)
@click.option(
    '--n-folds',
    type=int,
    default=5,
    help='Number of cross-validation folds'
)
def cross_validate(model_type: str, data_path: str, n_folds: int):
    """Perform cross-validation on a model."""
    logger.info(f"Starting {n_folds}-fold cross-validation for {model_type}")
    
    try:
        trainer = ModelTrainer(data_path=Path(data_path))
        trainer.n_folds = n_folds
        
        cv_results = trainer.cross_validate(model_type=model_type)
        
        # Display results
        click.echo("\n" + "="*60)
        click.echo(f"CROSS-VALIDATION RESULTS ({n_folds} folds)")
        click.echo("="*60)
        click.echo(f"Model Type: {model_type}")
        click.echo(f"\nMetrics:")
        for key, value in cv_results.items():
            if '_mean' in key or '_std' in key:
                click.echo(f"  {key}: {value:.4f}")
        
        click.echo("\n✅ Cross-validation completed successfully!")
        
    except Exception as e:
        logger.error(f"Cross-validation failed: {e}", exc_info=True)
        click.echo(f"❌ Cross-validation failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    '--version',
    type=str,
    required=True,
    help='Model version to evaluate'
)
@click.option(
    '--data-path',
    type=click.Path(exists=True),
    default='data/processed',
    help='Path to test data directory'
)
@click.option(
    '--output-path',
    type=click.Path(),
    default='results/evaluation',
    help='Path to save evaluation plots'
)
def evaluate(version: str, data_path: str, output_path: str):
    """Evaluate a trained model on test set."""
    logger.info(f"Evaluating model version {version}")
    
    try:
        # Load model
        registry = ModelRegistry()
        model = registry.load_model(version)
        
        # Load test data
        data_pipeline = DataPreparationPipeline()
        train_data, val_data, test_data, feature_names = data_pipeline.prepare_training_data(
            Path(data_path)
        )
        
        X_test = test_data.drop('lap_time', axis=1)
        y_test = test_data['lap_time']
        
        # Evaluate
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(
            model,
            X_test,
            y_test,
            test_data,
            output_path=Path(output_path)
        )
        
        # Display results
        click.echo("\n" + "="*60)
        click.echo("EVALUATION RESULTS")
        click.echo("="*60)
        click.echo(f"Model Version: {version}")
        click.echo(f"\nMetrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                click.echo(f"  {key}: {value:.4f}")
        
        click.echo(f"\nPlots saved to: {output_path}")
        click.echo("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        click.echo(f"❌ Evaluation failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    '--data-path',
    type=click.Path(exists=True),
    default='data/processed',
    help='Path to test data directory'
)
def compare(data_path: str):
    """Compare all model types."""
    logger.info("Comparing all model types")
    
    try:
        trainer = ModelTrainer(data_path=Path(data_path))
        comparison_df = trainer.compare_models()
        
        # Display results
        click.echo("\n" + "="*60)
        click.echo("MODEL COMPARISON")
        click.echo("="*60)
        click.echo(comparison_df.to_string())
        
        click.echo("\n✅ Comparison completed successfully!")
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        click.echo(f"❌ Comparison failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    '--version',
    type=str,
    required=True,
    help='Model version to promote'
)
@click.option(
    '--alias',
    type=click.Choice(['production', 'staging', 'latest']),
    default='production',
    help='Target alias'
)
def promote(version: str, alias: str):
    """Promote model version to alias."""
    logger.info(f"Promoting model {version} to {alias}")
    
    try:
        registry = ModelRegistry()
        registry.promote_model(version, alias)
        
        click.echo(f"✅ Model {version} promoted to {alias}")
        
    except Exception as e:
        logger.error(f"Promotion failed: {e}", exc_info=True)
        click.echo(f"❌ Promotion failed: {e}", err=True)
        sys.exit(1)


@cli.command('list-models')
def list_models():
    """List all registered model versions."""
    logger.info("Listing registered models")
    
    try:
        registry = ModelRegistry()
        versions = registry.list_versions()
        
        if not versions:
            click.echo("No models registered yet.")
            return
        
        # Display results
        click.echo("\n" + "="*60)
        click.echo("REGISTERED MODELS")
        click.echo("="*60)
        
        for i, version_info in enumerate(versions, 1):
            click.echo(f"\n{i}. Version: {version_info['version']}")
            click.echo(f"   Type: {version_info['model_type']}")
            click.echo(f"   Registered: {version_info['registered_at']}")
            if version_info.get('aliases'):
                click.echo(f"   Aliases: {', '.join(version_info['aliases'])}")
            
            # Display key metrics
            metrics = version_info.get('metrics', {})
            if 'test_rmse' in metrics:
                click.echo(f"   Test RMSE: {metrics['test_rmse']:.4f}s")
            if 'test_mae' in metrics:
                click.echo(f"   Test MAE: {metrics['test_mae']:.4f}s")
            if 'test_r2' in metrics:
                click.echo(f"   Test R²: {metrics['test_r2']:.4f}")
        
        click.echo("\n✅ Listed all registered models")
        
    except Exception as e:
        logger.error(f"Listing failed: {e}", exc_info=True)
        click.echo(f"❌ Listing failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    '--version',
    type=str,
    required=True,
    help='Model version to delete'
)
@click.option(
    '--force',
    is_flag=True,
    help='Force deletion even if version has aliases'
)
def delete(version: str, force: bool):
    """Delete a model version from registry."""
    logger.info(f"Deleting model version {version}")
    
    try:
        registry = ModelRegistry()
        
        # Confirm deletion
        if not force:
            click.confirm(
                f"Are you sure you want to delete model {version}?",
                abort=True
            )
        
        registry.delete_version(version, force=force)
        
        click.echo(f"✅ Model {version} deleted successfully")
        
    except Exception as e:
        logger.error(f"Deletion failed: {e}", exc_info=True)
        click.echo(f"❌ Deletion failed: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
