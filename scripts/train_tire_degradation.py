"""
Training script for tire degradation models.

CLI for training, evaluating, and managing tire degradation models.
"""

import click
import yaml
import pandas as pd
from pathlib import Path
from typing import Optional

from models.tire_degradation.training import ModelTrainer
from models.tire_degradation.registry import ModelRegistry
from models.tire_degradation.evaluation import ModelEvaluator
from models.tire_degradation.data_preparation import DataPreparationPipeline
from data_pipeline.schemas.historical_schema import HistoricalStint
from app.utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


@click.group()
def cli():
    """Tire degradation model training CLI."""
    pass


@cli.command()
@click.option('--model-type', type=click.Choice(['xgboost', 'lightgbm', 'ensemble']),
              default='ensemble', help='Model type to train')
@click.option('--config', type=click.Path(exists=True), default=None,
              help='Path to training config YAML')
@click.option('--data-path', type=click.Path(exists=True), required=True,
              help='Path to training data (CSV or Parquet)')
@click.option('--optimize/--no-optimize', default=True,
              help='Enable hyperparameter optimization')
@click.option('--n-trials', type=int, default=None,
              help='Number of Optuna trials (overrides config)')
@click.option('--version', type=str, default=None,
              help='Model version (semantic versioning)')
@click.option('--alias', type=str, default=None,
              help='Model alias (e.g., "production", "staging")')
def train(
    model_type: str,
    config: Optional[str],
    data_path: str,
    optimize: bool,
    n_trials: Optional[int],
    version: Optional[str],
    alias: Optional[str]
):
    """Train a tire degradation model."""
    logger.info(f"Starting training: model_type={model_type}")
    
    # Load data
    click.echo(f"Loading data from {data_path}")
    data_path = Path(data_path)
    
    if data_path.suffix == '.csv':
        data = pd.read_csv(data_path)
    elif data_path.suffix == '.parquet':
        data = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported data format: {data_path.suffix}")
    
    click.echo(f"Loaded {len(data)} samples")
    
    # Load config
    config_path = Path(config) if config else None
    
    # Initialize trainer
    trainer = ModelTrainer(
        model_type=model_type,
        config_path=config_path
    )
    
    # Train model
    click.echo("Training model...")
    model = trainer.train(
        data=data,
        optimize_hyperparams=optimize,
        n_trials=n_trials
    )
    
    # Register model
    if version:
        click.echo(f"Registering model version {version}")
        registry = ModelRegistry()
        registry.register_model(
            model=model,
            version=version,
            alias=alias
        )
    
    click.echo("Training complete!")


@cli.command()
@click.option('--version', type=str, default='latest',
              help='Model version to evaluate')
@click.option('--data-path', type=click.Path(exists=True), required=True,
              help='Path to test data')
@click.option('--output', type=click.Path(), default=None,
              help='Output path for evaluation report')
def evaluate(version: str, data_path: str, output: Optional[str]):
    """Evaluate a trained model."""
    logger.info(f"Evaluating model version: {version}")
    
    # Load model
    click.echo(f"Loading model version {version}")
    registry = ModelRegistry()
    model = registry.load_model(version)
    
    # Load test data
    click.echo(f"Loading test data from {data_path}")
    data_path = Path(data_path)
    
    if data_path.suffix == '.csv':
        data = pd.read_csv(data_path)
    elif data_path.suffix == '.parquet':
        data = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported data format: {data_path.suffix}")
    
    # Prepare data
    pipeline = DataPreparationPipeline()
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.prepare_training_data(
        data, target_column='degradation_rate'
    )
    
    # Evaluate
    click.echo("Evaluating model...")
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(model, X_test, y_test, additional_data=data)
    
    # Display metrics
    click.echo("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        click.echo(f"  {metric}: {value:.4f}")
    
    # Save report
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        click.echo(f"\nReport saved to {output_path}")


@cli.command()
@click.option('--model-type', type=click.Choice(['xgboost', 'lightgbm', 'ensemble']),
              multiple=True, help='Model types to compare')
@click.option('--data-path', type=click.Path(exists=True), required=True,
              help='Path to test data')
def compare(model_type: tuple, data_path: str):
    """Compare multiple models."""
    logger.info(f"Comparing models: {model_type}")
    
    if len(model_type) < 2:
        click.echo("Error: Specify at least 2 models to compare")
        return
    
    # Load models
    registry = ModelRegistry()
    models = {}
    
    for mt in model_type:
        click.echo(f"Loading {mt} model...")
        try:
            model = registry.load_model(f"{mt}_latest")
            models[mt] = model
        except Exception as e:
            click.echo(f"Warning: Could not load {mt}: {e}")
    
    if len(models) < 2:
        click.echo("Error: Could not load enough models for comparison")
        return
    
    # Load test data
    data_path = Path(data_path)
    if data_path.suffix == '.csv':
        data = pd.read_csv(data_path)
    else:
        data = pd.read_parquet(data_path)
    
    # Prepare data
    pipeline = DataPreparationPipeline()
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.prepare_training_data(
        data, target_column='degradation_rate'
    )
    
    # Compare
    click.echo("\nComparing models...")
    evaluator = ModelEvaluator()
    comparison = evaluator.compare_models(models, X_test, y_test)
    
    # Display comparison
    click.echo("\nModel Comparison:")
    click.echo(comparison.to_string(index=False))


@cli.command()
@click.option('--version', type=str, required=True,
              help='Model version to promote')
@click.option('--from-env', type=str, default='staging',
              help='Source environment')
@click.option('--to-env', type=str, default='production',
              help='Target environment')
def promote(version: str, from_env: str, to_env: str):
    """Promote a model to production."""
    logger.info(f"Promoting model {version} from {from_env} to {to_env}")
    
    registry = ModelRegistry()
    
    # Confirm promotion
    click.confirm(
        f"Promote model {version} to {to_env}?",
        abort=True
    )
    
    # Promote
    registry.promote_model(version, from_env, to_env)
    
    click.echo(f"Model {version} promoted to {to_env}")


@cli.command()
def list_models():
    """List all registered models."""
    registry = ModelRegistry()
    models = registry.list_models()
    
    if not models:
        click.echo("No models registered")
        return
    
    click.echo("\nRegistered Models:")
    click.echo("-" * 80)
    
    for model in models:
        version = model['version']
        model_type = model['model_type']
        registered_at = model['registered_at']
        metrics = model.get('performance_metrics', {})
        
        click.echo(f"\nVersion: {version}")
        click.echo(f"  Type: {model_type}")
        click.echo(f"  Registered: {registered_at}")
        
        if metrics:
            click.echo("  Metrics:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    click.echo(f"    {metric}: {value:.4f}")


if __name__ == '__main__':
    cli()
