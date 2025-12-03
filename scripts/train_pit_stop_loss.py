"""
Training script for Pit Stop Loss models.

CLI for training, evaluating, and managing pit stop loss models.
"""

import click
import yaml
import pandas as pd
from pathlib import Path
from typing import Optional

from models.pit_stop_loss.training import ModelTrainer as PitStopLossModelTrainer
from models.pit_stop_loss.registry import ModelRegistry as PitStopLossModelRegistry
from models.pit_stop_loss.evaluation import ModelEvaluator as PitStopLossModelEvaluator
from models.pit_stop_loss.xgboost_model import XGBoostPitStopLossModel
from models.pit_stop_loss.lightgbm_model import LightGBMPitStopLossModel
from models.pit_stop_loss.ensemble_model import EnsemblePitStopLossModel

import logging
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Pit Stop Loss model training CLI."""
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
@click.option('--n-trials', type=int, default=50,
              help='Number of Optuna trials')
@click.option('--version', type=str, default=None,
              help='Model version (semantic versioning)')
@click.option('--alias', type=str, default=None,
              help='Model alias (e.g., "production", "staging")')
def train(
    model_type: str,
    config: Optional[str],
    data_path: str,
    optimize: bool,
    n_trials: int,
    version: Optional[str],
    alias: Optional[str]
):
    """Train a Pit Stop Loss model."""
    logger.info(f"Starting training: model_type={model_type}")
    
    click.echo(f"Loading data from {data_path}")
    data_path = Path(data_path)
    
    if data_path.suffix == '.csv':
        data = pd.read_csv(data_path)
    elif data_path.suffix == '.parquet':
        data = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported data format: {data_path.suffix}")
    
    click.echo(f"Loaded {len(data)} samples")
    
    if model_type == 'xgboost':
        model = XGBoostPitStopLossModel()
    elif model_type == 'lightgbm':
        model = LightGBMPitStopLossModel()
    else:
        model = EnsemblePitStopLossModel()
    
    trainer = PitStopLossModelTrainer()
    click.echo("Training model...")
    result = trainer.train(model, data, optimize=optimize)
    
    click.echo(f"Training complete: {result['metrics']}")
    
    if version:
        registry = PitStopLossModelRegistry()
        registry.register_model(version, {"model": result.get("model"), "metrics": result.get("metrics")})
        click.echo(f"Model registered: {version}")
        if alias:
            registry.promote_model(version, alias)
            click.echo(f"Model promoted to {alias}")


@cli.command()
@click.option('--version', type=str, default='latest', help='Model version')
@click.option('--data-path', type=click.Path(exists=True), required=True,
              help='Path to evaluation data')
@click.option('--output', type=click.Path(), default=None,
              help='Path to save evaluation results (JSON)')
def evaluate(version: str, data_path: str, output: Optional[str]):
    """Evaluate a trained Pit Stop Loss model."""
    click.echo(f"Evaluating model {version}")
    
    registry = PitStopLossModelRegistry()
    model = registry.load_model(version)
    
    data = pd.read_csv(data_path) if data_path.endswith('.csv') else pd.read_parquet(data_path)
    
    evaluator = PitStopLossModelEvaluator()
    y_true = data.get('actual_pit_loss')
    y_pred = []  # placeholder
    metrics = evaluator.evaluate(y_true, y_pred)
    
    click.echo(f"Evaluation metrics: {metrics}")
    
    if output:
        import json
        with open(output, 'w') as fh:
            json.dump(metrics, fh, indent=2)
        click.echo(f"Results saved to {output}")


@cli.command()
@click.option('--version', type=str, default='latest', help='Model version to promote')
@click.option('--from-env', type=str, default='staging', help='Source environment')
@click.option('--to-env', type=str, default='production', help='Target environment')
def promote(version: str, from_env: str, to_env: str):
    """Promote a model between environments."""
    click.echo(f"Promoting {version} from {from_env} to {to_env}")
    
    registry = PitStopLossModelRegistry()
    registry.promote_model(version, to_env)
    
    click.echo(f"Model {version} promoted to {to_env}")


@cli.command(name='list-models')
def list_models():
    """List all registered models."""
    registry = PitStopLossModelRegistry()
    models = registry.list_models()
    
    click.echo("Registered models:")
    for ver, md in models.items():
        click.echo(f"  {ver}: {md}")


if __name__ == '__main__':
    cli()
