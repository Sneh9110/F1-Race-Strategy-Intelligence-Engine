#!/usr/bin/env python
"""
CLI tool for computing features using the feature engineering pipeline.

Supports:
- Batch computation for historical sessions
- Backfilling features across date ranges
- Real-time streaming computation
- Feature validation and listing

Usage:
    python scripts/compute_features.py compute-batch --session-ids 2024_MONACO_RACE
    python scripts/compute_features.py backfill --start-date 2024-01-01 --end-date 2024-12-31
    python scripts/compute_features.py compute-realtime --session-id 2024_MONACO_RACE
    python scripts/compute_features.py list-features
    python scripts/compute_features.py validate-features --session-id 2024_MONACO_RACE
"""

import asyncio
import click
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from features import BatchFeatureEngine, StreamingFeatureEngine, FeatureRegistry
from app.utils.logger import get_logger

logger = get_logger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose: bool):
    """F1 Feature Engineering CLI Tool"""
    if verbose:
        import logging
        logging.getLogger('features').setLevel(logging.DEBUG)


@cli.command('compute-batch')
@click.option('--session-ids', '-s', multiple=True, required=True, 
              help='Session IDs to compute features for (e.g., 2024_MONACO_RACE)')
@click.option('--feature-names', '-f', multiple=True, 
              help='Specific features to compute (default: all registered)')
@click.option('--parallel/--sequential', default=True, 
              help='Use parallel processing (default: True)')
@click.option('--num-workers', '-w', type=int, default=4, 
              help='Number of parallel workers (default: 4)')
@click.option('--output-dir', '-o', type=click.Path(), 
              help='Custom output directory for features')
@click.option('--force', is_flag=True, 
              help='Force recomputation even if features exist')
def compute_batch(
    session_ids: tuple,
    feature_names: tuple,
    parallel: bool,
    num_workers: int,
    output_dir: Optional[str],
    force: bool
):
    """
    Compute features for one or more historical sessions.
    
    Examples:
        compute-batch -s 2024_MONACO_RACE -s 2024_SILVERSTONE_RACE
        compute-batch -s 2024_MONACO_RACE -f stint_summary -f degradation_slope
        compute-batch -s 2024_MONACO_RACE --sequential
    """
    try:
        click.echo(f"🏎️  Computing features for {len(session_ids)} session(s)...")
        
        # Initialize engine
        engine_kwargs = {}
        if output_dir:
            engine_kwargs['feature_store_path'] = output_dir
        
        engine = BatchFeatureEngine(**engine_kwargs)
        
        # Compute features
        results = engine.compute_features(
            session_ids=list(session_ids),
            feature_names=list(feature_names) if feature_names else None,
            parallel=parallel,
            num_workers=num_workers if parallel else 1,
            force_recompute=force
        )
        
        # Display results
        total = len(results)
        successful = sum(1 for r in results.values() if r.get('status') == 'success')
        failed = total - successful
        
        click.echo(f"✅ Successfully computed: {successful}/{total}")
        if failed > 0:
            click.echo(f"❌ Failed: {failed}/{total}")
            for session_id, result in results.items():
                if result.get('status') != 'success':
                    click.echo(f"  - {session_id}: {result.get('error', 'Unknown error')}")
        
        # Show feature counts
        if successful > 0:
            click.echo("\n📊 Feature Statistics:")
            for session_id, result in results.items():
                if result.get('status') == 'success':
                    feature_count = len(result.get('features', {}))
                    click.echo(f"  - {session_id}: {feature_count} features")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        logger.exception("Feature computation failed")
        raise click.Abort()


@cli.command('backfill')
@click.option('--start-date', '-s', required=True, type=click.DateTime(formats=['%Y-%m-%d']),
              help='Start date (YYYY-MM-DD)')
@click.option('--end-date', '-e', required=True, type=click.DateTime(formats=['%Y-%m-%d']),
              help='End date (YYYY-MM-DD)')
@click.option('--feature-names', '-f', multiple=True,
              help='Specific features to backfill (default: all registered)')
@click.option('--parallel/--sequential', default=True,
              help='Use parallel processing (default: True)')
@click.option('--num-workers', '-w', type=int, default=4,
              help='Number of parallel workers (default: 4)')
@click.option('--batch-size', '-b', type=int, default=10,
              help='Number of sessions per batch (default: 10)')
@click.option('--output-dir', '-o', type=click.Path(),
              help='Custom output directory for features')
def backfill(
    start_date: datetime,
    end_date: datetime,
    feature_names: tuple,
    parallel: bool,
    num_workers: int,
    batch_size: int,
    output_dir: Optional[str]
):
    """
    Backfill features for all sessions in a date range.
    
    Examples:
        backfill -s 2024-01-01 -e 2024-12-31
        backfill -s 2024-06-01 -e 2024-06-30 -f stint_summary
        backfill -s 2024-01-01 -e 2024-03-31 --batch-size 20
    """
    try:
        click.echo(f"🏎️  Backfilling features from {start_date.date()} to {end_date.date()}...")
        
        # Initialize engine
        engine_kwargs = {}
        if output_dir:
            engine_kwargs['feature_store_path'] = output_dir
        
        engine = BatchFeatureEngine(**engine_kwargs)
        
        # Backfill features
        results = engine.backfill_features(
            start_date=start_date,
            end_date=end_date,
            feature_names=list(feature_names) if feature_names else None,
            parallel=parallel,
            num_workers=num_workers if parallel else 1,
            batch_size=batch_size
        )
        
        # Display results
        total = len(results)
        successful = sum(1 for r in results.values() if r.get('status') == 'success')
        failed = total - successful
        
        click.echo(f"✅ Successfully backfilled: {successful}/{total} sessions")
        if failed > 0:
            click.echo(f"❌ Failed: {failed}/{total} sessions")
        
        # Show date coverage
        if successful > 0:
            dates = [datetime.strptime(sid.split('_')[0], '%Y') for sid in results.keys() 
                    if results[sid].get('status') == 'success']
            if dates:
                click.echo(f"📅 Date range: {min(dates).date()} to {max(dates).date()}")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        logger.exception("Feature backfill failed")
        raise click.Abort()


@cli.command('compute-realtime')
@click.option('--session-id', '-s', required=True,
              help='Session ID for real-time computation')
@click.option('--feature-names', '-f', multiple=True,
              help='Specific features to compute (default: all registered)')
@click.option('--update-interval', '-i', type=float, default=1.0,
              help='Update interval in seconds (default: 1.0)')
@click.option('--duration', '-d', type=int,
              help='Duration to run in seconds (default: infinite)')
@click.option('--cache-ttl', type=int, default=300,
              help='Redis cache TTL in seconds (default: 300)')
def compute_realtime(
    session_id: str,
    feature_names: tuple,
    update_interval: float,
    duration: Optional[int],
    cache_ttl: int
):
    """
    Compute features in real-time for a live session.
    
    Examples:
        compute-realtime -s 2024_MONACO_RACE
        compute-realtime -s 2024_MONACO_RACE -f stint_summary -i 0.5
        compute-realtime -s 2024_MONACO_RACE -d 3600
    """
    async def _compute_realtime():
        try:
            click.echo(f"🏎️  Starting real-time feature computation for {session_id}...")
            click.echo(f"⏱️  Update interval: {update_interval}s")
            if duration:
                click.echo(f"⏱️  Duration: {duration}s")
            
            # Initialize engine
            engine = StreamingFeatureEngine(cache_ttl=cache_ttl)
            
            start_time = datetime.utcnow()
            iteration = 0
            
            while True:
                iteration += 1
                
                # Compute features
                results = await engine.compute_realtime(
                    session_id=session_id,
                    feature_names=list(feature_names) if feature_names else None
                )
                
                # Display results
                successful = sum(1 for r in results.values() if r.success)
                total = len(results)
                latency = sum(r.compute_time for r in results.values() if r.success) / max(successful, 1)
                
                click.echo(
                    f"[{iteration:04d}] ✅ {successful}/{total} features | "
                    f"⏱️  {latency*1000:.1f}ms avg latency"
                )
                
                # Check duration
                if duration and (datetime.utcnow() - start_time).total_seconds() >= duration:
                    click.echo("⏹️  Duration reached, stopping...")
                    break
                
                # Wait for next iteration
                await asyncio.sleep(update_interval)
            
            # Cleanup
            await engine.cleanup_old_states()
            click.echo("✅ Real-time computation completed")
            
        except KeyboardInterrupt:
            click.echo("\n⏹️  Stopped by user")
        except Exception as e:
            click.echo(f"❌ Error: {str(e)}", err=True)
            logger.exception("Real-time computation failed")
            raise click.Abort()
    
    asyncio.run(_compute_realtime())


@cli.command('list-features')
@click.option('--category', '-c', 
              type=click.Choice(['stint', 'pace', 'degradation', 'pitstop', 'tire', 
                               'weather', 'safety_car', 'traffic', 'telemetry', 'all']),
              default='all',
              help='Filter by feature category (default: all)')
@click.option('--show-dependencies', '-d', is_flag=True,
              help='Show feature dependencies')
def list_features(category: str, show_dependencies: bool):
    """
    List all registered features with their metadata.
    
    Examples:
        list-features
        list-features -c degradation
        list-features --show-dependencies
    """
    try:
        registry = FeatureRegistry()
        features = registry.list_features()
        
        # Filter by category
        if category != 'all':
            features = {
                name: info for name, info in features.items()
                if category in name.lower()
            }
        
        if not features:
            click.echo(f"No features found for category: {category}")
            return
        
        click.echo(f"\n📊 Registered Features ({len(features)}):\n")
        
        for name, info in sorted(features.items()):
            click.echo(f"  • {click.style(name, bold=True)}")
            click.echo(f"    Type: {info.feature_class.__name__}")
            click.echo(f"    Version: {info.version}")
            
            if show_dependencies and info.dependencies:
                deps = ', '.join(info.dependencies)
                click.echo(f"    Dependencies: {deps}")
            
            click.echo()
        
        # Show execution order
        if show_dependencies:
            click.echo("\n📋 Execution Order (topologically sorted):")
            execution_order = registry.compute_execution_order(list(features.keys()))
            for i, feature_name in enumerate(execution_order, 1):
                click.echo(f"  {i:2d}. {feature_name}")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        logger.exception("Failed to list features")
        raise click.Abort()


@cli.command('validate-features')
@click.option('--session-id', '-s', required=True,
              help='Session ID to validate features for')
@click.option('--feature-names', '-f', multiple=True,
              help='Specific features to validate (default: all stored)')
@click.option('--check-quality', '-q', is_flag=True,
              help='Perform quality checks (nulls, outliers, distributions)')
def validate_features(session_id: str, feature_names: tuple, check_quality: bool):
    """
    Validate computed features for a session.
    
    Examples:
        validate-features -s 2024_MONACO_RACE
        validate-features -s 2024_MONACO_RACE -f stint_summary --check-quality
    """
    try:
        from features.store import FeatureStore
        import pandas as pd
        
        click.echo(f"🔍 Validating features for {session_id}...")
        
        store = FeatureStore()
        
        # Load features
        features_to_check = list(feature_names) if feature_names else None
        df = store.load_features(session_id=session_id, feature_names=features_to_check)
        
        if df is None or df.empty:
            click.echo(f"❌ No features found for session: {session_id}")
            return
        
        click.echo(f"✅ Loaded {len(df)} rows across {len(df.columns)} feature columns")
        
        # Basic validation
        click.echo("\n📊 Basic Statistics:")
        click.echo(f"  - Rows: {len(df)}")
        click.echo(f"  - Columns: {len(df.columns)}")
        click.echo(f"  - Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Null check
        null_counts = df.isnull().sum()
        if null_counts.any():
            click.echo("\n⚠️  Null Values Found:")
            for col, count in null_counts[null_counts > 0].items():
                pct = count / len(df) * 100
                click.echo(f"  - {col}: {count} ({pct:.1f}%)")
        else:
            click.echo("\n✅ No null values found")
        
        # Quality checks
        if check_quality:
            click.echo("\n🔍 Quality Checks:")
            
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            for col in numeric_cols:
                # Outliers (using IQR method)
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                
                if outliers > 0:
                    pct = outliers / len(df) * 100
                    click.echo(f"  - {col}: {outliers} outliers ({pct:.1f}%)")
            
            # Distribution check
            click.echo("\n📈 Feature Ranges:")
            for col in numeric_cols[:10]:  # Show first 10
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = df[col].mean()
                click.echo(f"  - {col}: [{min_val:.3f}, {max_val:.3f}] (mean: {mean_val:.3f})")
        
        click.echo("\n✅ Validation completed")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        logger.exception("Feature validation failed")
        raise click.Abort()


if __name__ == '__main__':
    cli()
