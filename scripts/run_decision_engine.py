"""CLI for Decision Engine."""

import click
import json
from pathlib import Path
from typing import Optional

from decision_engine import DecisionEngine, DecisionInput, DecisionContext, SimulationContext
from decision_engine.explainer import DecisionExplainer


@click.group()
def cli():
    """Decision Engine CLI."""
    pass


@cli.command()
@click.option('--session-id', required=True, help='Session ID')
@click.option('--lap-number', required=True, type=int, help='Current lap')
@click.option('--driver-number', required=True, type=int, help='Driver number')
@click.option('--track-name', required=True, help='Track name')
@click.option('--config', default='config/decision_engine.yaml', help='Config path')
def decide(session_id: str, lap_number: int, driver_number: int, track_name: str, config: str):
    """Make single decision."""
    
    # Create sample context (would load from DB in production)
    context = DecisionContext(
        session_id=session_id,
        lap_number=lap_number,
        driver_number=driver_number,
        track_name=track_name,
        total_laps=78,
        current_position=3,
        tire_age=20,
        tire_compound='MEDIUM',
        fuel_load=85.0,
        stint_number=2,
        pit_stops_completed=1,
        recent_lap_times=[90.5, 90.8, 91.2, 91.5, 91.8],
    )
    
    # Create input
    decision_input = DecisionInput(context=context)
    
    # Make decision
    engine = DecisionEngine(config_path=config)
    output = engine.make_decision(decision_input)
    
    # Print results
    click.echo(f"\n{'='*60}")
    click.echo(f"Decision for Driver {driver_number} at Lap {lap_number}")
    click.echo(f"{'='*60}\n")
    
    if output.recommendations:
        for i, rec in enumerate(output.recommendations, 1):
            click.echo(f"{i}. {DecisionExplainer.generate_explanation_text(rec)}\n")
        
        click.echo(f"\nComparison Table:")
        click.echo(DecisionExplainer.generate_comparison_table(output.recommendations))
    else:
        click.echo("No recommendations")
    
    click.echo(f"\nComputation time: {output.computation_time_ms:.1f}ms")


@cli.command()
@click.option('--session-id', required=True, help='Session ID')
@click.option('--lap-number', required=True, type=int, help='Current lap')
@click.option('--config', default='config/decision_engine.yaml', help='Config path')
def decide_batch(session_id: str, lap_number: int, config: str):
    """Make decisions for all drivers."""
    
    click.echo(f"Batch decision for session {session_id} lap {lap_number}")
    click.echo("(Full implementation would load all driver states)")
    
    # Simplified: just show one driver
    engine = DecisionEngine(config_path=config)
    stats = engine.get_stats()
    
    click.echo(f"\nEngine stats:")
    click.echo(f"  Decisions: {stats['decision_count']}")
    click.echo(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
    click.echo(f"  Modules: {stats['module_count']}")


@cli.command()
@click.option('--num-runs', default=100, help='Number of benchmark runs')
@click.option('--config', default='config/decision_engine.yaml', help='Config path')
def benchmark(num_runs: int, config: str):
    """Benchmark decision engine performance."""
    import time
    
    click.echo(f"Benchmarking {num_runs} runs...")
    
    engine = DecisionEngine(config_path=config, enable_cache=False)
    
    # Create sample input
    context = DecisionContext(
        session_id='benchmark',
        lap_number=25,
        driver_number=44,
        track_name='Monaco',
        total_laps=78,
        current_position=3,
        tire_age=20,
        tire_compound='MEDIUM',
        fuel_load=85.0,
        stint_number=2,
        pit_stops_completed=1,
        recent_lap_times=[90.5, 90.8, 91.2],
    )
    
    decision_input = DecisionInput(context=context)
    
    # Benchmark
    latencies = []
    for i in range(num_runs):
        start = time.time()
        output = engine.make_decision(decision_input)
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        
        if (i + 1) % 10 == 0:
            click.echo(f"  Progress: {i+1}/{num_runs}")
    
    # Stats
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    
    click.echo(f"\nBenchmark Results:")
    click.echo(f"  Runs: {num_runs}")
    click.echo(f"  p50: {p50:.1f}ms")
    click.echo(f"  p95: {p95:.1f}ms")
    click.echo(f"  p99: {p99:.1f}ms")
    click.echo(f"  Target: <200ms (p95 {'PASS' if p95 < 200 else 'FAIL'})")


@cli.command()
@click.option('--session-id', required=True, help='Session ID')
@click.option('--output', default='decision_history.json', help='Output path')
def audit(session_id: str, output: str):
    """Audit decision accuracy post-race."""
    
    click.echo(f"Auditing session {session_id}...")
    click.echo("(Full implementation would compare decisions vs outcomes)")
    click.echo(f"Results would be saved to {output}")


@cli.command()
@click.option('--category', help='Filter by category')
@click.option('--enabled-only', is_flag=True, default=True, help='Only show enabled')
def list_modules(category: Optional[str], enabled_only: bool):
    """List registered decision modules."""
    from decision_engine.registry import DecisionModuleRegistry
    from decision_engine.schemas import DecisionCategory
    
    registry = DecisionModuleRegistry()
    
    # Parse category
    cat_filter = None
    if category:
        try:
            cat_filter = DecisionCategory(category.lower())
        except ValueError:
            click.echo(f"Invalid category: {category}")
            return
    
    modules = registry.list_modules(category=cat_filter, enabled_only=enabled_only)
    
    click.echo(f"\nRegistered Decision Modules ({len(modules)}):\n")
    click.echo(f"{'Name':<25} {'Version':<10} {'Category':<20} {'Priority':<10} {'Enabled'}")
    click.echo("-" * 80)
    
    for module in modules:
        click.echo(
            f"{module.name:<25} {module.version:<10} "
            f"{module.category.value:<20} {module.priority:<10} "
            f"{'✓' if module.enabled else '✗'}"
        )


if __name__ == '__main__':
    cli()
