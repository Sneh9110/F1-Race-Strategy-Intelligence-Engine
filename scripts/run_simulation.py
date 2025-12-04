"""
CLI interface for race simulation engine.

Provides command-line tools for running simulations, strategy analysis, and what-if scenarios.
"""

import click
import json
import yaml
from pathlib import Path
from typing import Optional, List
import logging

from simulation.core import RaceSimulator
from simulation.strategy_tree import StrategyTreeExplorer
from simulation.monte_carlo import MonteCarloSimulator, MonteCarloConfig
from simulation.what_if import WhatIfEngine, ScenarioType
from simulation.schemas import (
    SimulationInput,
    RaceConfig,
    DriverState,
    StrategyOption,
    TireCompound,
    PaceTarget,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """F1 Race Strategy Intelligence Engine - Simulation CLI"""
    pass


@cli.command()
@click.option('--input-file', '-i', required=True, type=click.Path(exists=True), help='Input JSON file with race config')
@click.option('--output-file', '-o', type=click.Path(), help='Output JSON file for results')
@click.option('--mode', type=click.Choice(['single', 'monte_carlo', 'strategy_tree', 'what_if']), default='single', help='Simulation mode')
@click.option('--mc-runs', type=int, default=1000, help='Monte Carlo runs (for monte_carlo mode)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def simulate(input_file: str, output_file: Optional[str], mode: str, mc_runs: int, verbose: bool):
    """Run race simulation."""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load input
    logger.info(f"Loading input from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    sim_input = SimulationInput(**input_data)
    
    # Initialize simulator
    simulator = RaceSimulator()
    
    # Run based on mode
    if mode == 'single':
        logger.info("Running single deterministic simulation")
        result = simulator.simulate_race(sim_input)
        output_data = result.dict()
    
    elif mode == 'monte_carlo':
        logger.info(f"Running Monte Carlo simulation with {mc_runs} runs")
        config = MonteCarloConfig(num_runs=mc_runs)
        mc_sim = MonteCarloSimulator(simulator, config)
        result = mc_sim.run_monte_carlo(sim_input)
        output_data = result
    
    elif mode == 'strategy_tree':
        logger.info("Exploring strategy tree")
        explorer = StrategyTreeExplorer(simulator)
        rankings = explorer.explore_strategies(sim_input)
        output_data = {
            "strategies": [r.dict() for r in rankings],
            "best_strategy": rankings[0].dict() if rankings else None,
        }
    
    elif mode == 'what_if':
        logger.info("Running what-if scenario analysis")
        engine = WhatIfEngine(simulator)
        recommendations = engine.generate_scenario_recommendations(sim_input)
        output_data = {"recommendations": recommendations}
    
    # Output results
    if output_file:
        logger.info(f"Writing results to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, default=str)
        click.echo(f"✓ Results written to {output_file}")
    else:
        click.echo(json.dumps(output_data, indent=2, default=str))
    
    click.echo(f"✓ Simulation complete")


@cli.command()
@click.option('--track', required=True, help='Track name (e.g., Monaco, Monza)')
@click.option('--total-laps', type=int, required=True, help='Total race laps')
@click.option('--current-lap', type=int, default=1, help='Current lap')
@click.option('--output-file', '-o', type=click.Path(), help='Output template file')
def create_input(track: str, total_laps: int, current_lap: int, output_file: Optional[str]):
    """Create simulation input template."""
    
    # Create template race config
    race_config = RaceConfig(
        track_name=track,
        total_laps=total_laps,
        current_lap=current_lap,
        weather_temp=25.0,
        track_temp=35.0,
        grid_positions=list(range(1, 21)),
        safety_car_active=False,
        vsc_active=False,
    )
    
    # Create template drivers (20 drivers)
    drivers = []
    for i in range(1, 21):
        driver = DriverState(
            driver_number=i,
            current_position=i,
            tire_compound=TireCompound.MEDIUM,
            tire_age=0,
            fuel_load=110.0,
            gap_to_ahead=0.0 if i == 1 else 1.0,
            gap_to_behind=1.0 if i < 20 else 0.0,
            recent_lap_times=[90.0] * 5,
            num_pit_stops=0,
            current_stint=1,
            cumulative_race_time=0.0,
        )
        drivers.append(driver)
    
    # Create default strategy
    strategy = StrategyOption(
        pit_laps=[total_laps // 2],
        tire_sequence=[TireCompound.MEDIUM, TireCompound.HARD],
        target_pace=PaceTarget.BALANCED,
    )
    
    # Create simulation input
    sim_input = SimulationInput(
        race_config=race_config,
        drivers=drivers,
        strategy_to_evaluate=strategy,
        monte_carlo_runs=0,
    )
    
    # Output
    output_data = sim_input.dict()
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, default=str)
        click.echo(f"✓ Template created: {output_file}")
    else:
        click.echo(json.dumps(output_data, indent=2, default=str))


@cli.command()
@click.option('--input-file', '-i', required=True, type=click.Path(exists=True), help='Base input JSON file')
@click.option('--max-stops', type=int, default=3, help='Maximum pit stops')
@click.option('--output-file', '-o', type=click.Path(), help='Output file for ranked strategies')
def explore_strategies(input_file: str, max_stops: int, output_file: Optional[str]):
    """Explore and rank race strategies."""
    
    logger.info(f"Loading input from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    sim_input = SimulationInput(**input_data)
    
    # Initialize
    simulator = RaceSimulator()
    explorer = StrategyTreeExplorer(simulator, max_strategies=50)
    
    # Explore
    logger.info(f"Exploring strategies (max {max_stops} stops)")
    rankings = explorer.explore_strategies(sim_input, max_pit_stops=max_stops)
    
    # Format output
    output_data = {
        "total_strategies": len(rankings),
        "rankings": [
            {
                "rank": idx + 1,
                "pit_laps": r.strategy.pit_laps,
                "tire_sequence": [c.value for c in r.strategy.tire_sequence],
                "expected_time": r.expected_race_time,
                "win_probability": r.win_probability,
                "expected_position": r.expected_position,
            }
            for idx, r in enumerate(rankings[:10])  # Top 10
        ],
    }
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        click.echo(f"✓ Strategy rankings written to {output_file}")
    else:
        click.echo(json.dumps(output_data, indent=2))
    
    # Print summary
    if rankings:
        best = rankings[0]
        click.echo(f"\n✓ Best strategy:")
        click.echo(f"  Pit laps: {best.strategy.pit_laps}")
        click.echo(f"  Tires: {' → '.join(c.value for c in best.strategy.tire_sequence)}")
        click.echo(f"  Expected time: {best.expected_race_time:.2f}s")
        click.echo(f"  Win probability: {best.win_probability:.1%}")


@cli.command()
@click.option('--input-file', '-i', required=True, type=click.Path(exists=True), help='Base input JSON file')
@click.option('--scenario', type=click.Choice([s.value for s in ScenarioType]), required=True, help='Scenario type')
@click.option('--output-file', '-o', type=click.Path(), help='Output file')
def what_if(input_file: str, scenario: str, output_file: Optional[str]):
    """Run what-if scenario analysis."""
    
    logger.info(f"Loading input from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    sim_input = SimulationInput(**input_data)
    
    # Initialize
    simulator = RaceSimulator()
    engine = WhatIfEngine(simulator)
    
    # Get scenario
    scenario_type = ScenarioType(scenario)
    scenario_obj = engine.scenario_templates[scenario_type]
    
    # Analyze
    logger.info(f"Analyzing scenario: {scenario}")
    comparison = engine.analyze_scenario(sim_input, scenario_obj)
    
    # Format output
    output_data = {
        "scenario": scenario,
        "description": scenario_obj.description,
        "baseline_position": comparison.baseline_result.results[0].final_position,
        "scenario_position": comparison.scenario_result.results[0].final_position,
        "position_delta": comparison.position_delta,
        "time_delta": comparison.time_delta,
        "win_probability_delta": comparison.win_probability_delta,
    }
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        click.echo(f"✓ Analysis written to {output_file}")
    else:
        click.echo(json.dumps(output_data, indent=2))
    
    # Print summary
    click.echo(f"\n✓ What-if Analysis: {scenario}")
    click.echo(f"  {scenario_obj.description}")
    click.echo(f"  Position change: {comparison.position_delta:+d}")
    click.echo(f"  Time delta: {comparison.time_delta:+.2f}s")
    click.echo(f"  Win probability change: {comparison.win_probability_delta:+.1%}")


@cli.command()
def list_scenarios():
    """List available what-if scenarios."""
    
    simulator = RaceSimulator()
    engine = WhatIfEngine(simulator)
    
    click.echo("Available What-If Scenarios:\n")
    
    for scenario_type, scenario in engine.scenario_templates.items():
        click.echo(f"  {scenario_type.value}")
        click.echo(f"    {scenario.description}")
        click.echo()


if __name__ == '__main__':
    cli()
