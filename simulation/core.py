"""
Core race simulation engine.

Orchestrates all ML models to simulate full race outcomes with lap-by-lap predictions.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
import time
import yaml
import logging
import hashlib
import json
from pathlib import Path

from models import (
    DegradationPredictor,
    LapTimePredictor,
    SafetyCarPredictor,
    PitStopLossPredictor,
)
from models.tire_degradation.base import PredictionInput as DegInput
from models.lap_time.base import PredictionInput as LapTimeInput, RaceCondition
from models.safety_car.base import PredictionInput as SCInput, IncidentLog
from models.pit_stop_loss.base import PredictionInput as PitInput

from .schemas import (
    SimulationInput,
    SimulationOutput,
    DriverSimulationResult,
    LapResult,
    StintResult,
    PitStopInfo,
    TireCompound,
    StrategyRanking,
)
from .race_state import RaceState

logger = logging.getLogger(__name__)


class RaceSimulator:
    """Main race simulation engine."""
    
    def __init__(
        self,
        config_path: str = "config/simulation.yaml",
        track_config_path: str = "config/tracks.yaml",
        cache_client: Optional[Any] = None,
    ):
        """
        Initialize race simulator.
        
        Args:
            config_path: Path to simulation config
            track_config_path: Path to track configs
            cache_client: Optional Redis cache client
        """
        # Load configurations
        self.config = self._load_config(config_path)
        self.track_configs = self._load_track_configs(track_config_path)
        self.cache_client = cache_client
        
        # Initialize ML model predictors
        self.predictors = {
            "degradation": DegradationPredictor(),
            "lap_time": LapTimePredictor(),
            "safety_car": SafetyCarPredictor(),
            "pit_stop_loss": PitStopLossPredictor(),
        }
        
        # Performance tracking
        self.performance_stats = {
            "total_simulations": 0,
            "total_latency_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "model_calls": {name: 0 for name in self.predictors},
        }
        
        logger.info("RaceSimulator initialized with all 4 ML models")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load simulation configuration."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded simulation config from {config_path}")
            return config or {}
        except FileNotFoundError:
            logger.warning(f"Config not found: {config_path}, using defaults")
            return {
                "performance": {"max_latency_ms": 500, "enable_cache": True},
                "safety_car": {"duration_laps": 3, "lap_time_multiplier": 1.3},
                "traffic": {"dirty_air_threshold": 1.0, "penalty": 0.3},
            }
    
    def _load_track_configs(self, track_config_path: str) -> Dict[str, Any]:
        """Load track-specific configurations."""
        try:
            with open(track_config_path, 'r', encoding='utf-8') as f:
                configs = yaml.safe_load(f)
            logger.info(f"Loaded track configs from {track_config_path}")
            return configs or {}
        except FileNotFoundError:
            logger.warning(f"Track config not found: {track_config_path}, using defaults")
            return {}
    
    def simulate_race(self, sim_input: SimulationInput) -> SimulationOutput:
        """
        Simulate full race.
        
        Args:
            sim_input: Simulation input parameters
        
        Returns:
            Complete simulation output with results
        """
        start_time = time.time()
        
        # Check cache
        cache_key = self._generate_cache_key(sim_input)
        if self.cache_client and self.config.get("performance", {}).get("enable_cache"):
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.performance_stats["cache_hits"] += 1
                logger.info("Returning cached simulation result")
                return cached_result
            self.performance_stats["cache_misses"] += 1
        
        # Initialize race state
        race_state = RaceState(sim_input.race_config, sim_input.drivers)
        track_config = self._get_track_config(sim_input.race_config.track_name)
        
        # Track lap history for each driver
        lap_history = {driver_number: [] for driver_number in race_state.drivers.keys()}
        
        # Get strategies (per-driver or baseline)
        if sim_input.per_driver_strategies:
            strategies = sim_input.per_driver_strategies
            baseline_strategy = sim_input.strategy_to_evaluate
        else:
            # Use single strategy for all drivers
            baseline_strategy = sim_input.strategy_to_evaluate
            strategies = {driver_number: baseline_strategy for driver_number in race_state.drivers.keys()}
        
        # Simulation loop
        logger.info(
            f"Starting simulation: {sim_input.race_config.track_name}, "
            f"{sim_input.race_config.total_laps} laps, {len(sim_input.drivers)} drivers"
        )
        
        for lap in range(race_state.current_lap, sim_input.race_config.total_laps + 1):
            self._simulate_lap(
                race_state=race_state,
                lap_number=lap,
                strategies=strategies,
                track_config=track_config,
                sim_input=sim_input,
                lap_history=lap_history,
            )
            
            # Advance to next lap
            fuel_consumption = track_config.get("fuel_consumption_per_lap_kg", 1.5)
            race_state.advance_lap(fuel_consumption)
        
        # Build results
        results = self._build_driver_results(race_state, sim_input, lap_history)
        
        # Build strategy ranking for evaluated strategy
        target_driver_result = next((r for r in results if r.driver_number == sim_input.drivers[0].driver_number), results[0])
        eval_strategy = baseline_strategy if baseline_strategy else list(strategies.values())[0]
        strategy_ranking = StrategyRanking(
            strategy_id=f"strat_{'-'.join(map(str, eval_strategy.pit_laps))}",
            expected_position=float(target_driver_result.final_position),
            win_probability=target_driver_result.win_probability,
            total_time=target_driver_result.total_race_time,
            confidence=0.9,  # High confidence for deterministic sim
            risk_score=len(eval_strategy.pit_laps) * 0.1,
            pit_laps=eval_strategy.pit_laps,
            tire_sequence=eval_strategy.tire_sequence,
        )
        
        # Calculate metadata
        elapsed_ms = (time.time() - start_time) * 1000
        metadata = {
            "computation_time_ms": elapsed_ms,
            "track_name": sim_input.race_config.track_name,
            "total_laps": sim_input.race_config.total_laps,
            "model_versions": {
                name: "1.0.0" for name in self.predictors
            },
            "cache_hits": self.performance_stats["cache_hits"],
        }
        
        output = SimulationOutput(
            race_config=sim_input.race_config,
            results=results,
            strategy_rankings=[strategy_ranking],
            optimal_strategy=strategy_ranking,
            metadata=metadata,
        )
        
        # Cache result
        if self.cache_client and self.config.get("performance", {}).get("enable_cache"):
            self._cache_result(cache_key, output)
        
        self.performance_stats["total_simulations"] += 1
        self.performance_stats["total_latency_ms"] += elapsed_ms
        
        logger.info(f"Simulation complete in {elapsed_ms:.0f}ms")
        
        return output
    
    def _simulate_lap(
        self,
        race_state: RaceState,
        lap_number: int,
        strategies: Dict[int, Any],
        track_config: Dict[str, Any],
        sim_input: SimulationInput,
        lap_history: Dict[int, List[Dict[str, Any]]],
    ) -> None:
        """Simulate a single lap for all drivers."""
        
        # Predict safety car
        sc_probability = self._predict_safety_car(race_state, sim_input)
        
        # Deploy SC if probability threshold met (simplified: random not used here for determinism)
        if sc_probability > 0.7 and not race_state.safety_car_active:
            race_state.deploy_safety_car(lap_number)
        
        # Simulate each driver
        for driver_number, driver in race_state.drivers.items():
            # Get driver's strategy (fall back to any strategy if not found)
            strategy = strategies.get(driver_number, list(strategies.values())[0])
            
            # Check if pit lap
            if lap_number in strategy.pit_laps:
                pit_index = strategy.pit_laps.index(lap_number)
                new_compound = strategy.tire_sequence[pit_index + 1]
                
                # Predict pit loss
                pit_loss = self._predict_pit_loss(
                    driver=driver,
                    new_compound=new_compound,
                    race_state=race_state,
                    track_config=track_config,
                )
                
                race_state.execute_pit_stop(driver_number, new_compound, pit_loss)
            
            # Predict lap time
            lap_time = self._predict_lap_time(
                driver=driver,
                race_state=race_state,
                track_config=track_config,
                sim_input=sim_input,
            )
            
            # Update driver cumulative time
            driver.cumulative_race_time += lap_time
            
            # Update recent lap times
            driver.recent_lap_times.append(lap_time)
            if len(driver.recent_lap_times) > 5:
                driver.recent_lap_times.pop(0)
            
            # Record lap data
            gaps = race_state.get_driver_gaps()
            lap_history[driver_number].append({
                'lap_number': lap_number,
                'lap_time': lap_time,
                'tire_age': driver.tire_age,
                'fuel_load': driver.fuel_load,
                'position': driver.current_position,
                'gap_to_leader': gaps[driver_number],
                'tire_compound': driver.tire_compound,
                'safety_car_active': race_state.safety_car_active,
            })
        
        # Recalculate positions
        race_state.recalculate_positions()
    
    def _predict_lap_time(
        self,
        driver: Any,
        race_state: RaceState,
        track_config: Dict[str, Any],
        sim_input: SimulationInput,
    ) -> float:
        """Predict lap time with fallback."""
        try:
            self.performance_stats["model_calls"]["lap_time"] += 1
            
            # Predict degradation first
            degradation_rate = self._predict_degradation(driver, sim_input)
            
            # Map to RaceCondition enum
            if race_state.safety_car_active:
                race_condition = RaceCondition.SAFETY_CAR
            else:
                traffic_state = race_state.get_traffic_state(driver.driver_number)
                race_condition = RaceCondition.DIRTY_AIR if traffic_state.value == "DIRTY_AIR" else RaceCondition.CLEAN_AIR
            
            # Build prediction input
            lap_time_input = LapTimeInput(
                tire_age=driver.tire_age,
                tire_compound=driver.tire_compound,
                fuel_load=driver.fuel_load,
                traffic_state=race_condition,
                gap_to_ahead=driver.gap_to_ahead or 999.0,
                safety_car_active=race_state.safety_car_active,
                weather_temp=sim_input.race_config.weather_temp,
                track_temp=sim_input.race_config.track_temp,
                track_name=sim_input.race_config.track_name,
                driver_number=driver.driver_number,
                lap_number=race_state.current_lap,
                session_progress=race_state.current_lap / sim_input.race_config.total_laps,
            )
            
            # Predict
            prediction = self.predictors["lap_time"].predict(lap_time_input)
            lap_time = prediction.predicted_lap_time
            
            # Apply degradation effect (additional time per lap from tire wear)
            lap_time += degradation_rate
            
            # Apply safety car effect
            if race_state.safety_car_active:
                sc_multiplier = self.config.get("safety_car", {}).get("lap_time_multiplier", 1.3)
                lap_time *= sc_multiplier
            
            return lap_time
            
        except Exception as e:
            logger.warning(f"Lap time prediction failed: {e}, using fallback")
            return self._fallback_lap_time(driver, track_config, race_state)
    
    def _predict_degradation(
        self,
        driver: Any,
        sim_input: SimulationInput,
    ) -> float:
        """Predict tire degradation rate with fallback."""
        try:
            self.performance_stats["model_calls"]["degradation"] += 1
            
            # Build stint history from recent lap times
            stint_history = [
                {'lap': i + 1, 'lap_time': lap_time}
                for i, lap_time in enumerate(driver.recent_lap_times)
            ]
            
            # Build degradation input
            deg_input = DegInput(
                tire_compound=driver.tire_compound,
                tire_age=driver.tire_age,
                stint_history=stint_history,
                weather_temp=sim_input.race_config.weather_temp,
                driver_aggression=0.5,  # Default moderate aggression
                track_name=sim_input.race_config.track_name,
            )
            
            # Predict
            prediction = self.predictors["degradation"].predict(deg_input)
            
            # Return degradation rate (seconds per lap lost to tire wear)
            return prediction.degradation_rate if hasattr(prediction, 'degradation_rate') else 0.0
            
        except Exception as e:
            logger.warning(f"Degradation prediction failed: {e}, using fallback")
            # Fallback: compound-specific rates
            compound_rates = {"SOFT": 0.05, "MEDIUM": 0.03, "HARD": 0.02}
            return compound_rates.get(driver.tire_compound, 0.03)
    
    def _predict_pit_loss(
        self,
        driver: Any,
        new_compound: TireCompound,
        race_state: RaceState,
        track_config: Dict[str, Any],
    ) -> float:
        """Predict pit stop loss with fallback."""
        try:
            self.performance_stats["model_calls"]["pit_stop_loss"] += 1
            
            # Count cars in pit window (simplified: count cars within 10 positions)
            cars_in_window = min(5, abs(driver.current_position - 10))
            
            pit_input = PitInput(
                track_name=race_state.race_config.track_name,
                current_lap=race_state.current_lap,
                cars_in_pit_window=cars_in_window,
                pit_stop_duration=2.5,
                traffic_density=cars_in_window / 20.0,
                tire_compound_change=(driver.tire_compound != new_compound),
                current_position=driver.current_position,
                gap_to_ahead=driver.gap_to_ahead,
                gap_to_behind=driver.gap_to_behind,
            )
            
            prediction = self.predictors["pit_stop_loss"].predict(pit_input)
            pit_loss = prediction.total_pit_loss
            
            # Adjust for safety car
            if race_state.safety_car_active:
                pit_loss *= 0.5  # Pit loss halved under SC
            
            return pit_loss
            
        except Exception as e:
            logger.warning(f"Pit loss prediction failed: {e}, using fallback")
            base_loss = track_config.get("pit_loss_seconds", 22.0)
            return base_loss * (0.5 if race_state.safety_car_active else 1.0)
    
    def _predict_safety_car(
        self,
        race_state: RaceState,
        sim_input: SimulationInput,
    ) -> float:
        """Predict safety car probability."""
        try:
            self.performance_stats["model_calls"]["safety_car"] += 1
            
            # Build incident logs from event log
            incidents = [
                IncidentLog(lap=e["lap"], sector="T1", severity="moderate")
                for e in race_state.event_log[-5:]
                if e.get("event_type") == "incident"
            ]
            
            sc_input = SCInput(
                track_name=sim_input.race_config.track_name,
                current_lap=race_state.current_lap,
                total_laps=sim_input.race_config.total_laps,
                race_progress=race_state.current_lap / sim_input.race_config.total_laps,
                incident_logs=incidents,
                sector_risks={"T1": 0.3, "T2": 0.2, "T3": 0.25},
            )
            
            prediction = self.predictors["safety_car"].predict(sc_input)
            return prediction.sc_probability
            
        except Exception as e:
            logger.warning(f"SC prediction failed: {e}, using fallback")
            track_config = self._get_track_config(sim_input.race_config.track_name)
            return track_config.get("safety_car_probability", 0.15)
    
    def _fallback_lap_time(
        self,
        driver: Any,
        track_config: Dict[str, Any],
        race_state: RaceState,
    ) -> float:
        """Fallback lap time calculation."""
        base_lap_time = track_config.get("base_lap_time_seconds", 90.0)
        
        # Tire degradation
        compound_deg_rates = {"SOFT": 0.05, "MEDIUM": 0.03, "HARD": 0.02}
        deg_rate = compound_deg_rates.get(driver.tire_compound, 0.03)
        tire_effect = driver.tire_age * deg_rate
        
        # Fuel effect
        fuel_effect_per_kg = track_config.get("fuel_effect_per_lap_seconds", 0.03)
        fuel_effect = driver.fuel_load * fuel_effect_per_kg
        
        # Safety car
        sc_effect = 1.3 if race_state.safety_car_active else 1.0
        
        lap_time = (base_lap_time + tire_effect + fuel_effect) * sc_effect
        
        return lap_time
    
    def _build_driver_results(
        self,
        race_state: RaceState,
        sim_input: SimulationInput,
        lap_history: Dict[int, List[Dict[str, Any]]],
    ) -> List[DriverSimulationResult]:
        """Build driver simulation results."""
        results = []
        
        for driver_number, driver in race_state.drivers.items():
            # Build lap results from actual history
            laps = [
                LapResult(
                    lap_number=lap_data['lap_number'],
                    lap_time=lap_data['lap_time'],
                    tire_age=lap_data['tire_age'],
                    fuel_load=lap_data['fuel_load'],
                    position=lap_data['position'],
                    gap_to_leader=lap_data['gap_to_leader'],
                    tire_compound=lap_data['tire_compound'],
                    safety_car_active=lap_data['safety_car_active'],
                )
                for lap_data in lap_history.get(driver_number, [])
            ]
            
            # Build stint results from pit stop events
            pit_events = [e for e in race_state.event_log if e.get('event_type') == 'pit_stop' and e.get('driver_number') == driver_number]
            stints = []
            stint_start = 1
            stint_num = 1
            
            for pit_event in pit_events:
                pit_lap = pit_event['lap']
                stint_laps = [ld for ld in lap_history.get(driver_number, []) if stint_start <= ld['lap_number'] < pit_lap]
                if stint_laps:
                    avg_time = sum(ld['lap_time'] for ld in stint_laps) / len(stint_laps)
                    # Estimate degradation from lap time trend
                    if len(stint_laps) > 1:
                        deg_rate = (stint_laps[-1]['lap_time'] - stint_laps[0]['lap_time']) / len(stint_laps)
                    else:
                        deg_rate = 0.0
                    
                    stints.append(StintResult(
                        stint_number=stint_num,
                        start_lap=stint_start,
                        end_lap=pit_lap - 1,
                        tire_compound=stint_laps[0]['tire_compound'],
                        avg_lap_time=avg_time,
                        degradation_rate=max(0.0, deg_rate),
                    ))
                stint_start = pit_lap
                stint_num += 1
            
            # Final stint
            final_stint_laps = [ld for ld in lap_history.get(driver_number, []) if ld['lap_number'] >= stint_start]
            if final_stint_laps:
                avg_time = sum(ld['lap_time'] for ld in final_stint_laps) / len(final_stint_laps)
                if len(final_stint_laps) > 1:
                    deg_rate = (final_stint_laps[-1]['lap_time'] - final_stint_laps[0]['lap_time']) / len(final_stint_laps)
                else:
                    deg_rate = 0.0
                stints.append(StintResult(
                    stint_number=stint_num,
                    start_lap=stint_start,
                    end_lap=sim_input.race_config.total_laps,
                    tire_compound=final_stint_laps[0]['tire_compound'],
                    avg_lap_time=avg_time,
                    degradation_rate=max(0.0, deg_rate),
                ))
            
            # Build pit stop info from event log
            pit_stops = []
            for event in race_state.event_log:
                if event.get('event_type') == 'pit_stop' and event.get('driver_number') == driver_number:
                    pit_stops.append(PitStopInfo(
                        lap=event['lap'],
                        duration=event.get('pit_loss', 22.0),
                        loss=event.get('pit_loss', 22.0),
                        old_compound=event.get('old_compound', driver.tire_compound),
                        new_compound=event.get('new_compound', driver.tire_compound),
                    ))
            
            result = DriverSimulationResult(
                driver_number=driver_number,
                final_position=driver.current_position,
                total_race_time=driver.cumulative_race_time,
                laps=laps,
                stints=stints,
                pit_stops=pit_stops,
                win_probability=1.0 if driver.current_position == 1 else 0.0,
                podium_probability=1.0 if driver.current_position <= 3 else 0.0,
            )
            
            results.append(result)
        
        # Sort by final position
        results.sort(key=lambda r: r.final_position)
        
        return results
    
    def _get_track_config(self, track_name: str) -> Dict[str, Any]:
        """Get track configuration."""
        return self.track_configs.get(track_name, {
            "base_lap_time_seconds": 90.0,
            "pit_loss_seconds": 22.0,
            "fuel_consumption_per_lap_kg": 1.5,
            "fuel_effect_per_lap_seconds": 0.03,
            "safety_car_probability": 0.15,
        })
    
    def _generate_cache_key(self, sim_input: SimulationInput) -> str:
        """Generate cache key from input."""
        input_str = json.dumps(sim_input.dict(), sort_keys=True, default=str)
        return hashlib.md5(input_str.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[SimulationOutput]:
        """Retrieve cached result."""
        if not self.cache_client:
            return None
        
        try:
            cached = self.cache_client.get(f"sim:{cache_key}")
            if cached:
                return SimulationOutput(**json.loads(cached))
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    def _cache_result(self, cache_key: str, output: SimulationOutput) -> None:
        """Cache simulation result."""
        if not self.cache_client:
            return
        
        try:
            ttl = self.config.get("performance", {}).get("cache_ttl_seconds", 300)
            self.cache_client.setex(
                f"sim:{cache_key}",
                ttl,
                json.dumps(output.dict(), default=str),
            )
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
