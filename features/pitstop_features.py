"""
Pit stop strategy feature calculators.

Analyzes undercut/overcut opportunities, pit loss, and optimal pit windows.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from features.base import BaseFeature, FeatureConfig
from features.registry import register_feature
from config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def load_tire_config() -> Dict[str, Any]:
    """Load tire compound configuration."""
    config_path = Path(settings.CONFIG_DIR) / "tire_compounds.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_track_config(track_name: str) -> Dict[str, Any]:
    """Load track-specific configuration."""
    config_path = Path(settings.CONFIG_DIR) / "tracks.yaml"
    with open(config_path, 'r') as f:
        tracks = yaml.safe_load(f)
    return tracks.get(track_name, {})


@register_feature(
    name='undercut_delta',
    version='v1.0.0',
    description='Calculate undercut time advantage',
    dependencies=[],
    computation_cost_ms=70,
    tags=['strategy', 'pitstop', 'undercut']
)
class UndercutDeltaFeature(BaseFeature):
    """
    Calculates undercut advantage: pitting before opponent.
    
    Formula:
        time_gained = (opponent_old_tire_pace - own_new_tire_pace) * laps_until_opponent_pits
                      - pit_loss - tire_warmup_penalty
    
    Output columns:
    - driver_number: Driver considering undercut
    - opponent_number: Target opponent
    - laps_until_opponent_pits: Expected laps until opponent pits
    - own_new_tire_pace: Expected pace on new tires
    - opponent_old_tire_pace: Opponent's pace on old tires
    - pit_loss: Time lost in pit stop
    - warmup_penalty: Time lost during tire warmup
    - undercut_delta: Net time gained/lost (positive = advantage)
    """
    
    def __init__(self, config: FeatureConfig = None, track_name: str = "generic"):
        super().__init__(
            name='undercut_delta',
            version='v1.0.0',
            config=config or FeatureConfig(thresholds={'undercut_window_laps': 3})
        )
        self.track_name = track_name
        self.tire_config = load_tire_config()
        self.track_config = load_track_config(track_name)
    
    def _calculate(
        self,
        data: pd.DataFrame,
        tire_compound: str = "MEDIUM",
        **kwargs
    ) -> pd.DataFrame:
        """
        Calculate undercut advantage.
        
        Args:
            data: DataFrame with columns: driver_number, current_pace, tire_age
            tire_compound: Target tire compound for undercut
        """
        results = []
        window_laps = self.config.thresholds.get('undercut_window_laps', 3)
        
        # Tire warmup penalty
        compound_config = self.tire_config.get('compounds', {}).get(tire_compound, {})
        warmup_laps = compound_config.get('warmup_laps', 2)
        warmup_penalty_per_lap = compound_config.get('warmup_penalty', 0.5)
        total_warmup_penalty = warmup_laps * warmup_penalty_per_lap
        
        # Pit loss
        pit_loss = self.track_config.get('pit_loss_seconds', 20.0)
        
        # New tire pace improvement
        new_tire_advantage = compound_config.get('new_tire_advantage', 0.3)
        
        for _, driver_row in data.iterrows():
            driver_num = driver_row['driver_number']
            
            # Compare against other drivers
            for _, opp_row in data.iterrows():
                if opp_row['driver_number'] == driver_num:
                    continue
                
                # Opponent's degraded pace
                opponent_pace = opp_row['current_pace']
                
                # Own pace on new tires (improvement from fresh rubber)
                own_new_pace = driver_row['current_pace'] - new_tire_advantage
                
                # Time gained per lap
                pace_delta_per_lap = opponent_pace - own_new_pace
                
                # Time gained over window
                time_gained = pace_delta_per_lap * window_laps
                
                # Subtract pit loss and warmup
                undercut_delta = time_gained - pit_loss - total_warmup_penalty
                
                results.append({
                    'driver_number': driver_num,
                    'opponent_number': opp_row['driver_number'],
                    'laps_until_opponent_pits': window_laps,
                    'own_new_tire_pace': float(own_new_pace),
                    'opponent_old_tire_pace': float(opponent_pace),
                    'pit_loss': float(pit_loss),
                    'warmup_penalty': float(total_warmup_penalty),
                    'undercut_delta': float(undercut_delta),
                    'undercut_viable': undercut_delta > 0.0
                })
        
        return pd.DataFrame(results)
    
    def _get_dependencies(self) -> List[str]:
        return []


@register_feature(
    name='overcut_delta',
    version='v1.0.0',
    description='Calculate overcut time advantage',
    dependencies=[],
    computation_cost_ms=60,
    tags=['strategy', 'pitstop', 'overcut']
)
class OvercutDeltaFeature(BaseFeature):
    """
    Calculates overcut advantage: staying out longer than opponent.
    
    Formula:
        time_gained = (own_extended_stint_pace - opponent_new_tire_pace) * extra_laps - pit_loss
    
    Output columns:
    - driver_number: Driver attempting overcut
    - opponent_number: Target opponent
    - extra_laps: Additional laps staying out
    - own_extended_pace: Own pace on old tires
    - opponent_new_tire_pace: Opponent's pace on new tires
    - overcut_delta: Net time advantage
    """
    
    def __init__(self, config: FeatureConfig = None, track_name: str = "generic"):
        super().__init__(
            name='overcut_delta',
            version='v1.0.0',
            config=config or FeatureConfig(thresholds={'overcut_window_laps': 5})
        )
        self.track_name = track_name
        self.tire_config = load_tire_config()
        self.track_config = load_track_config(track_name)
    
    def _calculate(
        self,
        data: pd.DataFrame,
        tire_compound: str = "MEDIUM",
        **kwargs
    ) -> pd.DataFrame:
        """Calculate overcut advantage."""
        results = []
        extra_laps = self.config.thresholds.get('overcut_window_laps', 5)
        
        compound_config = self.tire_config.get('compounds', {}).get(tire_compound, {})
        new_tire_advantage = compound_config.get('new_tire_advantage', 0.3)
        degradation_per_lap = compound_config.get('degradation_per_lap', 0.05)
        
        pit_loss = self.track_config.get('pit_loss_seconds', 20.0)
        
        for _, driver_row in data.iterrows():
            driver_num = driver_row['driver_number']
            current_pace = driver_row['current_pace']
            
            # Own pace will degrade over extra laps
            extended_pace = current_pace + (degradation_per_lap * extra_laps)
            
            for _, opp_row in data.iterrows():
                if opp_row['driver_number'] == driver_num:
                    continue
                
                # Opponent's new tire pace (after pitting)
                opp_new_pace = opp_row['current_pace'] - new_tire_advantage
                
                # Time delta per lap (negative = losing time)
                pace_delta_per_lap = opp_new_pace - extended_pace
                
                # Total time gained/lost
                overcut_delta = (pace_delta_per_lap * extra_laps) - pit_loss
                
                results.append({
                    'driver_number': driver_num,
                    'opponent_number': opp_row['driver_number'],
                    'extra_laps': extra_laps,
                    'own_extended_pace': float(extended_pace),
                    'opponent_new_tire_pace': float(opp_new_pace),
                    'pit_loss': float(pit_loss),
                    'overcut_delta': float(overcut_delta),
                    'overcut_viable': overcut_delta > 0.0
                })
        
        return pd.DataFrame(results)
    
    def _get_dependencies(self) -> List[str]:
        return []


@register_feature(
    name='pit_loss_model',
    version='v1.0.0',
    description='Calculate track-specific pit loss',
    dependencies=[],
    computation_cost_ms=30,
    tags=['strategy', 'pitstop', 'pit_loss']
)
class PitLossModelFeature(BaseFeature):
    """
    Computes total pit loss including congestion.
    
    Formula:
        total_pit_loss = base_pit_loss + congestion_penalty + pit_stop_duration
        congestion_penalty = count(cars_in_pit_window) * penalty_per_car
    
    Output columns:
    - driver_number: Driver number
    - base_pit_loss: Track-specific base pit loss
    - congestion_penalty: Additional time from traffic
    - pit_stop_duration: Actual pit stop time
    - total_pit_loss: Total time lost
    """
    
    def __init__(self, config: FeatureConfig = None, track_name: str = "generic"):
        super().__init__(
            name='pit_loss_model',
            version='v1.0.0',
            config=config or FeatureConfig(thresholds={'congestion_penalty_per_car': 2.0})
        )
        self.track_config = load_track_config(track_name)
    
    def _calculate(
        self,
        data: pd.DataFrame,
        cars_in_pit_window: int = 0,
        **kwargs
    ) -> pd.DataFrame:
        """Calculate pit loss."""
        base_pit_loss = self.track_config.get('pit_loss_seconds', 20.0)
        penalty_per_car = self.config.thresholds.get('congestion_penalty_per_car', 2.0)
        congestion_penalty = cars_in_pit_window * penalty_per_car
        
        results = []
        for _, row in data.iterrows():
            pit_stop_duration = row.get('pit_stop_duration', 2.5)
            
            total_pit_loss = base_pit_loss + congestion_penalty + pit_stop_duration
            
            results.append({
                'driver_number': row['driver_number'],
                'base_pit_loss': float(base_pit_loss),
                'congestion_penalty': float(congestion_penalty),
                'pit_stop_duration': float(pit_stop_duration),
                'total_pit_loss': float(total_pit_loss)
            })
        
        return pd.DataFrame(results)
    
    def _get_dependencies(self) -> List[str]:
        return []


@register_feature(
    name='pit_window',
    version='v1.0.0',
    description='Identify optimal pit stop window',
    dependencies=['cliff_detection'],
    computation_cost_ms=50,
    tags=['strategy', 'pitstop', 'window']
)
class PitWindowFeature(BaseFeature):
    """
    Identifies optimal pit stop window based on tire cliff.
    
    Output columns:
    - driver_number: Driver number
    - earliest_pit_lap: Earliest recommended pit lap
    - latest_pit_lap: Latest recommended pit lap
    - optimal_pit_lap: Optimal pit lap (minimize total race time)
    - window_size: Number of laps in pit window
    """
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(
            name='pit_window',
            version='v1.0.0',
            config=config or FeatureConfig(thresholds={'window_margin': 3})
        )
    
    def _calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Identify pit windows."""
        results = []
        margin = self.config.thresholds.get('window_margin', 3)
        
        for _, row in data.iterrows():
            if row.get('cliff_detected', False):
                cliff_lap = row['cliff_lap']
                earliest = max(1, cliff_lap - margin)
                latest = cliff_lap + 2
                optimal = cliff_lap - 1  # Pit just before cliff
            else:
                # No cliff detected, use conservative estimates
                earliest = 10
                latest = 25
                optimal = 15
            
            results.append({
                'driver_number': row['driver_number'],
                'earliest_pit_lap': int(earliest),
                'latest_pit_lap': int(latest),
                'optimal_pit_lap': int(optimal),
                'window_size': int(latest - earliest + 1)
            })
        
        return pd.DataFrame(results)
    
    def _get_dependencies(self) -> List[str]:
        return ['cliff_detection']


@register_feature(
    name='strategy_convergence',
    version='v1.0.0',
    description='Analyze strategy viability',
    dependencies=[],
    computation_cost_ms=40,
    tags=['strategy', 'pitstop', 'convergence']
)
class StrategyConvergenceFeature(BaseFeature):
    """
    Analyzes viability of different pit stop strategies.
    
    Output columns:
    - one_stop_viable: Whether 1-stop is feasible
    - two_stop_optimal: Whether 2-stop is faster
    - three_stop_risky: Whether 3-stop requires safety car
    - recommended_strategy: Recommended number of stops
    """
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(
            name='strategy_convergence',
            version='v1.0.0',
            config=config or FeatureConfig()
        )
    
    def _calculate(
        self,
        data: pd.DataFrame,
        race_laps: int = 50,
        **kwargs
    ) -> pd.DataFrame:
        """Analyze strategy convergence."""
        results = []
        
        for _, row in data.iterrows():
            tire_life = row.get('usable_tire_life', 25)
            
            # 1-stop viable if tires can last half the race
            one_stop_viable = tire_life >= (race_laps / 2)
            
            # 2-stop optimal (simplified heuristic)
            two_stop_optimal = tire_life < (race_laps / 2) and tire_life >= (race_laps / 3)
            
            # 3-stop risky (needs SC or very short tire life)
            three_stop_risky = tire_life < (race_laps / 3)
            
            if one_stop_viable:
                recommended = 1
            elif two_stop_optimal:
                recommended = 2
            else:
                recommended = 3
            
            results.append({
                'driver_number': row['driver_number'],
                'one_stop_viable': one_stop_viable,
                'two_stop_optimal': two_stop_optimal,
                'three_stop_risky': three_stop_risky,
                'recommended_strategy': recommended
            })
        
        return pd.DataFrame(results)
    
    def _get_dependencies(self) -> List[str]:
        return []
