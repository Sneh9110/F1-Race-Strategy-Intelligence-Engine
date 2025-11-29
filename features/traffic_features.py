"""
Traffic and clean air feature calculators.

Analyzes dirty air penalties, traffic density, and lapping impacts.
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from features.base import BaseFeature, FeatureConfig
from features.registry import register_feature
from data_pipeline.schemas.timing_schema import TimingPoint
from config.settings import settings


def load_track_config(track_name: str) -> Dict[str, Any]:
    """Load track-specific configuration."""
    config_path = Path(settings.CONFIG_DIR) / "tracks.yaml"
    with open(config_path, 'r') as f:
        tracks = yaml.safe_load(f)
    return tracks.get(track_name, {})


@register_feature(
    name='clean_air_penalty',
    version='v1.0.0',
    description='Calculate pace loss from dirty air',
    dependencies=[],
    computation_cost_ms=50,
    tags=['traffic', 'dirty_air', 'pace']
)
class CleanAirPenaltyFeature(BaseFeature):
    """
    Calculates pace loss when following another car.
    
    Formula:
        penalty_by_gap = base_penalty * exp(-gap / decay_distance)
        Typical: 0.4s when gap < 1.0s, exponential decay
    
    Output columns:
    - lap_number: Lap number
    - driver_number: Driver number
    - gap_to_ahead: Gap to car ahead (seconds)
    - dirty_air_penalty: Estimated time loss (seconds)
    - in_dirty_air: Boolean flag (gap < threshold)
    - clean_air_laps_count: Count of clean air laps
    """
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(
            name='clean_air_penalty',
            version='v1.0.0',
            config=config or FeatureConfig(thresholds={
                'gap_threshold': 1.0,
                'base_penalty': 0.4,
                'decay_distance': 1.0
            })
        )
    
    def _calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Calculate dirty air penalties."""
        results = []
        gap_threshold = self.config.thresholds.get('gap_threshold', 1.0)
        base_penalty = self.config.thresholds.get('base_penalty', 0.4)
        decay = self.config.thresholds.get('decay_distance', 1.0)
        
        for _, row in data.iterrows():
            gap = row.get('gap_to_ahead', 999.0)
            
            # Calculate penalty with exponential decay
            if gap < gap_threshold * 3:
                penalty = base_penalty * np.exp(-gap / decay)
            else:
                penalty = 0.0
            
            in_dirty_air = gap < gap_threshold
            
            results.append({
                'lap_number': row['lap_number'],
                'driver_number': row['driver_number'],
                'gap_to_ahead': float(gap),
                'dirty_air_penalty': float(penalty),
                'in_dirty_air': bool(in_dirty_air)
            })
        
        # Count clean air laps per driver
        results_df = pd.DataFrame(results)
        clean_air_counts = results_df.groupby('driver_number')['in_dirty_air'].apply(
            lambda x: (~x).sum()
        ).to_dict()
        
        results_df['clean_air_laps_count'] = results_df['driver_number'].map(clean_air_counts)
        
        return results_df
    
    def _get_dependencies(self) -> List[str]:
        return []


@register_feature(
    name='traffic_density',
    version='v1.0.0',
    description='Quantify traffic congestion',
    dependencies=[],
    computation_cost_ms=40,
    tags=['traffic', 'density', 'congestion']
)
class TrafficDensityFeature(BaseFeature):
    """
    Quantifies traffic density around each car.
    
    Output columns:
    - driver_number: Driver number
    - cars_within_1s: Number of cars within 1s
    - cars_within_3s: Number of cars within 3s
    - traffic_density_score: Weighted density score
    - overtaking_difficulty_multiplier: Track-specific multiplier
    """
    
    def __init__(self, config: FeatureConfig = None, track_name: str = "generic"):
        super().__init__(
            name='traffic_density',
            version='v1.0.0',
            config=config or FeatureConfig()
        )
        self.track_config = load_track_config(track_name)
    
    def _calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Calculate traffic density."""
        results = []
        overtaking_difficulty = self.track_config.get('overtaking_difficulty', 5)
        
        for _, row in data.iterrows():
            driver_num = row['driver_number']
            
            # Find other cars
            other_cars = data[data['driver_number'] != driver_num].copy()
            
            # Calculate gaps
            if 'gap_to_leader' in data.columns:
                driver_gap = row['gap_to_leader']
                other_cars['gap_diff'] = abs(other_cars['gap_to_leader'] - driver_gap)
                
                cars_within_1s = (other_cars['gap_diff'] < 1.0).sum()
                cars_within_3s = (other_cars['gap_diff'] < 3.0).sum()
            else:
                cars_within_1s = 0
                cars_within_3s = 0
            
            # Density score
            density_score = cars_within_1s * 1.0 + cars_within_3s * 0.3
            
            # Apply track difficulty
            adjusted_density = density_score * (overtaking_difficulty / 5.0)
            
            results.append({
                'driver_number': driver_num,
                'lap_number': row.get('lap_number', 0),
                'cars_within_1s': int(cars_within_1s),
                'cars_within_3s': int(cars_within_3s),
                'traffic_density_score': float(density_score),
                'overtaking_difficulty_multiplier': float(overtaking_difficulty),
                'adjusted_density_score': float(adjusted_density)
            })
        
        return pd.DataFrame(results)
    
    def _get_dependencies(self) -> List[str]:
        return []


@register_feature(
    name='lapping_impact',
    version='v1.0.0',
    description='Measure impact of lapping backmarkers',
    dependencies=[],
    computation_cost_ms=40,
    tags=['traffic', 'lapping', 'backmarkers']
)
class LappingImpactFeature(BaseFeature):
    """
    Measures time lost lapping backmarkers.
    
    Output columns:
    - driver_number: Driver number
    - laps_with_blue_flags: Count of laps with blue flags
    - laps_lost_to_backmarkers: Estimated time lost (seconds)
    - time_lost_lapping: Total time lost (seconds)
    - lapping_frequency: Lapping events per lap
    """
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(
            name='lapping_impact',
            version='v1.0.0',
            config=config or FeatureConfig(thresholds={'time_loss_per_lap': 0.5})
        )
    
    def _calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Calculate lapping impact."""
        results = []
        time_loss_per_lap = self.config.thresholds.get('time_loss_per_lap', 0.5)
        
        for driver_num in data['driver_number'].unique():
            driver_data = data[data['driver_number'] == driver_num]
            
            # Count blue flag laps (simplified: assume from data)
            blue_flag_laps = driver_data.get('blue_flag', pd.Series([False] * len(driver_data))).sum()
            
            # Estimate time lost
            laps_lost = blue_flag_laps * time_loss_per_lap
            
            # Lapping frequency
            total_laps = len(driver_data)
            lapping_frequency = blue_flag_laps / total_laps if total_laps > 0 else 0.0
            
            results.append({
                'driver_number': driver_num,
                'laps_with_blue_flags': int(blue_flag_laps),
                'laps_lost_to_backmarkers': float(laps_lost),
                'time_lost_lapping': float(laps_lost),
                'lapping_frequency': float(lapping_frequency),
                'total_laps': int(total_laps)
            })
        
        return pd.DataFrame(results)
    
    def _get_dependencies(self) -> List[str]:
        return []


@register_feature(
    name='position_battle',
    version='v1.0.0',
    description='Identify close racing battles',
    dependencies=[],
    computation_cost_ms=40,
    tags=['traffic', 'battle', 'racing']
)
class PositionBattleFeature(BaseFeature):
    """
    Identifies close position battles.
    
    Output columns:
    - driver_number: Driver number
    - battle_intensity: Battle intensity score (0-1)
    - position_changes_per_lap: Rate of position swaps
    - defensive_driving_penalty: Time penalty estimate (seconds)
    - avg_gap_to_ahead: Average gap to car ahead
    """
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(
            name='position_battle',
            version='v1.0.0',
            config=config or FeatureConfig(thresholds={
                'battle_gap_threshold': 2.0,
                'defensive_penalty': 0.2
            })
        )
    
    def _calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Calculate battle intensity."""
        results = []
        battle_threshold = self.config.thresholds.get('battle_gap_threshold', 2.0)
        defensive_penalty = self.config.thresholds.get('defensive_penalty', 0.2)
        
        for driver_num in data['driver_number'].unique():
            driver_data = data[data['driver_number'] == driver_num].copy()
            driver_data = driver_data.sort_values('lap_number')
            
            if 'gap_to_ahead' in driver_data.columns:
                avg_gap = driver_data['gap_to_ahead'].mean()
                
                # Battle intensity (inverse of gap when close)
                if avg_gap < battle_threshold:
                    battle_intensity = 1.0 / max(avg_gap, 0.1)
                else:
                    battle_intensity = 0.0
                
                # Count position changes
                if 'position' in driver_data.columns:
                    position_changes = (driver_data['position'].diff() != 0).sum()
                    changes_per_lap = position_changes / len(driver_data)
                else:
                    changes_per_lap = 0.0
                
                # Defensive penalty if in battle
                in_battle = avg_gap < battle_threshold
                penalty = defensive_penalty if in_battle else 0.0
            else:
                avg_gap = 999.0
                battle_intensity = 0.0
                changes_per_lap = 0.0
                penalty = 0.0
            
            results.append({
                'driver_number': driver_num,
                'battle_intensity': float(battle_intensity),
                'position_changes_per_lap': float(changes_per_lap),
                'defensive_driving_penalty': float(penalty),
                'avg_gap_to_ahead': float(avg_gap),
                'in_battle': bool(avg_gap < battle_threshold)
            })
        
        return pd.DataFrame(results)
    
    def _get_dependencies(self) -> List[str]:
        return []
