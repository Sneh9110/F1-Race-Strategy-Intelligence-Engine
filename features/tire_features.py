"""
Tire-specific feature calculators.

Analyzes tire warmup, dropoff, performance windows, and compound comparisons.
"""

from typing import List, Dict, Any
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


@register_feature(
    name='tire_warmup_curve',
    version='v1.0.0',
    description='Model tire performance during warmup phase',
    dependencies=[],
    computation_cost_ms=50,
    tags=['tire', 'warmup', 'performance']
)
class TireWarmupCurveFeature(BaseFeature):
    """
    Models tire warmup behavior in initial laps of stint.
    
    Output columns:
    - stint_id: Stint identifier
    - tire_compound: Tire compound
    - warmup_laps: Number of laps to reach optimal temp
    - warmup_delta: Time lost during warmup (first lap - optimal lap)
    - warmup_rate: Rate of improvement (seconds/lap)
    - lap_1_time: First lap time
    - optimal_lap_time: Best lap time after warmup
    """
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(
            name='tire_warmup_curve',
            version='v1.0.0',
            config=config or FeatureConfig(thresholds={'warmup_detection_laps': 5})
        )
        self.tire_config = load_tire_config()
    
    def _calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Calculate warmup curves."""
        results = []
        warmup_laps_config = self.config.thresholds.get('warmup_detection_laps', 5)
        
        for stint_id, stint_df in data.groupby('stint_id'):
            if len(stint_df) < 3:
                continue
            
            stint_df = stint_df.sort_values('tire_age')
            lap_times = stint_df['lap_time'].values
            tire_compound = stint_df['tire_compound'].iloc[0]
            
            # Warmup period
            warmup_period = min(warmup_laps_config, len(lap_times))
            warmup_times = lap_times[:warmup_period]
            
            # Find optimal (fastest) lap after warmup
            post_warmup = lap_times[warmup_period:] if len(lap_times) > warmup_period else warmup_times
            optimal_time = post_warmup.min() if len(post_warmup) > 0 else warmup_times.min()
            
            # Warmup delta
            first_lap_time = lap_times[0]
            warmup_delta = first_lap_time - optimal_time
            
            # Warmup rate (linear fit)
            if len(warmup_times) >= 2:
                x = np.arange(len(warmup_times))
                slope = np.polyfit(x, warmup_times, 1)[0]
                warmup_rate = float(slope)
            else:
                warmup_rate = 0.0
            
            # Expected warmup laps from config
            compound_config = self.tire_config.get('compounds', {}).get(tire_compound, {})
            expected_warmup_laps = compound_config.get('warmup_laps', 2)
            
            results.append({
                'stint_id': stint_id,
                'tire_compound': tire_compound,
                'warmup_laps': int(warmup_period),
                'expected_warmup_laps': int(expected_warmup_laps),
                'warmup_delta': float(warmup_delta),
                'warmup_rate': warmup_rate,
                'lap_1_time': float(first_lap_time),
                'optimal_lap_time': float(optimal_time)
            })
        
        return pd.DataFrame(results)
    
    def _get_dependencies(self) -> List[str]:
        return []


@register_feature(
    name='tire_dropoff',
    version='v1.0.0',
    description='Detect tire performance dropoff',
    dependencies=[],
    computation_cost_ms=40,
    tags=['tire', 'dropoff', 'degradation']
)
class TireDropoffFeature(BaseFeature):
    """
    Detects tire performance dropoff (cliff).
    
    Output columns:
    - stint_id: Stint identifier
    - dropoff_detected: Whether dropoff was detected
    - dropoff_lap: Lap number where dropoff occurred
    - dropoff_magnitude: Pace loss at dropoff (seconds)
    - pace_before_dropoff: Average pace before dropoff
    - pace_after_dropoff: Average pace after dropoff
    - usable_life: Number of laps before dropoff
    """
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(
            name='tire_dropoff',
            version='v1.0.0',
            config=config or FeatureConfig(thresholds={'dropoff_threshold': 0.5})
        )
        self.tire_config = load_tire_config()
    
    def _calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Detect tire dropoff."""
        results = []
        dropoff_threshold = self.config.thresholds.get('dropoff_threshold', 0.5)
        
        for stint_id, stint_df in data.groupby('stint_id'):
            if len(stint_df) < 5:
                continue
            
            stint_df = stint_df.sort_values('tire_age')
            lap_times = stint_df['lap_time'].values
            tire_compound = stint_df['tire_compound'].iloc[0]
            
            # Baseline: first 5 laps
            baseline = lap_times[:5].mean()
            
            # Find dropoff point
            dropoff_lap = None
            for i in range(5, len(lap_times)):
                if lap_times[i] > baseline + dropoff_threshold:
                    dropoff_lap = i + 1
                    break
            
            if dropoff_lap:
                dropoff_idx = dropoff_lap - 1
                pace_before = lap_times[:dropoff_idx].mean()
                pace_after = lap_times[dropoff_idx:].mean()
                dropoff_magnitude = pace_after - pace_before
                usable_life = dropoff_lap - 1
                
                results.append({
                    'stint_id': stint_id,
                    'tire_compound': tire_compound,
                    'dropoff_detected': True,
                    'dropoff_lap': int(dropoff_lap),
                    'dropoff_magnitude': float(dropoff_magnitude),
                    'pace_before_dropoff': float(pace_before),
                    'pace_after_dropoff': float(pace_after),
                    'usable_life': int(usable_life)
                })
            else:
                results.append({
                    'stint_id': stint_id,
                    'tire_compound': tire_compound,
                    'dropoff_detected': False,
                    'dropoff_lap': None,
                    'dropoff_magnitude': None,
                    'pace_before_dropoff': float(baseline),
                    'pace_after_dropoff': None,
                    'usable_life': len(lap_times)
                })
        
        return pd.DataFrame(results)
    
    def _get_dependencies(self) -> List[str]:
        return []


@register_feature(
    name='tire_performance_window',
    version='v1.0.0',
    description='Identify optimal tire usage window',
    dependencies=['tire_warmup_curve', 'tire_dropoff'],
    computation_cost_ms=30,
    tags=['tire', 'performance', 'window']
)
class TirePerformanceWindowFeature(BaseFeature):
    """
    Identifies the optimal tire performance window.
    
    Window is between warmup completion and dropoff onset.
    
    Output columns:
    - stint_id: Stint identifier
    - optimal_start_lap: First lap of optimal window
    - optimal_end_lap: Last lap of optimal window
    - optimal_window_length: Number of laps in window
    - avg_pace_in_window: Average lap time in optimal window
    """
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(
            name='tire_performance_window',
            version='v1.0.0',
            config=config or FeatureConfig()
        )
    
    def _calculate(
        self,
        warmup_data: pd.DataFrame,
        dropoff_data: pd.DataFrame,
        stint_data: pd.DataFrame,
        **kwargs
    ) -> pd.DataFrame:
        """Identify optimal performance windows."""
        results = []
        
        # Merge warmup and dropoff data
        merged = warmup_data.merge(dropoff_data, on='stint_id', suffixes=('_warmup', '_dropoff'))
        
        for _, row in merged.iterrows():
            stint_id = row['stint_id']
            warmup_laps = row['warmup_laps']
            
            # Start of optimal window (after warmup)
            optimal_start = warmup_laps + 1
            
            # End of optimal window (before dropoff or end of stint)
            if row['dropoff_detected']:
                optimal_end = row['dropoff_lap'] - 1
            else:
                # No dropoff, use full stint
                stint_laps = stint_data[stint_data['stint_id'] == stint_id]
                optimal_end = len(stint_laps)
            
            optimal_window_length = max(0, optimal_end - optimal_start + 1)
            
            # Average pace in window
            window_laps = stint_data[
                (stint_data['stint_id'] == stint_id) &
                (stint_data['tire_age'] >= optimal_start) &
                (stint_data['tire_age'] <= optimal_end)
            ]
            avg_pace = window_laps['lap_time'].mean() if len(window_laps) > 0 else None
            
            results.append({
                'stint_id': stint_id,
                'tire_compound': row['tire_compound_warmup'],
                'optimal_start_lap': int(optimal_start),
                'optimal_end_lap': int(optimal_end),
                'optimal_window_length': int(optimal_window_length),
                'avg_pace_in_window': float(avg_pace) if avg_pace else None
            })
        
        return pd.DataFrame(results)
    
    def _get_dependencies(self) -> List[str]:
        return ['tire_warmup_curve', 'tire_dropoff']


@register_feature(
    name='compound_comparison',
    version='v1.0.0',
    description='Compare tire compound performance',
    dependencies=[],
    computation_cost_ms=40,
    tags=['tire', 'compound', 'comparison']
)
class CompoundComparisonFeature(BaseFeature):
    """
    Compares performance across tire compounds.
    
    Output columns:
    - compound_pair: Comparison pair (e.g., 'SOFT_vs_MEDIUM')
    - pace_delta: Pace difference (seconds, positive = first compound slower)
    - durability_ratio: Durability ratio (first / second)
    - performance_per_lap: Pace advantage per lap of durability
    """
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(
            name='compound_comparison',
            version='v1.0.0',
            config=config or FeatureConfig()
        )
        self.tire_config = load_tire_config()
    
    def _calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Compare tire compounds."""
        results = []
        compounds = data['tire_compound'].unique()
        
        # Compare all pairs
        for i, compound1 in enumerate(compounds):
            for compound2 in compounds[i+1:]:
                data1 = data[data['tire_compound'] == compound1]
                data2 = data[data['tire_compound'] == compound2]
                
                if len(data1) == 0 or len(data2) == 0:
                    continue
                
                # Average pace
                avg_pace1 = data1['lap_time'].mean()
                avg_pace2 = data2['lap_time'].mean()
                pace_delta = avg_pace1 - avg_pace2
                
                # Durability
                durability1 = data1['usable_life'].mean() if 'usable_life' in data1.columns else 20
                durability2 = data2['usable_life'].mean() if 'usable_life' in data2.columns else 20
                durability_ratio = durability1 / durability2 if durability2 > 0 else 1.0
                
                # Performance per lap
                performance_per_lap = pace_delta / durability_ratio if durability_ratio > 0 else 0.0
                
                results.append({
                    'compound_pair': f"{compound1}_vs_{compound2}",
                    'compound_1': compound1,
                    'compound_2': compound2,
                    'pace_delta': float(pace_delta),
                    'compound_1_durability': float(durability1),
                    'compound_2_durability': float(durability2),
                    'durability_ratio': float(durability_ratio),
                    'performance_per_lap': float(performance_per_lap)
                })
        
        return pd.DataFrame(results)
    
    def _get_dependencies(self) -> List[str]:
        return []
