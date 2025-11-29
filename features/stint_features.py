"""
Stint-level feature calculators.

Aggregates lap-level timing data into stint-level summaries and
analyzes pace evolution within stints.
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np

from features.base import BaseFeature, FeatureConfig
from features.registry import register_feature
from data_pipeline.schemas.timing_schema import LapData
from app.utils.validators import detect_outliers

@register_feature(
    name='stint_summary',
    version='v1.0.0',
    description='Aggregate lap-level data into stint-level summaries',
    dependencies=[],
    computation_cost_ms=50,
    tags=['timing', 'stint', 'aggregation']
)
class StintSummaryFeature(BaseFeature):
    """
    Aggregates lap-level timing data into stint summaries.
    
    A stint is defined as a continuous set of laps on the same tire compound.
    Stint changes are detected via tire_age resets or pit stop indicators.
    
    Output columns:
    - stint_number: Sequential stint number (1, 2, 3, ...)
    - tire_compound: Tire compound used ('SOFT', 'MEDIUM', 'HARD')
    - start_lap: First lap of stint
    - end_lap: Last lap of stint
    - stint_length: Number of laps in stint
    - avg_lap_time: Mean lap time in seconds
    - median_lap_time: Median lap time in seconds
    - std_lap_time: Standard deviation of lap times
    - fastest_lap: Fastest lap time in stint
    - slowest_lap: Slowest lap time in stint
    - total_time: Sum of all lap times
    - avg_tire_age: Mean tire age across stint
    - pit_stop_duration: Duration of pit stop ending stint (if applicable)
    """
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(
            name='stint_summary',
            version='v1.0.0',
            config=config or FeatureConfig(
                min_data_points=1,  # Allow single-lap stints
                outlier_removal=True,
                outlier_threshold=3.0
            )
        )
    
    def _calculate(self, data: List[LapData], **kwargs) -> pd.DataFrame:
        """
        Calculate stint summaries from lap data.
        
        Args:
            data: List of LapData objects
            
        Returns:
            DataFrame with stint summaries
        """
        if not data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        laps_df = pd.DataFrame([lap.dict() for lap in data])
        laps_df = laps_df.sort_values('lap_number')
        
        # Detect stint changes (tire_age resets or explicit pit stops)
        laps_df['stint_change'] = (
            (laps_df['tire_age'].diff() < 0) |  # Tire age reset
            (laps_df['pit_stop_duration'] > 0)   # Pit stop occurred
        )
        laps_df['stint_number'] = laps_df['stint_change'].cumsum() + 1
        
        # Group by stint
        stint_summaries = []
        
        for stint_num, stint_df in laps_df.groupby('stint_number'):
            # Remove outliers if enabled
            lap_times = stint_df['lap_time'].copy()
            if self.config.outlier_removal and len(lap_times) >= 3:
                outlier_mask = detect_outliers(
                    lap_times.values,
                    threshold=self.config.outlier_threshold
                )
                lap_times_clean = lap_times[~outlier_mask]
            else:
                lap_times_clean = lap_times
            
            # Handle empty after outlier removal
            if len(lap_times_clean) == 0:
                lap_times_clean = lap_times
            
            # Calculate summary statistics
            summary = {
                'stint_number': int(stint_num),
                'tire_compound': stint_df['tire_compound'].iloc[0],
                'start_lap': int(stint_df['lap_number'].min()),
                'end_lap': int(stint_df['lap_number'].max()),
                'stint_length': len(stint_df),
                'avg_lap_time': float(lap_times_clean.mean()),
                'median_lap_time': float(lap_times_clean.median()),
                'std_lap_time': float(lap_times_clean.std()) if len(lap_times_clean) > 1 else 0.0,
                'fastest_lap': float(lap_times_clean.min()),
                'slowest_lap': float(lap_times_clean.max()),
                'total_time': float(lap_times_clean.sum()),
                'avg_tire_age': float(stint_df['tire_age'].mean()),
                'pit_stop_duration': float(stint_df['pit_stop_duration'].max())  # Last lap's pit duration
            }
            
            stint_summaries.append(summary)
        
        return pd.DataFrame(stint_summaries)
    
    def _get_dependencies(self) -> List[str]:
        """No dependencies for this feature."""
        return []


@register_feature(
    name='stint_pace_evolution',
    version='v1.0.0',
    description='Analyze lap-by-lap pace evolution within stints',
    dependencies=[],
    computation_cost_ms=40,
    tags=['timing', 'stint', 'pace']
)
class StintPaceEvolutionFeature(BaseFeature):
    """
    Analyzes pace evolution within each stint.
    
    Calculates pace deltas relative to stint start and previous lap,
    plus cumulative degradation over the stint.
    
    Output columns:
    - stint_number: Stint identifier
    - lap_number: Lap number
    - lap_time: Lap time in seconds
    - tire_age: Tire age at this lap
    - pace_delta_from_start: Difference from first lap of stint (seconds)
    - pace_delta_from_previous: Difference from previous lap (seconds)
    - cumulative_degradation: Sum of all pace deltas from start (seconds)
    - pace_trend: Rolling pace trend (positive = getting slower)
    """
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(
            name='stint_pace_evolution',
            version='v1.0.0',
            config=config or FeatureConfig(min_data_points=1)
        )
    
    def _calculate(self, data: List[LapData], **kwargs) -> pd.DataFrame:
        """
        Calculate pace evolution metrics.
        
        Args:
            data: List of LapData objects
            
        Returns:
            DataFrame with pace evolution metrics
        """
        if not data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        laps_df = pd.DataFrame([lap.dict() for lap in data])
        laps_df = laps_df.sort_values('lap_number')
        
        # Detect stints
        laps_df['stint_change'] = (
            (laps_df['tire_age'].diff() < 0) |
            (laps_df['pit_stop_duration'] > 0)
        )
        laps_df['stint_number'] = laps_df['stint_change'].cumsum() + 1
        
        # Calculate pace deltas per stint
        results = []
        
        for stint_num, stint_df in laps_df.groupby('stint_number'):
            stint_df = stint_df.copy()
            
            # Pace delta from stint start
            first_lap_time = stint_df['lap_time'].iloc[0]
            stint_df['pace_delta_from_start'] = stint_df['lap_time'] - first_lap_time
            
            # Pace delta from previous lap
            stint_df['pace_delta_from_previous'] = stint_df['lap_time'].diff().fillna(0.0)
            
            # Cumulative degradation
            stint_df['cumulative_degradation'] = stint_df['pace_delta_from_start'].cumsum()
            
            # Pace trend (3-lap rolling slope)
            if len(stint_df) >= 3:
                pace_trend = []
                lap_times = stint_df['lap_time'].values
                for i in range(len(lap_times)):
                    if i < 2:
                        pace_trend.append(0.0)
                    else:
                        window = lap_times[i-2:i+1]
                        x = np.arange(len(window))
                        slope = np.polyfit(x, window, 1)[0]
                        pace_trend.append(float(slope))
                stint_df['pace_trend'] = pace_trend
            else:
                stint_df['pace_trend'] = 0.0
            
            # Select output columns
            output_df = stint_df[[
                'stint_number', 'lap_number', 'lap_time', 'tire_age',
                'pace_delta_from_start', 'pace_delta_from_previous',
                'cumulative_degradation', 'pace_trend'
            ]]
            
            results.append(output_df)
        
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _get_dependencies(self) -> List[str]:
        """No dependencies for this feature."""
        return []
