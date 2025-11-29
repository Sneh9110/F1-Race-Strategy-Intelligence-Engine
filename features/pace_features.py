"""
Pace-related feature calculators.

Calculates pace deltas, rolling pace metrics, and sector-level pace analysis.
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np

from features.base import BaseFeature, FeatureConfig
from features.registry import register_feature
from data_pipeline.schemas.timing_schema import SessionTiming, TimingPoint


@register_feature(
    name='lap_pace_delta',
    version='v1.0.0',
    description='Calculate pace deltas relative to various benchmarks',
    dependencies=[],
    computation_cost_ms=60,
    tags=['timing', 'pace', 'delta']
)
class LapPaceDeltaFeature(BaseFeature):
    """
    Calculates lap time deltas relative to various benchmarks.
    
    Output columns:
    - lap_number: Lap number
    - driver_number: Driver number
    - lap_time: Actual lap time
    - delta_to_leader: Delta to leader's lap time
    - delta_to_teammate: Delta to teammate's lap time
    - delta_to_previous_lap: Delta to own previous lap
    - delta_to_session_average: Delta to session average
    - percentile_rank: Percentile rank in session (0-100)
    """
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(
            name='lap_pace_delta',
            version='v1.0.0',
            config=config or FeatureConfig()
        )
    
    def _calculate(self, data: SessionTiming, **kwargs) -> pd.DataFrame:
        """Calculate pace deltas."""
        # Convert session timing to DataFrame
        all_laps = []
        for driver_timing in data.timing_data:
            for lap in driver_timing.laps:
                all_laps.append({
                    'lap_number': lap.lap_number,
                    'driver_number': driver_timing.driver_number,
                    'team': driver_timing.team,
                    'lap_time': lap.lap_time
                })
        
        if not all_laps:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_laps)
        results = []
        
        for lap_num in df['lap_number'].unique():
            lap_data = df[df['lap_number'] == lap_num].copy()
            
            # Find leader's time
            leader_time = lap_data['lap_time'].min()
            
            # Session average for this lap
            session_avg = lap_data['lap_time'].mean()
            
            for _, row in lap_data.iterrows():
                driver_num = row['driver_number']
                lap_time = row['lap_time']
                
                # Delta to leader
                delta_to_leader = lap_time - leader_time
                
                # Delta to teammate
                teammates = lap_data[
                    (lap_data['team'] == row['team']) & 
                    (lap_data['driver_number'] != driver_num)
                ]
                delta_to_teammate = (
                    lap_time - teammates['lap_time'].mean()
                    if len(teammates) > 0 else None
                )
                
                # Delta to previous lap
                prev_lap = df[
                    (df['driver_number'] == driver_num) & 
                    (df['lap_number'] == lap_num - 1)
                ]
                delta_to_previous = (
                    lap_time - prev_lap['lap_time'].values[0]
                    if len(prev_lap) > 0 else None
                )
                
                # Delta to session average
                delta_to_session_avg = lap_time - session_avg
                
                # Percentile rank
                percentile_rank = (lap_data['lap_time'] <= lap_time).sum() / len(lap_data) * 100
                
                results.append({
                    'lap_number': lap_num,
                    'driver_number': driver_num,
                    'lap_time': lap_time,
                    'delta_to_leader': delta_to_leader,
                    'delta_to_teammate': delta_to_teammate,
                    'delta_to_previous_lap': delta_to_previous,
                    'delta_to_session_average': delta_to_session_avg,
                    'percentile_rank': percentile_rank
                })
        
        return pd.DataFrame(results)
    
    def _get_dependencies(self) -> List[str]:
        return []


@register_feature(
    name='rolling_pace',
    version='v1.0.0',
    description='Calculate rolling pace statistics',
    dependencies=[],
    computation_cost_ms=40,
    tags=['timing', 'pace', 'rolling']
)
class RollingPaceFeature(BaseFeature):
    """
    Calculates rolling pace statistics.
    
    Output columns:
    - driver_number: Driver number
    - lap_number: Lap number
    - rolling_avg_3lap: 3-lap rolling average
    - rolling_avg_5lap: 5-lap rolling average
    - rolling_std_3lap: 3-lap rolling std deviation
    - pace_trend: Linear regression slope over last 5 laps (s/lap)
    """
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(
            name='rolling_pace',
            version='v1.0.0',
            config=config or FeatureConfig(window_size=5)
        )
    
    def _calculate(self, data: SessionTiming, **kwargs) -> pd.DataFrame:
        """Calculate rolling pace metrics."""
        results = []
        
        for driver_timing in data.timing_data:
            laps_df = pd.DataFrame([lap.dict() for lap in driver_timing.laps])
            if laps_df.empty:
                continue
            
            laps_df = laps_df.sort_values('lap_number')
            
            # Rolling averages
            laps_df['rolling_avg_3lap'] = laps_df['lap_time'].rolling(window=3, min_periods=1).mean()
            laps_df['rolling_avg_5lap'] = laps_df['lap_time'].rolling(window=5, min_periods=1).mean()
            laps_df['rolling_std_3lap'] = laps_df['lap_time'].rolling(window=3, min_periods=2).std()
            
            # Pace trend (linear regression slope)
            pace_trends = []
            lap_times = laps_df['lap_time'].values
            
            for i in range(len(lap_times)):
                if i < 4:  # Need at least 5 points
                    pace_trends.append(None)
                else:
                    window = lap_times[i-4:i+1]
                    x = np.arange(len(window))
                    slope = np.polyfit(x, window, 1)[0]
                    pace_trends.append(float(slope))
            
            laps_df['pace_trend'] = pace_trends
            laps_df['driver_number'] = driver_timing.driver_number
            
            results.append(laps_df[[
                'driver_number', 'lap_number', 'rolling_avg_3lap',
                'rolling_avg_5lap', 'rolling_std_3lap', 'pace_trend'
            ]])
        
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _get_dependencies(self) -> List[str]:
        return []


@register_feature(
    name='sector_pace',
    version='v1.0.0',
    description='Analyze sector-level pace performance',
    dependencies=[],
    computation_cost_ms=50,
    tags=['timing', 'pace', 'sectors']
)
class SectorPaceFeature(BaseFeature):
    """
    Analyzes sector-level pace performance.
    
    Output columns:
    - lap_number: Lap number
    - driver_number: Driver number
    - sector_1_time: Sector 1 time
    - sector_2_time: Sector 2 time
    - sector_3_time: Sector 3 time
    - sector_1_delta_to_fastest: Delta to fastest S1
    - sector_2_delta_to_fastest: Delta to fastest S2
    - sector_3_delta_to_fastest: Delta to fastest S3
    - sector_consistency: Std dev of sector deltas
    - strongest_sector: Best performing sector (1, 2, or 3)
    """
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(
            name='sector_pace',
            version='v1.0.0',
            config=config or FeatureConfig()
        )
    
    def _calculate(self, data: SessionTiming, **kwargs) -> pd.DataFrame:
        """Calculate sector pace metrics."""
        # Collect all sector times
        all_sectors = []
        for driver_timing in data.timing_data:
            for lap in driver_timing.laps:
                if hasattr(lap, 'sector_1_time') and lap.sector_1_time:
                    all_sectors.append({
                        'lap_number': lap.lap_number,
                        'driver_number': driver_timing.driver_number,
                        'sector_1_time': lap.sector_1_time,
                        'sector_2_time': lap.sector_2_time,
                        'sector_3_time': lap.sector_3_time
                    })
        
        if not all_sectors:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_sectors)
        results = []
        
        for lap_num in df['lap_number'].unique():
            lap_data = df[df['lap_number'] == lap_num]
            
            # Find fastest sectors
            fastest_s1 = lap_data['sector_1_time'].min()
            fastest_s2 = lap_data['sector_2_time'].min()
            fastest_s3 = lap_data['sector_3_time'].min()
            
            for _, row in lap_data.iterrows():
                s1_delta = row['sector_1_time'] - fastest_s1
                s2_delta = row['sector_2_time'] - fastest_s2
                s3_delta = row['sector_3_time'] - fastest_s3
                
                # Sector consistency
                sector_deltas = [s1_delta, s2_delta, s3_delta]
                sector_consistency = float(np.std(sector_deltas))
                
                # Strongest sector
                strongest = np.argmin(sector_deltas) + 1
                
                results.append({
                    'lap_number': lap_num,
                    'driver_number': row['driver_number'],
                    'sector_1_time': row['sector_1_time'],
                    'sector_2_time': row['sector_2_time'],
                    'sector_3_time': row['sector_3_time'],
                    'sector_1_delta_to_fastest': s1_delta,
                    'sector_2_delta_to_fastest': s2_delta,
                    'sector_3_delta_to_fastest': s3_delta,
                    'sector_consistency': sector_consistency,
                    'strongest_sector': int(strongest)
                })
        
        return pd.DataFrame(results)
    
    def _get_dependencies(self) -> List[str]:
        return []
