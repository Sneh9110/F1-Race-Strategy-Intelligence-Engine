"""
Telemetry-derived feature calculators.

Analyzes driver style, fuel effects, tire temperatures, and energy management.
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from features.base import BaseFeature, FeatureConfig
from features.registry import register_feature
from data_pipeline.schemas.telemetry_schema import TelemetryPoint
from config.settings import settings


def load_track_config(track_name: str) -> Dict[str, Any]:
    """Load track-specific configuration."""
    config_path = Path(settings.CONFIG_DIR) / "tracks.yaml"
    with open(config_path, 'r') as f:
        tracks = yaml.safe_load(f)
    return tracks.get(track_name, {})


def load_tire_config() -> Dict[str, Any]:
    """Load tire compound configuration."""
    config_path = Path(settings.CONFIG_DIR) / "tire_compounds.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@register_feature(
    name='driver_style',
    version='v1.0.0',
    description='Characterize driver input style',
    dependencies=[],
    computation_cost_ms=60,
    tags=['telemetry', 'driver', 'style']
)
class DriverStyleFeature(BaseFeature):
    """
    Characterizes driving style from telemetry.
    
    Output columns:
    - driver_number: Driver number
    - avg_throttle_application: Mean throttle % (0-100)
    - avg_brake_application: Mean brake % (0-100)
    - aggression_score: Aggression metric (0-100)
    - smoothness_score: Smoothness metric (0-1, higher=smoother)
    - tire_management_score: Tire management metric (0-1)
    """
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(
            name='driver_style',
            version='v1.0.0',
            config=config or FeatureConfig(thresholds={'optimal_tire_temp': 90.0})
        )
        self.tire_config = load_tire_config()
    
    def _calculate(
        self,
        telemetry_data: List[TelemetryPoint],
        **kwargs
    ) -> pd.DataFrame:
        """Calculate driver style metrics."""
        if not telemetry_data:
            return pd.DataFrame()
        
        df = pd.DataFrame([t.dict() for t in telemetry_data])
        results = []
        
        optimal_tire_temp = self.config.thresholds.get('optimal_tire_temp', 90.0)
        
        for driver_num in df['driver_number'].unique():
            driver_df = df[df['driver_number'] == driver_num]
            
            # Throttle and brake application
            avg_throttle = driver_df['throttle_percent'].mean()
            avg_brake = driver_df['brake_percent'].mean()
            
            # Aggression score (combination of max inputs)
            max_throttle = driver_df['throttle_percent'].max()
            max_brake = driver_df['brake_percent'].max()
            aggression = (max_throttle + max_brake) / 2.0
            
            # Smoothness (inverse of input variance)
            throttle_changes = driver_df['throttle_percent'].diff().abs()
            smoothness = 1.0 / (throttle_changes.std() + 1.0)
            
            # Tire management (deviation from optimal temp)
            if 'tire_temp_front_left' in driver_df.columns:
                avg_tire_temp = driver_df[[
                    'tire_temp_front_left', 'tire_temp_front_right',
                    'tire_temp_rear_left', 'tire_temp_rear_right'
                ]].mean().mean()
                
                temp_deviation = abs(avg_tire_temp - optimal_tire_temp)
                tire_management = max(0.0, 1.0 - (temp_deviation / optimal_tire_temp))
            else:
                tire_management = 0.5
            
            results.append({
                'driver_number': driver_num,
                'avg_throttle_application': float(avg_throttle),
                'avg_brake_application': float(avg_brake),
                'aggression_score': float(aggression),
                'smoothness_score': float(smoothness),
                'tire_management_score': float(tire_management)
            })
        
        return pd.DataFrame(results)
    
    def _get_dependencies(self) -> List[str]:
        return []


@register_feature(
    name='fuel_effect',
    version='v1.0.0',
    description='Model fuel load impact on pace',
    dependencies=[],
    computation_cost_ms=40,
    tags=['telemetry', 'fuel', 'pace']
)
class FuelEffectFeature(BaseFeature):
    """
    Models fuel load effect on pace.
    
    Formula:
        pace_improvement_per_lap = fuel_per_lap * fuel_effect_per_kg
        cumulative_fuel_effect = sum(pace_improvements)
    
    Output columns:
    - driver_number: Driver number
    - fuel_per_lap: Fuel consumption per lap (kg)
    - fuel_effect_per_kg: Pace effect per kg (seconds)
    - pace_improvement_per_lap: Pace gain per lap (seconds)
    - cumulative_fuel_effect: Total fuel effect (seconds)
    - laps_analyzed: Number of laps in analysis
    """
    
    def __init__(self, config: FeatureConfig = None, track_name: str = "generic"):
        super().__init__(
            name='fuel_effect',
            version='v1.0.0',
            config=config or FeatureConfig()
        )
        self.track_config = load_track_config(track_name)
    
    def _calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Calculate fuel effects."""
        fuel_effect_per_kg = self.track_config.get('fuel_effect_per_lap_seconds', 0.035)
        
        results = []
        
        for driver_num in data['driver_number'].unique():
            driver_data = data[data['driver_number'] == driver_num].copy()
            driver_data = driver_data.sort_values('lap_number')
            
            if 'fuel_kg' in driver_data.columns:
                # Calculate fuel consumption per lap
                driver_data['fuel_consumed'] = driver_data['fuel_kg'].diff().abs()
                fuel_per_lap = driver_data['fuel_consumed'].mean()
                
                # Pace improvement
                pace_improvement_per_lap = fuel_per_lap * fuel_effect_per_kg
                
                # Cumulative effect
                cumulative_effect = pace_improvement_per_lap * len(driver_data)
            else:
                # Default values
                fuel_per_lap = 1.5  # kg
                pace_improvement_per_lap = fuel_per_lap * fuel_effect_per_kg
                cumulative_effect = pace_improvement_per_lap * len(driver_data)
            
            results.append({
                'driver_number': driver_num,
                'fuel_per_lap': float(fuel_per_lap),
                'fuel_effect_per_kg': float(fuel_effect_per_kg),
                'pace_improvement_per_lap': float(pace_improvement_per_lap),
                'cumulative_fuel_effect': float(cumulative_effect),
                'laps_analyzed': len(driver_data)
            })
        
        return pd.DataFrame(results)
    
    def _get_dependencies(self) -> List[str]:
        return []


@register_feature(
    name='tire_temperature',
    version='v1.0.0',
    description='Analyze tire temperature patterns',
    dependencies=[],
    computation_cost_ms=50,
    tags=['telemetry', 'tire', 'temperature']
)
class TireTemperatureFeature(BaseFeature):
    """
    Analyzes tire temperature patterns.
    
    Output columns:
    - driver_number: Driver number
    - avg_tire_temp: Average tire temperature across all corners
    - temp_imbalance: Std deviation of tire temps
    - overheating_risk: Risk of overheating (0-1)
    - underheating_risk: Risk of underheating (0-1)
    - optimal_temp_deviation: Deviation from optimal (°C)
    """
    
    def __init__(self, config: FeatureConfig = None, tire_compound: str = "MEDIUM"):
        super().__init__(
            name='tire_temperature',
            version='v1.0.0',
            config=config or FeatureConfig()
        )
        tire_config = load_tire_config()
        compound_config = tire_config.get('compounds', {}).get(tire_compound, {})
        self.optimal_range = compound_config.get('optimal_temp_range_celsius', [85, 95])
    
    def _calculate(
        self,
        telemetry_data: List[TelemetryPoint],
        **kwargs
    ) -> pd.DataFrame:
        """Calculate tire temperature metrics."""
        if not telemetry_data:
            return pd.DataFrame()
        
        df = pd.DataFrame([t.dict() for t in telemetry_data])
        results = []
        
        temp_cols = [
            'tire_temp_front_left', 'tire_temp_front_right',
            'tire_temp_rear_left', 'tire_temp_rear_right'
        ]
        
        # Check if temp columns exist
        if not all(col in df.columns for col in temp_cols):
            return pd.DataFrame()
        
        for driver_num in df['driver_number'].unique():
            driver_df = df[df['driver_number'] == driver_num]
            
            # Average temps
            tire_temps = driver_df[temp_cols].values
            avg_temp = np.mean(tire_temps)
            temp_imbalance = np.std(tire_temps)
            
            # Risk scores
            optimal_min, optimal_max = self.optimal_range
            overheating_risk = 1.0 if avg_temp > optimal_max else 0.0
            underheating_risk = 1.0 if avg_temp < optimal_min else 0.0
            
            # Deviation from optimal
            optimal_center = (optimal_min + optimal_max) / 2.0
            deviation = abs(avg_temp - optimal_center)
            
            results.append({
                'driver_number': driver_num,
                'avg_tire_temp': float(avg_temp),
                'temp_imbalance': float(temp_imbalance),
                'overheating_risk': float(overheating_risk),
                'underheating_risk': float(underheating_risk),
                'optimal_temp_deviation': float(deviation)
            })
        
        return pd.DataFrame(results)
    
    def _get_dependencies(self) -> List[str]:
        return []


@register_feature(
    name='energy_management',
    version='v1.0.0',
    description='Track energy deployment and efficiency',
    dependencies=[],
    computation_cost_ms=40,
    tags=['telemetry', 'energy', 'drs']
)
class EnergyManagementFeature(BaseFeature):
    """
    Tracks energy deployment efficiency.
    
    Output columns:
    - driver_number: Driver number
    - drs_usage_rate: DRS activations / opportunities
    - avg_speed_in_drs_zones: Average speed in DRS zones
    - energy_efficiency: Distance / fuel consumed
    - drs_opportunities: Total DRS opportunities
    - drs_activations: Total DRS activations
    """
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(
            name='energy_management',
            version='v1.0.0',
            config=config or FeatureConfig()
        )
    
    def _calculate(
        self,
        telemetry_data: List[TelemetryPoint],
        **kwargs
    ) -> pd.DataFrame:
        """Calculate energy management metrics."""
        if not telemetry_data:
            return pd.DataFrame()
        
        df = pd.DataFrame([t.dict() for t in telemetry_data])
        results = []
        
        for driver_num in df['driver_number'].unique():
            driver_df = df[df['driver_number'] == driver_num]
            
            # DRS usage
            if 'drs_active' in driver_df.columns:
                drs_activations = driver_df['drs_active'].sum()
                drs_opportunities = len(driver_df)
                drs_usage_rate = drs_activations / drs_opportunities if drs_opportunities > 0 else 0.0
                
                # Speed in DRS zones
                drs_speeds = driver_df[driver_df['drs_active']]['speed']
                avg_drs_speed = drs_speeds.mean() if len(drs_speeds) > 0 else 0.0
            else:
                drs_activations = 0
                drs_opportunities = 0
                drs_usage_rate = 0.0
                avg_drs_speed = 0.0
            
            # Energy efficiency
            if 'distance' in driver_df.columns and 'fuel_kg' in driver_df.columns:
                total_distance = driver_df['distance'].max() - driver_df['distance'].min()
                fuel_consumed = driver_df['fuel_kg'].iloc[0] - driver_df['fuel_kg'].iloc[-1]
                energy_efficiency = total_distance / fuel_consumed if fuel_consumed > 0 else 0.0
            else:
                energy_efficiency = 0.0
            
            results.append({
                'driver_number': driver_num,
                'drs_usage_rate': float(drs_usage_rate),
                'avg_speed_in_drs_zones': float(avg_drs_speed),
                'energy_efficiency': float(energy_efficiency),
                'drs_opportunities': int(drs_opportunities),
                'drs_activations': int(drs_activations)
            })
        
        return pd.DataFrame(results)
    
    def _get_dependencies(self) -> List[str]:
        return []
