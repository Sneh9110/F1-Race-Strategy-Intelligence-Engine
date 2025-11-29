"""
F1 Race Strategy Intelligence Engine - Feature Engineering Module

This module provides a comprehensive feature engineering pipeline for F1 race data.

Architecture:
    BaseFeature → Feature Calculators → Engines (Batch/Streaming) → FeatureStore

Main Components:
    - BaseFeature: Abstract base class for all feature calculators
    - FeatureStore: Parquet-based storage with Redis caching
    - FeatureRegistry: Feature discovery and dependency management
    - BatchFeatureEngine: Historical data processing
    - StreamingFeatureEngine: Real-time feature computation

Usage Example:
    from features import BatchFeatureEngine, FeatureRegistry
    
    engine = BatchFeatureEngine()
    results = engine.compute_features(
        session_ids=['2024_MONACO_RACE'],
        feature_names=['stint_summary', 'degradation_slope'],
        parallel=True
    )
"""

# Base infrastructure
from features.base import BaseFeature, FeatureConfig, FeatureResult
from features.store import FeatureStore, FeatureMetadata
from features.registry import FeatureRegistry, register_feature, FeatureInfo

# Engines
from features.batch_engine import BatchFeatureEngine
from features.streaming_engine import StreamingFeatureEngine

# Feature calculators - Stint & Pace
from features.stint_features import StintSummaryFeature, StintPaceEvolutionFeature
from features.pace_features import (
    LapPaceDeltaFeature,
    RollingPaceFeature,
    SectorPaceFeature
)

# Feature calculators - Degradation & Pitstop
from features.degradation_features import (
    DegradationSlopeFeature,
    ExponentialDegradationFeature,
    CliffDetectionFeature,
    DegradationAnomalyFeature
)
from features.pitstop_features import (
    UndercutDeltaFeature,
    OvercutDeltaFeature,
    PitLossModelFeature,
    PitWindowFeature,
    StrategyConvergenceFeature
)

# Feature calculators - Tire
from features.tire_features import (
    TireWarmupCurveFeature,
    TireDropoffFeature,
    TirePerformanceWindowFeature,
    CompoundComparisonFeature
)

# Feature calculators - Weather
from features.weather_features import (
    WeatherAdjustedPaceFeature,
    TrackEvolutionFeature,
    WeatherTrendFeature,
    CompoundWeatherSuitabilityFeature
)

# Feature calculators - Safety Car
from features.safety_car_features import (
    HistoricalSCProbabilityFeature,
    RealTimeSCProbabilityFeature,
    SCImpactFeature,
    SectorRiskFeature
)

# Feature calculators - Traffic
from features.traffic_features import (
    CleanAirPenaltyFeature,
    TrafficDensityFeature,
    LappingImpactFeature,
    PositionBattleFeature
)

# Feature calculators - Telemetry
from features.telemetry_features import (
    DriverStyleFeature,
    FuelEffectFeature,
    TireTemperatureFeature,
    EnergyManagementFeature
)

# Utilities
from features import utils


__all__ = [
    # Base
    'BaseFeature',
    'FeatureConfig',
    'FeatureResult',
    'FeatureStore',
    'FeatureMetadata',
    'FeatureRegistry',
    'register_feature',
    'FeatureInfo',
    
    # Engines
    'BatchFeatureEngine',
    'StreamingFeatureEngine',
    
    # Stint & Pace Features
    'StintSummaryFeature',
    'StintPaceEvolutionFeature',
    'LapPaceDeltaFeature',
    'RollingPaceFeature',
    'SectorPaceFeature',
    
    # Degradation & Pitstop Features
    'DegradationSlopeFeature',
    'ExponentialDegradationFeature',
    'CliffDetectionFeature',
    'DegradationAnomalyFeature',
    'UndercutDeltaFeature',
    'OvercutDeltaFeature',
    'PitLossModelFeature',
    'PitWindowFeature',
    'StrategyConvergenceFeature',
    
    # Tire Features
    'TireWarmupCurveFeature',
    'TireDropoffFeature',
    'TirePerformanceWindowFeature',
    'CompoundComparisonFeature',
    
    # Weather Features
    'WeatherAdjustedPaceFeature',
    'TrackEvolutionFeature',
    'WeatherTrendFeature',
    'CompoundWeatherSuitabilityFeature',
    
    # Safety Car Features
    'HistoricalSCProbabilityFeature',
    'RealTimeSCProbabilityFeature',
    'SCImpactFeature',
    'SectorRiskFeature',
    
    # Traffic Features
    'CleanAirPenaltyFeature',
    'TrafficDensityFeature',
    'LappingImpactFeature',
    'PositionBattleFeature',
    
    # Telemetry Features
    'DriverStyleFeature',
    'FuelEffectFeature',
    'TireTemperatureFeature',
    'EnergyManagementFeature',
    
    # Utilities
    'utils'
]


__version__ = '1.0.0'
__author__ = 'F1 Race Strategy Intelligence Team'

