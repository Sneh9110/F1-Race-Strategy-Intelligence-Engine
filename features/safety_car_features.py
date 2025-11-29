"""
Safety car probability feature calculators.

Analyzes safety car likelihood based on historical data and real-time conditions.
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from features.base import BaseFeature, FeatureConfig
from features.registry import register_feature
from data_pipeline.schemas.safety_car_schema import SafetyCarSession, IncidentLog
from config.settings import settings


def load_track_config(track_name: str) -> Dict[str, Any]:
    """Load track-specific configuration."""
    config_path = Path(settings.CONFIG_DIR) / "tracks.yaml"
    with open(config_path, 'r') as f:
        tracks = yaml.safe_load(f)
    return tracks.get(track_name, {})


@register_feature(
    name='historical_sc_probability',
    version='v1.0.0',
    description='Calculate track-specific safety car probability',
    dependencies=[],
    computation_cost_ms=40,
    tags=['safety_car', 'probability', 'historical']
)
class HistoricalSCProbabilityFeature(BaseFeature):
    """
    Calculates safety car probability from historical data.
    
    Formula:
        adjusted_probability = (base_probability + historical_sc_rate) / 2
        historical_sc_rate = count(SC_events) / count(races)
    
    Output columns:
    - track_name: Track name
    - base_probability: Base SC probability from config
    - historical_sc_rate: Historical SC rate
    - adjusted_probability: Combined probability
    - sc_events_count: Number of historical SC events
    - total_races: Total races analyzed
    """
    
    def __init__(self, config: FeatureConfig = None, track_name: str = "generic"):
        super().__init__(
            name='historical_sc_probability',
            version='v1.0.0',
            config=config or FeatureConfig()
        )
        self.track_name = track_name
        self.track_config = load_track_config(track_name)
    
    def _calculate(
        self,
        historical_sessions: List[SafetyCarSession],
        **kwargs
    ) -> pd.DataFrame:
        """Calculate historical SC probability."""
        base_probability = self.track_config.get('safety_car_probability', 0.3)
        
        # Count SC events
        sc_events = sum(1 for session in historical_sessions if session.safety_car_deployed)
        total_races = len(historical_sessions)
        
        historical_rate = sc_events / total_races if total_races > 0 else base_probability
        adjusted_probability = (base_probability + historical_rate) / 2.0
        
        return pd.DataFrame([{
            'track_name': self.track_name,
            'base_probability': float(base_probability),
            'historical_sc_rate': float(historical_rate),
            'adjusted_probability': float(adjusted_probability),
            'sc_events_count': int(sc_events),
            'total_races': int(total_races)
        }])
    
    def _get_dependencies(self) -> List[str]:
        return []


@register_feature(
    name='realtime_sc_probability',
    version='v1.0.0',
    description='Update SC probability in real-time',
    dependencies=[],
    computation_cost_ms=50,
    tags=['safety_car', 'probability', 'realtime']
)
class RealTimeSCProbabilityFeature(BaseFeature):
    """
    Updates safety car probability during race based on incidents.
    
    Formula:
        current_probability = base * (1 + incident_risk + proximity_risk + lap_progress_factor)
    
    Output columns:
    - current_lap: Current lap number
    - base_probability: Base SC probability
        - incident_risk_score: Risk from recent incidents
        - proximity_risk_score: Risk from close racing
        - lap_progress_factor: Adjustment for race stage
        - current_sc_probability: Updated probability
    """
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(
            name='realtime_sc_probability',
            version='v1.0.0',
            config=config or FeatureConfig(thresholds={
                'incident_weight': 0.1,
                'proximity_threshold': 1.0,
                'proximity_weight': 0.05
            })
        )
    
    def _calculate(
        self,
        data: pd.DataFrame,
        base_probability: float = 0.3,
        recent_incidents: List[IncidentLog] = None,
        current_lap: int = 1,
        total_laps: int = 50,
        **kwargs
    ) -> pd.DataFrame:
        """Calculate real-time SC probability."""
        incident_weight = self.config.thresholds.get('incident_weight', 0.1)
        proximity_threshold = self.config.thresholds.get('proximity_threshold', 1.0)
        proximity_weight = self.config.thresholds.get('proximity_weight', 0.05)
        
        # Incident risk score
        if recent_incidents:
            severity_weights = {'MINOR': 0.3, 'MODERATE': 0.6, 'MAJOR': 1.0}
            incident_risk = sum(
                severity_weights.get(inc.severity, 0.5)
                for inc in recent_incidents
            ) * incident_weight
        else:
            incident_risk = 0.0
        
        # Lap progress factor (higher early and late in race)
        progress = current_lap / total_laps
        if progress < 0.2 or progress > 0.8:
            lap_progress_factor = 0.2
        else:
            lap_progress_factor = 0.0
        
        # Driver proximity risk
        if 'gap_to_ahead' in data.columns:
            close_battles = (data['gap_to_ahead'] < proximity_threshold).sum()
            proximity_risk = close_battles * proximity_weight
        else:
            proximity_risk = 0.0
        
        # Combined probability
        current_probability = min(
            1.0,
            base_probability * (1 + incident_risk + proximity_risk + lap_progress_factor)
        )
        
        return pd.DataFrame([{
            'current_lap': int(current_lap),
            'base_probability': float(base_probability),
            'incident_risk_score': float(incident_risk),
            'proximity_risk_score': float(proximity_risk),
            'lap_progress_factor': float(lap_progress_factor),
            'current_sc_probability': float(current_probability)
        }])
    
    def _get_dependencies(self) -> List[str]:
        return []


@register_feature(
    name='sc_impact',
    version='v1.0.0',
    description='Quantify strategic impact of safety car',
    dependencies=[],
    computation_cost_ms=40,
    tags=['safety_car', 'impact', 'strategy']
)
class SCImpactFeature(BaseFeature):
    """
    Quantifies strategic impact of safety car deployment.
    
    Output columns:
    - field_compression: Reduction in gaps (0-1)
    - pit_opportunity_score: Value of pit window (0-1)
    - position_change_potential: Number of positions that could change
    - undercut_range: Number of cars within undercut range
    """
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(
            name='sc_impact',
            version='v1.0.0',
            config=config or FeatureConfig(thresholds={'undercut_range': 3.0})
        )
    
    def _calculate(
        self,
        gaps_before_sc: pd.DataFrame,
        gaps_after_sc: pd.DataFrame = None,
        in_pit_window: pd.DataFrame = None,
        **kwargs
    ) -> pd.DataFrame:
        """Calculate SC impact."""
        # Field compression
        if gaps_after_sc is not None:
            gap_before = gaps_before_sc['gap_to_leader'].sum()
            gap_after = gaps_after_sc['gap_to_leader'].sum()
            field_compression = (gap_before - gap_after) / gap_before if gap_before > 0 else 0.0
        else:
            field_compression = 0.7  # Typical compression
        
        # Pit opportunity
        if in_pit_window is not None:
            pit_opportunity_score = in_pit_window['in_pit_window'].mean()
        else:
            pit_opportunity_score = 0.5
        
        # Position change potential
        undercut_range = self.config.thresholds.get('undercut_range', 3.0)
        if 'gap_to_ahead' in gaps_before_sc.columns:
            position_changes = (gaps_before_sc['gap_to_ahead'] < undercut_range).sum()
        else:
            position_changes = 5
        
        return pd.DataFrame([{
            'field_compression': float(field_compression),
            'pit_opportunity_score': float(pit_opportunity_score),
            'position_change_potential': int(position_changes),
            'undercut_range_cars': int(position_changes)
        }])
    
    def _get_dependencies(self) -> List[str]:
        return []


@register_feature(
    name='sector_risk',
    version='v1.0.0',
    description='Identify high-risk track sectors for incidents',
    dependencies=[],
    computation_cost_ms=30,
    tags=['safety_car', 'risk', 'sectors']
)
class SectorRiskFeature(BaseFeature):
    """
    Identifies high-risk track sectors.
    
    Output columns:
    - sector_1_risk: Sector 1 incident risk (0-1)
    - sector_2_risk: Sector 2 incident risk (0-1)
    - sector_3_risk: Sector 3 incident risk (0-1)
    - highest_risk_sector: Sector with highest risk
    """
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(
            name='sector_risk',
            version='v1.0.0',
            config=config or FeatureConfig()
        )
    
    def _calculate(
        self,
        incident_logs: List[IncidentLog],
        **kwargs
    ) -> pd.DataFrame:
        """Calculate sector risk scores."""
        if not incident_logs:
            return pd.DataFrame([{
                'sector_1_risk': 0.33,
                'sector_2_risk': 0.33,
                'sector_3_risk': 0.33,
                'highest_risk_sector': 1
            }])
        
        # Count incidents per sector
        sector_counts = {1: 0, 2: 0, 3: 0}
        for incident in incident_logs:
            if hasattr(incident, 'sector') and incident.sector in sector_counts:
                sector_counts[incident.sector] += 1
        
        total_incidents = sum(sector_counts.values())
        
        # Calculate risk scores
        sector_risks = {
            sector: count / total_incidents if total_incidents > 0 else 0.33
            for sector, count in sector_counts.items()
        }
        
        highest_risk_sector = max(sector_risks, key=sector_risks.get)
        
        return pd.DataFrame([{
            'sector_1_risk': float(sector_risks[1]),
            'sector_2_risk': float(sector_risks[2]),
            'sector_3_risk': float(sector_risks[3]),
            'highest_risk_sector': int(highest_risk_sector),
            'total_incidents': int(total_incidents)
        }])
    
    def _get_dependencies(self) -> List[str]:
        return []
