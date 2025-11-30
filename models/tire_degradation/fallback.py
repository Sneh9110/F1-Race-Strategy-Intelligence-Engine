"""
Physics-based fallback heuristics for tire degradation.

Used when ML models fail or for cold-start scenarios.
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

from models.tire_degradation.base import (
    PredictionInput,
    PredictionOutput,
    TireCompound
)
from app.utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


class FallbackHeuristics:
    """
    Physics-based heuristics for tire degradation prediction.
    
    Uses empirical rules from tire_compounds.yaml and tracks.yaml
    when ML models are unavailable.
    """
    
    def __init__(self):
        """Initialize fallback heuristics."""
        self.compound_data = self._load_compound_data()
        self.track_data = self._load_track_data()
    
    def _load_compound_data(self) -> Dict[str, Any]:
        """Load tire compound characteristics."""
        compound_path = Path(settings.CONFIG_DIR) / 'tire_compounds.yaml'
        
        if compound_path.exists():
            with open(compound_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default compound data
        return {
            'SOFT': {
                'base_degradation_rate': 0.08,
                'expected_life': 15,
                'cliff_tendency': 0.8,
                'temp_sensitivity': 1.2
            },
            'MEDIUM': {
                'base_degradation_rate': 0.05,
                'expected_life': 25,
                'cliff_tendency': 0.5,
                'temp_sensitivity': 1.0
            },
            'HARD': {
                'base_degradation_rate': 0.03,
                'expected_life': 40,
                'cliff_tendency': 0.3,
                'temp_sensitivity': 0.8
            },
            'INTERMEDIATE': {
                'base_degradation_rate': 0.06,
                'expected_life': 20,
                'cliff_tendency': 0.6,
                'temp_sensitivity': 1.1
            },
            'WET': {
                'base_degradation_rate': 0.07,
                'expected_life': 18,
                'cliff_tendency': 0.7,
                'temp_sensitivity': 1.0
            }
        }
    
    def _load_track_data(self) -> Dict[str, Any]:
        """Load track characteristics."""
        track_path = Path(settings.CONFIG_DIR) / 'tracks.yaml'
        
        if track_path.exists():
            with open(track_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default track data
        return {
            'default': {
                'tire_wear_factor': 1.0,
                'thermal_severity': 1.0
            }
        }
    
    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """
        Make prediction using physics-based heuristics.
        
        Args:
            input_data: Prediction input
        
        Returns:
            Heuristic-based prediction
        """
        logger.info(f"Using fallback heuristics for {input_data.tire_compound}")
        
        # Get base degradation rate
        deg_rate = self._calculate_degradation_rate(input_data)
        
        # Generate degradation curve
        curve = self._generate_curve(input_data, deg_rate)
        
        # Calculate usable life
        usable_life = self._calculate_usable_life(input_data, curve)
        
        # Predict dropoff lap
        dropoff_lap = self._predict_dropoff(input_data, curve)
        
        # Confidence is lower for heuristics
        confidence = 0.6
        
        return PredictionOutput(
            degradation_curve=curve,
            usable_life=usable_life,
            dropoff_lap=dropoff_lap,
            confidence=confidence,
            degradation_rate=deg_rate,
            metadata={
                'model_type': 'fallback_heuristic',
                'version': '1.0.0',
                'method': 'physics_based',
                'tire_age': input_data.tire_age,
                'tire_compound': input_data.tire_compound
            }
        )
    
    def _calculate_degradation_rate(self, input_data: PredictionInput) -> float:
        """
        Calculate base degradation rate.
        
        Args:
            input_data: Prediction input
        
        Returns:
            Degradation rate (s/lap)
        """
        # Get compound characteristics
        compound_info = self.compound_data.get(
            input_data.tire_compound,
            self.compound_data['MEDIUM']
        )
        
        base_rate = compound_info['base_degradation_rate']
        
        # Adjust for tire age (exponential aging)
        age_factor = np.exp(0.02 * input_data.tire_age)
        
        # Adjust for temperature
        temp_factor = 1.0
        if input_data.weather_temp > 25:
            temp_sensitivity = compound_info['temp_sensitivity']
            temp_factor = 1.0 + (input_data.weather_temp - 25) * 0.01 * temp_sensitivity
        
        # Adjust for driver aggression
        aggression_factor = 1.0 + (input_data.driver_aggression * 0.3)
        
        # Adjust for track
        track_info = self.track_data.get(
            input_data.track_name,
            self.track_data['default']
        )
        track_factor = track_info.get('tire_wear_factor', 1.0)
        
        # Combined degradation rate
        deg_rate = base_rate * age_factor * temp_factor * aggression_factor * track_factor
        
        return float(deg_rate)
    
    def _generate_curve(
        self,
        input_data: PredictionInput,
        base_rate: float,
        num_laps: int = 50
    ) -> List[float]:
        """
        Generate degradation curve.
        
        Args:
            input_data: Prediction input
            base_rate: Base degradation rate
            num_laps: Number of laps to predict
        
        Returns:
            Degradation curve
        """
        compound_info = self.compound_data.get(
            input_data.tire_compound,
            self.compound_data['MEDIUM']
        )
        
        curve = []
        current_age = input_data.tire_age
        
        # Cliff parameters
        cliff_tendency = compound_info['cliff_tendency']
        expected_life = compound_info['expected_life']
        
        for lap in range(num_laps):
            age = current_age + lap
            
            # Base exponential degradation
            deg = base_rate * np.exp(0.015 * age)
            
            # Add cliff effect near expected life
            if age > expected_life * 0.7:
                # Sudden increase as tire approaches end of life
                cliff_progress = (age - expected_life * 0.7) / (expected_life * 0.3)
                cliff_factor = 1.0 + (cliff_tendency * cliff_progress ** 2)
                deg *= cliff_factor
            
            # Clamp to realistic values
            deg = np.clip(deg, 0.0, 5.0)
            curve.append(float(deg))
        
        return curve
    
    def _calculate_usable_life(
        self,
        input_data: PredictionInput,
        curve: List[float]
    ) -> int:
        """
        Calculate usable tire life.
        
        Args:
            input_data: Prediction input
            curve: Degradation curve
        
        Returns:
            Usable life in laps
        """
        # Threshold depends on compound
        thresholds = {
            'SOFT': 1.5,
            'MEDIUM': 2.0,
            'HARD': 2.5,
            'INTERMEDIATE': 1.8,
            'WET': 1.8
        }
        threshold = thresholds.get(input_data.tire_compound, 2.0)
        
        # Find when degradation exceeds threshold
        for i, deg in enumerate(curve):
            if deg > threshold:
                return max(1, i)  # At least 1 lap
        
        # If never exceeds threshold, return curve length
        return len(curve)
    
    def _predict_dropoff(
        self,
        input_data: PredictionInput,
        curve: List[float]
    ) -> Optional[int]:
        """
        Predict cliff/dropoff lap.
        
        Args:
            input_data: Prediction input
            curve: Degradation curve
        
        Returns:
            Dropoff lap or None
        """
        if len(curve) < 5:
            return None
        
        # Look for sudden jump in degradation
        for i in range(2, len(curve)):
            # Check acceleration (second derivative)
            if i < 2:
                continue
            
            accel = (curve[i] - 2 * curve[i-1] + curve[i-2])
            
            # Significant acceleration indicates cliff
            if accel > 0.2:
                return i + input_data.tire_age
        
        return None
    
    def get_compound_info(self, compound: str) -> Dict[str, Any]:
        """
        Get compound characteristics.
        
        Args:
            compound: Tire compound name
        
        Returns:
            Compound info dict
        """
        return self.compound_data.get(compound, {})
    
    def get_track_info(self, track_name: str) -> Dict[str, Any]:
        """
        Get track characteristics.
        
        Args:
            track_name: Track name
        
        Returns:
            Track info dict
        """
        return self.track_data.get(track_name, self.track_data.get('default', {}))
