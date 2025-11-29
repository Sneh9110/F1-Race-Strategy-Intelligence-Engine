"""
Tire degradation feature calculators.

Analyzes tire wear patterns, degradation rates, and performance cliff detection.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

from features.base import BaseFeature, FeatureConfig
from features.registry import register_feature
from data_pipeline.schemas.historical_schema import HistoricalStint
from app.utils.logger import get_logger

logger = get_logger(__name__)


@register_feature(
    name='degradation_slope',
    version='v1.0.0',
    description='Calculate linear tire degradation rate',
    dependencies=[],
    computation_cost_ms=80,
    tags=['tire', 'degradation', 'regression']
)
class DegradationSlopeFeature(BaseFeature):
    """
    Calculates linear regression on lap times vs tire age.
    
    Formula:
        degradation_rate = Σ[(x_i - x̄)(y_i - ȳ)] / Σ[(x_i - x̄)²]
        where x = tire_age, y = lap_time
    
    Output columns:
    - stint_id: Stint identifier
    - degradation_rate: Slope in seconds/lap
    - intercept: Y-intercept (baseline lap time)
    - r_squared: Coefficient of determination (fit quality)
    - predicted_lap_time_at_age_10: Predicted lap time at age 10
    - predicted_lap_time_at_age_20: Predicted lap time at age 20
    """
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(
            name='degradation_slope',
            version='v1.0.0',
            config=config or FeatureConfig(min_data_points=5)
        )
    
    def _calculate(self, data: List[HistoricalStint], **kwargs) -> pd.DataFrame:
        """Calculate degradation slope for each stint."""
        results = []
        
        for stint in data:
            lap_times = np.array(stint.lap_times)
            tire_ages = np.arange(1, len(lap_times) + 1)
            
            if len(lap_times) < self.config.min_data_points:
                logger.warning(
                    f"Insufficient data for stint {stint.stint_id}: "
                    f"{len(lap_times)} < {self.config.min_data_points}"
                )
                continue
            
            # Linear regression
            try:
                slope, intercept, r_value, _, _ = stats.linregress(tire_ages, lap_times)
                r_squared = r_value ** 2
                
                # Predictions
                pred_age_10 = intercept + slope * 10
                pred_age_20 = intercept + slope * 20
                
                results.append({
                    'stint_id': stint.stint_id,
                    'driver_number': stint.driver_number,
                    'tire_compound': stint.tire_compound,
                    'degradation_rate': float(slope),
                    'intercept': float(intercept),
                    'r_squared': float(r_squared),
                    'predicted_lap_time_at_age_10': float(pred_age_10),
                    'predicted_lap_time_at_age_20': float(pred_age_20),
                    'num_laps': len(lap_times)
                })
            except Exception as e:
                logger.error(f"Failed to compute degradation for stint {stint.stint_id}: {e}")
        
        return pd.DataFrame(results)
    
    def _get_dependencies(self) -> List[str]:
        return []


@register_feature(
    name='exponential_degradation',
    version='v1.0.0',
    description='Fit exponential tire degradation model',
    dependencies=[],
    computation_cost_ms=150,
    tags=['tire', 'degradation', 'exponential']
)
class ExponentialDegradationFeature(BaseFeature):
    """
    Fits exponential degradation model: lap_time = a * exp(b * tire_age) + c
    
    This model better captures non-linear degradation and "cliff" behavior.
    
    Output columns:
    - stint_id: Stint identifier
    - param_a: Exponential coefficient
    - param_b: Exponential rate
    - param_c: Baseline offset
    - fit_error: Mean squared error
    - convergence_success: Whether optimization converged
    """
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(
            name='exponential_degradation',
            version='v1.0.0',
            config=config or FeatureConfig(min_data_points=8)
        )
    
    def _calculate(self, data: List[HistoricalStint], **kwargs) -> pd.DataFrame:
        """Fit exponential degradation model."""
        def exponential_model(x, a, b, c):
            return a * np.exp(b * x) + c
        
        results = []
        
        for stint in data:
            lap_times = np.array(stint.lap_times)
            tire_ages = np.arange(1, len(lap_times) + 1)
            
            if len(lap_times) < self.config.min_data_points:
                continue
            
            try:
                # Initial guess
                p0 = [0.1, 0.01, lap_times.mean()]
                
                # Curve fitting
                params, _ = curve_fit(
                    exponential_model,
                    tire_ages,
                    lap_times,
                    p0=p0,
                    maxfev=1000
                )
                
                # Calculate fit error
                predicted = exponential_model(tire_ages, *params)
                mse = np.mean((lap_times - predicted) ** 2)
                
                results.append({
                    'stint_id': stint.stint_id,
                    'driver_number': stint.driver_number,
                    'tire_compound': stint.tire_compound,
                    'param_a': float(params[0]),
                    'param_b': float(params[1]),
                    'param_c': float(params[2]),
                    'fit_error': float(mse),
                    'convergence_success': True
                })
            except Exception as e:
                logger.warning(f"Exponential fit failed for stint {stint.stint_id}: {e}")
                results.append({
                    'stint_id': stint.stint_id,
                    'driver_number': stint.driver_number,
                    'tire_compound': stint.tire_compound,
                    'param_a': None,
                    'param_b': None,
                    'param_c': None,
                    'fit_error': None,
                    'convergence_success': False
                })
        
        return pd.DataFrame(results)
    
    def _get_dependencies(self) -> List[str]:
        return []


@register_feature(
    name='cliff_detection',
    version='v1.0.0',
    description='Detect tire performance cliff',
    dependencies=[],
    computation_cost_ms=60,
    tags=['tire', 'degradation', 'cliff']
)
class CliffDetectionFeature(BaseFeature):
    """
    Identifies lap where tire performance drops sharply.
    
    Cliff definition: First lap where lap_time > (avg_first_5_laps + threshold * std)
    
    Output columns:
    - stint_id: Stint identifier
    - cliff_detected: Whether cliff was detected
    - cliff_lap: Lap number where cliff occurred
    - cliff_tire_age: Tire age at cliff
    - cliff_magnitude: Lap time increase at cliff (seconds)
    - pre_cliff_avg: Average lap time before cliff
    - post_cliff_avg: Average lap time after cliff
    """
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(
            name='cliff_detection',
            version='v1.0.0',
            config=config or FeatureConfig(
                min_data_points=8,
                thresholds={'cliff_threshold': 2.0}
            )
        )
    
    def _calculate(self, data: List[HistoricalStint], **kwargs) -> pd.DataFrame:
        """Detect performance cliffs."""
        results = []
        threshold = self.config.thresholds.get('cliff_threshold', 2.0)
        
        for stint in data:
            lap_times = np.array(stint.lap_times)
            
            if len(lap_times) < self.config.min_data_points:
                continue
            
            # Baseline: first 5 laps
            baseline_size = min(5, len(lap_times) // 2)
            baseline = lap_times[:baseline_size]
            baseline_mean = baseline.mean()
            baseline_std = baseline.std() if len(baseline) > 1 else 0.1
            
            # Find cliff
            cliff_threshold_time = baseline_mean + threshold * baseline_std
            cliff_lap = None
            
            for i in range(baseline_size, len(lap_times)):
                if lap_times[i] > cliff_threshold_time:
                    cliff_lap = i + 1  # 1-indexed
                    break
            
            if cliff_lap:
                cliff_idx = cliff_lap - 1  # 0-indexed
                pre_cliff_avg = lap_times[:cliff_idx].mean() if cliff_idx > 0 else baseline_mean
                post_cliff_avg = lap_times[cliff_idx:].mean()
                cliff_magnitude = lap_times[cliff_idx] - baseline_mean
                
                results.append({
                    'stint_id': stint.stint_id,
                    'driver_number': stint.driver_number,
                    'tire_compound': stint.tire_compound,
                    'cliff_detected': True,
                    'cliff_lap': int(cliff_lap),
                    'cliff_tire_age': int(cliff_lap),
                    'cliff_magnitude': float(cliff_magnitude),
                    'pre_cliff_avg': float(pre_cliff_avg),
                    'post_cliff_avg': float(post_cliff_avg)
                })
            else:
                results.append({
                    'stint_id': stint.stint_id,
                    'driver_number': stint.driver_number,
                    'tire_compound': stint.tire_compound,
                    'cliff_detected': False,
                    'cliff_lap': None,
                    'cliff_tire_age': None,
                    'cliff_magnitude': None,
                    'pre_cliff_avg': float(baseline_mean),
                    'post_cliff_avg': None
                })
        
        return pd.DataFrame(results)
    
    def _get_dependencies(self) -> List[str]:
        return []


@register_feature(
    name='degradation_anomaly',
    version='v1.0.0',
    description='Detect unusual degradation patterns',
    dependencies=['degradation_slope'],
    computation_cost_ms=40,
    tags=['tire', 'degradation', 'anomaly']
)
class DegradationAnomalyFeature(BaseFeature):
    """
    Detects unusual degradation by comparing to historical averages.
    
    Uses z-score to identify anomalies:
        z_score = (degradation_rate - historical_avg) / historical_std
        is_anomaly = z_score > threshold
    
    Output columns:
    - stint_id: Stint identifier
    - degradation_rate: Current stint degradation rate
    - historical_mean: Historical average degradation rate
    - historical_std: Historical std deviation
    - z_score: Standardized score
    - is_anomaly: Whether degradation is anomalous (z > 2.0)
    """
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(
            name='degradation_anomaly',
            version='v1.0.0',
            config=config or FeatureConfig(thresholds={'anomaly_threshold': 2.0})
        )
    
    def _calculate(
        self,
        data: pd.DataFrame,
        historical_data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Detect degradation anomalies.
        
        Args:
            data: Current degradation slope features
            historical_data: Historical degradation data for comparison
        """
        if historical_data is None or historical_data.empty:
            logger.warning("No historical data provided, using current data statistics")
            historical_data = data
        
        results = []
        threshold = self.config.thresholds.get('anomaly_threshold', 2.0)
        
        # Group by tire compound
        for compound in data['tire_compound'].unique():
            compound_data = data[data['tire_compound'] == compound]
            hist_compound = historical_data[historical_data['tire_compound'] == compound]
            
            if len(hist_compound) < 3:
                continue
            
            # Historical statistics
            hist_mean = hist_compound['degradation_rate'].mean()
            hist_std = hist_compound['degradation_rate'].std()
            
            if hist_std == 0:
                hist_std = 0.01  # Avoid division by zero
            
            # Calculate z-scores
            for _, row in compound_data.iterrows():
                z_score = (row['degradation_rate'] - hist_mean) / hist_std
                is_anomaly = abs(z_score) > threshold
                
                results.append({
                    'stint_id': row['stint_id'],
                    'driver_number': row['driver_number'],
                    'tire_compound': compound,
                    'degradation_rate': row['degradation_rate'],
                    'historical_mean': float(hist_mean),
                    'historical_std': float(hist_std),
                    'z_score': float(z_score),
                    'is_anomaly': is_anomaly
                })
        
        return pd.DataFrame(results)
    
    def _get_dependencies(self) -> List[str]:
        return ['degradation_slope']
