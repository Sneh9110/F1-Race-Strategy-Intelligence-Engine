"""
Data preparation pipeline for tire degradation models.

Transforms HistoricalStint data and computed features into ML-ready format.
"""

from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from features.store import FeatureStore
from data_pipeline.schemas.historical_schema import HistoricalStint
from features.degradation_features import (
    DegradationSlopeFeature,
    CliffDetectionFeature
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DataPreparationConfig:
    """Configuration for data preparation."""
    min_laps_per_stint: int = 5
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    stratify_columns: List[str] = None
    lookback_window: int = 10
    normalize_features: bool = True
    
    def __post_init__(self):
        if self.stratify_columns is None:
            self.stratify_columns = ['tire_compound', 'track_name']
        
        # Validate ratios sum to 1.0
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {total}")


class DataPreparationPipeline:
    """
    Pipeline for preparing tire degradation training data.
    
    Handles feature extraction, encoding, splitting, and augmentation with
    computed features from the feature store.
    """
    
    def __init__(self, config: Optional[DataPreparationConfig] = None):
        """
        Initialize pipeline.
        
        Args:
            config: Data preparation configuration
        """
        self.config = config or DataPreparationConfig()
        self.feature_store = FeatureStore()
        self.label_encoders = {}
        self.scaler = StandardScaler() if self.config.normalize_features else None
        self.feature_names = []
        
        logger.info("Initialized DataPreparationPipeline")
    
    def prepare_training_data(
        self,
        stints: List[HistoricalStint]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare complete training dataset from historical stints.
        
        Args:
            stints: List of historical stint data
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Preparing training data from {len(stints)} stints")
        
        # Extract features from all stints
        data_list = []
        for stint in stints:
            if len(stint.lap_times) < self.config.min_laps_per_stint:
                continue
            
            features = self.extract_features_from_stint(stint)
            if features is not None:
                data_list.append(features)
        
        if not data_list:
            raise ValueError("No valid stints found for training")
        
        # Create DataFrame
        df = pd.DataFrame(data_list)
        logger.info(f"Extracted features: {df.shape}")
        
        # Encode categoricals
        df = self.encode_categoricals(df)
        
        # Split data
        train_df, val_df, test_df = self.split_train_val_test(df)
        
        logger.info(
            f"Split data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )
        
        return train_df, val_df, test_df
    
    def extract_features_from_stint(self, stint: HistoricalStint) -> Optional[Dict[str, Any]]:
        """
        Extract features from a single stint.
        
        Args:
            stint: Historical stint data
        
        Returns:
            Dictionary of features, or None if stint is invalid
        """
        try:
            lap_times = np.array(stint.lap_times)
            
            if len(lap_times) < self.config.min_laps_per_stint:
                return None
            
            # Basic features
            features = {
                'tire_compound': stint.tire_compound,
                'tire_age_start': stint.tire_age_start,
                'tire_age_end': stint.tire_age_end,
                'stint_length': len(lap_times),
                'stint_number': stint.stint_number,
                'track_name': stint.session_id.split('_')[1] if '_' in stint.session_id else 'UNKNOWN',
                'session_id': stint.session_id,
                'driver_id': stint.driver_id,
            }
            
            # Lap time statistics
            features['avg_lap_time'] = float(np.mean(lap_times))
            features['min_lap_time'] = float(np.min(lap_times))
            features['max_lap_time'] = float(np.max(lap_times))
            features['std_lap_time'] = float(np.std(lap_times))
            features['lap_time_range'] = features['max_lap_time'] - features['min_lap_time']
            
            # Pace evolution
            if len(lap_times) >= 3:
                first_third = lap_times[:len(lap_times)//3]
                last_third = lap_times[-len(lap_times)//3:]
                features['pace_evolution'] = float(np.mean(last_third) - np.mean(first_third))
            else:
                features['pace_evolution'] = 0.0
            
            # Degradation rate (linear fit)
            if len(lap_times) >= 5:
                laps = np.arange(len(lap_times))
                coeffs = np.polyfit(laps, lap_times, 1)
                features['degradation_rate'] = float(coeffs[0])
            else:
                features['degradation_rate'] = 0.0
            
            # Targets
            features['degradation_rate_target'] = features['degradation_rate']
            features['usable_life_target'] = stint.tire_age_end
            
            # Detect dropoff lap
            features['dropoff_lap_target'] = self._detect_dropoff_lap(lap_times)
            
            # Try to augment with computed features
            try:
                computed_features = self.augment_with_computed_features(stint.session_id)
                features.update(computed_features)
            except Exception as e:
                logger.warning(f"Failed to augment features for {stint.session_id}: {e}")
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract features from stint: {e}")
            return None
    
    def augment_with_computed_features(self, session_id: str) -> Dict[str, Any]:
        """
        Augment with features from feature store.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Dictionary of computed features
        """
        computed = {}
        
        try:
            # Load degradation features
            deg_features = self.feature_store.load_features(
                session_id=session_id,
                feature_names=['degradation_slope', 'cliff_detection'],
                version='latest'
            )
            
            if deg_features is not None and not deg_features.empty:
                computed['deg_slope_computed'] = float(deg_features['degradation_rate'].iloc[0])
                computed['cliff_detected'] = bool(deg_features['cliff_detected'].iloc[0])
            
            # Load tire features
            tire_features = self.feature_store.load_features(
                session_id=session_id,
                feature_names=['tire_warmup', 'tire_dropoff'],
                version='latest'
            )
            
            if tire_features is not None and not tire_features.empty:
                computed['warmup_laps'] = int(tire_features['warmup_laps'].iloc[0])
                computed['dropoff_detected'] = bool(tire_features.get('dropoff_lap', pd.Series([0])).iloc[0] > 0)
            
            # Load weather features
            weather_features = self.feature_store.load_features(
                session_id=session_id,
                feature_names=['weather_adjusted_pace'],
                version='latest'
            )
            
            if weather_features is not None and not weather_features.empty:
                computed['weather_temp'] = float(weather_features.get('air_temp', pd.Series([25.0])).iloc[0])
                computed['weather_correction'] = float(weather_features.get('temp_correction', pd.Series([0.0])).iloc[0])
            
            # Load driver style features
            driver_features = self.feature_store.load_features(
                session_id=session_id,
                feature_names=['driver_style'],
                version='latest'
            )
            
            if driver_features is not None and not driver_features.empty:
                computed['driver_aggression'] = float(driver_features.get('aggression_score', pd.Series([0.5])).iloc[0])
            
        except Exception as e:
            logger.debug(f"Could not load computed features for {session_id}: {e}")
            # Provide defaults
            computed['weather_temp'] = 25.0
            computed['driver_aggression'] = 0.5
        
        return computed
    
    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with encoded categoricals
        """
        df = df.copy()
        
        categorical_cols = ['tire_compound', 'track_name']
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def split_train_val_test(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test sets with stratification.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Create stratification column
        stratify_col = None
        if self.config.stratify_columns:
            valid_cols = [c for c in self.config.stratify_columns if c in df.columns]
            if valid_cols:
                stratify_col = df[valid_cols].astype(str).agg('_'.join, axis=1)
        
        # First split: train+val vs test
        train_val_size = self.config.train_ratio + self.config.val_ratio
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.config.test_ratio,
            stratify=stratify_col,
            random_state=42
        )
        
        # Second split: train vs val
        val_size_adjusted = self.config.val_ratio / train_val_size
        if stratify_col is not None:
            stratify_train_val = train_val_df[self.config.stratify_columns].astype(str).agg('_'.join, axis=1)
        else:
            stratify_train_val = None
        
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            stratify=stratify_train_val,
            random_state=42
        )
        
        return train_df, val_df, test_df
    
    def create_sequences(
        self,
        df: pd.DataFrame,
        lookback: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time-series sequences for sequential models.
        
        Args:
            df: Input DataFrame
            lookback: Lookback window size
        
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        if lookback is None:
            lookback = self.config.lookback_window
        
        # Group by session/driver/stint
        grouped = df.groupby(['session_id', 'driver_id', 'stint_number'])
        
        sequences_X = []
        sequences_y = []
        
        for _, group in grouped:
            if len(group) < lookback:
                continue
            
            # Extract features and targets
            feature_cols = [c for c in group.columns if c.endswith('_encoded') or 
                          c in ['tire_age_start', 'avg_lap_time', 'std_lap_time', 
                                'weather_temp', 'driver_aggression']]
            
            X = group[feature_cols].values
            y = group['degradation_rate_target'].values
            
            # Create sequences
            for i in range(len(X) - lookback):
                sequences_X.append(X[i:i+lookback])
                sequences_y.append(y[i+lookback])
        
        return np.array(sequences_X), np.array(sequences_y)
    
    def _detect_dropoff_lap(self, lap_times: np.ndarray) -> Optional[int]:
        """
        Detect dropoff/cliff lap in stint.
        
        Args:
            lap_times: Array of lap times
        
        Returns:
            Dropoff lap number, or None if no dropoff detected
        """
        if len(lap_times) < 5:
            return None
        
        # Calculate rolling mean and std
        window = min(3, len(lap_times) // 3)
        rolling_mean = pd.Series(lap_times).rolling(window=window).mean()
        rolling_std = pd.Series(lap_times).rolling(window=window).std()
        
        # Detect cliff: lap time > mean + 2*std
        for i in range(window, len(lap_times)):
            if lap_times[i] > rolling_mean.iloc[i] + 2 * rolling_std.iloc[i]:
                return i + 1  # 1-indexed lap number
        
        return None
