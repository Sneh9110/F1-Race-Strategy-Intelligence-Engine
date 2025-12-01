"""
Data preparation pipeline for lap time model training.

Handles feature extraction, normalization, and train/val/test splitting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from features.pace_features import LapPaceDelta, RollingPace, SectorPace
from features.tire_features import TireWarmup, TireDropoff, TireDegradation
from features.traffic_features import CleanAirPenalty, TrafficDensity
from features.weather_features import WeatherAdjustedPace, TrackEvolution
from features.safety_car_features import SafetyCarProbability, SafetyCarImpact
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DataPreparationPipeline:
    """
    Data preparation pipeline for lap time prediction.
    
    Features:
    - Feature extraction from historical lap data
    - Feature store integration
    - Train/val/test splitting with stratification
    - Categorical encoding and normalization
    - Data quality checks
    """
    
    def __init__(self):
        """Initialize data preparation pipeline."""
        self.feature_names = []
        self.categorical_encoders = {}
        self.numeric_scaler = StandardScaler()
        self.feature_importance = {}
        
        # Initialize feature extractors
        self.pace_features = {
            'lap_pace_delta': LapPaceDelta(),
            'rolling_pace': RollingPace(),
            'sector_pace': SectorPace()
        }
        
        self.tire_features = {
            'tire_warmup': TireWarmup(),
            'tire_dropoff': TireDropoff(),
            'tire_degradation': TireDegradation()
        }
        
        self.traffic_features = {
            'clean_air_penalty': CleanAirPenalty(),
            'traffic_density': TrafficDensity()
        }
        
        self.weather_features = {
            'weather_adjusted_pace': WeatherAdjustedPace(),
            'track_evolution': TrackEvolution()
        }
        
        self.safety_car_features = {
            'safety_car_probability': SafetyCarProbability(),
            'safety_car_impact': SafetyCarImpact()
        }
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        target_column: str = 'lap_time',
        test_size: float = 0.15,
        val_size: float = 0.15,
        stratify_columns: List[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training, validation, and test sets.
        
        Args:
            df: Historical lap data DataFrame
            target_column: Target column name
            test_size: Test set ratio
            val_size: Validation set ratio
            stratify_columns: Columns for stratification
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        logger.info(f"Preparing training data from {len(df)} samples")
        
        # Data quality checks
        df = self._quality_checks(df)
        
        # Create features
        df_features = self.create_features(df)
        
        # Separate features and target
        if target_column not in df_features.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        X = df_features.drop(columns=[target_column])
        y = df_features[target_column].values
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Encode categorical features
        X = self._encode_categoricals(X)
        
        # Create stratification key if specified
        stratify = None
        if stratify_columns:
            stratify = self._create_stratify_key(df_features, stratify_columns)
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
        
        # Second split: separate train and validation
        val_ratio = val_size / (1 - test_size)
        stratify_temp = None
        if stratify is not None:
            _, stratify_temp = train_test_split(
                stratify, stratify, test_size=test_size, random_state=42
            )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=stratify_temp
        )
        
        # Normalize numeric features
        X_train = self.normalize_features(X_train, fit=True)
        X_val = self.normalize_features(X_val, fit=False)
        X_test = self.normalize_features(X_test, fit=False)
        
        self.feature_names = list(X_train.columns)
        
        logger.info(f"Data prepared: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from historical lap data.
        
        Args:
            df: Historical lap data
        
        Returns:
            DataFrame with extracted features
        """
        logger.info("Creating features from historical data")
        
        df_features = df.copy()
        
        # Basic features (already present or extracted)
        required_base_features = [
            'tire_age', 'tire_compound', 'fuel_load', 'weather_temp',
            'track_name', 'driver_number', 'lap_number', 'lap_time'
        ]
        
        for feat in required_base_features:
            if feat not in df_features.columns:
                logger.warning(f"Missing required feature: {feat}")
        
        # Traffic state features
        if 'gap_to_ahead' in df_features.columns:
            df_features['traffic_state'] = df_features['gap_to_ahead'].apply(
                lambda x: 'DIRTY_AIR' if x < 1.0 else 'CLEAN_AIR'
            )
        else:
            df_features['traffic_state'] = 'CLEAN_AIR'
        
        # Safety car features
        if 'safety_car_active' not in df_features.columns:
            df_features['safety_car_active'] = False
        
        # Track temperature (estimate if missing)
        if 'track_temp' not in df_features.columns and 'weather_temp' in df_features.columns:
            df_features['track_temp'] = df_features['weather_temp'] + 10
        
        # Session progress
        if 'session_progress' not in df_features.columns and 'lap_number' in df_features.columns:
            df_features['session_progress'] = df_features['lap_number'] / df_features['lap_number'].max()
        
        # Stint number (if missing)
        if 'stint_number' not in df_features.columns:
            df_features['stint_number'] = 1
        
        # Extract pace features
        try:
            if 'lap_pace_delta' not in df_features.columns:
                df_features['rolling_avg_pace'] = df_features.groupby('driver_number')['lap_time'].transform(
                    lambda x: x.rolling(window=5, min_periods=1).mean()
                )
            
            if 'pace_consistency' not in df_features.columns:
                df_features['pace_consistency'] = df_features.groupby('driver_number')['lap_time'].transform(
                    lambda x: x.rolling(window=5, min_periods=1).std().fillna(0)
                )
        except Exception as e:
            logger.warning(f"Could not extract pace features: {e}")
        
        # Extract tire features
        try:
            if 'degradation_slope' not in df_features.columns and 'tire_age' in df_features.columns:
                df_features['degradation_slope'] = df_features.groupby(['driver_number', 'stint_number'])['lap_time'].transform(
                    lambda x: (x.iloc[-1] - x.iloc[0]) / len(x) if len(x) > 1 else 0
                )
        except Exception as e:
            logger.warning(f"Could not extract tire degradation features: {e}")
        
        # Traffic penalty from clean air baseline
        try:
            if 'traffic_penalty' not in df_features.columns and 'gap_to_ahead' in df_features.columns:
                df_features['traffic_penalty'] = df_features['gap_to_ahead'].apply(
                    lambda x: 0.4 * np.exp(-x / 1.0) if x < 3.0 else 0.0
                )
        except Exception as e:
            logger.warning(f"Could not extract traffic features: {e}")
        
        # Driver aggression (if available from feature store)
        if 'driver_aggression' not in df_features.columns:
            df_features['driver_aggression'] = 0.5  # Default medium aggression
        
        # Sector consistency (if sector times available)
        if all(col in df_features.columns for col in ['sector1_time', 'sector2_time', 'sector3_time']):
            df_features['sector_consistency'] = df_features[['sector1_time', 'sector2_time', 'sector3_time']].std(axis=1)
        
        logger.info(f"Created {len(df_features.columns)} features")
        
        return df_features
    
    def normalize_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Normalize numeric features.
        
        Args:
            df: Feature DataFrame
            fit: Whether to fit scaler (True for training set)
        
        Returns:
            Normalized DataFrame
        """
        df_normalized = df.copy()
        
        # Identify numeric columns (exclude encoded categoricals)
        numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude binary/categorical encoded columns
        numeric_cols = [col for col in numeric_cols if not col.endswith('_encoded')]
        
        if not numeric_cols:
            return df_normalized
        
        if fit:
            df_normalized[numeric_cols] = self.numeric_scaler.fit_transform(df_normalized[numeric_cols])
            logger.info(f"Fitted scaler on {len(numeric_cols)} numeric features")
        else:
            df_normalized[numeric_cols] = self.numeric_scaler.transform(df_normalized[numeric_cols])
        
        return df_normalized
    
    def augment_data(
        self,
        df: pd.DataFrame,
        rare_conditions: List[str] = None
    ) -> pd.DataFrame:
        """
        Augment data with synthetic scenarios for rare conditions.
        
        Args:
            df: Original DataFrame
            rare_conditions: List of rare conditions to augment
        
        Returns:
            Augmented DataFrame
        """
        if rare_conditions is None:
            rare_conditions = ['safety_car', 'extreme_weather']
        
        df_augmented = df.copy()
        
        # Augment safety car scenarios
        if 'safety_car' in rare_conditions:
            sc_samples = df[df['safety_car_active'] == True]
            if len(sc_samples) < len(df) * 0.1:  # Less than 10%
                # Duplicate and slightly modify
                augmented_sc = sc_samples.sample(n=min(len(sc_samples), 100), replace=True, random_state=42)
                augmented_sc['lap_time'] *= np.random.uniform(0.98, 1.02, len(augmented_sc))
                df_augmented = pd.concat([df_augmented, augmented_sc], ignore_index=True)
                logger.info(f"Augmented {len(augmented_sc)} safety car samples")
        
        return df_augmented
    
    def _quality_checks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform data quality checks.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        initial_count = len(df)
        
        # Remove outliers in lap times (Z-score > 3)
        if 'lap_time' in df.columns:
            z_scores = np.abs((df['lap_time'] - df['lap_time'].mean()) / df['lap_time'].std())
            df = df[z_scores < 3.0]
            logger.info(f"Removed {initial_count - len(df)} outliers (Z-score > 3)")
        
        # Remove invalid lap times
        if 'lap_time' in df.columns:
            df = df[(df['lap_time'] >= 60) & (df['lap_time'] <= 150)]
        
        # Check missing value thresholds
        missing_pct = df.isnull().sum() / len(df) * 100
        high_missing = missing_pct[missing_pct > 50]
        if not high_missing.empty:
            logger.warning(f"Columns with >50% missing: {high_missing.to_dict()}")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        df_filled = df.copy()
        
        # Numeric columns: fill with median
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_filled[col].isnull().any():
                df_filled[col].fillna(df_filled[col].median(), inplace=True)
        
        # Categorical columns: fill with mode
        categorical_cols = df_filled.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df_filled[col].isnull().any():
                df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)
        
        return df_filled
    
    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        df_encoded = df.copy()
        
        categorical_cols = ['tire_compound', 'track_name', 'traffic_state']
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                if col not in self.categorical_encoders:
                    self.categorical_encoders[col] = LabelEncoder()
                    df_encoded[f'{col}_encoded'] = self.categorical_encoders[col].fit_transform(
                        df_encoded[col].astype(str)
                    )
                else:
                    df_encoded[f'{col}_encoded'] = self.categorical_encoders[col].transform(
                        df_encoded[col].astype(str)
                    )
                
                # Drop original categorical column
                df_encoded = df_encoded.drop(columns=[col])
        
        return df_encoded
    
    def _create_stratify_key(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> np.ndarray:
        """Create stratification key from multiple columns."""
        stratify_key = df[columns].astype(str).agg('_'.join, axis=1)
        return stratify_key.values
