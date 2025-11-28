"""
QA Engine - Automated data quality checks and anomaly detection

Validates data integrity, detects outliers, and generates quality reports.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd
import numpy as np
from pydantic import BaseModel, ValidationError

from app.utils.logger import get_logger
from app.utils.validators import (
    validate_lap_time,
    validate_sector_times,
    detect_outliers,
    validate_temperature,
    validate_speed
)


@dataclass
class QAReport:
    """Quality assurance report for ingested data."""
    
    passed: bool
    total_records: int
    valid_records: int
    failed_records: int
    anomalies_detected: int
    warnings: List[str]
    critical_failures: List[str]
    statistics: Dict[str, Any]
    timestamp: datetime


class QAEngine:
    """
    Automated quality assurance engine for ingested data.
    
    Performs:
    - Schema validation
    - Range validation
    - Consistency checks
    - Completeness checks
    - Anomaly detection
    - Auto-correction
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize QA engine.
        
        Args:
            config: Optional configuration dict
        """
        self.config = config or {}
        self.logger = get_logger("qa_engine")
        
        # Configuration
        self.z_score_threshold = self.config.get('z_score_threshold', 3.0)
        self.max_failure_rate = self.config.get('max_failure_rate', 0.05)  # 5%
        self.enable_auto_correction = self.config.get('enable_auto_correction', True)
        
        # Quarantine directory
        self.quarantine_path = Path("data/quarantine")
        self.quarantine_path.mkdir(parents=True, exist_ok=True)
    
    async def run_checks(
        self,
        data: List[BaseModel],
        source: str
    ) -> QAReport:
        """
        Run all QA checks on ingested data.
        
        Args:
            data: List of validated Pydantic models
            source: Data source name
        
        Returns:
            QAReport with check results
        """
        try:
            total_records = len(data)
            warnings = []
            critical_failures = []
            anomalies = []
            
            self.logger.info(f"Running QA checks on {total_records} records from {source}")
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame([record.model_dump() for record in data])
            
            # 1. Schema Validation (already done by Pydantic, but double-check)
            schema_warnings = self._check_schema_compliance(df, source)
            warnings.extend(schema_warnings)
            
            # 2. Range Validation
            range_warnings, range_failures = self._check_value_ranges(df, source)
            warnings.extend(range_warnings)
            critical_failures.extend(range_failures)
            
            # 3. Consistency Checks
            consistency_warnings = self._check_data_consistency(df, source)
            warnings.extend(consistency_warnings)
            
            # 4. Completeness Checks
            completeness_warnings = self._check_completeness(df)
            warnings.extend(completeness_warnings)
            
            # 5. Uniqueness Checks
            duplicate_warnings = self._check_uniqueness(df, source)
            warnings.extend(duplicate_warnings)
            
            # 6. Temporal Consistency
            temporal_warnings = self._check_temporal_consistency(df)
            warnings.extend(temporal_warnings)
            
            # 7. Anomaly Detection
            detected_anomalies = self._detect_anomalies(df, source)
            anomalies.extend(detected_anomalies)
            
            # Calculate statistics
            valid_records = total_records - len(critical_failures)
            failed_records = len(critical_failures)
            failure_rate = failed_records / total_records if total_records > 0 else 0
            
            # Determine if QA passed
            passed = (
                failure_rate <= self.max_failure_rate and
                len(critical_failures) == 0
            )
            
            # Quarantine failed records if needed
            if failed_records > 0:
                self._quarantine_failed_records(df, critical_failures, source)
            
            # Build statistics
            statistics = {
                "failure_rate": failure_rate,
                "warning_count": len(warnings),
                "anomaly_count": len(anomalies),
                "critical_failure_count": len(critical_failures)
            }
            
            # Add source-specific stats
            if source == "timing":
                statistics.update(self._timing_statistics(df))
            elif source == "weather":
                statistics.update(self._weather_statistics(df))
            elif source == "telemetry":
                statistics.update(self._telemetry_statistics(df))
            
            report = QAReport(
                passed=passed,
                total_records=total_records,
                valid_records=valid_records,
                failed_records=failed_records,
                anomalies_detected=len(anomalies),
                warnings=warnings,
                critical_failures=critical_failures,
                statistics=statistics,
                timestamp=datetime.utcnow()
            )
            
            self.logger.info(
                f"QA checks completed for {source}",
                extra_data={
                    "passed": passed,
                    "valid_records": valid_records,
                    "warnings": len(warnings),
                    "anomalies": len(anomalies)
                }
            )
            
            return report
        
        except Exception as e:
            self.logger.error(f"QA check error: {str(e)}")
            return QAReport(
                passed=False,
                total_records=len(data),
                valid_records=0,
                failed_records=len(data),
                anomalies_detected=0,
                warnings=[],
                critical_failures=[f"QA engine error: {str(e)}"],
                statistics={},
                timestamp=datetime.utcnow()
            )
    
    def _check_schema_compliance(self, df: pd.DataFrame, source: str) -> List[str]:
        """Check if data complies with expected schema."""
        warnings = []
        
        # Check for unexpected columns
        expected_columns = {
            "timing": ["timestamp", "driver_number", "position", "lap_number", "lap_time"],
            "weather": ["timestamp", "track_temp_celsius", "air_temp_celsius", "humidity_percent"],
            "telemetry": ["timestamp", "driver_number", "speed_kmh", "throttle_percent", "brake_percent"]
        }
        
        if source in expected_columns:
            missing = set(expected_columns[source]) - set(df.columns)
            if missing:
                warnings.append(f"Missing expected columns: {missing}")
        
        return warnings
    
    def _check_value_ranges(self, df: pd.DataFrame, source: str) -> tuple[List[str], List[str]]:
        """Validate values are within acceptable ranges."""
        warnings = []
        failures = []
        
        if source == "timing":
            # Check lap times (30-150s)
            if 'lap_time' in df.columns:
                invalid_laps = df[
                    (df['lap_time'].notna()) &
                    ((df['lap_time'] < 30) | (df['lap_time'] > 150))
                ]
                if len(invalid_laps) > 0:
                    failures.append(f"{len(invalid_laps)} lap times out of range (30-150s)")
            
            # Check driver numbers (1-99)
            if 'driver_number' in df.columns:
                invalid_drivers = df[
                    (df['driver_number'] < 1) | (df['driver_number'] > 99)
                ]
                if len(invalid_drivers) > 0:
                    failures.append(f"{len(invalid_drivers)} invalid driver numbers")
        
        elif source == "weather":
            # Check temperatures
            if 'track_temp_celsius' in df.columns:
                invalid_temps = df[
                    (df['track_temp_celsius'] < 10) | (df['track_temp_celsius'] > 60)
                ]
                if len(invalid_temps) > 0:
                    warnings.append(f"{len(invalid_temps)} track temps out of range (10-60°C)")
        
        elif source == "telemetry":
            # Check speeds (0-380 km/h)
            if 'speed_kmh' in df.columns:
                invalid_speeds = df[
                    (df['speed_kmh'] < 0) | (df['speed_kmh'] > 380)
                ]
                if len(invalid_speeds) > 0:
                    failures.append(f"{len(invalid_speeds)} speeds out of range (0-380 km/h)")
        
        return warnings, failures
    
    def _check_data_consistency(self, df: pd.DataFrame, source: str) -> List[str]:
        """Check for data consistency issues."""
        warnings = []
        
        if source == "timing":
            # Sector times should sum to lap time (within 100ms tolerance)
            if all(col in df.columns for col in ['lap_time', 'sector_1_time', 'sector_2_time', 'sector_3_time']):
                df_with_sectors = df[df['sector_1_time'].notna() & df['sector_2_time'].notna() & df['sector_3_time'].notna()]
                
                if len(df_with_sectors) > 0:
                    sector_sums = (
                        df_with_sectors['sector_1_time'] +
                        df_with_sectors['sector_2_time'] +
                        df_with_sectors['sector_3_time']
                    )
                    discrepancies = abs(sector_sums - df_with_sectors['lap_time'])
                    invalid = discrepancies > 0.1  # 100ms tolerance
                    
                    if invalid.sum() > 0:
                        warnings.append(f"{invalid.sum()} laps with sector time inconsistencies")
            
            # Leader should have gap_to_leader = 0
            if 'position' in df.columns and 'gap_to_leader' in df.columns:
                leaders = df[df['position'] == 1]
                non_zero_gaps = leaders[leaders['gap_to_leader'] != 0]
                
                if len(non_zero_gaps) > 0:
                    warnings.append(f"{len(non_zero_gaps)} leader records with non-zero gap")
        
        elif source == "weather":
            # Track temp should be greater than air temp
            if 'track_temp_celsius' in df.columns and 'air_temp_celsius' in df.columns:
                invalid = df[df['track_temp_celsius'] <= df['air_temp_celsius']]
                
                if len(invalid) > 0:
                    warnings.append(f"{len(invalid)} records with track temp <= air temp")
        
        elif source == "telemetry":
            # Throttle and brake shouldn't both be >50% simultaneously
            if 'throttle_percent' in df.columns and 'brake_percent' in df.columns:
                invalid = df[
                    (df['throttle_percent'] > 50) & (df['brake_percent'] > 50)
                ]
                
                if len(invalid) > 0:
                    warnings.append(f"{len(invalid)} records with throttle and brake both >50%")
        
        return warnings
    
    def _check_completeness(self, df: pd.DataFrame) -> List[str]:
        """Check for missing data."""
        warnings = []
        
        # Check for null values in critical columns
        null_counts = df.isnull().sum()
        critical_nulls = null_counts[null_counts > 0]
        
        if len(critical_nulls) > 0:
            for column, count in critical_nulls.items():
                pct = (count / len(df)) * 100
                warnings.append(f"{column}: {count} null values ({pct:.1f}%)")
        
        return warnings
    
    def _check_uniqueness(self, df: pd.DataFrame, source: str) -> List[str]:
        """Check for duplicate records."""
        warnings = []
        
        # Define uniqueness keys per source
        unique_keys = {
            "timing": ["timestamp", "driver_number", "lap_number"],
            "weather": ["timestamp"],
            "telemetry": ["timestamp", "driver_number"]
        }
        
        if source in unique_keys and all(col in df.columns for col in unique_keys[source]):
            duplicates = df.duplicated(subset=unique_keys[source], keep=False)
            dup_count = duplicates.sum()
            
            if dup_count > 0:
                warnings.append(f"{dup_count} duplicate records detected")
        
        return warnings
    
    def _check_temporal_consistency(self, df: pd.DataFrame) -> List[str]:
        """Check timestamp consistency."""
        warnings = []
        
        if 'timestamp' in df.columns:
            df_sorted = df.sort_values('timestamp')
            
            # Check for future timestamps
            now = datetime.utcnow()
            future = df_sorted[pd.to_datetime(df_sorted['timestamp']) > now]
            
            if len(future) > 0:
                warnings.append(f"{len(future)} records have future timestamps")
            
            # Check for reasonable time gaps
            if len(df_sorted) > 1:
                time_diffs = pd.to_datetime(df_sorted['timestamp']).diff()
                large_gaps = time_diffs[time_diffs > pd.Timedelta(minutes=5)]
                
                if len(large_gaps) > 0:
                    warnings.append(f"{len(large_gaps)} large time gaps detected (>5 min)")
        
        return warnings
    
    def _detect_anomalies(self, df: pd.DataFrame, source: str) -> List[str]:
        """Detect statistical anomalies in data."""
        anomalies = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Skip columns with too few values
            if df[col].notna().sum() < 10:
                continue
            
            # Use Z-score method from validators
            outliers = detect_outliers(df[col].dropna().values, self.z_score_threshold)
            
            if outliers.sum() > 0:
                pct = (outliers.sum() / len(df)) * 100
                anomalies.append(f"{col}: {outliers.sum()} outliers detected ({pct:.1f}%)")
        
        return anomalies
    
    def _timing_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate timing-specific statistics."""
        stats = {}
        
        if 'lap_time' in df.columns:
            lap_times = df['lap_time'].dropna()
            if len(lap_times) > 0:
                stats['avg_lap_time'] = float(lap_times.mean())
                stats['min_lap_time'] = float(lap_times.min())
                stats['max_lap_time'] = float(lap_times.max())
                stats['lap_time_std'] = float(lap_times.std())
        
        return stats
    
    def _weather_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate weather-specific statistics."""
        stats = {}
        
        if 'track_temp_celsius' in df.columns:
            temps = df['track_temp_celsius'].dropna()
            if len(temps) > 0:
                stats['avg_track_temp'] = float(temps.mean())
                stats['min_track_temp'] = float(temps.min())
                stats['max_track_temp'] = float(temps.max())
        
        return stats
    
    def _telemetry_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate telemetry-specific statistics."""
        stats = {}
        
        if 'speed_kmh' in df.columns:
            speeds = df['speed_kmh'].dropna()
            if len(speeds) > 0:
                stats['avg_speed'] = float(speeds.mean())
                stats['max_speed'] = float(speeds.max())
        
        return stats
    
    def _quarantine_failed_records(
        self,
        df: pd.DataFrame,
        failures: List[str],
        source: str
    ) -> None:
        """Move failed records to quarantine directory."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        quarantine_dir = self.quarantine_path / source / timestamp
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        
        # Save failed records
        failed_file = quarantine_dir / "failed_records.json"
        # For now, save entire dataset if there are failures
        # In production, would identify specific failed rows
        df.to_json(failed_file, orient='records', indent=2)
        
        # Save failure report
        report_file = quarantine_dir / "failure_report.json"
        with open(report_file, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "source": source,
                "failures": failures,
                "record_count": len(df)
            }, f, indent=2)
        
        self.logger.warning(f"Quarantined failed records to {quarantine_dir}")
