"""
Ingestion Metrics - Prometheus metrics for monitoring

Tracks ingestion performance and data quality.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, Summary

from app.utils.logger import get_logger


class IngestionMetrics:
    """
    Centralized metrics for data ingestion monitoring.
    
    Tracks:
    - Ingestion throughput
    - Error rates
    - Data quality scores
    - Processing latency
    - Storage utilization
    """
    
    def __init__(self):
        """Initialize Prometheus metrics."""
        self.logger = get_logger(__name__)
        
        # Throughput metrics
        self.records_ingested = Counter(
            'ingestion_records_total',
            'Total records ingested',
            ['source', 'status']
        )
        
        self.bytes_ingested = Counter(
            'ingestion_bytes_total',
            'Total bytes ingested',
            ['source', 'format']
        )
        
        # Latency metrics
        self.ingestion_duration = Histogram(
            'ingestion_duration_seconds',
            'Ingestion duration',
            ['source'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )
        
        self.validation_duration = Histogram(
            'validation_duration_seconds',
            'Validation duration',
            ['source'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )
        
        # Error metrics
        self.ingestion_errors = Counter(
            'ingestion_errors_total',
            'Total ingestion errors',
            ['source', 'error_type']
        )
        
        self.validation_failures = Counter(
            'validation_failures_total',
            'Total validation failures',
            ['source', 'check_type']
        )
        
        # Quality metrics
        self.data_quality_score = Gauge(
            'data_quality_score',
            'Data quality score (0-100)',
            ['source']
        )
        
        self.anomalies_detected = Counter(
            'anomalies_detected_total',
            'Anomalies detected',
            ['source', 'anomaly_type']
        )
        
        # Active state
        self.active_ingestors = Gauge(
            'active_ingestors',
            'Number of active ingestors',
            ['source']
        )
        
        # Storage metrics
        self.storage_usage = Gauge(
            'storage_usage_bytes',
            'Storage usage',
            ['storage_type', 'source']
        )
        
        self.logger.info("Ingestion metrics initialized")
    
    def record_ingestion(
        self,
        source: str,
        records: int,
        bytes_size: int,
        duration: float,
        status: str = "success"
    ):
        """Record successful ingestion."""
        self.records_ingested.labels(source=source, status=status).inc(records)
        self.bytes_ingested.labels(source=source, format="parquet").inc(bytes_size)
        self.ingestion_duration.labels(source=source).observe(duration)
    
    def record_validation(
        self,
        source: str,
        duration: float,
        failures: int = 0,
        check_type: Optional[str] = None
    ):
        """Record validation metrics."""
        self.validation_duration.labels(source=source).observe(duration)
        
        if failures > 0 and check_type:
            self.validation_failures.labels(source=source, check_type=check_type).inc(failures)
    
    def record_error(self, source: str, error_type: str, count: int = 1):
        """Record ingestion error."""
        self.ingestion_errors.labels(source=source, error_type=error_type).inc(count)
    
    def update_quality_score(self, source: str, score: float):
        """Update data quality score (0-100)."""
        self.data_quality_score.labels(source=source).set(score)
    
    def record_anomaly(self, source: str, anomaly_type: str, count: int = 1):
        """Record detected anomaly."""
        self.anomalies_detected.labels(source=source, anomaly_type=anomaly_type).inc(count)
    
    def set_active_ingestor(self, source: str, active: bool):
        """Set ingestor active state."""
        self.active_ingestors.labels(source=source).set(1 if active else 0)
    
    def update_storage_usage(self, storage_type: str, source: str, bytes_size: int):
        """Update storage utilization."""
        self.storage_usage.labels(storage_type=storage_type, source=source).set(bytes_size)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        # This would typically query Prometheus
        # For now, return basic info
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics_active": True
        }
