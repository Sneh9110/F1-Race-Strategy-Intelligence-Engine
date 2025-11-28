"""
Base Ingestor - Abstract base class for all data ingestors

Provides common functionality: validation, storage, error handling, metrics collection.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from datetime import datetime
from dataclasses import dataclass
import time
import asyncio
from functools import wraps

from pydantic import BaseModel, ValidationError
from prometheus_client import Counter, Histogram, Gauge

from app.utils.logger import get_logger
from app.utils.validators import validate_driver_number
from config.settings import get_settings


@dataclass
class IngestionResult:
    """Result of an ingestion operation."""
    
    success: bool
    records_ingested: int
    records_failed: int
    duration_seconds: float
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]
    timestamp: datetime


class CircuitBreaker:
    """Circuit breaker pattern to prevent cascading failures."""
    
    def __init__(self, failure_threshold: int = 3, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half_open
    
    def call(self, func):
        """Decorator to wrap function with circuit breaker."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == "open":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "half_open"
                else:
                    raise Exception("Circuit breaker is OPEN - service unavailable")
            
            try:
                result = func(*args, **kwargs)
                if self.state == "half_open":
                    self.state = "closed"
                    self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                
                raise e
        
        return wrapper


class BaseIngestor(ABC):
    """
    Abstract base class for all data ingestors.
    
    Provides common ingestion functionality:
    - Data fetching (abstract method)
    - Pydantic validation
    - QA engine integration
    - Storage management
    - Error handling with retry logic
    - Metrics collection
    - Logging with correlation IDs
    """
    
    # Prometheus metrics (class-level, shared across instances)
    ingestion_records = Counter(
        'ingestion_records_total',
        'Total records ingested',
        ['source', 'status']
    )
    ingestion_duration = Histogram(
        'ingestion_duration_seconds',
        'Time spent ingesting data',
        ['source']
    )
    ingestion_errors = Counter(
        'ingestion_errors_total',
        'Total ingestion errors',
        ['source', 'error_type']
    )
    active_ingestions = Gauge(
        'ingestion_active',
        'Number of active ingestions',
        ['source']
    )
    
    def __init__(
        self,
        source_name: str,
        schema_class: Type[BaseModel],
        storage_manager: 'StorageManager',
        qa_engine: 'QAEngine',
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize base ingestor.
        
        Args:
            source_name: Name of data source (timing, weather, telemetry, etc.)
            schema_class: Pydantic model for validation
            storage_manager: Storage manager instance
            qa_engine: QA engine instance
            config: Optional configuration dict
        """
        self.source_name = source_name
        self.schema_class = schema_class
        self.storage_manager = storage_manager
        self.qa_engine = qa_engine
        self.config = config or {}
        
        # Load settings
        self.settings = get_settings()
        
        # Initialize logger
        self.logger = get_logger(f"ingestor.{source_name}")
        
        # Circuit breaker for API calls
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout=60
        )
        
        # Retry configuration
        self.max_retries = self.settings.data_pipeline.max_retry_attempts
        self.retry_backoff_base = self.settings.data_pipeline.retry_backoff_seconds
        
        # Mock mode configuration
        self.mock_mode = self.config.get('mock_mode', False)
        
        # State tracking
        self.is_running = False
        self.last_ingestion_time: Optional[datetime] = None
        self.cached_data: Optional[Any] = None
        
        self.logger.info(
            f"Initialized {source_name} ingestor",
            extra_data={
                "mock_mode": self.mock_mode,
                "schema": schema_class.__name__
            }
        )
    
    @abstractmethod
    async def ingest(self) -> Any:
        """
        Fetch data from source (abstract method).
        
        Subclasses must implement this method to fetch data from their specific source.
        
        Returns:
            Raw data from source (will be validated against schema)
        
        Raises:
            Exception: If data fetching fails
        """
        pass
    
    def validate(self, raw_data: Any) -> List[BaseModel]:
        """
        Validate raw data against Pydantic schema.
        
        Args:
            raw_data: Raw data to validate
        
        Returns:
            List of validated Pydantic model instances
        
        Raises:
            ValidationError: If data doesn't match schema
        """
        validated_records = []
        errors = []
        
        # Handle single record vs list of records
        data_list = raw_data if isinstance(raw_data, list) else [raw_data]
        
        for idx, record in enumerate(data_list):
            try:
                # Validate using Pydantic model
                validated = self.schema_class(**record) if isinstance(record, dict) else record
                validated_records.append(validated)
            except ValidationError as e:
                error_msg = f"Validation error at index {idx}: {str(e)}"
                errors.append(error_msg)
                self.logger.warning(error_msg, extra_data={"record": record})
                self.ingestion_errors.labels(
                    source=self.source_name,
                    error_type="validation"
                ).inc()
        
        if errors and len(errors) == len(data_list):
            raise ValidationError(f"All records failed validation: {errors}")
        
        self.logger.info(
            f"Validated {len(validated_records)}/{len(data_list)} records",
            extra_data={
                "success_rate": len(validated_records) / len(data_list) if data_list else 0
            }
        )
        
        return validated_records
    
    async def store(self, validated_data: List[BaseModel]) -> None:
        """
        Store validated data using storage manager.
        
        Args:
            validated_data: List of validated Pydantic models
        """
        try:
            # Convert to dict for storage
            records = [record.model_dump() for record in validated_data]
            
            # Save raw data
            await self.storage_manager.save_raw(
                source=self.source_name,
                data=records,
                metadata={
                    "ingestion_time": datetime.utcnow().isoformat(),
                    "record_count": len(records),
                    "schema_version": getattr(self.schema_class, '__version__', '1.0.0')
                }
            )
            
            self.logger.info(f"Stored {len(records)} records to storage")
        except Exception as e:
            self.logger.error(f"Storage error: {str(e)}")
            self.ingestion_errors.labels(
                source=self.source_name,
                error_type="storage"
            ).inc()
            raise
    
    async def run_with_retry(self, max_retries: Optional[int] = None) -> IngestionResult:
        """
        Run ingestion with retry logic and exponential backoff.
        
        Args:
            max_retries: Maximum retry attempts (uses config default if None)
        
        Returns:
            IngestionResult with operation summary
        """
        retries = max_retries if max_retries is not None else self.max_retries
        last_exception = None
        
        for attempt in range(retries + 1):
            try:
                result = await self.run()
                return result
            except Exception as e:
                last_exception = e
                
                if attempt < retries:
                    delay = self.retry_backoff_base * (2 ** attempt)
                    self.logger.warning(
                        f"Ingestion attempt {attempt + 1} failed, retrying in {delay}s",
                        extra_data={"error": str(e)}
                    )
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(
                        f"Ingestion failed after {retries + 1} attempts",
                        extra_data={"error": str(e)}
                    )
        
        # All retries failed
        return IngestionResult(
            success=False,
            records_ingested=0,
            records_failed=0,
            duration_seconds=0.0,
            errors=[str(last_exception)],
            warnings=[],
            metadata={},
            timestamp=datetime.utcnow()
        )
    
    async def run(self) -> IngestionResult:
        """
        Execute full ingestion pipeline: ingest → validate → QA → store.
        
        Returns:
            IngestionResult with operation summary
        """
        start_time = time.time()
        errors = []
        warnings = []
        records_ingested = 0
        records_failed = 0
        
        self.active_ingestions.labels(source=self.source_name).inc()
        
        try:
            self.is_running = True
            self.logger.info(f"Starting {self.source_name} ingestion")
            
            # Step 1: Ingest data
            try:
                raw_data = await self.ingest()
                if raw_data is None:
                    warnings.append("No data returned from source")
                    # Try fallback to cache
                    if self.cached_data:
                        warnings.append("Using cached data")
                        raw_data = self.cached_data
                    else:
                        raise Exception("No data available and no cache")
            except Exception as e:
                error_msg = f"Ingestion error: {str(e)}"
                errors.append(error_msg)
                self.logger.error(error_msg)
                self.ingestion_errors.labels(
                    source=self.source_name,
                    error_type="ingestion"
                ).inc()
                raise
            
            # Step 2: Validate data
            try:
                validated_data = self.validate(raw_data)
                records_ingested = len(validated_data)
                
                # Cache successful data for fallback
                self.cached_data = raw_data
            except ValidationError as e:
                error_msg = f"Validation error: {str(e)}"
                errors.append(error_msg)
                records_failed = len(raw_data) if isinstance(raw_data, list) else 1
                raise
            
            # Step 3: Run QA checks
            try:
                qa_report = await self.qa_engine.run_checks(
                    data=validated_data,
                    source=self.source_name
                )
                
                if not qa_report.passed:
                    warnings.extend(qa_report.warnings)
                    if qa_report.critical_failures:
                        errors.extend(qa_report.critical_failures)
                        raise Exception(f"Critical QA failures: {qa_report.critical_failures}")
            except Exception as e:
                warnings.append(f"QA check error: {str(e)}")
                # Don't fail ingestion on QA errors, just log
            
            # Step 4: Store data
            try:
                await self.store(validated_data)
            except Exception as e:
                error_msg = f"Storage error: {str(e)}"
                errors.append(error_msg)
                raise
            
            # Update metrics
            self.ingestion_records.labels(
                source=self.source_name,
                status="success"
            ).inc(records_ingested)
            
            self.last_ingestion_time = datetime.utcnow()
            
            duration = time.time() - start_time
            self.ingestion_duration.labels(source=self.source_name).observe(duration)
            
            self.logger.info(
                f"Completed {self.source_name} ingestion",
                extra_data={
                    "records": records_ingested,
                    "duration": f"{duration:.2f}s",
                    "warnings": len(warnings)
                }
            )
            
            return IngestionResult(
                success=True,
                records_ingested=records_ingested,
                records_failed=records_failed,
                duration_seconds=duration,
                errors=errors,
                warnings=warnings,
                metadata={
                    "last_ingestion_time": self.last_ingestion_time.isoformat(),
                    "source": self.source_name
                },
                timestamp=datetime.utcnow()
            )
        
        except Exception as e:
            duration = time.time() - start_time
            self.ingestion_records.labels(
                source=self.source_name,
                status="failure"
            ).inc()
            
            return IngestionResult(
                success=False,
                records_ingested=records_ingested,
                records_failed=records_failed,
                duration_seconds=duration,
                errors=errors + [str(e)],
                warnings=warnings,
                metadata={"source": self.source_name},
                timestamp=datetime.utcnow()
            )
        
        finally:
            self.is_running = False
            self.active_ingestions.labels(source=self.source_name).dec()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current ingestor status."""
        return {
            "source": self.source_name,
            "is_running": self.is_running,
            "last_ingestion_time": self.last_ingestion_time.isoformat() if self.last_ingestion_time else None,
            "mock_mode": self.mock_mode,
            "has_cached_data": self.cached_data is not None,
            "circuit_breaker_state": self.circuit_breaker.state
        }
