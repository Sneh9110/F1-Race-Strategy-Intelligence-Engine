# Data Ingestion Guide

Complete guide for the F1 Race Strategy Intelligence Engine data ingestion system.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Setup](#setup)
4. [Configuration](#configuration)
5. [Running Ingestors](#running-ingestors)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)

## Overview

The data ingestion system provides production-ready pipelines for collecting, validating, and storing F1 race data from multiple sources:

- **Timing Data**: Lap times, sector times, positions, gaps
- **Weather Data**: Track/air temperature, humidity, wind, rainfall, forecasts
- **Telemetry Data**: High-frequency car telemetry (10-60 Hz)
- **Historical Data**: Complete race history using FastF1
- **Safety Car Events**: SC/VSC/Red Flag detection and logging

### Key Features

- **Async/Concurrent**: Multiple sources ingested simultaneously
- **Resilient**: Circuit breaker pattern, exponential backoff retry
- **Quality Assured**: Automated validation and anomaly detection
- **Versioned**: Timestamp-based data versioning with rollback
- **Monitored**: Prometheus metrics for observability
- **Tested**: Mock generators for development without live APIs

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Ingestion Orchestrator                      │
│  (Session lifecycle, scheduling, health monitoring)          │
└──────────┬──────────────────────────────────────────────────┘
           │
           ├──> Timing Ingestor ─────┐
           ├──> Weather Ingestor ────┤
           ├──> Telemetry Ingestor ──┼──> BaseIngestor
           ├──> Historical Ingestor ─│    - Circuit Breaker
           └──> Safety Car Ingestor ─┘    - Retry Logic
                                           - Metrics
                       │
                       ▼
            ┌──────────────────────┐
            │   Validation Layer   │
            │  - Pydantic schemas  │
            │  - Custom validators │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │     QA Engine        │
            │  - Schema checks     │
            │  - Range validation  │
            │  - Anomaly detection │
            │  - Quarantine failed │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  Storage Manager     │
            │  - Parquet files     │
            │  - PostgreSQL/TSDB   │
            │  - Versioning        │
            └──────────────────────┘
```

## Setup

### Prerequisites

```bash
Python 3.10+
PostgreSQL 14+ (with TimescaleDB extension)
Required Python packages (see requirements.txt)
```

### Installation

```bash
# Clone repository
cd "F1 Race Strategy Intelligence Engine"

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Initialize database
python scripts/init_db.py
```

### Directory Structure

```
data/
├── raw/          # Raw ingested data
├── processed/    # Validated data
├── features/     # Engineered features
├── metadata/     # Version manifests
├── quarantine/   # Failed records
└── cache/        # Temporary cache
```

## Configuration

### Settings File (`config/settings.py`)

```python
INGESTION_CONFIG = {
    "storage": {
        "base_path": "data",
        "retention_days": 90,
        "format": "parquet",  # parquet, json, csv
        "compression": "snappy"
    },
    "qa": {
        "anomaly_threshold": 3.0,  # Z-score
        "quarantine_failures": True
    },
    "timing": {
        "poll_interval": 1,  # seconds
        "mock_mode": False
    },
    "weather": {
        "poll_interval": 60,
        "api_key": "your_openweathermap_key",
        "mock_mode": False
    },
    "telemetry": {
        "sample_rate": 10,  # Hz
        "buffer_size": 1000,
        "mock_mode": False
    }
}
```

### Database Configuration

```python
DATABASE_URL = "postgresql://user:pass@localhost:5432/f1_strategy"
```

## Running Ingestors

### Command Line Interface

The `scripts/run_ingestion.py` script provides a CLI for all ingestion operations.

#### Live Session

```bash
# Run live race session ingestion
python scripts/run_ingestion.py live \
    --session-name "Monaco GP 2024" \
    --track "Monaco"
```

This starts all real-time ingestors:
- Timing: 1 Hz
- Weather: 60s interval
- Telemetry: 10-60 Hz
- Safety Car: 5s detection

#### Historical Batch

```bash
# Ingest historical data for specific year
python scripts/run_ingestion.py historical --year 2024

# Specific rounds
python scripts/run_ingestion.py historical --year 2024 --rounds 1 5 10
```

#### Test Mode

```bash
# Test individual ingestor with mock data
python scripts/run_ingestion.py test --source timing
python scripts/run_ingestion.py test --source weather
python scripts/run_ingestion.py test --source telemetry
```

#### Health Check

```bash
# Check orchestrator and ingestor health
python scripts/run_ingestion.py health
```

### Programmatic Usage

```python
from data_pipeline.orchestrator import IngestionOrchestrator
from config.settings import Settings

# Initialize
settings = Settings()
orchestrator = IngestionOrchestrator(settings.dict())

# Run live session
session_info = {"name": "Race", "track": "Monaco"}
await orchestrator.run_live_session(session_info)

# Run historical batch
result = await orchestrator.run_historical_batch(year=2024)
```

### Individual Ingestor

```python
from data_pipeline.ingestors.timing_ingestor import TimingIngestor
from data_pipeline.base.storage_manager import StorageManager
from data_pipeline.base.qa_engine import QAEngine

# Initialize components
storage = StorageManager(config)
qa = QAEngine(config)

# Create ingestor
timing = TimingIngestor(storage, qa, {"mock_mode": True})

# Run ingestion
result = await timing.run()

print(f"Ingested {result.records_ingested} records in {result.duration:.2f}s")
```

## Monitoring

### Prometheus Metrics

The ingestion system exports comprehensive metrics:

```python
# Throughput
ingestion_records_total{source="timing", status="success"}
ingestion_bytes_total{source="timing", format="parquet"}

# Latency
ingestion_duration_seconds{source="timing"}
validation_duration_seconds{source="timing"}

# Errors
ingestion_errors_total{source="timing", error_type="network"}
validation_failures_total{source="timing", check_type="range"}

# Quality
data_quality_score{source="timing"}
anomalies_detected_total{source="timing", anomaly_type="outlier"}

# State
active_ingestors{source="timing"}
```

### Grafana Dashboard

Import the provided dashboard (`monitoring/grafana_dashboard.json`):

- Ingestion rates by source
- Error rates and types
- Data quality scores
- Processing latency percentiles
- Storage utilization

### Logs

```python
# Structured logging with context
logger.info("Ingestion started", extra={
    "source": "timing",
    "session_id": "MON_2024_R",
    "lap_number": 42
})
```

Logs include:
- Ingestion lifecycle events
- Validation warnings/failures
- QA anomalies detected
- Circuit breaker state changes
- Storage operations

## Troubleshooting

### Common Issues

#### 1. Circuit Breaker Open

**Symptom**: `CircuitBreakerOpen` errors, ingestion fails immediately

**Cause**: Repeated failures exceeded threshold (default: 3)

**Solution**:
```python
# Check circuit breaker state
health = orchestrator.get_health_status()
print(health["ingestors"]["timing"]["circuit_breaker_state"])

# Reset circuit breaker
ingestor.circuit_breaker.reset()
```

#### 2. QA Failures

**Symptom**: Records quarantined, low quality scores

**Cause**: Data quality issues (out of range, inconsistencies, anomalies)

**Solution**:
```python
# Check quarantine directory
ls data/quarantine/timing/

# Review QA report
qa_report = qa_engine.run_checks(data, source="timing")
print(qa_report.warnings)
print(qa_report.critical_failures)

# Adjust QA thresholds if needed
qa_engine.config["anomaly_threshold"] = 4.0  # Less sensitive
```

#### 3. Storage Issues

**Symptom**: Disk full, slow writes

**Cause**: Retention not cleaning old data, inefficient format

**Solution**:
```bash
# Manual cleanup
python scripts/cleanup_storage.py --days 30 --dry-run

# Check storage usage
du -sh data/raw/*

# Switch to more efficient format
INGESTION_CONFIG["storage"]["format"] = "parquet"
INGESTION_CONFIG["storage"]["compression"] = "snappy"
```

#### 4. Database Connection Errors

**Symptom**: `psycopg2.OperationalError`

**Cause**: Database unreachable, connection pool exhausted

**Solution**:
```python
# Check database connectivity
psql -h localhost -U user -d f1_strategy

# Increase connection pool
DATABASE_CONFIG["pool_size"] = 20
DATABASE_CONFIG["max_overflow"] = 10

# Enable connection pooling
from sqlalchemy.pool import QueuePool
```

#### 5. Mock Mode Not Working

**Symptom**: No data generated in test mode

**Cause**: Mock generators not initialized

**Solution**:
```python
# Ensure mock_mode enabled
config = {"mock_mode": True}
ingestor = TimingIngestor(storage, qa, config)

# Check mock generator
assert ingestor.mock_generator is not None
```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or via environment
export LOG_LEVEL=DEBUG
python scripts/run_ingestion.py test --source timing
```

### Testing

Run full test suite:

```bash
# All tests
pytest tests/test_data_pipeline/

# Specific test file
pytest tests/test_data_pipeline/test_timing_ingestor.py -v

# With coverage
pytest tests/test_data_pipeline/ --cov=data_pipeline --cov-report=html
```

## Best Practices

1. **Use Mock Mode for Development**: Test logic without live APIs
2. **Monitor QA Reports**: Review anomalies and adjust thresholds
3. **Set Appropriate Retention**: Balance storage vs. historical needs
4. **Tune Circuit Breakers**: Adjust thresholds for your reliability needs
5. **Schedule Historical Batches**: Run during off-peak hours
6. **Version Important Datasets**: Tag significant versions for rollback
7. **Alert on Quality Degradation**: Set up alerts for quality score drops

## Support

For issues or questions:
- Review logs in `logs/ingestion.log`
- Check Prometheus metrics dashboard
- Consult `DATA_SCHEMAS.md` for data formats
- Run health checks: `python scripts/run_ingestion.py health`
