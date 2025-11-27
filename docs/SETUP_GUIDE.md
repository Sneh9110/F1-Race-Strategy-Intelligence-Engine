# F1 Race Strategy Intelligence Engine - Setup Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Database Setup](#database-setup)
4. [Configuration](#configuration)
5. [Data Download](#data-download)
6. [Running the Application](#running-the-application)
7. [Development Workflow](#development-workflow)
8. [Testing](#testing)
9. [Troubleshooting](#troubleshooting)
10. [Docker Deployment](#docker-deployment)

---

## Prerequisites

### Required Software

- **Python 3.10 or higher**
  - Check version: `python --version`
  - Install from: https://www.python.org/downloads/

- **PostgreSQL 14 or higher**
  - Check version: `psql --version`
  - Install from: https://www.postgresql.org/download/

- **Redis 7 or higher**
  - Check version: `redis-cli --version`
  - Install from: https://redis.io/download

- **Git**
  - For version control and cloning the repository
  - Install from: https://git-scm.com/downloads

### Optional (for Docker deployment)

- **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)
  - Install from: https://docs.docker.com/get-docker/
- **Docker Compose**
  - Usually included with Docker Desktop

### System Requirements

- **Minimum**: 8GB RAM, 4 CPU cores, 20GB disk space
- **Recommended**: 16GB RAM, 8 CPU cores, 50GB disk space (for historical data)
- **Operating System**: Linux, macOS, or Windows 10+

---

## Installation

### 1. Clone or Navigate to Repository

```bash
# If repository is already set up locally
cd "d:\PROGRAMINGGGGGGG\F1 Race Strategy Intelligence Engine"

# If cloning from remote
git clone <repository-url>
cd f1-strategy-engine
```

### 2. Create Virtual Environment

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

You should see `(venv)` in your terminal prompt.

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# For development (includes testing/linting tools)
pip install -r requirements-dev.txt
```

**Expected installation time**: 3-5 minutes depending on internet speed.

### 4. Verify Installation

```bash
# Check Python packages
pip list | grep -E "fastapi|pydantic|pandas|xgboost"

# Should see:
# fastapi             0.104.0
# pydantic            2.5.0
# pandas              2.1.0
# xgboost             2.0.0
```

---

## Database Setup

### 1. Start PostgreSQL Service

**Linux (systemd):**
```bash
sudo systemctl start postgresql
sudo systemctl enable postgresql  # Start on boot
```

**Mac (Homebrew):**
```bash
brew services start postgresql@14
```

**Windows:**
- Start PostgreSQL service from Services app or pgAdmin

### 2. Create Database and User

```bash
# Connect to PostgreSQL as superuser
psql -U postgres

# In psql prompt:
CREATE DATABASE f1_strategy;
CREATE USER f1_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE f1_strategy TO f1_user;

# Enable TimescaleDB extension (important!)
\c f1_strategy
CREATE EXTENSION IF NOT EXISTS timescaledb;

# Exit psql
\q
```

### 3. Run Database Setup Script

```bash
# Make sure virtual environment is activated
python scripts/setup_db.py
```

**Expected output:**
```
[INFO] Creating database f1_strategy...
[INFO] Database already exists
[INFO] Creating schema and tables...
[INFO] ✓ Created table: timing_data
[INFO] ✓ Created hypertable: timing_data (partitioned by timestamp)
[INFO] ✓ Created index: idx_timing_driver_lap
[INFO] ✓ Created table: weather_data
[INFO] ✓ Created hypertable: weather_data
[INFO] ✓ Created table: telemetry_data
[INFO] ✓ Created hypertable: telemetry_data
[INFO] ✓ Created table: historical_races
[INFO] ✓ Created index: idx_historical_driver_race
[INFO] ✓ Created table: safety_car_events
[INFO] ✓ Created table: model_predictions
[INFO] ✓ Created table: strategy_recommendations
[SUCCESS] Database setup complete!
```

### 4. Verify Database Setup

```bash
# Connect to database
psql -U f1_user -d f1_strategy

# List tables
\dt

# Should see:
# timing_data, weather_data, telemetry_data, historical_races,
# safety_car_events, model_predictions, strategy_recommendations

# Check TimescaleDB hypertables
SELECT * FROM timescaledb_information.hypertables;

# Exit
\q
```

---

## Configuration

### 1. Create Environment File

```bash
# Copy example environment file
cp .env.example .env
```

### 2. Edit `.env` File

Open `.env` in your text editor and configure:

```ini
# Environment (development, staging, production)
ENVIRONMENT=development

# Database Configuration
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=f1_strategy
DATABASE_USER=f1_user
DATABASE_PASSWORD=your_secure_password
DATABASE_POOL_SIZE=20

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=  # Leave empty if no password

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true  # Auto-reload on code changes (dev only)
API_WORKERS=4
API_LOG_LEVEL=info

# ML Model Configuration
MODEL_REGISTRY_PATH=./models/registry
MODEL_RETRAIN_INTERVAL_HOURS=24
MODEL_MIN_SAMPLES=1000

# Data Pipeline Configuration
DATA_INGEST_INTERVAL_SECONDS=5
DATA_BATCH_SIZE=100
DATA_MAX_RETRY_ATTEMPTS=3

# Logging Configuration
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE_PATH=./logs/app.log
LOG_ROTATION_SIZE_MB=100
LOG_BACKUP_COUNT=10

# Monitoring (Optional)
SENTRY_DSN=  # Add Sentry DSN for error tracking
PROMETHEUS_PORT=9090
```

### 3. Environment-Specific Configuration

The application uses YAML files for environment-specific settings:

- **Development**: `config/development.yaml` (debug enabled, mock data)
- **Staging**: `config/staging.yaml` (real data, monitoring)
- **Production**: `config/production.yaml` (HA cluster, strict security)

Set `ENVIRONMENT=development` in `.env` to use development config.

### 4. Verify Configuration

```bash
# Test configuration loading
python -c "from config.settings import get_settings; print(get_settings())"
```

Should print configuration without errors.

---

## Data Download

### 1. Start Redis Service

**Linux:**
```bash
sudo systemctl start redis
```

**Mac:**
```bash
brew services start redis
```

**Windows:**
- Start Redis service or run `redis-server.exe`

### 2. Download Historical F1 Data

```bash
# Download data for recent seasons (2020-2024)
python scripts/download_historical_data.py --start-year 2020 --end-year 2024

# For specific tracks only
python scripts/download_historical_data.py --start-year 2023 --tracks Monaco Silverstone Spa

# Full historical data (2018-2024, ~2GB)
python scripts/download_historical_data.py --start-year 2018 --end-year 2024 --include-telemetry
```

**Expected output:**
```
[INFO] Downloading F1 historical data...
[INFO] Fetching 2020 season...
[INFO] ✓ Downloaded 17 races for 2020
[INFO] Fetching 2021 season...
[INFO] ✓ Downloaded 22 races for 2021
[INFO] Fetching 2022 season...
[INFO] ✓ Downloaded 22 races for 2022
[INFO] Fetching 2023 season...
[INFO] ✓ Downloaded 23 races for 2023
[INFO] Fetching 2024 season...
[INFO] ✓ Downloaded 15 races for 2024 (ongoing season)
[SUCCESS] Downloaded 99 races, 1980 driver results
[INFO] Saved to: data/raw/historical/
```

**Download time**: 10-30 minutes depending on data range and internet speed.

### 3. Verify Data Download

```bash
# Check downloaded files
ls -lh data/raw/historical/

# Should see:
# races_2020.json, races_2021.json, ..., lap_times/, pit_stops/
```

---

## Running the Application

### 1. Start Backend API

```bash
# Make sure venv is activated and in project root
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Expected output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using StatReload
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 2. Access API Documentation

Open browser to:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### 3. Start Frontend (Optional)

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

**Expected output:**
```
VITE v4.5.0  ready in 1234 ms

➜  Local:   http://localhost:3000/
➜  Network: http://192.168.1.100:3000/
```

Open browser to http://localhost:3000

### 4. Run Background Workers (Optional)

For production, run data ingestion workers:

```bash
# Terminal 1: Data ingestion worker
python -m data_pipeline.workers.ingestion_worker

# Terminal 2: Model training worker
python -m models.workers.training_worker

# Terminal 3: Strategy computation worker
python -m decision_engine.workers.strategy_worker
```

---

## Development Workflow

### Code Formatting

```bash
# Format code with black
black .

# Sort imports
isort .
```

### Linting

```bash
# Run linting checks
ruff check .

# Auto-fix issues
ruff check . --fix

# Type checking
mypy app/ data_pipeline/ models/
```

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "Add new table"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

### Adding New Dependencies

```bash
# Install package
pip install <package-name>

# Update requirements.txt
pip freeze > requirements.txt
```

---

## Testing

### Run All Tests

```bash
# Run full test suite
pytest

# With coverage report
pytest --cov=app --cov=data_pipeline --cov=models --cov-report=html

# Open coverage report
open htmlcov/index.html  # Mac/Linux
start htmlcov/index.html  # Windows
```

### Run Specific Tests

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Specific test file
pytest tests/unit/test_timing_schema.py

# Specific test function
pytest tests/unit/test_timing_schema.py::test_lap_data_validation
```

### Test Database Setup

```bash
# Create test database
createdb f1_strategy_test

# Run tests with test database
TEST_DATABASE_URL=postgresql://f1_user:password@localhost/f1_strategy_test pytest
```

---

## Troubleshooting

### Common Issues

#### Issue: `psycopg2` installation fails

**Solution (Linux/Mac):**
```bash
# Install PostgreSQL development headers
sudo apt-get install libpq-dev python3-dev  # Ubuntu/Debian
brew install postgresql  # Mac

# Reinstall
pip install psycopg2-binary
```

**Solution (Windows):**
```bash
# Use binary wheel instead
pip uninstall psycopg2
pip install psycopg2-binary
```

#### Issue: TimescaleDB extension not found

**Solution:**
```bash
# Install TimescaleDB
# Ubuntu/Debian:
sudo add-apt-repository ppa:timescale/timescaledb-ppa
sudo apt update
sudo apt install timescaledb-2-postgresql-14

# Mac:
brew install timescaledb

# Run setup script
sudo timescaledb-tune

# Restart PostgreSQL
sudo systemctl restart postgresql
```

#### Issue: Redis connection refused

**Solution:**
```bash
# Check Redis is running
redis-cli ping  # Should return "PONG"

# If not running, start Redis
sudo systemctl start redis  # Linux
brew services start redis  # Mac

# Check Redis configuration
redis-cli config get bind  # Should include 127.0.0.1
```

#### Issue: Port 8000 already in use

**Solution:**
```bash
# Find process using port 8000
lsof -i :8000  # Mac/Linux
netstat -ano | findstr :8000  # Windows

# Kill process or use different port
uvicorn api.main:app --port 8001
```

#### Issue: Module not found errors

**Solution:**
```bash
# Ensure virtual environment is activated
# Reinstall dependencies
pip install -r requirements.txt

# Verify Python path
python -c "import sys; print('\n'.join(sys.path))"

# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
$env:PYTHONPATH="$(pwd)"  # Windows PowerShell
```

#### Issue: Database connection timeout

**Solution:**
```bash
# Check PostgreSQL is accepting connections
psql -U f1_user -d f1_strategy -h localhost

# Check pg_hba.conf allows local connections
sudo nano /etc/postgresql/14/main/pg_hba.conf

# Add line:
# local   all   all   md5

# Restart PostgreSQL
sudo systemctl restart postgresql
```

### Logs

Check logs for detailed error information:

```bash
# Application logs
tail -f logs/app.log

# PostgreSQL logs (Linux)
sudo tail -f /var/log/postgresql/postgresql-14-main.log

# Redis logs (Linux)
sudo tail -f /var/log/redis/redis-server.log
```

---

## Docker Deployment

### Quick Start with Docker Compose

```bash
# Build and start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Services Included

- **API**: FastAPI backend (port 8000)
- **PostgreSQL**: TimescaleDB-enabled (port 5432)
- **Redis**: Cache layer (port 6379)
- **Prometheus**: Metrics (port 9090)
- **Grafana**: Dashboards (port 3001)

### Production Deployment

```bash
# Build production image
docker build -t f1-strategy-api:latest .

# Run with production settings
docker run -d \
  --name f1-api \
  -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e DATABASE_HOST=prod-db.example.com \
  -v $(pwd)/config:/app/config:ro \
  f1-strategy-api:latest
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment
kubectl get pods -n f1-strategy

# View logs
kubectl logs -f deployment/f1-api -n f1-strategy
```

---

## Next Steps

After setup is complete:

1. **Read Architecture**: Review `docs/architecture/system_overview.md`
2. **Explore Schemas**: Check `docs/schemas/data_schemas.md`
3. **Try Examples**: Run code from `docs/schemas/schema_examples.md`
4. **Train Models**: Follow `docs/models/training_guide.md`
5. **API Integration**: See `docs/api/integration_guide.md`

## Support

- **Issues**: Report bugs on GitHub Issues
- **Documentation**: Full docs at `docs/`
- **Contributing**: See `CONTRIBUTING.md`
- **License**: MIT License (see `LICENSE`)

---

## Quick Reference

### Essential Commands

```bash
# Start services
sudo systemctl start postgresql redis  # Linux
brew services start postgresql@14 redis  # Mac

# Activate environment
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\Activate.ps1  # Windows

# Run application
uvicorn api.main:app --reload

# Run tests
pytest

# Format code
black . && isort .

# Database migration
alembic upgrade head
```

### Directory Structure

```
f1-strategy-engine/
├── app/                    # Core application
├── data_pipeline/          # Data ingestion and schemas
├── models/                 # ML models
├── simulation/             # Race simulation
├── decision_engine/        # Strategy decisions
├── api/                    # REST API
├── frontend/               # React dashboard
├── scripts/                # Setup and utilities
├── config/                 # Configuration files
├── data/                   # Data storage
├── tests/                  # Test suite
├── docs/                   # Documentation
└── requirements.txt        # Python dependencies
```

**Setup complete! 🏁 Ready to optimize F1 race strategies.**
