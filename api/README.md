# F1 Race Strategy Intelligence API

Production-grade REST API for F1 race strategy predictions and simulations.

## Features

- **Prediction Endpoints**: Lap times, tire degradation, safety car probability, pit stop loss
- **Simulation Engine**: Race strategy simulation with Monte Carlo analysis
- **Decision Engine**: Real-time strategy recommendations using 7 decision modules
- **Authentication**: JWT tokens + API key support with role-based access
- **Rate Limiting**: Redis-based sliding window (60/min predictions, 10/min simulations)
- **Performance**: <200ms latency for predictions, <5s for simulations
- **Monitoring**: Prometheus metrics, health checks, structured logging
- **Documentation**: Auto-generated OpenAPI/Swagger UI

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (optional)
export API_HOST=0.0.0.0
export API_PORT=8000
export SECRET_KEY=your-secret-key-here
export REDIS_HOST=localhost
export REDIS_PORT=6379
```

### Running the Server

```bash
# Development mode (with auto-reload)
python -m api.run

# Production mode with uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment

```bash
# Build image
docker build -t f1-strategy-api .

# Run container
docker run -p 8000:8000 -e REDIS_HOST=redis f1-strategy-api
```

## API Endpoints

### Authentication

- `POST /api/v1/auth/token` - Generate JWT access token
- `POST /api/v1/auth/refresh` - Refresh token
- `GET /api/v1/auth/me` - Get current user info

### Health & Monitoring

- `GET /api/v1/health` - Health check with component status
- `GET /api/v1/health/ready` - Readiness probe (K8s)
- `GET /api/v1/health/live` - Liveness probe (K8s)
- `GET /api/v1/metrics` - Prometheus metrics

### Predictions (Coming Soon)

- `POST /api/v1/predict/laptime` - Predict lap time
- `POST /api/v1/predict/degradation` - Predict tire degradation
- `POST /api/v1/predict/safety-car` - Predict safety car probability
- `POST /api/v1/predict/pit-stop-loss` - Predict pit stop time loss

### Simulation (Coming Soon)

- `POST /api/v1/simulate/strategy` - Run race strategy simulation
- `POST /api/v1/simulate/compare` - Compare multiple strategies
- `POST /api/v1/simulate/monte-carlo` - Monte Carlo simulation
- `POST /api/v1/simulate/what-if` - What-if scenario analysis

### Strategy Recommendations (Coming Soon)

- `POST /api/v1/strategy/recommend` - Get strategy recommendation
- `GET /api/v1/strategy/modules` - List decision modules

## Authentication

### JWT Token

```bash
# Get token
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123"

# Use token
curl http://localhost:8000/api/v1/health \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### API Key

```bash
# Use API key
curl http://localhost:8000/api/v1/health \
  -H "X-API-Key: test_key_12345"
```

## Configuration

Environment variables:

```env
# Server
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_RELOAD=False

# Security
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Rate Limits (requests per minute)
RATE_LIMIT_PREDICTIONS=60
RATE_LIMIT_SIMULATIONS=10
RATE_LIMIT_ADMIN=5

# Caching (seconds)
CACHE_TTL_PREDICTIONS=60
CACHE_TTL_SIMULATIONS=300
CACHE_TTL_RACE_STATE=5
```

## Performance Targets

- **Predictions**: <200ms response time
- **Simulations**: <5s response time
- **Race State Queries**: <100ms response time
- **Health Checks**: <50ms response time
- **Throughput**: 100+ requests/second

## Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=api tests/

# Run load tests
locust -f tests/load/locustfile.py --host http://localhost:8000
```

## Documentation

Interactive API documentation available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Architecture

```
api/
├── main.py              # FastAPI application
├── config.py            # Configuration
├── dependencies.py      # Dependency injection
├── auth.py              # Authentication logic
├── routers/
│   └── v1/
│       ├── auth.py      # Auth endpoints
│       ├── health.py    # Health/monitoring
│       ├── predictions.py   # ML predictions (TODO)
│       ├── simulation.py    # Race simulation (TODO)
│       └── strategy.py      # Strategy recommendations (TODO)
├── schemas/
│   ├── common.py        # Common types
│   └── predictions.py   # Prediction schemas
├── middleware/          # Custom middleware (TODO)
├── utils/               # Utility functions (TODO)
└── tests/               # Test suite (TODO)
```

## Status

**Current Version**: 0.1.0-alpha

✅ Completed:
- Core FastAPI application
- Authentication (JWT + API key)
- Health monitoring endpoints
- Configuration management
- Basic error handling
- OpenAPI documentation

⏳ In Progress:
- Rate limiting middleware
- Prediction endpoints
- Simulation endpoints
- Strategy recommendation endpoints
- Caching layer
- Comprehensive test suite

## License

MIT License - see LICENSE file for details.
