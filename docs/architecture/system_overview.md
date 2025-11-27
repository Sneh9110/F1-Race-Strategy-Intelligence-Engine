# F1 Race Strategy Intelligence Engine - System Architecture

## Overview

The F1 Race Strategy Intelligence Engine is a production-grade platform designed to provide real-time race strategy optimization using machine learning, simulation, and data-driven decision-making. The system handles 20+ cars × 70+ laps × multiple data streams with sub-200ms latency for predictions.

## Architecture Principles

### Microservices Approach
- **Modular Design**: Each component can be deployed independently
- **Scalability**: Horizontal scaling for high-load components
- **Fault Isolation**: Failure in one service doesn't cascade
- **Technology Flexibility**: Use best tool for each job

### Real-Time Performance
- **Latency Requirements**:
  - Predictions: <200ms
  - Simulations: <2s
  - Data ingestion: <100ms
- **Caching Strategy**: Redis for hot data, PostgreSQL for persistence
- **Event-Driven**: Asynchronous processing where possible

## System Components

### 1. Data Pipeline (`data_pipeline/`)
**Responsibility**: Ingest real-time and historical F1 data

**Sub-components**:
- **Ingestors**: Separate modules for timing, weather, telemetry data
- **Schemas**: Pydantic models for validation
- **Schedulers**: Automated polling with error handling

**Technology**: Python, FastF1, Ergast API, APScheduler

**Data Sources**:
- FIA Live Timing (real-time)
- Weather APIs (5-minute intervals)
- Historical race database (FastF1, Ergast)
- Telemetry streams (high-frequency)

**Output**: Validated data to TimescaleDB and message queue

### 2. Feature Engineering (`features/`)
**Responsibility**: Transform raw data into ML-ready features

**Features Computed**:
- Stint summaries (avg pace, degradation rate)
- Undercut/overcut deltas
- Traffic impact models
- Tire temperature profiles
- Weather impact factors

**Storage**: Feature Store (Parquet files with versioning)

**Technology**: Pandas, NumPy, Parquet

### 3. ML Models (`models/`)
**Responsibility**: Predictive modeling for strategy optimization

**Model Types**:
1. **Tire Degradation Model**
   - Predicts lap time loss per lap
   - Inputs: Compound, age, track temp, driver style
   - Algorithm: XGBoost Regression

2. **Lap Time Prediction Model**
   - Forecasts lap times given conditions
   - Inputs: Tire state, fuel load, traffic, weather
   - Algorithm: LightGBM

3. **Safety Car Probability Model**
   - Estimates SC likelihood per lap
   - Inputs: Track history, race position, lap number
   - Algorithm: Random Forest Classifier

4. **Pit Stop Loss Model**
   - Calculates time lost during pit stop
   - Inputs: Track layout, traffic, pit crew performance
   - Algorithm: Linear Regression with priors

**Model Registry**: Versioned artifacts with metadata

**Technology**: scikit-learn, XGBoost, LightGBM, MLflow

### 4. Simulation Engine (`simulation/`)
**Responsibility**: Monte Carlo race simulation and strategy trees

**Capabilities**:
- Run 10,000+ race simulations in <2 seconds
- Model uncertainty (SC, weather, degradation variance)
- Generate strategy tree with probability distributions
- What-if scenario analysis

**Algorithm**: Event-driven simulation with stochastic elements

**Technology**: NumPy (vectorized operations), Numba (JIT compilation)

### 5. Decision Engine (`decision_engine/`)
**Responsibility**: Real-time strategy recommendations

**Logic**:
- **Rule-Based**: Hard constraints (regulations, physics)
- **ML-Driven**: Probabilistic recommendations
- **Hybrid**: Combines both approaches

**Recommendations**:
- Optimal pit window
- Tire compound selection
- Undercut/overcut opportunities
- Safety car strategy

**Output**: Prioritized actions with confidence scores

### 6. REST API (`api/`)
**Responsibility**: Expose platform capabilities via HTTP endpoints

**Endpoints**:
- `/predictions/lap-time` - Lap time forecast
- `/predictions/degradation` - Tire degradation
- `/predictions/safety-car` - SC probability
- `/simulations/race` - Full race simulation
- `/strategies/optimal` - Best strategy recommendation
- `/strategies/compare` - Compare strategies
- `/admin/health` - Health check
- `/admin/metrics` - Performance metrics

**Technology**: FastAPI, Uvicorn, Pydantic

**Features**:
- OpenAPI documentation (auto-generated)
- JWT authentication
- Rate limiting
- CORS configuration
- Request validation

### 7. Frontend Dashboard (`frontend/`)
**Responsibility**: Real-time visualization for race strategists

**Pages**:
- **Live Race**: Real-time data, current standings, recommendations
- **Strategy View**: Pit stop timings, tire choices, fuel loads
- **Analytics**: Historical comparison, what-if scenarios
- **Telemetry**: Driver inputs, tire temps, lap time analysis

**Technology**: React, TypeScript, D3.js (charts), WebSockets (live data)

## Data Flow

```
External Sources → Data Pipeline → Database (TimescaleDB)
                                        ↓
                              Feature Engineering
                                        ↓
                              ML Models (Inference)
                                        ↓
                              Decision Engine
                                        ↓
                              REST API
                                        ↓
                              Frontend Dashboard
```

### Real-Time Flow (Race Day)
1. **Timing data** arrives every 100ms
2. **Pipeline** validates and stores in TimescaleDB
3. **Features** computed on-the-fly (cached for 5s)
4. **Models** predict next 10 laps
5. **Decision Engine** generates recommendations
6. **API** serves results to dashboard
7. **Frontend** updates visualization

**End-to-End Latency**: 150-200ms

## Technology Stack

### Core
- **Language**: Python 3.10+
- **Framework**: FastAPI (API), React (Frontend)
- **Database**: PostgreSQL 14+ with TimescaleDB extension
- **Cache**: Redis 7+
- **Message Queue**: (Optional) RabbitMQ for async processing

### Data & ML
- **Data Processing**: Pandas, NumPy, Parquet
- **ML Libraries**: scikit-learn, XGBoost, LightGBM
- **Experiment Tracking**: MLflow (model registry)
- **Feature Store**: Custom (Parquet + metadata)

### Infrastructure
- **Containerization**: Docker, Docker Compose
- **Orchestration**: Kubernetes (production)
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus + Grafana
- **Error Tracking**: Sentry
- **Logging**: Structured JSON logs

## Scalability Considerations

### Horizontal Scaling
- **API**: Load-balanced with 4-8 workers
- **Models**: Separate inference service, GPU optional
- **Database**: Read replicas for queries, write master for ingestion

### Vertical Scaling
- **Database**: 16GB RAM minimum, 32GB recommended
- **Model Inference**: 8-core CPU or 1x GPU
- **API**: 4-core CPU per instance

### Performance Optimization
- **Caching**: 
  - Model predictions (5-minute TTL)
  - Feature computations (10-second TTL)
  - Static data (track configs, tire compounds)
- **Database Indexing**: Multi-column indexes on query patterns
- **Query Optimization**: Continuous aggregates (TimescaleDB)

## Latency Budget (Target <200ms)

| Component | Latency | Notes |
|-----------|---------|-------|
| Data ingestion | 20ms | Network + validation |
| Feature computation | 30ms | Cached after first computation |
| Model inference | 80ms | Batch prediction (32 samples) |
| Decision logic | 20ms | Rule evaluation |
| API overhead | 30ms | Serialization + network |
| Frontend rendering | 20ms | Client-side |
| **Total** | **200ms** | Target achieved |

## Deployment Architecture

### Local Development
```
Docker Compose:
- PostgreSQL container
- Redis container
- API container (hot reload)
- Frontend container (hot reload)
```

### Cloud Production (AWS Example)
```
- ECS/EKS for container orchestration
- RDS PostgreSQL (Multi-AZ)
- ElastiCache Redis (Cluster mode)
- ALB for load balancing
- CloudWatch for monitoring
- S3 for model artifacts and data
```

### Hybrid Deployment
```
- Cloud: API, Database, Cache
- Edge: Model inference (low latency)
- Client: Frontend dashboard
```

## Failover & High Availability

### Database
- Primary-replica setup
- Automatic failover (30s RPO)
- Point-in-time recovery (PITR)

### API
- Multiple instances behind load balancer
- Health checks every 10s
- Circuit breaker pattern

### Models
- Fallback to simpler models if ML fails
- Rule-based backup logic
- Model versioning with rollback

## Security

### Authentication
- JWT tokens (60-minute expiry)
- API keys for external integrations
- Role-based access control (RBAC)

### Data Protection
- Encryption at rest (database)
- TLS 1.3 for data in transit
- Secrets management (AWS Secrets Manager)

### Rate Limiting
- 60 requests/minute per user (production)
- 1000 requests/minute per user (development)

## Monitoring & Observability

### Metrics (Prometheus)
- Request latency (p50, p95, p99)
- Model inference time
- Database query performance
- Cache hit rate
- API error rate

### Logging (Structured JSON)
- All requests logged with correlation ID
- Model predictions with input/output
- Error stacktraces with context
- Performance bottleneck identification

### Alerting
- API downtime > 1 minute
- Model inference > 500ms
- Database connection pool exhausted
- Prediction accuracy drop > 10%

## Future Enhancements

1. **Real-Time Streaming**: Replace polling with WebSocket streams
2. **Reinforcement Learning**: Train agents on race strategy
3. **Multi-Team Support**: Isolate data per team
4. **Mobile App**: iOS/Android dashboard
5. **Voice Interface**: Alexa/Google Assistant integration

## Conclusion

The F1 Race Strategy Intelligence Engine is architected for production workloads with emphasis on real-time performance, scalability, and reliability. The microservices design allows independent scaling and deployment while maintaining tight latency requirements critical for race day operations.
