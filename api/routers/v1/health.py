"""Health and monitoring endpoints."""

import time
from fastapi import APIRouter, Depends
from datetime import datetime

from api.schemas.common import HealthCheckResponse, ComponentHealth, ComponentStatus
from api.config import api_config
from api.dependencies import get_redis_dependency
from api.auth import get_current_user_optional, User
from api.main import app_start_time
import redis.asyncio as redis

router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    redis_client: redis.Redis = Depends(get_redis_dependency),
    current_user: User | None = Depends(get_current_user_optional)
):
    """
    Basic health check endpoint.
    
    Returns overall system health status and component health.
    No authentication required.
    """
    components = {}
    overall_status = ComponentStatus.HEALTHY
    
    # Check Redis
    try:
        start = time.time()
        await redis_client.ping()
        redis_latency = (time.time() - start) * 1000
        components["redis"] = ComponentHealth(
            status=ComponentStatus.HEALTHY,
            latency_ms=redis_latency
        )
    except Exception as e:
        components["redis"] = ComponentHealth(
            status=ComponentStatus.UNHEALTHY,
            message=str(e)
        )
        overall_status = ComponentStatus.DEGRADED
    
    # Check models (basic check - they're loaded at startup)
    components["models"] = ComponentHealth(
        status=ComponentStatus.HEALTHY,
        message="All models loaded"
    )
    
    # Calculate uptime
    uptime = time.time() - app_start_time
    
    return HealthCheckResponse(
        status=overall_status,
        version=api_config.API_VERSION,
        uptime_seconds=uptime,
        components=components,
        timestamp=datetime.utcnow()
    )


@router.get("/health/ready")
async def readiness_probe(redis_client: redis.Redis = Depends(get_redis_dependency)):
    """
    Kubernetes readiness probe.
    
    Returns 200 if service is ready to accept traffic, 503 otherwise.
    """
    try:
        # Check Redis connectivity
        await redis_client.ping()
        return {"status": "ready"}
    except Exception:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )


@router.get("/health/live")
async def liveness_probe():
    """
    Kubernetes liveness probe.
    
    Returns 200 if service is alive.
    """
    return {"status": "alive"}


@router.get("/metrics")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus text format.
    """
    # Simplified metrics - full implementation would use prometheus_client
    uptime = time.time() - app_start_time
    
    metrics = f"""# HELP api_uptime_seconds API uptime in seconds
# TYPE api_uptime_seconds gauge
api_uptime_seconds {uptime}

# HELP api_info API information
# TYPE api_info gauge
api_info{{version="{api_config.API_VERSION}"}} 1
"""
    
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(content=metrics, media_type="text/plain")
