"""Common API schemas."""

from datetime import datetime
from typing import Generic, TypeVar, Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum


class ErrorCode(str, Enum):
    """Standard error codes."""
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    NOT_FOUND = "not_found"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    TIMEOUT = "timeout"
    INTERNAL_ERROR = "internal_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    BAD_REQUEST = "bad_request"


class ErrorResponse(BaseModel):
    """Standard error response."""
    error_code: ErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "error_code": "validation_error",
                "message": "Invalid input parameters",
                "details": {"field": "tire_age", "error": "must be positive"},
                "timestamp": "2024-12-06T10:30:00Z",
                "request_id": "req_abc123"
            }
        }


class ComponentStatus(str, Enum):
    """Component health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Health status for a component."""
    status: ComponentStatus
    latency_ms: Optional[float] = None
    error_rate: Optional[float] = None
    message: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: ComponentStatus
    version: str
    uptime_seconds: float
    components: Dict[str, ComponentHealth]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "uptime_seconds": 3600.5,
                "components": {
                    "redis": {"status": "healthy", "latency_ms": 2.5},
                    "models": {"status": "healthy", "latency_ms": 150.0}
                },
                "timestamp": "2024-12-06T10:30:00Z"
            }
        }


class PaginationParams(BaseModel):
    """Pagination query parameters."""
    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")
    sort_by: Optional[str] = Field(default=None, description="Field to sort by")
    sort_order: str = Field(default="asc", pattern="^(asc|desc)$", description="Sort order")


T = TypeVar('T')


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response."""
    items: List[T]
    total: int
    page: int
    page_size: int
    total_pages: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "items": [],
                "total": 100,
                "page": 1,
                "page_size": 20,
                "total_pages": 5
            }
        }


class MetadataBase(BaseModel):
    """Base metadata for responses."""
    request_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    latency_ms: float
    cache_hit: bool = False


class ModelStats(BaseModel):
    """Model performance statistics."""
    model_name: str
    model_version: str
    prediction_count: int
    cache_hit_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    fallback_count: int
    error_count: int
    last_updated: datetime


class RateLimitInfo(BaseModel):
    """Rate limit information."""
    limit: int
    remaining: int
    reset_at: datetime
    window_seconds: int


class APIResponse(BaseModel, Generic[T]):
    """Generic API response wrapper."""
    data: T
    metadata: MetadataBase
    rate_limit: Optional[RateLimitInfo] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "data": {},
                "metadata": {
                    "request_id": "req_abc123",
                    "timestamp": "2024-12-06T10:30:00Z",
                    "latency_ms": 150.5,
                    "cache_hit": False
                }
            }
        }
