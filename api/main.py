"""Main FastAPI application entry point."""

import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.config import api_config
from api.dependencies import (
    get_tire_degradation_predictor,
    get_lap_time_predictor,
    get_safety_car_predictor,
    get_pit_stop_loss_predictor,
    get_race_simulator,
    get_decision_engine,
    get_redis_client,
)
from api.schemas.common import ErrorResponse, ErrorCode
from api.routers.v1 import auth as auth_router, health, predictions, simulation, strategy
from app.utils.logger import get_logger, setup_logging, set_correlation_id
from config.settings import settings

# Setup logging once at module level
setup_logging(
    level=settings.logging.level,
    format_type=settings.logging.format,
    log_file=settings.logging.file
)

logger = get_logger(__name__)

# Application start time for uptime tracking
app_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    # Startup
    logger.info("Starting F1 Strategy Intelligence API...")
    logger.info(f"Environment: {settings.env}")
    logger.info(f"Debug mode: {settings.debug}")
    
    try:
        # Initialize all predictors (singleton pattern via lru_cache)
        logger.info("Loading ML model predictors...")
        degradation_pred = get_tire_degradation_predictor()
        lap_time_pred = get_lap_time_predictor()
        safety_car_pred = get_safety_car_predictor()
        pit_stop_pred = get_pit_stop_loss_predictor()
        logger.info("✓ ML models loaded successfully")
        
        # Initialize simulation engine
        logger.info("Loading race simulator...")
        simulator = get_race_simulator()
        logger.info("✓ Race simulator initialized")
        
        # Initialize decision engine
        logger.info("Loading decision engine...")
        decision_engine = get_decision_engine()
        logger.info("✓ Decision engine initialized")
        
        # Initialize Redis connection
        logger.info("Connecting to Redis...")
        redis_client = get_redis_client()
        await redis_client.ping()
        logger.info("✓ Redis connected")
        
        logger.info("🚀 API startup complete")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        raise
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down F1 Strategy Intelligence API...")
    try:
        # Close Redis connection
        redis_client = get_redis_client()
        await redis_client.close()
        logger.info("✓ Redis connection closed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    
    logger.info("👋 API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=api_config.API_TITLE,
    version=api_config.API_VERSION,
    description=api_config.API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/v1/openapi.json",
    lifespan=lifespan,
)


# Add middleware
# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=api_config.CORS_ORIGINS,
    allow_credentials=api_config.CORS_CREDENTIALS,
    allow_methods=api_config.CORS_METHODS,
    allow_headers=api_config.CORS_HEADERS,
)

# GZip compression
if api_config.ENABLE_COMPRESSION:
    app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=api_config.COMPRESSION_LEVEL)


# Global exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    error_code_map = {
        400: ErrorCode.BAD_REQUEST,
        401: ErrorCode.AUTHENTICATION_ERROR,
        403: ErrorCode.AUTHORIZATION_ERROR,
        404: ErrorCode.NOT_FOUND,
        429: ErrorCode.RATE_LIMIT_EXCEEDED,
        503: ErrorCode.SERVICE_UNAVAILABLE,
        504: ErrorCode.TIMEOUT,
    }
    
    error_response = ErrorResponse(
        error_code=error_code_map.get(exc.status_code, ErrorCode.INTERNAL_ERROR),
        message=exc.detail,
        request_id=request.headers.get("X-Request-ID"),
    )
    
    logger.warning(
        f"HTTP {exc.status_code}: {exc.detail}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "status_code": exc.status_code,
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    error_response = ErrorResponse(
        error_code=ErrorCode.VALIDATION_ERROR,
        message="Request validation failed",
        details={"errors": exc.errors()},
        request_id=request.headers.get("X-Request-ID"),
    )
    
    logger.warning(
        f"Validation error: {exc.errors()}",
        extra={"path": request.url.path, "method": request.method}
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    error_response = ErrorResponse(
        error_code=ErrorCode.INTERNAL_ERROR,
        message="An unexpected error occurred",
        details={"error": str(exc)} if api_config.LOG_LEVEL == "debug" else None,
        request_id=request.headers.get("X-Request-ID"),
    )
    
    logger.error(
        f"Unexpected error: {exc}",
        exc_info=True,
        extra={"path": request.url.path, "method": request.method}
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump(),
    )


# Request/Response logging middleware with correlation ID
@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """
    Add correlation ID to each request and set it in the logging context.
    
    The correlation ID allows tracking all logs for a single request.
    """
    # Get or generate correlation ID
    correlation_id = request.headers.get(
        "X-Correlation-ID",
        request.headers.get("X-Request-ID", str(uuid.uuid4()))
    )
    
    # Set correlation ID in logging context for this request
    set_correlation_id(correlation_id)
    
    # Store in request state for use in endpoints
    request.state.correlation_id = correlation_id
    request.state.request_id = correlation_id
    
    # Process request
    response = await call_next(request)
    
    # Add correlation ID to response headers
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Request-ID"] = correlation_id
    
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and responses with timing."""
    correlation_id = getattr(request.state, "correlation_id", "unknown")
    start_time = time.time()
    
    # Log request
    logger.info(
        f"→ {request.method} {request.url.path}",
        extra={
            "correlation_id": correlation_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.query_params),
            "client_host": request.client.host if request.client else None,
        }
    )
    
    # Process request
    response = await call_next(request)
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log response
    logger.info(
        f"← {response.status_code} ({latency_ms:.2f}ms)",
        extra={
            "correlation_id": correlation_id,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
        }
    )
    
    # Add timing header
    response.headers["X-Response-Time"] = f"{latency_ms:.2f}ms"
    
    return response


# Include routers
app.include_router(auth_router.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(predictions.router, prefix="/api/v1/predict", tags=["Predictions"])
app.include_router(simulation.router, prefix="/api/v1/simulate", tags=["Simulation"])
app.include_router(strategy.router, prefix="/api/v1/strategy", tags=["Strategy"])
app.include_router(health.router, prefix="/api/v1", tags=["Health & Monitoring"])


# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    return {
        "name": api_config.API_TITLE,
        "version": api_config.API_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=api_config.HOST,
        port=api_config.PORT,
        workers=api_config.WORKERS,
        reload=api_config.RELOAD,
        log_level=api_config.LOG_LEVEL,
    )
