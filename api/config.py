"""API-specific configuration."""

from typing import List
from pydantic_settings import BaseSettings


class APIConfig(BaseSettings):
    """API-specific configuration settings."""
    
    # API Metadata
    API_TITLE: str = "F1 Race Strategy Intelligence API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Production-grade REST API for F1 race strategy optimization, predictions, and simulations"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    RELOAD: bool = False
    LOG_LEVEL: str = "info"
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: List[str] = ["*"]
    CORS_HEADERS: List[str] = ["*"]
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Rate Limiting
    RATE_LIMIT_PREDICTIONS: int = 60  # per minute
    RATE_LIMIT_SIMULATIONS: int = 10  # per minute
    RATE_LIMIT_ADMIN: int = 5  # per minute
    RATE_LIMIT_DEFAULT: int = 100  # per minute
    
    # Performance
    REQUEST_TIMEOUT_SECONDS: int = 60
    MAX_REQUEST_SIZE_MB: int = 10
    ENABLE_COMPRESSION: bool = True
    COMPRESSION_LEVEL: int = 6
    
    # Caching
    CACHE_TTL_PREDICTIONS: int = 60  # seconds
    CACHE_TTL_SIMULATIONS: int = 300  # seconds
    CACHE_TTL_RACE_STATE: int = 5  # seconds
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    class Config:
        env_file = ".env"
        env_prefix = "API_"


# Global config instance
api_config = APIConfig()
