"""
Centralized Settings Module - Environment-based configuration

Uses Pydantic BaseSettings for type-safe configuration management.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import List, Optional
from functools import lru_cache
import os
from pathlib import Path


class DatabaseSettings(BaseSettings):
    """Database connection configuration."""

    url: str = Field(
        default="postgresql://postgres:password@localhost:5432/f1_strategy",
        description="PostgreSQL connection string"
    )
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Maximum overflow connections")
    echo: bool = Field(default=False, description="Echo SQL statements")

    class Config:
        env_prefix = "DATABASE_"


class RedisSettings(BaseSettings):
    """Redis cache configuration."""

    url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection string"
    )
    ttl_seconds: int = Field(default=300, description="Default cache TTL")

    class Config:
        env_prefix = "REDIS_"


class APISettings(BaseSettings):
    """API server configuration."""

    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, ge=1024, le=65535, description="API port")
    secret_key: str = Field(default="dev-secret-key-change-in-production", description="JWT secret key")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins"
    )
    rate_limit_per_minute: int = Field(default=60, description="API rate limit")
    workers: int = Field(default=4, description="Number of worker processes")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    class Config:
        env_prefix = "API_"


class ModelSettings(BaseSettings):
    """ML model configuration."""

    registry_path: str = Field(
        default="./models/registry",
        description="Path to model registry"
    )
    inference_batch_size: int = Field(default=32, description="Batch size for inference")
    enable_cache: bool = Field(default=True, description="Enable prediction caching")
    fallback_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for fallback"
    )
    model_timeout_seconds: int = Field(default=5, description="Model inference timeout")

    class Config:
        env_prefix = "MODEL_"


class DataPipelineSettings(BaseSettings):
    """Data pipeline configuration."""

    ingestion_interval_seconds: int = Field(
        default=10,
        description="Data ingestion frequency"
    )
    data_retention_days: int = Field(
        default=90,
        description="Data retention period"
    )
    enable_mock_data: bool = Field(
        default=False,
        description="Use mock data sources"
    )
    weather_api_key: Optional[str] = Field(
        default=None,
        description="Weather API key"
    )
    f1_data_api_key: Optional[str] = Field(
        default=None,
        description="F1 data API key"
    )

    class Config:
        env_prefix = "DATA_"


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="json", description="Log format (json/text)")
    file_path: Optional[str] = Field(default=None, description="Log file path")
    enable_console: bool = Field(default=True, description="Enable console logging")

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v_upper

    class Config:
        env_prefix = "LOG_"


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration."""

    sentry_dsn: Optional[str] = Field(
        default=None,
        description="Sentry DSN for error tracking"
    )
    prometheus_port: int = Field(
        default=9090,
        ge=1024,
        le=65535,
        description="Prometheus metrics port"
    )
    enable_tracing: bool = Field(
        default=False,
        description="Enable distributed tracing"
    )

    class Config:
        env_prefix = "MONITORING_"


class Settings(BaseSettings):
    """Main settings aggregating all configuration sections."""

    # Environment
    env: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Debug mode")

    # Sub-settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    api: APISettings = Field(default_factory=APISettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    data_pipeline: DataPipelineSettings = Field(default_factory=DataPipelineSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)

    @field_validator("env")
    @classmethod
    def validate_env(cls, v: str) -> str:
        """Validate environment is recognized."""
        valid_envs = ["development", "staging", "production", "testing"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Invalid environment. Must be one of: {valid_envs}")
        return v.lower()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        env_nested_delimiter = "__"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Application settings
    """
    return Settings()


# Module-level settings instance for backwards compatibility
# This allows existing code to import: from config.settings import settings
settings = get_settings()


# Example usage
if __name__ == "__main__":
    settings = get_settings()

    print("Environment:", settings.env)
    print("Debug Mode:", settings.debug)
    print("Database URL:", settings.database.url)
    print("API Host:", settings.api.host, ":", settings.api.port)
    print("Log Level:", settings.logging.level)
    print("Model Registry:", settings.model.registry_path)
