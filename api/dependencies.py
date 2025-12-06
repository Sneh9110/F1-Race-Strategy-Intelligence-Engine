"""Dependency injection for FastAPI endpoints."""

from functools import lru_cache
from typing import Optional
import redis.asyncio as redis

from models.tire_degradation.inference import DegradationPredictor
from models.lap_time.inference import LapTimePredictor
from models.safety_car.inference import SafetyCarPredictor
from models.pit_stop_loss.inference import PitStopLossPredictor
from simulation import RaceSimulator
from decision_engine import DecisionEngine
from config.settings import Settings


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


@lru_cache()
def get_redis_client() -> redis.Redis:
    """Get Redis client for caching and rate limiting."""
    settings = get_settings()
    return redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        decode_responses=True,
    )


@lru_cache()
def get_tire_degradation_predictor() -> DegradationPredictor:
    """Get singleton tire degradation predictor."""
    settings = get_settings()
    return DegradationPredictor(
        model_version='production',
        config_path='config/tire_degradation.yaml'
    )


@lru_cache()
def get_lap_time_predictor() -> LapTimePredictor:
    """Get singleton lap time predictor."""
    settings = get_settings()
    return LapTimePredictor(
        model_version='production',
        config_path='config/lap_time.yaml'
    )


@lru_cache()
def get_safety_car_predictor() -> SafetyCarPredictor:
    """Get singleton safety car predictor."""
    settings = get_settings()
    return SafetyCarPredictor(
        model_version='production',
        config_path='config/safety_car.yaml'
    )


@lru_cache()
def get_pit_stop_loss_predictor() -> PitStopLossPredictor:
    """Get singleton pit stop loss predictor."""
    settings = get_settings()
    return PitStopLossPredictor(
        model_version='production',
        config_path='config/pit_stop_loss.yaml'
    )


@lru_cache()
def get_race_simulator() -> RaceSimulator:
    """Get singleton race simulator."""
    return RaceSimulator()


@lru_cache()
def get_decision_engine() -> DecisionEngine:
    """Get singleton decision engine."""
    return DecisionEngine(config_path='config/decision_engine.yaml')


# Dependency functions for FastAPI
async def get_settings_dependency() -> Settings:
    """FastAPI dependency for settings."""
    return get_settings()


async def get_redis_dependency() -> redis.Redis:
    """FastAPI dependency for Redis client."""
    return get_redis_client()


async def get_degradation_predictor_dependency() -> DegradationPredictor:
    """FastAPI dependency for degradation predictor."""
    return get_tire_degradation_predictor()


async def get_lap_time_predictor_dependency() -> LapTimePredictor:
    """FastAPI dependency for lap time predictor."""
    return get_lap_time_predictor()


async def get_safety_car_predictor_dependency() -> SafetyCarPredictor:
    """FastAPI dependency for safety car predictor."""
    return get_safety_car_predictor()


async def get_pit_stop_loss_predictor_dependency() -> PitStopLossPredictor:
    """FastAPI dependency for pit stop loss predictor."""
    return get_pit_stop_loss_predictor()


async def get_race_simulator_dependency() -> RaceSimulator:
    """FastAPI dependency for race simulator."""
    return get_race_simulator()


async def get_decision_engine_dependency() -> DecisionEngine:
    """FastAPI dependency for decision engine."""
    return get_decision_engine()
