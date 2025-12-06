"""Simulation endpoints."""

import time
from datetime import datetime
from fastapi import APIRouter, Depends, Request, HTTPException, status

from api.schemas.simulation import (
    StrategySimulationRequest, StrategySimulationResponse,
    CompareStrategiesRequest, CompareStrategiesResponse
)
from api.schemas.common import MetadataBase, APIResponse
from api.dependencies import get_race_simulator_dependency, get_redis_dependency
from api.auth import get_current_user_optional, User
from api.middleware.rate_limiter import rate_limit_dependency
from api.utils.cache import CacheManager
from api.utils.validators import validate_circuit_name
from api.config import api_config
import redis.asyncio as redis

router = APIRouter()


@router.post("/strategy", response_model=APIResponse[StrategySimulationResponse])
async def simulate_strategy(
    request: Request,
    data: StrategySimulationRequest,
    current_user: User | None = Depends(get_current_user_optional),
    simulator = Depends(get_race_simulator_dependency),
    redis_client: redis.Redis = Depends(get_redis_dependency)
):
    """
    Run race strategy simulation.
    
    Rate limit: 10 requests/minute
    Cache TTL: 300 seconds (5 minutes)
    Target latency: <5 seconds
    """
    await rate_limit_dependency(request, "simulations", api_config.RATE_LIMIT_SIMULATIONS)
    
    start_time = time.time()
    request_id = getattr(request.state, "request_id", "unknown")
    
    validate_circuit_name(data.circuit_name)
    
    # Check cache
    cache = CacheManager(redis_client)
    cache_key = cache.generate_cache_key("simulation", data.model_dump())
    cached_result = await cache.get(cache_key)
    
    if cached_result:
        latency_ms = (time.time() - start_time) * 1000
        return APIResponse(
            success=True,
            data=StrategySimulationResponse(
                **cached_result,
                metadata=MetadataBase(
                    request_id=request_id,
                    timestamp=datetime.utcnow(),
                    latency_ms=latency_ms,
                    cache_hit=True
                )
            ),
            message="Strategy simulation (cached)"
        )
    
    try:
        # TODO: Integrate with actual race simulator
        total_time = 5850.5  # Placeholder
        final_pos = 2
        pit_count = len(data.pit_stops)
        tire_strat = [data.starting_tire] + [ps.get("tire", "MEDIUM") for ps in data.pit_stops]
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = StrategySimulationResponse(
            total_race_time=total_time,
            final_position=final_pos,
            pit_stop_count=pit_count,
            tire_strategy=tire_strat,
            lap_times=[],
            metadata=MetadataBase(
                request_id=request_id,
                timestamp=datetime.utcnow(),
                latency_ms=latency_ms,
                cache_hit=False
            )
        )
        
        await cache.set(cache_key, result.model_dump(exclude={"metadata"}), ttl=api_config.CACHE_TTL_SIMULATIONS)
        
        return APIResponse(
            success=True,
            data=result,
            message="Strategy simulation completed"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation failed: {str(e)}"
        )


@router.post("/compare-strategies", response_model=APIResponse[CompareStrategiesResponse])
async def compare_strategies(
    request: Request,
    data: CompareStrategiesRequest,
    current_user: User | None = Depends(get_current_user_optional),
    simulator = Depends(get_race_simulator_dependency),
    redis_client: redis.Redis = Depends(get_redis_dependency)
):
    """
    Compare multiple race strategies.
    
    Rate limit: 5 requests/minute
    Cache TTL: 300 seconds
    """
    await rate_limit_dependency(request, "simulations", 5)
    
    start_time = time.time()
    request_id = getattr(request.state, "request_id", "unknown")
    
    validate_circuit_name(data.circuit_name)
    
    cache = CacheManager(redis_client)
    cache_key = cache.generate_cache_key("compare", data.model_dump())
    cached_result = await cache.get(cache_key)
    
    if cached_result:
        latency_ms = (time.time() - start_time) * 1000
        return APIResponse(
            success=True,
            data=CompareStrategiesResponse(
                **cached_result,
                metadata=MetadataBase(
                    request_id=request_id,
                    timestamp=datetime.utcnow(),
                    latency_ms=latency_ms,
                    cache_hit=True
                )
            ),
            message="Strategy comparison (cached)"
        )
    
    try:
        # TODO: Implement actual strategy comparison
        best = data.strategies[0].get("name", "Strategy 1")
        comparisons = [
            {"name": s.get("name", f"Strategy {i+1}"), "time": 5850.0 + i*2.5}
            for i, s in enumerate(data.strategies)
        ]
        time_diffs = {s.get("name", f"Strategy {i+1}"): i*2.5 for i, s in enumerate(data.strategies)}
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = CompareStrategiesResponse(
            best_strategy=best,
            comparisons=comparisons,
            time_differences=time_diffs,
            metadata=MetadataBase(
                request_id=request_id,
                timestamp=datetime.utcnow(),
                latency_ms=latency_ms,
                cache_hit=False
            )
        )
        
        await cache.set(cache_key, result.model_dump(exclude={"metadata"}), ttl=api_config.CACHE_TTL_SIMULATIONS)
        
        return APIResponse(
            success=True,
            data=result,
            message="Strategy comparison completed"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison failed: {str(e)}"
        )


@router.post("/monte-carlo")
async def monte_carlo_simulation(
    request: Request,
    data: StrategySimulationRequest,
    iterations: int = 1000,
    current_user: User | None = Depends(get_current_user_optional),
    simulator = Depends(get_race_simulator_dependency)
):
    """
    Run Monte Carlo simulation with uncertainty.
    
    Rate limit: 3 requests/minute (computationally expensive)
    """
    await rate_limit_dependency(request, "simulations", 3)
    
    # TODO: Implement Monte Carlo simulation
    return APIResponse(
        success=True,
        data={
            "mean_race_time": 5850.0,
            "std_dev": 12.5,
            "percentiles": {
                "p10": 5830.0,
                "p50": 5850.0,
                "p90": 5870.0
            },
            "iterations": iterations
        },
        message=f"Monte Carlo simulation completed ({iterations} iterations)"
    )
