"""Strategy recommendation endpoints."""

import time
from datetime import datetime
from fastapi import APIRouter, Depends, Request, HTTPException, status

from api.schemas.simulation import DecisionRequest, DecisionResponse, DecisionModule
from api.schemas.common import MetadataBase, APIResponse
from api.dependencies import get_decision_engine_dependency, get_redis_dependency
from api.auth import get_current_user_optional, require_auth, User
from api.middleware.rate_limiter import rate_limit_dependency
from api.utils.cache import CacheManager
from api.utils.validators import validate_circuit_name
from api.config import api_config
import redis.asyncio as redis

router = APIRouter()


@router.post("/recommend", response_model=APIResponse[DecisionResponse])
async def recommend_strategy(
    request: Request,
    data: DecisionRequest,
    current_user: User | None = Depends(get_current_user_optional),
    decision_engine = Depends(get_decision_engine_dependency),
    redis_client: redis.Redis = Depends(get_redis_dependency)
):
    """
    Get real-time strategy recommendation.
    
    Rate limit: 30 requests/minute
    Cache TTL: 5 seconds (race state changes rapidly)
    """
    await rate_limit_dependency(request, "predictions", 30)
    
    start_time = time.time()
    request_id = getattr(request.state, "request_id", "unknown")
    
    validate_circuit_name(data.circuit_name)
    
    # Short cache for rapidly changing race conditions
    cache = CacheManager(redis_client)
    cache_key = cache.generate_cache_key("decision", data.model_dump())
    cached_result = await cache.get(cache_key)
    
    if cached_result:
        latency_ms = (time.time() - start_time) * 1000
        return APIResponse(
            success=True,
            data=DecisionResponse(
                **cached_result,
                metadata=MetadataBase(
                    request_id=request_id,
                    timestamp=datetime.utcnow(),
                    latency_ms=latency_ms,
                    cache_hit=True
                )
            ),
            message="Strategy recommendation (cached)"
        )
    
    try:
        # TODO: Integrate with actual decision engine
        recommendation = "Continue on current tire for 5 more laps, then pit for SOFT"
        confidence = 0.82
        reasoning = "Tire degradation is moderate, and pit window opens in 3 laps. Track position favorable for undercut."
        alternatives = [
            "Pit now for MEDIUM tires",
            "Extend current stint by 10 laps"
        ]
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = DecisionResponse(
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning,
            alternative_options=alternatives,
            risk_assessment="LOW",
            metadata=MetadataBase(
                request_id=request_id,
                timestamp=datetime.utcnow(),
                latency_ms=latency_ms,
                cache_hit=False
            )
        )
        
        # Short cache (5 seconds) for race state
        await cache.set(cache_key, result.model_dump(exclude={"metadata"}), ttl=api_config.CACHE_TTL_RACE_STATE)
        
        return APIResponse(
            success=True,
            data=result,
            message="Strategy recommendation generated"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Decision engine failed: {str(e)}"
        )


@router.get("/modules", response_model=APIResponse[list[DecisionModule]])
async def list_decision_modules(
    current_user: User = Depends(require_auth),
    decision_engine = Depends(get_decision_engine_dependency)
):
    """
    List all decision engine modules.
    
    Requires authentication.
    """
    # TODO: Get actual modules from decision engine
    modules = [
        DecisionModule(
            name="Tire Strategy Module",
            priority=10,
            enabled=True,
            description="Optimizes tire compound selection and pit stop timing"
        ),
        DecisionModule(
            name="Fuel Management Module",
            priority=8,
            enabled=True,
            description="Manages fuel consumption and strategy"
        ),
        DecisionModule(
            name="Safety Car Response Module",
            priority=9,
            enabled=True,
            description="Adapts strategy during safety car periods"
        ),
        DecisionModule(
            name="Weather Adaptation Module",
            priority=7,
            enabled=True,
            description="Adjusts strategy based on weather conditions"
        ),
        DecisionModule(
            name="Overtaking Opportunity Module",
            priority=6,
            enabled=True,
            description="Identifies optimal overtaking windows"
        ),
        DecisionModule(
            name="Track Position Module",
            priority=5,
            enabled=True,
            description="Balances track position vs optimal strategy"
        ),
        DecisionModule(
            name="Risk Assessment Module",
            priority=8,
            enabled=True,
            description="Evaluates risk of different strategic options"
        )
    ]
    
    return APIResponse(
        success=True,
        data=modules,
        message=f"Retrieved {len(modules)} decision modules"
    )
