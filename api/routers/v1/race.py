"""Race state management endpoints."""

import time
from datetime import datetime
from fastapi import APIRouter, Depends, Request, HTTPException, status, WebSocket, WebSocketDisconnect
from typing import Dict

from api.schemas.race import (
    RaceStateRequest,
    RaceStateResponse,
    RaceState,
    WebSocketMessage,
    LapCompletedData,
)
from api.schemas.common import MetadataBase, APIResponse
from api.dependencies import get_redis_dependency
from api.auth import get_current_user_optional, User
from api.middleware.rate_limiter import rate_limit_dependency
from api.utils.cache import CacheManager
import redis.asyncio as redis
import json

router = APIRouter()

# In-memory storage for race states (replace with Redis/database in production)
race_states: Dict[str, RaceState] = {}

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)

    def disconnect(self, websocket: WebSocket, session_id: str):
        if session_id in self.active_connections:
            self.active_connections[session_id].remove(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]

    async def broadcast(self, message: WebSocketMessage, session_id: str):
        if session_id in self.active_connections:
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_text(message.model_dump_json())
                except Exception as e:
                    print(f"Error broadcasting to websocket: {e}")

manager = ConnectionManager()


@router.get("/state/{session_id}", response_model=APIResponse[RaceStateResponse])
async def get_race_state(
    session_id: str,
    request: Request,
    current_user: User | None = Depends(get_current_user_optional),
    redis_client: redis.Redis = Depends(get_redis_dependency)
):
    """
    Get current race state for a session.
    
    Rate limit: 60 requests/minute
    Cache TTL: 5 seconds
    """
    await rate_limit_dependency(request, "race_state", 60)
    
    start_time = time.time()
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Check cache first
    cache = CacheManager(redis_client)
    cache_key = f"race_state:{session_id}"
    cached_state = await cache.get(cache_key)
    
    if cached_state:
        latency_ms = (time.time() - start_time) * 1000
        return APIResponse(
            data=RaceStateResponse(
                **cached_state,
                metadata=MetadataBase(
                    request_id=request_id,
                    timestamp=datetime.utcnow(),
                    latency_ms=latency_ms,
                    cache_hit=True
                )
            ),
            metadata=MetadataBase(
                request_id=request_id,
                timestamp=datetime.utcnow(),
                latency_ms=latency_ms,
                cache_hit=True
            )
        )
    
    # Get from in-memory storage
    if session_id not in race_states:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Race state not found for session {session_id}"
        )
    
    race_state = race_states[session_id]
    latency_ms = (time.time() - start_time) * 1000
    
    response = RaceStateResponse(
        **race_state.model_dump(),
        metadata=MetadataBase(
            request_id=request_id,
            timestamp=datetime.utcnow(),
            latency_ms=latency_ms,
            cache_hit=False
        )
    )
    
    # Cache for 5 seconds
    await cache.set(cache_key, response.model_dump(exclude={"metadata"}), ttl=5)
    
    return APIResponse(
        data=response,
        metadata=MetadataBase(
            request_id=request_id,
            timestamp=datetime.utcnow(),
            latency_ms=latency_ms,
            cache_hit=False
        )
    )


@router.post("/state/{session_id}", response_model=APIResponse[RaceStateResponse])
async def create_or_update_race_state(
    session_id: str,
    data: RaceStateRequest,
    request: Request,
    current_user: User | None = Depends(get_current_user_optional),
    redis_client: redis.Redis = Depends(get_redis_dependency)
):
    """
    Create or update race state for a session.
    
    Rate limit: 30 requests/minute
    """
    await rate_limit_dependency(request, "race_state_update", 30)
    
    start_time = time.time()
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Create race state
    race_state = RaceState(**data.model_dump())
    race_states[session_id] = race_state
    
    # Invalidate cache
    cache = CacheManager(redis_client)
    cache_key = f"race_state:{session_id}"
    await cache.delete(cache_key)
    
    # Broadcast update via WebSocket
    ws_message = WebSocketMessage(
        type="RACE_STATE_UPDATE",
        data=race_state,
        timestamp=datetime.utcnow()
    )
    await manager.broadcast(ws_message, session_id)
    
    latency_ms = (time.time() - start_time) * 1000
    
    response = RaceStateResponse(
        **race_state.model_dump(),
        metadata=MetadataBase(
            request_id=request_id,
            timestamp=datetime.utcnow(),
            latency_ms=latency_ms,
            cache_hit=False
        )
    )
    
    return APIResponse(
        data=response,
        metadata=MetadataBase(
            request_id=request_id,
            timestamp=datetime.utcnow(),
            latency_ms=latency_ms,
            cache_hit=False
        )
    )


@router.delete("/state/{session_id}")
async def delete_race_state(
    session_id: str,
    request: Request,
    current_user: User | None = Depends(get_current_user_optional),
    redis_client: redis.Redis = Depends(get_redis_dependency)
):
    """
    Delete race state for a session.
    
    Rate limit: 10 requests/minute
    """
    await rate_limit_dependency(request, "race_state_delete", 10)
    
    if session_id not in race_states:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Race state not found for session {session_id}"
        )
    
    # Delete from storage
    del race_states[session_id]
    
    # Invalidate cache
    cache = CacheManager(redis_client)
    cache_key = f"race_state:{session_id}"
    await cache.delete(cache_key)
    
    return {"status": "deleted", "session_id": session_id}


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time race updates.
    
    Sends messages with type and data fields:
    - LAP_COMPLETED: Driver completed a lap
    - PIT_STOP: Driver pit stop
    - SAFETY_CAR: Safety car deployed/withdrawn
    - POSITION_CHANGE: Driver position change
    - RACE_STATE_UPDATE: Full race state update
    """
    await manager.connect(websocket, session_id)
    
    try:
        # Send initial race state if available
        if session_id in race_states:
            initial_message = WebSocketMessage(
                type="RACE_STATE_UPDATE",
                data=race_states[session_id],
                timestamp=datetime.utcnow()
            )
            await websocket.send_text(initial_message.model_dump_json())
        
        # Keep connection alive and handle heartbeat
        while True:
            try:
                data = await websocket.receive_text()
                # Handle ping/pong for heartbeat
                if data == "ping":
                    await websocket.send_text("pong")
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket error: {e}")
                break
    
    finally:
        manager.disconnect(websocket, session_id)
