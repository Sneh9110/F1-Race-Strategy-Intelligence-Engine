"""
Ingestion Orchestrator - Coordinates all data ingestors

Manages scheduling, health monitoring, session lifecycle.
"""

from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timedelta
import asyncio
from enum import Enum

from data_pipeline.ingestors.timing_ingestor import TimingIngestor
from data_pipeline.ingestors.weather_ingestor import WeatherIngestor
from data_pipeline.ingestors.historical_ingestor import HistoricalDataIngestor
from data_pipeline.ingestors.safety_car_ingestor import SafetyCarIngestor
from data_pipeline.ingestors.telemetry_ingestor import TelemetryIngestor
from data_pipeline.base.storage_manager import StorageManager
from data_pipeline.base.qa_engine import QAEngine
from app.utils.logger import get_logger


class SessionState(str, Enum):
    """Session lifecycle states."""
    PRE_SESSION = "pre_session"
    ACTIVE = "active"
    POST_SESSION = "post_session"
    COMPLETED = "completed"


class IngestionOrchestrator:
    """
    Orchestrates all data ingestion operations.
    
    Manages:
    - Session lifecycle
    - Ingestor scheduling
    - Health monitoring
    - Error recovery
    - Resource coordination
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize orchestrator."""
        self.config = config
        self.logger = get_logger(__name__)
        
        self.storage_manager = StorageManager(config.get('storage', {}))
        self.qa_engine = QAEngine(config.get('qa', {}))
        
        # Initialize all ingestors
        self.ingestors = {
            'timing': TimingIngestor(self.storage_manager, self.qa_engine, config.get('timing', {})),
            'weather': WeatherIngestor(self.storage_manager, self.qa_engine, config.get('weather', {})),
            'historical': HistoricalDataIngestor(self.storage_manager, self.qa_engine, config.get('historical', {})),
            'safety_car': SafetyCarIngestor(self.storage_manager, self.qa_engine, config.get('safety_car', {})),
            'telemetry': TelemetryIngestor(self.storage_manager, self.qa_engine, config.get('telemetry', {}))
        }
        
        self.session_state = SessionState.PRE_SESSION
        self.active_tasks: Set[asyncio.Task] = set()
        
        self.logger.info("Ingestion orchestrator initialized")
    
    async def run_live_session(self, session_info: Dict[str, Any]):
        """Run live race session ingestion."""
        self.logger.info(f"Starting live session: {session_info.get('name')}")
        
        try:
            self.session_state = SessionState.ACTIVE
            
            # Start all live ingestors concurrently
            tasks = [
                asyncio.create_task(self._run_ingestor('timing', interval=1)),
                asyncio.create_task(self._run_ingestor('weather', interval=60)),
                asyncio.create_task(self._run_ingestor('safety_car', interval=5)),
                asyncio.create_task(self._run_ingestor('telemetry', interval=0.1))
            ]
            
            self.active_tasks.update(tasks)
            
            # Wait for session end or error
            await asyncio.gather(*tasks, return_exceptions=True)
            
            self.session_state = SessionState.POST_SESSION
            
        except Exception as e:
            self.logger.error(f"Session error: {str(e)}")
            raise
        finally:
            await self._cleanup_session()
    
    async def run_historical_batch(self, year: int, rounds: Optional[List[int]] = None):
        """Run historical data batch ingestion."""
        self.logger.info(f"Starting historical batch for {year}")
        
        historical_ingestor = self.ingestors['historical']
        historical_ingestor.config['year'] = year
        
        try:
            result = await historical_ingestor.run()
            self.logger.info(f"Historical batch complete: {result.records_ingested} records")
            return result
        except Exception as e:
            self.logger.error(f"Historical batch error: {str(e)}")
            raise
    
    async def _run_ingestor(self, name: str, interval: float):
        """Run single ingestor with periodic execution."""
        ingestor = self.ingestors[name]
        
        while self.session_state == SessionState.ACTIVE:
            try:
                await ingestor.run()
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(f"{name} ingestor error: {str(e)}")
                # Continue on error (circuit breaker handles retries)
    
    async def _cleanup_session(self):
        """Clean up after session."""
        self.logger.info("Cleaning up session")
        
        # Cancel all active tasks
        for task in self.active_tasks:
            if not task.done():
                task.cancel()
        
        await asyncio.gather(*self.active_tasks, return_exceptions=True)
        self.active_tasks.clear()
        
        self.session_state = SessionState.COMPLETED
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get orchestrator and ingestor health."""
        return {
            "session_state": self.session_state.value,
            "active_tasks": len(self.active_tasks),
            "ingestors": {
                name: {
                    "circuit_breaker_state": ing.circuit_breaker.state.value,
                    "mock_mode": ing.mock_mode
                }
                for name, ing in self.ingestors.items()
            }
        }
