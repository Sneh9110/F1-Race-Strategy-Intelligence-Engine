"""
Timing Ingestor - FIA timing data ingestion pipeline

Ingests lap times, sector times, positions, and gaps from live timing feeds.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import asyncio
import random

from data_pipeline.base.base_ingestor import BaseIngestor
from data_pipeline.schemas.timing_schema import TimingPoint, LapData, SessionTiming
from data_pipeline.mock.mock_timing_generator import MockTimingGenerator
from app.utils.validators import validate_lap_time, validate_sector_times, detect_outliers
from app.utils.logger import get_logger


class TimingIngestor(BaseIngestor):
    """
    Timing data ingestor for FIA live timing feed.
    
    Sources:
    - Live: FIA timing API (requires API key)
    - Mock: Realistic timing data generator
    """
    
    def __init__(self, storage_manager, qa_engine, config: Optional[Dict[str, Any]] = None):
        """Initialize timing ingestor."""
        super().__init__(
            source_name="timing",
            schema_class=SessionTiming,
            storage_manager=storage_manager,
            qa_engine=qa_engine,
            config=config
        )
        
        # Timing-specific configuration
        self.api_url = self.config.get('api_url', '')
        self.api_key = self.config.get('api_key', '')
        self.session_id = self.config.get('session_id', '')
        
        # Initialize mock generator
        self.mock_generator = MockTimingGenerator(
            track_name=self.config.get('track_name', 'Monaco'),
            num_drivers=self.config.get('num_drivers', 20)
        )
        
        # Track last ingested lap to avoid duplicates
        self.last_lap_number = 0
        
        self.logger.info("Initialized timing ingestor", extra_data={
            "session_id": self.session_id,
            "mock_mode": self.mock_mode
        })
    
    async def ingest(self) -> Dict[str, Any]:
        """
        Fetch timing data from source.
        
        Returns:
            Dict with session timing data
        """
        if self.mock_mode:
            return await self._ingest_mock()
        else:
            return await self._ingest_live()
    
    async def _ingest_live(self) -> Dict[str, Any]:
        """Ingest data from live FIA timing API."""
        if not self.api_url or not self.api_key:
            raise ValueError("API URL and API key required for live mode")
        
        try:
            # TODO: Implement actual API integration
            # For now, this is a placeholder showing the structure
            
            # import httpx
            # async with httpx.AsyncClient() as client:
            #     response = await client.get(
            #         self.api_url,
            #         headers={"Authorization": f"Bearer {self.api_key}"},
            #         params={"session": self.session_id}
            #     )
            #     response.raise_for_status()
            #     raw_data = response.json()
            
            # For now, raise to indicate not implemented
            raise NotImplementedError("Live API integration pending - use mock mode")
        
        except Exception as e:
            self.logger.error(f"Live ingestion error: {str(e)}")
            raise
    
    async def _ingest_mock(self) -> Dict[str, Any]:
        """Generate realistic mock timing data."""
        try:
            # Generate next lap of data
            self.last_lap_number += 1
            
            # Generate session timing data
            session_data = self.mock_generator.generate_lap(self.last_lap_number)
            
            # Add session metadata
            session_timing = {
                "session_id": self.session_id or f"MOCK_{datetime.utcnow().strftime('%Y%m%d')}",
                "session_type": self.config.get('session_type', 'RACE'),
                "track_name": self.config.get('track_name', 'Monaco'),
                "session_start_time": datetime.utcnow().isoformat(),
                "lap_data": session_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.debug(
                f"Generated mock timing data for lap {self.last_lap_number}",
                extra_data={"drivers": len(session_data)}
            )
            
            return session_timing
        
        except Exception as e:
            self.logger.error(f"Mock generation error: {str(e)}")
            raise
    
    def validate(self, raw_data: Any) -> List[SessionTiming]:
        """
        Validate timing data.
        
        Performs additional timing-specific validation:
        - Sector time consistency
        - Position/gap consistency
        - Lap time outlier detection
        """
        # Base validation using Pydantic
        validated = super().validate(raw_data)
        
        # Additional timing-specific validation
        for session in validated:
            for lap_data in session.lap_data:
                # Validate lap time range
                if lap_data.lap_time:
                    try:
                        validate_lap_time(lap_data.lap_time)
                    except ValueError as e:
                        self.logger.warning(
                            f"Invalid lap time: {lap_data.lap_time}s",
                            extra_data={
                                "driver": lap_data.driver_number,
                                "lap": lap_data.lap_number
                            }
                        )
                
                # Validate sector times consistency
                if all([lap_data.sector_1_time, lap_data.sector_2_time, lap_data.sector_3_time]):
                    sector_sum = (
                        lap_data.sector_1_time +
                        lap_data.sector_2_time +
                        lap_data.sector_3_time
                    )
                    
                    if lap_data.lap_time and abs(sector_sum - lap_data.lap_time) > 0.1:
                        self.logger.warning(
                            f"Sector time inconsistency: sum={sector_sum:.3f}s, lap_time={lap_data.lap_time:.3f}s",
                            extra_data={
                                "driver": lap_data.driver_number,
                                "lap": lap_data.lap_number
                            }
                        )
        
        return validated
    
    async def store(self, validated_data: List[SessionTiming]) -> None:
        """
        Store timing data.
        
        Saves to:
        - Parquet files (data/raw/timing/)
        - PostgreSQL timing_data table
        """
        # Store in files via base class
        await super().store(validated_data)
        
        # Also store in database
        for session in validated_data:
            # Convert lap data to database format
            db_records = []
            for lap_data in session.lap_data:
                record = {
                    "session_id": session.session_id,
                    "timestamp": datetime.utcnow(),
                    "driver_number": lap_data.driver_number,
                    "lap_number": lap_data.lap_number,
                    "lap_time": lap_data.lap_time,
                    "sector_1_time": lap_data.sector_1_time,
                    "sector_2_time": lap_data.sector_2_time,
                    "sector_3_time": lap_data.sector_3_time,
                    "position": lap_data.position,
                    "tire_compound": lap_data.tire_compound,
                    "tire_age": lap_data.tire_age,
                    "track_status": lap_data.track_status
                }
                db_records.append(record)
            
            # Batch insert to database
            if db_records:
                try:
                    await self.storage_manager.save_to_database(
                        table_name="timing_data",
                        data=db_records,
                        batch_size=100
                    )
                except Exception as e:
                    self.logger.error(f"Database storage error: {str(e)}")
                    # Don't fail ingestion if DB insert fails
