"""
Weather Ingestor - Weather data ingestion pipeline

Ingests track temperature, air temperature, humidity, wind, rainfall, and forecasts.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import asyncio

from data_pipeline.base.base_ingestor import BaseIngestor
from data_pipeline.schemas.weather_schema import WeatherData, WeatherForecast, WeatherSession
from data_pipeline.mock.mock_weather_generator import MockWeatherGenerator
from app.utils.validators import validate_temperature
from app.utils.logger import get_logger


class WeatherIngestor(BaseIngestor):
    """
    Weather data ingestor for track conditions and forecasts.
    
    Sources:
    - Live: Weather API (OpenWeatherMap, WeatherAPI, etc.)
    - Mock: Realistic weather generator based on track location
    """
    
    def __init__(self, storage_manager, qa_engine, config: Optional[Dict[str, Any]] = None):
        """Initialize weather ingestor."""
        super().__init__(
            source_name="weather",
            schema_class=WeatherSession,
            storage_manager=storage_manager,
            qa_engine=qa_engine,
            config=config
        )
        
        # Weather-specific configuration
        self.api_url = self.config.get('weather_api_url', 'https://api.openweathermap.org/data/2.5')
        self.api_key = self.config.get('weather_api_key', '')
        self.track_location = self.config.get('track_location', {'lat': 43.7347, 'lon': 7.4206})  # Monaco
        
        # Initialize mock generator
        self.mock_generator = MockWeatherGenerator(
            track_name=self.config.get('track_name', 'Monaco')
        )
        
        self.logger.info("Initialized weather ingestor", extra_data={
            "track": self.config.get('track_name'),
            "mock_mode": self.mock_mode
        })
    
    async def ingest(self) -> Dict[str, Any]:
        """Fetch weather data from source."""
        if self.mock_mode:
            return await self._ingest_mock()
        else:
            return await self._ingest_live()
    
    async def _ingest_live(self) -> Dict[str, Any]:
        """Ingest data from weather API."""
        if not self.api_key:
            raise ValueError("Weather API key required for live mode")
        
        try:
            # TODO: Implement actual weather API integration
            # Example for OpenWeatherMap:
            # import httpx
            # async with httpx.AsyncClient() as client:
            #     # Current weather
            #     current_response = await client.get(
            #         f"{self.api_url}/weather",
            #         params={
            #             "lat": self.track_location['lat'],
            #             "lon": self.track_location['lon'],
            #             "appid": self.api_key,
            #             "units": "metric"
            #         }
            #     )
            #     current_data = current_response.json()
            #     
            #     # Forecast
            #     forecast_response = await client.get(
            #         f"{self.api_url}/forecast",
            #         params={
            #             "lat": self.track_location['lat'],
            #             "lon": self.track_location['lon'],
            #             "appid": self.api_key,
            #             "units": "metric"
            #         }
            #     )
            #     forecast_data = forecast_response.json()
            
            raise NotImplementedError("Live weather API integration pending - use mock mode")
        
        except Exception as e:
            self.logger.error(f"Live weather ingestion error: {str(e)}")
            raise
    
    async def _ingest_mock(self) -> Dict[str, Any]:
        """Generate realistic mock weather data."""
        try:
            # Generate current weather observation
            weather_data = self.mock_generator.generate_observation()
            
            # Generate forecasts (3 hours ahead)
            forecasts = self.mock_generator.generate_forecasts(hours_ahead=3)
            
            # Build weather session
            session = {
                "session_id": self.config.get('session_id', f"MOCK_{datetime.utcnow().strftime('%Y%m%d')}"),
                "track_name": self.config.get('track_name', 'Monaco'),
                "observations": [weather_data],
                "forecasts": forecasts,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.debug(
                "Generated mock weather data",
                extra_data={
                    "track_temp": weather_data.get('track_temp_celsius'),
                    "condition": weather_data.get('track_condition')
                }
            )
            
            return session
        
        except Exception as e:
            self.logger.error(f"Mock weather generation error: {str(e)}")
            raise
    
    async def store(self, validated_data: List[WeatherSession]) -> None:
        """Store weather data in files and database."""
        # Store in files via base class
        await super().store(validated_data)
        
        # Also store in database
        for session in validated_data:
            db_records = []
            
            # Store observations
            for obs in session.observations:
                record = {
                    "session_id": session.session_id,
                    "timestamp": obs.timestamp if hasattr(obs, 'timestamp') else datetime.utcnow(),
                    "track_temp_celsius": obs.track_temp_celsius if hasattr(obs, 'track_temp_celsius') else None,
                    "air_temp_celsius": obs.air_temp_celsius if hasattr(obs, 'air_temp_celsius') else None,
                    "humidity_percent": obs.humidity_percent if hasattr(obs, 'humidity_percent') else None,
                    "wind_speed_kmh": obs.wind_speed_kmh if hasattr(obs, 'wind_speed_kmh') else None,
                    "rainfall_mm": obs.rainfall_mm if hasattr(obs, 'rainfall_mm') else None,
                    "track_condition": obs.track_condition if hasattr(obs, 'track_condition') else None
                }
                db_records.append(record)
            
            if db_records:
                try:
                    await self.storage_manager.save_to_database(
                        table_name="weather_data",
                        data=db_records,
                        batch_size=50
                    )
                except Exception as e:
                    self.logger.error(f"Database storage error: {str(e)}")
