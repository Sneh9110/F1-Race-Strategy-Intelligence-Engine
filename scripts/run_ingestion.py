"""
Run Ingestion Script - CLI for running data ingestors

Provides command-line interface for ingestion operations.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.orchestrator import IngestionOrchestrator
from data_pipeline.ingestors.timing_ingestor import TimingIngestor
from data_pipeline.ingestors.weather_ingestor import WeatherIngestor
from data_pipeline.ingestors.historical_ingestor import HistoricalDataIngestor
from data_pipeline.base.storage_manager import StorageManager
from data_pipeline.base.qa_engine import QAEngine
from config.settings import Settings
from app.utils.logger import get_logger


logger = get_logger(__name__)


async def run_live_session(args):
    """Run live session ingestion."""
    logger.info("Starting live session ingestion")
    
    settings = Settings()
    orchestrator = IngestionOrchestrator(settings.dict())
    
    session_info = {
        "name": args.session_name or "Live Session",
        "track": args.track or "Monaco"
    }
    
    try:
        await orchestrator.run_live_session(session_info)
        logger.info("Live session completed successfully")
    except KeyboardInterrupt:
        logger.info("Session interrupted by user")
    except Exception as e:
        logger.error(f"Session failed: {str(e)}")
        sys.exit(1)


async def run_historical_batch(args):
    """Run historical data batch ingestion."""
    logger.info(f"Starting historical batch for year {args.year}")
    
    settings = Settings()
    orchestrator = IngestionOrchestrator(settings.dict())
    
    try:
        result = await orchestrator.run_historical_batch(
            year=args.year,
            rounds=args.rounds
        )
        logger.info(f"Historical batch completed: {result.records_ingested} records")
    except Exception as e:
        logger.error(f"Historical batch failed: {str(e)}")
        sys.exit(1)


async def run_test_mode(args):
    """Run ingestor in test/mock mode."""
    logger.info(f"Running {args.source} ingestor in test mode")
    
    settings = Settings()
    storage = StorageManager(settings.storage)
    qa = QAEngine(settings.qa)
    
    # Select ingestor
    if args.source == "timing":
        ingestor = TimingIngestor(storage, qa, {"mock_mode": True})
    elif args.source == "weather":
        ingestor = WeatherIngestor(storage, qa, {"mock_mode": True})
    else:
        logger.error(f"Unknown source: {args.source}")
        sys.exit(1)
    
    try:
        result = await ingestor.run()
        logger.info(f"Test ingestion completed: {result.records_ingested} records")
        logger.info(f"Success: {result.success}, Duration: {result.duration:.2f}s")
    except Exception as e:
        logger.error(f"Test ingestion failed: {str(e)}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="F1 Race Strategy Data Ingestion CLI"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Live session command
    live_parser = subparsers.add_parser('live', help='Run live session ingestion')
    live_parser.add_argument('--session-name', type=str, help='Session name')
    live_parser.add_argument('--track', type=str, help='Track name')
    
    # Historical batch command
    hist_parser = subparsers.add_parser('historical', help='Run historical batch ingestion')
    hist_parser.add_argument('--year', type=int, required=True, help='Year to ingest')
    hist_parser.add_argument('--rounds', type=int, nargs='+', help='Specific rounds')
    
    # Test mode command
    test_parser = subparsers.add_parser('test', help='Run test ingestion')
    test_parser.add_argument('--source', type=str, required=True,
                            choices=['timing', 'weather', 'telemetry'],
                            help='Data source to test')
    
    # Health check command
    health_parser = subparsers.add_parser('health', help='Check orchestrator health')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate handler
    if args.command == 'live':
        asyncio.run(run_live_session(args))
    elif args.command == 'historical':
        asyncio.run(run_historical_batch(args))
    elif args.command == 'test':
        asyncio.run(run_test_mode(args))
    elif args.command == 'health':
        settings = Settings()
        orchestrator = IngestionOrchestrator(settings.dict())
        health = orchestrator.get_health_status()
        print(f"Orchestrator Health: {health}")


if __name__ == "__main__":
    main()
