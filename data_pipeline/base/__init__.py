"""
Base Ingestion Framework - Core components for data ingestion pipelines
"""

from data_pipeline.base.base_ingestor import BaseIngestor, IngestionResult
from data_pipeline.base.storage_manager import StorageManager
from data_pipeline.base.qa_engine import QAEngine, QAReport

__all__ = [
    "BaseIngestor",
    "IngestionResult",
    "StorageManager",
    "QAEngine",
    "QAReport",
]
