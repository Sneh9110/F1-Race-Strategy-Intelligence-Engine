"""
Storage Manager - Versioned data persistence with multiple formats

Handles storage of raw, processed, and feature-engineered data with versioning.
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import hashlib
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psycopg2
from psycopg2.extras import execute_batch

from app.utils.logger import get_logger
from config.settings import get_settings


class StorageManager:
    """
    Manages versioned data storage across multiple formats and destinations.
    
    Supports:
    - Parquet (primary format, compressed)
    - JSON (metadata, human-readable exports)
    - CSV (legacy compatibility)
    - PostgreSQL/TimescaleDB (relational queries)
    """
    
    def __init__(self, base_path: Optional[Path] = None, db_connection_string: Optional[str] = None):
        """
        Initialize storage manager.
        
        Args:
            base_path: Base directory for file storage (default: ./data)
            db_connection_string: PostgreSQL connection string
        """
        self.settings = get_settings()
        self.logger = get_logger("storage_manager")
        
        # Set base paths
        self.base_path = base_path or Path("data")
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"
        self.features_path = self.base_path / "features"
        self.metadata_path = self.base_path / "metadata"
        
        # Create directories
        for path in [self.raw_path, self.processed_path, self.features_path, self.metadata_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Database connection
        self.db_connection_string = db_connection_string or self.settings.database.url
        self.db_pool_size = self.settings.database.pool_size
        
        # Retention policy (days)
        self.retention_days = self.settings.data_pipeline.data_retention_days
        
        self.logger.info("Initialized storage manager", extra_data={
            "base_path": str(self.base_path),
            "retention_days": self.retention_days
        })
    
    def _get_db_connection(self):
        """Get database connection from pool."""
        try:
            return psycopg2.connect(self.db_connection_string)
        except Exception as e:
            self.logger.error(f"Database connection error: {str(e)}")
            raise
    
    def _generate_version(self) -> str:
        """Generate timestamp-based version string."""
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    def _calculate_checksum(self, data: Any) -> str:
        """Calculate SHA256 checksum of data."""
        if isinstance(data, pd.DataFrame):
            data_bytes = data.to_json().encode('utf-8')
        elif isinstance(data, (list, dict)):
            data_bytes = json.dumps(data, sort_keys=True).encode('utf-8')
        else:
            data_bytes = str(data).encode('utf-8')
        
        return hashlib.sha256(data_bytes).hexdigest()
    
    def _get_storage_path(
        self,
        storage_type: str,
        source: str,
        date: Optional[datetime] = None,
        version: Optional[str] = None
    ) -> Path:
        """
        Get storage path for data.
        
        Args:
            storage_type: Type of storage (raw, processed, features)
            source: Data source name
            date: Date for partitioning (default: today)
            version: Version string (default: auto-generated)
        
        Returns:
            Path object for storage location
        """
        if storage_type == "raw":
            base = self.raw_path
        elif storage_type == "processed":
            base = self.processed_path
        elif storage_type == "features":
            base = self.features_path
        else:
            raise ValueError(f"Invalid storage type: {storage_type}")
        
        date = date or datetime.utcnow()
        date_str = date.strftime("%Y-%m-%d")
        
        version = version or self._generate_version()
        
        # Structure: {base}/{source}/{date}/{version}.parquet
        path = base / source / date_str
        path.mkdir(parents=True, exist_ok=True)
        
        return path / f"{version}.parquet"
    
    def _save_metadata(
        self,
        storage_type: str,
        source: str,
        version: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Save metadata manifest for data version."""
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        metadata_dir = self.metadata_path / source / date_str
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_path = metadata_dir / "manifest.json"
        
        # Load existing manifest or create new
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        else:
            manifest = {"versions": []}
        
        # Add new version entry
        manifest["versions"].append({
            "version": version,
            "storage_type": storage_type,
            "timestamp": datetime.utcnow().isoformat(),
            **metadata
        })
        
        # Save manifest
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        self.logger.debug(f"Saved metadata for {source} version {version}")
    
    async def save_raw(
        self,
        source: str,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
        format: str = "parquet"
    ) -> str:
        """
        Save raw ingested data.
        
        Args:
            source: Data source name (timing, weather, telemetry, etc.)
            data: Data to save (list of dicts, DataFrame, or dict)
            metadata: Optional metadata to store
            format: Storage format (parquet, json, csv)
        
        Returns:
            Version string for saved data
        """
        try:
            version = self._generate_version()
            
            # Convert to DataFrame if needed
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
            
            # Get storage path
            storage_path = self._get_storage_path("raw", source, version=version)
            
            # Save based on format
            if format == "parquet":
                # Save as Parquet with compression
                df.to_parquet(
                    storage_path,
                    engine='pyarrow',
                    compression='snappy',
                    index=False
                )
            elif format == "json":
                json_path = storage_path.with_suffix('.json')
                df.to_json(json_path, orient='records', indent=2)
            elif format == "csv":
                csv_path = storage_path.with_suffix('.csv')
                df.to_csv(csv_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Calculate metadata
            file_size = storage_path.stat().st_size if storage_path.exists() else 0
            checksum = self._calculate_checksum(df)
            
            # Save metadata
            meta = {
                "record_count": len(df),
                "file_size_bytes": file_size,
                "checksum": checksum,
                "format": format,
                **(metadata or {})
            }
            self._save_metadata("raw", source, version, meta)
            
            self.logger.info(
                f"Saved raw data for {source}",
                extra_data={
                    "version": version,
                    "records": len(df),
                    "size_mb": file_size / 1024 / 1024
                }
            )
            
            return version
        
        except Exception as e:
            self.logger.error(f"Error saving raw data: {str(e)}")
            raise
    
    async def save_processed(
        self,
        source: str,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save processed/validated data.
        
        Args:
            source: Data source name
            data: Processed data
            metadata: Optional metadata
        
        Returns:
            Version string
        """
        version = self._generate_version()
        
        # Convert to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame([data])
        
        # Get storage path
        storage_path = self._get_storage_path("processed", source, version=version)
        
        # Save as Parquet
        df.to_parquet(
            storage_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        
        # Save metadata
        meta = {
            "record_count": len(df),
            "file_size_bytes": storage_path.stat().st_size,
            "checksum": self._calculate_checksum(df),
            **(metadata or {})
        }
        self._save_metadata("processed", source, version, meta)
        
        self.logger.info(f"Saved processed data for {source}", extra_data={"version": version})
        
        return version
    
    def load_latest(
        self,
        source: str,
        storage_type: str = "raw",
        date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load most recent data for a source.
        
        Args:
            source: Data source name
            storage_type: Type of storage (raw, processed, features)
            date: Optional date to load from (default: today)
        
        Returns:
            DataFrame with data, or None if not found
        """
        try:
            date = date or datetime.utcnow()
            date_str = date.strftime("%Y-%m-%d")
            
            # Get storage directory
            if storage_type == "raw":
                base_path = self.raw_path
            elif storage_type == "processed":
                base_path = self.processed_path
            else:
                base_path = self.features_path
            
            source_date_path = base_path / source / date_str
            
            if not source_date_path.exists():
                self.logger.warning(f"No data found for {source} on {date_str}")
                return None
            
            # Find latest version (newest timestamp)
            parquet_files = sorted(source_date_path.glob("*.parquet"), reverse=True)
            
            if not parquet_files:
                return None
            
            latest_file = parquet_files[0]
            
            # Load Parquet file
            df = pd.read_parquet(latest_file, engine='pyarrow')
            
            self.logger.info(f"Loaded latest {storage_type} data for {source}", extra_data={
                "file": latest_file.name,
                "records": len(df)
            })
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return None
    
    def load_version(
        self,
        source: str,
        version: str,
        storage_type: str = "raw"
    ) -> Optional[pd.DataFrame]:
        """Load specific version of data."""
        try:
            # Parse date from version (YYYYMMDD_HHMMSS)
            date_str = datetime.strptime(version.split('_')[0], "%Y%m%d").strftime("%Y-%m-%d")
            
            # Get storage path
            if storage_type == "raw":
                base_path = self.raw_path
            elif storage_type == "processed":
                base_path = self.processed_path
            else:
                base_path = self.features_path
            
            file_path = base_path / source / date_str / f"{version}.parquet"
            
            if not file_path.exists():
                self.logger.warning(f"Version {version} not found for {source}")
                return None
            
            df = pd.read_parquet(file_path, engine='pyarrow')
            
            self.logger.info(f"Loaded {storage_type} data version {version} for {source}")
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error loading version: {str(e)}")
            return None
    
    def list_versions(
        self,
        source: str,
        storage_type: str = "raw",
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        List available versions for a source.
        
        Args:
            source: Data source name
            storage_type: Type of storage
            days: Number of days to look back
        
        Returns:
            List of version metadata dicts
        """
        versions = []
        
        # Get base path
        if storage_type == "raw":
            base_path = self.raw_path
        elif storage_type == "processed":
            base_path = self.processed_path
        else:
            base_path = self.features_path
        
        source_path = base_path / source
        
        if not source_path.exists():
            return versions
        
        # Look through recent dates
        for i in range(days):
            date = datetime.utcnow() - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            date_path = source_path / date_str
            
            if date_path.exists():
                # Find all parquet files
                for file_path in date_path.glob("*.parquet"):
                    version = file_path.stem
                    stat = file_path.stat()
                    
                    versions.append({
                        "version": version,
                        "date": date_str,
                        "file_size_bytes": stat.st_size,
                        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
        
        return sorted(versions, key=lambda x: x['version'], reverse=True)
    
    async def save_to_database(
        self,
        table_name: str,
        data: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        """
        Save data to PostgreSQL database.
        
        Args:
            table_name: Database table name
            data: List of records to insert
            batch_size: Batch size for inserts
        
        Returns:
            Number of records inserted
        """
        if not data:
            return 0
        
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # Build insert query (assuming all records have same keys)
            columns = list(data[0].keys())
            placeholders = ', '.join(['%s'] * len(columns))
            query = f"""
                INSERT INTO {table_name} ({', '.join(columns)})
                VALUES ({placeholders})
                ON CONFLICT DO NOTHING
            """
            
            # Convert records to tuples
            values = [tuple(record[col] for col in columns) for record in data]
            
            # Batch insert
            execute_batch(cursor, query, values, page_size=batch_size)
            
            conn.commit()
            rows_inserted = cursor.rowcount
            
            cursor.close()
            conn.close()
            
            self.logger.info(f"Inserted {rows_inserted} records into {table_name}")
            
            return rows_inserted
        
        except Exception as e:
            self.logger.error(f"Database insert error: {str(e)}")
            raise
    
    def cleanup_old_data(self, source: Optional[str] = None, dry_run: bool = False) -> Dict[str, int]:
        """
        Delete data older than retention period.
        
        Args:
            source: Specific source to clean (None = all sources)
            dry_run: If True, don't delete, just report what would be deleted
        
        Returns:
            Dict with cleanup statistics
        """
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        stats = {"files_deleted": 0, "bytes_freed": 0}
        
        for storage_type in ["raw", "processed", "features"]:
            if storage_type == "raw":
                base_path = self.raw_path
            elif storage_type == "processed":
                base_path = self.processed_path
            else:
                base_path = self.features_path
            
            # Get sources to clean
            sources = [source] if source else [p.name for p in base_path.iterdir() if p.is_dir()]
            
            for src in sources:
                source_path = base_path / src
                
                if not source_path.exists():
                    continue
                
                # Find old date directories
                for date_dir in source_path.iterdir():
                    if not date_dir.is_dir():
                        continue
                    
                    try:
                        dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
                        
                        if dir_date < cutoff_date:
                            # Calculate size
                            dir_size = sum(f.stat().st_size for f in date_dir.rglob('*') if f.is_file())
                            file_count = len(list(date_dir.rglob('*.parquet')))
                            
                            if not dry_run:
                                shutil.rmtree(date_dir)
                                self.logger.info(f"Deleted old data: {date_dir}")
                            
                            stats["files_deleted"] += file_count
                            stats["bytes_freed"] += dir_size
                    
                    except ValueError:
                        # Skip directories that don't match date format
                        continue
        
        self.logger.info(
            f"Cleanup {'(dry run) ' if dry_run else ''}completed",
            extra_data=stats
        )
        
        return stats
