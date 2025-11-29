"""
Feature store for persisting and retrieving computed features.

Provides storage backend using Parquet format with versioning,
metadata tracking, and Redis caching for real-time access.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime, timedelta
import json
import hashlib

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import redis

from config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureMetadata:
    """
    Metadata for stored features.
    
    Attributes:
        feature_name: Name of the feature
        version: Feature version (semantic versioning)
        computation_timestamp: When features were computed
        schema: DataFrame schema as dict
        dependencies: List of dependent feature names
        statistics: Statistical summary of features
        session_id: Session identifier
        checksum: SHA256 checksum for data integrity
    """
    feature_name: str
    version: str
    computation_timestamp: str
    schema: Dict[str, str]
    dependencies: List[str]
    statistics: Dict[str, Any]
    session_id: str
    checksum: str
    num_records: int = 0
    file_size_bytes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureMetadata':
        """Create metadata from dictionary."""
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert metadata to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'FeatureMetadata':
        """Create metadata from JSON string."""
        return cls.from_dict(json.loads(json_str))


class FeatureStore:
    """
    Feature store for persisting and retrieving features.
    
    Features are stored in Parquet format with the following structure:
    data/features/{feature_name}/{version}/{date}/session_{session_id}.parquet
    
    Metadata is stored alongside as JSON:
    data/features/{feature_name}/{version}/{date}/session_{session_id}.metadata.json
    
    Redis is used for caching frequently accessed features.
    """
    
    def __init__(
        self,
        base_path: Optional[Path] = None,
        redis_client: Optional[redis.Redis] = None,
        enable_cache: bool = True
    ):
        """
        Initialize feature store.
        
        Args:
            base_path: Base directory for feature storage
            redis_client: Redis client for caching
            enable_cache: Whether to enable Redis caching
        """
        self.base_path = base_path or Path(settings.DATA_DIR) / "features"
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.enable_cache = enable_cache and redis_client is not None
        self.redis_client = redis_client
        
        if self.enable_cache and self.redis_client:
            try:
                self.redis_client.ping()
                logger.info("Feature store initialized with Redis caching enabled")
            except redis.ConnectionError:
                logger.warning("Redis connection failed, caching disabled")
                self.enable_cache = False
        else:
            logger.info("Feature store initialized without caching")
    
    def save_features(
        self,
        features: pd.DataFrame,
        feature_name: str,
        version: str,
        session_id: str,
        dependencies: Optional[List[str]] = None,
        statistics: Optional[Dict[str, Any]] = None
    ) -> FeatureMetadata:
        """
        Save features to Parquet with metadata.
        
        Args:
            features: Features DataFrame
            feature_name: Name of the feature
            version: Feature version
            session_id: Session identifier
            dependencies: List of dependent features
            statistics: Feature statistics
            
        Returns:
            FeatureMetadata object
        """
        # Create directory structure
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        feature_dir = self.base_path / feature_name / version / date_str
        feature_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        parquet_path = feature_dir / f"session_{session_id}.parquet"
        metadata_path = feature_dir / f"session_{session_id}.metadata.json"
        
        # Save Parquet with compression
        table = pa.Table.from_pandas(features)
        pq.write_table(
            table,
            parquet_path,
            compression='snappy',
            use_dictionary=True
        )
        
        # Calculate checksum
        checksum = self._calculate_checksum(parquet_path)
        
        # Build schema dict
        schema = {col: str(dtype) for col, dtype in features.dtypes.items()}
        
        # Create metadata
        metadata = FeatureMetadata(
            feature_name=feature_name,
            version=version,
            computation_timestamp=datetime.utcnow().isoformat(),
            schema=schema,
            dependencies=dependencies or [],
            statistics=statistics or {},
            session_id=session_id,
            checksum=checksum,
            num_records=len(features),
            file_size_bytes=parquet_path.stat().st_size
        )
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            f.write(metadata.to_json())
        
        logger.info(
            f"Saved features '{feature_name}' v{version} for session {session_id}",
            extra={
                'feature_name': feature_name,
                'version': version,
                'session_id': session_id,
                'num_records': len(features),
                'file_size_bytes': metadata.file_size_bytes
            }
        )
        
        return metadata
    
    def load_features(
        self,
        feature_name: str,
        session_id: str,
        version: Optional[str] = None,
        date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load features from Parquet.
        
        Args:
            feature_name: Name of the feature
            session_id: Session identifier
            version: Feature version (defaults to latest)
            date: Date string (YYYY-MM-DD, defaults to today)
            
        Returns:
            Features DataFrame or None if not found
        """
        # Check cache first
        if self.enable_cache:
            cached = self._get_from_cache(feature_name, session_id, version)
            if cached is not None:
                logger.debug(f"Cache hit for feature '{feature_name}' session {session_id}")
                return cached
        
        # Determine version (latest if not specified)
        if version is None:
            version = self._get_latest_version(feature_name)
            if version is None:
                logger.warning(f"No versions found for feature '{feature_name}'")
                return None
        
        # Determine date
        date_str = date or datetime.utcnow().strftime("%Y-%m-%d")
        
        # File path
        parquet_path = (
            self.base_path / feature_name / version / date_str / 
            f"session_{session_id}.parquet"
        )
        
        if not parquet_path.exists():
            logger.warning(
                f"Features not found: '{feature_name}' v{version} session {session_id}"
            )
            return None
        
        # Load Parquet
        features = pd.read_parquet(parquet_path)
        
        logger.info(
            f"Loaded features '{feature_name}' v{version} for session {session_id}",
            extra={
                'feature_name': feature_name,
                'version': version,
                'session_id': session_id,
                'num_records': len(features)
            }
        )
        
        return features
    
    def load_metadata(
        self,
        feature_name: str,
        session_id: str,
        version: Optional[str] = None,
        date: Optional[str] = None
    ) -> Optional[FeatureMetadata]:
        """
        Load feature metadata.
        
        Args:
            feature_name: Name of the feature
            session_id: Session identifier
            version: Feature version (defaults to latest)
            date: Date string (defaults to today)
            
        Returns:
            FeatureMetadata or None if not found
        """
        if version is None:
            version = self._get_latest_version(feature_name)
            if version is None:
                return None
        
        date_str = date or datetime.utcnow().strftime("%Y-%m-%d")
        metadata_path = (
            self.base_path / feature_name / version / date_str /
            f"session_{session_id}.metadata.json"
        )
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            return FeatureMetadata.from_json(f.read())
    
    def cache_features(
        self,
        key: str,
        features: pd.DataFrame,
        ttl: int = 300
    ) -> bool:
        """
        Cache features in Redis.
        
        Args:
            key: Cache key
            features: Features DataFrame
            ttl: Time-to-live in seconds
            
        Returns:
            True if cached successfully
        """
        if not self.enable_cache:
            return False
        
        try:
            # Serialize DataFrame to JSON
            features_json = features.to_json(orient='records')
            self.redis_client.setex(key, ttl, features_json)
            logger.debug(f"Cached features with key '{key}' (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.error(f"Failed to cache features: {e}")
            return False
    
    def _get_from_cache(
        self,
        feature_name: str,
        session_id: str,
        version: Optional[str]
    ) -> Optional[pd.DataFrame]:
        """Get features from Redis cache."""
        if not self.enable_cache:
            return None
        
        try:
            key = self._cache_key(feature_name, session_id, version)
            cached = self.redis_client.get(key)
            if cached:
                return pd.read_json(cached, orient='records')
        except Exception as e:
            logger.error(f"Failed to retrieve from cache: {e}")
        
        return None
    
    def _cache_key(
        self,
        feature_name: str,
        session_id: str,
        version: Optional[str]
    ) -> str:
        """Generate cache key."""
        version_str = version or 'latest'
        return f"feature:{feature_name}:{version_str}:{session_id}"
    
    def get_feature_versions(self, feature_name: str) -> List[str]:
        """
        Get all available versions for a feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            List of version strings (sorted newest first)
        """
        feature_dir = self.base_path / feature_name
        if not feature_dir.exists():
            return []
        
        versions = [d.name for d in feature_dir.iterdir() if d.is_dir()]
        # Sort by semantic versioning
        versions.sort(reverse=True)
        return versions
    
    def _get_latest_version(self, feature_name: str) -> Optional[str]:
        """Get latest version for a feature."""
        versions = self.get_feature_versions(feature_name)
        return versions[0] if versions else None
    
    def cleanup_old_versions(
        self,
        feature_name: str,
        retention_days: int = 30,
        keep_latest: int = 3
    ) -> int:
        """
        Clean up old feature versions.
        
        Args:
            feature_name: Name of the feature
            retention_days: Delete data older than this
            keep_latest: Always keep this many latest versions
            
        Returns:
            Number of files deleted
        """
        deleted_count = 0
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        versions = self.get_feature_versions(feature_name)
        versions_to_check = versions[keep_latest:]  # Skip latest N
        
        for version in versions_to_check:
            version_dir = self.base_path / feature_name / version
            
            for date_dir in version_dir.iterdir():
                if not date_dir.is_dir():
                    continue
                
                try:
                    date_obj = datetime.strptime(date_dir.name, "%Y-%m-%d")
                    if date_obj < cutoff_date:
                        # Delete all files in date directory
                        for file in date_dir.iterdir():
                            file.unlink()
                            deleted_count += 1
                        date_dir.rmdir()
                        logger.info(f"Cleaned up old features: {date_dir}")
                except ValueError:
                    continue
        
        return deleted_count
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum for file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def list_sessions(
        self,
        feature_name: str,
        version: Optional[str] = None,
        date: Optional[str] = None
    ) -> List[str]:
        """
        List all sessions with features available.
        
        Args:
            feature_name: Name of the feature
            version: Feature version (defaults to latest)
            date: Date string (defaults to today)
            
        Returns:
            List of session IDs
        """
        if version is None:
            version = self._get_latest_version(feature_name)
            if version is None:
                return []
        
        date_str = date or datetime.utcnow().strftime("%Y-%m-%d")
        date_dir = self.base_path / feature_name / version / date_str
        
        if not date_dir.exists():
            return []
        
        sessions = []
        for file in date_dir.glob("session_*.parquet"):
            session_id = file.stem.replace("session_", "")
            sessions.append(session_id)
        
        return sorted(sessions)
