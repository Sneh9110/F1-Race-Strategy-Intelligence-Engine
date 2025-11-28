"""
Data Version Manager - Semantic and timestamp-based versioning

Manages data versioning with Git-like tagging.
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json
import hashlib

from app.utils.logger import get_logger


class DataVersionManager:
    """
    Manage data versions with semantic versioning.
    
    Version format: v{major}.{minor}.{patch}_{timestamp}
    Example: v1.2.3_20240315_143022
    """
    
    def __init__(self, base_path: str = "data"):
        """Initialize version manager."""
        self.base_path = Path(base_path)
        self.versions_file = self.base_path / "versions.json"
        self.logger = get_logger(__name__)
        
        self._load_versions()
    
    def _load_versions(self):
        """Load version registry."""
        if self.versions_file.exists():
            with open(self.versions_file, 'r') as f:
                self.versions = json.load(f)
        else:
            self.versions = {}
    
    def _save_versions(self):
        """Save version registry."""
        self.versions_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.versions_file, 'w') as f:
            json.dump(self.versions, f, indent=2, default=str)
    
    def create_version(
        self,
        source: str,
        data_path: str,
        version_type: str = "patch",
        description: Optional[str] = None
    ) -> str:
        """
        Create new version.
        
        Args:
            source: Data source name
            data_path: Path to data file
            version_type: "major", "minor", or "patch"
            description: Optional version description
        
        Returns:
            Version string (e.g., "v1.2.3_20240315_143022")
        """
        # Get current version
        if source not in self.versions:
            self.versions[source] = {
                "current": {"major": 0, "minor": 0, "patch": 0},
                "history": []
            }
        
        current = self.versions[source]["current"]
        
        # Increment version
        if version_type == "major":
            current["major"] += 1
            current["minor"] = 0
            current["patch"] = 0
        elif version_type == "minor":
            current["minor"] += 1
            current["patch"] = 0
        else:  # patch
            current["patch"] += 1
        
        # Generate version string
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        version_str = f"v{current['major']}.{current['minor']}.{current['patch']}_{timestamp}"
        
        # Calculate checksum
        with open(data_path, 'rb') as f:
            checksum = hashlib.sha256(f.read()).hexdigest()
        
        # Record version
        version_record = {
            "version": version_str,
            "timestamp": timestamp,
            "data_path": str(data_path),
            "checksum": checksum,
            "description": description or f"{version_type.capitalize()} update"
        }
        
        self.versions[source]["history"].append(version_record)
        self._save_versions()
        
        self.logger.info(f"Created version {version_str} for {source}")
        return version_str
    
    def get_version(self, source: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Get version information."""
        if source not in self.versions:
            raise ValueError(f"No versions for source: {source}")
        
        if version is None:
            # Return latest
            return self.versions[source]["history"][-1]
        
        # Find specific version
        for v in self.versions[source]["history"]:
            if v["version"] == version:
                return v
        
        raise ValueError(f"Version not found: {version}")
    
    def list_versions(self, source: str, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent versions."""
        if source not in self.versions:
            return []
        
        history = self.versions[source]["history"]
        return history[-limit:]
    
    def rollback(self, source: str, version: str) -> Dict[str, Any]:
        """Rollback to specific version."""
        version_info = self.get_version(source, version)
        
        self.logger.info(f"Rolling back {source} to {version}")
        return version_info
