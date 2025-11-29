"""
Batch feature engine for historical data processing.

Orchestrates large-scale feature computation for historical sessions
with parallel processing and progress tracking.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm

from features.base import BaseFeature
from features.registry import FeatureRegistry
from features.store import FeatureStore
from config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BatchFeatureEngine:
    """
    Batch feature engine for historical data processing.
    
    Orchestrates computation of features across multiple sessions
    with dependency resolution, parallel processing, and error recovery.
    """
    
    def __init__(
        self,
        feature_store: Optional[FeatureStore] = None,
        registry: Optional[FeatureRegistry] = None,
        batch_size: int = 10,
        max_workers: Optional[int] = None
    ):
        """
        Initialize batch feature engine.
        
        Args:
            feature_store: FeatureStore instance
            registry: Feature registry
            batch_size: Number of sessions to process in batch
            max_workers: Maximum parallel workers (None = CPU count)
        """
        self.feature_store = feature_store or FeatureStore()
        self.registry = registry or FeatureRegistry()
        self.batch_size = batch_size
        self.max_workers = max_workers or mp.cpu_count()
        
        logger.info(
            f"Initialized BatchFeatureEngine with {self.max_workers} workers",
            extra={'batch_size': batch_size}
        )
    
    def compute_features(
        self,
        session_ids: List[str],
        feature_names: List[str],
        parallel: bool = True,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Compute features for multiple sessions.
        
        Args:
            session_ids: List of session identifiers
            feature_names: List of feature names to compute
            parallel: Whether to use parallel processing
            save_results: Whether to save to feature store
            
        Returns:
            Dictionary with computation results and statistics
        """
        start_time = datetime.utcnow()
        
        logger.info(
            f"Computing {len(feature_names)} features for {len(session_ids)} sessions",
            extra={
                'feature_names': feature_names,
                'session_count': len(session_ids),
                'parallel': parallel
            }
        )
        
        # Resolve execution order
        try:
            execution_order = self.registry.compute_execution_order(feature_names)
        except ValueError as e:
            logger.error(f"Failed to resolve feature dependencies: {e}")
            return {'success': False, 'error': str(e)}
        
        logger.info(f"Execution order: {execution_order}")
        
        # Process sessions
        results = {
            'success': True,
            'sessions_processed': 0,
            'sessions_failed': 0,
            'features_computed': {},
            'errors': []
        }
        
        if parallel and len(session_ids) > 1:
            results = self._compute_parallel(
                session_ids, execution_order, save_results
            )
        else:
            results = self._compute_sequential(
                session_ids, execution_order, save_results
            )
        
        # Summary
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        results['duration_seconds'] = duration
        results['start_time'] = start_time.isoformat()
        results['end_time'] = end_time.isoformat()
        
        logger.info(
            f"Batch computation completed",
            extra={
                'duration_seconds': duration,
                'sessions_processed': results['sessions_processed'],
                'sessions_failed': results['sessions_failed']
            }
        )
        
        return results
    
    def _compute_sequential(
        self,
        session_ids: List[str],
        execution_order: List[str],
        save_results: bool
    ) -> Dict[str, Any]:
        """Compute features sequentially."""
        results = {
            'success': True,
            'sessions_processed': 0,
            'sessions_failed': 0,
            'features_computed': {},
            'errors': []
        }
        
        for session_id in tqdm(session_ids, desc="Processing sessions"):
            try:
                session_features = self._compute_session_features(
                    session_id, execution_order
                )
                
                if save_results:
                    for feature_name, feature_df in session_features.items():
                        self.feature_store.save_features(
                            features=feature_df,
                            feature_name=feature_name,
                            version='v1.0.0',
                            session_id=session_id
                        )
                
                results['sessions_processed'] += 1
                for feature_name in session_features:
                    results['features_computed'][feature_name] = \
                        results['features_computed'].get(feature_name, 0) + 1
                
            except Exception as e:
                logger.error(f"Failed to process session {session_id}: {e}")
                results['sessions_failed'] += 1
                results['errors'].append({
                    'session_id': session_id,
                    'error': str(e)
                })
        
        return results
    
    def _compute_parallel(
        self,
        session_ids: List[str],
        execution_order: List[str],
        save_results: bool
    ) -> Dict[str, Any]:
        """Compute features in parallel."""
        results = {
            'success': True,
            'sessions_processed': 0,
            'sessions_failed': 0,
            'features_computed': {},
            'errors': []
        }
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            futures = {
                executor.submit(
                    self._compute_session_features_wrapper,
                    session_id,
                    execution_order
                ): session_id
                for session_id in session_ids
            }
            
            # Process results
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing sessions"
            ):
                session_id = futures[future]
                
                try:
                    session_features = future.result()
                    
                    if save_results:
                        for feature_name, feature_df in session_features.items():
                            self.feature_store.save_features(
                                features=feature_df,
                                feature_name=feature_name,
                                version='v1.0.0',
                                session_id=session_id
                            )
                    
                    results['sessions_processed'] += 1
                    for feature_name in session_features:
                        results['features_computed'][feature_name] = \
                            results['features_computed'].get(feature_name, 0) + 1
                
                except Exception as e:
                    logger.error(f"Failed to process session {session_id}: {e}")
                    results['sessions_failed'] += 1
                    results['errors'].append({
                        'session_id': session_id,
                        'error': str(e)
                    })
        
        return results
    
    def _compute_session_features(
        self,
        session_id: str,
        execution_order: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """Compute features for a single session."""
        session_features = {}
        
        for feature_name in execution_order:
            feature_calculator = self.registry.get_feature_instance(feature_name)
            
            if feature_calculator is None:
                logger.warning(f"Feature '{feature_name}' not found in registry")
                continue
            
            # Load dependencies
            dependencies = {}
            for dep_name in feature_calculator._get_dependencies():
                if dep_name in session_features:
                    dependencies[dep_name] = session_features[dep_name]
                else:
                    # Try to load from store
                    dep_df = self.feature_store.load_features(
                        dep_name, session_id
                    )
                    if dep_df is not None:
                        dependencies[dep_name] = dep_df
            
            # Load raw data (simplified - should load from data pipeline)
            raw_data = self._load_raw_data(session_id)
            
            # Compute feature
            result = feature_calculator.compute(raw_data, **dependencies)
            
            if result.success:
                session_features[feature_name] = result.features
            else:
                logger.warning(
                    f"Feature '{feature_name}' computation failed: {result.errors}"
                )
        
        return session_features
    
    @staticmethod
    def _compute_session_features_wrapper(
        session_id: str,
        execution_order: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """Wrapper for parallel execution."""
        engine = BatchFeatureEngine()
        return engine._compute_session_features(session_id, execution_order)
    
    def _load_raw_data(self, session_id: str) -> Any:
        """Load raw data for session (placeholder)."""
        # TODO: Implement actual data loading from data pipeline
        return pd.DataFrame()
    
    def backfill_features(
        self,
        start_date: str,
        end_date: str,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Backfill features for date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            feature_names: Features to compute
            
        Returns:
            Computation results
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        logger.info(f"Backfilling features from {start_date} to {end_date}")
        
        # Find sessions in date range (placeholder)
        session_ids = self._find_sessions_in_range(start, end)
        
        return self.compute_features(
            session_ids=session_ids,
            feature_names=feature_names,
            parallel=True,
            save_results=True
        )
    
    def _find_sessions_in_range(
        self,
        start: datetime,
        end: datetime
    ) -> List[str]:
        """Find session IDs in date range (placeholder)."""
        # TODO: Implement actual session discovery
        return []
    
    def update_features(
        self,
        session_id: str,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Recompute specific features for a session.
        
        Args:
            session_id: Session identifier
            feature_names: Features to recompute
            
        Returns:
            Computation results
        """
        logger.info(f"Updating features for session {session_id}")
        
        return self.compute_features(
            session_ids=[session_id],
            feature_names=feature_names,
            parallel=False,
            save_results=True
        )
