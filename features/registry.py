"""
Feature registry for discovery and dependency management.

Provides a centralized registry for all feature calculators,
dependency resolution, and execution order computation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Type, Callable
from collections import defaultdict
import inspect

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureInfo:
    """
    Information about a registered feature.
    
    Attributes:
        name: Feature name
        version: Feature version
        description: Feature description
        calculator_class: Feature calculator class
        input_schema: Expected input schema
        output_schema: Expected output schema
        dependencies: List of feature dependencies
        computation_cost_ms: Estimated computation time in milliseconds
        tags: Optional tags for categorization
    """
    name: str
    version: str
    description: str
    calculator_class: Type
    input_schema: Dict[str, str]
    output_schema: Dict[str, str]
    dependencies: List[str]
    computation_cost_ms: int = 100
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class FeatureRegistry:
    """
    Singleton registry for feature discovery and management.
    
    Features are registered using the @register_feature decorator.
    The registry maintains a DAG of feature dependencies and provides
    methods for topological sorting to determine execution order.
    """
    
    _instance = None
    _features: Dict[str, FeatureInfo] = {}
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._features = {}
        return cls._instance
    
    def register(
        self,
        name: str,
        version: str,
        calculator_class: Type,
        description: str = "",
        input_schema: Optional[Dict[str, str]] = None,
        output_schema: Optional[Dict[str, str]] = None,
        dependencies: Optional[List[str]] = None,
        computation_cost_ms: int = 100,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Register a feature calculator.
        
        Args:
            name: Feature name
            version: Feature version
            calculator_class: Feature calculator class
            description: Feature description
            input_schema: Expected input schema
            output_schema: Expected output schema
            dependencies: List of feature dependencies
            computation_cost_ms: Estimated computation time
            tags: Optional tags for categorization
        """
        if name in self._features:
            logger.warning(f"Feature '{name}' already registered, overwriting")
        
        feature_info = FeatureInfo(
            name=name,
            version=version,
            description=description,
            calculator_class=calculator_class,
            input_schema=input_schema or {},
            output_schema=output_schema or {},
            dependencies=dependencies or [],
            computation_cost_ms=computation_cost_ms,
            tags=tags or []
        )
        
        self._features[name] = feature_info
        
        logger.info(
            f"Registered feature '{name}' v{version}",
            extra={
                'feature_name': name,
                'version': version,
                'dependencies': dependencies or []
            }
        )
    
    def get_feature(self, name: str) -> Optional[FeatureInfo]:
        """
        Get feature info by name.
        
        Args:
            name: Feature name
            
        Returns:
            FeatureInfo or None if not found
        """
        return self._features.get(name)
    
    def get_feature_instance(self, name: str, **kwargs) -> Optional[Any]:
        """
        Get feature calculator instance.
        
        Args:
            name: Feature name
            **kwargs: Arguments for calculator initialization
            
        Returns:
            Feature calculator instance or None
        """
        feature_info = self.get_feature(name)
        if feature_info is None:
            logger.error(f"Feature '{name}' not found in registry")
            return None
        
        try:
            return feature_info.calculator_class(**kwargs)
        except Exception as e:
            logger.error(f"Failed to instantiate feature '{name}': {e}")
            return None
    
    def list_features(self, tags: Optional[List[str]] = None) -> List[FeatureInfo]:
        """
        List all registered features.
        
        Args:
            tags: Optional tags to filter by
            
        Returns:
            List of FeatureInfo objects
        """
        features = list(self._features.values())
        
        if tags:
            features = [
                f for f in features
                if any(tag in f.tags for tag in tags)
            ]
        
        return sorted(features, key=lambda f: f.name)
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Build dependency graph for all features.
        
        Returns:
            Dictionary mapping feature names to their dependencies
        """
        graph = {}
        for name, info in self._features.items():
            graph[name] = info.dependencies
        return graph
    
    def compute_execution_order(self, feature_names: List[str]) -> List[str]:
        """
        Compute topological execution order for features.
        
        Uses Kahn's algorithm for topological sorting.
        
        Args:
            feature_names: List of feature names to compute
            
        Returns:
            Ordered list of feature names
            
        Raises:
            ValueError: If circular dependencies detected
        """
        # Build subgraph for requested features
        subgraph = self._build_subgraph(feature_names)
        
        # Validate no circular dependencies
        if self._has_cycle(subgraph):
            raise ValueError("Circular dependencies detected in feature graph")
        
        # Topological sort using Kahn's algorithm
        in_degree = defaultdict(int)
        for node in subgraph:
            in_degree[node] = 0
        
        for node, deps in subgraph.items():
            for dep in deps:
                in_degree[node] += 1
        
        queue = [node for node in subgraph if in_degree[node] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            # Find nodes that depend on current node
            for other_node, deps in subgraph.items():
                if node in deps:
                    in_degree[other_node] -= 1
                    if in_degree[other_node] == 0:
                        queue.append(other_node)
        
        if len(result) != len(subgraph):
            raise ValueError("Failed to compute execution order (cycle detected)")
        
        return result
    
    def _build_subgraph(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """
        Build dependency subgraph for requested features.
        
        Includes all transitive dependencies.
        """
        subgraph = {}
        visited = set()
        
        def visit(name: str):
            if name in visited:
                return
            visited.add(name)
            
            feature_info = self.get_feature(name)
            if feature_info is None:
                raise ValueError(f"Feature '{name}' not found in registry")
            
            subgraph[name] = feature_info.dependencies
            
            for dep in feature_info.dependencies:
                visit(dep)
        
        for name in feature_names:
            visit(name)
        
        return subgraph
    
    def _has_cycle(self, graph: Dict[str, List[str]]) -> bool:
        """
        Detect cycles in dependency graph using DFS.
        
        Args:
            graph: Dependency graph
            
        Returns:
            True if cycle detected
        """
        visited = set()
        rec_stack = set()
        
        def visit(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if visit(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                if visit(node):
                    return True
        
        return False
    
    def validate_dependencies(self, feature_name: str) -> Dict[str, Any]:
        """
        Validate feature dependencies.
        
        Args:
            feature_name: Feature name to validate
            
        Returns:
            Validation result dictionary
        """
        errors = []
        warnings = []
        
        feature_info = self.get_feature(feature_name)
        if feature_info is None:
            errors.append(f"Feature '{feature_name}' not found")
            return {'valid': False, 'errors': errors, 'warnings': warnings}
        
        # Check all dependencies are registered
        for dep in feature_info.dependencies:
            if self.get_feature(dep) is None:
                errors.append(f"Dependency '{dep}' not found in registry")
        
        # Check for circular dependencies
        try:
            self.compute_execution_order([feature_name])
        except ValueError as e:
            errors.append(str(e))
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def get_features_by_tag(self, tag: str) -> List[FeatureInfo]:
        """
        Get all features with a specific tag.
        
        Args:
            tag: Tag to filter by
            
        Returns:
            List of FeatureInfo objects
        """
        return [
            info for info in self._features.values()
            if tag in info.tags
        ]
    
    def clear(self) -> None:
        """Clear all registered features (mainly for testing)."""
        self._features.clear()
        logger.info("Feature registry cleared")


# Decorator for feature registration
def register_feature(
    name: str,
    version: str,
    description: str = "",
    input_schema: Optional[Dict[str, str]] = None,
    output_schema: Optional[Dict[str, str]] = None,
    dependencies: Optional[List[str]] = None,
    computation_cost_ms: int = 100,
    tags: Optional[List[str]] = None
) -> Callable:
    """
    Decorator for automatic feature registration.
    
    Usage:
        @register_feature(
            name='stint_summary',
            version='v1.0.0',
            description='Aggregate lap data into stint summaries',
            dependencies=['lap_data']
        )
        class StintSummaryFeature(BaseFeature):
            ...
    
    Args:
        name: Feature name
        version: Feature version
        description: Feature description
        input_schema: Expected input schema
        output_schema: Expected output schema
        dependencies: List of feature dependencies
        computation_cost_ms: Estimated computation time
        tags: Optional tags for categorization
        
    Returns:
        Decorator function
    """
    def decorator(cls: Type) -> Type:
        registry = FeatureRegistry()
        registry.register(
            name=name,
            version=version,
            calculator_class=cls,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            dependencies=dependencies,
            computation_cost_ms=computation_cost_ms,
            tags=tags
        )
        return cls
    
    return decorator
