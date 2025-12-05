"""Decision module registry for versioning and management."""

from typing import Dict, List, Optional, Type, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from decision_engine.base import BaseDecisionModule
from decision_engine.schemas import DecisionCategory
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModuleInfo:
    """Module metadata."""
    name: str
    version: str
    module_class: Type[BaseDecisionModule]
    category: DecisionCategory
    priority: int
    enabled: bool
    registration_time: datetime
    config_path: Optional[str] = None
    description: str = ""


class DecisionModuleRegistry:
    """Singleton registry for decision modules."""
    
    _instance = None
    
    def __new__(cls):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._modules = {}  # type: Dict[str, Dict[str, ModuleInfo]]
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize registry."""
        if not self._initialized:
            self._modules = {}  # type: Dict[str, Dict[str, ModuleInfo]]
            self._initialized = True
            logger.info("Decision module registry initialized")
    
    def register_module(
        self,
        name: str,
        version: str,
        module_class: Type[BaseDecisionModule],
        category: DecisionCategory,
        priority: int,
        enabled: bool = True,
        config_path: Optional[str] = None,
        description: str = ""
    ):
        """
        Register decision module.
        
        Args:
            name: Module name
            version: Module version
            module_class: Module class
            category: Decision category
            priority: Priority (1-10)
            enabled: Whether module is enabled
            config_path: Path to config file
            description: Module description
        """
        if name not in self._modules:
            self._modules[name] = {}
        
        if version in self._modules[name]:
            logger.warning(f"Overwriting module {name} v{version}")
        
        module_info = ModuleInfo(
            name=name,
            version=version,
            module_class=module_class,
            category=category,
            priority=priority,
            enabled=enabled,
            registration_time=datetime.utcnow(),
            config_path=config_path,
            description=description,
        )
        
        self._modules[name][version] = module_info
        
        logger.info(
            f"Registered module: {name} v{version} "
            f"(category={category.value}, priority={priority}, enabled={enabled})"
        )
    
    def get_module(
        self, 
        name: str, 
        version: str = 'latest'
    ) -> Optional[BaseDecisionModule]:
        """
        Get module instance by name and version.
        
        Args:
            name: Module name
            version: Module version ('latest' for most recent)
            
        Returns:
            Module instance or None
        """
        if name not in self._modules:
            return None
        
        versions = self._modules[name]
        
        if version == 'latest':
            # Get most recent version
            latest_info = max(
                versions.values(), 
                key=lambda info: info.registration_time
            )
            module_info = latest_info
        else:
            module_info = versions.get(version)
        
        if not module_info:
            return None
        
        if not module_info.enabled:
            logger.debug(f"Module {name} v{version} is disabled")
            return None
        
        # Instantiate module
        try:
            module = module_info.module_class(config_path=module_info.config_path)
            return module
        except Exception as e:
            logger.error(f"Failed to instantiate module {name} v{version}: {e}")
            return None
    
    def list_modules(
        self,
        category: Optional[DecisionCategory] = None,
        enabled_only: bool = True
    ) -> List[ModuleInfo]:
        """
        List registered modules.
        
        Args:
            category: Filter by category
            enabled_only: Only return enabled modules
            
        Returns:
            List of module info
        """
        modules = []
        
        for name, versions in self._modules.items():
            for version, info in versions.items():
                if enabled_only and not info.enabled:
                    continue
                
                if category and info.category != category:
                    continue
                
                modules.append(info)
        
        # Sort by priority descending
        modules.sort(key=lambda m: m.priority, reverse=True)
        
        return modules
    
    def enable_module(self, name: str, version: str = 'latest') -> bool:
        """
        Enable a module.
        
        Args:
            name: Module name
            version: Module version
            
        Returns:
            True if successful
        """
        if name not in self._modules:
            return False
        
        versions = self._modules[name]
        
        if version == 'latest':
            # Enable latest version
            latest_info = max(
                versions.values(),
                key=lambda info: info.registration_time
            )
            latest_info.enabled = True
            logger.info(f"Enabled module {name} v{latest_info.version}")
            return True
        else:
            if version in versions:
                versions[version].enabled = True
                logger.info(f"Enabled module {name} v{version}")
                return True
        
        return False
    
    def disable_module(self, name: str, version: str = 'latest') -> bool:
        """
        Disable a module.
        
        Args:
            name: Module name
            version: Module version
            
        Returns:
            True if successful
        """
        if name not in self._modules:
            return False
        
        versions = self._modules[name]
        
        if version == 'latest':
            # Disable latest version
            latest_info = max(
                versions.values(),
                key=lambda info: info.registration_time
            )
            latest_info.enabled = False
            logger.info(f"Disabled module {name} v{latest_info.version}")
            return True
        else:
            if version in versions:
                versions[version].enabled = False
                logger.info(f"Disabled module {name} v{version}")
                return True
        
        return False
    
    def get_execution_order(self) -> List[str]:
        """
        Get module names in execution order (by priority).
        
        Returns:
            List of module names
        """
        modules = self.list_modules(enabled_only=True)
        return [m.name for m in modules]
    
    def validate_module(self, name: str) -> Dict[str, Any]:
        """
        Validate module configuration.
        
        Args:
            name: Module name
            
        Returns:
            Validation result
        """
        if name not in self._modules:
            return {
                'valid': False,
                'errors': [f"Module {name} not found"],
                'warnings': [],
            }
        
        errors = []
        warnings = []
        
        versions = self._modules[name]
        latest_info = max(versions.values(), key=lambda info: info.registration_time)
        
        # Check module class
        if not issubclass(latest_info.module_class, BaseDecisionModule):
            errors.append(f"Module class does not inherit from BaseDecisionModule")
        
        # Check config file exists
        if latest_info.config_path:
            config_path = Path(latest_info.config_path)
            if not config_path.exists():
                warnings.append(f"Config file not found: {latest_info.config_path}")
        
        valid = len(errors) == 0
        
        return {
            'valid': valid,
            'errors': errors,
            'warnings': warnings,
            'module': latest_info.name,
            'version': latest_info.version,
        }
    
    def clear(self):
        """Clear all registered modules (for testing)."""
        self._modules.clear()
        logger.info("Cleared module registry")


def register_decision_module(
    name: str,
    version: str,
    category: DecisionCategory,
    priority: int,
    enabled: bool = True
):
    """
    Decorator to register decision module.
    
    Args:
        name: Module name
        version: Module version
        category: Decision category
        priority: Priority (1-10)
        enabled: Whether enabled
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        registry = DecisionModuleRegistry()
        registry.register_module(
            name=name,
            version=version,
            module_class=cls,
            category=category,
            priority=priority,
            enabled=enabled,
        )
        return cls
    
    return decorator
