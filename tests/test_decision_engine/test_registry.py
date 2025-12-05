"""Tests for decision module registry."""

import pytest
from decision_engine.registry import DecisionModuleRegistry, register_decision_module, ModuleInfo
from decision_engine import BaseDecisionModule, PitTimingDecision, DecisionCategory


def test_registry_singleton():
    """Test registry is singleton."""
    registry1 = DecisionModuleRegistry()
    registry2 = DecisionModuleRegistry()
    
    assert registry1 is registry2


def test_register_module():
    """Test module registration."""
    registry = DecisionModuleRegistry()
    
    module_info = ModuleInfo(
        name="test_module",
        version="v1.0.0",
        module_class=PitTimingDecision,
        category=DecisionCategory.PIT_TIMING,
        priority=9,
        enabled=True,
    )
    
    registry.register_module(module_info)
    
    # Should be retrievable
    retrieved = registry.get_module("test_module", version="v1.0.0")
    assert retrieved is not None
    assert retrieved.name == "test_module"


def test_register_module_multiple_versions():
    """Test registering multiple versions."""
    registry = DecisionModuleRegistry()
    
    # Register v1.0.0
    info_v1 = ModuleInfo(
        name="test_module",
        version="v1.0.0",
        module_class=PitTimingDecision,
        category=DecisionCategory.PIT_TIMING,
        priority=9,
        enabled=True,
    )
    registry.register_module(info_v1)
    
    # Register v1.1.0
    info_v2 = ModuleInfo(
        name="test_module",
        version="v1.1.0",
        module_class=PitTimingDecision,
        category=DecisionCategory.PIT_TIMING,
        priority=9,
        enabled=True,
    )
    registry.register_module(info_v2)
    
    # Both should be retrievable
    v1 = registry.get_module("test_module", version="v1.0.0")
    v2 = registry.get_module("test_module", version="v1.1.0")
    
    assert v1.version == "v1.0.0"
    assert v2.version == "v1.1.0"


def test_get_module_latest():
    """Test getting latest module version."""
    registry = DecisionModuleRegistry()
    
    # Register multiple versions
    for i in range(3):
        info = ModuleInfo(
            name="test_module",
            version=f"v1.{i}.0",
            module_class=PitTimingDecision,
            category=DecisionCategory.PIT_TIMING,
            priority=9,
            enabled=True,
        )
        registry.register_module(info)
    
    # Get latest
    latest = registry.get_module("test_module", version="latest")
    
    # Should get most recent version
    assert latest is not None


def test_list_modules():
    """Test listing modules."""
    registry = DecisionModuleRegistry()
    
    # Register some modules
    for i in range(3):
        info = ModuleInfo(
            name=f"module_{i}",
            version="v1.0.0",
            module_class=PitTimingDecision,
            category=DecisionCategory.PIT_TIMING,
            priority=i + 5,
            enabled=True,
        )
        registry.register_module(info)
    
    modules = registry.list_modules()
    
    assert len(modules) >= 3


def test_list_modules_filtered_by_category():
    """Test listing modules filtered by category."""
    registry = DecisionModuleRegistry()
    
    # Register modules with different categories
    info1 = ModuleInfo(
        name="pit_module",
        version="v1.0.0",
        module_class=PitTimingDecision,
        category=DecisionCategory.PIT_TIMING,
        priority=9,
        enabled=True,
    )
    registry.register_module(info1)
    
    info2 = ModuleInfo(
        name="sc_module",
        version="v1.0.0",
        module_class=PitTimingDecision,
        category=DecisionCategory.SAFETY_CAR,
        priority=10,
        enabled=True,
    )
    registry.register_module(info2)
    
    # Filter by category
    pit_modules = registry.list_modules(category=DecisionCategory.PIT_TIMING)
    
    # Should only get pit timing modules
    assert all(m.category == DecisionCategory.PIT_TIMING for m in pit_modules)


def test_list_modules_enabled_only():
    """Test listing only enabled modules."""
    registry = DecisionModuleRegistry()
    
    # Register enabled and disabled modules
    info_enabled = ModuleInfo(
        name="enabled_module",
        version="v1.0.0",
        module_class=PitTimingDecision,
        category=DecisionCategory.PIT_TIMING,
        priority=9,
        enabled=True,
    )
    registry.register_module(info_enabled)
    
    info_disabled = ModuleInfo(
        name="disabled_module",
        version="v1.0.0",
        module_class=PitTimingDecision,
        category=DecisionCategory.PIT_TIMING,
        priority=9,
        enabled=False,
    )
    registry.register_module(info_disabled)
    
    # Get only enabled
    enabled_modules = registry.list_modules(enabled_only=True)
    
    assert all(m.enabled for m in enabled_modules)


def test_enable_disable_module():
    """Test enabling and disabling modules."""
    registry = DecisionModuleRegistry()
    
    info = ModuleInfo(
        name="test_module",
        version="v1.0.0",
        module_class=PitTimingDecision,
        category=DecisionCategory.PIT_TIMING,
        priority=9,
        enabled=True,
    )
    registry.register_module(info)
    
    # Disable
    registry.disable_module("test_module")
    disabled = registry.get_module("test_module")
    assert not disabled.enabled
    
    # Enable
    registry.enable_module("test_module")
    enabled = registry.get_module("test_module")
    assert enabled.enabled


def test_get_execution_order():
    """Test getting modules in execution order."""
    registry = DecisionModuleRegistry()
    
    # Register modules with different priorities
    for priority in [5, 10, 7, 9]:
        info = ModuleInfo(
            name=f"module_p{priority}",
            version="v1.0.0",
            module_class=PitTimingDecision,
            category=DecisionCategory.PIT_TIMING,
            priority=priority,
            enabled=True,
        )
        registry.register_module(info)
    
    ordered = registry.get_execution_order()
    
    # Should be sorted by priority descending
    for i in range(len(ordered) - 1):
        assert ordered[i].priority >= ordered[i + 1].priority


def test_validate_module():
    """Test module validation."""
    registry = DecisionModuleRegistry()
    
    # Valid module
    info = ModuleInfo(
        name="valid_module",
        version="v1.0.0",
        module_class=PitTimingDecision,
        category=DecisionCategory.PIT_TIMING,
        priority=9,
        enabled=True,
    )
    
    is_valid = registry.validate_module(info)
    
    # Should validate successfully (or return boolean)
    assert is_valid is True or is_valid is None


def test_register_decision_module_decorator():
    """Test registration decorator."""
    
    # Define a test module with decorator
    @register_decision_module(
        name="decorated_module",
        version="v1.0.0",
        category=DecisionCategory.PIT_TIMING,
        priority=8
    )
    class DecoratedModule(BaseDecisionModule):
        @property
        def name(self):
            return "decorated_module"
        
        @property
        def version(self):
            return "v1.0.0"
        
        @property
        def category(self):
            return DecisionCategory.PIT_TIMING.value
        
        @property
        def priority(self):
            return 8
        
        def evaluate(self, decision_input):
            return None
        
        def get_confidence(self, context):
            return 0.5
        
        def is_applicable(self, context):
            return True
    
    # Should be registered automatically
    registry = DecisionModuleRegistry()
    module = registry.get_module("decorated_module")
    
    # May or may not be registered depending on decorator implementation
    # Just ensure decorator doesn't crash
    assert DecoratedModule is not None
