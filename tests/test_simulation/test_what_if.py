"""Tests for WhatIfEngine."""

import pytest

from simulation.core import RaceSimulator
from simulation.what_if import WhatIfEngine, WhatIfScenario, ScenarioType
from simulation.schemas import TireCompound, StrategyOption, PaceTarget


class TestWhatIfEngine:
    """Test WhatIfEngine initialization."""
    
    def test_init(self):
        """Test initialization."""
        simulator = RaceSimulator()
        engine = WhatIfEngine(simulator)
        
        assert len(engine.scenario_templates) == 10
    
    def test_scenario_templates(self):
        """Test all scenario templates present."""
        simulator = RaceSimulator()
        engine = WhatIfEngine(simulator)
        
        expected_scenarios = [
            ScenarioType.EARLY_SAFETY_CAR,
            ScenarioType.LATE_SAFETY_CAR,
            ScenarioType.DOUBLE_SAFETY_CAR,
            ScenarioType.UNDERCUT_ATTEMPT,
            ScenarioType.OVERCUT_ATTEMPT,
            ScenarioType.TIRE_OFFSET,
            ScenarioType.AGGRESSIVE_PACING,
            ScenarioType.CONSERVATIVE_PACING,
            ScenarioType.RAIN_TRANSITION,
            ScenarioType.TIRE_FAILURE,
        ]
        
        for scenario_type in expected_scenarios:
            assert scenario_type in engine.scenario_templates


class TestScenarioAnalysis:
    """Test scenario analysis."""
    
    def test_analyze_early_safety_car(self, sample_simulation_input):
        """Test early safety car scenario."""
        simulator = RaceSimulator()
        engine = WhatIfEngine(simulator)
        
        scenario = engine.scenario_templates[ScenarioType.EARLY_SAFETY_CAR]
        comparison = engine.analyze_scenario(sample_simulation_input, scenario)
        
        assert comparison is not None
        assert comparison.scenario.scenario_type == ScenarioType.EARLY_SAFETY_CAR
        assert isinstance(comparison.position_delta, int)
        assert isinstance(comparison.time_delta, float)
    
    def test_analyze_undercut_attempt(self, sample_simulation_input):
        """Test undercut attempt scenario."""
        simulator = RaceSimulator()
        engine = WhatIfEngine(simulator)
        
        scenario = engine.scenario_templates[ScenarioType.UNDERCUT_ATTEMPT]
        comparison = engine.analyze_scenario(sample_simulation_input, scenario)
        
        assert comparison is not None
        assert comparison.scenario.scenario_type == ScenarioType.UNDERCUT_ATTEMPT
    
    def test_analyze_overcut_attempt(self, sample_simulation_input):
        """Test overcut attempt scenario."""
        simulator = RaceSimulator()
        engine = WhatIfEngine(simulator)
        
        scenario = engine.scenario_templates[ScenarioType.OVERCUT_ATTEMPT]
        comparison = engine.analyze_scenario(sample_simulation_input, scenario)
        
        assert comparison is not None
        assert comparison.scenario.scenario_type == ScenarioType.OVERCUT_ATTEMPT


class TestScenarioModification:
    """Test scenario modifications."""
    
    def test_apply_early_safety_car(self, sample_simulation_input):
        """Test early SC scenario modifies input."""
        simulator = RaceSimulator()
        engine = WhatIfEngine(simulator)
        
        scenario = engine.scenario_templates[ScenarioType.EARLY_SAFETY_CAR]
        modified_input = engine._apply_scenario(sample_simulation_input, scenario)
        
        assert modified_input.what_if_params is not None
        assert "inject_safety_car" in modified_input.what_if_params
    
    def test_apply_undercut(self, sample_simulation_input):
        """Test undercut scenario modifies pit laps."""
        simulator = RaceSimulator()
        engine = WhatIfEngine(simulator)
        
        original_pit_laps = sample_simulation_input.strategy_to_evaluate.pit_laps.copy()
        
        scenario = engine.scenario_templates[ScenarioType.UNDERCUT_ATTEMPT]
        modified_input = engine._apply_scenario(sample_simulation_input, scenario)
        
        # Pit laps should be earlier
        for orig, mod in zip(original_pit_laps, modified_input.strategy_to_evaluate.pit_laps):
            assert mod <= orig
    
    def test_apply_tire_offset(self, sample_simulation_input):
        """Test tire offset scenario modifies tire sequence."""
        simulator = RaceSimulator()
        engine = WhatIfEngine(simulator)
        
        scenario = engine.scenario_templates[ScenarioType.TIRE_OFFSET]
        modified_input = engine._apply_scenario(sample_simulation_input, scenario)
        
        # First tire should be changed
        assert modified_input.strategy_to_evaluate.tire_sequence[0] == TireCompound.HARD


class TestScenarioComparison:
    """Test scenario comparison."""
    
    def test_compare_strategies_under_scenarios(self, sample_simulation_input):
        """Test comparing strategies under scenarios."""
        simulator = RaceSimulator()
        engine = WhatIfEngine(simulator)
        
        strategies = [
            StrategyOption(
                pit_laps=[25],
                tire_sequence=[TireCompound.SOFT, TireCompound.MEDIUM],
                target_pace=PaceTarget.BALANCED,
            ),
            StrategyOption(
                pit_laps=[35],
                tire_sequence=[TireCompound.MEDIUM, TireCompound.HARD],
                target_pace=PaceTarget.BALANCED,
            ),
        ]
        
        scenarios = [
            engine.scenario_templates[ScenarioType.EARLY_SAFETY_CAR],
            engine.scenario_templates[ScenarioType.LATE_SAFETY_CAR],
        ]
        
        result = engine.compare_strategies_under_scenarios(
            sample_simulation_input,
            strategies,
            scenarios,
        )
        
        assert "strategies" in result
        assert "scenarios" in result
        assert "results_matrix" in result
        assert "most_robust_strategy_index" in result


class TestRecommendations:
    """Test scenario recommendations."""
    
    def test_generate_recommendations(self, sample_simulation_input):
        """Test generating recommendations."""
        simulator = RaceSimulator()
        engine = WhatIfEngine(simulator)
        
        recommendations = engine.generate_scenario_recommendations(sample_simulation_input)
        
        assert len(recommendations) > 0
        
        for rec in recommendations:
            assert "scenario" in rec
            assert "impact" in rec
            assert "recommendation" in rec
            assert rec["impact"] in ["positive", "negative", "neutral"]
    
    def test_recommendation_confidence(self, sample_simulation_input):
        """Test recommendations have confidence levels."""
        simulator = RaceSimulator()
        engine = WhatIfEngine(simulator)
        
        recommendations = engine.generate_scenario_recommendations(sample_simulation_input)
        
        for rec in recommendations:
            assert "confidence" in rec
            assert rec["confidence"] in ["high", "moderate", "low"]


class TestScenarioType:
    """Test ScenarioType enum."""
    
    def test_all_scenario_types(self):
        """Test all scenario types exist."""
        types = [
            ScenarioType.EARLY_SAFETY_CAR,
            ScenarioType.LATE_SAFETY_CAR,
            ScenarioType.DOUBLE_SAFETY_CAR,
            ScenarioType.UNDERCUT_ATTEMPT,
            ScenarioType.OVERCUT_ATTEMPT,
            ScenarioType.TIRE_OFFSET,
            ScenarioType.AGGRESSIVE_PACING,
            ScenarioType.CONSERVATIVE_PACING,
            ScenarioType.RAIN_TRANSITION,
            ScenarioType.TIRE_FAILURE,
        ]
        
        assert len(types) == 10


class TestWhatIfScenario:
    """Test WhatIfScenario dataclass."""
    
    def test_scenario_creation(self):
        """Test scenario creation."""
        scenario = WhatIfScenario(
            scenario_type=ScenarioType.EARLY_SAFETY_CAR,
            description="Test SC scenario",
            parameters={"lap": 15},
            injection_lap=15,
        )
        
        assert scenario.scenario_type == ScenarioType.EARLY_SAFETY_CAR
        assert scenario.description == "Test SC scenario"
        assert scenario.parameters["lap"] == 15
