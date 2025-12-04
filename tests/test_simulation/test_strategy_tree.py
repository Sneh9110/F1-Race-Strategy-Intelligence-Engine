"""Tests for StrategyTreeExplorer."""

import pytest

from simulation.core import RaceSimulator
from simulation.strategy_tree import StrategyTreeExplorer, StrategyNode
from simulation.schemas import TireCompound, PaceTarget


class TestStrategyTreeExplorer:
    """Test StrategyTreeExplorer initialization."""
    
    def test_init(self):
        """Test initialization."""
        simulator = RaceSimulator()
        explorer = StrategyTreeExplorer(simulator)
        
        assert explorer.max_workers == 4
        assert explorer.max_strategies == 50
        assert explorer.pruning_threshold == 5.0


class TestStrategyGeneration:
    """Test strategy generation."""
    
    def test_generate_one_stop_strategies(self):
        """Test 1-stop strategy generation."""
        simulator = RaceSimulator()
        explorer = StrategyTreeExplorer(simulator)
        
        nodes = explorer._generate_n_stop_strategies(
            total_laps=78,
            num_stops=1,
            available_compounds=[TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD],
        )
        
        assert len(nodes) > 0
        assert all(len(n.pit_laps) == 1 for n in nodes)
        assert all(len(n.tire_sequence) == 2 for n in nodes)
    
    def test_generate_two_stop_strategies(self):
        """Test 2-stop strategy generation."""
        simulator = RaceSimulator()
        explorer = StrategyTreeExplorer(simulator)
        
        nodes = explorer._generate_n_stop_strategies(
            total_laps=78,
            num_stops=2,
            available_compounds=[TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD],
        )
        
        assert len(nodes) > 0
        assert all(len(n.pit_laps) == 2 for n in nodes)
        assert all(len(n.tire_sequence) == 3 for n in nodes)
    
    def test_pit_lap_ordering(self):
        """Test pit laps are in ascending order."""
        simulator = RaceSimulator()
        explorer = StrategyTreeExplorer(simulator)
        
        nodes = explorer._generate_n_stop_strategies(
            total_laps=78,
            num_stops=2,
            available_compounds=[TireCompound.SOFT, TireCompound.MEDIUM],
        )
        
        for node in nodes:
            assert node.pit_laps == sorted(node.pit_laps)


class TestStrategyExploration:
    """Test strategy exploration."""
    
    def test_explore_strategies(self, sample_simulation_input):
        """Test strategy exploration."""
        simulator = RaceSimulator()
        explorer = StrategyTreeExplorer(simulator, max_strategies=10)
        
        rankings = explorer.explore_strategies(sample_simulation_input, max_pit_stops=2)
        
        assert len(rankings) > 0
        assert len(rankings) <= 10
    
    def test_rankings_sorted(self, sample_simulation_input):
        """Test rankings sorted by expected time."""
        simulator = RaceSimulator()
        explorer = StrategyTreeExplorer(simulator, max_strategies=10)
        
        rankings = explorer.explore_strategies(sample_simulation_input, max_pit_stops=1)
        
        for i in range(len(rankings) - 1):
            assert rankings[i].expected_race_time <= rankings[i + 1].expected_race_time


class TestStrategyNode:
    """Test StrategyNode."""
    
    def test_node_creation(self):
        """Test node creation."""
        node = StrategyNode(
            pit_laps=[25],
            tire_sequence=[TireCompound.SOFT, TireCompound.MEDIUM],
        )
        
        assert len(node.pit_laps) == 1
        assert len(node.tire_sequence) == 2
        assert node.simulated is False
    
    def test_to_strategy_option(self):
        """Test conversion to strategy option."""
        node = StrategyNode(
            pit_laps=[25, 50],
            tire_sequence=[TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD],
        )
        
        strategy = node.to_strategy_option(PaceTarget.AGGRESSIVE)
        
        assert strategy.pit_laps == [25, 50]
        assert strategy.target_pace == PaceTarget.AGGRESSIVE


class TestUndercutAnalysis:
    """Test undercut analysis."""
    
    def test_undercut_window_analysis(self, sample_simulation_input):
        """Test undercut window analysis."""
        simulator = RaceSimulator()
        explorer = StrategyTreeExplorer(simulator, max_strategies=5)
        
        target_strategy = sample_simulation_input.strategy_to_evaluate
        
        results = explorer.analyze_undercut_window(
            sample_simulation_input,
            target_strategy,
            lap_range=(20, 25),
        )
        
        assert len(results) > 0
        assert all(isinstance(v, float) for v in results.values())
    
    def test_undercut_opportunity_estimation(self):
        """Test undercut opportunity estimation."""
        simulator = RaceSimulator()
        explorer = StrategyTreeExplorer(simulator)
        
        from simulation.schemas import StrategyOption
        
        early_strategy = StrategyOption(
            pit_laps=[18],
            tire_sequence=[TireCompound.SOFT, TireCompound.MEDIUM],
            target_pace=PaceTarget.BALANCED,
        )
        
        opportunity = explorer._estimate_undercut_opportunity(early_strategy)
        
        assert 0.0 <= opportunity <= 1.0
        assert opportunity > 0.5  # Early pit = high undercut opportunity


class TestOvercutAnalysis:
    """Test overcut analysis."""
    
    def test_overcut_opportunity_estimation(self):
        """Test overcut opportunity estimation."""
        simulator = RaceSimulator()
        explorer = StrategyTreeExplorer(simulator)
        
        from simulation.schemas import StrategyOption
        
        late_strategy = StrategyOption(
            pit_laps=[40],
            tire_sequence=[TireCompound.MEDIUM, TireCompound.HARD],
            target_pace=PaceTarget.BALANCED,
        )
        
        opportunity = explorer._estimate_overcut_opportunity(late_strategy)
        
        assert 0.0 <= opportunity <= 1.0
        assert opportunity > 0.5  # Late pit = high overcut opportunity


class TestTireSequences:
    """Test tire sequence generation."""
    
    def test_common_sequences(self):
        """Test common tire sequences generated."""
        simulator = RaceSimulator()
        explorer = StrategyTreeExplorer(simulator)
        
        sequences = explorer._generate_tire_sequences(
            num_stints=2,
            available_compounds=[TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD],
        )
        
        assert len(sequences) > 0
        assert all(len(seq) == 2 for seq in sequences)
        
        # Check for common sequences
        soft_medium = [TireCompound.SOFT, TireCompound.MEDIUM]
        assert soft_medium in sequences or [TireCompound.MEDIUM, TireCompound.SOFT] in sequences
