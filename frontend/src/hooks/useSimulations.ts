// Custom hooks for simulations using TanStack Query

import { useMutation, UseMutationResult } from '@tanstack/react-query';
import simulationService from '@/services/simulationService';
import {
  StrategySimulationRequest,
  StrategySimulationResponse,
  CompareStrategiesRequest,
  CompareStrategiesResponse,
  MonteCarloRequest,
  MonteCarloResponse,
} from '@/types/simulation.types';

/**
 * Hook for strategy simulation
 */
export function useStrategySimulation(): UseMutationResult<
  StrategySimulationResponse,
  Error,
  StrategySimulationRequest
> {
  return useMutation({
    mutationFn: (request: StrategySimulationRequest) =>
      simulationService.simulateStrategy(request),
  });
}

/**
 * Hook for comparing strategies
 */
export function useCompareStrategies(): UseMutationResult<
  CompareStrategiesResponse,
  Error,
  CompareStrategiesRequest
> {
  return useMutation({
    mutationFn: (request: CompareStrategiesRequest) =>
      simulationService.compareStrategies(request),
  });
}

/**
 * Hook for Monte Carlo simulation
 */
export function useMonteCarloSimulation(): UseMutationResult<
  MonteCarloResponse,
  Error,
  MonteCarloRequest
> {
  return useMutation({
    mutationFn: (request: MonteCarloRequest) =>
      simulationService.runMonteCarlo(request),
  });
}
