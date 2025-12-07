// Simulation Service - Updated to match backend API schemas

import apiClient from './apiClient';
import { APIResponse } from '@/types/api.types';
import {
  StrategySimulationRequest,
  StrategySimulationResponse,
  CompareStrategiesRequest,
  CompareStrategiesResponse,
  MonteCarloRequest,
  MonteCarloResponse,
} from '@/types/simulation.types';

export const simulationService = {
  /**
   * Simulate single strategy
   */
  async simulateStrategy(
    request: StrategySimulationRequest
  ): Promise<StrategySimulationResponse> {
    const response = await apiClient.post<APIResponse<StrategySimulationResponse>>(
      '/api/v1/simulate/strategy',
      request
    );
    return response.data.data;
  },

  /**
   * Compare multiple strategies
   */
  async compareStrategies(
    request: CompareStrategiesRequest
  ): Promise<CompareStrategiesResponse> {
    const response = await apiClient.post<APIResponse<CompareStrategiesResponse>>(
      '/api/v1/simulate/compare-strategies',
      request
    );
    return response.data.data;
  },

  /**
   * Run Monte Carlo simulation
   */
  async runMonteCarlo(request: MonteCarloRequest): Promise<MonteCarloResponse> {
    const response = await apiClient.post<APIResponse<MonteCarloResponse>>(
      '/api/v1/simulate/monte-carlo',
      request
    );
    return response.data.data;
  },
};

export default simulationService;
