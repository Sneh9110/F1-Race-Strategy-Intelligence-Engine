// Simulation API types matching backend schemas (api/schemas/simulation.py)

import { MetadataBase } from './api.types';

// Strategy Simulation Schemas
export interface StrategySimulationRequest {
  circuit_name: string;
  total_laps: number;
  starting_tire: string;
  fuel_load: number;
  weather_condition?: string;
  pit_stops?: Array<{ lap: number; tire: string }>;
}

export interface StrategySimulationResponse {
  total_race_time: number; // in seconds
  final_position: number;
  pit_stop_count: number;
  tire_strategy: string[];
  lap_times?: number[];
  metadata: MetadataBase;
}

// Compare Strategies Schemas
export interface CompareStrategiesRequest {
  circuit_name: string;
  total_laps: number;
  strategies: Array<{
    name: string;
    pit_stops: Array<{ lap: number; tire: string }>;
  }>;
}

export interface CompareStrategiesResponse {
  best_strategy: string;
  comparisons: Array<Record<string, any>>;
  time_differences: Record<string, number>;
  metadata: MetadataBase;
}

// Monte Carlo Simulation (if needed for future)
export interface MonteCarloRequest {
  circuit_name: string;
  total_laps: number;
  starting_tire: string;
  strategies: Array<{
    name: string;
    pit_stops: Array<{ lap: number; tire: string }>;
  }>;
  num_simulations?: number;
  safety_car_probability?: number;
}

export interface MonteCarloResult {
  strategy_name: string;
  avg_position: number;
  position_distribution: Record<string, number>;
  win_probability: number;
  podium_probability: number;
  points_probability: number;
  avg_race_time: number;
  time_std_dev: number;
  risk_score: number;
}

export interface MonteCarloResponse {
  results: MonteCarloResult[];
  best_strategy: string;
  confidence_level: number;
  simulations_run: number;
}
