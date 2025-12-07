// Strategy Decision API types matching backend schemas (api/schemas/simulation.py)

import { MetadataBase } from './api.types';

export enum TrafficLight {
  GREEN = 'GREEN',
  AMBER = 'AMBER',
  RED = 'RED',
}

// Decision Request/Response matching backend
export interface DecisionRequest {
  circuit_name: string;
  current_lap: number;
  total_laps: number;
  current_position: number;
  current_tire: string;
  tire_age: number;
  fuel_remaining: number;
  gap_to_leader?: number;
  gap_to_next?: number;
  weather_condition?: string;
  safety_car_deployed?: boolean;
}

export interface DecisionResponse {
  recommendation: string;
  confidence: number;
  reasoning: string;
  alternative_options: string[];
  risk_assessment: string; // LOW, MEDIUM, HIGH
  metadata: MetadataBase;
}

// Decision Module Information
export interface DecisionModule {
  name: string;
  priority: number;
  enabled: boolean;
  description: string;
}

export interface ModuleListResponse {
  modules: DecisionModule[];
  total_enabled: number;
}
