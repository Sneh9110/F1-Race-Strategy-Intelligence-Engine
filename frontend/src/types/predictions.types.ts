// Prediction API types matching backend schemas (api/schemas/predictions.py)

import { MetadataBase } from './api.types';

export type TireCompound = 'SOFT' | 'MEDIUM' | 'HARD' | 'INTERMEDIATE' | 'WET';

// Lap Time Prediction Schemas
export interface LapTimePredictionRequest {
  circuit_name: string;
  driver: string;
  team: string;
  tire_compound: string; // SOFT, MEDIUM, HARD
  tire_age: number;
  fuel_load: number;
  track_temp: number;
  air_temp: number;
  weather_condition?: string; // Dry, Light Rain, Heavy Rain
}

export interface LapTimePredictionResponse {
  predicted_lap_time: number; // in seconds
  confidence: number; // 0-1
  metadata: MetadataBase;
}

// Tire Degradation Prediction Schemas
export interface TireDegradationRequest {
  circuit_name: string;
  tire_compound: string;
  laps: number;
  track_temp: number;
  fuel_load: number;
  downforce_level?: string; // HIGH, MEDIUM, LOW
}

export interface TireDegradationResponse {
  degradation_per_lap: number; // % degradation per lap
  total_degradation: number; // Total % degradation
  remaining_performance: number; // Remaining tire performance %
  metadata: MetadataBase;
}

// Safety Car Prediction Schemas
export interface SafetyCarRequest {
  circuit_name: string;
  lap: number;
  total_laps: number;
  weather_condition: string;
  incidents_so_far?: number;
}

export interface SafetyCarResponse {
  probability: number; // 0-1
  risk_level: string; // LOW, MEDIUM, HIGH
  metadata: MetadataBase;
}

// Pit Stop Loss Prediction Schemas
export interface PitStopLossRequest {
  circuit_name: string;
  pit_lane_type?: string; // Standard or Short
  traffic_density?: number; // 0-1
}

export interface PitStopLossResponse {
  time_loss: number; // Expected time loss in seconds
  range_min: number; // Minimum expected time loss
  range_max: number; // Maximum expected time loss
  metadata: MetadataBase;
}

// Batch Prediction Schemas
export interface BatchPredictionRequest {
  predictions: Record<string, any>[];
}

export interface BatchPredictionResponse {
  results: Record<string, any>[];
  total: number;
  successful: number;
}

export interface PredictionStatsResponse {
  total_predictions: number;
  avg_confidence: number;
  cache_hit_rate: number;
  predictions_by_type: Record<string, number>;
}
