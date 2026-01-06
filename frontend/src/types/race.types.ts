// Race State types

import { TireCompound } from './predictions.types';

export interface LapData {
  lap_number: number;
  lap_time: number;
  sector_1: number;
  sector_2: number;
  sector_3: number;
  position: number;
  tire_compound: TireCompound;
  tire_age: number;
  fuel_load: number;
}

export interface StintData {
  stint_number: number;
  tire_compound: TireCompound;
  start_lap: number;
  end_lap: number;
  laps_completed: number;
  avg_lap_time: number;
  degradation_rate: number;
  pit_stop_duration?: number;
}

export interface DriverState {
  driver_number: number;
  driver_name: string;
  team: string;
  current_position: number;
  laps_completed: number;
  tire_compound: TireCompound;
  tire_age: number;
  fuel_load: number;
  gap_to_leader: number;
  gap_to_next: number;
  gap_to_previous: number;
  last_lap_time: number;
  best_lap_time: number;
  pit_stops: number;
  stints: StintData[];
  recent_laps: LapData[];
}

export interface RaceState {
  session_id: string;
  session_type: 'PRACTICE' | 'QUALIFYING' | 'SPRINT' | 'RACE';
  circuit_id: string;
  circuit_name: string;
  current_lap: number;
  total_laps: number;
  safety_car_active: boolean;
  virtual_safety_car_active: boolean;
  red_flag_active: boolean;
  weather_condition: WeatherCondition;
  track_temp: number;
  air_temp: number;
  humidity: number;
  rain_probability: number;
  wind_speed: number;
  wind_direction: number;
  drivers: DriverState[];
  last_update: string;
}

export interface RaceStateUpdate {
  type: 'RACE_STATE_UPDATE';
  data: RaceState;
}

export interface LapCompletedEvent {
  type: 'LAP_COMPLETED';
  data: {
    driver_number: number;
    lap_number: number;
    lap_time: number;
    position: number;
  };
}

export interface PitStopEvent {
  type: 'PIT_STOP';
  data: {
    driver_number: number;
    lap_number: number;
    duration: number;
    tire_compound_out: TireCompound;
    tire_compound_in: TireCompound;
  };
}

export interface SafetyCarEvent {
  type: 'SAFETY_CAR';
  data: {
    deployed: boolean;
    lap_number: number;
    reason?: string;
  };
}

export interface StrategyRecommendationEvent {
  type: 'STRATEGY_RECOMMENDATION';
  data: {
    driver_number: number;
    recommendation: string;
    confidence: number;
    traffic_light: string;
  };
}

export type WebSocketMessage =
  | RaceStateUpdate
  | LapCompletedEvent
  | PitStopEvent
  | SafetyCarEvent
  | StrategyRecommendationEvent;
