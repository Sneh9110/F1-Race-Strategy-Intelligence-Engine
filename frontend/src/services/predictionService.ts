// Prediction Service - Updated to match backend API schemas

import apiClient from './apiClient';
import { APIResponse } from '@/types/api.types';
import {
  LapTimePredictionRequest,
  LapTimePredictionResponse,
  TireDegradationRequest,
  TireDegradationResponse,
  SafetyCarRequest,
  SafetyCarResponse,
  PitStopLossRequest,
  PitStopLossResponse,
  PredictionStatsResponse,
} from '@/types/predictions.types';

export const predictionService = {
  /**
   * Predict lap time
   */
  async predictLapTime(request: LapTimePredictionRequest): Promise<LapTimePredictionResponse> {
    const response = await apiClient.post<APIResponse<LapTimePredictionResponse>>(
      '/api/v1/predict/laptime',
      request
    );
    return response.data.data;
  },

  /**
   * Predict tire degradation
   */
  async predictTireDegradation(
    request: TireDegradationRequest
  ): Promise<TireDegradationResponse> {
    const response = await apiClient.post<APIResponse<TireDegradationResponse>>(
      '/api/v1/predict/degradation',
      request
    );
    return response.data.data;
  },

  /**
   * Predict safety car probability
   */
  async predictSafetyCar(request: SafetyCarRequest): Promise<SafetyCarResponse> {
    const response = await apiClient.post<APIResponse<SafetyCarResponse>>(
      '/api/v1/predict/safety-car',
      request
    );
    return response.data.data;
  },

  /**
   * Predict pit stop loss
   */
  async predictPitStopLoss(request: PitStopLossRequest): Promise<PitStopLossResponse> {
    const response = await apiClient.post<APIResponse<PitStopLossResponse>>(
      '/api/v1/predict/pit-stop-loss',
      request
    );
    return response.data.data;
  },

  /**
   * Get prediction statistics
   */
  async getPredictionStats(): Promise<PredictionStatsResponse> {
    const response = await apiClient.get<PredictionStatsResponse>('/api/v1/predict/stats');
    return response.data;
  },
};

export default predictionService;
