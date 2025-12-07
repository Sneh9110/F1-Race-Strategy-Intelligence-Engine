// Health Service - Updated to match backend API

import apiClient from './apiClient';
import {
  HealthCheckResponse,
  ReadinessResponse,
  LivenessResponse,
  MetricsResponse,
} from '@/types/api.types';

export const healthService = {
  /**
   * Check API health
   */
  async checkHealth(): Promise<HealthCheckResponse> {
    const response = await apiClient.get<HealthCheckResponse>('/api/v1/health');
    return response.data;
  },

  /**
   * Check API readiness
   */
  async checkReadiness(): Promise<ReadinessResponse> {
    const response = await apiClient.get<ReadinessResponse>('/api/v1/health/ready');
    return response.data;
  },

  /**
   * Check API liveness
   */
  async checkLiveness(): Promise<LivenessResponse> {
    const response = await apiClient.get<LivenessResponse>('/api/v1/health/live');
    return response.data;
  },

  /**
   * Get API metrics (Prometheus text format)
   */
  async getMetrics(): Promise<MetricsResponse> {
    const response = await apiClient.get<string>('/api/v1/metrics', {
      headers: {
        'Accept': 'text/plain',
      },
    });
    return response.data;
  },
};

export default healthService;
