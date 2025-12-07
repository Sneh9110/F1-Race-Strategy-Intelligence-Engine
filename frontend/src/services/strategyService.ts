// Strategy Service - Updated to match backend API schemas

import apiClient from './apiClient';
import { APIResponse } from '@/types/api.types';
import {
  DecisionRequest,
  DecisionResponse,
  ModuleListResponse,
} from '@/types/strategy.types';

export const strategyService = {
  /**
   * Get strategy recommendation
   */
  async getRecommendation(request: DecisionRequest): Promise<DecisionResponse> {
    const response = await apiClient.post<APIResponse<DecisionResponse>>(
      '/api/v1/strategy/recommend',
      request
    );
    return response.data.data;
  },

  /**
   * List available decision modules
   */
  async listModules(): Promise<ModuleListResponse> {
    const response = await apiClient.get<ModuleListResponse>('/api/v1/strategy/modules');
    return response.data;
  },
};

export default strategyService;
