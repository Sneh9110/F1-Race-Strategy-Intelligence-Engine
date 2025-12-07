// Race State Service

import apiClient from './apiClient';
import { RaceState } from '@/types/race.types';

export const raceStateService = {
  /**
   * Get current race state
   */
  async getRaceState(sessionId: string): Promise<RaceState> {
    const response = await apiClient.get<RaceState>(`/api/v1/race/state/${sessionId}`);
    return response.data;
  },

  /**
   * Create/update race state
   */
  async updateRaceState(sessionId: string, state: Partial<RaceState>): Promise<RaceState> {
    const response = await apiClient.post<RaceState>(`/api/v1/race/state/${sessionId}`, state);
    return response.data;
  },

  /**
   * Delete race state
   */
  async deleteRaceState(sessionId: string): Promise<void> {
    await apiClient.delete(`/api/v1/race/state/${sessionId}`);
  },
};

export default raceStateService;
