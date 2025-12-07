// Custom hooks for predictions using TanStack Query

import { useQuery, UseQueryResult } from '@tanstack/react-query';
import predictionService from '@/services/predictionService';
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

/**
 * Hook for lap time prediction
 */
export function useLapTimePrediction(
  request: LapTimePredictionRequest,
  enabled = true
): UseQueryResult<LapTimePredictionResponse> {
  return useQuery({
    queryKey: ['prediction', 'laptime', request],
    queryFn: () => predictionService.predictLapTime(request),
    staleTime: 60000, // 60s cache (matches API TTL)
    enabled,
  });
}

/**
 * Hook for tire degradation prediction
 */
export function useTireDegradation(
  request: TireDegradationRequest,
  enabled = true
): UseQueryResult<TireDegradationResponse> {
  return useQuery({
    queryKey: ['prediction', 'degradation', request],
    queryFn: () => predictionService.predictTireDegradation(request),
    staleTime: 60000,
    enabled,
  });
}

/**
 * Hook for safety car probability
 */
export function useSafetyCarProbability(
  request: SafetyCarRequest,
  options: { enabled?: boolean; refetchInterval?: number } = {}
): UseQueryResult<SafetyCarResponse> {
  const { enabled = true, refetchInterval = 10000 } = options;

  return useQuery({
    queryKey: ['prediction', 'safety-car', request],
    queryFn: () => predictionService.predictSafetyCar(request),
    staleTime: 10000,
    refetchInterval,
    enabled,
  });
}

/**
 * Hook for pit stop loss prediction
 */
export function usePitStopLoss(
  request: PitStopLossRequest,
  enabled = true
): UseQueryResult<PitStopLossResponse> {
  return useQuery({
    queryKey: ['prediction', 'pit-stop-loss', request],
    queryFn: () => predictionService.predictPitStopLoss(request),
    staleTime: 60000,
    enabled,
  });
}

/**
 * Hook for prediction statistics
 */
export function usePredictionStats(): UseQueryResult<PredictionStatsResponse> {
  return useQuery({
    queryKey: ['prediction', 'stats'],
    queryFn: () => predictionService.getPredictionStats(),
    staleTime: 300000, // 5 minutes
  });
}
