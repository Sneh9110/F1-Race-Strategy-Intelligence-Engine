// Custom hooks for strategy recommendations

import { useQuery, UseQueryResult } from '@tanstack/react-query';
import strategyService from '@/services/strategyService';
import {
  DecisionRequest,
  DecisionResponse,
  ModuleListResponse,
} from '@/types/strategy.types';

/**
 * Hook for strategy recommendation
 */
export function useStrategyRecommendation(
  request: DecisionRequest,
  options: { enabled?: boolean; refetchInterval?: number } = {}
): UseQueryResult<DecisionResponse> {
  const { enabled = true, refetchInterval = 10000 } = options;

  return useQuery({
    queryKey: ['strategy', 'recommendation', request],
    queryFn: () => strategyService.getRecommendation(request),
    staleTime: 5000, // 5s cache (race state changes rapidly)
    refetchInterval, // Poll every 10s during race
    enabled,
  });
}

/**
 * Hook for listing decision modules
 */
export function useDecisionModules(): UseQueryResult<ModuleListResponse> {
  return useQuery({
    queryKey: ['strategy', 'modules'],
    queryFn: () => strategyService.listModules(),
    staleTime: 300000, // 5 minutes (modules don't change often)
  });
}
