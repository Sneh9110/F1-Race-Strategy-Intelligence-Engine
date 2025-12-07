// Custom hooks for race state

import { useQuery, useMutation, useQueryClient, UseQueryResult } from '@tanstack/react-query';
import raceStateService from '@/services/raceStateService';
import { RaceState } from '@/types/race.types';
import { useRaceStore } from '@/stores/raceStore';
import { useEffect } from 'react';

/**
 * Hook for fetching race state (fallback when WebSocket disconnected)
 */
export function useRaceState(
  sessionId: string | null,
  options: { enabled?: boolean; refetchInterval?: number | false } = {}
): UseQueryResult<RaceState> {
  const { enabled = true, refetchInterval = 5000 } = options;
  const setRaceState = useRaceStore((state) => state.setRaceState);

  const query = useQuery({
    queryKey: ['raceState', sessionId],
    queryFn: () => raceStateService.getRaceState(sessionId!),
    staleTime: 3000,
    refetchInterval,
    enabled: enabled && !!sessionId,
  });

  // Update store when data changes
  useEffect(() => {
    if (query.data) {
      setRaceState(query.data);
    }
  }, [query.data, setRaceState]);

  return query;
}

/**
 * Hook for updating race state
 */
export function useUpdateRaceState(sessionId: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (state: Partial<RaceState>) =>
      raceStateService.updateRaceState(sessionId, state),
    onSuccess: () => {
      // Invalidate race state query
      queryClient.invalidateQueries({ queryKey: ['raceState', sessionId] });
    },
  });
}

/**
 * Hook for deleting race state
 */
export function useDeleteRaceState(sessionId: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () => raceStateService.deleteRaceState(sessionId),
    onSuccess: () => {
      // Invalidate race state query
      queryClient.invalidateQueries({ queryKey: ['raceState', sessionId] });
    },
  });
}
