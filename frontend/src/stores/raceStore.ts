// Zustand Store for Race State

import { create } from 'zustand';
import { RaceState, DriverState, WebSocketMessage } from '@/types/race.types';

interface RaceStoreState {
  raceState: RaceState | null;
  selectedDriverNumber: number | null;
  isLive: boolean;
  lastUpdate: Date | null;
}

interface RaceStoreActions {
  setRaceState: (state: RaceState) => void;
  updateRaceState: (update: Partial<RaceState>) => void;
  updateDriverState: (driverNumber: number, update: Partial<DriverState>) => void;
  setSelectedDriver: (driverNumber: number) => void;
  handleWebSocketMessage: (message: WebSocketMessage) => void;
  reset: () => void;
}

export const useRaceStore = create<RaceStoreState & RaceStoreActions>((set, get) => ({
  // State
  raceState: null,
  selectedDriverNumber: null,
  isLive: false,
  lastUpdate: null,

  // Actions
  setRaceState: (state: RaceState) => {
    set({
      raceState: state,
      isLive: true,
      lastUpdate: new Date(),
    });
  },

  updateRaceState: (update: Partial<RaceState>) => {
    const currentState = get().raceState;
    if (!currentState) return;

    set({
      raceState: { ...currentState, ...update },
      lastUpdate: new Date(),
    });
  },

  updateDriverState: (driverNumber: number, update: Partial<DriverState>) => {
    const currentState = get().raceState;
    if (!currentState) return;

    const updatedDrivers = currentState.drivers.map((driver) =>
      driver.driver_number === driverNumber ? { ...driver, ...update } : driver
    );

    set({
      raceState: { ...currentState, drivers: updatedDrivers },
      lastUpdate: new Date(),
    });
  },

  setSelectedDriver: (driverNumber: number) => {
    set({ selectedDriverNumber: driverNumber });
  },

  handleWebSocketMessage: (message: WebSocketMessage) => {
    const { raceState } = get();

    switch (message.type) {
      case 'RACE_STATE_UPDATE':
        set({
          raceState: message.data,
          isLive: true,
          lastUpdate: new Date(),
        });
        break;

      case 'LAP_COMPLETED':
        if (raceState) {
          const updatedDrivers = raceState.drivers.map((driver) =>
            driver.driver_number === message.data.driver_number
              ? {
                  ...driver,
                  laps_completed: message.data.lap_number,
                  last_lap_time: message.data.lap_time,
                  current_position: message.data.position,
                }
              : driver
          );

          set({
            raceState: { ...raceState, drivers: updatedDrivers },
            lastUpdate: new Date(),
          });
        }
        break;

      case 'PIT_STOP':
        if (raceState) {
          const updatedDrivers = raceState.drivers.map((driver) =>
            driver.driver_number === message.data.driver_number
              ? {
                  ...driver,
                  tire_compound: message.data.tire_compound_in,
                  tire_age: 0,
                  pit_stops: driver.pit_stops + 1,
                }
              : driver
          );

          set({
            raceState: { ...raceState, drivers: updatedDrivers },
            lastUpdate: new Date(),
          });
        }
        break;

      case 'SAFETY_CAR':
        if (raceState) {
          set({
            raceState: {
              ...raceState,
              safety_car_active: message.data.deployed,
            },
            lastUpdate: new Date(),
          });
        }
        break;

      case 'STRATEGY_RECOMMENDATION':
        // Handle strategy recommendation if needed
        console.log('[Race Store] Strategy recommendation', message.data);
        break;
    }
  },

  reset: () => {
    set({
      raceState: null,
      selectedDriverNumber: null,
      isLive: false,
      lastUpdate: null,
    });
  },
}));

export default useRaceStore;
