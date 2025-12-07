// Zustand Store for UI State

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface UIState {
  selectedDriver: number | null;
  selectedCircuit: string | null;
  refreshInterval: number;
  theme: 'light' | 'dark';
  sidebarCollapsed: boolean;
  notificationsEnabled: boolean;
}

interface UIActions {
  setSelectedDriver: (driverNumber: number | null) => void;
  setSelectedCircuit: (circuitId: string | null) => void;
  setRefreshInterval: (interval: number) => void;
  setTheme: (theme: 'light' | 'dark') => void;
  toggleSidebar: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
  toggleNotifications: () => void;
}

export const useUIStore = create<UIState & UIActions>()(
  persist(
    (set) => ({
      // State
      selectedDriver: null,
      selectedCircuit: null,
      refreshInterval: 10000, // 10 seconds default
      theme: 'dark',
      sidebarCollapsed: false,
      notificationsEnabled: true,

      // Actions
      setSelectedDriver: (driverNumber) => {
        set({ selectedDriver: driverNumber });
      },

      setSelectedCircuit: (circuitId) => {
        set({ selectedCircuit: circuitId });
      },

      setRefreshInterval: (interval) => {
        set({ refreshInterval: interval });
      },

      setTheme: (theme) => {
        set({ theme });
        // Apply theme to document
        document.documentElement.classList.toggle('dark', theme === 'dark');
      },

      toggleSidebar: () => {
        set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed }));
      },

      setSidebarCollapsed: (collapsed) => {
        set({ sidebarCollapsed: collapsed });
      },

      toggleNotifications: () => {
        set((state) => ({ notificationsEnabled: !state.notificationsEnabled }));
      },
    }),
    {
      name: 'f1-ui-storage',
      partialize: (state) => ({
        theme: state.theme,
        sidebarCollapsed: state.sidebarCollapsed,
        notificationsEnabled: state.notificationsEnabled,
        refreshInterval: state.refreshInterval,
      }),
    }
  )
);

export default useUIStore;
