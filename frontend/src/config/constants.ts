// Configuration constants

export const API_CONFIG = {
  BASE_URL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  WS_URL: import.meta.env.VITE_WS_URL || 'ws://localhost:8000',
  VERSION: import.meta.env.VITE_API_VERSION || 'v1',
  TIMEOUT: 30000,
};

export const REFRESH_INTERVALS = {
  RACE_STATE: Number(import.meta.env.VITE_REFRESH_INTERVAL_RACE_STATE) || 5000,
  PREDICTIONS: Number(import.meta.env.VITE_REFRESH_INTERVAL_PREDICTIONS) || 10000,
  SIMULATIONS: Number(import.meta.env.VITE_REFRESH_INTERVAL_SIMULATIONS) || 30000,
  WEATHER: 300000, // 5 minutes
};

export const TIRE_COMPOUNDS = {
  SOFT: { color: '#FF0000', label: 'Soft', shortCode: 'S' },
  MEDIUM: { color: '#FFFF00', label: 'Medium', shortCode: 'M' },
  HARD: { color: '#FFFFFF', label: 'Hard', shortCode: 'H' },
  INTERMEDIATE: { color: '#00FF00', label: 'Intermediate', shortCode: 'I' },
  WET: { color: '#0000FF', label: 'Wet', shortCode: 'W' },
} as const;

export const TRAFFIC_LIGHT_COLORS = {
  GREEN: '#10B981',
  AMBER: '#F59E0B',
  RED: '#EF4444',
} as const;

export const WEATHER_CONDITIONS = {
  DRY: { icon: '☀️', label: 'Dry' },
  LIGHT_RAIN: { icon: '🌦️', label: 'Light Rain' },
  HEAVY_RAIN: { icon: '🌧️', label: 'Heavy Rain' },
  MIXED: { icon: '⛅', label: 'Mixed' },
} as const;

export const DRIVER_NUMBERS = [
  1, 2, 3, 4, 10, 11, 14, 16, 18, 20, 21, 22, 23, 24, 27, 31, 44, 55, 63, 81,
] as const;

export const F1_TEAMS = {
  'Red Bull Racing': { color: '#0600EF', shortName: 'RBR' },
  'Ferrari': { color: '#DC0000', shortName: 'FER' },
  'Mercedes': { color: '#00D2BE', shortName: 'MER' },
  'McLaren': { color: '#FF8700', shortName: 'MCL' },
  'Aston Martin': { color: '#006F62', shortName: 'AMR' },
  'Alpine': { color: '#0090FF', shortName: 'ALP' },
  'Williams': { color: '#005AFF', shortName: 'WIL' },
  'AlphaTauri': { color: '#2B4562', shortName: 'AT' },
  'Alfa Romeo': { color: '#900000', shortName: 'ARR' },
  'Haas': { color: '#FFFFFF', shortName: 'HAS' },
} as const;
