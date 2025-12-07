/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string;
  readonly VITE_WS_URL: string;
  readonly VITE_API_VERSION: string;
  readonly VITE_REFRESH_INTERVAL_RACE_STATE: string;
  readonly VITE_REFRESH_INTERVAL_PREDICTIONS: string;
  readonly VITE_REFRESH_INTERVAL_SIMULATIONS: string;
  readonly VITE_ENABLE_WEBSOCKET: string;
  readonly VITE_ENABLE_MOCK_DATA: string;
  readonly DEV: boolean; // Vite built-in environment variable
  readonly PROD: boolean;
  readonly MODE: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
