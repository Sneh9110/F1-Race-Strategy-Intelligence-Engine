// API Client with interceptors

import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig } from 'axios';
import { API_CONFIG } from '@/config/constants';
import { ErrorResponse } from '@/types/api.types';

// Create axios instance
const apiClient: AxiosInstance = axios.create({
  baseURL: API_CONFIG.BASE_URL,
  timeout: API_CONFIG.TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
    'X-API-Version': API_CONFIG.VERSION,
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    // Attach JWT token from localStorage
    const token = localStorage.getItem('f1_auth_token');
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    // Generate request ID for tracking
    const requestId = generateRequestId();
    if (config.headers) {
      config.headers['X-Request-ID'] = requestId;
    }

    // Log request in development
    if (import.meta.env.DEV) {
      console.log(`[API Request] ${config.method?.toUpperCase()} ${config.url}`, {
        requestId,
        params: config.params,
        data: config.data,
      });
    }

    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => {
    // Log response in development
    if (import.meta.env.DEV) {
      console.log(`[API Response] ${response.config.url}`, {
        status: response.status,
        data: response.data,
        latency: response.headers['x-latency-ms'],
      });
    }

    // Extract data from APIResponse wrapper if present
    if (response.data && 'data' in response.data && 'metadata' in response.data) {
      return {
        ...response,
        data: response.data.data,
        metadata: response.data.metadata,
      };
    }

    return response;
  },
  async (error: AxiosError<ErrorResponse>) => {
    // Handle different error scenarios
    if (error.response) {
      const { status, data } = error.response;

      // 401 Unauthorized - redirect to login
      if (status === 401) {
        localStorage.removeItem('f1_auth_token');
        window.location.href = '/login';
        return Promise.reject(new Error('Authentication required'));
      }

      // 429 Rate Limit
      if (status === 429) {
        const retryAfter = error.response.headers['retry-after'];
        console.warn(`Rate limited. Retry after ${retryAfter}s`);
        return Promise.reject(new Error('Rate limit exceeded. Please wait and try again.'));
      }

      // 5xx Server errors - retry with exponential backoff
      if (status >= 500 && status < 600) {
        const config = error.config;
        if (config && (!config.headers || !config.headers['X-Retry-Count'])) {
          return retryRequest(config, 1);
        }
      }

      // Return formatted error - use 'message' field to match backend
      const errorMessage = data?.message || `Request failed with status ${status}`;
      return Promise.reject(new Error(errorMessage));
    }

    // Network error
    if (error.request) {
      console.error('[API Error] Network error', error.message);
      return Promise.reject(new Error('Network error. Please check your connection.'));
    }

    return Promise.reject(error);
  }
);

// Retry logic with exponential backoff
async function retryRequest(
  config: InternalAxiosRequestConfig,
  retryCount: number
): Promise<any> {
  const maxRetries = 3;
  const baseDelay = 1000; // 1 second

  if (retryCount > maxRetries) {
    return Promise.reject(new Error('Max retries exceeded'));
  }

  // Exponential backoff: 1s, 2s, 4s
  const delay = baseDelay * Math.pow(2, retryCount - 1);
  await new Promise((resolve) => setTimeout(resolve, delay));

  // Add retry count to headers
  if (!config.headers) {
    config.headers = {} as any;
  }
  config.headers['X-Retry-Count'] = retryCount.toString();

  console.log(`[API Retry] Attempt ${retryCount} for ${config.url}`);

  try {
    return await apiClient.request(config);
  } catch (error) {
    if (axios.isAxiosError(error) && error.response?.status && error.response.status >= 500) {
      return retryRequest(config, retryCount + 1);
    }
    throw error;
  }
}

// Generate unique request ID
function generateRequestId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
}

export default apiClient;
