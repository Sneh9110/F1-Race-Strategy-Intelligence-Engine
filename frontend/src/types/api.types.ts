// Core API types matching backend schemas (api/schemas/common.py)

export interface MetadataBase {
  request_id: string;
  timestamp: string;
  latency_ms: number;
  cache_hit?: boolean;
}

export interface APIResponse<T> {
  data: T;
  metadata: MetadataBase;
}

export type ErrorCode =
  | 'validation_error'
  | 'authentication_error'
  | 'authorization_error'
  | 'not_found'
  | 'rate_limit_exceeded'
  | 'timeout'
  | 'internal_error'
  | 'service_unavailable'
  | 'bad_request';

export interface ErrorResponse {
  error_code: ErrorCode;
  message: string; // Changed from 'detail' to 'message' to match backend
  details?: Record<string, any>;
  timestamp: string;
  request_id?: string;
}

export type ComponentStatus = 'healthy' | 'degraded' | 'unhealthy';

export interface ComponentHealth {
  status: ComponentStatus;
  latency_ms?: number;
  error_rate?: number;
  message?: string;
}

export interface HealthCheckResponse {
  status: ComponentStatus;
  version: string;
  uptime_seconds: number; // Added to match backend
  components: Record<string, ComponentHealth>; // Changed to ComponentHealth
  timestamp: string;
}

export interface ReadinessResponse {
  ready: boolean;
  checks: Record<string, boolean>;
}

export interface LivenessResponse {
  alive: boolean;
}

// Metrics returned as plain text (Prometheus format)
export type MetricsResponse = string;

