// Auth types

export interface LoginRequest {
  username: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  refresh_token?: string;
  token_type: string;
  expires_in: number;
}

export interface UserInfo {
  id: string;
  username: string;
  email?: string;
  full_name?: string;
  roles: string[];
}

export interface APIKeyRequest {
  name: string;
  expiry_days?: number;
}

export interface APIKeyResponse {
  api_key: string;
  name: string;
  created_at: string;
  expires_at?: string;
}
