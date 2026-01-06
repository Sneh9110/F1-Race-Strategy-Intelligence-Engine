// Authentication Service

import apiClient from './apiClient';
import { TokenResponse, UserInfo, APIKeyResponse } from '@/types/auth.types';

const TOKEN_KEY = 'f1_auth_token';
const REFRESH_TOKEN_KEY = 'f1_refresh_token';
const USER_KEY = 'f1_user_info';

export const authService = {
  /**
   * Login with username and password
   */
  async login(username: string, password: string): Promise<{ token: string; user: UserInfo }> {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    const response = await apiClient.post<TokenResponse>('/api/v1/auth/token', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });

    const { access_token, refresh_token } = response.data;

    // Store tokens
    localStorage.setItem(TOKEN_KEY, access_token);
    if (refresh_token) {
      localStorage.setItem(REFRESH_TOKEN_KEY, refresh_token);
    }

    // Get user info
    const user = await this.getCurrentUser();

    // Store user info
    localStorage.setItem(USER_KEY, JSON.stringify(user));

    return { token: access_token, user };
  },

  /**
   * Logout user
   */
  logout(): void {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(REFRESH_TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    window.location.href = '/login';
  },

  /**
   * Refresh access token
   */
  async refreshToken(): Promise<string> {
    const refreshToken = localStorage.getItem(REFRESH_TOKEN_KEY);
    if (!refreshToken) {
      throw new Error('No refresh token available');
    }

    const response = await apiClient.post<TokenResponse>('/api/v1/auth/refresh', {
      refresh_token: refreshToken,
    });

    const { access_token } = response.data;
    localStorage.setItem(TOKEN_KEY, access_token);

    return access_token;
  },

  /**
   * Get current user info
   */
  async getCurrentUser(): Promise<UserInfo> {
    const response = await apiClient.get<UserInfo>('/api/v1/auth/me');
    return response.data;
  },

  /**
   * Check if user is authenticated
   */
  isAuthenticated(): boolean {
    const token = localStorage.getItem(TOKEN_KEY);
    if (!token) return false;

    // Check if token is expired
    try {
      const payload = JSON.parse(atob(token.split('.')[1]));
      const exp = payload.exp * 1000; // Convert to milliseconds
      return Date.now() < exp;
    } catch {
      return false;
    }
  },

  /**
   * Get stored token
   */
  getToken(): string | null {
    return localStorage.getItem(TOKEN_KEY);
  },

  /**
   * Get stored user info
   */
  getUserInfo(): UserInfo | null {
    const userJson = localStorage.getItem(USER_KEY);
    if (!userJson) return null;

    try {
      return JSON.parse(userJson);
    } catch {
      return null;
    }
  },

  /**
   * Create API key
   */
  async createApiKey(name: string, expiryDays?: number): Promise<APIKeyResponse> {
    const response = await apiClient.post<APIKeyResponse>('/api/v1/auth/api-keys', {
      name,
      expiry_days: expiryDays,
    });

    return response.data;
  },

  /**
   * Check auth status and refresh if needed
   */
  async checkAuth(): Promise<boolean> {
    if (!this.isAuthenticated()) {
      // Try to refresh token
      try {
        await this.refreshToken();
        return true;
      } catch {
        return false;
      }
    }
    return true;
  },
};

export default authService;
