// WebSocket Manager for real-time race updates

import { API_CONFIG } from '@/config/constants';
import { WebSocketMessage } from '@/types/race.types';

type MessageHandler = (message: WebSocketMessage) => void;
type ConnectionStatusHandler = (connected: boolean) => void;

export class WebSocketManager {
  private ws: WebSocket | null = null;
  private sessionId: string | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelay = 1000; // Start with 1 second
  private messageHandlers: Set<MessageHandler> = new Set();
  private statusHandlers: Set<ConnectionStatusHandler> = new Set();
  private heartbeatInterval: number | null = null; // Changed from NodeJS.Timeout to number for browser compatibility
  private isManuallyDisconnected = false;

  /**
   * Connect to WebSocket
   */
  connect(sessionId: string, token?: string): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      console.warn('[WebSocket] Already connected');
      return;
    }

    this.sessionId = sessionId;
    this.isManuallyDisconnected = false;

    // Build WebSocket URL with optional token
    const wsUrl = `${API_CONFIG.WS_URL}/ws/race/${sessionId}${token ? `?token=${token}` : ''}`;

    console.log('[WebSocket] Connecting to', wsUrl);

    try {
      this.ws = new WebSocket(wsUrl);
      this.setupEventHandlers();
    } catch (error) {
      console.error('[WebSocket] Connection error', error);
      this.handleReconnect();
    }
  }

  /**
   * Disconnect WebSocket
   */
  disconnect(): void {
    this.isManuallyDisconnected = true;
    this.stopHeartbeat();

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.notifyStatusHandlers(false);
    console.log('[WebSocket] Disconnected');
  }

  /**
   * Subscribe to messages
   */
  onMessage(handler: MessageHandler): () => void {
    this.messageHandlers.add(handler);
    
    // Return unsubscribe function
    return () => {
      this.messageHandlers.delete(handler);
    };
  }

  /**
   * Subscribe to connection status changes
   */
  onStatusChange(handler: ConnectionStatusHandler): () => void {
    this.statusHandlers.add(handler);
    
    // Return unsubscribe function
    return () => {
      this.statusHandlers.delete(handler);
    };
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  /**
   * Send message to server
   */
  send(message: any): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('[WebSocket] Cannot send message, not connected');
    }
  }

  /**
   * Setup WebSocket event handlers
   */
  private setupEventHandlers(): void {
    if (!this.ws) return;

    this.ws.onopen = () => {
      console.log('[WebSocket] Connected');
      this.reconnectAttempts = 0;
      this.reconnectDelay = 1000;
      this.startHeartbeat();
      this.notifyStatusHandlers(true);
    };

    this.ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        
        if (import.meta.env.DEV) {
          console.log('[WebSocket] Message received', message.type, message);
        }

        // Notify all handlers
        this.messageHandlers.forEach((handler) => {
          try {
            handler(message);
          } catch (error) {
            console.error('[WebSocket] Handler error', error);
          }
        });
      } catch (error) {
        console.error('[WebSocket] Failed to parse message', error);
      }
    };

    this.ws.onerror = (error) => {
      console.error('[WebSocket] Error', error);
    };

    this.ws.onclose = (event) => {
      console.log('[WebSocket] Closed', event.code, event.reason);
      this.stopHeartbeat();
      this.notifyStatusHandlers(false);

      // Attempt reconnection if not manually disconnected
      if (!this.isManuallyDisconnected) {
        this.handleReconnect();
      }
    };
  }

  /**
   * Handle reconnection with exponential backoff
   */
  private handleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[WebSocket] Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    
    // Exponential backoff: 1s, 2s, 4s, 8s, ..., max 30s
    const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1), 30000);

    console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

    setTimeout(() => {
      if (this.sessionId && !this.isManuallyDisconnected) {
        const token = localStorage.getItem('f1_auth_token');
        this.connect(this.sessionId, token || undefined);
      }
    }, delay);
  }

  /**
   * Start heartbeat/ping
   */
  private startHeartbeat(): void {
    this.stopHeartbeat();
    
    this.heartbeatInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.send({ type: 'ping', timestamp: Date.now() });
      }
    }, 30000); // Send ping every 30 seconds
  }

  /**
   * Stop heartbeat
   */
  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  /**
   * Notify status handlers
   */
  private notifyStatusHandlers(connected: boolean): void {
    this.statusHandlers.forEach((handler) => {
      try {
        handler(connected);
      } catch (error) {
        console.error('[WebSocket] Status handler error', error);
      }
    });
  }
}

// Singleton instance
let wsManager: WebSocketManager | null = null;

export function getWebSocketManager(): WebSocketManager {
  if (!wsManager) {
    wsManager = new WebSocketManager();
  }
  return wsManager;
}

export default getWebSocketManager;
