// Custom hook for WebSocket connection

import { useEffect, useState, useCallback } from 'react';
import { getWebSocketManager } from '@/services/websocketManager';
import { WebSocketMessage } from '@/types/race.types';
import { useAuthStore } from '@/stores/authStore';

interface UseWebSocketOptions {
  onMessage?: (message: WebSocketMessage) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  autoConnect?: boolean;
}

export function useWebSocket(sessionId: string | null, options: UseWebSocketOptions = {}) {
  const { onMessage, onConnect, onDisconnect, autoConnect = true } = options;
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { token } = useAuthStore();

  const wsManager = getWebSocketManager();

  // Handle connection
  const connect = useCallback(() => {
    if (!sessionId) {
      setError('No session ID provided');
      return;
    }

    try {
      wsManager.connect(sessionId, token || undefined);
      setError(null);
    } catch (err: any) {
      setError(err.message || 'Failed to connect');
    }
  }, [sessionId, token, wsManager]);

  // Handle disconnection
  const disconnect = useCallback(() => {
    wsManager.disconnect();
  }, [wsManager]);

  // Setup message handler
  useEffect(() => {
    const unsubscribe = wsManager.onMessage((message) => {
      setLastMessage(message);
      onMessage?.(message);
    });

    return unsubscribe;
  }, [wsManager, onMessage]);

  // Setup status handler
  useEffect(() => {
    const unsubscribe = wsManager.onStatusChange((connected) => {
      setIsConnected(connected);
      if (connected) {
        onConnect?.();
      } else {
        onDisconnect?.();
      }
    });

    return unsubscribe;
  }, [wsManager, onConnect, onDisconnect]);

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect && sessionId) {
      connect();
    }

    return () => {
      if (autoConnect) {
        disconnect();
      }
    };
  }, [autoConnect, sessionId, connect, disconnect]);

  return {
    isConnected,
    lastMessage,
    error,
    connect,
    disconnect,
  };
}

export default useWebSocket;
