import { useState, useEffect, useCallback } from 'react';
import { wsService } from '@/lib/websocket';
import { WebSocketContextType } from '@/types';

export function useWebSocket(userId?: string): WebSocketContextType {
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    if (userId) {
      wsService.connect(userId);
      
      // Set up connection status tracking
      const checkConnection = () => {
        setIsConnected(wsService.getConnectionStatus());
      };

      const interval = setInterval(checkConnection, 1000);
      checkConnection();

      return () => {
        clearInterval(interval);
      };
    }
  }, [userId]);

  const connect = useCallback(() => {
    wsService.connect(userId);
  }, [userId]);

  const disconnect = useCallback(() => {
    wsService.disconnect();
    setIsConnected(false);
  }, []);

  const emit = useCallback((event: string, data: any) => {
    wsService.emit(event, data);
  }, []);

  const on = useCallback((event: string, callback: (data: any) => void) => {
    wsService.on(event, callback);
  }, []);

  const off = useCallback((event: string, callback?: (data: any) => void) => {
    wsService.off(event, callback);
  }, []);

  return {
    socket: wsService,
    isConnected,
    connect,
    disconnect,
    emit,
    on,
    off,
  };
}
