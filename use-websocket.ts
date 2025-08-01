import { useEffect, useState, useRef } from 'react';
import { WebSocketManager } from '@/lib/websocket';

export function useWebSocket() {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<string | null>(null);
  const wsManager = useRef<WebSocketManager | null>(null);

  useEffect(() => {
    // Determine WebSocket URL
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    // Create WebSocket manager
    wsManager.current = new WebSocketManager(wsUrl);

    // Set up event listeners
    wsManager.current.on('connect', () => {
      setIsConnected(true);
    });

    wsManager.current.on('disconnect', () => {
      setIsConnected(false);
    });

    wsManager.current.on('message', (data) => {
      setLastMessage(JSON.stringify(data));
    });

    wsManager.current.on('error', (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    });

    // Connect
    wsManager.current.connect().catch(error => {
      console.error('Failed to connect to WebSocket:', error);
    });

    // Cleanup on unmount
    return () => {
      if (wsManager.current) {
        wsManager.current.disconnect();
      }
    };
  }, []);

  const sendMessage = (data: any) => {
    if (wsManager.current) {
      wsManager.current.send(data);
    }
  };

  return {
    isConnected,
    lastMessage,
    sendMessage,
  };
}
