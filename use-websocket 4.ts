import { useState, useEffect } from 'react';
import { wsClient } from '@/lib/websocket';

export const useWebSocket = () => {
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const handleConnect = () => setIsConnected(true);
    const handleDisconnect = () => setIsConnected(false);

    wsClient.onConnect(handleConnect);
    wsClient.onDisconnect(handleDisconnect);

    setIsConnected(wsClient.isConnected());

    return () => {
      // Cleanup handled by WebSocket client
    };
  }, []);

  const subscribe = (messageType: string, handler: (data: any) => void) => {
    wsClient.subscribe(messageType, handler);
  };

  const unsubscribe = (messageType: string) => {
    wsClient.unsubscribe(messageType);
  };

  const send = (type: string, data: any) => {
    wsClient.send(type, data);
  };

  return {
    isConnected,
    subscribe,
    unsubscribe,
    send
  };
};
