import { useEffect } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { useWebSocket } from './use-websocket';
import { WebSocketMessage } from '@shared/schema';

export function useRealTimeData() {
  const queryClient = useQueryClient();

  const handleWebSocketMessage = (message: WebSocketMessage) => {
    switch (message.type) {
      case 'agent_update':
        // Invalidate agent-related queries
        queryClient.invalidateQueries({ queryKey: ['/api/agents'] });
        queryClient.invalidateQueries({ queryKey: ['/api/dashboard'] });
        
        // Update specific agent if we have the data
        if (message.data?.id) {
          queryClient.setQueryData(['/api/agents', message.data.id], message.data);
        }
        break;

      case 'system_metrics':
        // Update system metrics queries
        queryClient.invalidateQueries({ queryKey: ['/api/monitoring/system-metrics'] });
        queryClient.invalidateQueries({ queryKey: ['/api/monitoring/health'] });
        queryClient.invalidateQueries({ queryKey: ['/api/dashboard'] });
        
        // Set the latest metrics data
        queryClient.setQueryData(['/api/monitoring/system-metrics/latest'], {
          success: true,
          data: message.data,
          timestamp: message.timestamp
        });
        break;

      case 'activity_log':
        // Invalidate activity log queries
        queryClient.invalidateQueries({ queryKey: ['/api/monitoring/activity-logs'] });
        
        // Prepend new activity to existing data
        queryClient.setQueryData(['/api/monitoring/activity-logs'], (oldData: any) => {
          if (!oldData?.data) return oldData;
          
          return {
            ...oldData,
            data: [message.data, ...oldData.data].slice(0, 50) // Keep only latest 50
          };
        });
        break;

      case 'error_log':
        // Invalidate error log queries
        queryClient.invalidateQueries({ queryKey: ['/api/monitoring/error-logs'] });
        queryClient.invalidateQueries({ queryKey: ['/api/monitoring/health'] });
        
        // Add new error to existing data
        queryClient.setQueryData(['/api/monitoring/error-logs'], (oldData: any) => {
          if (!oldData?.data) return oldData;
          
          return {
            ...oldData,
            data: [message.data, ...oldData.data]
          };
        });
        break;

      case 'learning_update':
        // Invalidate learning-related queries
        queryClient.invalidateQueries({ queryKey: ['/api/learning/metrics'] });
        queryClient.invalidateQueries({ queryKey: ['/api/learning/activities'] });
        queryClient.invalidateQueries({ queryKey: ['/api/learning/overview'] });
        queryClient.invalidateQueries({ queryKey: ['/api/dashboard'] });
        
        // Update learning activities if it's an activity
        if (message.data?.activity_type) {
          queryClient.setQueryData(['/api/learning/activities'], (oldData: any) => {
            if (!oldData?.data) return oldData;
            
            return {
              ...oldData,
              data: [message.data, ...oldData.data].slice(0, 100) // Keep latest 100
            };
          });
        }
        break;

      default:
        console.log('Unknown WebSocket message type:', message.type);
    }
  };

  const handleConnect = () => {
    console.log('✅ Real-time connection established');
    
    // Refresh all data when reconnecting
    queryClient.invalidateQueries();
  };

  const handleDisconnect = () => {
    console.log('❌ Real-time connection lost');
  };

  const handleError = (error: Event) => {
    console.error('❌ Real-time connection error:', error);
  };

  // Determine WebSocket URL based on current location
  const getWebSocketUrl = () => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    return `${protocol}//${host}/ws`;
  };

  const { isConnected, connectionStatus, reconnectAttempts } = useWebSocket(
    getWebSocketUrl(),
    {
      onMessage: handleWebSocketMessage,
      onConnect: handleConnect,
      onDisconnect: handleDisconnect,
      onError: handleError,
      reconnectInterval: 3000,
      maxReconnectAttempts: 10
    }
  );

  // Log connection status changes
  useEffect(() => {
    console.log(`🔌 WebSocket status: ${connectionStatus}${reconnectAttempts > 0 ? ` (attempt ${reconnectAttempts})` : ''}`);
  }, [connectionStatus, reconnectAttempts]);

  return {
    isConnected,
    connectionStatus,
    reconnectAttempts
  };
}
