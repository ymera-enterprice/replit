import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';

// WebSocket message types
export interface WebSocketMessage {
  type: string;
  data?: any;
  fileId?: string;
}

// WebSocket context type
export interface WebSocketContextType {
  connectionStatus: 'disconnected' | 'connecting' | 'connected' | 'error';
  activeUsers: number;
  processingFiles: number;
  sendMessage: (message: WebSocketMessage) => void;
  lastMessage: WebSocketMessage | null;
}

// Create context
const WebSocketContext = createContext<WebSocketContextType | null>(null);

// Hook to use WebSocket context
export function useWebSocket(): WebSocketContextType {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
}

interface WebSocketProviderProps {
  children: ReactNode;
}

// WebSocket provider component
export function WebSocketProvider({ children }: WebSocketProviderProps) {
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected');
  const [activeUsers, setActiveUsers] = useState(0);
  const [processingFiles, setProcessingFiles] = useState(0);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [socket, setSocket] = useState<WebSocket | null>(null);

  const sendMessage = (message: WebSocketMessage) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected. Cannot send message:', message);
    }
  };

  // Initialize WebSocket connection
  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnectionStatus('connected');
      setSocket(ws);
      
      // Authenticate user
      sendMessage({
        type: 'authenticate',
        data: {
          userId: 'demo-user-123',
          userName: 'Demo User'
        }
      });
    };
    
    ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        setLastMessage(message);
        
        // Handle specific message types
        switch (message.type) {
          case 'active_users_update':
            setActiveUsers(message.data?.count || 0);
            break;
          case 'processing_update':
            setProcessingFiles(message.data?.count || 0);
            break;
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setConnectionStatus('disconnected');
      setSocket(null);
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnectionStatus('error');
    };
    
    setConnectionStatus('connecting');
    
    return () => {
      ws.close();
    };
  }, []);

  // Simulate some activity for demo purposes
  useEffect(() => {
    const interval = setInterval(() => {
      setActiveUsers(prev => {
        const change = Math.random() > 0.5 ? 1 : -1;
        return Math.max(0, Math.min(20, prev + change));
      });
      
      setProcessingFiles(prev => {
        if (Math.random() > 0.8) {
          return Math.max(0, Math.min(10, prev + (Math.random() > 0.5 ? 1 : -1)));
        }
        return prev;
      });
    }, 5000);

    // Initialize with some values
    setActiveUsers(Math.floor(Math.random() * 10) + 3);
    setProcessingFiles(Math.floor(Math.random() * 5));

    return () => clearInterval(interval);
  }, []);

  const value: WebSocketContextType = {
    connectionStatus,
    activeUsers,
    processingFiles,
    sendMessage,
    lastMessage,
  };

  return React.createElement(
    WebSocketContext.Provider,
    { value },
    children
  );
}

// Custom hook for file-specific WebSocket operations
export function useFileWebSocket(fileId?: string) {
  const { sendMessage, lastMessage } = useWebSocket();

  const joinFile = (fileId: string) => {
    sendMessage({
      type: 'join_file',
      fileId,
    });
  };

  const leaveFile = () => {
    sendMessage({
      type: 'leave_file',
    });
  };

  const sendFileActivity = (activity: any) => {
    sendMessage({
      type: 'file_activity',
      data: activity,
    });
  };

  const sendComment = (content: string, parentId?: string) => {
    sendMessage({
      type: 'comment',
      data: {
        content,
        parentId,
      },
    });
  };

  const sendCursorPosition = (position: { x: number; y: number }) => {
    sendMessage({
      type: 'cursor_position',
      data: position,
    });
  };

  return {
    joinFile,
    leaveFile,
    sendFileActivity,
    sendComment,
    sendCursorPosition,
    lastMessage,
  };
}