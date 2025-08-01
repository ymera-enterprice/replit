import { io, Socket } from 'socket.io-client';

interface WebSocketMessage {
  timestamp: Date;
  data: any;
}

interface WebSocketCallbacks {
  onMetricsUpdate?: (data: any) => void;
  onHealthUpdate?: (data: any) => void;
  onAlertNew?: (data: any) => void;
  onTaskCreated?: (data: any) => void;
  onTaskCompleted?: (data: any) => void;
  onTaskFailed?: (data: any) => void;
  onPatternDiscovered?: (data: any) => void;
  onKnowledgeValidated?: (data: any) => void;
}

class WebSocketManager {
  private socket: Socket | null = null;
  private callbacks: WebSocketCallbacks = {};
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectInterval = 5000;

  constructor() {
    this.connect();
  }

  private connect() {
    const token = localStorage.getItem('accessToken');
    if (!token) {
      console.warn('No auth token available for WebSocket connection');
      return;
    }

    this.socket = io(window.location.origin, {
      auth: {
        token,
      },
      transports: ['websocket', 'polling'],
    });

    this.setupEventListeners();
  }

  private setupEventListeners() {
    if (!this.socket) return;

    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
    });

    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      this.handleReconnect();
    });

    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      this.handleReconnect();
    });

    // Metrics updates
    this.socket.on('metrics:update', (message: WebSocketMessage) => {
      this.callbacks.onMetricsUpdate?.(message.data);
    });

    // Health updates
    this.socket.on('health:update', (message: WebSocketMessage) => {
      this.callbacks.onHealthUpdate?.(message.data);
    });

    // Alert notifications
    this.socket.on('alert:new', (message: WebSocketMessage) => {
      this.callbacks.onAlertNew?.(message.data);
    });

    // Task updates
    this.socket.on('task:created', (message: WebSocketMessage) => {
      this.callbacks.onTaskCreated?.(message.data);
    });

    this.socket.on('task:completed', (message: WebSocketMessage) => {
      this.callbacks.onTaskCompleted?.(message.data);
    });

    this.socket.on('task:failed', (message: WebSocketMessage) => {
      this.callbacks.onTaskFailed?.(message.data);
    });

    // Learning engine updates
    this.socket.on('learning:pattern-discovered', (message: WebSocketMessage) => {
      this.callbacks.onPatternDiscovered?.(message.data);
    });

    this.socket.on('learning:knowledge-validated', (message: WebSocketMessage) => {
      this.callbacks.onKnowledgeValidated?.(message.data);
    });
  }

  private handleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

    setTimeout(() => {
      this.connect();
    }, this.reconnectInterval * this.reconnectAttempts);
  }

  public subscribe(callbacks: WebSocketCallbacks) {
    this.callbacks = { ...this.callbacks, ...callbacks };

    if (this.socket?.connected) {
      // Subscribe to specific event streams
      if (callbacks.onMetricsUpdate) {
        this.socket.emit('subscribe:metrics');
      }
      if (callbacks.onAlertNew) {
        this.socket.emit('subscribe:alerts');
      }
      if (callbacks.onTaskCreated || callbacks.onTaskCompleted || callbacks.onTaskFailed) {
        this.socket.emit('subscribe:tasks');
      }
    }
  }

  public unsubscribe() {
    this.callbacks = {};
  }

  public disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  public isConnected(): boolean {
    return this.socket?.connected || false;
  }

  public reconnect() {
    this.disconnect();
    this.reconnectAttempts = 0;
    this.connect();
  }
}

// Singleton instance
export const websocketManager = new WebSocketManager();

// React hook for WebSocket
import { useEffect, useRef } from 'react';

export function useWebSocket(callbacks: WebSocketCallbacks) {
  const callbacksRef = useRef(callbacks);
  callbacksRef.current = callbacks;

  useEffect(() => {
    websocketManager.subscribe(callbacksRef.current);

    return () => {
      websocketManager.unsubscribe();
    };
  }, []);

  return {
    isConnected: websocketManager.isConnected(),
    reconnect: () => websocketManager.reconnect(),
  };
}
