import { Server as SocketIOServer } from 'socket.io';
import { Server as HttpServer } from 'http';
import jwt from 'jsonwebtoken';
import { createLogger } from './logger';
import { JWTManager, SessionManager } from './auth';
import { appCache } from './cache';

const logger = createLogger('websocket');

export interface SocketUser {
  id: string;
  username: string;
  email: string;
  sessionId: string;
}

export interface AuthenticatedSocket {
  id: string;
  user: SocketUser;
  emit: (event: string, data?: any) => void;
  join: (room: string) => void;
  leave: (room: string) => void;
  disconnect: () => void;
}

export class WebSocketManager {
  private io: SocketIOServer;
  private connectedUsers = new Map<string, AuthenticatedSocket>();
  private userSockets = new Map<string, Set<string>>(); // userId -> socketIds
  private cleanupInterval: NodeJS.Timeout;

  constructor(httpServer: HttpServer) {
    this.io = new SocketIOServer(httpServer, {
      cors: {
        origin: process.env.NODE_ENV === 'development' 
          ? ["http://localhost:5000", "http://127.0.0.1:5000"]
          : process.env.ALLOWED_ORIGINS?.split(',') || [],
        methods: ["GET", "POST"],
        credentials: true,
      },
      transports: ['websocket', 'polling'],
      pingTimeout: 60000,
      pingInterval: 25000,
    });

    this.setupMiddleware();
    this.setupEventHandlers();
    this.startCleanupTask();

    logger.info('WebSocket server initialized');
  }

  private setupMiddleware(): void {
    // Authentication middleware
    this.io.use(async (socket, next) => {
      try {
        const token = socket.handshake.auth.token || socket.handshake.headers.authorization?.replace('Bearer ', '');
        
        if (!token) {
          return next(new Error('Authentication token required'));
        }

        const payload = JWTManager.verifyToken(token);
        if (!payload) {
          return next(new Error('Invalid authentication token'));
        }

        // Validate session
        const sessionData = await SessionManager.validateSession(payload.sessionId);
        if (!sessionData) {
          return next(new Error('Session expired or invalid'));
        }

        // Attach user data to socket
        (socket as any).user = {
          id: sessionData.user.id,
          username: sessionData.user.username,
          email: sessionData.user.email,
          sessionId: payload.sessionId,
        };

        logger.info('Socket authenticated', {
          socketId: socket.id,
          userId: sessionData.user.id,
          username: sessionData.user.username,
        });

        next();
      } catch (error) {
        logger.error('Socket authentication error', { error: error.message });
        next(new Error('Authentication failed'));
      }
    });

    // Rate limiting middleware
    this.io.use(async (socket, next) => {
      const userId = (socket as any).user?.id;
      if (!userId) return next();

      // Check connection limit per user
      const userConnections = this.userSockets.get(userId)?.size || 0;
      const maxConnectionsPerUser = 5;

      if (userConnections >= maxConnectionsPerUser) {
        return next(new Error('Maximum connections per user exceeded'));
      }

      next();
    });
  }

  private setupEventHandlers(): void {
    this.io.on('connection', (socket) => {
      const user = (socket as any).user as SocketUser;
      
      // Track user connections
      if (!this.userSockets.has(user.id)) {
        this.userSockets.set(user.id, new Set());
      }
      this.userSockets.get(user.id)!.add(socket.id);

      // Create authenticated socket wrapper
      const authSocket: AuthenticatedSocket = {
        id: socket.id,
        user,
        emit: socket.emit.bind(socket),
        join: socket.join.bind(socket),
        leave: socket.leave.bind(socket),
        disconnect: socket.disconnect.bind(socket),
      };

      this.connectedUsers.set(socket.id, authSocket);

      logger.info('User connected via WebSocket', {
        socketId: socket.id,
        userId: user.id,
        username: user.username,
        totalConnections: this.connectedUsers.size,
      });

      // Join user-specific room
      socket.join(`user:${user.id}`);

      // Setup event handlers
      this.setupSocketEventHandlers(socket, authSocket);

      // Send welcome message
      socket.emit('connected', {
        message: 'Connected to YMERA WebSocket server',
        timestamp: new Date().toISOString(),
        user: {
          id: user.id,
          username: user.username,
        },
      });

      // Handle disconnection
      socket.on('disconnect', (reason) => {
        this.handleDisconnection(socket, reason);
      });

      // Handle errors
      socket.on('error', (error) => {
        logger.error('Socket error', {
          socketId: socket.id,
          userId: user.id,
          error: error.message,
        });
      });
    });
  }

  private setupSocketEventHandlers(socket: any, authSocket: AuthenticatedSocket): void {
    // Ping/Pong for connection health
    socket.on('ping', () => {
      socket.emit('pong', { timestamp: Date.now() });
    });

    // Subscribe to real-time updates
    socket.on('subscribe', async (data: { channels: string[] }) => {
      if (!data.channels || !Array.isArray(data.channels)) {
        socket.emit('error', { message: 'Invalid subscription data' });
        return;
      }

      for (const channel of data.channels) {
        if (this.isChannelAllowed(channel, authSocket.user)) {
          socket.join(channel);
          logger.debug('User subscribed to channel', {
            userId: authSocket.user.id,
            channel,
          });
        } else {
          socket.emit('subscription_error', {
            channel,
            message: 'Access denied to channel',
          });
        }
      }

      socket.emit('subscribed', { channels: data.channels });
    });

    // Unsubscribe from updates
    socket.on('unsubscribe', (data: { channels: string[] }) => {
      if (!data.channels || !Array.isArray(data.channels)) {
        return;
      }

      for (const channel of data.channels) {
        socket.leave(channel);
        logger.debug('User unsubscribed from channel', {
          userId: authSocket.user.id,
          channel,
        });
      }

      socket.emit('unsubscribed', { channels: data.channels });
    });

    // Agent status updates
    socket.on('agent_status_request', async (data: { agentId: string }) => {
      if (!data.agentId) {
        socket.emit('error', { message: 'Agent ID required' });
        return;
      }

      try {
        // Check if user has access to this agent
        const agentData = await appCache.getAgentData(data.agentId);
        if (agentData) {
          socket.emit('agent_status', {
            agentId: data.agentId,
            status: agentData.status,
            lastHeartbeat: agentData.lastHeartbeat,
          });
        } else {
          socket.emit('agent_status', {
            agentId: data.agentId,
            status: 'unknown',
          });
        }
      } catch (error) {
        socket.emit('error', {
          message: 'Failed to get agent status',
          agentId: data.agentId,
        });
      }
    });

    // Real-time learning data updates
    socket.on('learning_data_stream', (data: { agentId: string; enabled: boolean }) => {
      if (data.enabled) {
        socket.join(`learning:${data.agentId}`);
      } else {
        socket.leave(`learning:${data.agentId}`);
      }

      socket.emit('learning_stream_status', {
        agentId: data.agentId,
        enabled: data.enabled,
      });
    });

    // System notifications
    socket.on('system_notifications', (data: { enabled: boolean }) => {
      if (data.enabled) {
        socket.join('system_notifications');
      } else {
        socket.leave('system_notifications');
      }

      socket.emit('notification_status', { enabled: data.enabled });
    });
  }

  private handleDisconnection(socket: any, reason: string): void {
    const authSocket = this.connectedUsers.get(socket.id);
    
    if (authSocket) {
      const userId = authSocket.user.id;
      
      // Remove from tracking
      this.connectedUsers.delete(socket.id);
      
      const userConnections = this.userSockets.get(userId);
      if (userConnections) {
        userConnections.delete(socket.id);
        if (userConnections.size === 0) {
          this.userSockets.delete(userId);
        }
      }

      logger.info('User disconnected from WebSocket', {
        socketId: socket.id,
        userId,
        username: authSocket.user.username,
        reason,
        totalConnections: this.connectedUsers.size,
      });
    }
  }

  private isChannelAllowed(channel: string, user: SocketUser): boolean {
    // Define channel access rules
    const channelRules: { [pattern: string]: (user: SocketUser) => boolean } = {
      [`user:${user.id}`]: () => true, // User's own channel
      'system_notifications': () => true, // All authenticated users
      'agent:*': () => true, // All agents (could be restricted based on permissions)
      'learning:*': () => true, // Learning data (could be restricted)
    };

    for (const [pattern, rule] of Object.entries(channelRules)) {
      const regex = new RegExp(pattern.replace('*', '.*'));
      if (regex.test(channel)) {
        return rule(user);
      }
    }

    return false; // Deny by default
  }

  private startCleanupTask(): void {
    // Clean up stale connections every 5 minutes
    this.cleanupInterval = setInterval(() => {
      this.cleanupStaleConnections();
    }, 5 * 60 * 1000);
  }

  private async cleanupStaleConnections(): Promise<void> {
    const staleThreshold = Date.now() - (10 * 60 * 1000); // 10 minutes
    let cleanedCount = 0;

    for (const [socketId, socket] of this.connectedUsers.entries()) {
      try {
        // Check if session is still valid
        const sessionData = await SessionManager.validateSession(socket.user.sessionId);
        if (!sessionData) {
          socket.disconnect();
          cleanedCount++;
        }
      } catch (error) {
        logger.warn('Error during connection cleanup', {
          socketId,
          error: error.message,
        });
      }
    }

    if (cleanedCount > 0) {
      logger.info('Cleaned up stale WebSocket connections', { count: cleanedCount });
    }
  }

  // Public methods for sending messages
  public sendToUser(userId: string, event: string, data: any): void {
    this.io.to(`user:${userId}`).emit(event, data);
  }

  public sendToChannel(channel: string, event: string, data: any): void {
    this.io.to(channel).emit(event, data);
  }

  public broadcastToAll(event: string, data: any): void {
    this.io.emit(event, data);
  }

  public sendSystemNotification(message: string, type: 'info' | 'warning' | 'error' = 'info'): void {
    this.sendToChannel('system_notifications', 'notification', {
      type,
      message,
      timestamp: new Date().toISOString(),
    });
  }

  public sendAgentUpdate(agentId: string, status: string, data?: any): void {
    this.sendToChannel(`agent:${agentId}`, 'agent_update', {
      agentId,
      status,
      data,
      timestamp: new Date().toISOString(),
    });
  }

  public sendLearningDataUpdate(agentId: string, learningData: any): void {
    this.sendToChannel(`learning:${agentId}`, 'learning_update', {
      agentId,
      data: learningData,
      timestamp: new Date().toISOString(),
    });
  }

  // Get statistics
  public getStats(): {
    totalConnections: number;
    uniqueUsers: number;
    channels: string[];
  } {
    return {
      totalConnections: this.connectedUsers.size,
      uniqueUsers: this.userSockets.size,
      channels: Array.from(this.io.sockets.adapter.rooms.keys()),
    };
  }

  // Cleanup
  public destroy(): void {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
    }
    this.io.close();
    this.connectedUsers.clear();
    this.userSockets.clear();
    logger.info('WebSocket manager destroyed');
  }
}

// Export singleton instance placeholder
export let wsManager: WebSocketManager | null = null;

export const initializeWebSocket = (httpServer: HttpServer): WebSocketManager => {
  wsManager = new WebSocketManager(httpServer);
  return wsManager;
};

// Utility functions for other modules
export const sendUserNotification = (userId: string, message: string, type: 'info' | 'warning' | 'error' = 'info'): void => {
  if (wsManager) {
    wsManager.sendToUser(userId, 'notification', {
      type,
      message,
      timestamp: new Date().toISOString(),
    });
  }
};

export const broadcastSystemMessage = (message: string, type: 'info' | 'warning' | 'error' = 'info'): void => {
  if (wsManager) {
    wsManager.sendSystemNotification(message, type);
  }
};
