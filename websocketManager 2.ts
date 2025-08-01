import { WebSocket, WebSocketServer } from 'ws';
import { IncomingMessage } from 'http';
import { storage } from '../storage';
import type { NewWebsocketConnection, NewActivityLog } from '@shared/schema';

export interface WebSocketMessage {
  type: string;
  data?: any;
  fileId?: string;
  userId?: string;
  userName?: string;
}

export interface AuthenticatedWebSocket extends WebSocket {
  userId?: string;
  userName?: string;
  connectionId?: string;
  fileId?: string;
}

export class WebSocketManager {
  private wss: WebSocketServer;
  private connections = new Map<string, AuthenticatedWebSocket>();
  private fileRooms = new Map<string, Set<string>>(); // fileId -> Set of connectionIds

  constructor(server: any) {
    this.wss = new WebSocketServer({ server, path: '/ws' });
    this.setupWebSocketServer();
    this.startCleanupInterval();
  }

  private setupWebSocketServer() {
    this.wss.on('connection', (ws: AuthenticatedWebSocket, request: IncomingMessage) => {
      console.log('New WebSocket connection');
      
      ws.on('message', async (message: string) => {
        try {
          const data: WebSocketMessage = JSON.parse(message);
          await this.handleMessage(ws, data);
        } catch (error) {
          console.error('WebSocket message error:', error);
          this.sendError(ws, 'Invalid message format');
        }
      });

      ws.on('close', async () => {
        await this.handleDisconnection(ws);
      });

      ws.on('error', (error) => {
        console.error('WebSocket error:', error);
      });

      // Send connection acknowledgment
      this.send(ws, { type: 'connected' });
    });
  }

  private async handleMessage(ws: AuthenticatedWebSocket, message: WebSocketMessage) {
    switch (message.type) {
      case 'authenticate':
        await this.handleAuthentication(ws, message);
        break;
      
      case 'join_file':
        await this.handleJoinFile(ws, message);
        break;
      
      case 'leave_file':
        await this.handleLeaveFile(ws, message);
        break;
      
      case 'file_activity':
        await this.handleFileActivity(ws, message);
        break;
      
      case 'comment':
        await this.handleComment(ws, message);
        break;
      
      case 'cursor_position':
        await this.handleCursorPosition(ws, message);
        break;
      
      case 'ping':
        await this.handlePing(ws);
        break;
      
      default:
        console.warn('Unknown message type:', message.type);
    }
  }

  private async handleAuthentication(ws: AuthenticatedWebSocket, message: WebSocketMessage) {
    if (!message.data?.userId || !message.data?.userName) {
      this.sendError(ws, 'Authentication requires userId and userName');
      return;
    }

    const connectionId = this.generateConnectionId();
    ws.userId = message.data.userId;
    ws.userName = message.data.userName;
    ws.connectionId = connectionId;

    this.connections.set(connectionId, ws);

    // Store connection in database
    try {
      await storage.createWebsocketConnection({
        userId: ws.userId!,
        userName: ws.userName!,
        connectionId: connectionId,
        fileId: undefined,
      });

      this.send(ws, {
        type: 'authenticated',
        data: { connectionId }
      });

      console.log(`User ${ws.userName} authenticated with connection ${connectionId}`);
    } catch (error) {
      console.error('Authentication error:', error);
      this.sendError(ws, 'Authentication failed');
    }
  }

  private async handleJoinFile(ws: AuthenticatedWebSocket, message: WebSocketMessage) {
    if (!ws.userId || !message.fileId) {
      this.sendError(ws, 'Authentication and fileId required');
      return;
    }

    // Leave current file if any
    if (ws.fileId) {
      await this.handleLeaveFile(ws, { type: 'leave_file' });
    }

    ws.fileId = message.fileId;
    
    // Add to file room
    if (!this.fileRooms.has(message.fileId)) {
      this.fileRooms.set(message.fileId, new Set());
    }
    this.fileRooms.get(message.fileId)!.add(ws.connectionId!);

    // Update database connection
    try {
      const connection = await storage.createWebsocketConnection({
        userId: ws.userId,
        userName: ws.userName!,
        connectionId: ws.connectionId!,
        fileId: message.fileId,
      });

      // Log activity
      await storage.createActivityLog({
        fileId: message.fileId,
        userId: ws.userId,
        userName: ws.userName!,
        action: 'join_file',
        details: { connectionId: ws.connectionId },
        ipAddress: this.getClientIP(ws),
        userAgent: 'websocket',
      });

      // Notify other users in the file
      await this.broadcastToFile(message.fileId, {
        type: 'user_joined',
        data: {
          userId: ws.userId,
          userName: ws.userName,
          connectionId: ws.connectionId
        }
      }, [ws.connectionId!]);

      // Send current file collaborators
      const activeConnections = await storage.getActiveConnections(message.fileId);
      this.send(ws, {
        type: 'file_joined',
        data: {
          fileId: message.fileId,
          collaborators: activeConnections.map(conn => ({
            userId: conn.userId,
            userName: conn.userName,
            connectionId: conn.connectionId,
            lastActivity: conn.lastActivity
          }))
        }
      });

      console.log(`User ${ws.userName} joined file ${message.fileId}`);
    } catch (error) {
      console.error('Join file error:', error);
      this.sendError(ws, 'Failed to join file');
    }
  }

  private async handleLeaveFile(ws: AuthenticatedWebSocket, message: WebSocketMessage) {
    if (!ws.fileId || !ws.connectionId) return;

    const fileId = ws.fileId;
    
    // Remove from file room
    this.fileRooms.get(fileId)?.delete(ws.connectionId);
    if (this.fileRooms.get(fileId)?.size === 0) {
      this.fileRooms.delete(fileId);
    }

    // Notify other users
    await this.broadcastToFile(fileId, {
      type: 'user_left',
      data: {
        userId: ws.userId,
        userName: ws.userName,
        connectionId: ws.connectionId
      }
    }, [ws.connectionId]);

    // Log activity
    await storage.createActivityLog({
      fileId: fileId,
      userId: ws.userId!,
      userName: ws.userName!,
      action: 'leave_file',
      details: { connectionId: ws.connectionId },
      ipAddress: this.getClientIP(ws),
      userAgent: 'websocket',
    });

    ws.fileId = undefined;
    console.log(`User ${ws.userName} left file ${fileId}`);
  }

  private async handleFileActivity(ws: AuthenticatedWebSocket, message: WebSocketMessage) {
    if (!ws.fileId || !ws.userId) return;

    // Broadcast activity to other users in the file
    await this.broadcastToFile(ws.fileId, {
      type: 'file_activity',
      data: {
        userId: ws.userId,
        userName: ws.userName,
        activity: message.data,
        timestamp: new Date().toISOString()
      }
    }, [ws.connectionId!]);

    // Update connection activity
    await storage.updateConnectionActivity(ws.connectionId!);
  }

  private async handleComment(ws: AuthenticatedWebSocket, message: WebSocketMessage) {
    if (!ws.fileId || !ws.userId || !message.data?.content) return;

    try {
      // Create comment in database
      const comment = await storage.createFileComment({
        fileId: ws.fileId,
        content: message.data.content,
        authorId: ws.userId,
        authorName: ws.userName!,
        parentId: message.data.parentId,
      });

      // Log activity
      await storage.createActivityLog({
        fileId: ws.fileId,
        userId: ws.userId,
        userName: ws.userName!,
        action: 'comment',
        details: { commentId: comment.id, content: message.data.content },
        ipAddress: this.getClientIP(ws),
        userAgent: 'websocket',
      });

      // Broadcast comment to all users in the file
      await this.broadcastToFile(ws.fileId, {
        type: 'new_comment',
        data: {
          comment,
          timestamp: new Date().toISOString()
        }
      });

    } catch (error) {
      console.error('Comment error:', error);
      this.sendError(ws, 'Failed to create comment');
    }
  }

  private async handleCursorPosition(ws: AuthenticatedWebSocket, message: WebSocketMessage) {
    if (!ws.fileId || !ws.userId) return;

    // Broadcast cursor position to other users in the file
    await this.broadcastToFile(ws.fileId, {
      type: 'cursor_position',
      data: {
        userId: ws.userId,
        userName: ws.userName,
        position: message.data.position,
        timestamp: new Date().toISOString()
      }
    }, [ws.connectionId!]);
  }

  private async handlePing(ws: AuthenticatedWebSocket) {
    if (ws.connectionId) {
      await storage.updateConnectionActivity(ws.connectionId);
    }
    this.send(ws, { type: 'pong' });
  }

  private async handleDisconnection(ws: AuthenticatedWebSocket) {
    if (!ws.connectionId) return;

    // Remove from file room if applicable
    if (ws.fileId) {
      await this.handleLeaveFile(ws, { type: 'leave_file' });
    }

    // Remove from connections
    this.connections.delete(ws.connectionId);

    // Remove from database
    try {
      await storage.removeWebsocketConnection(ws.connectionId);
      console.log(`User ${ws.userName} disconnected (${ws.connectionId})`);
    } catch (error) {
      console.error('Disconnection cleanup error:', error);
    }
  }

  private async broadcastToFile(fileId: string, message: WebSocketMessage, excludeConnections: string[] = []) {
    const connectionIds = this.fileRooms.get(fileId);
    if (!connectionIds) return;

    const promises = Array.from(connectionIds)
      .filter(id => !excludeConnections.includes(id))
      .map(async (connectionId) => {
        const ws = this.connections.get(connectionId);
        if (ws && ws.readyState === WebSocket.OPEN) {
          this.send(ws, message);
        }
      });

    await Promise.all(promises);
  }

  private send(ws: AuthenticatedWebSocket, message: WebSocketMessage) {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
    }
  }

  private sendError(ws: AuthenticatedWebSocket, error: string) {
    this.send(ws, {
      type: 'error',
      data: { message: error }
    });
  }

  private generateConnectionId(): string {
    return `conn_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private getClientIP(ws: AuthenticatedWebSocket): string {
    // This would need to be properly implemented based on your proxy setup
    return 'unknown';
  }

  private startCleanupInterval() {
    // Clean up inactive connections every 5 minutes
    setInterval(async () => {
      const fiveMinutesAgo = new Date(Date.now() - 5 * 60 * 1000);
      
      for (const [connectionId, ws] of this.connections) {
        if (ws.readyState !== WebSocket.OPEN) {
          await this.handleDisconnection(ws);
        }
      }
    }, 5 * 60 * 1000);
  }

  // Public methods for external use
  async notifyFileProcessing(fileId: string, status: any) {
    await this.broadcastToFile(fileId, {
      type: 'file_processing',
      data: { fileId, status }
    });
  }

  async notifyFileUpdate(fileId: string, update: any) {
    await this.broadcastToFile(fileId, {
      type: 'file_updated',
      data: { fileId, update }
    });
  }

  getActiveConnections(): number {
    return this.connections.size;
  }

  getFileConnections(fileId: string): number {
    return this.fileRooms.get(fileId)?.size || 0;
  }
}
