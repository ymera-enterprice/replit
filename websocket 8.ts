import { WebSocketServer, WebSocket } from 'ws';
import { Server } from 'http';
import jwt from 'jsonwebtoken';
import { storage } from './storage';
import type { User } from '@shared/schema';

const JWT_SECRET = process.env.JWT_SECRET || process.env.Service || 'ymera-default-secret-2024';

interface WebSocketConnection {
  ws: WebSocket;
  user?: User;
  subscriptions: Set<string>;
  lastPing: number;
}

export class WebSocketManager {
  private wss: WebSocketServer;
  private connections = new Map<string, WebSocketConnection>();
  private heartbeatInterval: NodeJS.Timeout;

  constructor(server: Server) {
    this.wss = new WebSocketServer({ 
      server, 
      path: '/ws',
      verifyClient: this.verifyClient.bind(this)
    });

    this.setupEventHandlers();
    this.startHeartbeat();
  }

  private verifyClient(info: any): boolean {
    // Basic verification - can be enhanced with more security checks
    return true;
  }

  private setupEventHandlers(): void {
    this.wss.on('connection', (ws, req) => {
      const connectionId = this.generateConnectionId();
      
      const connection: WebSocketConnection = {
        ws,
        subscriptions: new Set(),
        lastPing: Date.now()
      };

      this.connections.set(connectionId, connection);
      
      console.log(`WebSocket client connected: ${connectionId}`);
      
      // Send welcome message
      this.sendMessage(ws, {
        type: 'welcome',
        data: { 
          connectionId, 
          timestamp: new Date().toISOString(),
          serverVersion: '2.0.0'
        }
      });

      // Handle messages
      ws.on('message', async (message) => {
        try {
          const data = JSON.parse(message.toString());
          await this.handleMessage(connectionId, data);
        } catch (error) {
          console.error('WebSocket message error:', error);
          this.sendMessage(ws, {
            type: 'error',
            data: { message: 'Invalid message format' }
          });
        }
      });

      // Handle connection close
      ws.on('close', () => {
        this.connections.delete(connectionId);
        console.log(`WebSocket client disconnected: ${connectionId}`);
      });

      // Handle errors
      ws.on('error', (error) => {
        console.error(`WebSocket error for client ${connectionId}:`, error);
        this.connections.delete(connectionId);
      });
    });
  }

  private async handleMessage(connectionId: string, data: any): Promise<void> {
    const connection = this.connections.get(connectionId);
    if (!connection) return;

    const { ws } = connection;

    switch (data.type) {
      case 'authenticate':
        await this.handleAuthentication(connectionId, data.token);
        break;

      case 'ping':
        connection.lastPing = Date.now();
        this.sendMessage(ws, {
          type: 'pong',
          data: { timestamp: new Date().toISOString() }
        });
        break;

      case 'subscribe':
        this.handleSubscription(connectionId, data.topic);
        break;

      case 'unsubscribe':
        this.handleUnsubscription(connectionId, data.topic);
        break;

      case 'agent_heartbeat':
        await this.handleAgentHeartbeat(data);
        break;

      case 'collaboration_message':
        await this.handleCollaborationMessage(connectionId, data);
        break;

      case 'learning_update':
        await this.handleLearningUpdate(data);
        break;

      default:
        this.sendMessage(ws, {
          type: 'error',
          data: { message: `Unknown message type: ${data.type}` }
        });
    }
  }

  private async handleAuthentication(connectionId: string, token: string): Promise<void> {
    const connection = this.connections.get(connectionId);
    if (!connection) return;

    try {
      const decoded = jwt.verify(token, JWT_SECRET) as { userId: number; username: string };
      const user = await storage.getUser(decoded.userId);

      if (user && user.isActive) {
        connection.user = user;
        
        this.sendMessage(connection.ws, {
          type: 'authenticated',
          data: { 
            userId: user.id, 
            username: user.username,
            role: user.role
          }
        });
      } else {
        this.sendMessage(connection.ws, {
          type: 'authentication_failed',
          data: { message: 'Invalid or inactive user' }
        });
      }
    } catch (error) {
      this.sendMessage(connection.ws, {
        type: 'authentication_failed',
        data: { message: 'Invalid token' }
      });
    }
  }

  private handleSubscription(connectionId: string, topic: string): void {
    const connection = this.connections.get(connectionId);
    if (!connection) return;

    connection.subscriptions.add(topic);
    
    this.sendMessage(connection.ws, {
      type: 'subscribed',
      data: { topic }
    });

    console.log(`Client ${connectionId} subscribed to ${topic}`);
  }

  private handleUnsubscription(connectionId: string, topic: string): void {
    const connection = this.connections.get(connectionId);
    if (!connection) return;

    connection.subscriptions.delete(topic);
    
    this.sendMessage(connection.ws, {
      type: 'unsubscribed',
      data: { topic }
    });

    console.log(`Client ${connectionId} unsubscribed from ${topic}`);
  }

  private async handleAgentHeartbeat(data: any): Promise<void> {
    if (!data.agentId) return;

    // Update agent metrics in database
    const agent = await storage.getAgent(data.agentId);
    if (agent) {
      await storage.updateAgent(data.agentId, {
        lastActivity: new Date(),
        cpuUsage: data.metrics?.cpuUsage || agent.cpuUsage,
        memoryUsage: data.metrics?.memoryUsage || agent.memoryUsage,
        healthScore: data.metrics?.healthScore || agent.healthScore,
      });

      // Broadcast to subscribers
      this.broadcast('agents', {
        type: 'agent_heartbeat',
        data: { 
          agentId: data.agentId, 
          timestamp: new Date().toISOString(),
          metrics: data.metrics
        }
      });
    }
  }

  private async handleCollaborationMessage(connectionId: string, data: any): Promise<void> {
    const connection = this.connections.get(connectionId);
    if (!connection || !connection.user) return;

    // Broadcast collaboration message to other participants
    if (data.collaborationId && data.participantAgents) {
      this.broadcast(`collaboration:${data.collaborationId}`, {
        type: 'collaboration_message',
        data: {
          ...data,
          from: connection.user.username,
          timestamp: new Date().toISOString()
        }
      });
    }
  }

  private async handleLearningUpdate(data: any): Promise<void> {
    // Broadcast learning updates
    this.broadcast('learning', {
      type: 'learning_update',
      data: {
        ...data,
        timestamp: new Date().toISOString()
      }
    });
  }

  public broadcast(topic: string, message: any): void {
    const messageString = JSON.stringify(message);
    
    this.connections.forEach((connection, connectionId) => {
      if (connection.subscriptions.has(topic) && connection.ws.readyState === WebSocket.OPEN) {
        try {
          connection.ws.send(messageString);
        } catch (error) {
          console.error(`Failed to send message to client ${connectionId}:`, error);
          this.connections.delete(connectionId);
        }
      }
    });
  }

  public broadcastToAll(message: any): void {
    const messageString = JSON.stringify(message);
    
    this.connections.forEach((connection, connectionId) => {
      if (connection.ws.readyState === WebSocket.OPEN) {
        try {
          connection.ws.send(messageString);
        } catch (error) {
          console.error(`Failed to send message to client ${connectionId}:`, error);
          this.connections.delete(connectionId);
        }
      }
    });
  }

  private sendMessage(ws: WebSocket, message: any): void {
    if (ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify(message));
      } catch (error) {
        console.error('Failed to send WebSocket message:', error);
      }
    }
  }

  private generateConnectionId(): string {
    return Math.random().toString(36).substring(2) + Date.now().toString(36);
  }

  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      const now = Date.now();
      const timeout = 30000; // 30 seconds

      this.connections.forEach((connection, connectionId) => {
        if (now - connection.lastPing > timeout) {
          console.log(`Client ${connectionId} timed out, closing connection`);
          connection.ws.terminate();
          this.connections.delete(connectionId);
        }
      });
    }, 10000); // Check every 10 seconds
  }

  public getConnectionCount(): number {
    return this.connections.size;
  }

  public getSubscriberCount(topic: string): number {
    let count = 0;
    this.connections.forEach(connection => {
      if (connection.subscriptions.has(topic)) {
        count++;
      }
    });
    return count;
  }

  public close(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
    }
    
    this.connections.forEach((connection) => {
      connection.ws.terminate();
    });
    
    this.connections.clear();
    this.wss.close();
  }
}

// Export singleton instance
let wsManagerInstance: WebSocketManager | null = null;

export const getWebSocketManager = (server?: Server): WebSocketManager => {
  if (!wsManagerInstance && server) {
    wsManagerInstance = new WebSocketManager(server);
  }
  if (!wsManagerInstance) {
    throw new Error('WebSocketManager not initialized');
  }
  return wsManagerInstance;
};
