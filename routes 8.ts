import type { Express } from "express";
import { createServer, type Server } from "http";
import { WebSocketServer, WebSocket } from "ws";
import multer from "multer";
import path from "path";
import { promises as fs } from "fs";
import rateLimit from "express-rate-limit";
import { storage } from "./storage";
import { setupAuth, isAuthenticated } from "./replitAuth";
import {
  insertProjectSchema,
  insertFileSchema,
  insertMessageSchema,
  insertAgentSchema,
  insertAgentTaskSchema,
  insertKnowledgeNodeSchema,
  insertSystemMetricSchema,
} from "@shared/schema";

// Rate limiting
const apiLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 1000, // limit each IP to 1000 requests per windowMs
  message: "Too many requests from this IP",
});

// File upload configuration
const upload = multer({
  dest: 'uploads/',
  limits: {
    fileSize: 50 * 1024 * 1024, // 50MB limit
  },
});

// WebSocket connection map
const wsConnections = new Map<string, { ws: WebSocket; userId: string }>();

export async function registerRoutes(app: Express): Promise<Server> {
  // Apply rate limiting
  app.use('/api', apiLimiter);

  // Auth middleware
  await setupAuth(app);

  // Auth routes
  app.get('/api/auth/user', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const user = await storage.getUser(userId);
      res.json(user);
    } catch (error) {
      console.error("Error fetching user:", error);
      res.status(500).json({ message: "Failed to fetch user" });
    }
  });

  // Project routes
  app.get('/api/projects', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const projects = await storage.getProjects(userId);
      res.json({ data: projects });
    } catch (error) {
      console.error("Error fetching projects:", error);
      res.status(500).json({ message: "Failed to fetch projects" });
    }
  });

  app.get('/api/projects/:id', isAuthenticated, async (req: any, res) => {
    try {
      const project = await storage.getProject(req.params.id);
      if (!project) {
        return res.status(404).json({ message: "Project not found" });
      }
      res.json({ data: project });
    } catch (error) {
      console.error("Error fetching project:", error);
      res.status(500).json({ message: "Failed to fetch project" });
    }
  });

  app.post('/api/projects', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const projectData = insertProjectSchema.parse({ ...req.body, userId });
      const project = await storage.createProject(projectData);
      res.status(201).json({ data: project });
    } catch (error) {
      console.error("Error creating project:", error);
      res.status(400).json({ message: "Failed to create project" });
    }
  });

  app.patch('/api/projects/:id', isAuthenticated, async (req: any, res) => {
    try {
      const updates = insertProjectSchema.partial().parse(req.body);
      const project = await storage.updateProject(req.params.id, updates);
      if (!project) {
        return res.status(404).json({ message: "Project not found" });
      }
      res.json({ data: project });
    } catch (error) {
      console.error("Error updating project:", error);
      res.status(400).json({ message: "Failed to update project" });
    }
  });

  app.delete('/api/projects/:id', isAuthenticated, async (req: any, res) => {
    try {
      const deleted = await storage.deleteProject(req.params.id);
      if (!deleted) {
        return res.status(404).json({ message: "Project not found" });
      }
      res.status(204).send();
    } catch (error) {
      console.error("Error deleting project:", error);
      res.status(500).json({ message: "Failed to delete project" });
    }
  });

  // File routes
  app.get('/api/files', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const projectId = req.query.projectId as string;
      const files = await storage.getFiles(userId, projectId);
      res.json({ data: files });
    } catch (error) {
      console.error("Error fetching files:", error);
      res.status(500).json({ message: "Failed to fetch files" });
    }
  });

  app.post('/api/files/upload', isAuthenticated, upload.array('files'), async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const files = req.files as Express.Multer.File[];
      const projectId = req.body.projectId;

      const uploadedFiles = [];
      for (const file of files) {
        const fileData = {
          name: file.filename,
          originalName: file.originalname,
          mimeType: file.mimetype,
          size: file.size,
          path: file.path,
          projectId,
          userId,
        };
        
        const savedFile = await storage.createFile(fileData);
        uploadedFiles.push(savedFile);
      }

      res.status(201).json({ data: uploadedFiles });
    } catch (error) {
      console.error("Error uploading files:", error);
      res.status(400).json({ message: "Failed to upload files" });
    }
  });

  app.get('/api/files/:id/download', isAuthenticated, async (req: any, res) => {
    try {
      const file = await storage.getFile(req.params.id);
      if (!file) {
        return res.status(404).json({ message: "File not found" });
      }

      // Increment download count
      await storage.updateFile(file.id, { downloadCount: file.downloadCount + 1 });

      res.download(file.path, file.originalName);
    } catch (error) {
      console.error("Error downloading file:", error);
      res.status(500).json({ message: "Failed to download file" });
    }
  });

  app.delete('/api/files/:id', isAuthenticated, async (req: any, res) => {
    try {
      const file = await storage.getFile(req.params.id);
      if (!file) {
        return res.status(404).json({ message: "File not found" });
      }

      // Delete file from filesystem
      try {
        await fs.unlink(file.path);
      } catch (fsError) {
        console.warn("File not found on filesystem:", fsError);
      }

      const deleted = await storage.deleteFile(req.params.id);
      if (!deleted) {
        return res.status(404).json({ message: "File not found" });
      }

      res.status(204).send();
    } catch (error) {
      console.error("Error deleting file:", error);
      res.status(500).json({ message: "Failed to delete file" });
    }
  });

  // Communication routes
  app.get('/api/messages', isAuthenticated, async (req: any, res) => {
    try {
      const projectId = req.query.projectId as string;
      const limit = parseInt(req.query.limit as string) || 100;
      const messages = await storage.getMessages(projectId, limit);
      res.json({ data: messages });
    } catch (error) {
      console.error("Error fetching messages:", error);
      res.status(500).json({ message: "Failed to fetch messages" });
    }
  });

  app.post('/api/messages', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const messageData = insertMessageSchema.parse({ ...req.body, userId });
      const message = await storage.createMessage(messageData);

      // Broadcast message to WebSocket connections
      const messageWithUser = { ...message, user: await storage.getUser(userId) };
      broadcastToProject(messageData.projectId, 'new_message', messageWithUser);

      res.status(201).json({ data: message });
    } catch (error) {
      console.error("Error creating message:", error);
      res.status(400).json({ message: "Failed to create message" });
    }
  });

  // Agent routes
  app.get('/api/agents', isAuthenticated, async (req: any, res) => {
    try {
      const agents = await storage.getAgents();
      res.json({ data: agents });
    } catch (error) {
      console.error("Error fetching agents:", error);
      res.status(500).json({ message: "Failed to fetch agents" });
    }
  });

  app.get('/api/agents/:id', isAuthenticated, async (req: any, res) => {
    try {
      const agent = await storage.getAgent(req.params.id);
      if (!agent) {
        return res.status(404).json({ message: "Agent not found" });
      }
      res.json({ data: agent });
    } catch (error) {
      console.error("Error fetching agent:", error);
      res.status(500).json({ message: "Failed to fetch agent" });
    }
  });

  app.post('/api/agents', isAuthenticated, async (req: any, res) => {
    try {
      const agentData = insertAgentSchema.parse(req.body);
      const agent = await storage.createAgent(agentData);
      res.status(201).json({ data: agent });
    } catch (error) {
      console.error("Error creating agent:", error);
      res.status(400).json({ message: "Failed to create agent" });
    }
  });

  app.patch('/api/agents/:id', isAuthenticated, async (req: any, res) => {
    try {
      const updates = insertAgentSchema.partial().parse(req.body);
      const agent = await storage.updateAgent(req.params.id, updates);
      if (!agent) {
        return res.status(404).json({ message: "Agent not found" });
      }
      res.json({ data: agent });
    } catch (error) {
      console.error("Error updating agent:", error);
      res.status(400).json({ message: "Failed to update agent" });
    }
  });

  app.delete('/api/agents/:id', isAuthenticated, async (req: any, res) => {
    try {
      const deleted = await storage.deleteAgent(req.params.id);
      if (!deleted) {
        return res.status(404).json({ message: "Agent not found" });
      }
      res.status(204).send();
    } catch (error) {
      console.error("Error deleting agent:", error);
      res.status(500).json({ message: "Failed to delete agent" });
    }
  });

  // Agent task routes
  app.get('/api/agent-tasks', isAuthenticated, async (req: any, res) => {
    try {
      const agentId = req.query.agentId as string;
      const limit = parseInt(req.query.limit as string) || 100;
      const tasks = await storage.getAgentTasks(agentId, limit);
      res.json({ data: tasks });
    } catch (error) {
      console.error("Error fetching agent tasks:", error);
      res.status(500).json({ message: "Failed to fetch agent tasks" });
    }
  });

  app.post('/api/agent-tasks', isAuthenticated, async (req: any, res) => {
    try {
      const taskData = insertAgentTaskSchema.parse(req.body);
      const task = await storage.createAgentTask(taskData);
      res.status(201).json({ data: task });
    } catch (error) {
      console.error("Error creating agent task:", error);
      res.status(400).json({ message: "Failed to create agent task" });
    }
  });

  // Knowledge graph routes
  app.get('/api/knowledge/nodes', isAuthenticated, async (req: any, res) => {
    try {
      const type = req.query.type as string;
      const nodes = await storage.getKnowledgeNodes(type);
      res.json({ data: nodes });
    } catch (error) {
      console.error("Error fetching knowledge nodes:", error);
      res.status(500).json({ message: "Failed to fetch knowledge nodes" });
    }
  });

  app.post('/api/knowledge/nodes', isAuthenticated, async (req: any, res) => {
    try {
      const nodeData = insertKnowledgeNodeSchema.parse(req.body);
      const node = await storage.createKnowledgeNode(nodeData);
      res.status(201).json({ data: node });
    } catch (error) {
      console.error("Error creating knowledge node:", error);
      res.status(400).json({ message: "Failed to create knowledge node" });
    }
  });

  app.get('/api/knowledge/relationships', isAuthenticated, async (req: any, res) => {
    try {
      const nodeId = req.query.nodeId as string;
      const relationships = await storage.getKnowledgeRelationships(nodeId);
      res.json({ data: relationships });
    } catch (error) {
      console.error("Error fetching knowledge relationships:", error);
      res.status(500).json({ message: "Failed to fetch knowledge relationships" });
    }
  });

  // System metrics routes
  app.get('/api/metrics', isAuthenticated, async (req: any, res) => {
    try {
      const type = req.query.type as string;
      const hours = parseInt(req.query.hours as string) || 24;
      const metrics = await storage.getSystemMetrics(type, hours);
      res.json({ data: metrics });
    } catch (error) {
      console.error("Error fetching metrics:", error);
      res.status(500).json({ message: "Failed to fetch metrics" });
    }
  });

  app.post('/api/metrics', isAuthenticated, async (req: any, res) => {
    try {
      const metricData = insertSystemMetricSchema.parse(req.body);
      const metric = await storage.createSystemMetric(metricData);
      res.status(201).json({ data: metric });
    } catch (error) {
      console.error("Error creating metric:", error);
      res.status(400).json({ message: "Failed to create metric" });
    }
  });

  // Dashboard metrics route
  app.get('/api/dashboard/metrics', isAuthenticated, async (req: any, res) => {
    try {
      const metrics = await storage.getDashboardMetrics();
      res.json({ data: metrics });
    } catch (error) {
      console.error("Error fetching dashboard metrics:", error);
      res.status(500).json({ message: "Failed to fetch dashboard metrics" });
    }
  });

  // WebSocket connections route
  app.get('/api/websocket/connections', isAuthenticated, async (req: any, res) => {
    try {
      const connections = await storage.getActiveConnections();
      res.json({ data: connections });
    } catch (error) {
      console.error("Error fetching connections:", error);
      res.status(500).json({ message: "Failed to fetch connections" });
    }
  });

  // Create HTTP server
  const httpServer = createServer(app);

  // Setup WebSocket server
  const wss = new WebSocketServer({ server: httpServer, path: '/ws' });

  wss.on('connection', async (ws: WebSocket, req) => {
    const connectionId = Math.random().toString(36).substring(7);
    
    ws.on('message', async (message) => {
      try {
        const data = JSON.parse(message.toString());
        
        if (data.type === 'auth' && data.userId) {
          wsConnections.set(connectionId, { ws, userId: data.userId });
          
          // Store connection in database
          await storage.createConnection({
            userId: data.userId,
            connectionId,
            status: 'connected',
          });

          ws.send(JSON.stringify({ type: 'auth_success', connectionId }));
        } else if (data.type === 'ping') {
          ws.send(JSON.stringify({ type: 'pong' }));
        } else if (data.type === 'agent_message' && data.agentId && data.message) {
          // Broadcast agent message to all connected clients
          broadcast('agent_communication', {
            agentId: data.agentId,
            message: data.message,
            timestamp: new Date().toISOString(),
          });
        }
      } catch (error) {
        console.error('WebSocket message error:', error);
      }
    });

    ws.on('close', async () => {
      const connection = wsConnections.get(connectionId);
      if (connection) {
        wsConnections.delete(connectionId);
        
        // Update connection status in database
        await storage.updateConnection(connectionId, { status: 'disconnected' });
      }
    });

    // Send welcome message
    ws.send(JSON.stringify({ type: 'welcome', connectionId }));
  });

  // WebSocket utility functions
  function broadcast(type: string, data: any) {
    const message = JSON.stringify({ type, data });
    wsConnections.forEach(({ ws }) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(message);
      }
    });
  }

  function broadcastToProject(projectId: string | null, type: string, data: any) {
    const message = JSON.stringify({ type, data, projectId });
    wsConnections.forEach(({ ws }) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(message);
      }
    });
  }

  // Periodic system metrics collection
  setInterval(async () => {
    try {
      const metrics = await storage.getDashboardMetrics();
      
      // Store metrics in database
      await storage.createSystemMetric({
        metricType: 'active_users',
        value: metrics.activeUsers,
        unit: 'count',
      });

      await storage.createSystemMetric({
        metricType: 'active_connections',
        value: metrics.activeConnections,
        unit: 'count',
      });

      // Broadcast to connected clients
      broadcast('metrics_update', metrics);
    } catch (error) {
      console.error('Error collecting metrics:', error);
    }
  }, 30000); // Every 30 seconds

  return httpServer;
}
