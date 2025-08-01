import {
  users,
  projects,
  files,
  messages,
  agents,
  agentTasks,
  knowledgeNodes,
  knowledgeRelationships,
  systemMetrics,
  webSocketConnections,
  type User,
  type UpsertUser,
  type Project,
  type InsertProject,
  type File,
  type InsertFile,
  type Message,
  type InsertMessage,
  type Agent,
  type InsertAgent,
  type AgentTask,
  type InsertAgentTask,
  type KnowledgeNode,
  type InsertKnowledgeNode,
  type KnowledgeRelationship,
  type InsertKnowledgeRelationship,
  type SystemMetric,
  type InsertSystemMetric,
  type WebSocketConnection,
  type InsertWebSocketConnection,
} from "@shared/schema";
import { db } from "./db";
import { eq, desc, and, sql, count } from "drizzle-orm";

export interface IStorage {
  // User operations (required for Replit Auth)
  getUser(id: string): Promise<User | undefined>;
  upsertUser(user: UpsertUser): Promise<User>;
  
  // Project operations
  getProjects(userId: string): Promise<Project[]>;
  getProject(id: string): Promise<Project | undefined>;
  createProject(project: InsertProject): Promise<Project>;
  updateProject(id: string, updates: Partial<InsertProject>): Promise<Project | undefined>;
  deleteProject(id: string): Promise<boolean>;
  
  // File operations
  getFiles(userId: string, projectId?: string): Promise<File[]>;
  getFile(id: string): Promise<File | undefined>;
  createFile(file: InsertFile): Promise<File>;
  updateFile(id: string, updates: Partial<InsertFile>): Promise<File | undefined>;
  deleteFile(id: string): Promise<boolean>;
  
  // Message operations
  getMessages(projectId?: string, limit?: number): Promise<Message[]>;
  createMessage(message: InsertMessage): Promise<Message>;
  
  // Agent operations
  getAgents(): Promise<Agent[]>;
  getAgent(id: string): Promise<Agent | undefined>;
  createAgent(agent: InsertAgent): Promise<Agent>;
  updateAgent(id: string, updates: Partial<InsertAgent>): Promise<Agent | undefined>;
  deleteAgent(id: string): Promise<boolean>;
  
  // Agent task operations
  getAgentTasks(agentId?: string, limit?: number): Promise<AgentTask[]>;
  createAgentTask(task: InsertAgentTask): Promise<AgentTask>;
  updateAgentTask(id: string, updates: Partial<InsertAgentTask>): Promise<AgentTask | undefined>;
  
  // Knowledge graph operations
  getKnowledgeNodes(type?: string): Promise<KnowledgeNode[]>;
  createKnowledgeNode(node: InsertKnowledgeNode): Promise<KnowledgeNode>;
  getKnowledgeRelationships(nodeId?: string): Promise<KnowledgeRelationship[]>;
  createKnowledgeRelationship(relationship: InsertKnowledgeRelationship): Promise<KnowledgeRelationship>;
  
  // System metrics operations
  getSystemMetrics(type?: string, hours?: number): Promise<SystemMetric[]>;
  createSystemMetric(metric: InsertSystemMetric): Promise<SystemMetric>;
  
  // WebSocket operations
  getActiveConnections(): Promise<WebSocketConnection[]>;
  createConnection(connection: InsertWebSocketConnection): Promise<WebSocketConnection>;
  updateConnection(id: string, updates: Partial<InsertWebSocketConnection>): Promise<WebSocketConnection | undefined>;
  deleteConnection(id: string): Promise<boolean>;
  
  // Dashboard metrics
  getDashboardMetrics(): Promise<{
    totalUsers: number;
    activeUsers: number;
    totalProjects: number;
    activeProjects: number;
    totalFiles: number;
    totalFileSize: number;
    totalMessages: number;
    activeAgents: number;
    knowledgeNodes: number;
    activeConnections: number;
  }>;
}

export class DatabaseStorage implements IStorage {
  // User operations
  async getUser(id: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.id, id));
    return user;
  }

  async upsertUser(userData: UpsertUser): Promise<User> {
    const [user] = await db
      .insert(users)
      .values(userData)
      .onConflictDoUpdate({
        target: users.id,
        set: {
          ...userData,
          updatedAt: new Date(),
        },
      })
      .returning();
    return user;
  }

  // Project operations
  async getProjects(userId: string): Promise<Project[]> {
    return await db
      .select()
      .from(projects)
      .where(eq(projects.userId, userId))
      .orderBy(desc(projects.updatedAt));
  }

  async getProject(id: string): Promise<Project | undefined> {
    const [project] = await db.select().from(projects).where(eq(projects.id, id));
    return project;
  }

  async createProject(project: InsertProject): Promise<Project> {
    const [newProject] = await db.insert(projects).values(project).returning();
    return newProject;
  }

  async updateProject(id: string, updates: Partial<InsertProject>): Promise<Project | undefined> {
    const [updated] = await db
      .update(projects)
      .set({ ...updates, updatedAt: new Date() })
      .where(eq(projects.id, id))
      .returning();
    return updated;
  }

  async deleteProject(id: string): Promise<boolean> {
    const result = await db.delete(projects).where(eq(projects.id, id));
    return (result.rowCount ?? 0) > 0;
  }

  // File operations
  async getFiles(userId: string, projectId?: string): Promise<File[]> {
    let query = db.select().from(files).where(eq(files.userId, userId));
    if (projectId) {
      query = db.select().from(files).where(and(eq(files.userId, userId), eq(files.projectId, projectId)));
    }
    return await query.orderBy(desc(files.createdAt));
  }

  async getFile(id: string): Promise<File | undefined> {
    const [file] = await db.select().from(files).where(eq(files.id, id));
    return file;
  }

  async createFile(file: InsertFile): Promise<File> {
    const [newFile] = await db.insert(files).values(file).returning();
    return newFile;
  }

  async updateFile(id: string, updates: Partial<InsertFile>): Promise<File | undefined> {
    const [updated] = await db
      .update(files)
      .set({ ...updates, updatedAt: new Date() })
      .where(eq(files.id, id))
      .returning();
    return updated;
  }

  async deleteFile(id: string): Promise<boolean> {
    const result = await db.delete(files).where(eq(files.id, id));
    return (result.rowCount ?? 0) > 0;
  }

  // Message operations
  async getMessages(projectId?: string, limit = 100): Promise<Message[]> {
    const query = db.select().from(messages);
    if (projectId) {
      query.where(eq(messages.projectId, projectId));
    }
    return await query.orderBy(desc(messages.createdAt)).limit(limit);
  }

  async createMessage(message: InsertMessage): Promise<Message> {
    const [newMessage] = await db.insert(messages).values(message).returning();
    return newMessage;
  }

  // Agent operations
  async getAgents(): Promise<Agent[]> {
    return await db.select().from(agents).orderBy(desc(agents.lastActivity));
  }

  async getAgent(id: string): Promise<Agent | undefined> {
    const [agent] = await db.select().from(agents).where(eq(agents.id, id));
    return agent;
  }

  async createAgent(agent: InsertAgent): Promise<Agent> {
    const [newAgent] = await db.insert(agents).values(agent).returning();
    return newAgent;
  }

  async updateAgent(id: string, updates: Partial<InsertAgent>): Promise<Agent | undefined> {
    const [updated] = await db
      .update(agents)
      .set({ ...updates, updatedAt: new Date() })
      .where(eq(agents.id, id))
      .returning();
    return updated;
  }

  async deleteAgent(id: string): Promise<boolean> {
    const result = await db.delete(agents).where(eq(agents.id, id));
    return (result.rowCount ?? 0) > 0;
  }

  // Agent task operations
  async getAgentTasks(agentId?: string, limit = 100): Promise<AgentTask[]> {
    const query = db.select().from(agentTasks);
    if (agentId) {
      query.where(eq(agentTasks.agentId, agentId));
    }
    return await query.orderBy(desc(agentTasks.createdAt)).limit(limit);
  }

  async createAgentTask(task: InsertAgentTask): Promise<AgentTask> {
    const [newTask] = await db.insert(agentTasks).values(task).returning();
    return newTask;
  }

  async updateAgentTask(id: string, updates: Partial<InsertAgentTask>): Promise<AgentTask | undefined> {
    const [updated] = await db
      .update(agentTasks)
      .set(updates)
      .where(eq(agentTasks.id, id))
      .returning();
    return updated;
  }

  // Knowledge graph operations
  async getKnowledgeNodes(type?: string): Promise<KnowledgeNode[]> {
    const query = db.select().from(knowledgeNodes);
    if (type) {
      query.where(eq(knowledgeNodes.type, type));
    }
    return await query.orderBy(desc(knowledgeNodes.updatedAt));
  }

  async createKnowledgeNode(node: InsertKnowledgeNode): Promise<KnowledgeNode> {
    const [newNode] = await db.insert(knowledgeNodes).values(node).returning();
    return newNode;
  }

  async getKnowledgeRelationships(nodeId?: string): Promise<KnowledgeRelationship[]> {
    const query = db.select().from(knowledgeRelationships);
    if (nodeId) {
      query.where(eq(knowledgeRelationships.fromNodeId, nodeId));
    }
    return await query.orderBy(desc(knowledgeRelationships.createdAt));
  }

  async createKnowledgeRelationship(relationship: InsertKnowledgeRelationship): Promise<KnowledgeRelationship> {
    const [newRelationship] = await db.insert(knowledgeRelationships).values(relationship).returning();
    return newRelationship;
  }

  // System metrics operations
  async getSystemMetrics(type?: string, hours = 24): Promise<SystemMetric[]> {
    const query = db.select().from(systemMetrics);
    const conditions = [];
    
    if (type) {
      conditions.push(eq(systemMetrics.metricType, type));
    }
    
    const timeLimit = new Date(Date.now() - hours * 60 * 60 * 1000);
    conditions.push(sql`${systemMetrics.timestamp} >= ${timeLimit}`);
    
    if (conditions.length > 0) {
      query.where(and(...conditions));
    }
    
    return await query.orderBy(desc(systemMetrics.timestamp));
  }

  async createSystemMetric(metric: InsertSystemMetric): Promise<SystemMetric> {
    const [newMetric] = await db.insert(systemMetrics).values(metric).returning();
    return newMetric;
  }

  // WebSocket operations
  async getActiveConnections(): Promise<WebSocketConnection[]> {
    return await db
      .select()
      .from(webSocketConnections)
      .where(eq(webSocketConnections.status, 'connected'))
      .orderBy(desc(webSocketConnections.lastPing));
  }

  async createConnection(connection: InsertWebSocketConnection): Promise<WebSocketConnection> {
    const [newConnection] = await db.insert(webSocketConnections).values(connection).returning();
    return newConnection;
  }

  async updateConnection(id: string, updates: Partial<InsertWebSocketConnection>): Promise<WebSocketConnection | undefined> {
    const [updated] = await db
      .update(webSocketConnections)
      .set(updates)
      .where(eq(webSocketConnections.id, id))
      .returning();
    return updated;
  }

  async deleteConnection(id: string): Promise<boolean> {
    const result = await db.delete(webSocketConnections).where(eq(webSocketConnections.id, id));
    return (result.rowCount ?? 0) > 0;
  }

  // Dashboard metrics
  async getDashboardMetrics() {
    const [userCounts] = await db
      .select({
        total: count(),
        active: sql<number>`count(*) filter (where ${users.updatedAt} > now() - interval '24 hours')`,
      })
      .from(users);

    const [projectCounts] = await db
      .select({
        total: count(),
        active: sql<number>`count(*) filter (where ${projects.status} = 'active')`,
      })
      .from(projects);

    const [fileCounts] = await db
      .select({
        total: count(),
        totalSize: sql<number>`coalesce(sum(${files.size}), 0)`,
      })
      .from(files);

    const [messageCounts] = await db
      .select({ total: count() })
      .from(messages);

    const [agentCounts] = await db
      .select({
        active: sql<number>`count(*) filter (where ${agents.status} = 'active')`,
      })
      .from(agents);

    const [knowledgeCounts] = await db
      .select({ total: count() })
      .from(knowledgeNodes);

    const [connectionCounts] = await db
      .select({
        active: sql<number>`count(*) filter (where ${webSocketConnections.status} = 'connected')`,
      })
      .from(webSocketConnections);

    return {
      totalUsers: userCounts.total,
      activeUsers: userCounts.active,
      totalProjects: projectCounts.total,
      activeProjects: projectCounts.active,
      totalFiles: fileCounts.total,
      totalFileSize: fileCounts.totalSize,
      totalMessages: messageCounts.total,
      activeAgents: agentCounts.active,
      knowledgeNodes: knowledgeCounts.total,
      activeConnections: connectionCounts.active,
    };
  }
}

export const storage = new DatabaseStorage();
