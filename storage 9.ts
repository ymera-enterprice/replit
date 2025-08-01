import { agents, projects, tasks, collaborations, knowledgeItems, learningSessions, systemMetrics, users } from "@shared/schema";
import type { Agent, InsertAgent, Project, InsertProject, Task, InsertTask, Collaboration, InsertCollaboration, KnowledgeItem, InsertKnowledgeItem, LearningSession, InsertLearningSession, SystemMetric, InsertSystemMetric, User, InsertUser } from "@shared/schema";
import { db } from "./db";
import { eq, and, desc, asc, like, count, sql } from "drizzle-orm";

export interface IStorage {
  // User operations
  getUser(id: number): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(insertUser: InsertUser): Promise<User>;
  
  // Agent operations
  getAgents(filters?: { status?: string; type?: string; search?: string; limit?: number; offset?: number }): Promise<{ agents: Agent[]; total: number }>;
  getAgent(id: string): Promise<Agent | undefined>;
  createAgent(insertAgent: InsertAgent): Promise<Agent>;
  updateAgent(id: string, updates: Partial<InsertAgent>): Promise<Agent | undefined>;
  deleteAgent(id: string): Promise<boolean>;
  
  // Project operations
  getProjects(filters?: { status?: string; ownerId?: number; limit?: number; offset?: number }): Promise<{ projects: Project[]; total: number }>;
  getProject(id: string): Promise<Project | undefined>;
  createProject(insertProject: InsertProject): Promise<Project>;
  updateProject(id: string, updates: Partial<InsertProject>): Promise<Project | undefined>;
  deleteProject(id: string): Promise<boolean>;
  
  // Task operations
  getTasks(filters?: { agentId?: string; projectId?: string; status?: string; limit?: number; offset?: number }): Promise<{ tasks: Task[]; total: number }>;
  getTask(id: string): Promise<Task | undefined>;
  createTask(insertTask: InsertTask): Promise<Task>;
  updateTask(id: string, updates: Partial<InsertTask>): Promise<Task | undefined>;
  deleteTask(id: string): Promise<boolean>;
  
  // Collaboration operations
  getCollaborations(filters?: { projectId?: string; status?: string; limit?: number; offset?: number }): Promise<{ collaborations: Collaboration[]; total: number }>;
  getCollaboration(id: string): Promise<Collaboration | undefined>;
  createCollaboration(insertCollaboration: InsertCollaboration): Promise<Collaboration>;
  updateCollaboration(id: string, updates: Partial<InsertCollaboration>): Promise<Collaboration | undefined>;
  deleteCollaboration(id: string): Promise<boolean>;
  
  // Knowledge operations
  getKnowledgeItems(filters?: { type?: string; tags?: string[]; projectId?: string; isPublic?: boolean; limit?: number; offset?: number }): Promise<{ items: KnowledgeItem[]; total: number }>;
  getKnowledgeItem(id: string): Promise<KnowledgeItem | undefined>;
  createKnowledgeItem(insertKnowledgeItem: InsertKnowledgeItem): Promise<KnowledgeItem>;
  updateKnowledgeItem(id: string, updates: Partial<InsertKnowledgeItem>): Promise<KnowledgeItem | undefined>;
  deleteKnowledgeItem(id: string): Promise<boolean>;
  
  // Learning operations
  getLearningSessions(filters?: { agentId?: string; sessionType?: string; limit?: number; offset?: number }): Promise<{ sessions: LearningSession[]; total: number }>;
  getLearningSession(id: string): Promise<LearningSession | undefined>;
  createLearningSession(insertLearningSession: InsertLearningSession): Promise<LearningSession>;
  updateLearningSession(id: string, updates: Partial<InsertLearningSession>): Promise<LearningSession | undefined>;
  
  // System metrics operations
  getSystemMetrics(filters?: { metricType?: string; limit?: number; from?: Date; to?: Date }): Promise<SystemMetric[]>;
  createSystemMetric(insertSystemMetric: InsertSystemMetric): Promise<SystemMetric>;
  
  // Dashboard operations
  getDashboardStats(): Promise<{
    activeAgents: number;
    totalProjects: number;
    learningSessions: number;
    knowledgeItems: number;
    systemHealth: number;
  }>;
}

export class DatabaseStorage implements IStorage {
  // User operations
  async getUser(id: number): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.id, id));
    return user || undefined;
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.username, username));
    return user || undefined;
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const [user] = await db.insert(users).values({
      ...insertUser,
      createdAt: new Date(),
      updatedAt: new Date(),
    }).returning();
    return user;
  }

  // Agent operations
  async getAgents(filters?: { status?: string; type?: string; search?: string; limit?: number; offset?: number }): Promise<{ agents: Agent[]; total: number }> {
    let query = db.select().from(agents);
    let countQuery = db.select({ count: count() }).from(agents);
    
    const conditions = [];
    
    if (filters?.status) {
      conditions.push(eq(agents.status, filters.status as any));
    }
    
    if (filters?.type) {
      conditions.push(eq(agents.type, filters.type as any));
    }
    
    if (filters?.search) {
      conditions.push(
        sql`${agents.name} ILIKE ${`%${filters.search}%`} OR ${agents.description} ILIKE ${`%${filters.search}%`}`
      );
    }
    
    if (conditions.length > 0) {
      query = query.where(and(...conditions));
      countQuery = countQuery.where(and(...conditions));
    }
    
    query = query.orderBy(desc(agents.updatedAt));
    
    if (filters?.limit) {
      query = query.limit(filters.limit);
    }
    
    if (filters?.offset) {
      query = query.offset(filters.offset);
    }
    
    const [agentResults, countResult] = await Promise.all([
      query,
      countQuery
    ]);
    
    return {
      agents: agentResults,
      total: countResult[0]?.count || 0
    };
  }

  async getAgent(id: string): Promise<Agent | undefined> {
    const [agent] = await db.select().from(agents).where(eq(agents.id, id));
    return agent || undefined;
  }

  async createAgent(insertAgent: InsertAgent): Promise<Agent> {
    const [agent] = await db.insert(agents).values({
      ...insertAgent,
      createdAt: new Date(),
      updatedAt: new Date(),
    }).returning();
    return agent;
  }

  async updateAgent(id: string, updates: Partial<InsertAgent>): Promise<Agent | undefined> {
    const [agent] = await db.update(agents)
      .set({ ...updates, updatedAt: new Date() })
      .where(eq(agents.id, id))
      .returning();
    return agent || undefined;
  }

  async deleteAgent(id: string): Promise<boolean> {
    const result = await db.delete(agents).where(eq(agents.id, id));
    return result.rowCount > 0;
  }

  // Project operations
  async getProjects(filters?: { status?: string; ownerId?: number; limit?: number; offset?: number }): Promise<{ projects: Project[]; total: number }> {
    let query = db.select().from(projects);
    let countQuery = db.select({ count: count() }).from(projects);
    
    const conditions = [];
    
    if (filters?.status) {
      conditions.push(eq(projects.status, filters.status as any));
    }
    
    if (filters?.ownerId) {
      conditions.push(eq(projects.ownerId, filters.ownerId));
    }
    
    if (conditions.length > 0) {
      query = query.where(and(...conditions));
      countQuery = countQuery.where(and(...conditions));
    }
    
    query = query.orderBy(desc(projects.updatedAt));
    
    if (filters?.limit) {
      query = query.limit(filters.limit);
    }
    
    if (filters?.offset) {
      query = query.offset(filters.offset);
    }
    
    const [projectResults, countResult] = await Promise.all([
      query,
      countQuery
    ]);
    
    return {
      projects: projectResults,
      total: countResult[0]?.count || 0
    };
  }

  async getProject(id: string): Promise<Project | undefined> {
    const [project] = await db.select().from(projects).where(eq(projects.id, id));
    return project || undefined;
  }

  async createProject(insertProject: InsertProject): Promise<Project> {
    const [project] = await db.insert(projects).values({
      ...insertProject,
      createdAt: new Date(),
      updatedAt: new Date(),
    }).returning();
    return project;
  }

  async updateProject(id: string, updates: Partial<InsertProject>): Promise<Project | undefined> {
    const [project] = await db.update(projects)
      .set({ ...updates, updatedAt: new Date() })
      .where(eq(projects.id, id))
      .returning();
    return project || undefined;
  }

  async deleteProject(id: string): Promise<boolean> {
    const result = await db.delete(projects).where(eq(projects.id, id));
    return result.rowCount > 0;
  }

  // Task operations
  async getTasks(filters?: { agentId?: string; projectId?: string; status?: string; limit?: number; offset?: number }): Promise<{ tasks: Task[]; total: number }> {
    let query = db.select().from(tasks);
    let countQuery = db.select({ count: count() }).from(tasks);
    
    const conditions = [];
    
    if (filters?.agentId) {
      conditions.push(eq(tasks.agentId, filters.agentId));
    }
    
    if (filters?.projectId) {
      conditions.push(eq(tasks.projectId, filters.projectId));
    }
    
    if (filters?.status) {
      conditions.push(eq(tasks.status, filters.status as any));
    }
    
    if (conditions.length > 0) {
      query = query.where(and(...conditions));
      countQuery = countQuery.where(and(...conditions));
    }
    
    query = query.orderBy(desc(tasks.createdAt));
    
    if (filters?.limit) {
      query = query.limit(filters.limit);
    }
    
    if (filters?.offset) {
      query = query.offset(filters.offset);
    }
    
    const [taskResults, countResult] = await Promise.all([
      query,
      countQuery
    ]);
    
    return {
      tasks: taskResults,
      total: countResult[0]?.count || 0
    };
  }

  async getTask(id: string): Promise<Task | undefined> {
    const [task] = await db.select().from(tasks).where(eq(tasks.id, id));
    return task || undefined;
  }

  async createTask(insertTask: InsertTask): Promise<Task> {
    const [task] = await db.insert(tasks).values({
      ...insertTask,
      createdAt: new Date(),
      updatedAt: new Date(),
    }).returning();
    return task;
  }

  async updateTask(id: string, updates: Partial<InsertTask>): Promise<Task | undefined> {
    const [task] = await db.update(tasks)
      .set({ ...updates, updatedAt: new Date() })
      .where(eq(tasks.id, id))
      .returning();
    return task || undefined;
  }

  async deleteTask(id: string): Promise<boolean> {
    const result = await db.delete(tasks).where(eq(tasks.id, id));
    return result.rowCount > 0;
  }

  // Collaboration operations
  async getCollaborations(filters?: { projectId?: string; status?: string; limit?: number; offset?: number }): Promise<{ collaborations: Collaboration[]; total: number }> {
    let query = db.select().from(collaborations);
    let countQuery = db.select({ count: count() }).from(collaborations);
    
    const conditions = [];
    
    if (filters?.projectId) {
      conditions.push(eq(collaborations.projectId, filters.projectId));
    }
    
    if (filters?.status) {
      conditions.push(eq(collaborations.status, filters.status as any));
    }
    
    if (conditions.length > 0) {
      query = query.where(and(...conditions));
      countQuery = countQuery.where(and(...conditions));
    }
    
    query = query.orderBy(desc(collaborations.updatedAt));
    
    if (filters?.limit) {
      query = query.limit(filters.limit);
    }
    
    if (filters?.offset) {
      query = query.offset(filters.offset);
    }
    
    const [collaborationResults, countResult] = await Promise.all([
      query,
      countQuery
    ]);
    
    return {
      collaborations: collaborationResults,
      total: countResult[0]?.count || 0
    };
  }

  async getCollaboration(id: string): Promise<Collaboration | undefined> {
    const [collaboration] = await db.select().from(collaborations).where(eq(collaborations.id, id));
    return collaboration || undefined;
  }

  async createCollaboration(insertCollaboration: InsertCollaboration): Promise<Collaboration> {
    const [collaboration] = await db.insert(collaborations).values({
      ...insertCollaboration,
      createdAt: new Date(),
      updatedAt: new Date(),
    }).returning();
    return collaboration;
  }

  async updateCollaboration(id: string, updates: Partial<InsertCollaboration>): Promise<Collaboration | undefined> {
    const [collaboration] = await db.update(collaborations)
      .set({ ...updates, updatedAt: new Date() })
      .where(eq(collaborations.id, id))
      .returning();
    return collaboration || undefined;
  }

  async deleteCollaboration(id: string): Promise<boolean> {
    const result = await db.delete(collaborations).where(eq(collaborations.id, id));
    return result.rowCount > 0;
  }

  // Knowledge operations
  async getKnowledgeItems(filters?: { type?: string; tags?: string[]; projectId?: string; isPublic?: boolean; limit?: number; offset?: number }): Promise<{ items: KnowledgeItem[]; total: number }> {
    let query = db.select().from(knowledgeItems);
    let countQuery = db.select({ count: count() }).from(knowledgeItems);
    
    const conditions = [];
    
    if (filters?.type) {
      conditions.push(eq(knowledgeItems.type, filters.type));
    }
    
    if (filters?.projectId) {
      conditions.push(eq(knowledgeItems.projectId, filters.projectId));
    }
    
    if (filters?.isPublic !== undefined) {
      conditions.push(eq(knowledgeItems.isPublic, filters.isPublic));
    }
    
    if (conditions.length > 0) {
      query = query.where(and(...conditions));
      countQuery = countQuery.where(and(...conditions));
    }
    
    query = query.orderBy(desc(knowledgeItems.updatedAt));
    
    if (filters?.limit) {
      query = query.limit(filters.limit);
    }
    
    if (filters?.offset) {
      query = query.offset(filters.offset);
    }
    
    const [itemResults, countResult] = await Promise.all([
      query,
      countQuery
    ]);
    
    return {
      items: itemResults,
      total: countResult[0]?.count || 0
    };
  }

  async getKnowledgeItem(id: string): Promise<KnowledgeItem | undefined> {
    const [item] = await db.select().from(knowledgeItems).where(eq(knowledgeItems.id, id));
    return item || undefined;
  }

  async createKnowledgeItem(insertKnowledgeItem: InsertKnowledgeItem): Promise<KnowledgeItem> {
    const [item] = await db.insert(knowledgeItems).values({
      ...insertKnowledgeItem,
      createdAt: new Date(),
      updatedAt: new Date(),
    }).returning();
    return item;
  }

  async updateKnowledgeItem(id: string, updates: Partial<InsertKnowledgeItem>): Promise<KnowledgeItem | undefined> {
    const [item] = await db.update(knowledgeItems)
      .set({ ...updates, updatedAt: new Date() })
      .where(eq(knowledgeItems.id, id))
      .returning();
    return item || undefined;
  }

  async deleteKnowledgeItem(id: string): Promise<boolean> {
    const result = await db.delete(knowledgeItems).where(eq(knowledgeItems.id, id));
    return result.rowCount > 0;
  }

  // Learning operations
  async getLearningSessions(filters?: { agentId?: string; sessionType?: string; limit?: number; offset?: number }): Promise<{ sessions: LearningSession[]; total: number }> {
    let query = db.select().from(learningSessions);
    let countQuery = db.select({ count: count() }).from(learningSessions);
    
    const conditions = [];
    
    if (filters?.agentId) {
      conditions.push(eq(learningSessions.agentId, filters.agentId));
    }
    
    if (filters?.sessionType) {
      conditions.push(eq(learningSessions.sessionType, filters.sessionType));
    }
    
    if (conditions.length > 0) {
      query = query.where(and(...conditions));
      countQuery = countQuery.where(and(...conditions));
    }
    
    query = query.orderBy(desc(learningSessions.createdAt));
    
    if (filters?.limit) {
      query = query.limit(filters.limit);
    }
    
    if (filters?.offset) {
      query = query.offset(filters.offset);
    }
    
    const [sessionResults, countResult] = await Promise.all([
      query,
      countQuery
    ]);
    
    return {
      sessions: sessionResults,
      total: countResult[0]?.count || 0
    };
  }

  async getLearningSession(id: string): Promise<LearningSession | undefined> {
    const [session] = await db.select().from(learningSessions).where(eq(learningSessions.id, id));
    return session || undefined;
  }

  async createLearningSession(insertLearningSession: InsertLearningSession): Promise<LearningSession> {
    const [session] = await db.insert(learningSessions).values({
      ...insertLearningSession,
      createdAt: new Date(),
    }).returning();
    return session;
  }

  async updateLearningSession(id: string, updates: Partial<InsertLearningSession>): Promise<LearningSession | undefined> {
    const [session] = await db.update(learningSessions)
      .set(updates)
      .where(eq(learningSessions.id, id))
      .returning();
    return session || undefined;
  }

  // System metrics operations
  async getSystemMetrics(filters?: { metricType?: string; limit?: number; from?: Date; to?: Date }): Promise<SystemMetric[]> {
    let query = db.select().from(systemMetrics);
    
    const conditions = [];
    
    if (filters?.metricType) {
      conditions.push(eq(systemMetrics.metricType, filters.metricType));
    }
    
    if (filters?.from) {
      conditions.push(sql`${systemMetrics.timestamp} >= ${filters.from}`);
    }
    
    if (filters?.to) {
      conditions.push(sql`${systemMetrics.timestamp} <= ${filters.to}`);
    }
    
    if (conditions.length > 0) {
      query = query.where(and(...conditions));
    }
    
    query = query.orderBy(desc(systemMetrics.timestamp));
    
    if (filters?.limit) {
      query = query.limit(filters.limit);
    }
    
    return await query;
  }

  async createSystemMetric(insertSystemMetric: InsertSystemMetric): Promise<SystemMetric> {
    const [metric] = await db.insert(systemMetrics).values({
      ...insertSystemMetric,
      timestamp: new Date(),
    }).returning();
    return metric;
  }

  // Dashboard operations
  async getDashboardStats(): Promise<{
    activeAgents: number;
    totalProjects: number;
    learningSessions: number;
    knowledgeItems: number;
    systemHealth: number;
  }> {
    const [activeAgentsResult] = await db
      .select({ count: count() })
      .from(agents)
      .where(eq(agents.status, 'active'));
    
    const [totalProjectsResult] = await db
      .select({ count: count() })
      .from(projects);
    
    const [learningSessionsResult] = await db
      .select({ count: count() })
      .from(learningSessions)
      .where(sql`${learningSessions.createdAt} >= NOW() - INTERVAL '24 hours'`);
    
    const [knowledgeItemsResult] = await db
      .select({ count: count() })
      .from(knowledgeItems);
    
    // Calculate system health based on various metrics
    const [avgHealthResult] = await db
      .select({ avg: sql<number>`AVG(${agents.healthScore})` })
      .from(agents)
      .where(eq(agents.status, 'active'));
    
    return {
      activeAgents: activeAgentsResult?.count || 0,
      totalProjects: totalProjectsResult?.count || 0,
      learningSessions: learningSessionsResult?.count || 0,
      knowledgeItems: knowledgeItemsResult?.count || 0,
      systemHealth: Math.round(avgHealthResult?.avg || 100),
    };
  }
}

export const storage = new DatabaseStorage();
