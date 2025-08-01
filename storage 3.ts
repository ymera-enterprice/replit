import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import * as schema from '@shared/schema';
import { eq, and, desc, asc, count, sql, or, like, gte, lte } from 'drizzle-orm';
import { createLogger } from './logger';

const logger = createLogger('storage');

// Database connection configuration
const connectionString = process.env.DATABASE_URL;
if (!connectionString) {
  throw new Error('DATABASE_URL environment variable is required');
}

// Create PostgreSQL connection
const connection = postgres(connectionString, {
  max: 20, // Maximum connections in pool
  idle_timeout: 20, // Close idle connections after 20 seconds
  connect_timeout: 10, // Connection timeout in seconds
  prepare: false, // Disable prepared statements for better compatibility
});

// Create Drizzle database instance
export const db = drizzle(connection, { schema });

// Test database connection
export async function testConnection(): Promise<boolean> {
  try {
    await connection`SELECT 1`;
    logger.info('Database connection successful');
    return true;
  } catch (error) {
    logger.error('Database connection failed', { error: error.message });
    return false;
  }
}

// Storage interface implementation
export interface IStorage {
  // User management
  insertUser(user: schema.InsertUser): Promise<schema.SelectUser>;
  getUserById(id: string): Promise<schema.SelectUser | null>;
  getUserByUsername(username: string): Promise<schema.SelectUser | null>;
  getUserByEmail(email: string): Promise<schema.SelectUser | null>;
  updateUser(id: string, updates: Partial<schema.InsertUser>): Promise<schema.SelectUser | null>;
  deleteUser(id: string): Promise<boolean>;
  listUsers(filters?: {
    status?: string;
    limit?: number;
    offset?: number;
    search?: string;
  }): Promise<{ users: schema.SelectUser[]; total: number }>;

  // Session management
  insertSession(session: schema.InsertSession): Promise<schema.SelectSession>;
  getSessionByToken(token: string): Promise<schema.SelectSession | null>;
  updateSession(id: string, updates: Partial<schema.InsertSession>): Promise<schema.SelectSession | null>;
  deleteSession(id: string): Promise<boolean>;
  deleteUserSessions(userId: string): Promise<number>;
  cleanupExpiredSessions(): Promise<number>;

  // Activity tracking
  insertActivity(activity: schema.InsertActivity): Promise<schema.SelectActivity>;
  getUserActivities(userId: string, options?: {
    limit?: number;
    offset?: number;
    type?: string;
    startDate?: Date;
    endDate?: Date;
  }): Promise<{ activities: schema.SelectActivity[]; total: number }>;

  // Agent management
  insertAgent(agent: schema.InsertAgent): Promise<schema.SelectAgent>;
  getAgentById(id: string): Promise<schema.SelectAgent | null>;
  getAgentByAgentId(agentId: string): Promise<schema.SelectAgent | null>;
  updateAgent(id: string, updates: Partial<schema.InsertAgent>): Promise<schema.SelectAgent | null>;
  deleteAgent(id: string): Promise<boolean>;
  listAgents(filters?: {
    status?: string;
    type?: string;
    enabled?: boolean;
    limit?: number;
    offset?: number;
    search?: string;
  }): Promise<{ agents: schema.SelectAgent[]; total: number }>;

  // Learning data management
  insertLearningData(data: schema.InsertLearningData): Promise<schema.SelectLearningData>;
  getLearningDataById(id: string): Promise<schema.SelectLearningData | null>;
  listLearningData(filters?: {
    sourceAgentId?: string;
    knowledgeType?: string;
    minConfidence?: number;
    limit?: number;
    offset?: number;
  }): Promise<{ data: schema.SelectLearningData[]; total: number }>;
  deleteLearningData(id: string): Promise<boolean>;

  // System health
  insertHealthRecord(record: {
    component: string;
    status: string;
    details: any;
    responseTimeMs?: number;
  }): Promise<void>;
  getHealthHistory(component?: string, limit?: number): Promise<any[]>;
}

export class PostgreSQLStorage implements IStorage {
  // User management
  async insertUser(user: schema.InsertUser): Promise<schema.SelectUser> {
    try {
      const [newUser] = await db.insert(schema.users).values(user).returning();
      logger.info('User created', { userId: newUser.id, username: newUser.username });
      return newUser;
    } catch (error) {
      logger.error('Failed to create user', { error: error.message, username: user.username });
      throw error;
    }
  }

  async getUserById(id: string): Promise<schema.SelectUser | null> {
    try {
      const [user] = await db.select().from(schema.users).where(eq(schema.users.id, id)).limit(1);
      return user || null;
    } catch (error) {
      logger.error('Failed to get user by ID', { error: error.message, userId: id });
      throw error;
    }
  }

  async getUserByUsername(username: string): Promise<schema.SelectUser | null> {
    try {
      const [user] = await db.select().from(schema.users).where(eq(schema.users.username, username)).limit(1);
      return user || null;
    } catch (error) {
      logger.error('Failed to get user by username', { error: error.message, username });
      throw error;
    }
  }

  async getUserByEmail(email: string): Promise<schema.SelectUser | null> {
    try {
      const [user] = await db.select().from(schema.users).where(eq(schema.users.email, email)).limit(1);
      return user || null;
    } catch (error) {
      logger.error('Failed to get user by email', { error: error.message, email });
      throw error;
    }
  }

  async updateUser(id: string, updates: Partial<schema.InsertUser>): Promise<schema.SelectUser | null> {
    try {
      const [updatedUser] = await db
        .update(schema.users)
        .set({ ...updates, updatedAt: new Date() })
        .where(eq(schema.users.id, id))
        .returning();
      
      if (updatedUser) {
        logger.info('User updated', { userId: id });
      }
      
      return updatedUser || null;
    } catch (error) {
      logger.error('Failed to update user', { error: error.message, userId: id });
      throw error;
    }
  }

  async deleteUser(id: string): Promise<boolean> {
    try {
      const result = await db.delete(schema.users).where(eq(schema.users.id, id));
      const deleted = result.rowCount > 0;
      
      if (deleted) {
        logger.info('User deleted', { userId: id });
      }
      
      return deleted;
    } catch (error) {
      logger.error('Failed to delete user', { error: error.message, userId: id });
      throw error;
    }
  }

  async listUsers(filters: {
    status?: string;
    limit?: number;
    offset?: number;
    search?: string;
  } = {}): Promise<{ users: schema.SelectUser[]; total: number }> {
    try {
      const { status, limit = 50, offset = 0, search } = filters;
      
      let query = db.select().from(schema.users);
      let countQuery = db.select({ count: count() }).from(schema.users);
      
      const conditions = [];
      
      if (status) {
        conditions.push(eq(schema.users.userStatus, status));
      }
      
      if (search) {
        conditions.push(
          or(
            like(schema.users.username, `%${search}%`),
            like(schema.users.email, `%${search}%`),
            like(schema.users.firstName, `%${search}%`),
            like(schema.users.lastName, `%${search}%`)
          )
        );
      }
      
      if (conditions.length > 0) {
        query = query.where(and(...conditions));
        countQuery = countQuery.where(and(...conditions));
      }
      
      const [users, [{ count: total }]] = await Promise.all([
        query.orderBy(desc(schema.users.createdAt)).limit(limit).offset(offset),
        countQuery
      ]);
      
      return { users, total };
    } catch (error) {
      logger.error('Failed to list users', { error: error.message, filters });
      throw error;
    }
  }

  // Session management
  async insertSession(session: schema.InsertSession): Promise<schema.SelectSession> {
    try {
      const [newSession] = await db.insert(schema.userSessions).values(session).returning();
      logger.debug('Session created', { sessionId: newSession.id, userId: newSession.userId });
      return newSession;
    } catch (error) {
      logger.error('Failed to create session', { error: error.message });
      throw error;
    }
  }

  async getSessionByToken(token: string): Promise<schema.SelectSession | null> {
    try {
      const [session] = await db
        .select()
        .from(schema.userSessions)
        .where(eq(schema.userSessions.sessionToken, token))
        .limit(1);
      return session || null;
    } catch (error) {
      logger.error('Failed to get session by token', { error: error.message });
      throw error;
    }
  }

  async updateSession(id: string, updates: Partial<schema.InsertSession>): Promise<schema.SelectSession | null> {
    try {
      const [updatedSession] = await db
        .update(schema.userSessions)
        .set(updates)
        .where(eq(schema.userSessions.id, id))
        .returning();
      return updatedSession || null;
    } catch (error) {
      logger.error('Failed to update session', { error: error.message, sessionId: id });
      throw error;
    }
  }

  async deleteSession(id: string): Promise<boolean> {
    try {
      const result = await db.delete(schema.userSessions).where(eq(schema.userSessions.id, id));
      return result.rowCount > 0;
    } catch (error) {
      logger.error('Failed to delete session', { error: error.message, sessionId: id });
      throw error;
    }
  }

  async deleteUserSessions(userId: string): Promise<number> {
    try {
      const result = await db.delete(schema.userSessions).where(eq(schema.userSessions.userId, userId));
      logger.info('User sessions deleted', { userId, count: result.rowCount });
      return result.rowCount;
    } catch (error) {
      logger.error('Failed to delete user sessions', { error: error.message, userId });
      throw error;
    }
  }

  async cleanupExpiredSessions(): Promise<number> {
    try {
      const now = new Date();
      const result = await db
        .delete(schema.userSessions)
        .where(
          or(
            lte(schema.userSessions.expiresAt, now),
            eq(schema.userSessions.isActive, false)
          )
        );
      
      if (result.rowCount > 0) {
        logger.info('Expired sessions cleaned up', { count: result.rowCount });
      }
      
      return result.rowCount;
    } catch (error) {
      logger.error('Failed to cleanup expired sessions', { error: error.message });
      throw error;
    }
  }

  // Activity tracking
  async insertActivity(activity: schema.InsertActivity): Promise<schema.SelectActivity> {
    try {
      const [newActivity] = await db.insert(schema.userActivities).values(activity).returning();
      return newActivity;
    } catch (error) {
      logger.error('Failed to create activity', { error: error.message });
      throw error;
    }
  }

  async getUserActivities(
    userId: string,
    options: {
      limit?: number;
      offset?: number;
      type?: string;
      startDate?: Date;
      endDate?: Date;
    } = {}
  ): Promise<{ activities: schema.SelectActivity[]; total: number }> {
    try {
      const { limit = 50, offset = 0, type, startDate, endDate } = options;
      
      let query = db.select().from(schema.userActivities).where(eq(schema.userActivities.userId, userId));
      let countQuery = db
        .select({ count: count() })
        .from(schema.userActivities)
        .where(eq(schema.userActivities.userId, userId));
      
      const conditions = [eq(schema.userActivities.userId, userId)];
      
      if (type) {
        conditions.push(eq(schema.userActivities.activityType, type));
      }
      
      if (startDate) {
        conditions.push(gte(schema.userActivities.createdAt, startDate));
      }
      
      if (endDate) {
        conditions.push(lte(schema.userActivities.createdAt, endDate));
      }
      
      if (conditions.length > 1) {
        query = query.where(and(...conditions));
        countQuery = countQuery.where(and(...conditions));
      }
      
      const [activities, [{ count: total }]] = await Promise.all([
        query.orderBy(desc(schema.userActivities.createdAt)).limit(limit).offset(offset),
        countQuery
      ]);
      
      return { activities, total };
    } catch (error) {
      logger.error('Failed to get user activities', { error: error.message, userId });
      throw error;
    }
  }

  // Agent management
  async insertAgent(agent: schema.InsertAgent): Promise<schema.SelectAgent> {
    try {
      const [newAgent] = await db.insert(schema.agents).values(agent).returning();
      logger.info('Agent created', { agentId: newAgent.agentId, type: newAgent.agentType });
      return newAgent;
    } catch (error) {
      logger.error('Failed to create agent', { error: error.message, agentId: agent.agentId });
      throw error;
    }
  }

  async getAgentById(id: string): Promise<schema.SelectAgent | null> {
    try {
      const [agent] = await db.select().from(schema.agents).where(eq(schema.agents.id, id)).limit(1);
      return agent || null;
    } catch (error) {
      logger.error('Failed to get agent by ID', { error: error.message, agentId: id });
      throw error;
    }
  }

  async getAgentByAgentId(agentId: string): Promise<schema.SelectAgent | null> {
    try {
      const [agent] = await db.select().from(schema.agents).where(eq(schema.agents.agentId, agentId)).limit(1);
      return agent || null;
    } catch (error) {
      logger.error('Failed to get agent by agent ID', { error: error.message, agentId });
      throw error;
    }
  }

  async updateAgent(id: string, updates: Partial<schema.InsertAgent>): Promise<schema.SelectAgent | null> {
    try {
      const [updatedAgent] = await db
        .update(schema.agents)
        .set({ ...updates, updatedAt: new Date() })
        .where(eq(schema.agents.id, id))
        .returning();
      
      if (updatedAgent) {
        logger.info('Agent updated', { agentId: updatedAgent.agentId });
      }
      
      return updatedAgent || null;
    } catch (error) {
      logger.error('Failed to update agent', { error: error.message, agentId: id });
      throw error;
    }
  }

  async deleteAgent(id: string): Promise<boolean> {
    try {
      const result = await db.delete(schema.agents).where(eq(schema.agents.id, id));
      const deleted = result.rowCount > 0;
      
      if (deleted) {
        logger.info('Agent deleted', { agentId: id });
      }
      
      return deleted;
    } catch (error) {
      logger.error('Failed to delete agent', { error: error.message, agentId: id });
      throw error;
    }
  }

  async listAgents(filters: {
    status?: string;
    type?: string;
    enabled?: boolean;
    limit?: number;
    offset?: number;
    search?: string;
  } = {}): Promise<{ agents: schema.SelectAgent[]; total: number }> {
    try {
      const { status, type, enabled, limit = 50, offset = 0, search } = filters;
      
      let query = db.select().from(schema.agents);
      let countQuery = db.select({ count: count() }).from(schema.agents);
      
      const conditions = [];
      
      if (status) {
        conditions.push(eq(schema.agents.status, status));
      }
      
      if (type) {
        conditions.push(eq(schema.agents.agentType, type));
      }
      
      if (enabled !== undefined) {
        conditions.push(eq(schema.agents.enabled, enabled));
      }
      
      if (search) {
        conditions.push(
          or(
            like(schema.agents.agentId, `%${search}%`),
            like(schema.agents.name, `%${search}%`),
            like(schema.agents.description, `%${search}%`)
          )
        );
      }
      
      if (conditions.length > 0) {
        query = query.where(and(...conditions));
        countQuery = countQuery.where(and(...conditions));
      }
      
      const [agents, [{ count: total }]] = await Promise.all([
        query.orderBy(desc(schema.agents.createdAt)).limit(limit).offset(offset),
        countQuery
      ]);
      
      return { agents, total };
    } catch (error) {
      logger.error('Failed to list agents', { error: error.message, filters });
      throw error;
    }
  }

  // Learning data management
  async insertLearningData(data: schema.InsertLearningData): Promise<schema.SelectLearningData> {
    try {
      const [newLearningData] = await db.insert(schema.agentLearningData).values(data).returning();
      logger.debug('Learning data created', { 
        id: newLearningData.id,
        sourceAgentId: newLearningData.sourceAgentId,
        knowledgeType: newLearningData.knowledgeType 
      });
      return newLearningData;
    } catch (error) {
      logger.error('Failed to create learning data', { error: error.message });
      throw error;
    }
  }

  async getLearningDataById(id: string): Promise<schema.SelectLearningData | null> {
    try {
      const [learningData] = await db
        .select()
        .from(schema.agentLearningData)
        .where(eq(schema.agentLearningData.id, id))
        .limit(1);
      return learningData || null;
    } catch (error) {
      logger.error('Failed to get learning data by ID', { error: error.message, id });
      throw error;
    }
  }

  async listLearningData(filters: {
    sourceAgentId?: string;
    knowledgeType?: string;
    minConfidence?: number;
    limit?: number;
    offset?: number;
  } = {}): Promise<{ data: schema.SelectLearningData[]; total: number }> {
    try {
      const { sourceAgentId, knowledgeType, minConfidence, limit = 50, offset = 0 } = filters;
      
      let query = db.select().from(schema.agentLearningData);
      let countQuery = db.select({ count: count() }).from(schema.agentLearningData);
      
      const conditions = [];
      
      if (sourceAgentId) {
        conditions.push(eq(schema.agentLearningData.sourceAgentId, sourceAgentId));
      }
      
      if (knowledgeType) {
        conditions.push(eq(schema.agentLearningData.knowledgeType, knowledgeType));
      }
      
      if (minConfidence !== undefined) {
        conditions.push(gte(schema.agentLearningData.confidenceScore, minConfidence));
      }
      
      if (conditions.length > 0) {
        query = query.where(and(...conditions));
        countQuery = countQuery.where(and(...conditions));
      }
      
      const [data, [{ count: total }]] = await Promise.all([
        query.orderBy(desc(schema.agentLearningData.createdAt)).limit(limit).offset(offset),
        countQuery
      ]);
      
      return { data, total };
    } catch (error) {
      logger.error('Failed to list learning data', { error: error.message, filters });
      throw error;
    }
  }

  async deleteLearningData(id: string): Promise<boolean> {
    try {
      const result = await db.delete(schema.agentLearningData).where(eq(schema.agentLearningData.id, id));
      const deleted = result.rowCount > 0;
      
      if (deleted) {
        logger.info('Learning data deleted', { id });
      }
      
      return deleted;
    } catch (error) {
      logger.error('Failed to delete learning data', { error: error.message, id });
      throw error;
    }
  }

  // System health
  async insertHealthRecord(record: {
    component: string;
    status: string;
    details: any;
    responseTimeMs?: number;
  }): Promise<void> {
    try {
      await db.insert(schema.systemHealth).values(record);
    } catch (error) {
      logger.error('Failed to insert health record', { error: error.message, component: record.component });
      throw error;
    }
  }

  async getHealthHistory(component?: string, limit = 100): Promise<any[]> {
    try {
      let query = db.select().from(schema.systemHealth);
      
      if (component) {
        query = query.where(eq(schema.systemHealth.component, component));
      }
      
      const history = await query
        .orderBy(desc(schema.systemHealth.createdAt))
        .limit(limit);
      
      return history;
    } catch (error) {
      logger.error('Failed to get health history', { error: error.message, component });
      throw error;
    }
  }
}

// Create storage instance
export const storage = new PostgreSQLStorage();

// Initialize database connection
export async function initializeDatabase(): Promise<void> {
  try {
    const isConnected = await testConnection();
    if (!isConnected) {
      throw new Error('Failed to connect to database');
    }
    
    logger.info('Database initialized successfully');
  } catch (error) {
    logger.error('Database initialization failed', { error: error.message });
    throw error;
  }
}

// Cleanup function
export async function closeDatabase(): Promise<void> {
  try {
    await connection.end();
    logger.info('Database connection closed');
  } catch (error) {
    logger.error('Error closing database connection', { error: error.message });
  }
}

// Health check function
export async function databaseHealthCheck(): Promise<{
  status: 'healthy' | 'unhealthy';
  details: any;
}> {
  try {
    const start = Date.now();
    await connection`SELECT 1`;
    const responseTime = Date.now() - start;
    
    return {
      status: 'healthy',
      details: {
        responseTime,
        connected: true,
      },
    };
  } catch (error) {
    return {
      status: 'unhealthy',
      details: {
        error: error.message,
        connected: false,
      },
    };
  }
}

// Export for use in other modules
export { connection };
export default storage;
