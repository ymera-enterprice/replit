import { eq, and, desc, lt, count, like, or } from "drizzle-orm";
import { db } from "./db";
import {
  users,
  userSessions,
  files,
  fileOperations,
  collaborationSessions,
  activityLogs,
  type User,
  type InsertUser,
  type UserSession,
  type File,
  type InsertFile,
  type FileOperation,
  type InsertFileOperation,
  type CollaborationSession,
  type InsertCollaborationSession,
  type ActivityLog,
  type InsertActivityLog,
} from "@shared/schema";

export interface IStorage {
  // User management
  getUser(id: string): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  getUserByEmail(email: string): Promise<User | undefined>;
  createUser(insertUser: InsertUser): Promise<User>;
  updateUser(id: string, updates: Partial<User>): Promise<User | undefined>;
  deleteUser(id: string): Promise<boolean>;

  // Session management
  createSession(session: Omit<UserSession, 'id' | 'createdAt'>): Promise<UserSession>;
  getSession(sessionToken: string): Promise<UserSession | undefined>;
  updateSession(sessionToken: string, updates: Partial<UserSession>): Promise<UserSession | undefined>;
  deleteSession(sessionToken: string): Promise<boolean>;
  deleteUserSessions(userId: string): Promise<number>;
  cleanupExpiredSessions(): Promise<number>;

  // File management
  getFile(id: string): Promise<File | undefined>;
  getFiles(userId?: string, limit?: number, offset?: number): Promise<File[]>;
  createFile(insertFile: InsertFile): Promise<File>;
  updateFile(id: string, updates: Partial<File>): Promise<File | undefined>;
  deleteFile(id: string): Promise<boolean>;
  searchFiles(query: string, userId?: string): Promise<File[]>;

  // File operations
  getFileOperations(fileId?: string, userId?: string, limit?: number): Promise<FileOperation[]>;
  createFileOperation(operation: InsertFileOperation): Promise<FileOperation>;
  updateFileOperation(id: string, updates: Partial<FileOperation>): Promise<FileOperation | undefined>;

  // Collaboration sessions
  getCollaborationSessions(fileId?: string, userId?: string): Promise<CollaborationSession[]>;
  createCollaborationSession(session: InsertCollaborationSession): Promise<CollaborationSession>;
  updateCollaborationSession(id: string, updates: Partial<CollaborationSession>): Promise<CollaborationSession | undefined>;

  // Activity logs
  getActivityLogs(userId?: string, limit?: number): Promise<ActivityLog[]>;
  createActivityLog(log: InsertActivityLog): Promise<ActivityLog>;

  // Dashboard stats
  getDashboardStats(): Promise<{
    totalFiles: number;
    activeUsers: number;
    totalOperations: number;
    systemStatus: string;
  }>;
}

export class DatabaseStorage implements IStorage {
  // User management
  async getUser(id: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.id, id)).limit(1);
    return user || undefined;
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.username, username)).limit(1);
    return user || undefined;
  }

  async getUserByEmail(email: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.email, email)).limit(1);
    return user || undefined;
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const [user] = await db.insert(users).values({
      ...insertUser,
      updatedAt: new Date(),
    }).returning();
    return user;
  }

  async updateUser(id: string, updates: Partial<User>): Promise<User | undefined> {
    const [user] = await db
      .update(users)
      .set({ ...updates, updatedAt: new Date() })
      .where(eq(users.id, id))
      .returning();
    return user || undefined;
  }

  async deleteUser(id: string): Promise<boolean> {
    const result = await db.delete(users).where(eq(users.id, id));
    return result.rowCount > 0;
  }

  // Session management
  async createSession(session: Omit<UserSession, 'id' | 'createdAt'>): Promise<UserSession> {
    const [newSession] = await db.insert(userSessions).values(session).returning();
    return newSession;
  }

  async getSession(sessionToken: string): Promise<UserSession | undefined> {
    const [session] = await db
      .select()
      .from(userSessions)
      .where(and(eq(userSessions.sessionToken, sessionToken), eq(userSessions.isActive, true)))
      .limit(1);
    return session || undefined;
  }

  async updateSession(sessionToken: string, updates: Partial<UserSession>): Promise<UserSession | undefined> {
    const [session] = await db
      .update(userSessions)
      .set(updates)
      .where(eq(userSessions.sessionToken, sessionToken))
      .returning();
    return session || undefined;
  }

  async deleteSession(sessionToken: string): Promise<boolean> {
    const result = await db
      .update(userSessions)
      .set({ isActive: false })
      .where(eq(userSessions.sessionToken, sessionToken));
    return result.rowCount > 0;
  }

  async deleteUserSessions(userId: string): Promise<number> {
    const result = await db
      .update(userSessions)
      .set({ isActive: false })
      .where(eq(userSessions.userId, userId));
    return result.rowCount;
  }

  async cleanupExpiredSessions(): Promise<number> {
    const now = new Date();
    const result = await db
      .update(userSessions)
      .set({ isActive: false })
      .where(and(eq(userSessions.isActive, true), lt(userSessions.expiresAt, now)));
    return result.rowCount;
  }

  // File management
  async getFile(id: string): Promise<File | undefined> {
    const [file] = await db.select().from(files).where(eq(files.id, id)).limit(1);
    return file || undefined;
  }

  async getFiles(userId?: string, limit = 50, offset = 0): Promise<File[]> {
    let query = db.select().from(files).orderBy(desc(files.createdAt)).limit(limit).offset(offset);
    
    if (userId) {
      query = query.where(eq(files.uploadedBy, userId));
    }
    
    return await query;
  }

  async createFile(insertFile: InsertFile): Promise<File> {
    const [file] = await db.insert(files).values({
      ...insertFile,
      updatedAt: new Date(),
    }).returning();
    return file;
  }

  async updateFile(id: string, updates: Partial<File>): Promise<File | undefined> {
    const [file] = await db
      .update(files)
      .set({ ...updates, updatedAt: new Date() })
      .where(eq(files.id, id))
      .returning();
    return file || undefined;
  }

  async deleteFile(id: string): Promise<boolean> {
    const result = await db.delete(files).where(eq(files.id, id));
    return result.rowCount > 0;
  }

  async searchFiles(query: string, userId?: string): Promise<File[]> {
    let searchQuery = db
      .select()
      .from(files)
      .where(
        or(
          like(files.filename, `%${query}%`),
          like(files.originalName, `%${query}%`)
        )
      )
      .orderBy(desc(files.createdAt))
      .limit(50);

    if (userId) {
      searchQuery = searchQuery.where(eq(files.uploadedBy, userId));
    }

    return await searchQuery;
  }

  // File operations
  async getFileOperations(fileId?: string, userId?: string, limit = 50): Promise<FileOperation[]> {
    let query = db.select().from(fileOperations).orderBy(desc(fileOperations.createdAt)).limit(limit);

    if (fileId) {
      query = query.where(eq(fileOperations.fileId, fileId));
    }

    if (userId) {
      query = query.where(eq(fileOperations.userId, userId));
    }

    return await query;
  }

  async createFileOperation(operation: InsertFileOperation): Promise<FileOperation> {
    const [newOperation] = await db.insert(fileOperations).values(operation).returning();
    return newOperation;
  }

  async updateFileOperation(id: string, updates: Partial<FileOperation>): Promise<FileOperation | undefined> {
    const [operation] = await db
      .update(fileOperations)
      .set(updates)
      .where(eq(fileOperations.id, id))
      .returning();
    return operation || undefined;
  }

  // Collaboration sessions
  async getCollaborationSessions(fileId?: string, userId?: string): Promise<CollaborationSession[]> {
    let query = db.select().from(collaborationSessions).orderBy(desc(collaborationSessions.createdAt));

    if (fileId) {
      query = query.where(eq(collaborationSessions.fileId, fileId));
    }

    if (userId) {
      query = query.where(eq(collaborationSessions.createdBy, userId));
    }

    return await query;
  }

  async createCollaborationSession(session: InsertCollaborationSession): Promise<CollaborationSession> {
    const [newSession] = await db.insert(collaborationSessions).values(session).returning();
    return newSession;
  }

  async updateCollaborationSession(id: string, updates: Partial<CollaborationSession>): Promise<CollaborationSession | undefined> {
    const [session] = await db
      .update(collaborationSessions)
      .set(updates)
      .where(eq(collaborationSessions.id, id))
      .returning();
    return session || undefined;
  }

  // Activity logs
  async getActivityLogs(userId?: string, limit = 50): Promise<ActivityLog[]> {
    let query = db.select().from(activityLogs).orderBy(desc(activityLogs.createdAt)).limit(limit);

    if (userId) {
      query = query.where(eq(activityLogs.userId, userId));
    }

    return await query;
  }

  async createActivityLog(log: InsertActivityLog): Promise<ActivityLog> {
    const [newLog] = await db.insert(activityLogs).values(log).returning();
    return newLog;
  }

  // Dashboard stats
  async getDashboardStats(): Promise<{
    totalFiles: number;
    activeUsers: number;
    totalOperations: number;
    systemStatus: string;
  }> {
    const [filesCount] = await db.select({ count: count() }).from(files);
    const [usersCount] = await db.select({ count: count() }).from(users).where(eq(users.userStatus, 'active'));
    const [operationsCount] = await db.select({ count: count() }).from(fileOperations);

    return {
      totalFiles: filesCount.count,
      activeUsers: usersCount.count,
      totalOperations: operationsCount.count,
      systemStatus: 'healthy'
    };
  }
}

export const storage = new DatabaseStorage();
