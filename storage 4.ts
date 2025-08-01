import { eq, desc, like, and, or, sql } from "drizzle-orm";
import { db } from "./db";
import {
  files, fileMetadata, processingLogs, fileVersions, fileShares, fileComments,
  searchIndex, activityLogs, websocketConnections,
  type File, type NewFile, type FileMetadata, type NewFileMetadata,
  type ProcessingLog, type NewProcessingLog, type FileVersion, type NewFileVersion,
  type FileShare, type NewFileShare, type FileComment, type NewFileComment,
  type SearchIndex, type NewSearchIndex, type ActivityLog, type NewActivityLog,
  type WebsocketConnection, type NewWebsocketConnection
} from "@shared/schema";

export interface IStorage {
  // File operations
  createFile(file: NewFile): Promise<File>;
  getFile(id: string): Promise<File | undefined>;
  getFiles(userId: string, limit?: number, offset?: number): Promise<File[]>;
  updateFileStatus(id: string, status: string): Promise<void>;
  deleteFile(id: string): Promise<void>;
  
  // File metadata operations
  createFileMetadata(metadata: NewFileMetadata): Promise<FileMetadata>;
  getFileMetadata(fileId: string): Promise<FileMetadata | undefined>;
  updateFileMetadata(fileId: string, metadata: Partial<NewFileMetadata>): Promise<void>;
  
  // Processing logs
  createProcessingLog(log: NewProcessingLog): Promise<ProcessingLog>;
  getProcessingLogs(fileId: string): Promise<ProcessingLog[]>;
  
  // File versions
  createFileVersion(version: NewFileVersion): Promise<FileVersion>;
  getFileVersions(fileId: string): Promise<FileVersion[]>;
  
  // File sharing
  createFileShare(share: NewFileShare): Promise<FileShare>;
  getFileShares(fileId: string): Promise<FileShare[]>;
  getUserSharedFiles(userId: string): Promise<File[]>;
  
  // Comments
  createFileComment(comment: NewFileComment): Promise<FileComment>;
  getFileComments(fileId: string): Promise<FileComment[]>;
  
  // Search
  createSearchIndex(index: NewSearchIndex): Promise<SearchIndex>;
  searchFiles(query: string, userId: string): Promise<File[]>;
  
  // Activity logs
  createActivityLog(log: NewActivityLog): Promise<ActivityLog>;
  getUserActivity(userId: string, limit?: number): Promise<ActivityLog[]>;
  
  // WebSocket connections
  createWebsocketConnection(connection: NewWebsocketConnection): Promise<WebsocketConnection>;
  getActiveConnections(fileId?: string): Promise<WebsocketConnection[]>;
  removeWebsocketConnection(connectionId: string): Promise<void>;
  updateConnectionActivity(connectionId: string): Promise<void>;
}

export class DatabaseStorage implements IStorage {
  // File operations
  async createFile(file: NewFile): Promise<File> {
    const [newFile] = await db.insert(files).values(file).returning();
    return newFile;
  }

  async getFile(id: string): Promise<File | undefined> {
    const [file] = await db.select().from(files).where(eq(files.id, id));
    return file;
  }

  async getFiles(userId: string, limit = 50, offset = 0): Promise<File[]> {
    return await db
      .select()
      .from(files)
      .where(eq(files.uploadedBy, userId))
      .orderBy(desc(files.updatedAt))
      .limit(limit)
      .offset(offset);
  }

  async updateFileStatus(id: string, status: string): Promise<void> {
    await db
      .update(files)
      .set({ status: status as any, updatedAt: new Date() })
      .where(eq(files.id, id));
  }

  async deleteFile(id: string): Promise<void> {
    await db.delete(files).where(eq(files.id, id));
  }

  // File metadata operations
  async createFileMetadata(metadata: NewFileMetadata): Promise<FileMetadata> {
    const [newMetadata] = await db.insert(fileMetadata).values(metadata).returning();
    return newMetadata;
  }

  async getFileMetadata(fileId: string): Promise<FileMetadata | undefined> {
    const [metadata] = await db
      .select()
      .from(fileMetadata)
      .where(eq(fileMetadata.fileId, fileId));
    return metadata;
  }

  async updateFileMetadata(fileId: string, metadata: Partial<NewFileMetadata>): Promise<void> {
    await db
      .update(fileMetadata)
      .set(metadata)
      .where(eq(fileMetadata.fileId, fileId));
  }

  // Processing logs
  async createProcessingLog(log: NewProcessingLog): Promise<ProcessingLog> {
    const [newLog] = await db.insert(processingLogs).values(log).returning();
    return newLog;
  }

  async getProcessingLogs(fileId: string): Promise<ProcessingLog[]> {
    return await db
      .select()
      .from(processingLogs)
      .where(eq(processingLogs.fileId, fileId))
      .orderBy(desc(processingLogs.createdAt));
  }

  // File versions
  async createFileVersion(version: NewFileVersion): Promise<FileVersion> {
    const [newVersion] = await db.insert(fileVersions).values(version).returning();
    return newVersion;
  }

  async getFileVersions(fileId: string): Promise<FileVersion[]> {
    return await db
      .select()
      .from(fileVersions)
      .where(eq(fileVersions.fileId, fileId))
      .orderBy(desc(fileVersions.version));
  }

  // File sharing
  async createFileShare(share: NewFileShare): Promise<FileShare> {
    const [newShare] = await db.insert(fileShares).values(share).returning();
    return newShare;
  }

  async getFileShares(fileId: string): Promise<FileShare[]> {
    return await db
      .select()
      .from(fileShares)
      .where(eq(fileShares.fileId, fileId));
  }

  async getUserSharedFiles(userId: string): Promise<File[]> {
    return await db
      .select({
        id: files.id,
        name: files.name,
        originalName: files.originalName,
        mimeType: files.mimeType,
        size: files.size,
        path: files.path,
        checksum: files.checksum,
        status: files.status,
        uploadedBy: files.uploadedBy,
        createdAt: files.createdAt,
        updatedAt: files.updatedAt,
      })
      .from(files)
      .innerJoin(fileShares, eq(files.id, fileShares.fileId))
      .where(eq(fileShares.sharedWith, userId))
      .orderBy(desc(files.updatedAt));
  }

  // Comments
  async createFileComment(comment: NewFileComment): Promise<FileComment> {
    const [newComment] = await db.insert(fileComments).values(comment).returning();
    return newComment;
  }

  async getFileComments(fileId: string): Promise<FileComment[]> {
    return await db
      .select()
      .from(fileComments)
      .where(eq(fileComments.fileId, fileId))
      .orderBy(desc(fileComments.createdAt));
  }

  // Search
  async createSearchIndex(index: NewSearchIndex): Promise<SearchIndex> {
    const [newIndex] = await db.insert(searchIndex).values(index).returning();
    return newIndex;
  }

  async searchFiles(query: string, userId: string): Promise<File[]> {
    const searchTerm = `%${query}%`;
    
    return await db
      .select({
        id: files.id,
        name: files.name,
        originalName: files.originalName,
        mimeType: files.mimeType,
        size: files.size,
        path: files.path,
        checksum: files.checksum,
        status: files.status,
        uploadedBy: files.uploadedBy,
        createdAt: files.createdAt,
        updatedAt: files.updatedAt,
      })
      .from(files)
      .leftJoin(searchIndex, eq(files.id, searchIndex.fileId))
      .leftJoin(fileShares, eq(files.id, fileShares.fileId))
      .where(
        and(
          or(
            eq(files.uploadedBy, userId),
            eq(fileShares.sharedWith, userId)
          ),
          or(
            like(files.name, searchTerm),
            like(files.originalName, searchTerm),
            like(searchIndex.content, searchTerm)
          )
        )
      )
      .orderBy(desc(searchIndex.rankingScore))
      .limit(50);
  }

  // Activity logs
  async createActivityLog(log: NewActivityLog): Promise<ActivityLog> {
    const [newLog] = await db.insert(activityLogs).values(log).returning();
    return newLog;
  }

  async getUserActivity(userId: string, limit = 100): Promise<ActivityLog[]> {
    return await db
      .select()
      .from(activityLogs)
      .where(eq(activityLogs.userId, userId))
      .orderBy(desc(activityLogs.createdAt))
      .limit(limit);
  }

  // WebSocket connections
  async createWebsocketConnection(connection: NewWebsocketConnection): Promise<WebsocketConnection> {
    const [newConnection] = await db.insert(websocketConnections).values(connection).returning();
    return newConnection;
  }

  async getActiveConnections(fileId?: string): Promise<WebsocketConnection[]> {
    const query = db
      .select()
      .from(websocketConnections)
      .where(
        sql`${websocketConnections.lastActivity} > NOW() - INTERVAL '5 minutes'`
      );

    if (fileId) {
      return await query
        .where(eq(websocketConnections.fileId, fileId))
        .execute();
    }

    return await query.execute();
  }

  async removeWebsocketConnection(connectionId: string): Promise<void> {
    await db
      .delete(websocketConnections)
      .where(eq(websocketConnections.connectionId, connectionId));
  }

  async updateConnectionActivity(connectionId: string): Promise<void> {
    await db
      .update(websocketConnections)
      .set({ lastActivity: new Date() })
      .where(eq(websocketConnections.connectionId, connectionId));
  }
}

export const storage = new DatabaseStorage();
