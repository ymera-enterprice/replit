import express, { type Request, Response } from "express";
import multer from "multer";
import path from "path";
import { createServer } from "http";
import { WebSocketServer, WebSocket } from "ws";
import { z } from "zod";
import crypto from "crypto";
import fs from "fs/promises";
import { storage } from "./storage";
import { fileProcessor } from "./services/fileProcessor";
import { WebSocketManager } from "./services/websocketManager";
import { searchEngine } from "./services/searchEngine";
import { 
  insertFileSchema, 
  insertFileCommentSchema, 
  insertFileShareSchema,
  insertActivityLogSchema 
} from "@shared/schema";

const router = express.Router();

// Create HTTP server for WebSocket integration
const httpServer = createServer();

// Multer configuration for file uploads
const upload = multer({
  dest: process.env.UPLOAD_DIR || './uploads',
  limits: {
    fileSize: 100 * 1024 * 1024, // 100MB limit
  },
  fileFilter: (req, file, cb) => {
    // Allow all file types for now, but log them
    console.log(`Uploading file: ${file.originalname} (${file.mimetype})`);
    cb(null, true);
  },
});

// Mock authentication middleware (replace with real auth)
const authenticateUser = (req: Request, res: Response, next: any) => {
  // In a real app, this would validate JWT tokens
  req.user = {
    id: req.headers['x-user-id'] as string || 'user123',
    name: req.headers['x-user-name'] as string || 'John Doe',
  };
  next();
};

// Health check endpoint
router.get('/health', (req: Request, res: Response) => {
  res.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    memory: process.memoryUsage(),
  });
});

// File upload endpoint
router.post('/files/upload', upload.single('file'), async (req: Request, res: Response) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    const fileData = {
      name: `${Date.now()}_${req.file.originalname}`,
      originalName: req.file.originalname,
      mimeType: req.file.mimetype,
      size: req.file.size,
      path: req.file.path,
      checksum: await calculateFileChecksum(req.file.path),
      status: 'processing' as const,
      uploadedBy: req.user.id,
    };

    // Create file record
    const file = await storage.createFile(fileData);

    // Log upload activity
    await storage.createActivityLog({
      fileId: file.id,
      userId: req.user.id,
      userName: req.user.name,
      action: 'upload',
      details: { 
        originalName: req.file.originalname, 
        size: req.file.size,
        mimeType: req.file.mimetype 
      },
      ipAddress: req.ip,
      userAgent: req.headers['user-agent'],
    });

    // Start file processing asynchronously
    processFileAsync(file);

    res.json({ 
      message: 'File uploaded successfully', 
      file: {
        id: file.id,
        name: file.name,
        originalName: file.originalName,
        size: file.size,
        mimeType: file.mimeType,
        status: file.status,
        createdAt: file.createdAt,
      }
    });
  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({ error: 'Upload failed' });
  }
});

// Get user's files
router.get('/files', async (req: Request, res: Response) => {
  try {
    const limit = parseInt(req.query.limit as string) || 50;
    const offset = parseInt(req.query.offset as string) || 0;
    
    const files = await storage.getFiles(req.user.id, limit, offset);
    
    // Enhance files with metadata
    const enhancedFiles = await Promise.all(
      files.map(async (file) => {
        const metadata = await storage.getFileMetadata(file.id);
        const shares = await storage.getFileShares(file.id);
        const activeConnections = await storage.getActiveConnections(file.id);
        
        return {
          ...file,
          metadata: metadata ? {
            tags: metadata.tags,
            wordCount: metadata.wordCount,
            pageCount: metadata.pageCount,
            thumbnailPath: metadata.thumbnailPath,
          } : null,
          collaborators: activeConnections.length,
          shares: shares.length,
        };
      })
    );

    res.json({ files: enhancedFiles });
  } catch (error) {
    console.error('Get files error:', error);
    res.status(500).json({ error: 'Failed to fetch files' });
  }
});

// Get shared files
router.get('/files/shared', async (req: Request, res: Response) => {
  try {
    const files = await storage.getUserSharedFiles(req.user.id);
    res.json({ files });
  } catch (error) {
    console.error('Get shared files error:', error);
    res.status(500).json({ error: 'Failed to fetch shared files' });
  }
});

// Get file details
router.get('/files/:id', async (req: Request, res: Response) => {
  try {
    const file = await storage.getFile(req.params.id);
    if (!file) {
      return res.status(404).json({ error: 'File not found' });
    }

    // Check if user has access
    const hasAccess = file.uploadedBy === req.user.id || 
      (await storage.getFileShares(file.id)).some(share => share.sharedWith === req.user.id);

    if (!hasAccess) {
      return res.status(403).json({ error: 'Access denied' });
    }

    // Get additional data
    const metadata = await storage.getFileMetadata(file.id);
    const versions = await storage.getFileVersions(file.id);
    const shares = await storage.getFileShares(file.id);
    const comments = await storage.getFileComments(file.id);
    const activeConnections = await storage.getActiveConnections(file.id);

    // Log view activity
    await storage.createActivityLog({
      fileId: file.id,
      userId: req.user.id,
      userName: req.user.name,
      action: 'view',
      details: {},
      ipAddress: req.ip,
      userAgent: req.headers['user-agent'],
    });

    res.json({
      file,
      metadata,
      versions,
      shares,
      comments,
      activeCollaborators: activeConnections.map(conn => ({
        userId: conn.userId,
        userName: conn.userName,
        lastActivity: conn.lastActivity,
      })),
    });
  } catch (error) {
    console.error('Get file details error:', error);
    res.status(500).json({ error: 'Failed to fetch file details' });
  }
});

// Search files
router.get('/files/search', async (req: Request, res: Response) => {
  try {
    const query = req.query.q as string;
    if (!query) {
      return res.status(400).json({ error: 'Search query required' });
    }

    const fileTypes = req.query.types ? (req.query.types as string).split(',') : undefined;
    const tags = req.query.tags ? (req.query.tags as string).split(',') : undefined;
    const limit = parseInt(req.query.limit as string) || 50;
    const offset = parseInt(req.query.offset as string) || 0;

    const searchOptions = {
      userId: req.user.id,
      query,
      fileTypes,
      tags,
      limit,
      offset,
    };

    const results = await searchEngine.searchFiles(searchOptions);

    res.json({ results });
  } catch (error) {
    console.error('Search error:', error);
    res.status(500).json({ error: 'Search failed' });
  }
});

// Share file
router.post('/files/:id/share', async (req: Request, res: Response) => {
  try {
    const shareData = insertFileShareSchema.parse({
      ...req.body,
      fileId: req.params.id,
      sharedBy: req.user.id,
    });

    const file = await storage.getFile(req.params.id);
    if (!file || file.uploadedBy !== req.user.id) {
      return res.status(403).json({ error: 'Access denied' });
    }

    const share = await storage.createFileShare(shareData);

    // Log share activity
    await storage.createActivityLog({
      fileId: file.id,
      userId: req.user.id,
      userName: req.user.name,
      action: 'share',
      details: { 
        sharedWith: shareData.sharedWith, 
        permission: shareData.permission 
      },
      ipAddress: req.ip,
      userAgent: req.headers['user-agent'],
    });

    res.json({ message: 'File shared successfully', share });
  } catch (error) {
    console.error('Share file error:', error);
    res.status(500).json({ error: 'Failed to share file' });
  }
});

// Add comment
router.post('/files/:id/comments', async (req: Request, res: Response) => {
  try {
    const commentData = insertFileCommentSchema.parse({
      ...req.body,
      fileId: req.params.id,
      authorId: req.user.id,
      authorName: req.user.name,
    });

    const file = await storage.getFile(req.params.id);
    if (!file) {
      return res.status(404).json({ error: 'File not found' });
    }

    // Check access
    const hasAccess = file.uploadedBy === req.user.id || 
      (await storage.getFileShares(file.id)).some(share => share.sharedWith === req.user.id);

    if (!hasAccess) {
      return res.status(403).json({ error: 'Access denied' });
    }

    const comment = await storage.createFileComment(commentData);

    // Log comment activity
    await storage.createActivityLog({
      fileId: file.id,
      userId: req.user.id,
      userName: req.user.name,
      action: 'comment',
      details: { commentId: comment.id, content: req.body.content },
      ipAddress: req.ip,
      userAgent: req.headers['user-agent'],
    });

    // Notify via WebSocket
    await wsManager.notifyFileUpdate(file.id, {
      type: 'new_comment',
      comment,
    });

    res.json({ message: 'Comment added successfully', comment });
  } catch (error) {
    console.error('Add comment error:', error);
    res.status(500).json({ error: 'Failed to add comment' });
  }
});

// Get file processing status
router.get('/files/:id/processing', async (req: Request, res: Response) => {
  try {
    const logs = await storage.getProcessingLogs(req.params.id);
    res.json({ logs });
  } catch (error) {
    console.error('Get processing status error:', error);
    res.status(500).json({ error: 'Failed to fetch processing status' });
  }
});

// Get user activity
router.get('/activity', async (req: Request, res: Response) => {
  try {
    const limit = parseInt(req.query.limit as string) || 50;
    const activity = await storage.getUserActivity(req.user.id, limit);
    res.json({ activity });
  } catch (error) {
    console.error('Get activity error:', error);
    res.status(500).json({ error: 'Failed to fetch activity' });
  }
});

// Delete file
router.delete('/files/:id', async (req: Request, res: Response) => {
  try {
    const file = await storage.getFile(req.params.id);
    if (!file || file.uploadedBy !== req.user.id) {
      return res.status(403).json({ error: 'Access denied' });
    }

    // Delete file from filesystem
    try {
      await fs.unlink(file.path);
    } catch (fsError) {
      console.warn('Failed to delete file from filesystem:', fsError);
    }

    // Delete from database
    await storage.deleteFile(file.id);

    // Log delete activity
    await storage.createActivityLog({
      fileId: file.id,
      userId: req.user.id,
      userName: req.user.name,
      action: 'delete',
      details: { fileName: file.name },
      ipAddress: req.ip,
      userAgent: req.headers['user-agent'],
    });

    res.json({ message: 'File deleted successfully' });
  } catch (error) {
    console.error('Delete file error:', error);
    res.status(500).json({ error: 'Failed to delete file' });
  }
});

// WebSocket status endpoint
router.get('/websocket/status', (req: Request, res: Response) => {
  res.json({
    totalConnections: wsManager.getActiveConnections(),
    timestamp: new Date().toISOString(),
  });
});

// Helper functions
async function calculateFileChecksum(filePath: string): Promise<string> {
  const fileBuffer = await fs.readFile(filePath);
  return crypto.createHash('md5').update(fileBuffer).digest('hex');
}

async function processFileAsync(file: any) {
  try {
    console.log(`Starting async processing for file ${file.id}`);
    
    // Notify processing start via WebSocket
    await wsManager.notifyFileProcessing(file.id, {
      stage: 'processing',
      status: 'started',
    });

    const result = await fileProcessor.processFile(file);
    
    // Notify processing completion
    await wsManager.notifyFileProcessing(file.id, {
      stage: 'complete',
      status: 'completed',
      result,
    });

    console.log(`Completed processing for file ${file.id}`);
  } catch (error) {
    console.error(`Processing failed for file ${file.id}:`, error instanceof Error ? error.message : error);
    
    await wsManager.notifyFileProcessing(file.id, {
      stage: 'error',
      status: 'failed',
      error: error.message,
    });
  }
}

// Extend the Request interface
declare global {
  namespace Express {
    interface Request {
      user: {
        id: string;
        name: string;
      };
    }
  }
}

// Note: Routes are now registered via the registerRoutes function

// Function to setup routes and return server
export function registerRoutes(app: express.Application) {
  // Initialize WebSocket manager with the HTTP server
  const wsManager = new WebSocketManager(httpServer);
  
  // Use authentication middleware for all API routes
  app.use('/api', authenticateUser);
  app.use('/api', router);
  
  // Add route for serving uploaded files
  app.get('/uploads/:filename', authenticateUser, async (req: Request, res: Response) => {
    try {
      const filename = req.params.filename;
      const filePath = path.join(process.env.UPLOAD_DIR || './uploads', filename);
      
      // Find file in database to check access
      const files = await storage.getFiles(req.user.id, 1000);
      const file = files.find(f => f.path === filePath);
      
      if (!file) {
        // Check if user has access via shares
        const sharedFiles = await storage.getUserSharedFiles(req.user.id);
        const sharedFile = sharedFiles.find(f => f.path === filePath);
        
        if (!sharedFile) {
          return res.status(403).json({ error: 'Access denied' });
        }
      }

      res.sendFile(path.resolve(filePath));
    } catch (error) {
      console.error('File serve error:', error);
      res.status(500).json({ error: 'Failed to serve file' });
    }
  });

  // Add route for serving thumbnails
  app.get('/thumbnails/:filename', authenticateUser, async (req: Request, res: Response) => {
    try {
      const filename = req.params.filename;
      const thumbnailPath = path.join(process.env.THUMBNAIL_DIR || './thumbnails', filename);
      
      res.sendFile(path.resolve(thumbnailPath));
    } catch (error) {
      console.error('Thumbnail serve error:', error);
      res.status(404).json({ error: 'Thumbnail not found' });
    }
  });
  
  return httpServer;
}

export default httpServer;
