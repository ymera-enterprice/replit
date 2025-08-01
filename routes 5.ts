import { Router, Request, Response } from 'express';
import multer from 'multer';
import { z } from 'zod';
import { storage } from './storage';
import { 
  insertUserSchema, 
  insertAgentSchema, 
  insertTaskSchema, 
  insertFileSchema,
  insertKnowledgeItemSchema,
  insertLearningPatternSchema,
  insertMessageSchema,
  insertSystemMetricSchema,
  insertAuditLogSchema
} from '@shared/schema';
import { 
  authenticateToken, 
  optionalAuth, 
  AuthService,
  JWTManager 
} from './middleware/auth';
import { asyncHandler } from './middleware/errorHandler';
import { rateLimitMiddleware, rateLimitConfigs } from './middleware/rateLimit';
import { auditLog } from './middleware/logging';
import { FileManager } from './services/fileManager';
import { messageBroker } from './services/messageBroker';
import { taskDispatcher } from './services/taskDispatcher';
import { learningEngine } from './services/learningEngine';
import { monitoringService } from './services/monitoring';

const router = Router();
const upload = multer({ dest: 'temp/' });
const fileManager = new FileManager();

// ===============================================================================
// AUTHENTICATION ROUTES
// ===============================================================================

router.post('/auth/register', 
  rateLimitMiddleware(rateLimitConfigs.auth),
  asyncHandler(async (req: Request, res: Response) => {
    const validation = insertUserSchema.extend({
      password: z.string().min(8),
    }).safeParse(req.body);
    
    if (!validation.success) {
      return res.status(400).json({ 
        error: 'validation_error', 
        details: validation.error.errors 
      });
    }
    
    const { username, email, password, firstName, lastName } = validation.data;
    
    const { user, tokens } = await AuthService.register(
      username, 
      email, 
      password, 
      firstName, 
      lastName
    );
    
    auditLog('user_register', 'user', user.id, { username, email });
    
    res.status(201).json({ user, tokens });
  })
);

router.post('/auth/login',
  rateLimitMiddleware(rateLimitConfigs.auth),
  asyncHandler(async (req: Request, res: Response) => {
    const { username, password } = req.body;
    
    if (!username || !password) {
      return res.status(400).json({ 
        error: 'validation_error', 
        message: 'Username and password are required' 
      });
    }
    
    const { user, tokens } = await AuthService.login(
      username, 
      password, 
      req.ip, 
      req.headers['user-agent']
    );
    
    auditLog('user_login', 'user', user.id, { username, ip: req.ip });
    
    res.json({ user, tokens });
  })
);

router.post('/auth/refresh',
  asyncHandler(async (req: Request, res: Response) => {
    const { refreshToken } = req.body;
    
    if (!refreshToken) {
      return res.status(400).json({ 
        error: 'validation_error', 
        message: 'Refresh token is required' 
      });
    }
    
    const tokens = await AuthService.refreshToken(refreshToken);
    res.json({ tokens });
  })
);

router.post('/auth/logout',
  authenticateToken,
  asyncHandler(async (req: Request & { auth?: any }, res: Response) => {
    const sessionToken = req.auth?.session?.sessionToken;
    
    if (sessionToken) {
      await AuthService.logout(sessionToken);
      auditLog('user_logout', 'user', req.auth.user.id);
    }
    
    res.json({ message: 'Logged out successfully' });
  })
);

router.get('/auth/me',
  authenticateToken,
  asyncHandler(async (req: Request & { auth?: any }, res: Response) => {
    const user = await storage.getUserById(req.auth.user.id);
    res.json({ user });
  })
);

// ===============================================================================
// AGENT MANAGEMENT ROUTES
// ===============================================================================

router.get('/agents',
  authenticateToken,
  asyncHandler(async (req: Request, res: Response) => {
    const agents = await storage.getAllAgents();
    res.json({ agents });
  })
);

router.post('/agents',
  authenticateToken,
  rateLimitMiddleware(rateLimitConfigs.api),
  asyncHandler(async (req: Request & { auth?: any }, res: Response) => {
    const validation = insertAgentSchema.safeParse(req.body);
    
    if (!validation.success) {
      return res.status(400).json({ 
        error: 'validation_error', 
        details: validation.error.errors 
      });
    }
    
    const agent = await storage.createAgent(validation.data);
    
    auditLog('agent_created', 'agent', agent.id, { name: agent.name }, req.auth?.user?.id);
    
    res.status(201).json({ agent });
  })
);

router.get('/agents/:id',
  authenticateToken,
  asyncHandler(async (req: Request, res: Response) => {
    const agent = await storage.getAgentById(req.params.id);
    
    if (!agent) {
      return res.status(404).json({ error: 'Agent not found' });
    }
    
    res.json({ agent });
  })
);

router.put('/agents/:id',
  authenticateToken,
  asyncHandler(async (req: Request & { auth?: any }, res: Response) => {
    const updateData = req.body;
    const agent = await storage.updateAgent(req.params.id, updateData);
    
    if (!agent) {
      return res.status(404).json({ error: 'Agent not found' });
    }
    
    auditLog('agent_updated', 'agent', agent.id, updateData, req.auth?.user?.id);
    
    res.json({ agent });
  })
);

router.delete('/agents/:id',
  authenticateToken,
  asyncHandler(async (req: Request & { auth?: any }, res: Response) => {
    await storage.deleteAgent(req.params.id);
    
    auditLog('agent_deleted', 'agent', req.params.id, {}, req.auth?.user?.id);
    
    res.json({ message: 'Agent deleted successfully' });
  })
);

// ===============================================================================
// TASK MANAGEMENT ROUTES
// ===============================================================================

router.get('/tasks',
  authenticateToken,
  asyncHandler(async (req: Request, res: Response) => {
    const { status, agentId, limit = 50, offset = 0 } = req.query;
    
    const tasks = await storage.getTasks({
      status: status as string,
      agentId: agentId as string,
      limit: parseInt(limit as string),
      offset: parseInt(offset as string),
    });
    
    res.json({ tasks });
  })
);

router.post('/tasks',
  authenticateToken,
  rateLimitMiddleware(rateLimitConfigs.api),
  asyncHandler(async (req: Request & { auth?: any }, res: Response) => {
    const validation = insertTaskSchema.safeParse(req.body);
    
    if (!validation.success) {
      return res.status(400).json({ 
        error: 'validation_error', 
        details: validation.error.errors 
      });
    }
    
    const task = await taskDispatcher.createTask({
      name: validation.data.name,
      type: validation.data.type,
      payload: validation.data.payload,
      priority: validation.data.priority,
      maxRetries: validation.data.maxRetries,
      scheduledAt: validation.data.scheduledAt,
    });
    
    auditLog('task_created', 'task', task.id, { name: task.name, type: task.type }, req.auth?.user?.id);
    
    res.status(201).json({ task });
  })
);

router.get('/tasks/:id',
  authenticateToken,
  asyncHandler(async (req: Request, res: Response) => {
    const taskStatus = await taskDispatcher.getTaskStatus(req.params.id);
    res.json(taskStatus);
  })
);

router.post('/tasks/:id/cancel',
  authenticateToken,
  asyncHandler(async (req: Request & { auth?: any }, res: Response) => {
    await taskDispatcher.cancelTask(req.params.id);
    
    auditLog('task_cancelled', 'task', req.params.id, {}, req.auth?.user?.id);
    
    res.json({ message: 'Task cancelled successfully' });
  })
);

router.get('/tasks/stats',
  authenticateToken,
  asyncHandler(async (req: Request, res: Response) => {
    const stats = taskDispatcher.getStats();
    res.json({ stats });
  })
);

// ===============================================================================
// FILE MANAGEMENT ROUTES
// ===============================================================================

router.post('/files/upload',
  authenticateToken,
  rateLimitMiddleware(rateLimitConfigs.upload),
  upload.single('file'),
  asyncHandler(async (req: Request & { auth?: any }, res: Response) => {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }
    
    const metadata = {
      originalName: req.file.originalname,
      mimeType: req.file.mimetype,
      size: req.file.size,
      hash: '', // Will be calculated by FileManager
      uploadedBy: req.auth.user.id,
      tags: req.body.tags ? JSON.parse(req.body.tags) : [],
    };
    
    const options = {
      maxSize: 100 * 1024 * 1024, // 100MB
      virusScan: true,
      generateThumbnails: req.file.mimetype.startsWith('image/'),
    };
    
    const fileBuffer = await require('fs/promises').readFile(req.file.path);
    const file = await fileManager.uploadFile(fileBuffer, metadata, options);
    
    // Clean up temp file
    await require('fs/promises').unlink(req.file.path);
    
    auditLog('file_uploaded', 'file', file.id, { 
      filename: file.filename, 
      size: file.size 
    }, req.auth.user.id);
    
    res.status(201).json({ file });
  })
);

router.get('/files/:id/download',
  optionalAuth,
  asyncHandler(async (req: Request & { auth?: any }, res: Response) => {
    const fileStream = await fileManager.downloadFile(req.params.id, req.auth?.user?.id);
    const fileMetadata = await fileManager.getFileMetadata(req.params.id, req.auth?.user?.id);
    
    res.setHeader('Content-Disposition', `attachment; filename="${fileMetadata.originalName}"`);
    res.setHeader('Content-Type', fileMetadata.mimeType);
    res.setHeader('Content-Length', fileMetadata.size);
    
    fileStream.pipe(res);
  })
);

router.get('/files/:id',
  optionalAuth,
  asyncHandler(async (req: Request & { auth?: any }, res: Response) => {
    const file = await fileManager.getFileMetadata(req.params.id, req.auth?.user?.id);
    res.json({ file });
  })
);

router.delete('/files/:id',
  authenticateToken,
  asyncHandler(async (req: Request & { auth?: any }, res: Response) => {
    await fileManager.deleteFile(req.params.id, req.auth.user.id);
    
    auditLog('file_deleted', 'file', req.params.id, {}, req.auth.user.id);
    
    res.json({ message: 'File deleted successfully' });
  })
);

router.get('/files',
  authenticateToken,
  asyncHandler(async (req: Request, res: Response) => {
    const { tags, mimeType, filename, limit = 50, offset = 0 } = req.query;
    
    const result = await fileManager.searchFiles({
      tags: tags ? (tags as string).split(',') : undefined,
      mimeType: mimeType as string,
      filename: filename as string,
      limit: parseInt(limit as string),
      offset: parseInt(offset as string),
    });
    
    res.json(result);
  })
);

// ===============================================================================
// LEARNING ENGINE ROUTES
// ===============================================================================

router.post('/learning/experiences',
  authenticateToken,
  asyncHandler(async (req: Request & { auth?: any }, res: Response) => {
    const experience = await learningEngine.captureExperience({
      type: req.body.type,
      context: req.body.context,
      action: req.body.action,
      outcome: req.body.outcome,
      success: req.body.success,
      agentId: req.body.agentId,
      metadata: req.body.metadata,
    });
    
    res.status(201).json({ experience });
  })
);

router.get('/learning/patterns',
  authenticateToken,
  asyncHandler(async (req: Request, res: Response) => {
    const { type } = req.query;
    const patterns = await learningEngine.getPatterns(type as any);
    res.json({ patterns });
  })
);

router.post('/learning/knowledge',
  authenticateToken,
  asyncHandler(async (req: Request & { auth?: any }, res: Response) => {
    const validation = insertKnowledgeItemSchema.safeParse(req.body);
    
    if (!validation.success) {
      return res.status(400).json({ 
        error: 'validation_error', 
        details: validation.error.errors 
      });
    }
    
    const knowledge = await learningEngine.createKnowledge(validation.data);
    
    auditLog('knowledge_created', 'knowledge', knowledge.id, { type: knowledge.type }, req.auth.user.id);
    
    res.status(201).json({ knowledge });
  })
);

router.post('/learning/knowledge/:id/validate',
  authenticateToken,
  asyncHandler(async (req: Request & { auth?: any }, res: Response) => {
    await learningEngine.validateKnowledge(req.params.id);
    
    auditLog('knowledge_validated', 'knowledge', req.params.id, {}, req.auth.user.id);
    
    res.json({ message: 'Knowledge validation completed' });
  })
);

router.get('/learning/suggestions',
  authenticateToken,
  asyncHandler(async (req: Request, res: Response) => {
    const { agentId } = req.query;
    const suggestions = await learningEngine.getOptimizationSuggestions(agentId as string);
    res.json({ suggestions });
  })
);

router.get('/learning/metrics',
  authenticateToken,
  asyncHandler(async (req: Request, res: Response) => {
    const metrics = await learningEngine.getMetrics();
    res.json({ metrics });
  })
);

router.get('/learning/knowledge-graph',
  authenticateToken,
  asyncHandler(async (req: Request, res: Response) => {
    const graph = learningEngine.getKnowledgeGraph();
    res.json({ graph });
  })
);

// ===============================================================================
// MESSAGE BROKER ROUTES
// ===============================================================================

router.post('/messaging/publish',
  authenticateToken,
  rateLimitMiddleware(rateLimitConfigs.api),
  asyncHandler(async (req: Request, res: Response) => {
    const { queue, payload, options } = req.body;
    
    const messageId = await messageBroker.publish(queue, payload, options);
    res.status(201).json({ messageId });
  })
);

router.get('/messaging/queues/stats',
  authenticateToken,
  asyncHandler(async (req: Request, res: Response) => {
    const { queue } = req.query;
    const stats = await messageBroker.getQueueStats(queue as string);
    res.json({ stats });
  })
);

router.post('/messaging/queues/:name/purge',
  authenticateToken,
  asyncHandler(async (req: Request & { auth?: any }, res: Response) => {
    const purgedCount = await messageBroker.purgeQueue(req.params.name);
    
    auditLog('queue_purged', 'queue', req.params.name, { count: purgedCount }, req.auth.user.id);
    
    res.json({ purgedCount });
  })
);

router.post('/messaging/dead-letters/reprocess',
  authenticateToken,
  asyncHandler(async (req: Request & { auth?: any }, res: Response) => {
    const { fromQueue, toQueue } = req.body;
    const reprocessedCount = await messageBroker.reprocessDeadLetters(fromQueue, toQueue);
    
    auditLog('dead_letters_reprocessed', 'queue', fromQueue, { 
      toQueue, 
      count: reprocessedCount 
    }, req.auth.user.id);
    
    res.json({ reprocessedCount });
  })
);

// ===============================================================================
// MONITORING ROUTES
// ===============================================================================

router.get('/monitoring/health',
  asyncHandler(async (req: Request, res: Response) => {
    const health = await monitoringService.getSystemHealth();
    res.json({ health });
  })
);

router.get('/monitoring/metrics',
  authenticateToken,
  asyncHandler(async (req: Request, res: Response) => {
    const metrics = await monitoringService.getSystemMetrics();
    res.json({ metrics });
  })
);

router.get('/monitoring/alerts',
  authenticateToken,
  asyncHandler(async (req: Request, res: Response) => {
    const { type, acknowledged } = req.query;
    const alerts = monitoringService.getAlerts(
      type as any,
      acknowledged ? acknowledged === 'true' : undefined
    );
    res.json({ alerts });
  })
);

router.post('/monitoring/alerts/:id/acknowledge',
  authenticateToken,
  asyncHandler(async (req: Request & { auth?: any }, res: Response) => {
    await monitoringService.acknowledgeAlert(req.params.id);
    
    auditLog('alert_acknowledged', 'alert', req.params.id, {}, req.auth.user.id);
    
    res.json({ message: 'Alert acknowledged' });
  })
);

router.get('/monitoring/metrics/:name/history',
  authenticateToken,
  asyncHandler(async (req: Request, res: Response) => {
    const { hours = 24 } = req.query;
    const history = await monitoringService.getMetricHistory(
      req.params.name,
      parseInt(hours as string)
    );
    res.json({ history });
  })
);

router.get('/monitoring/service-stats',
  authenticateToken,
  asyncHandler(async (req: Request, res: Response) => {
    const stats = await monitoringService.getServiceStats();
    res.json({ stats });
  })
);

// ===============================================================================
// SYSTEM ADMINISTRATION ROUTES
// ===============================================================================

router.get('/admin/system-status',
  authenticateToken,
  asyncHandler(async (req: Request, res: Response) => {
    const health = await monitoringService.getSystemHealth();
    const metrics = await monitoringService.getSystemMetrics();
    const taskStats = taskDispatcher.getStats();
    const learningMetrics = await learningEngine.getMetrics();
    const queueStats = await messageBroker.getQueueStats();
    const fileStats = await fileManager.getStorageStats();
    
    res.json({
      system: {
        health: health.overall,
        uptime: health.uptime,
        timestamp: health.timestamp,
      },
      performance: metrics,
      services: {
        tasks: taskStats,
        learning: learningMetrics,
        messaging: queueStats,
        storage: fileStats,
      },
    });
  })
);

router.get('/admin/audit-logs',
  authenticateToken,
  asyncHandler(async (req: Request, res: Response) => {
    const { limit = 100, offset = 0, action, resource } = req.query;
    
    const logs = await storage.getAuditLogs({
      limit: parseInt(limit as string),
      offset: parseInt(offset as string),
      action: action as string,
      resource: resource as string,
    });
    
    res.json({ logs });
  })
);

// ===============================================================================
// WEBSOCKET ENDPOINTS (for real-time updates)
// ===============================================================================

router.get('/ws/metrics',
  authenticateToken,
  asyncHandler(async (req: Request, res: Response) => {
    // WebSocket endpoint info
    res.json({
      endpoint: '/ws/metrics',
      description: 'Real-time system metrics updates',
      authentication: 'JWT token required',
    });
  })
);

export { router };
