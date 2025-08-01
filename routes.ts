import type { Express, Request, Response } from "express";
import { createServer, type Server } from "http";
import { z } from 'zod';
import { storage } from "./storage";
import { 
  AuthService, 
  authenticateToken, 
  optionalAuth,
  AuthContext
} from './auth';
import { validateRequest, validator } from './validation';
import { createLogger } from './logger';
import { rateLimiter, appCache } from './cache';
import { createRateLimiter } from './middleware';
import { healthCheck, readinessCheck, livenessCheck, metricsEndpoint } from './health';
import { wsManager, sendUserNotification, broadcastSystemMessage } from './websocket';
import {
  loginSchema,
  registerSchema,
  insertAgentSchema,
  insertLearningDataSchema,
  UserStatus,
  ActivityType,
  AgentStatus
} from '@shared/schema';

const logger = createLogger('routes');

// Extend Request type to include auth context
interface AuthenticatedRequest extends Request {
  auth?: AuthContext;
  validatedBody?: any;
}

export async function registerRoutes(app: Express): Promise<Server> {
  // Health check endpoints (no auth required)
  app.get('/health', healthCheck);
  app.get('/health/ready', readinessCheck);
  app.get('/health/live', livenessCheck);
  app.get('/metrics', metricsEndpoint);

  // API prefix for all routes
  const apiRouter = app;
  
  // Rate limiting for auth endpoints
  const authLimiter = createRateLimiter(15 * 60 * 1000, 10); // 10 requests per 15 minutes
  const generalLimiter = createRateLimiter(60 * 1000, 100); // 100 requests per minute
  
  // Apply general rate limiting to all API routes
  apiRouter.use('/api', generalLimiter);

  // ============================================================================
  // AUTHENTICATION ROUTES
  // ============================================================================

  // User registration
  apiRouter.post('/api/auth/register', 
    authLimiter,
    validateRequest(registerSchema),
    async (req: AuthenticatedRequest, res: Response) => {
      try {
        const { username, email, password, firstName, lastName } = req.validatedBody;

        const result = await AuthService.register(
          username,
          email,
          password,
          firstName,
          lastName
        );

        // Log user activity
        await storage.insertActivity({
          userId: result.user.id,
          activityType: ActivityType.LOGIN,
          activityDetails: { 
            method: 'register',
            ipAddress: req.ip,
            userAgent: req.get('User-Agent')
          },
          ipAddress: req.ip,
          userAgent: req.get('User-Agent'),
          success: true
        });

        // Send welcome notification
        if (wsManager) {
          sendUserNotification(result.user.id, 'Welcome to YMERA Platform!', 'info');
        }

        logger.info('User registered successfully', { 
          userId: result.user.id, 
          username, 
          email 
        });

        res.status(201).json({
          message: 'Registration successful',
          user: {
            id: result.user.id,
            username: result.user.username,
            email: result.user.email,
            displayName: result.user.displayName,
            userStatus: result.user.userStatus
          },
          tokens: result.tokens
        });

      } catch (error) {
        logger.error('Registration failed', { error: error.message });
        
        if (error.message.includes('already exists') || error.message.includes('already taken')) {
          res.status(409).json({ error: error.message });
        } else {
          res.status(400).json({ error: 'Registration failed' });
        }
      }
    }
  );

  // User login
  apiRouter.post('/api/auth/login',
    authLimiter,
    validateRequest(loginSchema),
    async (req: AuthenticatedRequest, res: Response) => {
      try {
        const { username, password } = req.validatedBody;

        const result = await AuthService.login(
          username,
          password,
          req.ip,
          req.get('User-Agent')
        );

        // Log successful login
        await storage.insertActivity({
          userId: result.user.id,
          activityType: ActivityType.LOGIN,
          activityDetails: { 
            method: 'login',
            ipAddress: req.ip,
            userAgent: req.get('User-Agent')
          },
          ipAddress: req.ip,
          userAgent: req.get('User-Agent'),
          success: true
        });

        logger.info('User logged in successfully', { 
          userId: result.user.id, 
          username: result.user.username 
        });

        res.json({
          message: 'Login successful',
          user: {
            id: result.user.id,
            username: result.user.username,
            email: result.user.email,
            displayName: result.user.displayName,
            userStatus: result.user.userStatus
          },
          tokens: result.tokens
        });

      } catch (error) {
        logger.warn('Login failed', { 
          username: req.validatedBody?.username, 
          error: error.message,
          ip: req.ip 
        });

        // Log failed login attempt
        if (req.validatedBody?.username) {
          try {
            const user = await storage.getUserByUsername(req.validatedBody.username) ||
                          await storage.getUserByEmail(req.validatedBody.username);
            
            if (user) {
              await storage.insertActivity({
                userId: user.id,
                activityType: ActivityType.LOGIN,
                activityDetails: { 
                  method: 'login',
                  ipAddress: req.ip,
                  userAgent: req.get('User-Agent'),
                  error: error.message
                },
                ipAddress: req.ip,
                userAgent: req.get('User-Agent'),
                success: false,
                errorMessage: error.message
              });
            }
          } catch (logError) {
            logger.error('Failed to log failed login attempt', { error: logError.message });
          }
        }

        res.status(401).json({ error: error.message });
      }
    }
  );

  // Logout
  apiRouter.post('/api/auth/logout',
    authenticateToken,
    async (req: AuthenticatedRequest, res: Response) => {
      try {
        if (req.auth?.session) {
          await AuthService.logout(req.auth.session.sessionToken);
          
          // Log logout activity
          await storage.insertActivity({
            userId: req.auth.user.id,
            sessionId: req.auth.session.id,
            activityType: ActivityType.LOGOUT,
            activityDetails: { 
              ipAddress: req.ip,
              userAgent: req.get('User-Agent')
            },
            ipAddress: req.ip,
            userAgent: req.get('User-Agent'),
            success: true
          });

          logger.info('User logged out', { userId: req.auth.user.id });
        }

        res.json({ message: 'Logout successful' });
      } catch (error) {
        logger.error('Logout failed', { error: error.message });
        res.status(500).json({ error: 'Logout failed' });
      }
    }
  );

  // Refresh token
  apiRouter.post('/api/auth/refresh',
    validateRequest(z.object({ refreshToken: z.string() })),
    async (req: AuthenticatedRequest, res: Response) => {
      try {
        const { refreshToken } = req.validatedBody;
        const tokens = await AuthService.refreshToken(refreshToken);

        res.json({
          message: 'Token refreshed successfully',
          tokens
        });
      } catch (error) {
        logger.warn('Token refresh failed', { error: error.message });
        res.status(401).json({ error: error.message });
      }
    }
  );

  // Get current user profile
  apiRouter.get('/api/auth/me',
    authenticateToken,
    async (req: AuthenticatedRequest, res: Response) => {
      try {
        const user = await storage.getUserById(req.auth!.user.id);
        if (!user) {
          return res.status(404).json({ error: 'User not found' });
        }

        res.json({
          user: {
            id: user.id,
            username: user.username,
            email: user.email,
            firstName: user.firstName,
            lastName: user.lastName,
            displayName: user.displayName,
            avatarUrl: user.avatarUrl,
            timezone: user.timezone,
            language: user.language,
            userStatus: user.userStatus,
            isEmailVerified: user.isEmailVerified,
            isMfaEnabled: user.isMfaEnabled,
            lastLoginAt: user.lastLoginAt,
            createdAt: user.createdAt
          }
        });
      } catch (error) {
        logger.error('Failed to get user profile', { error: error.message });
        res.status(500).json({ error: 'Failed to get user profile' });
      }
    }
  );

  // ============================================================================
  // USER MANAGEMENT ROUTES
  // ============================================================================

  // Get user activities
  apiRouter.get('/api/users/activities',
    authenticateToken,
    async (req: AuthenticatedRequest, res: Response) => {
      try {
        const { limit = 50, offset = 0, type, startDate, endDate } = req.query;
        
        const result = await storage.getUserActivities(req.auth!.user.id, {
          limit: parseInt(limit as string),
          offset: parseInt(offset as string),
          type: type as string,
          startDate: startDate ? new Date(startDate as string) : undefined,
          endDate: endDate ? new Date(endDate as string) : undefined
        });

        res.json(result);
      } catch (error) {
        logger.error('Failed to get user activities', { error: error.message });
        res.status(500).json({ error: 'Failed to get user activities' });
      }
    }
  );

  // Update user profile
  apiRouter.patch('/api/users/profile',
    authenticateToken,
    validateRequest(z.object({
      firstName: z.string().optional(),
      lastName: z.string().optional(),
      displayName: z.string().optional(),
      timezone: z.string().optional(),
      language: z.string().optional()
    })),
    async (req: AuthenticatedRequest, res: Response) => {
      try {
        const updates = req.validatedBody;
        const updatedUser = await storage.updateUser(req.auth!.user.id, updates);

        if (!updatedUser) {
          return res.status(404).json({ error: 'User not found' });
        }

        // Log profile update
        await storage.insertActivity({
          userId: req.auth!.user.id,
          sessionId: req.auth!.session.id,
          activityType: ActivityType.PROFILE_UPDATE,
          activityDetails: { updates },
          ipAddress: req.ip,
          userAgent: req.get('User-Agent'),
          success: true
        });

        logger.info('User profile updated', { userId: req.auth!.user.id });

        res.json({
          message: 'Profile updated successfully',
          user: {
            id: updatedUser.id,
            username: updatedUser.username,
            email: updatedUser.email,
            firstName: updatedUser.firstName,
            lastName: updatedUser.lastName,
            displayName: updatedUser.displayName,
            timezone: updatedUser.timezone,
            language: updatedUser.language
          }
        });
      } catch (error) {
        logger.error('Failed to update user profile', { error: error.message });
        res.status(500).json({ error: 'Failed to update profile' });
      }
    }
  );

  // ============================================================================
  // AGENT MANAGEMENT ROUTES
  // ============================================================================

  // List agents
  apiRouter.get('/api/agents',
    authenticateToken,
    async (req: AuthenticatedRequest, res: Response) => {
      try {
        const { 
          status, 
          type, 
          enabled, 
          limit = 50, 
          offset = 0, 
          search 
        } = req.query;

        const result = await storage.listAgents({
          status: status as string,
          type: type as string,
          enabled: enabled ? enabled === 'true' : undefined,
          limit: parseInt(limit as string),
          offset: parseInt(offset as string),
          search: search as string
        });

        res.json(result);
      } catch (error) {
        logger.error('Failed to list agents', { error: error.message });
        res.status(500).json({ error: 'Failed to list agents' });
      }
    }
  );

  // Get single agent
  apiRouter.get('/api/agents/:id',
    authenticateToken,
    async (req: AuthenticatedRequest, res: Response) => {
      try {
        const { id } = req.params;
        const agent = await storage.getAgentById(id);

        if (!agent) {
          return res.status(404).json({ error: 'Agent not found' });
        }

        // Get cached agent data if available
        const cachedData = await appCache.getAgentData(agent.agentId);

        res.json({
          ...agent,
          cachedData
        });
      } catch (error) {
        logger.error('Failed to get agent', { error: error.message });
        res.status(500).json({ error: 'Failed to get agent' });
      }
    }
  );

  // Create agent
  apiRouter.post('/api/agents',
    authenticateToken,
    validateRequest(insertAgentSchema),
    async (req: AuthenticatedRequest, res: Response) => {
      try {
        const agentData = {
          ...req.validatedBody,
          createdBy: req.auth!.user.id,
          status: AgentStatus.INACTIVE
        };

        const agent = await storage.insertAgent(agentData);

        // Cache agent data
        await appCache.cacheAgentData(agent.agentId, {
          status: agent.status,
          lastHeartbeat: agent.lastHeartbeat,
          configuration: agent.configuration
        });

        // Log agent creation
        await storage.insertActivity({
          userId: req.auth!.user.id,
          sessionId: req.auth!.session.id,
          activityType: ActivityType.AGENT_CREATE,
          activityDetails: { 
            agentId: agent.agentId,
            agentType: agent.agentType 
          },
          resourceType: 'agent',
          resourceId: agent.id,
          ipAddress: req.ip,
          userAgent: req.get('User-Agent'),
          success: true
        });

        // Send WebSocket notification
        if (wsManager) {
          wsManager.sendAgentUpdate(agent.agentId, agent.status, {
            action: 'created',
            agent: agent
          });
        }

        logger.info('Agent created', { 
          agentId: agent.agentId, 
          createdBy: req.auth!.user.id 
        });

        res.status(201).json(agent);
      } catch (error) {
        logger.error('Failed to create agent', { error: error.message });
        res.status(400).json({ error: 'Failed to create agent' });
      }
    }
  );

  // Update agent
  apiRouter.patch('/api/agents/:id',
    authenticateToken,
    validateRequest(insertAgentSchema.partial()),
    async (req: AuthenticatedRequest, res: Response) => {
      try {
        const { id } = req.params;
        const updates = req.validatedBody;

        const agent = await storage.updateAgent(id, updates);
        if (!agent) {
          return res.status(404).json({ error: 'Agent not found' });
        }

        // Update cache
        await appCache.cacheAgentData(agent.agentId, {
          status: agent.status,
          lastHeartbeat: agent.lastHeartbeat,
          configuration: agent.configuration
        });

        // Log agent update
        await storage.insertActivity({
          userId: req.auth!.user.id,
          sessionId: req.auth!.session.id,
          activityType: ActivityType.AGENT_UPDATE,
          activityDetails: { 
            agentId: agent.agentId,
            updates 
          },
          resourceType: 'agent',
          resourceId: agent.id,
          ipAddress: req.ip,
          userAgent: req.get('User-Agent'),
          success: true
        });

        // Send WebSocket notification
        if (wsManager) {
          wsManager.sendAgentUpdate(agent.agentId, agent.status, {
            action: 'updated',
            updates: updates
          });
        }

        logger.info('Agent updated', { agentId: agent.agentId });

        res.json(agent);
      } catch (error) {
        logger.error('Failed to update agent', { error: error.message });
        res.status(400).json({ error: 'Failed to update agent' });
      }
    }
  );

  // Delete agent
  apiRouter.delete('/api/agents/:id',
    authenticateToken,
    async (req: AuthenticatedRequest, res: Response) => {
      try {
        const { id } = req.params;
        
        // Get agent info before deletion
        const agent = await storage.getAgentById(id);
        if (!agent) {
          return res.status(404).json({ error: 'Agent not found' });
        }

        await storage.deleteAgent(id);

        // Clear cache
        await appCache.invalidateAgentCache(agent.agentId);

        // Log agent deletion
        await storage.insertActivity({
          userId: req.auth!.user.id,
          sessionId: req.auth!.session.id,
          activityType: ActivityType.AGENT_DELETE,
          activityDetails: { 
            agentId: agent.agentId,
            agentType: agent.agentType 
          },
          resourceType: 'agent',
          resourceId: agent.id,
          ipAddress: req.ip,
          userAgent: req.get('User-Agent'),
          success: true
        });

        // Send WebSocket notification
        if (wsManager) {
          wsManager.sendAgentUpdate(agent.agentId, 'deleted', {
            action: 'deleted'
          });
        }

        logger.info('Agent deleted', { agentId: agent.agentId });

        res.json({ message: 'Agent deleted successfully' });
      } catch (error) {
        logger.error('Failed to delete agent', { error: error.message });
        res.status(500).json({ error: 'Failed to delete agent' });
      }
    }
  );

  // ============================================================================
  // LEARNING DATA ROUTES
  // ============================================================================

  // List learning data
  apiRouter.get('/api/learning-data',
    authenticateToken,
    async (req: AuthenticatedRequest, res: Response) => {
      try {
        const { 
          sourceAgentId, 
          knowledgeType, 
          minConfidence, 
          limit = 50, 
          offset = 0 
        } = req.query;

        const result = await storage.listLearningData({
          sourceAgentId: sourceAgentId as string,
          knowledgeType: knowledgeType as string,
          minConfidence: minConfidence ? parseInt(minConfidence as string) : undefined,
          limit: parseInt(limit as string),
          offset: parseInt(offset as string)
        });

        res.json(result);
      } catch (error) {
        logger.error('Failed to list learning data', { error: error.message });
        res.status(500).json({ error: 'Failed to list learning data' });
      }
    }
  );

  // Create learning data
  apiRouter.post('/api/learning-data',
    authenticateToken,
    validateRequest(insertLearningDataSchema),
    async (req: AuthenticatedRequest, res: Response) => {
      try {
        const learningData = await storage.insertLearningData(req.validatedBody);

        // Cache learning data
        await appCache.cacheLearningData(
          learningData.sourceAgentId,
          learningData.knowledgeType,
          learningData
        );

        // Log learning data creation
        await storage.insertActivity({
          userId: req.auth!.user.id,
          sessionId: req.auth!.session.id,
          activityType: ActivityType.LEARNING_DATA_ADD,
          activityDetails: { 
            sourceAgentId: learningData.sourceAgentId,
            knowledgeType: learningData.knowledgeType,
            confidenceScore: learningData.confidenceScore
          },
          resourceType: 'learning_data',
          resourceId: learningData.id,
          ipAddress: req.ip,
          userAgent: req.get('User-Agent'),
          success: true
        });

        // Send WebSocket notification
        if (wsManager) {
          wsManager.sendLearningDataUpdate(learningData.sourceAgentId, learningData);
        }

        logger.info('Learning data created', { 
          id: learningData.id,
          sourceAgentId: learningData.sourceAgentId 
        });

        res.status(201).json(learningData);
      } catch (error) {
        logger.error('Failed to create learning data', { error: error.message });
        res.status(400).json({ error: 'Failed to create learning data' });
      }
    }
  );

  // Get learning data by ID
  apiRouter.get('/api/learning-data/:id',
    authenticateToken,
    async (req: AuthenticatedRequest, res: Response) => {
      try {
        const { id } = req.params;
        const learningData = await storage.getLearningDataById(id);

        if (!learningData) {
          return res.status(404).json({ error: 'Learning data not found' });
        }

        res.json(learningData);
      } catch (error) {
        logger.error('Failed to get learning data', { error: error.message });
        res.status(500).json({ error: 'Failed to get learning data' });
      }
    }
  );

  // Delete learning data
  apiRouter.delete('/api/learning-data/:id',
    authenticateToken,
    async (req: AuthenticatedRequest, res: Response) => {
      try {
        const { id } = req.params;
        const deleted = await storage.deleteLearningData(id);

        if (!deleted) {
          return res.status(404).json({ error: 'Learning data not found' });
        }

        logger.info('Learning data deleted', { id });

        res.json({ message: 'Learning data deleted successfully' });
      } catch (error) {
        logger.error('Failed to delete learning data', { error: error.message });
        res.status(500).json({ error: 'Failed to delete learning data' });
      }
    }
  );

  // ============================================================================
  // SYSTEM ROUTES
  // ============================================================================

  // Get system information
  apiRouter.get('/api/system/info',
    authenticateToken,
    async (req: AuthenticatedRequest, res: Response) => {
      try {
        const info = {
          version: process.env.APP_VERSION || '1.0.0',
          environment: process.env.NODE_ENV || 'development',
          uptime: Math.round(process.uptime()),
          timestamp: new Date().toISOString(),
          features: {
            websocket: !!wsManager,
            caching: true,
            authentication: true,
            validation: true
          }
        };

        res.json(info);
      } catch (error) {
        logger.error('Failed to get system info', { error: error.message });
        res.status(500).json({ error: 'Failed to get system info' });
      }
    }
  );

  // ============================================================================
  // ERROR HANDLING
  // ============================================================================

  // 404 handler for API routes
  apiRouter.use('/api/*', (req, res) => {
    res.status(404).json({
      error: 'API endpoint not found',
      path: req.path,
      method: req.method
    });
  });

  // Setup Vite for frontend serving in development
  const httpServer = createServer(app);
  
  if (process.env.NODE_ENV === 'development') {
    const { setupVite } = await import('./vite');
    await setupVite(app, httpServer);
    logger.info('Vite development server configured');
  }

  return httpServer;
}
