import express from 'express';
import cors from 'cors';
import { createServer } from 'http';
import { Server } from 'socket.io';
import { ViteDevServer } from 'vite';
import { router } from './routes';
import { corsMiddleware, enterpriseCorsOptions } from './middleware/cors';
import { rateLimitMiddleware, rateLimitConfigs } from './middleware/rateLimit';
import { loggingMiddleware, StructuredLogger } from './middleware/logging';
import { errorHandler, notFoundHandler, setupUncaughtExceptionHandlers } from './middleware/errorHandler';
import { monitoringService } from './services/monitoring';
import { messageBroker } from './services/messageBroker';
import { taskDispatcher } from './services/taskDispatcher';
import { learningEngine } from './services/learningEngine';

const app = express();
const server = createServer(app);
const io = new Server(server, {
  cors: {
    origin: process.env.REPLIT_DOMAINS?.split(',').map(domain => `https://${domain}`) || ["http://localhost:5000"],
    methods: ["GET", "POST"],
    credentials: true
  }
});

const PORT = process.env.PORT || 5000;

// Setup uncaught exception handlers
setupUncaughtExceptionHandlers();

// ===============================================================================
// MIDDLEWARE SETUP
// ===============================================================================

// Trust proxy for rate limiting and IP detection
app.set('trust proxy', 1);

// CORS middleware
app.use(corsMiddleware(enterpriseCorsOptions));

// Body parsing
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Rate limiting
app.use(rateLimitMiddleware(rateLimitConfigs.api));

// Logging middleware
app.use(loggingMiddleware({
  logRequests: true,
  logResponses: true,
  logRequestBody: process.env.NODE_ENV === 'development',
  logResponseBody: false,
  maskSensitiveData: true,
  excludePaths: ['/health', '/metrics', '/favicon.ico'],
}));

// ===============================================================================
// WEBSOCKET SETUP FOR REAL-TIME UPDATES
// ===============================================================================

interface AuthenticatedSocket extends Socket {
  userId?: string;
  authenticated?: boolean;
}

import { Socket } from 'socket.io';
import { JWTManager } from './middleware/auth';

io.use(async (socket: AuthenticatedSocket, next) => {
  try {
    const token = socket.handshake.auth.token;
    if (!token) {
      return next(new Error('Authentication required'));
    }

    const payload = JWTManager.verifyToken(token);
    if (!payload) {
      return next(new Error('Invalid token'));
    }

    socket.userId = payload.sub;
    socket.authenticated = true;
    next();
  } catch (error) {
    next(new Error('Authentication failed'));
  }
});

io.on('connection', (socket: AuthenticatedSocket) => {
  StructuredLogger.info('WebSocket client connected', {
    socketId: socket.id,
    userId: socket.userId,
  });

  // Join user-specific room for personalized updates
  if (socket.userId) {
    socket.join(`user:${socket.userId}`);
  }

  // Join general monitoring room
  socket.join('monitoring');

  socket.on('subscribe:metrics', () => {
    socket.join('metrics');
    StructuredLogger.debug('Client subscribed to metrics', { socketId: socket.id });
  });

  socket.on('subscribe:alerts', () => {
    socket.join('alerts');
    StructuredLogger.debug('Client subscribed to alerts', { socketId: socket.id });
  });

  socket.on('subscribe:tasks', () => {
    socket.join('tasks');
    StructuredLogger.debug('Client subscribed to tasks', { socketId: socket.id });
  });

  socket.on('disconnect', () => {
    StructuredLogger.info('WebSocket client disconnected', {
      socketId: socket.id,
      userId: socket.userId,
    });
  });
});

// ===============================================================================
// REAL-TIME EVENT BROADCASTING
// ===============================================================================

// Broadcast system metrics updates
monitoringService.on('metrics:collected', (metrics) => {
  io.to('metrics').emit('metrics:update', {
    timestamp: new Date(),
    data: metrics,
  });
});

// Broadcast health status changes
monitoringService.on('health:checked', (health) => {
  io.to('monitoring').emit('health:update', {
    timestamp: new Date(),
    data: health,
  });
});

// Broadcast alerts
monitoringService.on('alert:created', (alert) => {
  io.to('alerts').emit('alert:new', {
    timestamp: new Date(),
    data: alert,
  });
});

// Broadcast task updates
taskDispatcher.on('task:created', (task) => {
  io.to('tasks').emit('task:created', {
    timestamp: new Date(),
    data: task,
  });
});

taskDispatcher.on('task:completed', ({ task, result }) => {
  io.to('tasks').emit('task:completed', {
    timestamp: new Date(),
    data: { task, result },
  });
});

taskDispatcher.on('task:failed', ({ task, error }) => {
  io.to('tasks').emit('task:failed', {
    timestamp: new Date(),
    data: { task, error: error.message },
  });
});

// Broadcast learning updates
learningEngine.on('pattern:discovered', (pattern) => {
  io.to('monitoring').emit('learning:pattern-discovered', {
    timestamp: new Date(),
    data: pattern,
  });
});

learningEngine.on('knowledge:validated', ({ item, validation }) => {
  io.to('monitoring').emit('learning:knowledge-validated', {
    timestamp: new Date(),
    data: { item, validation },
  });
});

// ===============================================================================
// API ROUTES
// ===============================================================================

// Health check endpoint (no auth required)
app.get('/health', async (req, res) => {
  try {
    const health = await monitoringService.getSystemHealth();
    res.json({
      status: 'ok',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      health: health.overall,
      version: process.env.npm_package_version || '1.0.0',
    });
  } catch (error) {
    res.status(503).json({
      status: 'error',
      message: 'Health check failed',
      timestamp: new Date().toISOString(),
    });
  }
});

// Metrics endpoint for monitoring tools
app.get('/metrics', async (req, res) => {
  try {
    const metrics = await monitoringService.getSystemMetrics();
    const serviceStats = await monitoringService.getServiceStats();
    
    // Return Prometheus-style metrics
    const metricsText = [
      `# HELP ymera_cpu_usage_percent CPU usage percentage`,
      `# TYPE ymera_cpu_usage_percent gauge`,
      `ymera_cpu_usage_percent ${metrics.cpu.usage}`,
      '',
      `# HELP ymera_memory_usage_percent Memory usage percentage`,
      `# TYPE ymera_memory_usage_percent gauge`,
      `ymera_memory_usage_percent ${metrics.memory.percentage}`,
      '',
      `# HELP ymera_disk_usage_percent Disk usage percentage`,
      `# TYPE ymera_disk_usage_percent gauge`,
      `ymera_disk_usage_percent ${metrics.disk.percentage}`,
      '',
      `# HELP ymera_requests_total Total number of requests`,
      `# TYPE ymera_requests_total counter`,
      `ymera_requests_total ${serviceStats.totalRequests}`,
      '',
      `# HELP ymera_errors_total Total number of errors`,
      `# TYPE ymera_errors_total counter`,
      `ymera_errors_total ${serviceStats.totalErrors}`,
      '',
    ].join('\n');
    
    res.set('Content-Type', 'text/plain');
    res.send(metricsText);
  } catch (error) {
    res.status(500).json({ error: 'Failed to collect metrics' });
  }
});

// API routes
app.use('/api', router);

// ===============================================================================
// ERROR HANDLING
// ===============================================================================

// 404 handler
app.use(notFoundHandler);

// Global error handler
app.use(errorHandler);

// ===============================================================================
// VITE INTEGRATION (Development)
// ===============================================================================

if (process.env.NODE_ENV !== 'production') {
  const vite = await import('./vite');
  vite.registerVite(app);
}

// ===============================================================================
// PERIODIC TASKS
// ===============================================================================

// Start periodic cleanup tasks
setInterval(async () => {
  try {
    // Clean up expired temp files
    const fileManager = (await import('./services/fileManager')).FileManager;
    const fm = new fileManager();
    await fm.cleanupExpiredFiles();
  } catch (error) {
    StructuredLogger.error('Cleanup task failed', error);
  }
}, 60 * 60 * 1000); // Run every hour

// ===============================================================================
// GRACEFUL SHUTDOWN
// ===============================================================================

const gracefulShutdown = async (signal: string) => {
  StructuredLogger.info(`Received ${signal}, starting graceful shutdown`);
  
  // Stop accepting new connections
  server.close(() => {
    StructuredLogger.info('HTTP server closed');
  });
  
  // Close WebSocket connections
  io.close(() => {
    StructuredLogger.info('WebSocket server closed');
  });
  
  // Cleanup services
  try {
    messageBroker.destroy();
    taskDispatcher.destroy();
    learningEngine.destroy();
    monitoringService.destroy();
    
    StructuredLogger.info('All services cleaned up');
  } catch (error) {
    StructuredLogger.error('Error during service cleanup', error);
  }
  
  // Force exit after timeout
  setTimeout(() => {
    StructuredLogger.error('Forced shutdown due to timeout');
    process.exit(1);
  }, 10000);
  
  process.exit(0);
};

// Handle shutdown signals
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

// ===============================================================================
// SERVER STARTUP
// ===============================================================================

server.listen(PORT, '0.0.0.0', () => {
  StructuredLogger.info('YMERA Enterprise Server started', {
    port: PORT,
    environment: process.env.NODE_ENV || 'development',
    nodeVersion: process.version,
    uptime: process.uptime(),
  });
  
  // Log startup summary
  console.log('\n🚀 YMERA Enterprise Platform');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log(`🌐 Server running on: http://0.0.0.0:${PORT}`);
  console.log(`📊 Health check: http://0.0.0.0:${PORT}/health`);
  console.log(`📈 Metrics: http://0.0.0.0:${PORT}/metrics`);
  console.log(`🔧 API docs: http://0.0.0.0:${PORT}/api`);
  console.log('═══════════════════════════════════════════════════════════════\n');
});

export { app, server, io };
