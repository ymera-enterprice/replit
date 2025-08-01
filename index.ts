import express from 'express';
import { createServer } from 'http';
import { registerRoutes } from './routes';
import { initializeDatabase, testConnection } from './storage';
import { initializeWebSocket } from './websocket';
import { createLogger } from './logger';
import {
  securityMiddleware,
  performanceMiddleware,
  developmentMiddleware,
  errorHandler,
  requestLogger,
  healthCheck
} from './middleware';
import { startPeriodicHealthChecks } from './health';
import { validator } from './validation';

const logger = createLogger('server');

async function startServer() {
  try {
    // Test database connection first
    logger.info('Testing database connection...');
    const dbConnected = await testConnection();
    if (!dbConnected) {
      throw new Error('Database connection failed');
    }

    // Initialize database
    await initializeDatabase();
    logger.info('Database initialized successfully');

    // Create Express app
    const app = express();
    
    // Add request tracking for health monitoring
    app.use(requestLogger);
    
    // Add health check before other middleware
    app.use(healthCheck);

    // Security middleware
    app.use(securityMiddleware);
    
    // Performance middleware
    app.use(performanceMiddleware);
    
    // Development middleware (logging, etc.)
    if (process.env.NODE_ENV === 'development') {
      app.use(developmentMiddleware);
    } else {
      app.use(requestLogger);
    }

    // Body parsing middleware
    app.use(express.json({ limit: '10mb' }));
    app.use(express.urlencoded({ extended: true, limit: '10mb' }));

    // Trust proxy for proper IP forwarding
    app.set('trust proxy', 1);

    // Register API routes
    const httpServer = await registerRoutes(app);
    
    // Initialize WebSocket server
    const wsManager = initializeWebSocket(httpServer);
    logger.info('WebSocket server initialized');

    // Global error handler (must be last)
    app.use(errorHandler);

    // Start periodic health checks
    startPeriodicHealthChecks();

    // Start server
    const PORT = parseInt(process.env.PORT || '8000');
    const HOST = process.env.HOST || '0.0.0.0';

    httpServer.listen(PORT, HOST, () => {
      logger.info('YMERA-Core server started', {
        port: PORT,
        host: HOST,
        env: process.env.NODE_ENV || 'development',
        version: process.env.APP_VERSION || '1.0.0'
      });
    });

    // Graceful shutdown handlers
    const gracefulShutdown = async (signal: string) => {
      logger.info(`Received ${signal}, starting graceful shutdown...`);
      
      try {
        // Stop accepting new connections
        httpServer.close(async () => {
          logger.info('HTTP server closed');
          
          try {
            // Clean up WebSocket connections
            if (wsManager) {
              wsManager.destroy();
            }
            
            // Close database connections
            const { closeDatabase } = await import('./storage');
            await closeDatabase();
            
            logger.info('Graceful shutdown completed');
            process.exit(0);
          } catch (error) {
            logger.error('Error during graceful shutdown', { error: error.message });
            process.exit(1);
          }
        });

        // Force exit after 30 seconds
        setTimeout(() => {
          logger.error('Forced shutdown after timeout');
          process.exit(1);
        }, 30000);
        
      } catch (error) {
        logger.error('Error during shutdown', { error: error.message });
        process.exit(1);
      }
    };

    // Handle shutdown signals
    process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
    process.on('SIGINT', () => gracefulShutdown('SIGINT'));
    
    // Handle uncaught exceptions
    process.on('uncaughtException', (error) => {
      logger.error('Uncaught exception', { error: error.message, stack: error.stack });
      gracefulShutdown('uncaughtException');
    });
    
    process.on('unhandledRejection', (reason, promise) => {
      logger.error('Unhandled rejection', { reason, promise });
      gracefulShutdown('unhandledRejection');
    });

  } catch (error) {
    logger.error('Failed to start server', { error: error.message, stack: error.stack });
    process.exit(1);
  }
}

// Start the server
startServer().catch((error) => {
  console.error('Fatal startup error:', error);
  process.exit(1);
});
