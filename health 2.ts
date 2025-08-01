import { Request, Response } from 'express';
import { db } from './storage';
import { systemHealth } from '@shared/schema';
import { createLogger } from './logger';
import { cacheHealthCheck } from './cache';
import { wsManager } from './websocket';

const logger = createLogger('health');

export interface HealthStatus {
  status: 'healthy' | 'unhealthy' | 'degraded';
  timestamp: string;
  uptime: number;
  version: string;
  components: {
    database: ComponentHealth;
    cache: ComponentHealth;
    websocket: ComponentHealth;
    api: ComponentHealth;
  };
  metrics?: {
    memoryUsage: NodeJS.MemoryUsage;
    cpuUsage: NodeJS.CpuUsage;
    activeConnections: number;
    requestCount?: number;
    errorRate?: number;
  };
}

export interface ComponentHealth {
  status: 'healthy' | 'unhealthy' | 'degraded';
  responseTime?: number;
  details?: any;
  lastCheck: string;
  error?: string;
}

class HealthMonitor {
  private requestCount = 0;
  private errorCount = 0;
  private startTime = Date.now();
  private cpuUsage = process.cpuUsage();

  constructor() {
    // Reset counters periodically
    setInterval(() => {
      this.requestCount = 0;
      this.errorCount = 0;
      this.cpuUsage = process.cpuUsage();
    }, 60000); // Reset every minute
  }

  incrementRequests(): void {
    this.requestCount++;
  }

  incrementErrors(): void {
    this.errorCount++;
  }

  async checkDatabase(): Promise<ComponentHealth> {
    const start = Date.now();
    try {
      // Simple query to test database connectivity
      await db.select().from(systemHealth).limit(1);
      
      const responseTime = Date.now() - start;
      
      return {
        status: 'healthy',
        responseTime,
        lastCheck: new Date().toISOString(),
        details: {
          connected: true,
          responseTimeMs: responseTime,
        },
      };
    } catch (error) {
      logger.error('Database health check failed', { error: error.message });
      return {
        status: 'unhealthy',
        responseTime: Date.now() - start,
        lastCheck: new Date().toISOString(),
        error: error.message,
        details: {
          connected: false,
        },
      };
    }
  }

  async checkCache(): Promise<ComponentHealth> {
    const start = Date.now();
    try {
      const cacheHealth = await cacheHealthCheck();
      const responseTime = Date.now() - start;
      
      return {
        status: cacheHealth.status,
        responseTime,
        lastCheck: new Date().toISOString(),
        details: cacheHealth.stats,
        error: cacheHealth.error,
      };
    } catch (error) {
      logger.error('Cache health check failed', { error: error.message });
      return {
        status: 'unhealthy',
        responseTime: Date.now() - start,
        lastCheck: new Date().toISOString(),
        error: error.message,
      };
    }
  }

  checkWebSocket(): ComponentHealth {
    try {
      if (!wsManager) {
        return {
          status: 'unhealthy',
          lastCheck: new Date().toISOString(),
          error: 'WebSocket manager not initialized',
        };
      }

      const stats = wsManager.getStats();
      
      return {
        status: 'healthy',
        lastCheck: new Date().toISOString(),
        details: {
          totalConnections: stats.totalConnections,
          uniqueUsers: stats.uniqueUsers,
          channels: stats.channels.length,
        },
      };
    } catch (error) {
      logger.error('WebSocket health check failed', { error: error.message });
      return {
        status: 'unhealthy',
        lastCheck: new Date().toISOString(),
        error: error.message,
      };
    }
  }

  checkAPI(): ComponentHealth {
    const errorRate = this.requestCount > 0 ? (this.errorCount / this.requestCount) * 100 : 0;
    const status = errorRate > 10 ? 'degraded' : errorRate > 50 ? 'unhealthy' : 'healthy';
    
    return {
      status,
      lastCheck: new Date().toISOString(),
      details: {
        requestCount: this.requestCount,
        errorCount: this.errorCount,
        errorRate: Math.round(errorRate * 100) / 100,
      },
    };
  }

  getSystemMetrics() {
    const memoryUsage = process.memoryUsage();
    const cpuUsage = process.cpuUsage(this.cpuUsage);
    
    return {
      memoryUsage: {
        rss: Math.round(memoryUsage.rss / 1024 / 1024), // MB
        heapTotal: Math.round(memoryUsage.heapTotal / 1024 / 1024), // MB
        heapUsed: Math.round(memoryUsage.heapUsed / 1024 / 1024), // MB
        external: Math.round(memoryUsage.external / 1024 / 1024), // MB
      },
      cpuUsage: {
        user: cpuUsage.user,
        system: cpuUsage.system,
      },
      uptime: Math.round(process.uptime()),
      pid: process.pid,
      platform: process.platform,
      nodeVersion: process.version,
    };
  }

  async getFullHealthStatus(): Promise<HealthStatus> {
    const [database, cache] = await Promise.all([
      this.checkDatabase(),
      this.checkCache(),
    ]);
    
    const websocket = this.checkWebSocket();
    const api = this.checkAPI();
    
    const components = { database, cache, websocket, api };
    
    // Determine overall status
    const unhealthyComponents = Object.values(components).filter(c => c.status === 'unhealthy').length;
    const degradedComponents = Object.values(components).filter(c => c.status === 'degraded').length;
    
    let overallStatus: 'healthy' | 'unhealthy' | 'degraded';
    if (unhealthyComponents > 0) {
      overallStatus = 'unhealthy';
    } else if (degradedComponents > 0) {
      overallStatus = 'degraded';
    } else {
      overallStatus = 'healthy';
    }

    const metrics = this.getSystemMetrics();
    
    return {
      status: overallStatus,
      timestamp: new Date().toISOString(),
      uptime: Math.round(process.uptime()),
      version: process.env.APP_VERSION || '1.0.0',
      components,
      metrics: {
        memoryUsage: process.memoryUsage(),
        cpuUsage: process.cpuUsage(),
        activeConnections: wsManager ? wsManager.getStats().totalConnections : 0,
        requestCount: this.requestCount,
        errorRate: this.requestCount > 0 ? (this.errorCount / this.requestCount) * 100 : 0,
      },
    };
  }

  async persistHealthStatus(status: HealthStatus): Promise<void> {
    try {
      await db.insert(systemHealth).values({
        component: 'system',
        status: status.status,
        details: {
          components: status.components,
          metrics: status.metrics,
          uptime: status.uptime,
          version: status.version,
        },
        responseTimeMs: undefined,
      });
    } catch (error) {
      logger.error('Failed to persist health status', { error: error.message });
    }
  }
}

// Create singleton instance
export const healthMonitor = new HealthMonitor();

// Health check endpoints
export const healthCheck = async (req: Request, res: Response): Promise<void> => {
  try {
    const healthStatus = await healthMonitor.getFullHealthStatus();
    
    // Set appropriate HTTP status code
    const httpStatus = healthStatus.status === 'healthy' ? 200 : 
                      healthStatus.status === 'degraded' ? 200 : 503;
    
    res.status(httpStatus).json(healthStatus);
    
    // Persist health status for monitoring
    await healthMonitor.persistHealthStatus(healthStatus);
    
  } catch (error) {
    logger.error('Health check failed', { error: error.message });
    res.status(500).json({
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      error: 'Health check failed',
      details: error.message,
    });
  }
};

// Lightweight readiness check
export const readinessCheck = async (req: Request, res: Response): Promise<void> => {
  try {
    // Quick checks for essential components
    const [dbHealth, cacheHealth] = await Promise.all([
      healthMonitor.checkDatabase(),
      healthMonitor.checkCache(),
    ]);
    
    const isReady = dbHealth.status === 'healthy' && cacheHealth.status !== 'unhealthy';
    
    if (isReady) {
      res.status(200).json({
        status: 'ready',
        timestamp: new Date().toISOString(),
        components: { database: dbHealth.status, cache: cacheHealth.status },
      });
    } else {
      res.status(503).json({
        status: 'not_ready',
        timestamp: new Date().toISOString(),
        components: { database: dbHealth.status, cache: cacheHealth.status },
      });
    }
  } catch (error) {
    logger.error('Readiness check failed', { error: error.message });
    res.status(503).json({
      status: 'not_ready',
      timestamp: new Date().toISOString(),
      error: error.message,
    });
  }
};

// Simple liveness check
export const livenessCheck = (req: Request, res: Response): void => {
  res.status(200).json({
    status: 'alive',
    timestamp: new Date().toISOString(),
    uptime: Math.round(process.uptime()),
    pid: process.pid,
  });
};

// Metrics endpoint
export const metricsEndpoint = (req: Request, res: Response): void => {
  const metrics = healthMonitor.getSystemMetrics();
  res.status(200).json({
    timestamp: new Date().toISOString(),
    ...metrics,
  });
};

// Middleware to track requests and errors
export const trackRequest = (req: Request, res: Response, next: any): void => {
  healthMonitor.incrementRequests();
  
  res.on('finish', () => {
    if (res.statusCode >= 400) {
      healthMonitor.incrementErrors();
    }
  });
  
  next();
};

// Periodic health checks
export const startPeriodicHealthChecks = (): void => {
  const interval = parseInt(process.env.HEALTH_CHECK_INTERVAL || '300000'); // 5 minutes
  
  setInterval(async () => {
    try {
      const healthStatus = await healthMonitor.getFullHealthStatus();
      await healthMonitor.persistHealthStatus(healthStatus);
      
      // Log warning if system is not healthy
      if (healthStatus.status !== 'healthy') {
        logger.warn('System health check warning', { 
          status: healthStatus.status,
          unhealthyComponents: Object.entries(healthStatus.components)
            .filter(([, component]) => component.status !== 'healthy')
            .map(([name]) => name),
        });
      }
      
    } catch (error) {
      logger.error('Periodic health check failed', { error: error.message });
    }
  }, interval);
  
  logger.info('Periodic health checks started', { intervalMs: interval });
};
