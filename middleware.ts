import { Request, Response, NextFunction } from 'express';
import helmet from 'helmet';
import cors from 'cors';
import compression from 'compression';
import { RateLimiterMemory } from 'rate-limiter-flexible';
import { createLogger } from './logger';
import { validator, ValidationError } from './validation';
import { rateLimiter } from './cache';

const logger = createLogger('middleware');

// Request logging middleware
export const requestLogger = (req: Request, res: Response, next: NextFunction): void => {
  const start = Date.now();
  const requestId = Math.random().toString(36).substring(7);
  
  // Add request ID to request object
  (req as any).requestId = requestId;
  
  logger.info('Request started', {
    requestId,
    method: req.method,
    url: req.url,
    ip: req.ip,
    userAgent: req.get('User-Agent'),
  });

  // Log response when finished
  res.on('finish', () => {
    const duration = Date.now() - start;
    logger.info('Request completed', {
      requestId,
      method: req.method,
      url: req.url,
      statusCode: res.statusCode,
      duration,
      contentLength: res.get('Content-Length'),
    });
  });

  next();
};

// Error handling middleware
export const errorHandler = (
  error: Error,
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  const requestId = (req as any).requestId;
  
  logger.error('Request error', {
    requestId,
    error: error.message,
    stack: error.stack,
    url: req.url,
    method: req.method,
  });

  // Handle validation errors
  if (error instanceof ValidationError) {
    res.status(400).json({
      error: 'Validation failed',
      details: error.errors,
      requestId,
    });
    return;
  }

  // Handle JWT errors
  if (error.name === 'JsonWebTokenError') {
    res.status(401).json({
      error: 'Invalid token',
      requestId,
    });
    return;
  }

  if (error.name === 'TokenExpiredError') {
    res.status(401).json({
      error: 'Token expired',
      requestId,
    });
    return;
  }

  // Handle database errors
  if (error.message.includes('duplicate key value')) {
    res.status(409).json({
      error: 'Resource already exists',
      requestId,
    });
    return;
  }

  if (error.message.includes('foreign key constraint')) {
    res.status(400).json({
      error: 'Invalid reference',
      requestId,
    });
    return;
  }

  // Default error response
  res.status(500).json({
    error: process.env.NODE_ENV === 'production' 
      ? 'Internal server error' 
      : error.message,
    requestId,
  });
};

// Security headers middleware
export const securityHeaders = helmet({
  contentSecurityPolicy: process.env.NODE_ENV === 'development' ? false : {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      scriptSrc: ["'self'"],
      imgSrc: ["'self'", "data:", "https:"],
      connectSrc: ["'self'"],
      fontSrc: ["'self'"],
      objectSrc: ["'none'"],
      mediaSrc: ["'self'"],
      frameSrc: ["'none'"],
    },
  },
  crossOriginResourcePolicy: { policy: "cross-origin" },
});

// CORS configuration
export const corsMiddleware = cors({
  origin: (origin, callback) => {
    // Allow requests with no origin (mobile apps, curl, etc.)
    if (!origin) return callback(null, true);
    
    // In development, allow all origins
    if (process.env.NODE_ENV === 'development') {
      return callback(null, true);
    }
    
    // Production origins
    const allowedOrigins = process.env.ALLOWED_ORIGINS?.split(',') || [];
    if (allowedOrigins.includes(origin)) {
      return callback(null, true);
    }
    
    callback(new Error('Not allowed by CORS'));
  },
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
  allowedHeaders: [
    'Origin',
    'X-Requested-With', 
    'Content-Type',
    'Accept',
    'Authorization',
    'X-Request-ID',
  ],
  credentials: true,
  maxAge: 86400, // 24 hours
});

// Compression middleware
export const compressionMiddleware = compression({
  filter: (req, res) => {
    if (req.headers['x-no-compression']) {
      return false;
    }
    return compression.filter(req, res);
  },
  threshold: 1024, // Only compress responses larger than 1KB
});

// Rate limiting middleware factory
export const createRateLimiter = (
  windowMs: number = 15 * 60 * 1000, // 15 minutes
  maxRequests: number = 100,
  keyGenerator?: (req: Request) => string
) => {
  const limiter = new RateLimiterMemory({
    keyBuilder: keyGenerator || ((req) => req.ip),
    points: maxRequests,
    duration: Math.floor(windowMs / 1000),
    blockDuration: Math.floor(windowMs / 1000),
  });

  return async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const key = keyGenerator ? keyGenerator(req) : req.ip;
      await limiter.consume(key);
      next();
    } catch (rateLimiterRes) {
      const secs = Math.round(rateLimiterRes.msBeforeNext / 1000) || 1;
      res.set('Retry-After', String(secs));
      res.status(429).json({
        error: 'Too many requests',
        retryAfter: secs,
      });
    }
  };
};

// Input sanitization middleware
export const sanitizeInput = (req: Request, res: Response, next: NextFunction): void => {
  try {
    if (req.body && typeof req.body === 'object') {
      req.body = validator.sanitizeObject(req.body);
    }
    
    if (req.query && typeof req.query === 'object') {
      req.query = validator.sanitizeObject(req.query);
    }
    
    if (req.params && typeof req.params === 'object') {
      req.params = validator.sanitizeObject(req.params);
    }
    
    next();
  } catch (error) {
    logger.error('Input sanitization error', { error: error.message });
    next(error);
  }
};

// Request size limiting middleware
export const requestSizeLimit = (maxSize: string = '10mb') => {
  return (req: Request, res: Response, next: NextFunction): void => {
    const contentLength = req.get('Content-Length');
    if (contentLength) {
      const sizeInBytes = parseInt(contentLength, 10);
      const maxSizeInBytes = parseSize(maxSize);
      
      if (sizeInBytes > maxSizeInBytes) {
        res.status(413).json({
          error: 'Request too large',
          maxSize,
        });
        return;
      }
    }
    next();
  };
};

// Request timeout middleware
export const requestTimeout = (timeoutMs: number = 30000) => {
  return (req: Request, res: Response, next: NextFunction): void => {
    const timer = setTimeout(() => {
      if (!res.headersSent) {
        res.status(408).json({
          error: 'Request timeout',
          timeout: timeoutMs,
        });
      }
    }, timeoutMs);
    
    res.on('finish', () => clearTimeout(timer));
    res.on('close', () => clearTimeout(timer));
    
    next();
  };
};

// Health check middleware
export const healthCheck = (req: Request, res: Response, next: NextFunction): void => {
  if (req.path === '/health' || req.path === '/ping') {
    res.status(200).json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      version: process.env.APP_VERSION || '1.0.0',
    });
    return;
  }
  next();
};

// Content type validation middleware
export const validateContentType = (allowedTypes: string[] = ['application/json']) => {
  return (req: Request, res: Response, next: NextFunction): void => {
    if (req.method === 'GET' || req.method === 'DELETE') {
      return next();
    }
    
    const contentType = req.get('Content-Type');
    if (!contentType) {
      res.status(400).json({
        error: 'Content-Type header required',
        allowed: allowedTypes,
      });
      return;
    }
    
    const isAllowed = allowedTypes.some(type => 
      contentType.toLowerCase().includes(type.toLowerCase())
    );
    
    if (!isAllowed) {
      res.status(415).json({
        error: 'Unsupported content type',
        provided: contentType,
        allowed: allowedTypes,
      });
      return;
    }
    
    next();
  };
};

// API versioning middleware
export const apiVersioning = (req: Request, res: Response, next: NextFunction): void => {
  const version = req.get('API-Version') || req.query.version || 'v1';
  (req as any).apiVersion = version;
  
  // Set response header
  res.set('API-Version', version);
  
  next();
};

// Security middleware for sensitive operations
export const requireSecureConnection = (req: Request, res: Response, next: NextFunction): void => {
  if (process.env.NODE_ENV === 'production' && !req.secure && req.get('X-Forwarded-Proto') !== 'https') {
    res.status(403).json({
      error: 'HTTPS required for this operation',
    });
    return;
  }
  next();
};

// Utility functions
function parseSize(size: string): number {
  const units: { [key: string]: number } = {
    b: 1,
    kb: 1024,
    mb: 1024 * 1024,
    gb: 1024 * 1024 * 1024,
  };
  
  const match = size.toLowerCase().match(/^(\d+(?:\.\d+)?)(b|kb|mb|gb)?$/);
  if (!match) {
    throw new Error(`Invalid size format: ${size}`);
  }
  
  const value = parseFloat(match[1]);
  const unit = match[2] || 'b';
  
  return Math.floor(value * units[unit]);
}

// Export middleware combinations
export const securityMiddleware = [
  securityHeaders,
  corsMiddleware,
  sanitizeInput,
  validateContentType(),
];

export const performanceMiddleware = [
  compressionMiddleware,
  requestTimeout(),
  requestSizeLimit(),
];

export const developmentMiddleware = process.env.NODE_ENV === 'development' ? [
  requestLogger,
] : [];
