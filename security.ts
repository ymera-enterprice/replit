import { Request, Response, NextFunction } from 'express';
import rateLimit from 'express-rate-limit';
import helmet from 'helmet';
import * as cors from 'cors';

// Rate limiting configuration
export const createRateLimiter = (windowMs: number, max: number, message?: string) => {
  return rateLimit({
    windowMs,
    max,
    message: {
      success: false,
      error: message || 'Too many requests from this IP, please try again later.'
    },
    standardHeaders: true,
    legacyHeaders: false,
  });
};

// Different rate limits for different endpoints
export const generalRateLimit = createRateLimiter(
  15 * 60 * 1000, // 15 minutes
  100, // limit each IP to 100 requests per windowMs
  'Too many requests, please try again later.'
);

export const authRateLimit = createRateLimiter(
  15 * 60 * 1000, // 15 minutes
  5, // limit each IP to 5 login attempts per windowMs
  'Too many authentication attempts, please try again later.'
);

export const apiRateLimit = createRateLimiter(
  1 * 60 * 1000, // 1 minute
  60, // limit each IP to 60 API requests per minute
  'API rate limit exceeded, please slow down.'
);

// CORS configuration
const corsOptions = {
  origin: (origin: string | undefined, callback: (err: Error | null, allow?: boolean) => void) => {
    // Allow requests with no origin (like mobile apps or curl requests)
    if (!origin) return callback(null, true);
    
    // Allow localhost for development
    if (origin.includes('localhost') || origin.includes('127.0.0.1')) {
      return callback(null, true);
    }
    
    // Allow Replit domains
    if (origin.includes('.replit.dev') || origin.includes('.repl.co')) {
      return callback(null, true);
    }
    
    // Check environment variables for allowed origins
    const allowedOrigins = process.env.ALLOWED_ORIGINS?.split(',') || [];
    if (allowedOrigins.includes(origin)) {
      return callback(null, true);
    }
    
    callback(new Error('Not allowed by CORS'), false);
  },
  credentials: true,
  optionsSuccessStatus: 200,
};

// Security headers configuration
const helmetOptions = {
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com", "https://cdn.tailwindcss.com"],
      fontSrc: ["'self'", "https://fonts.gstatic.com"],
      scriptSrc: ["'self'", "'unsafe-inline'", "https://cdn.tailwindcss.com"],
      imgSrc: ["'self'", "data:", "https:"],
      connectSrc: ["'self'", "ws:", "wss:"],
      frameSrc: ["'none'"],
      objectSrc: ["'none'"],
    },
  },
  crossOriginEmbedderPolicy: false,
};

// Request sanitization middleware
export const sanitizeRequest = (req: Request, res: Response, next: NextFunction) => {
  // Remove potentially dangerous characters from string inputs
  const sanitizeString = (str: string): string => {
    return str.replace(/[<>\"']/g, '');
  };
  
  const sanitizeObject = (obj: any): any => {
    if (typeof obj === 'string') {
      return sanitizeString(obj);
    }
    
    if (Array.isArray(obj)) {
      return obj.map(sanitizeObject);
    }
    
    if (obj && typeof obj === 'object') {
      const sanitized: any = {};
      for (const [key, value] of Object.entries(obj)) {
        sanitized[key] = sanitizeObject(value);
      }
      return sanitized;
    }
    
    return obj;
  };
  
  // Sanitize request body
  if (req.body) {
    req.body = sanitizeObject(req.body);
  }
  
  // Sanitize query parameters
  if (req.query) {
    req.query = sanitizeObject(req.query);
  }
  
  next();
};

// Request logging middleware
export const requestLogger = (req: Request, res: Response, next: NextFunction) => {
  const start = Date.now();
  
  res.on('finish', () => {
    const duration = Date.now() - start;
    const { method, url, ip } = req;
    const { statusCode } = res;
    
    console.log(`${method} ${url} - ${statusCode} - ${duration}ms - ${ip}`);
    
    // Log suspicious activity
    if (statusCode >= 400) {
      console.warn(`Suspicious activity: ${method} ${url} - ${statusCode} - ${ip}`);
    }
  });
  
  next();
};

// IP whitelist middleware (for admin endpoints)
export const ipWhitelist = (allowedIPs: string[]) => {
  return (req: Request, res: Response, next: NextFunction) => {
    const clientIP = req.ip || req.connection.remoteAddress || req.socket.remoteAddress;
    
    if (!clientIP || !allowedIPs.includes(clientIP)) {
      return res.status(403).json({
        success: false,
        error: 'Access denied from this IP address'
      });
    }
    
    next();
  };
};

// Security headers middleware
export const securityHeaders = (req: Request, res: Response, next: NextFunction) => {
  // Custom security headers
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('X-XSS-Protection', '1; mode=block');
  res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
  res.setHeader('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');
  
  next();
};

// Main security middleware that combines all security measures
export const securityMiddleware = [
  helmet(helmetOptions),
  cors.default(corsOptions),
  generalRateLimit,
  requestLogger,
  sanitizeRequest,
  securityHeaders,
];

// API-specific security middleware
export const apiSecurityMiddleware = [
  helmet(helmetOptions),
  cors.default(corsOptions),
  apiRateLimit,
  requestLogger,
  sanitizeRequest,
  securityHeaders,
];

// Auth-specific security middleware
export const authSecurityMiddleware = [
  helmet(helmetOptions),
  cors.default(corsOptions),
  authRateLimit,
  requestLogger,
  sanitizeRequest,
  securityHeaders,
];
