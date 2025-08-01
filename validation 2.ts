import { z } from 'zod';
import DOMPurify from 'isomorphic-dompurify';
import { createLogger } from './logger';

const logger = createLogger('validation');

// SQL injection patterns
const SQL_INJECTION_PATTERNS = [
  /('|(\\'))+.*(;|--|\|)/i,
  /(;|--|\||#|\/\*|\*\/)/i,
  /(union|select|insert|delete|update|drop|create|alter|exec|execute)/i,
  /(script|javascript|vbscript|onload|onerror|onclick)/i,
];

// XSS patterns
const XSS_PATTERNS = [
  /<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi,
  /javascript:/gi,
  /on\w+\s*=/gi,
];

export interface ValidationConfig {
  strictMode: boolean;
  allowEmptyStrings: boolean;
  maxStringLength: number;
  maxListLength: number;
  maxDictDepth: number;
  sanitizeHtml: boolean;
  checkSqlInjection: boolean;
  checkXss: boolean;
}

export const defaultValidationConfig: ValidationConfig = {
  strictMode: true,
  allowEmptyStrings: false,
  maxStringLength: 10000,
  maxListLength: 1000,
  maxDictDepth: 10,
  sanitizeHtml: true,
  checkSqlInjection: true,
  checkXss: true,
};

export interface ValidationResult<T = any> {
  isValid: boolean;
  sanitizedData?: T;
  errors: string[];
  warnings: string[];
  metadata: Record<string, any>;
}

export class ValidationError extends Error {
  constructor(
    message: string,
    public errors: string[],
    public code: string = 'VALIDATION_ERROR'
  ) {
    super(message);
    this.name = 'ValidationError';
  }
}

export class SecurityValidator {
  constructor(private config: ValidationConfig = defaultValidationConfig) {}

  // Check for SQL injection patterns
  checkSqlInjection(value: string): boolean {
    if (!this.config.checkSqlInjection) return true;
    
    const valueLower = value.toLowerCase();
    return !SQL_INJECTION_PATTERNS.some(pattern => pattern.test(valueLower));
  }

  // Check for XSS patterns
  checkXss(value: string): boolean {
    if (!this.config.checkXss) return true;
    
    return !XSS_PATTERNS.some(pattern => pattern.test(value));
  }

  // Sanitize string input
  sanitizeString(value: string): string {
    if (typeof value !== 'string') {
      value = String(value);
    }

    // HTML sanitization
    if (this.config.sanitizeHtml) {
      value = DOMPurify.sanitize(value, { 
        ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'u'],
        ALLOWED_ATTR: []
      });
    }

    // Length check
    if (value.length > this.config.maxStringLength) {
      value = value.substring(0, this.config.maxStringLength);
    }

    return value.trim();
  }

  // Recursively sanitize object
  sanitizeObject(data: any, depth = 0): any {
    if (depth > this.config.maxDictDepth) {
      logger.warn('Maximum object depth exceeded during sanitization');
      return {};
    }

    if (typeof data === 'string') {
      return this.sanitizeString(data);
    }

    if (Array.isArray(data)) {
      return data
        .slice(0, this.config.maxListLength)
        .map(item => this.sanitizeObject(item, depth + 1));
    }

    if (data && typeof data === 'object') {
      const sanitized: any = {};
      for (const [key, value] of Object.entries(data)) {
        const sanitizedKey = this.sanitizeString(key);
        sanitized[sanitizedKey] = this.sanitizeObject(value, depth + 1);
      }
      return sanitized;
    }

    return data;
  }

  // Validate and sanitize input
  validateAndSanitize<T>(
    schema: z.ZodSchema<T>,
    data: unknown,
    customConfig?: Partial<ValidationConfig>
  ): ValidationResult<T> {
    const config = { ...this.config, ...customConfig };
    const errors: string[] = [];
    const warnings: string[] = [];

    try {
      // First, sanitize the data
      const sanitizedData = this.sanitizeObject(data);

      // Security checks for string values
      const checkSecurityRecursive = (obj: any, path = ''): void => {
        if (typeof obj === 'string') {
          if (!this.checkSqlInjection(obj)) {
            errors.push(`Potential SQL injection detected at ${path || 'root'}`);
          }
          if (!this.checkXss(obj)) {
            errors.push(`Potential XSS detected at ${path || 'root'}`);
          }
        } else if (Array.isArray(obj)) {
          obj.forEach((item, index) => 
            checkSecurityRecursive(item, `${path}[${index}]`)
          );
        } else if (obj && typeof obj === 'object') {
          Object.entries(obj).forEach(([key, value]) =>
            checkSecurityRecursive(value, path ? `${path}.${key}` : key)
          );
        }
      };

      checkSecurityRecursive(sanitizedData);

      // If security checks failed, return early
      if (errors.length > 0) {
        return {
          isValid: false,
          errors,
          warnings,
          metadata: { securityChecksFailed: true }
        };
      }

      // Schema validation
      const result = schema.safeParse(sanitizedData);
      
      if (!result.success) {
        const schemaErrors = result.error.errors.map(err => 
          `${err.path.join('.')}: ${err.message}`
        );
        return {
          isValid: false,
          errors: schemaErrors,
          warnings,
          metadata: { schemaValidationFailed: true }
        };
      }

      return {
        isValid: true,
        sanitizedData: result.data,
        errors: [],
        warnings,
        metadata: { sanitized: true }
      };

    } catch (error) {
      logger.error('Validation error', { error: error.message, data });
      return {
        isValid: false,
        errors: [`Validation error: ${error.message}`],
        warnings,
        metadata: { validationError: true }
      };
    }
  }
}

// Default validator instance
export const validator = new SecurityValidator();

// Middleware for request validation
export const validateRequest = <T>(schema: z.ZodSchema<T>) => {
  return (req: any, res: any, next: any) => {
    const result = validator.validateAndSanitize(schema, req.body);
    
    if (!result.isValid) {
      return res.status(400).json({
        error: 'Validation failed',
        details: result.errors,
        warnings: result.warnings
      });
    }

    req.validatedBody = result.sanitizedData;
    next();
  };
};

// Utility functions
export const validateEmail = (email: string): boolean => {
  return z.string().email().safeParse(email).success;
};

export const validateUuid = (uuid: string): boolean => {
  return z.string().uuid().safeParse(uuid).success;
};

export const validatePassword = (password: string): { isValid: boolean; errors: string[] } => {
  const errors: string[] = [];

  if (!password || password.length < 8) {
    errors.push('Password must be at least 8 characters long');
  }

  if (!/[a-z]/.test(password)) {
    errors.push('Password must contain at least one lowercase letter');
  }

  if (!/[A-Z]/.test(password)) {
    errors.push('Password must contain at least one uppercase letter');
  }

  if (!/\d/.test(password)) {
    errors.push('Password must contain at least one digit');
  }

  if (!/[@$!%*?&]/.test(password)) {
    errors.push('Password must contain at least one special character');
  }

  return {
    isValid: errors.length === 0,
    errors
  };
};
