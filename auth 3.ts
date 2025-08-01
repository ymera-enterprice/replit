import jwt from 'jsonwebtoken';
import bcrypt from 'bcryptjs';
import crypto from 'crypto';
import { Request, Response, NextFunction } from 'express';
import { z } from 'zod';
import { eq, and, lt } from 'drizzle-orm';
import { db } from './storage';
import { users, userSessions, SelectUser, SelectSession } from '@shared/schema';
import { createLogger } from './logger';
import { validator } from './validation';

const logger = createLogger('auth');

// Configuration
const JWT_SECRET = process.env.JWT_SECRET || 'your-super-secret-jwt-key-change-in-production';
const JWT_EXPIRES_IN = process.env.JWT_EXPIRES_IN || '24h';
const REFRESH_TOKEN_EXPIRES_IN = process.env.REFRESH_TOKEN_EXPIRES_IN || '7d';
const BCRYPT_ROUNDS = 12;
const MAX_LOGIN_ATTEMPTS = 5;
const LOCKOUT_DURATION = 30 * 60 * 1000; // 30 minutes
const SESSION_TIMEOUT = 24 * 60 * 60 * 1000; // 24 hours

export interface AuthenticatedUser {
  id: string;
  username: string;
  email: string;
  displayName?: string;
  roles: string[];
}

export interface JWTPayload {
  sub: string;
  username: string;
  email: string;
  sessionId: string;
  iat?: number;
  exp?: number;
}

export interface AuthContext {
  user: AuthenticatedUser;
  session: SelectSession;
}

// Password utilities
export class PasswordManager {
  static async hashPassword(password: string): Promise<{ hash: string; salt: string }> {
    const salt = crypto.randomBytes(16).toString('hex');
    const hash = await bcrypt.hash(password + salt, BCRYPT_ROUNDS);
    return { hash, salt };
  }

  static async verifyPassword(
    password: string,
    hash: string,
    salt: string
  ): Promise<boolean> {
    try {
      return await bcrypt.compare(password + salt, hash);
    } catch (error) {
      logger.error('Password verification error', { error: error.message });
      return false;
    }
  }

  static generateSecureToken(length = 32): string {
    return crypto.randomBytes(length).toString('hex');
  }
}

// JWT utilities  
export class JWTManager {
  static generateTokens(payload: Omit<JWTPayload, 'iat' | 'exp'>): {
    accessToken: string;
    refreshToken: string;
  } {
    const accessToken = jwt.sign(payload, JWT_SECRET, {
      expiresIn: JWT_EXPIRES_IN,
      issuer: 'ymera-core',
      audience: 'ymera-platform',
    });

    const refreshToken = jwt.sign(
      { sub: payload.sub, sessionId: payload.sessionId },
      JWT_SECRET,
      {
        expiresIn: REFRESH_TOKEN_EXPIRES_IN,
        issuer: 'ymera-core',
        audience: 'ymera-platform',
      }
    );

    return { accessToken, refreshToken };
  }

  static verifyToken(token: string): JWTPayload | null {
    try {
      const decoded = jwt.verify(token, JWT_SECRET, {
        issuer: 'ymera-core',
        audience: 'ymera-platform',
      }) as JWTPayload;

      return decoded;
    } catch (error) {
      logger.warn('Token verification failed', { error: error.message });
      return null;
    }
  }

  static extractTokenFromHeader(authHeader?: string): string | null {
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return null;
    }
    return authHeader.substring(7);
  }
}

// Session management
export class SessionManager {
  static async createSession(
    userId: string,
    ipAddress?: string,
    userAgent?: string,
    deviceInfo?: any
  ): Promise<SelectSession> {
    const sessionToken = PasswordManager.generateSecureToken();
    const refreshToken = PasswordManager.generateSecureToken();
    const expiresAt = new Date(Date.now() + SESSION_TIMEOUT);

    const [session] = await db
      .insert(userSessions)
      .values({
        userId,
        sessionToken,
        refreshToken,
        expiresAt,
        ipAddress,
        userAgent,
        deviceInfo: deviceInfo || {},
      })
      .returning();

    logger.info('Session created', { 
      userId, 
      sessionId: session.id,
      ipAddress 
    });

    return session;
  }

  static async validateSession(sessionToken: string): Promise<{
    user: SelectUser;
    session: SelectSession;
  } | null> {
    try {
      const [result] = await db
        .select({
          user: users,
          session: userSessions,
        })
        .from(userSessions)
        .innerJoin(users, eq(users.id, userSessions.userId))
        .where(
          and(
            eq(userSessions.sessionToken, sessionToken),
            eq(userSessions.isActive, true)
          )
        )
        .limit(1);

      if (!result) {
        return null;
      }

      // Check if session is expired
      if (new Date() > result.session.expiresAt) {
        await this.invalidateSession(sessionToken);
        return null;
      }

      // Update last activity
      await db
        .update(userSessions)
        .set({ lastActivityAt: new Date() })
        .where(eq(userSessions.sessionToken, sessionToken));

      return result;
    } catch (error) {
      logger.error('Session validation error', { error: error.message });
      return null;
    }
  }

  static async invalidateSession(sessionToken: string): Promise<void> {
    await db
      .update(userSessions)
      .set({ isActive: false })
      .where(eq(userSessions.sessionToken, sessionToken));

    logger.info('Session invalidated', { sessionToken });
  }

  static async invalidateAllUserSessions(userId: string): Promise<void> {
    await db
      .update(userSessions)
      .set({ isActive: false })
      .where(eq(userSessions.userId, userId));

    logger.info('All user sessions invalidated', { userId });
  }

  static async cleanupExpiredSessions(): Promise<void> {
    const now = new Date();
    const result = await db
      .update(userSessions)
      .set({ isActive: false })
      .where(
        and(
          eq(userSessions.isActive, true),
          lt(userSessions.expiresAt, now)
        )
      );

    logger.info('Expired sessions cleaned up', { count: result.rowCount });
  }
}

// Authentication service
export class AuthService {
  static async register(
    username: string,
    email: string,
    password: string,
    firstName?: string,
    lastName?: string
  ): Promise<{ user: SelectUser; tokens: { accessToken: string; refreshToken: string } }> {
    // Check if user already exists
    const existingUser = await db
      .select()
      .from(users)
      .where(eq(users.email, email))
      .limit(1);

    if (existingUser.length > 0) {
      throw new Error('User already exists with this email');
    }

    const existingUsername = await db
      .select()
      .from(users)
      .where(eq(users.username, username))
      .limit(1);

    if (existingUsername.length > 0) {
      throw new Error('Username already taken');
    }

    // Hash password
    const { hash, salt } = await PasswordManager.hashPassword(password);

    // Create user
    const [newUser] = await db
      .insert(users)
      .values({
        username,
        email,
        passwordHash: hash,
        salt,
        firstName,
        lastName,
        displayName: firstName && lastName ? `${firstName} ${lastName}` : username,
        userStatus: 'pending_verification',
      })
      .returning();

    // Create session
    const session = await SessionManager.createSession(newUser.id);

    // Generate tokens
    const tokens = JWTManager.generateTokens({
      sub: newUser.id,
      username: newUser.username,
      email: newUser.email,
      sessionId: session.id,
    });

    logger.info('User registered successfully', { 
      userId: newUser.id, 
      username, 
      email 
    });

    return { user: newUser, tokens };
  }

  static async login(
    username: string,
    password: string,
    ipAddress?: string,
    userAgent?: string
  ): Promise<{ user: SelectUser; tokens: { accessToken: string; refreshToken: string } }> {
    // Find user by username or email
    const [user] = await db
      .select()
      .from(users)
      .where(eq(users.username, username))
      .limit(1);

    if (!user) {
      const [userByEmail] = await db
        .select()
        .from(users)
        .where(eq(users.email, username))
        .limit(1);
      
      if (!userByEmail) {
        throw new Error('Invalid credentials');
      }
      
      return this.login(userByEmail.username, password, ipAddress, userAgent);
    }

    // Check account status
    if (user.userStatus === 'locked') {
      // Check if lockout has expired
      if (user.lockedUntil && new Date() < user.lockedUntil) {
        throw new Error('Account is temporarily locked. Please try again later.');
      } else {
        // Unlock account
        await db
          .update(users)
          .set({
            userStatus: 'active',
            lockedUntil: null,
            failedLoginAttempts: 0,
          })
          .where(eq(users.id, user.id));
      }
    }

    if (user.userStatus === 'suspended') {
      throw new Error('Account is suspended');
    }

    // Verify password
    const isValidPassword = await PasswordManager.verifyPassword(
      password,
      user.passwordHash,
      user.salt
    );

    if (!isValidPassword) {
      // Increment failed attempts
      const newFailedAttempts = user.failedLoginAttempts + 1;
      const shouldLock = newFailedAttempts >= MAX_LOGIN_ATTEMPTS;

      await db
        .update(users)
        .set({
          failedLoginAttempts: newFailedAttempts,
          ...(shouldLock && {
            userStatus: 'locked',
            lockedUntil: new Date(Date.now() + LOCKOUT_DURATION),
          }),
        })
        .where(eq(users.id, user.id));

      throw new Error('Invalid credentials');
    }

    // Reset failed attempts and update last login
    await db
      .update(users)
      .set({
        failedLoginAttempts: 0,
        lastLoginAt: new Date(),
        lastLoginIp: ipAddress,
        userStatus: user.userStatus === 'pending_verification' ? 'active' : user.userStatus,
      })
      .where(eq(users.id, user.id));

    // Create session
    const session = await SessionManager.createSession(
      user.id,
      ipAddress,
      userAgent
    );

    // Generate tokens
    const tokens = JWTManager.generateTokens({
      sub: user.id,
      username: user.username,
      email: user.email,
      sessionId: session.id,
    });

    logger.info('User logged in successfully', { 
      userId: user.id, 
      username: user.username,
      ipAddress 
    });

    return { user: { ...user, userStatus: 'active' }, tokens };
  }

  static async logout(sessionToken: string): Promise<void> {
    await SessionManager.invalidateSession(sessionToken);
    logger.info('User logged out', { sessionToken });
  }

  static async refreshToken(refreshToken: string): Promise<{
    accessToken: string;
    refreshToken: string;
  }> {
    const payload = JWTManager.verifyToken(refreshToken);
    if (!payload) {
      throw new Error('Invalid refresh token');
    }

    // Validate session
    const sessionData = await db
      .select({
        user: users,
        session: userSessions,
      })
      .from(userSessions)
      .innerJoin(users, eq(users.id, userSessions.userId))
      .where(
        and(
          eq(userSessions.id, payload.sessionId),
          eq(userSessions.isActive, true)
        )
      )
      .limit(1);

    if (sessionData.length === 0) {
      throw new Error('Session not found or expired');
    }

    const { user, session } = sessionData[0];

    // Generate new tokens
    const tokens = JWTManager.generateTokens({
      sub: user.id,
      username: user.username,
      email: user.email,
      sessionId: session.id,
    });

    // Update session with new refresh token
    await db
      .update(userSessions)
      .set({
        refreshToken: tokens.refreshToken,
        lastActivityAt: new Date(),
      })
      .where(eq(userSessions.id, session.id));

    return tokens;
  }
}

// Middleware
export const authenticateToken = async (
  req: Request & { auth?: AuthContext },
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const token = JWTManager.extractTokenFromHeader(req.headers.authorization);
    
    if (!token) {
      res.status(401).json({ error: 'Access token required' });
      return;
    }

    const payload = JWTManager.verifyToken(token);
    if (!payload) {
      res.status(401).json({ error: 'Invalid or expired token' });
      return;
    }

    // Validate session
    const sessionData = await SessionManager.validateSession(payload.sessionId);
    if (!sessionData) {
      res.status(401).json({ error: 'Session expired or invalid' });
      return;
    }

    // Add auth context to request
    req.auth = {
      user: {
        id: sessionData.user.id,
        username: sessionData.user.username,
        email: sessionData.user.email,
        displayName: sessionData.user.displayName || undefined,
        roles: [], // TODO: Implement role loading
      },
      session: sessionData.session,
    };

    next();
  } catch (error) {
    logger.error('Authentication middleware error', { error: error.message });
    res.status(500).json({ error: 'Authentication error' });
  }
};

export const optionalAuth = async (
  req: Request & { auth?: AuthContext },
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const token = JWTManager.extractTokenFromHeader(req.headers.authorization);
    
    if (token) {
      const payload = JWTManager.verifyToken(token);
      if (payload) {
        const sessionData = await SessionManager.validateSession(payload.sessionId);
        if (sessionData) {
          req.auth = {
            user: {
              id: sessionData.user.id,
              username: sessionData.user.username,
              email: sessionData.user.email,
              displayName: sessionData.user.displayName || undefined,
              roles: [],
            },
            session: sessionData.session,
          };
        }
      }
    }

    next();
  } catch (error) {
    logger.warn('Optional auth middleware error', { error: error.message });
    next();
  }
};

// Initialize periodic cleanup
setInterval(() => {
  SessionManager.cleanupExpiredSessions().catch(error => 
    logger.error('Session cleanup error', { error: error.message })
  );
}, 60 * 60 * 1000); // Run every hour
