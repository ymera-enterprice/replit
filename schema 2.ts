import { pgTable, text, serial, integer, boolean, timestamp, jsonb, uuid, varchar, index, uniqueIndex } from "drizzle-orm/pg-core";
import { createInsertSchema, createSelectSchema } from "drizzle-zod";
import { z } from "zod";
import { relations } from "drizzle-orm";

// Users table
export const users = pgTable("users", {
  id: uuid("id").primaryKey().defaultRandom(),
  username: varchar("username", { length: 50 }).notNull().unique(),
  email: varchar("email", { length: 255 }).notNull().unique(),
  passwordHash: text("password_hash").notNull(),
  salt: varchar("salt", { length: 32 }).notNull(),
  firstName: varchar("first_name", { length: 100 }),
  lastName: varchar("last_name", { length: 100 }),
  displayName: varchar("display_name", { length: 200 }),
  avatarUrl: text("avatar_url"),
  isEmailVerified: boolean("is_email_verified").default(false).notNull(),
  emailVerificationToken: varchar("email_verification_token", { length: 255 }),
  emailVerificationExpires: timestamp("email_verification_expires"),
  isMfaEnabled: boolean("is_mfa_enabled").default(false).notNull(),
  mfaSecret: varchar("mfa_secret", { length: 32 }),
  backupCodes: jsonb("backup_codes").default([]).notNull(),
  failedLoginAttempts: integer("failed_login_attempts").default(0).notNull(),
  lockedUntil: timestamp("locked_until"),
  lastLoginAt: timestamp("last_login_at"),
  lastLoginIp: varchar("last_login_ip", { length: 45 }),
  passwordChangedAt: timestamp("password_changed_at"),
  userStatus: varchar("user_status", { length: 50 }).default("pending_verification").notNull(),
  timezone: varchar("timezone", { length: 50 }).default("UTC").notNull(),
  language: varchar("language", { length: 10 }).default("en").notNull(),
  metadata: jsonb("metadata").default({}).notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
}, (table) => ({
  emailIdx: uniqueIndex("users_email_idx").on(table.email),
  usernameIdx: uniqueIndex("users_username_idx").on(table.username),
  statusEmailIdx: index("users_status_email_idx").on(table.userStatus, table.email),
}));

// User sessions table
export const userSessions = pgTable("user_sessions", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: uuid("user_id").references(() => users.id, { onDelete: "cascade" }).notNull(),
  sessionToken: varchar("session_token", { length: 255 }).unique().notNull(),
  refreshToken: varchar("refresh_token", { length: 255 }).unique(),
  expiresAt: timestamp("expires_at").notNull(),
  ipAddress: varchar("ip_address", { length: 45 }),
  userAgent: text("user_agent"),
  deviceInfo: jsonb("device_info").default({}).notNull(),
  isActive: boolean("is_active").default(true).notNull(),
  lastActivityAt: timestamp("last_activity_at").defaultNow().notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
}, (table) => ({
  sessionTokenIdx: uniqueIndex("sessions_token_idx").on(table.sessionToken),
  userIdIdx: index("sessions_user_id_idx").on(table.userId),
  activeIdx: index("sessions_active_idx").on(table.isActive, table.expiresAt),
}));

// User activities table
export const userActivities = pgTable("user_activities", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: uuid("user_id").references(() => users.id, { onDelete: "cascade" }).notNull(),
  sessionId: uuid("session_id").references(() => userSessions.id, { onDelete: "set null" }),
  activityType: varchar("activity_type", { length: 50 }).notNull(),
  activityDetails: jsonb("activity_details").default({}).notNull(),
  ipAddress: varchar("ip_address", { length: 45 }),
  userAgent: text("user_agent"),
  resourceType: varchar("resource_type", { length: 50 }),
  resourceId: varchar("resource_id", { length: 255 }),
  success: boolean("success").default(true).notNull(),
  errorMessage: text("error_message"),
  durationMs: integer("duration_ms"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
}, (table) => ({
  userActivityIdx: index("activities_user_type_time_idx").on(table.userId, table.activityType, table.createdAt),
  resourceIdx: index("activities_resource_idx").on(table.resourceType, table.resourceId),
}));

// Roles table
export const roles = pgTable("roles", {
  id: uuid("id").primaryKey().defaultRandom(),
  code: varchar("code", { length: 50 }).unique().notNull(),
  name: varchar("name", { length: 100 }).notNull(),
  description: text("description"),
  permissions: jsonb("permissions").default([]).notNull(),
  isSystemRole: boolean("is_system_role").default(false).notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
}, (table) => ({
  codeIdx: uniqueIndex("roles_code_idx").on(table.code),
}));

// User roles junction table
export const userRoles = pgTable("user_roles", {
  userId: uuid("user_id").references(() => users.id, { onDelete: "cascade" }).notNull(),
  roleId: uuid("role_id").references(() => roles.id, { onDelete: "cascade" }).notNull(),
  assignedAt: timestamp("assigned_at").defaultNow().notNull(),
  assignedBy: uuid("assigned_by").references(() => users.id),
}, (table) => ({
  userRoleIdx: index("user_roles_user_role_idx").on(table.userId, table.roleId),
}));

// Agents table
export const agents = pgTable("agents", {
  id: uuid("id").primaryKey().defaultRandom(),
  agentId: varchar("agent_id", { length: 100 }).unique().notNull(),
  agentType: varchar("agent_type", { length: 50 }).notNull(),
  name: varchar("name", { length: 200 }).notNull(),
  description: text("description"),
  capabilities: jsonb("capabilities").default([]).notNull(),
  configuration: jsonb("configuration").default({}).notNull(),
  maxConcurrentTasks: integer("max_concurrent_tasks").default(10).notNull(),
  timeoutSeconds: integer("timeout_seconds").default(30).notNull(),
  memoryLimitMb: integer("memory_limit_mb").default(512).notNull(),
  enabled: boolean("enabled").default(true).notNull(),
  status: varchar("status", { length: 50 }).default("inactive").notNull(),
  lastHeartbeat: timestamp("last_heartbeat"),
  metadata: jsonb("metadata").default({}).notNull(),
  createdBy: uuid("created_by").references(() => users.id),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
}, (table) => ({
  agentIdIdx: uniqueIndex("agents_agent_id_idx").on(table.agentId),
  typeIdx: index("agents_type_idx").on(table.agentType),
  statusIdx: index("agents_status_idx").on(table.status),
}));

// Agent learning data table
export const agentLearningData = pgTable("agent_learning_data", {
  id: uuid("id").primaryKey().defaultRandom(),
  sourceAgentId: varchar("source_agent_id", { length: 100 }).notNull(),
  knowledgeType: varchar("knowledge_type", { length: 50 }).notNull(),
  content: jsonb("content").notNull(),
  confidenceScore: integer("confidence_score").notNull(), // 0-100 for better precision
  tags: jsonb("tags").default([]).notNull(),
  metadata: jsonb("metadata").default({}).notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
}, (table) => ({
  sourceAgentIdx: index("learning_data_source_agent_idx").on(table.sourceAgentId),
  knowledgeTypeIdx: index("learning_data_knowledge_type_idx").on(table.knowledgeType),
  confidenceIdx: index("learning_data_confidence_idx").on(table.confidenceScore),
}));

// System health table
export const systemHealth = pgTable("system_health", {
  id: uuid("id").primaryKey().defaultRandom(),
  component: varchar("component", { length: 50 }).notNull(),
  status: varchar("status", { length: 20 }).notNull(),
  details: jsonb("details").default({}).notNull(),
  responseTimeMs: integer("response_time_ms"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
}, (table) => ({
  componentIdx: index("health_component_idx").on(table.component),
  statusIdx: index("health_status_idx").on(table.status),
  timeIdx: index("health_time_idx").on(table.createdAt),
}));

// Relations
export const usersRelations = relations(users, ({ many }) => ({
  sessions: many(userSessions),
  activities: many(userActivities),
  userRoles: many(userRoles),
  createdAgents: many(agents),
}));

export const userSessionsRelations = relations(userSessions, ({ one, many }) => ({
  user: one(users, {
    fields: [userSessions.userId],
    references: [users.id],
  }),
  activities: many(userActivities),
}));

export const userActivitiesRelations = relations(userActivities, ({ one }) => ({
  user: one(users, {
    fields: [userActivities.userId],
    references: [users.id],
  }),
  session: one(userSessions, {
    fields: [userActivities.sessionId],
    references: [userSessions.id],
  }),
}));

export const rolesRelations = relations(roles, ({ many }) => ({
  userRoles: many(userRoles),
}));

export const userRolesRelations = relations(userRoles, ({ one }) => ({
  user: one(users, {
    fields: [userRoles.userId],
    references: [users.id],
  }),
  role: one(roles, {
    fields: [userRoles.roleId],
    references: [roles.id],
  }),
}));

export const agentsRelations = relations(agents, ({ one, many }) => ({
  creator: one(users, {
    fields: [agents.createdBy],
    references: [users.id],
  }),
  learningData: many(agentLearningData),
}));

export const agentLearningDataRelations = relations(agentLearningData, ({ one }) => ({
  agent: one(agents, {
    fields: [agentLearningData.sourceAgentId],
    references: [agents.agentId],
  }),
}));

// Zod schemas for validation
export const insertUserSchema = createInsertSchema(users, {
  email: z.string().email(),
  username: z.string().min(3).max(50).regex(/^[a-zA-Z0-9_.-]+$/),
  passwordHash: z.string().min(1),
  salt: z.string().min(1),
}).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const selectUserSchema = createSelectSchema(users);

export const insertSessionSchema = createInsertSchema(userSessions).omit({
  id: true,
  createdAt: true,
});

export const selectSessionSchema = createSelectSchema(userSessions);

export const insertActivitySchema = createInsertSchema(userActivities).omit({
  id: true,
  createdAt: true,
});

export const selectActivitySchema = createSelectSchema(userActivities);

export const insertAgentSchema = createInsertSchema(agents, {
  agentId: z.string().min(1).max(100).regex(/^[a-zA-Z0-9_-]+$/),
  agentType: z.string().min(1).max(50).regex(/^[a-zA-Z_][a-zA-Z0-9_]*$/),
  capabilities: z.array(z.string()).min(1),
  maxConcurrentTasks: z.number().int().min(1).max(100),
  timeoutSeconds: z.number().int().min(1).max(300),
  memoryLimitMb: z.number().int().min(64).max(4096),
}).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const selectAgentSchema = createSelectSchema(agents);

export const insertLearningDataSchema = createInsertSchema(agentLearningData, {
  sourceAgentId: z.string().min(1),
  knowledgeType: z.string().min(1).regex(/^[a-zA-Z_][a-zA-Z0-9_]*$/),
  content: z.object({}).passthrough(),
  confidenceScore: z.number().int().min(0).max(100),
}).omit({
  id: true,
  createdAt: true,
});

export const selectLearningDataSchema = createSelectSchema(agentLearningData);

export const loginSchema = z.object({
  username: z.string().min(1),
  password: z.string().min(1),
});

export const registerSchema = z.object({
  username: z.string().min(3).max(50).regex(/^[a-zA-Z0-9_.-]+$/),
  email: z.string().email(),
  password: z.string().min(8).regex(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/),
  firstName: z.string().optional(),
  lastName: z.string().optional(),
});

// Types
export type InsertUser = z.infer<typeof insertUserSchema>;
export type SelectUser = typeof users.$inferSelect;
export type InsertSession = z.infer<typeof insertSessionSchema>;
export type SelectSession = typeof userSessions.$inferSelect;
export type InsertActivity = z.infer<typeof insertActivitySchema>;
export type SelectActivity = typeof userActivities.$inferSelect;
export type InsertAgent = z.infer<typeof insertAgentSchema>;
export type SelectAgent = typeof agents.$inferSelect;
export type InsertLearningData = z.infer<typeof insertLearningDataSchema>;
export type SelectLearningData = typeof agentLearningData.$inferSelect;
export type LoginRequest = z.infer<typeof loginSchema>;
export type RegisterRequest = z.infer<typeof registerSchema>;

// Enums
export const UserStatus = {
  ACTIVE: "active",
  INACTIVE: "inactive",
  SUSPENDED: "suspended",
  PENDING_VERIFICATION: "pending_verification",
  LOCKED: "locked",
  DELETED: "deleted",
} as const;

export const ActivityType = {
  LOGIN: "login",
  LOGOUT: "logout", 
  PASSWORD_CHANGE: "password_change",
  PROFILE_UPDATE: "profile_update",
  AGENT_CREATE: "agent_create",
  AGENT_UPDATE: "agent_update",
  AGENT_DELETE: "agent_delete",
  LEARNING_DATA_ADD: "learning_data_add",
  SYSTEM_ACCESS: "system_access",
} as const;

export const AgentStatus = {
  ACTIVE: "active",
  INACTIVE: "inactive",
  ERROR: "error",
  MAINTENANCE: "maintenance",
} as const;
