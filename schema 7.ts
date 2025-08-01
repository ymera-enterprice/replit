import { pgTable, serial, text, timestamp, integer, boolean, jsonb, varchar, uuid, pgEnum } from 'drizzle-orm/pg-core';
import { relations } from 'drizzle-orm';
import { createInsertSchema } from 'drizzle-zod';
import { z } from 'zod';

// Enums
export const agentStatusEnum = pgEnum('agent_status', ['initializing', 'active', 'idle', 'busy', 'paused', 'error', 'shutdown']);
export const agentTypeEnum = pgEnum('agent_type', ['core', 'specialized', 'custom']);
export const taskStatusEnum = pgEnum('task_status', ['pending', 'running', 'completed', 'failed', 'cancelled']);
export const taskPriorityEnum = pgEnum('task_priority', ['low', 'medium', 'high', 'urgent']);
export const collaborationStatusEnum = pgEnum('collaboration_status', ['active', 'paused', 'completed', 'cancelled']);
export const projectStatusEnum = pgEnum('project_status', ['planning', 'active', 'paused', 'completed', 'cancelled']);

// Users table
export const users = pgTable('users', {
  id: serial('id').primaryKey(),
  username: varchar('username', { length: 255 }).notNull().unique(),
  email: varchar('email', { length: 255 }).notNull().unique(),
  passwordHash: text('password_hash').notNull(),
  role: varchar('role', { length: 50 }).notNull().default('user'),
  isActive: boolean('is_active').notNull().default(true),
  lastLogin: timestamp('last_login'),
  createdAt: timestamp('created_at').notNull().defaultNow(),
  updatedAt: timestamp('updated_at').notNull().defaultNow(),
});

// Agents table
export const agents = pgTable('agents', {
  id: uuid('id').primaryKey().defaultRandom(),
  name: varchar('name', { length: 255 }).notNull(),
  type: agentTypeEnum('type').notNull(),
  status: agentStatusEnum('status').notNull().default('initializing'),
  description: text('description'),
  capabilities: jsonb('capabilities').$type<string[]>().notNull().default([]),
  configuration: jsonb('configuration').$type<Record<string, any>>().notNull().default({}),
  resourceLimits: jsonb('resource_limits').$type<Record<string, number>>().notNull().default({}),
  learningEnabled: boolean('learning_enabled').notNull().default(true),
  autoScale: boolean('auto_scale').notNull().default(false),
  healthScore: integer('health_score').notNull().default(100),
  cpuUsage: integer('cpu_usage').notNull().default(0),
  memoryUsage: integer('memory_usage').notNull().default(0),
  taskCount: integer('task_count').notNull().default(0),
  successRate: integer('success_rate').notNull().default(100),
  lastActivity: timestamp('last_activity'),
  createdBy: integer('created_by').references(() => users.id),
  createdAt: timestamp('created_at').notNull().defaultNow(),
  updatedAt: timestamp('updated_at').notNull().defaultNow(),
});

// Projects table
export const projects = pgTable('projects', {
  id: uuid('id').primaryKey().defaultRandom(),
  name: varchar('name', { length: 255 }).notNull(),
  description: text('description'),
  status: projectStatusEnum('status').notNull().default('planning'),
  priority: taskPriorityEnum('priority').notNull().default('medium'),
  progress: integer('progress').notNull().default(0),
  metadata: jsonb('metadata').$type<Record<string, any>>().notNull().default({}),
  ownerId: integer('owner_id').references(() => users.id).notNull(),
  createdAt: timestamp('created_at').notNull().defaultNow(),
  updatedAt: timestamp('updated_at').notNull().defaultNow(),
});

// Tasks table
export const tasks = pgTable('tasks', {
  id: uuid('id').primaryKey().defaultRandom(),
  agentId: uuid('agent_id').references(() => agents.id),
  projectId: uuid('project_id').references(() => projects.id),
  taskType: varchar('task_type', { length: 100 }).notNull(),
  status: taskStatusEnum('status').notNull().default('pending'),
  priority: taskPriorityEnum('priority').notNull().default('medium'),
  payload: jsonb('payload').$type<Record<string, any>>().notNull(),
  context: jsonb('context').$type<Record<string, any>>(),
  result: jsonb('result').$type<Record<string, any>>(),
  error: text('error'),
  timeout: integer('timeout').notNull().default(300),
  retryCount: integer('retry_count').notNull().default(0),
  maxRetries: integer('max_retries').notNull().default(3),
  requiresLearning: boolean('requires_learning').notNull().default(false),
  executionTime: integer('execution_time'),
  startedAt: timestamp('started_at'),
  completedAt: timestamp('completed_at'),
  createdBy: integer('created_by').references(() => users.id),
  createdAt: timestamp('created_at').notNull().defaultNow(),
  updatedAt: timestamp('updated_at').notNull().defaultNow(),
});

// Collaborations table
export const collaborations = pgTable('collaborations', {
  id: uuid('id').primaryKey().defaultRandom(),
  name: varchar('name', { length: 255 }).notNull(),
  description: text('description'),
  status: collaborationStatusEnum('status').notNull().default('active'),
  projectId: uuid('project_id').references(() => projects.id),
  participantAgents: jsonb('participant_agents').$type<string[]>().notNull().default([]),
  progress: integer('progress').notNull().default(0),
  metadata: jsonb('metadata').$type<Record<string, any>>().notNull().default({}),
  startedAt: timestamp('started_at'),
  completedAt: timestamp('completed_at'),
  createdBy: integer('created_by').references(() => users.id),
  createdAt: timestamp('created_at').notNull().defaultNow(),
  updatedAt: timestamp('updated_at').notNull().defaultNow(),
});

// Knowledge items table
export const knowledgeItems = pgTable('knowledge_items', {
  id: uuid('id').primaryKey().defaultRandom(),
  title: varchar('title', { length: 255 }).notNull(),
  content: text('content').notNull(),
  type: varchar('type', { length: 100 }).notNull(),
  tags: jsonb('tags').$type<string[]>().notNull().default([]),
  metadata: jsonb('metadata').$type<Record<string, any>>().notNull().default({}),
  sourceAgentId: uuid('source_agent_id').references(() => agents.id),
  projectId: uuid('project_id').references(() => projects.id),
  isPublic: boolean('is_public').notNull().default(false),
  createdBy: integer('created_by').references(() => users.id),
  createdAt: timestamp('created_at').notNull().defaultNow(),
  updatedAt: timestamp('updated_at').notNull().defaultNow(),
});

// Learning sessions table
export const learningSessions = pgTable('learning_sessions', {
  id: uuid('id').primaryKey().defaultRandom(),
  agentId: uuid('agent_id').references(() => agents.id).notNull(),
  sessionType: varchar('session_type', { length: 100 }).notNull(),
  data: jsonb('data').$type<Record<string, any>>().notNull(),
  insights: jsonb('insights').$type<Record<string, any>>().notNull().default({}),
  performance: integer('performance').notNull().default(0),
  duration: integer('duration').notNull().default(0),
  startedAt: timestamp('started_at').notNull(),
  completedAt: timestamp('completed_at'),
  createdAt: timestamp('created_at').notNull().defaultNow(),
});

// System metrics table
export const systemMetrics = pgTable('system_metrics', {
  id: serial('id').primaryKey(),
  metricType: varchar('metric_type', { length: 100 }).notNull(),
  value: integer('value').notNull(),
  metadata: jsonb('metadata').$type<Record<string, any>>().notNull().default({}),
  timestamp: timestamp('timestamp').notNull().defaultNow(),
});

// Relations
export const usersRelations = relations(users, ({ many }) => ({
  createdAgents: many(agents),
  ownedProjects: many(projects),
  createdTasks: many(tasks),
  createdCollaborations: many(collaborations),
  createdKnowledge: many(knowledgeItems),
}));

export const agentsRelations = relations(agents, ({ one, many }) => ({
  creator: one(users, {
    fields: [agents.createdBy],
    references: [users.id],
  }),
  tasks: many(tasks),
  knowledgeItems: many(knowledgeItems),
  learningSessions: many(learningSessions),
}));

export const projectsRelations = relations(projects, ({ one, many }) => ({
  owner: one(users, {
    fields: [projects.ownerId],
    references: [users.id],
  }),
  tasks: many(tasks),
  collaborations: many(collaborations),
  knowledgeItems: many(knowledgeItems),
}));

export const tasksRelations = relations(tasks, ({ one }) => ({
  agent: one(agents, {
    fields: [tasks.agentId],
    references: [agents.id],
  }),
  project: one(projects, {
    fields: [tasks.projectId],
    references: [projects.id],
  }),
  creator: one(users, {
    fields: [tasks.createdBy],
    references: [users.id],
  }),
}));

export const collaborationsRelations = relations(collaborations, ({ one }) => ({
  project: one(projects, {
    fields: [collaborations.projectId],
    references: [projects.id],
  }),
  creator: one(users, {
    fields: [collaborations.createdBy],
    references: [users.id],
  }),
}));

export const knowledgeItemsRelations = relations(knowledgeItems, ({ one }) => ({
  sourceAgent: one(agents, {
    fields: [knowledgeItems.sourceAgentId],
    references: [agents.id],
  }),
  project: one(projects, {
    fields: [knowledgeItems.projectId],
    references: [projects.id],
  }),
  creator: one(users, {
    fields: [knowledgeItems.createdBy],
    references: [users.id],
  }),
}));

export const learningSessionsRelations = relations(learningSessions, ({ one }) => ({
  agent: one(agents, {
    fields: [learningSessions.agentId],
    references: [agents.id],
  }),
}));

// Insert schemas
export const insertUserSchema = createInsertSchema(users).omit({ id: true, createdAt: true, updatedAt: true });
export const insertAgentSchema = createInsertSchema(agents).omit({ id: true, createdAt: true, updatedAt: true });
export const insertProjectSchema = createInsertSchema(projects).omit({ id: true, createdAt: true, updatedAt: true });
export const insertTaskSchema = createInsertSchema(tasks).omit({ id: true, createdAt: true, updatedAt: true });
export const insertCollaborationSchema = createInsertSchema(collaborations).omit({ id: true, createdAt: true, updatedAt: true });
export const insertKnowledgeItemSchema = createInsertSchema(knowledgeItems).omit({ id: true, createdAt: true, updatedAt: true });
export const insertLearningSessionSchema = createInsertSchema(learningSessions).omit({ id: true, createdAt: true });
export const insertSystemMetricSchema = createInsertSchema(systemMetrics).omit({ id: true, timestamp: true });

// Types
export type User = typeof users.$inferSelect;
export type InsertUser = z.infer<typeof insertUserSchema>;
export type Agent = typeof agents.$inferSelect;
export type InsertAgent = z.infer<typeof insertAgentSchema>;
export type Project = typeof projects.$inferSelect;
export type InsertProject = z.infer<typeof insertProjectSchema>;
export type Task = typeof tasks.$inferSelect;
export type InsertTask = z.infer<typeof insertTaskSchema>;
export type Collaboration = typeof collaborations.$inferSelect;
export type InsertCollaboration = z.infer<typeof insertCollaborationSchema>;
export type KnowledgeItem = typeof knowledgeItems.$inferSelect;
export type InsertKnowledgeItem = z.infer<typeof insertKnowledgeItemSchema>;
export type LearningSession = typeof learningSessions.$inferSelect;
export type InsertLearningSession = z.infer<typeof insertLearningSessionSchema>;
export type SystemMetric = typeof systemMetrics.$inferSelect;
export type InsertSystemMetric = z.infer<typeof insertSystemMetricSchema>;

// Additional types for API responses
export type AgentStatus = 'initializing' | 'active' | 'idle' | 'busy' | 'paused' | 'error' | 'shutdown';
export type AgentType = 'core' | 'specialized' | 'custom';
export type TaskStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
export type TaskPriority = 'low' | 'medium' | 'high' | 'urgent';
export type CollaborationStatus = 'active' | 'paused' | 'completed' | 'cancelled';
export type ProjectStatus = 'planning' | 'active' | 'paused' | 'completed' | 'cancelled';
