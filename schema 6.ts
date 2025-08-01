import { sql } from 'drizzle-orm';
import {
  index,
  jsonb,
  pgTable,
  timestamp,
  varchar,
  text,
  integer,
  boolean,
  real,
  primaryKey,
} from "drizzle-orm/pg-core";
import { relations } from 'drizzle-orm';
import { createInsertSchema } from 'drizzle-zod';
import { z } from 'zod';

// Session storage table (required for Replit Auth)
export const sessions = pgTable(
  "sessions",
  {
    sid: varchar("sid").primaryKey(),
    sess: jsonb("sess").notNull(),
    expire: timestamp("expire").notNull(),
  },
  (table) => [index("IDX_session_expire").on(table.expire)],
);

// User storage table (required for Replit Auth)
export const users = pgTable("users", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  email: varchar("email").unique(),
  firstName: varchar("first_name"),
  lastName: varchar("last_name"),
  profileImageUrl: varchar("profile_image_url"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Projects table
export const projects = pgTable("projects", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: varchar("name", { length: 255 }).notNull(),
  description: text("description"),
  userId: varchar("user_id").notNull(),
  status: varchar("status", { length: 50 }).default("active"),
  isShared: boolean("is_shared").default(false),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Files table
export const files = pgTable("files", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: varchar("name", { length: 255 }).notNull(),
  originalName: varchar("original_name", { length: 255 }).notNull(),
  mimeType: varchar("mime_type", { length: 100 }),
  size: integer("size").notNull(),
  path: varchar("path", { length: 500 }).notNull(),
  projectId: varchar("project_id"),
  userId: varchar("user_id").notNull(),
  isShared: boolean("is_shared").default(false),
  downloadCount: integer("download_count").default(0),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Chat messages table
export const messages = pgTable("messages", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  content: text("content").notNull(),
  userId: varchar("user_id").notNull(),
  projectId: varchar("project_id"),
  type: varchar("type", { length: 50 }).default("text"), // text, file, system
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow(),
});

// AI Agents table
export const agents = pgTable("agents", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: varchar("name", { length: 255 }).notNull(),
  type: varchar("type", { length: 100 }).notNull(), // code_analyzer, security_scanner, etc.
  description: text("description"),
  status: varchar("status", { length: 50 }).default("idle"), // active, idle, learning, error
  config: jsonb("config"),
  healthScore: real("health_score").default(100),
  cpuUsage: real("cpu_usage").default(0),
  memoryUsage: real("memory_usage").default(0),
  taskCount: integer("task_count").default(0),
  successRate: real("success_rate").default(100),
  lastActivity: timestamp("last_activity").defaultNow(),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Agent tasks table
export const agentTasks = pgTable("agent_tasks", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  agentId: varchar("agent_id").notNull(),
  task: text("task").notNull(),
  context: jsonb("context"),
  status: varchar("status", { length: 50 }).default("pending"), // pending, running, completed, failed
  result: jsonb("result"),
  processingTime: real("processing_time"),
  tokensUsed: integer("tokens_used"),
  confidenceLevel: varchar("confidence_level", { length: 50 }),
  createdAt: timestamp("created_at").defaultNow(),
  completedAt: timestamp("completed_at"),
});

// Knowledge graph nodes
export const knowledgeNodes = pgTable("knowledge_nodes", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  type: varchar("type", { length: 100 }).notNull(),
  data: jsonb("data").notNull(),
  embedding: text("embedding"), // Vector embedding as text
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Knowledge graph relationships
export const knowledgeRelationships = pgTable("knowledge_relationships", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  fromNodeId: varchar("from_node_id").notNull(),
  toNodeId: varchar("to_node_id").notNull(),
  relationshipType: varchar("relationship_type", { length: 100 }).notNull(),
  strength: real("strength").default(1.0),
  metadata: jsonb("metadata"),
  createdAt: timestamp("created_at").defaultNow(),
});

// System metrics table
export const systemMetrics = pgTable("system_metrics", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  metricType: varchar("metric_type", { length: 100 }).notNull(),
  value: real("value").notNull(),
  unit: varchar("unit", { length: 50 }),
  metadata: jsonb("metadata"),
  timestamp: timestamp("timestamp").defaultNow(),
});

// WebSocket connections table
export const webSocketConnections = pgTable("websocket_connections", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull(),
  connectionId: varchar("connection_id").notNull(),
  status: varchar("status", { length: 50 }).default("connected"),
  lastPing: timestamp("last_ping").defaultNow(),
  createdAt: timestamp("created_at").defaultNow(),
});

// Define relations
export const usersRelations = relations(users, ({ many }) => ({
  projects: many(projects),
  files: many(files),
  messages: many(messages),
  webSocketConnections: many(webSocketConnections),
}));

export const projectsRelations = relations(projects, ({ one, many }) => ({
  user: one(users, {
    fields: [projects.userId],
    references: [users.id],
  }),
  files: many(files),
  messages: many(messages),
}));

export const filesRelations = relations(files, ({ one }) => ({
  user: one(users, {
    fields: [files.userId],
    references: [users.id],
  }),
  project: one(projects, {
    fields: [files.projectId],
    references: [projects.id],
  }),
}));

export const messagesRelations = relations(messages, ({ one }) => ({
  user: one(users, {
    fields: [messages.userId],
    references: [users.id],
  }),
  project: one(projects, {
    fields: [messages.projectId],
    references: [projects.id],
  }),
}));

export const agentsRelations = relations(agents, ({ many }) => ({
  tasks: many(agentTasks),
}));

export const agentTasksRelations = relations(agentTasks, ({ one }) => ({
  agent: one(agents, {
    fields: [agentTasks.agentId],
    references: [agents.id],
  }),
}));

export const knowledgeRelationshipsRelations = relations(knowledgeRelationships, ({ one }) => ({
  fromNode: one(knowledgeNodes, {
    fields: [knowledgeRelationships.fromNodeId],
    references: [knowledgeNodes.id],
  }),
  toNode: one(knowledgeNodes, {
    fields: [knowledgeRelationships.toNodeId],
    references: [knowledgeNodes.id],
  }),
}));

export const webSocketConnectionsRelations = relations(webSocketConnections, ({ one }) => ({
  user: one(users, {
    fields: [webSocketConnections.userId],
    references: [users.id],
  }),
}));

// Types
export type UpsertUser = typeof users.$inferInsert;
export type User = typeof users.$inferSelect;

export type InsertProject = typeof projects.$inferInsert;
export type Project = typeof projects.$inferSelect;

export type InsertFile = typeof files.$inferInsert;
export type File = typeof files.$inferSelect;

export type InsertMessage = typeof messages.$inferInsert;
export type Message = typeof messages.$inferSelect;

export type InsertAgent = typeof agents.$inferInsert;
export type Agent = typeof agents.$inferSelect;
export type AgentStatus = 'active' | 'idle' | 'learning' | 'error';

export type InsertAgentTask = typeof agentTasks.$inferInsert;
export type AgentTask = typeof agentTasks.$inferSelect;

export type InsertKnowledgeNode = typeof knowledgeNodes.$inferInsert;
export type KnowledgeNode = typeof knowledgeNodes.$inferSelect;

export type InsertKnowledgeRelationship = typeof knowledgeRelationships.$inferInsert;
export type KnowledgeRelationship = typeof knowledgeRelationships.$inferSelect;

export type InsertSystemMetric = typeof systemMetrics.$inferInsert;
export type SystemMetric = typeof systemMetrics.$inferSelect;

export type InsertWebSocketConnection = typeof webSocketConnections.$inferInsert;
export type WebSocketConnection = typeof webSocketConnections.$inferSelect;

// Zod schemas for validation
export const insertProjectSchema = createInsertSchema(projects).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertFileSchema = createInsertSchema(files).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertMessageSchema = createInsertSchema(messages).omit({
  id: true,
  createdAt: true,
});

export const insertAgentSchema = createInsertSchema(agents).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertAgentTaskSchema = createInsertSchema(agentTasks).omit({
  id: true,
  createdAt: true,
  completedAt: true,
});

export const insertKnowledgeNodeSchema = createInsertSchema(knowledgeNodes).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertSystemMetricSchema = createInsertSchema(systemMetrics).omit({
  id: true,
  timestamp: true,
});
