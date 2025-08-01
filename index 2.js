// server/index.ts
import express from "express";
import { createServer } from "http";
import { WebSocketServer } from "ws";

// server/routes.ts
import { Router } from "express";
import { z as z2 } from "zod";

// shared/schema.ts
import { z } from "zod";
var AgentStatus = z.enum(["active", "learning", "idle", "error", "maintenance"]);
var AgentType = z.enum([
  "editing",
  "enhancement",
  "monitoring",
  "orchestration",
  "project",
  "validation",
  "examination",
  "learning_engine"
]);
var LearningActivityType = z.enum([
  "pattern_discovery",
  "knowledge_consolidation",
  "inter_agent_transfer",
  "performance_optimization",
  "error_correction"
]);
var AgentSchema = z.object({
  id: z.string(),
  name: z.string(),
  type: AgentType,
  status: AgentStatus,
  description: z.string(),
  capabilities: z.array(z.string()),
  cpu_usage: z.number().min(0).max(100),
  memory_usage: z.number().min(0).max(100),
  created_at: z.string().datetime(),
  updated_at: z.string().datetime(),
  last_activity: z.string().datetime(),
  config: z.record(z.any()).optional(),
  health_score: z.number().min(0).max(100),
  task_count: z.number().min(0),
  success_rate: z.number().min(0).max(100)
});
var insertAgentSchema = AgentSchema.omit({ id: true, created_at: true, updated_at: true });
var LearningMetricsSchema = z.object({
  id: z.string(),
  agent_id: z.string(),
  pattern_accuracy: z.number().min(0).max(100),
  knowledge_growth: z.number(),
  learning_rate: z.number().min(0),
  consolidation_score: z.number().min(0).max(100),
  transfer_success_rate: z.number().min(0).max(100),
  created_at: z.string().datetime(),
  updated_at: z.string().datetime()
});
var insertLearningMetricsSchema = LearningMetricsSchema.omit({ id: true, created_at: true, updated_at: true });
var LearningActivitySchema = z.object({
  id: z.string(),
  agent_id: z.string(),
  activity_type: LearningActivityType,
  description: z.string(),
  impact_score: z.number().min(0).max(100),
  accuracy_improvement: z.number().optional(),
  knowledge_units: z.number().optional(),
  success_rate: z.number().min(0).max(100).optional(),
  created_at: z.string().datetime()
});
var insertLearningActivitySchema = LearningActivitySchema.omit({ id: true, created_at: true });
var KnowledgeNodeSchema = z.object({
  id: z.string(),
  label: z.string(),
  type: z.string(),
  properties: z.record(z.any()),
  agent_id: z.string().optional(),
  created_at: z.string().datetime(),
  updated_at: z.string().datetime()
});
var insertKnowledgeNodeSchema = KnowledgeNodeSchema.omit({ id: true, created_at: true, updated_at: true });
var KnowledgeRelationshipSchema = z.object({
  id: z.string(),
  source_id: z.string(),
  target_id: z.string(),
  relationship_type: z.string(),
  properties: z.record(z.any()),
  strength: z.number().min(0).max(1),
  created_at: z.string().datetime()
});
var insertKnowledgeRelationshipSchema = KnowledgeRelationshipSchema.omit({ id: true, created_at: true });
var SystemMetricsSchema = z.object({
  id: z.string(),
  cpu_usage: z.number().min(0).max(100),
  memory_usage: z.number().min(0).max(100),
  network_latency: z.number().min(0),
  storage_used: z.number().min(0),
  storage_total: z.number().min(0),
  active_agents: z.number().min(0),
  tasks_completed: z.number().min(0),
  error_count: z.number().min(0),
  warning_count: z.number().min(0),
  created_at: z.string().datetime()
});
var insertSystemMetricsSchema = SystemMetricsSchema.omit({ id: true, created_at: true });
var ActivityLogSchema = z.object({
  id: z.string(),
  agent_id: z.string().optional(),
  agent_type: AgentType.optional(),
  message: z.string(),
  level: z.enum(["info", "warning", "error", "success"]),
  details: z.record(z.any()).optional(),
  created_at: z.string().datetime()
});
var insertActivityLogSchema = ActivityLogSchema.omit({ id: true, created_at: true });
var ErrorLogSchema = z.object({
  id: z.string(),
  agent_id: z.string().optional(),
  error_type: z.string(),
  message: z.string(),
  stack_trace: z.string().optional(),
  severity: z.enum(["low", "medium", "high", "critical"]),
  resolved: z.boolean().default(false),
  resolution_notes: z.string().optional(),
  created_at: z.string().datetime(),
  resolved_at: z.string().datetime().optional()
});
var insertErrorLogSchema = ErrorLogSchema.omit({ id: true, created_at: true });
var TaskSchema = z.object({
  id: z.string(),
  agent_id: z.string(),
  title: z.string(),
  description: z.string(),
  status: z.enum(["pending", "running", "completed", "failed", "cancelled"]),
  priority: z.enum(["low", "medium", "high", "urgent"]),
  progress: z.number().min(0).max(100),
  started_at: z.string().datetime().optional(),
  completed_at: z.string().datetime().optional(),
  created_at: z.string().datetime(),
  updated_at: z.string().datetime()
});
var insertTaskSchema = TaskSchema.omit({ id: true, created_at: true, updated_at: true });
var WebSocketMessageSchema = z.object({
  type: z.enum(["agent_update", "system_metrics", "activity_log", "error_log", "learning_update"]),
  data: z.any(),
  timestamp: z.string().datetime()
});
var ApiResponseSchema = z.object({
  success: z.boolean(),
  data: z.any().optional(),
  error: z.string().optional(),
  timestamp: z.string().datetime()
});
var DashboardDataSchema = z.object({
  agents: z.array(AgentSchema),
  system_metrics: SystemMetricsSchema,
  learning_metrics: z.array(LearningMetricsSchema),
  recent_activities: z.array(ActivityLogSchema),
  error_summary: z.object({
    critical: z.number(),
    high: z.number(),
    medium: z.number(),
    low: z.number()
  }),
  knowledge_graph_stats: z.object({
    node_count: z.number(),
    relationship_count: z.number(),
    last_update: z.string().datetime()
  })
});

// server/storage.ts
import { randomUUID } from "crypto";
var MemStorage = class {
  agents = [];
  learningMetrics = [];
  learningActivities = [];
  knowledgeNodes = [];
  knowledgeRelationships = [];
  systemMetrics = [];
  activityLogs = [];
  errorLogs = [];
  tasks = [];
  constructor() {
    this.initializeData();
  }
  initializeData() {
    const now = (/* @__PURE__ */ new Date()).toISOString();
    const agentTypes = [
      {
        type: "editing",
        name: "Editing Agent",
        description: "Content modification and optimization with contextual understanding",
        capabilities: ["content_editing", "grammar_correction", "style_optimization"]
      },
      {
        type: "enhancement",
        name: "Enhancement Agent",
        description: "Quality improvement and feature augmentation capabilities",
        capabilities: ["performance_optimization", "feature_enhancement", "quality_improvement"]
      },
      {
        type: "monitoring",
        name: "Monitoring Agent",
        description: "Real-time system health and performance tracking",
        capabilities: ["health_monitoring", "performance_tracking", "alerting"]
      },
      {
        type: "orchestration",
        name: "Orchestration Agent",
        description: "Task coordination and workflow management",
        capabilities: ["task_coordination", "workflow_management", "resource_allocation"]
      },
      {
        type: "project",
        name: "Project Agent",
        description: "Project lifecycle management and resource allocation",
        capabilities: ["project_management", "resource_planning", "timeline_tracking"]
      },
      {
        type: "validation",
        name: "Validation Agent",
        description: "Quality assurance and compliance verification",
        capabilities: ["quality_assurance", "compliance_checking", "validation_testing"]
      },
      {
        type: "examination",
        name: "Examination Agent",
        description: "Deep analysis and pattern recognition capabilities",
        capabilities: ["pattern_recognition", "deep_analysis", "anomaly_detection"]
      },
      {
        type: "learning_engine",
        name: "Learning Engine",
        description: "Continuous learning and knowledge synthesis",
        capabilities: ["machine_learning", "knowledge_synthesis", "pattern_learning"]
      }
    ];
    this.agents = agentTypes.map((agentType) => ({
      id: randomUUID(),
      name: agentType.name,
      type: agentType.type,
      status: ["active", "learning", "idle"][Math.floor(Math.random() * 3)],
      description: agentType.description,
      capabilities: agentType.capabilities,
      cpu_usage: Math.floor(Math.random() * 80) + 10,
      memory_usage: Math.floor(Math.random() * 70) + 20,
      created_at: now,
      updated_at: now,
      last_activity: now,
      health_score: Math.floor(Math.random() * 20) + 80,
      task_count: Math.floor(Math.random() * 50) + 10,
      success_rate: Math.floor(Math.random() * 15) + 85
    }));
    this.systemMetrics.push({
      id: randomUUID(),
      cpu_usage: 34,
      memory_usage: 67,
      network_latency: 12,
      storage_used: 23e12,
      // 23TB in bytes
      storage_total: 5e13,
      // 50TB in bytes
      active_agents: this.agents.filter((a) => a.status === "active").length,
      tasks_completed: 24e5,
      error_count: 2,
      warning_count: 7,
      created_at: now
    });
  }
  async getAgents() {
    return [...this.agents];
  }
  async getAgent(id) {
    return this.agents.find((agent) => agent.id === id) || null;
  }
  async createAgent(agent) {
    const now = (/* @__PURE__ */ new Date()).toISOString();
    const newAgent = {
      id: randomUUID(),
      ...agent,
      created_at: now,
      updated_at: now
    };
    this.agents.push(newAgent);
    return newAgent;
  }
  async updateAgent(id, updates) {
    const index = this.agents.findIndex((agent) => agent.id === id);
    if (index === -1) return null;
    const now = (/* @__PURE__ */ new Date()).toISOString();
    this.agents[index] = {
      ...this.agents[index],
      ...updates,
      updated_at: now
    };
    return this.agents[index];
  }
  async deleteAgent(id) {
    const index = this.agents.findIndex((agent) => agent.id === id);
    if (index === -1) return false;
    this.agents.splice(index, 1);
    return true;
  }
  async getLearningMetrics(agentId) {
    return agentId ? this.learningMetrics.filter((m) => m.agent_id === agentId) : [...this.learningMetrics];
  }
  async createLearningMetrics(metrics) {
    const now = (/* @__PURE__ */ new Date()).toISOString();
    const newMetrics = {
      id: randomUUID(),
      ...metrics,
      created_at: now,
      updated_at: now
    };
    this.learningMetrics.push(newMetrics);
    return newMetrics;
  }
  async updateLearningMetrics(id, updates) {
    const index = this.learningMetrics.findIndex((m) => m.id === id);
    if (index === -1) return null;
    const now = (/* @__PURE__ */ new Date()).toISOString();
    this.learningMetrics[index] = {
      ...this.learningMetrics[index],
      ...updates,
      updated_at: now
    };
    return this.learningMetrics[index];
  }
  async getLearningActivities(agentId, limit) {
    let activities = agentId ? this.learningActivities.filter((a) => a.agent_id === agentId) : [...this.learningActivities];
    activities.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
    return limit ? activities.slice(0, limit) : activities;
  }
  async createLearningActivity(activity) {
    const now = (/* @__PURE__ */ new Date()).toISOString();
    const newActivity = {
      id: randomUUID(),
      ...activity,
      created_at: now
    };
    this.learningActivities.push(newActivity);
    return newActivity;
  }
  async getKnowledgeNodes() {
    return [...this.knowledgeNodes];
  }
  async createKnowledgeNode(node) {
    const now = (/* @__PURE__ */ new Date()).toISOString();
    const newNode = {
      id: randomUUID(),
      ...node,
      created_at: now,
      updated_at: now
    };
    this.knowledgeNodes.push(newNode);
    return newNode;
  }
  async getKnowledgeRelationships() {
    return [...this.knowledgeRelationships];
  }
  async createKnowledgeRelationship(relationship) {
    const now = (/* @__PURE__ */ new Date()).toISOString();
    const newRelationship = {
      id: randomUUID(),
      ...relationship,
      created_at: now
    };
    this.knowledgeRelationships.push(newRelationship);
    return newRelationship;
  }
  async getSystemMetrics(limit) {
    const sorted = [...this.systemMetrics].sort(
      (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    );
    return limit ? sorted.slice(0, limit) : sorted;
  }
  async createSystemMetrics(metrics) {
    const now = (/* @__PURE__ */ new Date()).toISOString();
    const newMetrics = {
      id: randomUUID(),
      ...metrics,
      created_at: now
    };
    this.systemMetrics.push(newMetrics);
    return newMetrics;
  }
  async getLatestSystemMetrics() {
    const metrics = await this.getSystemMetrics(1);
    return metrics[0] || null;
  }
  async getActivityLogs(limit, agentId) {
    let logs = agentId ? this.activityLogs.filter((log) => log.agent_id === agentId) : [...this.activityLogs];
    logs.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
    return limit ? logs.slice(0, limit) : logs;
  }
  async createActivityLog(log) {
    const now = (/* @__PURE__ */ new Date()).toISOString();
    const newLog = {
      id: randomUUID(),
      ...log,
      created_at: now
    };
    this.activityLogs.push(newLog);
    return newLog;
  }
  async getErrorLogs(limit, resolved) {
    let logs = [...this.errorLogs];
    if (resolved !== void 0) {
      logs = logs.filter((log) => log.resolved === resolved);
    }
    logs.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
    return limit ? logs.slice(0, limit) : logs;
  }
  async createErrorLog(error) {
    const now = (/* @__PURE__ */ new Date()).toISOString();
    const newError = {
      id: randomUUID(),
      ...error,
      created_at: now
    };
    this.errorLogs.push(newError);
    return newError;
  }
  async resolveError(id, resolutionNotes) {
    const index = this.errorLogs.findIndex((error) => error.id === id);
    if (index === -1) return null;
    const now = (/* @__PURE__ */ new Date()).toISOString();
    this.errorLogs[index] = {
      ...this.errorLogs[index],
      resolved: true,
      resolution_notes: resolutionNotes,
      resolved_at: now
    };
    return this.errorLogs[index];
  }
  async getTasks(agentId) {
    return agentId ? this.tasks.filter((task) => task.agent_id === agentId) : [...this.tasks];
  }
  async createTask(task) {
    const now = (/* @__PURE__ */ new Date()).toISOString();
    const newTask = {
      id: randomUUID(),
      ...task,
      created_at: now,
      updated_at: now
    };
    this.tasks.push(newTask);
    return newTask;
  }
  async updateTask(id, updates) {
    const index = this.tasks.findIndex((task) => task.id === id);
    if (index === -1) return null;
    const now = (/* @__PURE__ */ new Date()).toISOString();
    this.tasks[index] = {
      ...this.tasks[index],
      ...updates,
      updated_at: now
    };
    return this.tasks[index];
  }
  async getDashboardData() {
    const agents = await this.getAgents();
    const systemMetrics = await this.getLatestSystemMetrics();
    const learningMetrics = await this.getLearningMetrics();
    const recentActivities = await this.getActivityLogs(10);
    const errorLogs = await this.getErrorLogs();
    const errorSummary = {
      critical: errorLogs.filter((e) => e.severity === "critical" && !e.resolved).length,
      high: errorLogs.filter((e) => e.severity === "high" && !e.resolved).length,
      medium: errorLogs.filter((e) => e.severity === "medium" && !e.resolved).length,
      low: errorLogs.filter((e) => e.severity === "low" && !e.resolved).length
    };
    return {
      agents,
      system_metrics: systemMetrics || {
        id: randomUUID(),
        cpu_usage: 0,
        memory_usage: 0,
        network_latency: 0,
        storage_used: 0,
        storage_total: 0,
        active_agents: 0,
        tasks_completed: 0,
        error_count: 0,
        warning_count: 0,
        created_at: (/* @__PURE__ */ new Date()).toISOString()
      },
      learning_metrics: learningMetrics,
      recent_activities: recentActivities,
      error_summary: errorSummary,
      knowledge_graph_stats: {
        node_count: this.knowledgeNodes.length,
        relationship_count: this.knowledgeRelationships.length,
        last_update: (/* @__PURE__ */ new Date()).toISOString()
      }
    };
  }
};
var storage = new MemStorage();

// server/services/agents.ts
var AgentService = class {
  async getAllAgents() {
    return await storage.getAgents();
  }
  async getAgent(id) {
    return await storage.getAgent(id);
  }
  async createAgent(agentData) {
    return await storage.createAgent(agentData);
  }
  async updateAgent(id, updates) {
    return await storage.updateAgent(id, updates);
  }
  async updateAgentStatus(id, status) {
    return await storage.updateAgent(id, {
      status,
      last_activity: (/* @__PURE__ */ new Date()).toISOString()
    });
  }
  async deleteAgent(id) {
    return await storage.deleteAgent(id);
  }
  async getAgentHealth(id) {
    const agent = await storage.getAgent(id);
    if (!agent) return null;
    return {
      health_score: agent.health_score,
      cpu_usage: agent.cpu_usage,
      memory_usage: agent.memory_usage
    };
  }
  async updateAgentMetrics(id, cpu_usage, memory_usage) {
    const health_score = this.calculateHealthScore(cpu_usage, memory_usage);
    return await storage.updateAgent(id, {
      cpu_usage,
      memory_usage,
      health_score,
      last_activity: (/* @__PURE__ */ new Date()).toISOString()
    });
  }
  calculateHealthScore(cpu_usage, memory_usage) {
    const cpuScore = Math.max(0, 100 - cpu_usage);
    const memoryScore = Math.max(0, 100 - memory_usage);
    return Math.round((cpuScore + memoryScore) / 2);
  }
  async getAgentsByStatus(status) {
    const agents = await storage.getAgents();
    return agents.filter((agent) => agent.status === status);
  }
  async getAgentTaskLoad(id) {
    const agent = await storage.getAgent(id);
    if (!agent) return null;
    return {
      task_count: agent.task_count,
      success_rate: agent.success_rate
    };
  }
};
var agentService = new AgentService();

// server/services/learning.ts
var LearningService = class {
  async getLearningMetrics(agentId) {
    return await storage.getLearningMetrics(agentId);
  }
  async createLearningMetrics(metrics) {
    return await storage.createLearningMetrics(metrics);
  }
  async updateLearningMetrics(id, updates) {
    return await storage.updateLearningMetrics(id, updates);
  }
  async getLearningActivities(agentId, limit) {
    return await storage.getLearningActivities(agentId, limit);
  }
  async createLearningActivity(activity) {
    return await storage.createLearningActivity(activity);
  }
  async recordPatternDiscovery(agentId, description, accuracyImprovement) {
    return await this.createLearningActivity({
      agent_id: agentId,
      activity_type: "pattern_discovery",
      description,
      impact_score: Math.min(100, accuracyImprovement * 10),
      accuracy_improvement: accuracyImprovement
    });
  }
  async recordKnowledgeConsolidation(agentId, description, knowledgeUnits) {
    return await this.createLearningActivity({
      agent_id: agentId,
      activity_type: "knowledge_consolidation",
      description,
      impact_score: Math.min(100, knowledgeUnits / 10),
      knowledge_units: knowledgeUnits
    });
  }
  async recordInterAgentTransfer(sourceAgentId, targetAgentId, successRate) {
    return await this.createLearningActivity({
      agent_id: sourceAgentId,
      activity_type: "inter_agent_transfer",
      description: `Knowledge transfer to agent ${targetAgentId}`,
      impact_score: successRate,
      success_rate: successRate
    });
  }
  async getAgentLearningProgress(agentId) {
    const activities = await storage.getLearningActivities(agentId);
    return {
      total_activities: activities.length,
      pattern_discoveries: activities.filter((a) => a.activity_type === "pattern_discovery").length,
      knowledge_consolidations: activities.filter((a) => a.activity_type === "knowledge_consolidation").length,
      avg_impact_score: activities.length > 0 ? activities.reduce((sum, a) => sum + a.impact_score, 0) / activities.length : 0
    };
  }
  async calculateOverallLearningMetrics() {
    const allMetrics = await storage.getLearningMetrics();
    const allActivities = await storage.getLearningActivities();
    if (allMetrics.length === 0) {
      return {
        total_pattern_accuracy: 0,
        avg_knowledge_growth: 0,
        total_activities: 0,
        most_active_agent: null
      };
    }
    const totalPatternAccuracy = allMetrics.reduce((sum, m) => sum + m.pattern_accuracy, 0) / allMetrics.length;
    const avgKnowledgeGrowth = allMetrics.reduce((sum, m) => sum + m.knowledge_growth, 0) / allMetrics.length;
    const agentActivityCounts = allActivities.reduce((acc, activity) => {
      acc[activity.agent_id] = (acc[activity.agent_id] || 0) + 1;
      return acc;
    }, {});
    const mostActiveAgent = Object.entries(agentActivityCounts).length > 0 ? Object.entries(agentActivityCounts).reduce((a, b) => a[1] > b[1] ? a : b)[0] : null;
    return {
      total_pattern_accuracy: Math.round(totalPatternAccuracy * 10) / 10,
      avg_knowledge_growth: Math.round(avgKnowledgeGrowth * 10) / 10,
      total_activities: allActivities.length,
      most_active_agent: mostActiveAgent
    };
  }
};
var learningService = new LearningService();

// server/services/monitoring.ts
var MonitoringService = class {
  async getSystemMetrics(limit) {
    return await storage.getSystemMetrics(limit);
  }
  async getLatestSystemMetrics() {
    return await storage.getLatestSystemMetrics();
  }
  async createSystemMetrics(metrics) {
    return await storage.createSystemMetrics(metrics);
  }
  async getActivityLogs(limit, agentId) {
    return await storage.getActivityLogs(limit, agentId);
  }
  async createActivityLog(log) {
    return await storage.createActivityLog(log);
  }
  async getErrorLogs(limit, resolved) {
    return await storage.getErrorLogs(limit, resolved);
  }
  async createErrorLog(error) {
    return await storage.createErrorLog(error);
  }
  async resolveError(id, resolutionNotes) {
    return await storage.resolveError(id, resolutionNotes);
  }
  async logAgentActivity(agentId, agentType, message, level = "info") {
    return await this.createActivityLog({
      agent_id: agentId,
      agent_type: agentType,
      message,
      level
    });
  }
  async logSystemError(errorType, message, severity, agentId, stackTrace) {
    return await this.createErrorLog({
      agent_id: agentId,
      error_type: errorType,
      message,
      severity,
      stack_trace: stackTrace
    });
  }
  async getSystemHealth() {
    const latestMetrics = await this.getLatestSystemMetrics();
    const errorLogs = await this.getErrorLogs(100, false);
    if (!latestMetrics) {
      return {
        overall_health: "poor",
        cpu_status: "normal",
        memory_status: "normal",
        error_rate: "low",
        active_agents: 0
      };
    }
    let cpu_status = "normal";
    if (latestMetrics.cpu_usage > 80) cpu_status = "critical";
    else if (latestMetrics.cpu_usage > 60) cpu_status = "high";
    let memory_status = "normal";
    if (latestMetrics.memory_usage > 85) memory_status = "critical";
    else if (latestMetrics.memory_usage > 70) memory_status = "high";
    const criticalErrors = errorLogs.filter((e) => e.severity === "critical").length;
    const highErrors = errorLogs.filter((e) => e.severity === "high").length;
    let error_rate = "low";
    if (criticalErrors > 0 || highErrors > 5) error_rate = "high";
    else if (highErrors > 2 || errorLogs.length > 10) error_rate = "medium";
    let overall_health = "excellent";
    if (cpu_status === "critical" || memory_status === "critical" || error_rate === "high") {
      overall_health = "poor";
    } else if (cpu_status === "high" || memory_status === "high" || error_rate === "medium") {
      overall_health = "fair";
    } else if (cpu_status === "high" || memory_status === "high" || errorLogs.length > 5) {
      overall_health = "good";
    }
    return {
      overall_health,
      cpu_status,
      memory_status,
      error_rate,
      active_agents: latestMetrics.active_agents
    };
  }
  async generateSystemReport() {
    const metrics = await this.getSystemMetrics(24);
    const errorLogs = await this.getErrorLogs();
    const performanceMetrics = metrics.length > 0 ? {
      avg_cpu_usage: Math.round(metrics.reduce((sum, m) => sum + m.cpu_usage, 0) / metrics.length * 10) / 10,
      avg_memory_usage: Math.round(metrics.reduce((sum, m) => sum + m.memory_usage, 0) / metrics.length * 10) / 10,
      avg_network_latency: Math.round(metrics.reduce((sum, m) => sum + m.network_latency, 0) / metrics.length * 10) / 10
    } : {
      avg_cpu_usage: 0,
      avg_memory_usage: 0,
      avg_network_latency: 0
    };
    const errorSummary = {
      critical: errorLogs.filter((e) => e.severity === "critical").length,
      high: errorLogs.filter((e) => e.severity === "high").length,
      medium: errorLogs.filter((e) => e.severity === "medium").length,
      low: errorLogs.filter((e) => e.severity === "low").length
    };
    const uptime_percentage = 99.9;
    const latestMetrics = await this.getLatestSystemMetrics();
    const total_tasks_completed = latestMetrics?.tasks_completed || 0;
    return {
      uptime_percentage,
      total_tasks_completed,
      error_summary,
      performance_metrics
    };
  }
};
var monitoringService = new MonitoringService();

// server/services/websocket.ts
import { WebSocket } from "ws";
var WebSocketService = class {
  clients = /* @__PURE__ */ new Set();
  messageQueue = [];
  addClient(ws) {
    this.clients.add(ws);
    ws.on("close", () => {
      this.clients.delete(ws);
    });
    ws.on("error", (error) => {
      console.error("WebSocket error:", error);
      this.clients.delete(ws);
    });
    this.messageQueue.forEach((message) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(message));
      }
    });
  }
  broadcast(message) {
    const messageString = JSON.stringify(message);
    this.messageQueue.push(message);
    if (this.messageQueue.length > 100) {
      this.messageQueue.shift();
    }
    this.clients.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        try {
          client.send(messageString);
        } catch (error) {
          console.error("Error sending WebSocket message:", error);
          this.clients.delete(client);
        }
      } else {
        this.clients.delete(client);
      }
    });
  }
  broadcastAgentUpdate(agentData) {
    this.broadcast({
      type: "agent_update",
      data: agentData,
      timestamp: (/* @__PURE__ */ new Date()).toISOString()
    });
  }
  broadcastSystemMetrics(metrics) {
    this.broadcast({
      type: "system_metrics",
      data: metrics,
      timestamp: (/* @__PURE__ */ new Date()).toISOString()
    });
  }
  broadcastActivityLog(log) {
    this.broadcast({
      type: "activity_log",
      data: log,
      timestamp: (/* @__PURE__ */ new Date()).toISOString()
    });
  }
  broadcastErrorLog(error) {
    this.broadcast({
      type: "error_log",
      data: error,
      timestamp: (/* @__PURE__ */ new Date()).toISOString()
    });
  }
  broadcastLearningUpdate(learning) {
    this.broadcast({
      type: "learning_update",
      data: learning,
      timestamp: (/* @__PURE__ */ new Date()).toISOString()
    });
  }
  getClientCount() {
    return this.clients.size;
  }
  closeAllConnections() {
    this.clients.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        client.close();
      }
    });
    this.clients.clear();
  }
};
var webSocketService = new WebSocketService();

// server/routes.ts
var router = Router();
var createResponse = (success, data, error) => ({
  success,
  data,
  error,
  timestamp: (/* @__PURE__ */ new Date()).toISOString()
});
router.get("/api/agents", async (req, res) => {
  try {
    const agents = await agentService.getAllAgents();
    res.json(createResponse(true, agents));
  } catch (error) {
    console.error("Error fetching agents:", error);
    res.status(500).json(createResponse(false, null, "Failed to fetch agents"));
  }
});
router.get("/api/agents/:id", async (req, res) => {
  try {
    const agent = await agentService.getAgent(req.params.id);
    if (!agent) {
      return res.status(404).json(createResponse(false, null, "Agent not found"));
    }
    res.json(createResponse(true, agent));
  } catch (error) {
    console.error("Error fetching agent:", error);
    res.status(500).json(createResponse(false, null, "Failed to fetch agent"));
  }
});
router.post("/api/agents", async (req, res) => {
  try {
    const agentData = insertAgentSchema.parse(req.body);
    const agent = await agentService.createAgent(agentData);
    webSocketService.broadcastAgentUpdate(agent);
    res.status(201).json(createResponse(true, agent));
  } catch (error) {
    if (error instanceof z2.ZodError) {
      return res.status(400).json(createResponse(false, null, "Invalid agent data"));
    }
    console.error("Error creating agent:", error);
    res.status(500).json(createResponse(false, null, "Failed to create agent"));
  }
});
router.put("/api/agents/:id", async (req, res) => {
  try {
    const updates = req.body;
    const agent = await agentService.updateAgent(req.params.id, updates);
    if (!agent) {
      return res.status(404).json(createResponse(false, null, "Agent not found"));
    }
    webSocketService.broadcastAgentUpdate(agent);
    res.json(createResponse(true, agent));
  } catch (error) {
    console.error("Error updating agent:", error);
    res.status(500).json(createResponse(false, null, "Failed to update agent"));
  }
});
router.delete("/api/agents/:id", async (req, res) => {
  try {
    const success = await agentService.deleteAgent(req.params.id);
    if (!success) {
      return res.status(404).json(createResponse(false, null, "Agent not found"));
    }
    res.json(createResponse(true, { deleted: true }));
  } catch (error) {
    console.error("Error deleting agent:", error);
    res.status(500).json(createResponse(false, null, "Failed to delete agent"));
  }
});
router.get("/api/learning/metrics", async (req, res) => {
  try {
    const agentId = req.query.agentId;
    const metrics = await learningService.getLearningMetrics(agentId);
    res.json(createResponse(true, metrics));
  } catch (error) {
    console.error("Error fetching learning metrics:", error);
    res.status(500).json(createResponse(false, null, "Failed to fetch learning metrics"));
  }
});
router.post("/api/learning/metrics", async (req, res) => {
  try {
    const metricsData = insertLearningMetricsSchema.parse(req.body);
    const metrics = await learningService.createLearningMetrics(metricsData);
    webSocketService.broadcastLearningUpdate(metrics);
    res.status(201).json(createResponse(true, metrics));
  } catch (error) {
    if (error instanceof z2.ZodError) {
      return res.status(400).json(createResponse(false, null, "Invalid metrics data"));
    }
    console.error("Error creating learning metrics:", error);
    res.status(500).json(createResponse(false, null, "Failed to create learning metrics"));
  }
});
router.get("/api/learning/activities", async (req, res) => {
  try {
    const agentId = req.query.agentId;
    const limit = req.query.limit ? parseInt(req.query.limit) : void 0;
    const activities = await learningService.getLearningActivities(agentId, limit);
    res.json(createResponse(true, activities));
  } catch (error) {
    console.error("Error fetching learning activities:", error);
    res.status(500).json(createResponse(false, null, "Failed to fetch learning activities"));
  }
});
router.post("/api/learning/activities", async (req, res) => {
  try {
    const activityData = insertLearningActivitySchema.parse(req.body);
    const activity = await learningService.createLearningActivity(activityData);
    webSocketService.broadcastLearningUpdate(activity);
    res.status(201).json(createResponse(true, activity));
  } catch (error) {
    if (error instanceof z2.ZodError) {
      return res.status(400).json(createResponse(false, null, "Invalid activity data"));
    }
    console.error("Error creating learning activity:", error);
    res.status(500).json(createResponse(false, null, "Failed to create learning activity"));
  }
});
router.get("/api/learning/overview", async (req, res) => {
  try {
    const overview = await learningService.calculateOverallLearningMetrics();
    res.json(createResponse(true, overview));
  } catch (error) {
    console.error("Error fetching learning overview:", error);
    res.status(500).json(createResponse(false, null, "Failed to fetch learning overview"));
  }
});
router.get("/api/monitoring/system-metrics", async (req, res) => {
  try {
    const limit = req.query.limit ? parseInt(req.query.limit) : void 0;
    const metrics = await monitoringService.getSystemMetrics(limit);
    res.json(createResponse(true, metrics));
  } catch (error) {
    console.error("Error fetching system metrics:", error);
    res.status(500).json(createResponse(false, null, "Failed to fetch system metrics"));
  }
});
router.get("/api/monitoring/system-metrics/latest", async (req, res) => {
  try {
    const metrics = await monitoringService.getLatestSystemMetrics();
    res.json(createResponse(true, metrics));
  } catch (error) {
    console.error("Error fetching latest system metrics:", error);
    res.status(500).json(createResponse(false, null, "Failed to fetch latest system metrics"));
  }
});
router.post("/api/monitoring/system-metrics", async (req, res) => {
  try {
    const metricsData = insertSystemMetricsSchema.parse(req.body);
    const metrics = await monitoringService.createSystemMetrics(metricsData);
    webSocketService.broadcastSystemMetrics(metrics);
    res.status(201).json(createResponse(true, metrics));
  } catch (error) {
    if (error instanceof z2.ZodError) {
      return res.status(400).json(createResponse(false, null, "Invalid metrics data"));
    }
    console.error("Error creating system metrics:", error);
    res.status(500).json(createResponse(false, null, "Failed to create system metrics"));
  }
});
router.get("/api/monitoring/activity-logs", async (req, res) => {
  try {
    const limit = req.query.limit ? parseInt(req.query.limit) : void 0;
    const agentId = req.query.agentId;
    const logs = await monitoringService.getActivityLogs(limit, agentId);
    res.json(createResponse(true, logs));
  } catch (error) {
    console.error("Error fetching activity logs:", error);
    res.status(500).json(createResponse(false, null, "Failed to fetch activity logs"));
  }
});
router.post("/api/monitoring/activity-logs", async (req, res) => {
  try {
    const logData = insertActivityLogSchema.parse(req.body);
    const log = await monitoringService.createActivityLog(logData);
    webSocketService.broadcastActivityLog(log);
    res.status(201).json(createResponse(true, log));
  } catch (error) {
    if (error instanceof z2.ZodError) {
      return res.status(400).json(createResponse(false, null, "Invalid log data"));
    }
    console.error("Error creating activity log:", error);
    res.status(500).json(createResponse(false, null, "Failed to create activity log"));
  }
});
router.get("/api/monitoring/error-logs", async (req, res) => {
  try {
    const limit = req.query.limit ? parseInt(req.query.limit) : void 0;
    const resolved = req.query.resolved === "true" ? true : req.query.resolved === "false" ? false : void 0;
    const logs = await monitoringService.getErrorLogs(limit, resolved);
    res.json(createResponse(true, logs));
  } catch (error) {
    console.error("Error fetching error logs:", error);
    res.status(500).json(createResponse(false, null, "Failed to fetch error logs"));
  }
});
router.post("/api/monitoring/error-logs", async (req, res) => {
  try {
    const errorData = insertErrorLogSchema.parse(req.body);
    const error = await monitoringService.createErrorLog(errorData);
    webSocketService.broadcastErrorLog(error);
    res.status(201).json(createResponse(true, error));
  } catch (error) {
    if (error instanceof z2.ZodError) {
      return res.status(400).json(createResponse(false, null, "Invalid error data"));
    }
    console.error("Error creating error log:", error);
    res.status(500).json(createResponse(false, null, "Failed to create error log"));
  }
});
router.patch("/api/monitoring/error-logs/:id/resolve", async (req, res) => {
  try {
    const { resolutionNotes } = req.body;
    const error = await monitoringService.resolveError(req.params.id, resolutionNotes);
    if (!error) {
      return res.status(404).json(createResponse(false, null, "Error log not found"));
    }
    res.json(createResponse(true, error));
  } catch (error) {
    console.error("Error resolving error log:", error);
    res.status(500).json(createResponse(false, null, "Failed to resolve error log"));
  }
});
router.get("/api/monitoring/health", async (req, res) => {
  try {
    const health = await monitoringService.getSystemHealth();
    res.json(createResponse(true, health));
  } catch (error) {
    console.error("Error fetching system health:", error);
    res.status(500).json(createResponse(false, null, "Failed to fetch system health"));
  }
});
router.get("/api/monitoring/report", async (req, res) => {
  try {
    const report = await monitoringService.generateSystemReport();
    res.json(createResponse(true, report));
  } catch (error) {
    console.error("Error generating system report:", error);
    res.status(500).json(createResponse(false, null, "Failed to generate system report"));
  }
});
router.get("/api/knowledge/nodes", async (req, res) => {
  try {
    const nodes = await storage.getKnowledgeNodes();
    res.json(createResponse(true, nodes));
  } catch (error) {
    console.error("Error fetching knowledge nodes:", error);
    res.status(500).json(createResponse(false, null, "Failed to fetch knowledge nodes"));
  }
});
router.get("/api/knowledge/relationships", async (req, res) => {
  try {
    const relationships = await storage.getKnowledgeRelationships();
    res.json(createResponse(true, relationships));
  } catch (error) {
    console.error("Error fetching knowledge relationships:", error);
    res.status(500).json(createResponse(false, null, "Failed to fetch knowledge relationships"));
  }
});
router.get("/api/dashboard", async (req, res) => {
  try {
    const dashboardData = await storage.getDashboardData();
    res.json(createResponse(true, dashboardData));
  } catch (error) {
    console.error("Error fetching dashboard data:", error);
    res.status(500).json(createResponse(false, null, "Failed to fetch dashboard data"));
  }
});
router.get("/api/websocket/status", (req, res) => {
  res.json(createResponse(true, {
    connected_clients: webSocketService.getClientCount(),
    status: "active"
  }));
});
var routes_default = router;

// server/index.ts
import path from "path";
var app = express();
var server = createServer(app);
app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
  res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept, Authorization");
  if (req.method === "OPTIONS") {
    res.sendStatus(200);
  } else {
    next();
  }
});
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));
app.use(routes_default);
var clientPath = path.join(process.cwd(), "client", "dist");
app.use(express.static(clientPath));
app.get("*", (req, res) => {
  if (!req.path.startsWith("/api")) {
    res.sendFile(path.join(clientPath, "index.html"));
  }
});
var wss = new WebSocketServer({ server, path: "/ws" });
wss.on("connection", (ws) => {
  console.log("New WebSocket connection established");
  webSocketService.addClient(ws);
});
var startRealTimeUpdates = () => {
  setInterval(async () => {
    try {
      const agents = await agentService.getAllAgents();
      const activeAgents = agents.filter((a) => a.status === "active").length;
      const metrics = {
        cpu_usage: Math.floor(Math.random() * 30) + 20,
        // 20-50%
        memory_usage: Math.floor(Math.random() * 40) + 40,
        // 40-80%
        network_latency: Math.floor(Math.random() * 20) + 5,
        // 5-25ms
        storage_used: 23e12 + Math.floor(Math.random() * 1e12),
        // ~23TB + variation
        storage_total: 5e13,
        // 50TB
        active_agents: activeAgents,
        tasks_completed: 24e5 + Math.floor(Math.random() * 1e3),
        error_count: Math.floor(Math.random() * 5),
        warning_count: Math.floor(Math.random() * 10) + 3
      };
      const systemMetrics = await monitoringService.createSystemMetrics(metrics);
      webSocketService.broadcastSystemMetrics(systemMetrics);
    } catch (error) {
      console.error("Error updating system metrics:", error);
    }
  }, 3e4);
  setInterval(async () => {
    try {
      const agents = await agentService.getAllAgents();
      for (const agent of agents) {
        const cpuVariation = (Math.random() - 0.5) * 10;
        const memoryVariation = (Math.random() - 0.5) * 10;
        const newCpuUsage = Math.max(5, Math.min(95, agent.cpu_usage + cpuVariation));
        const newMemoryUsage = Math.max(10, Math.min(90, agent.memory_usage + memoryVariation));
        const updatedAgent = await agentService.updateAgentMetrics(agent.id, newCpuUsage, newMemoryUsage);
        if (updatedAgent) {
          webSocketService.broadcastAgentUpdate(updatedAgent);
        }
      }
    } catch (error) {
      console.error("Error updating agent metrics:", error);
    }
  }, 15e3);
  const generateActivityLog = async () => {
    try {
      const agents = await agentService.getAllAgents();
      if (agents.length === 0) return;
      const randomAgent = agents[Math.floor(Math.random() * agents.length)];
      const activities = [
        "Task batch completed successfully",
        "Performance optimization applied",
        "Pattern analysis complete",
        "Quality check passed",
        "Resource allocation updated",
        "Data processing completed",
        "Knowledge consolidation finished",
        "Inter-agent communication established"
      ];
      const activity = activities[Math.floor(Math.random() * activities.length)];
      const log = await monitoringService.logAgentActivity(
        randomAgent.id,
        randomAgent.type,
        activity,
        "info"
      );
      webSocketService.broadcastActivityLog(log);
    } catch (error) {
      console.error("Error generating activity log:", error);
    }
  };
  const scheduleNextActivityLog = () => {
    const delay = Math.floor(Math.random() * 2e4) + 1e4;
    setTimeout(() => {
      generateActivityLog();
      scheduleNextActivityLog();
    }, delay);
  };
  scheduleNextActivityLog();
};
startRealTimeUpdates();
var PORT = process.env.PORT || 5e3;
server.listen(PORT, "0.0.0.0", () => {
  console.log(`\u{1F680} YMERA Enterprise Platform running on port ${PORT}`);
  console.log(`\u{1F4CA} Dashboard: http://localhost:${PORT}`);
  console.log(`\u{1F50C} WebSocket: ws://localhost:${PORT}/ws`);
});
process.on("SIGTERM", () => {
  console.log("Shutting down gracefully...");
  webSocketService.closeAllConnections();
  server.close(() => {
    console.log("Server closed");
    process.exit(0);
  });
});
