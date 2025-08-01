export interface User {
  id: number;
  username: string;
  email: string;
  role: string;
  isActive: boolean;
  lastLogin?: string;
  createdAt: string;
  updatedAt: string;
}

export interface Agent {
  id: string;
  name: string;
  type: 'core' | 'specialized' | 'custom';
  status: 'initializing' | 'active' | 'idle' | 'busy' | 'paused' | 'error' | 'shutdown';
  description?: string;
  capabilities: string[];
  configuration: Record<string, any>;
  resourceLimits: Record<string, number>;
  learningEnabled: boolean;
  autoScale: boolean;
  healthScore: number;
  cpuUsage: number;
  memoryUsage: number;
  taskCount: number;
  successRate: number;
  lastActivity?: string;
  createdBy?: number;
  createdAt: string;
  updatedAt: string;
}

export interface Project {
  id: string;
  name: string;
  description?: string;
  status: 'planning' | 'active' | 'paused' | 'completed' | 'cancelled';
  priority: 'low' | 'medium' | 'high' | 'urgent';
  progress: number;
  metadata: Record<string, any>;
  ownerId: number;
  createdAt: string;
  updatedAt: string;
}

export interface Task {
  id: string;
  agentId?: string;
  projectId?: string;
  taskType: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  priority: 'low' | 'medium' | 'high' | 'urgent';
  payload: Record<string, any>;
  context?: Record<string, any>;
  result?: Record<string, any>;
  error?: string;
  timeout: number;
  retryCount: number;
  maxRetries: number;
  requiresLearning: boolean;
  executionTime?: number;
  startedAt?: string;
  completedAt?: string;
  createdBy?: number;
  createdAt: string;
  updatedAt: string;
}

export interface Collaboration {
  id: string;
  name: string;
  description?: string;
  status: 'active' | 'paused' | 'completed' | 'cancelled';
  projectId?: string;
  participantAgents: string[];
  progress: number;
  metadata: Record<string, any>;
  startedAt?: string;
  completedAt?: string;
  createdBy?: number;
  createdAt: string;
  updatedAt: string;
}

export interface KnowledgeItem {
  id: string;
  title: string;
  content: string;
  type: string;
  tags: string[];
  metadata: Record<string, any>;
  sourceAgentId?: string;
  projectId?: string;
  isPublic: boolean;
  createdBy?: number;
  createdAt: string;
  updatedAt: string;
}

export interface LearningSession {
  id: string;
  agentId: string;
  sessionType: string;
  data: Record<string, any>;
  insights: Record<string, any>;
  performance: number;
  duration: number;
  startedAt: string;
  completedAt?: string;
  createdAt: string;
}

export interface SystemMetric {
  id: number;
  metricType: string;
  value: number;
  metadata: Record<string, any>;
  timestamp: string;
}

export interface DashboardStats {
  activeAgents: number;
  totalProjects: number;
  learningSessions: number;
  knowledgeItems: number;
  systemHealth: number;
}

export interface WebSocketMessage {
  type: string;
  data: any;
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  pagination?: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
}
