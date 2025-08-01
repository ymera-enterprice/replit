export interface Agent {
  id: string;
  agentId: string;
  name: string;
  type: 'task_executor' | 'data_processor' | 'knowledge_manager' | 'coordinator' | 'analyzer' | 'communicator';
  status: 'initializing' | 'active' | 'learning' | 'idle' | 'error' | 'maintenance';
  description?: string;
  capabilities: Record<string, any>;
  learningConfig: Record<string, any>;
  collaborationConfig: Record<string, any>;
  
  // State tracking
  currentState: string;
  healthScore: number;
  
  // Performance metrics
  cpuUsage: number;
  memoryUsage: number;
  taskCount: number;
  successRate: number;
  
  // Learning metrics
  knowledgeItemsCount: number;
  learningVelocity: number;
  retentionRate: number;
  collaborationScore: number;
  patternDiscoveryCount: number;
  externalIntegrationSuccessRate: number;
  knowledgeApplicationRate: number;
  collectiveIntelligenceScore: number;
  
  // Collaboration metrics
  knowledgeTransfersSent: number;
  knowledgeTransfersReceived: number;
  successfulCollaborations: number;
  collaborationEfficiency: number;
  knowledgeDiversityScore: number;
  peerRating: number;
  responseTimeAvg: number;
  availabilityScore: number;
  
  // Timestamps
  createdAt: string;
  updatedAt: string;
  lastLearningCycle?: string;
  lastActivity: string;
}

export interface AgentKnowledge {
  id: string;
  knowledgeId: string;
  agentId: string;
  knowledgeType: string;
  content: Record<string, any>;
  encryptedContent?: string;
  confidenceScore: number;
  validationStatus: string;
  usageCount: number;
  successRate: number;
  applicableContexts: string[];
  sourceContext: Record<string, any>;
  learningSource: string;
  validationAttempts: number;
  lastApplied?: string;
  effectivenessScore: number;
  createdAt: string;
  updatedAt: string;
  expiresAt?: string;
}

export interface AgentCollaboration {
  id: string;
  collaborationId: string;
  sourceAgentId: string;
  targetAgentId: string;
  collaborationType: string;
  status: string;
  priority: number;
  requestData: Record<string, any>;
  responseData?: Record<string, any>;
  responseTimeMs?: number;
  successIndicator?: boolean;
  qualityScore?: number;
  requestedAt: string;
  respondedAt?: string;
  completedAt?: string;
  timeoutAt: string;
}

export interface LearningCycle {
  id: string;
  cycleId: string;
  agentId: string;
  cycleType: string;
  triggerSource: string;
  knowledgeItemsProcessed: number;
  knowledgeItemsLearned: number;
  knowledgeItemsValidated: number;
  knowledgeItemsApplied: number;
  cycleDurationMs: number;
  learningEfficiency: number;
  memoryConsolidationScore: number;
  patternsDiscovered: number;
  patternsValidated: number;
  behavioralInsights: Record<string, any>;
  knowledgeShared: number;
  knowledgeReceived: number;
  collaborationOpportunities: number;
  status: string;
  completionPercentage: number;
  errorDetails?: Record<string, any>;
  startedAt: string;
  completedAt?: string;
  nextCycleAt?: string;
}

export type AgentStatus = Agent['status'];
export type AgentType = Agent['type'];
