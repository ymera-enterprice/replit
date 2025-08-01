import { storage } from '../storage';
import { 
  LearningMetrics, 
  InsertLearningMetrics,
  LearningActivity,
  InsertLearningActivity,
  LearningActivityType 
} from '@shared/schema';

export class LearningService {
  async getLearningMetrics(agentId?: string): Promise<LearningMetrics[]> {
    return await storage.getLearningMetrics(agentId);
  }

  async createLearningMetrics(metrics: InsertLearningMetrics): Promise<LearningMetrics> {
    return await storage.createLearningMetrics(metrics);
  }

  async updateLearningMetrics(id: string, updates: Partial<InsertLearningMetrics>): Promise<LearningMetrics | null> {
    return await storage.updateLearningMetrics(id, updates);
  }

  async getLearningActivities(agentId?: string, limit?: number): Promise<LearningActivity[]> {
    return await storage.getLearningActivities(agentId, limit);
  }

  async createLearningActivity(activity: InsertLearningActivity): Promise<LearningActivity> {
    return await storage.createLearningActivity(activity);
  }

  async recordPatternDiscovery(agentId: string, description: string, accuracyImprovement: number): Promise<LearningActivity> {
    return await this.createLearningActivity({
      agent_id: agentId,
      activity_type: 'pattern_discovery',
      description,
      impact_score: Math.min(100, accuracyImprovement * 10),
      accuracy_improvement: accuracyImprovement
    });
  }

  async recordKnowledgeConsolidation(agentId: string, description: string, knowledgeUnits: number): Promise<LearningActivity> {
    return await this.createLearningActivity({
      agent_id: agentId,
      activity_type: 'knowledge_consolidation',
      description,
      impact_score: Math.min(100, knowledgeUnits / 10),
      knowledge_units: knowledgeUnits
    });
  }

  async recordInterAgentTransfer(sourceAgentId: string, targetAgentId: string, successRate: number): Promise<LearningActivity> {
    return await this.createLearningActivity({
      agent_id: sourceAgentId,
      activity_type: 'inter_agent_transfer',
      description: `Knowledge transfer to agent ${targetAgentId}`,
      impact_score: successRate,
      success_rate: successRate
    });
  }

  async getAgentLearningProgress(agentId: string): Promise<{
    total_activities: number;
    pattern_discoveries: number;
    knowledge_consolidations: number;
    avg_impact_score: number;
  }> {
    const activities = await storage.getLearningActivities(agentId);
    
    return {
      total_activities: activities.length,
      pattern_discoveries: activities.filter(a => a.activity_type === 'pattern_discovery').length,
      knowledge_consolidations: activities.filter(a => a.activity_type === 'knowledge_consolidation').length,
      avg_impact_score: activities.length > 0 
        ? activities.reduce((sum, a) => sum + a.impact_score, 0) / activities.length 
        : 0
    };
  }

  async calculateOverallLearningMetrics(): Promise<{
    total_pattern_accuracy: number;
    avg_knowledge_growth: number;
    total_activities: number;
    most_active_agent: string | null;
  }> {
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
    
    // Find most active agent
    const agentActivityCounts = allActivities.reduce((acc, activity) => {
      acc[activity.agent_id] = (acc[activity.agent_id] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    
    const mostActiveAgent = Object.entries(agentActivityCounts).length > 0
      ? Object.entries(agentActivityCounts).reduce((a, b) => a[1] > b[1] ? a : b)[0]
      : null;

    return {
      total_pattern_accuracy: Math.round(totalPatternAccuracy * 10) / 10,
      avg_knowledge_growth: Math.round(avgKnowledgeGrowth * 10) / 10,
      total_activities: allActivities.length,
      most_active_agent: mostActiveAgent
    };
  }
}

export const learningService = new LearningService();
