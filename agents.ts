import { storage } from '../storage';
import { Agent, InsertAgent, AgentStatus } from '@shared/schema';

export class AgentService {
  async getAllAgents(): Promise<Agent[]> {
    return await storage.getAgents();
  }

  async getAgent(id: string): Promise<Agent | null> {
    return await storage.getAgent(id);
  }

  async createAgent(agentData: InsertAgent): Promise<Agent> {
    return await storage.createAgent(agentData);
  }

  async updateAgent(id: string, updates: Partial<InsertAgent>): Promise<Agent | null> {
    return await storage.updateAgent(id, updates);
  }

  async updateAgentStatus(id: string, status: AgentStatus): Promise<Agent | null> {
    return await storage.updateAgent(id, { 
      status, 
      last_activity: new Date().toISOString() 
    });
  }

  async deleteAgent(id: string): Promise<boolean> {
    return await storage.deleteAgent(id);
  }

  async getAgentHealth(id: string): Promise<{ health_score: number; cpu_usage: number; memory_usage: number } | null> {
    const agent = await storage.getAgent(id);
    if (!agent) return null;

    return {
      health_score: agent.health_score,
      cpu_usage: agent.cpu_usage,
      memory_usage: agent.memory_usage
    };
  }

  async updateAgentMetrics(id: string, cpu_usage: number, memory_usage: number): Promise<Agent | null> {
    const health_score = this.calculateHealthScore(cpu_usage, memory_usage);
    
    return await storage.updateAgent(id, {
      cpu_usage,
      memory_usage,
      health_score,
      last_activity: new Date().toISOString()
    });
  }

  private calculateHealthScore(cpu_usage: number, memory_usage: number): number {
    // Simple health score calculation based on resource usage
    const cpuScore = Math.max(0, 100 - cpu_usage);
    const memoryScore = Math.max(0, 100 - memory_usage);
    return Math.round((cpuScore + memoryScore) / 2);
  }

  async getAgentsByStatus(status: AgentStatus): Promise<Agent[]> {
    const agents = await storage.getAgents();
    return agents.filter(agent => agent.status === status);
  }

  async getAgentTaskLoad(id: string): Promise<{ task_count: number; success_rate: number } | null> {
    const agent = await storage.getAgent(id);
    if (!agent) return null;

    return {
      task_count: agent.task_count,
      success_rate: agent.success_rate
    };
  }
}

export const agentService = new AgentService();
