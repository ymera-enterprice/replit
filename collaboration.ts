import { storage } from '../storage';
import type { Collaboration, InsertCollaboration, Agent, Task } from '@shared/schema';

export class CollaborationService {
  async createCollaboration(collaborationData: InsertCollaboration): Promise<Collaboration> {
    const collaboration = await storage.createCollaboration({
      ...collaborationData,
      startedAt: new Date(),
    });

    // Initialize collaboration
    setTimeout(async () => {
      await this.initializeCollaboration(collaboration.id);
    }, 1000);

    return collaboration;
  }

  async updateCollaboration(id: string, updates: Partial<InsertCollaboration>): Promise<Collaboration | undefined> {
    return await storage.updateCollaboration(id, updates);
  }

  async addAgentToCollaboration(collaborationId: string, agentId: string): Promise<Collaboration | undefined> {
    const collaboration = await storage.getCollaboration(collaborationId);
    if (!collaboration) return undefined;

    const agent = await storage.getAgent(agentId);
    if (!agent) throw new Error('Agent not found');

    if (collaboration.participantAgents.includes(agentId)) {
      throw new Error('Agent is already part of this collaboration');
    }

    const updatedAgents = [...collaboration.participantAgents, agentId];
    
    return await storage.updateCollaboration(collaborationId, {
      participantAgents: updatedAgents,
    });
  }

  async removeAgentFromCollaboration(collaborationId: string, agentId: string): Promise<Collaboration | undefined> {
    const collaboration = await storage.getCollaboration(collaborationId);
    if (!collaboration) return undefined;

    const updatedAgents = collaboration.participantAgents.filter(id => id !== agentId);
    
    return await storage.updateCollaboration(collaborationId, {
      participantAgents: updatedAgents,
    });
  }

  async getCollaborationProgress(collaborationId: string): Promise<{
    overallProgress: number;
    agentContributions: Array<{ agentId: string; contribution: number }>;
    completedTasks: number;
    totalTasks: number;
    timeElapsed: number;
    estimatedCompletion: number;
  } | null> {
    const collaboration = await storage.getCollaboration(collaborationId);
    if (!collaboration) return null;

    // Get all tasks for this collaboration's project
    const tasks = await storage.getTasks({
      projectId: collaboration.projectId || undefined,
      limit: 1000
    });

    const collaborationTasks = tasks.tasks.filter(task => 
      collaboration.participantAgents.includes(task.agentId || '')
    );

    const completedTasks = collaborationTasks.filter(task => task.status === 'completed').length;
    const totalTasks = collaborationTasks.length;
    const overallProgress = totalTasks > 0 ? Math.round((completedTasks / totalTasks) * 100) : 0;

    // Calculate agent contributions
    const agentContributions = collaboration.participantAgents.map(agentId => {
      const agentTasks = collaborationTasks.filter(task => task.agentId === agentId);
      const agentCompletedTasks = agentTasks.filter(task => task.status === 'completed').length;
      const contribution = agentTasks.length > 0 
        ? Math.round((agentCompletedTasks / agentTasks.length) * 100)
        : 0;
      
      return { agentId, contribution };
    });

    // Calculate time metrics
    const startTime = new Date(collaboration.startedAt || collaboration.createdAt).getTime();
    const currentTime = Date.now();
    const timeElapsed = Math.round((currentTime - startTime) / (1000 * 60)); // minutes

    // Estimate completion time based on current progress
    const estimatedCompletion = overallProgress > 0 
      ? Math.round((timeElapsed / overallProgress) * (100 - overallProgress))
      : 0;

    return {
      overallProgress,
      agentContributions,
      completedTasks,
      totalTasks,
      timeElapsed,
      estimatedCompletion
    };
  }

  async getActiveCollaborations(): Promise<Array<Collaboration & { 
    progress: number; 
    participantCount: number; 
    duration: number;
  }>> {
    const collaborations = await storage.getCollaborations({
      status: 'active',
      limit: 100
    });

    const enrichedCollaborations = await Promise.all(
      collaborations.collaborations.map(async (collaboration) => {
        const progress = await this.getCollaborationProgress(collaboration.id);
        const duration = collaboration.startedAt 
          ? Math.round((Date.now() - new Date(collaboration.startedAt).getTime()) / (1000 * 60))
          : 0;

        return {
          ...collaboration,
          progress: progress?.overallProgress || 0,
          participantCount: collaboration.participantAgents.length,
          duration
        };
      })
    );

    return enrichedCollaborations;
  }

  async pauseCollaboration(collaborationId: string): Promise<Collaboration | undefined> {
    return await storage.updateCollaboration(collaborationId, {
      status: 'paused',
    });
  }

  async resumeCollaboration(collaborationId: string): Promise<Collaboration | undefined> {
    return await storage.updateCollaboration(collaborationId, {
      status: 'active',
    });
  }

  async completeCollaboration(collaborationId: string): Promise<Collaboration | undefined> {
    const collaboration = await storage.updateCollaboration(collaborationId, {
      status: 'completed',
      completedAt: new Date(),
    });

    if (collaboration) {
      // Update progress to 100%
      await storage.updateCollaboration(collaborationId, {
        progress: 100,
      });
    }

    return collaboration;
  }

  private async initializeCollaboration(collaborationId: string): Promise<void> {
    try {
      const collaboration = await storage.getCollaboration(collaborationId);
      if (!collaboration) return;

      // Verify all participant agents exist and are available
      const agents = await Promise.all(
        collaboration.participantAgents.map(agentId => storage.getAgent(agentId))
      );

      const availableAgents = agents.filter(agent => 
        agent && ['idle', 'active'].includes(agent.status)
      );

      if (availableAgents.length < collaboration.participantAgents.length) {
        console.warn(`Some agents not available for collaboration ${collaborationId}`);
      }

      // Set initial progress and metadata
      await storage.updateCollaboration(collaborationId, {
        progress: 0,
        metadata: {
          ...collaboration.metadata,
          initializedAt: new Date().toISOString(),
          availableAgents: availableAgents.length,
          totalAgents: collaboration.participantAgents.length,
        }
      });

      console.log(`Collaboration ${collaborationId} initialized with ${availableAgents.length} agents`);
    } catch (error) {
      console.error(`Failed to initialize collaboration ${collaborationId}:`, error);
    }
  }

  async getCollaborationInsights(collaborationId: string): Promise<{
    efficiency: number;
    communicationScore: number;
    knowledgeSharing: number;
    recommendations: string[];
  } | null> {
    const collaboration = await storage.getCollaboration(collaborationId);
    if (!collaboration) return null;

    const progress = await this.getCollaborationProgress(collaborationId);
    if (!progress) return null;

    // Calculate efficiency based on time vs progress
    const efficiency = progress.timeElapsed > 0 
      ? Math.min(100, Math.round((progress.overallProgress / progress.timeElapsed) * 60))
      : 50;

    // Simulate communication score based on agent participation
    const communicationScore = Math.round(
      progress.agentContributions.reduce((sum, contrib) => sum + contrib.contribution, 0) / 
      progress.agentContributions.length
    );

    // Simulate knowledge sharing score
    const knowledgeSharing = Math.round(Math.random() * 30 + 70); // 70-100%

    // Generate recommendations
    const recommendations: string[] = [];
    
    if (efficiency < 50) {
      recommendations.push('Consider reassigning tasks to optimize workload distribution');
    }
    
    if (communicationScore < 70) {
      recommendations.push('Improve communication protocols between agents');
    }
    
    if (knowledgeSharing < 80) {
      recommendations.push('Implement better knowledge sharing mechanisms');
    }
    
    if (progress.overallProgress < 30 && progress.timeElapsed > 120) {
      recommendations.push('Review project scope and deadlines');
    }

    if (recommendations.length === 0) {
      recommendations.push('Collaboration is performing well, continue current approach');
    }

    return {
      efficiency,
      communicationScore,
      knowledgeSharing,
      recommendations
    };
  }
}

export const collaborationService = new CollaborationService();
