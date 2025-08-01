import { EventEmitter } from 'events';
import { storage } from '../storage';
import { KnowledgeItem, InsertKnowledgeItem, LearningPattern, InsertLearningPattern, Agent } from '@shared/schema';
import { StructuredLogger } from '../middleware/logging';
import { AppError } from '../middleware/errorHandler';

export interface Experience {
  id: string;
  type: string;
  context: Record<string, any>;
  action: Record<string, any>;
  outcome: Record<string, any>;
  success: boolean;
  timestamp: Date;
  agentId?: string;
  metadata?: Record<string, any>;
}

export interface Pattern {
  id: string;
  name: string;
  type: 'behavioral' | 'performance' | 'error' | 'optimization';
  conditions: Record<string, any>;
  actions: Record<string, any>;
  confidence: number;
  frequency: number;
  lastSeen: Date;
  isActive: boolean;
}

export interface LearningMetrics {
  totalExperiences: number;
  patternsDiscovered: number;
  knowledgeValidated: number;
  improvementSuggestions: number;
  confidenceScore: number;
  learningVelocity: number;
}

export interface KnowledgeGraph {
  nodes: Array<{
    id: string;
    type: string;
    label: string;
    properties: Record<string, any>;
  }>;
  edges: Array<{
    source: string;
    target: string;
    relationship: string;
    weight: number;
  }>;
}

class PatternDiscovery {
  private static readonly MIN_FREQUENCY = 3;
  private static readonly MIN_CONFIDENCE = 0.7;
  
  static async analyzeExperiences(experiences: Experience[]): Promise<Pattern[]> {
    const patterns: Pattern[] = [];
    
    // Group experiences by type and context similarity
    const grouped = this.groupSimilarExperiences(experiences);
    
    for (const [key, group] of grouped.entries()) {
      if (group.length < this.MIN_FREQUENCY) continue;
      
      const pattern = await this.extractPattern(key, group);
      if (pattern && pattern.confidence >= this.MIN_CONFIDENCE) {
        patterns.push(pattern);
      }
    }
    
    return patterns;
  }
  
  private static groupSimilarExperiences(experiences: Experience[]): Map<string, Experience[]> {
    const groups = new Map<string, Experience[]>();
    
    for (const experience of experiences) {
      const key = this.generateExperienceKey(experience);
      const group = groups.get(key) || [];
      group.push(experience);
      groups.set(key, group);
    }
    
    return groups;
  }
  
  private static generateExperienceKey(experience: Experience): string {
    // Create a key based on experience type and context similarity
    const contextKeys = Object.keys(experience.context).sort();
    const actionKeys = Object.keys(experience.action).sort();
    
    return `${experience.type}:${contextKeys.join(',')}:${actionKeys.join(',')}`;
  }
  
  private static async extractPattern(key: string, experiences: Experience[]): Promise<Pattern | null> {
    const successfulExperiences = experiences.filter(e => e.success);
    const confidence = successfulExperiences.length / experiences.length;
    
    if (confidence < this.MIN_CONFIDENCE) return null;
    
    // Extract common conditions and actions
    const conditions = this.extractCommonConditions(experiences);
    const actions = this.extractCommonActions(successfulExperiences);
    
    const pattern: Pattern = {
      id: crypto.randomUUID(),
      name: this.generatePatternName(key, experiences),
      type: this.determinePatternType(experiences),
      conditions,
      actions,
      confidence,
      frequency: experiences.length,
      lastSeen: new Date(Math.max(...experiences.map(e => e.timestamp.getTime()))),
      isActive: true,
    };
    
    return pattern;
  }
  
  private static extractCommonConditions(experiences: Experience[]): Record<string, any> {
    const conditions: Record<string, any> = {};
    const contextKeys = new Set<string>();
    
    // Collect all context keys
    experiences.forEach(e => {
      Object.keys(e.context).forEach(key => contextKeys.add(key));
    });
    
    // Find common values
    for (const key of contextKeys) {
      const values = experiences.map(e => e.context[key]).filter(v => v !== undefined);
      const uniqueValues = [...new Set(values)];
      
      if (uniqueValues.length === 1) {
        conditions[key] = uniqueValues[0];
      } else if (values.length === experiences.length) {
        // Calculate ranges for numeric values
        if (typeof values[0] === 'number') {
          conditions[key] = {
            min: Math.min(...values),
            max: Math.max(...values),
            avg: values.reduce((a, b) => a + b, 0) / values.length,
          };
        }
      }
    }
    
    return conditions;
  }
  
  private static extractCommonActions(experiences: Experience[]): Record<string, any> {
    const actions: Record<string, any> = {};
    const actionKeys = new Set<string>();
    
    experiences.forEach(e => {
      Object.keys(e.action).forEach(key => actionKeys.add(key));
    });
    
    for (const key of actionKeys) {
      const values = experiences.map(e => e.action[key]).filter(v => v !== undefined);
      const uniqueValues = [...new Set(values)];
      
      if (uniqueValues.length === 1) {
        actions[key] = uniqueValues[0];
      }
    }
    
    return actions;
  }
  
  private static generatePatternName(key: string, experiences: Experience[]): string {
    const [type] = key.split(':');
    const avgSuccess = experiences.filter(e => e.success).length / experiences.length;
    
    if (avgSuccess > 0.9) return `Optimal ${type} Pattern`;
    if (avgSuccess > 0.7) return `Effective ${type} Pattern`;
    return `Common ${type} Pattern`;
  }
  
  private static determinePatternType(experiences: Experience[]): Pattern['type'] {
    const type = experiences[0].type;
    
    if (type.includes('error') || type.includes('failure')) return 'error';
    if (type.includes('performance') || type.includes('speed')) return 'performance';
    if (type.includes('optimization') || type.includes('improve')) return 'optimization';
    return 'behavioral';
  }
}

class KnowledgeValidator {
  static async validateKnowledge(item: KnowledgeItem, relatedExperiences: Experience[]): Promise<{
    isValid: boolean;
    confidence: number;
    issues: string[];
  }> {
    const issues: string[] = [];
    let confidence = parseFloat(item.confidence.toString());
    
    // Check consistency with experiences
    if (relatedExperiences.length > 0) {
      const consistencyScore = this.checkConsistency(item, relatedExperiences);
      confidence = (confidence + consistencyScore) / 2;
      
      if (consistencyScore < 0.5) {
        issues.push('Knowledge conflicts with recent experiences');
      }
    }
    
    // Check recency
    const age = Date.now() - item.createdAt.getTime();
    const ageWeeks = age / (7 * 24 * 60 * 60 * 1000);
    
    if (ageWeeks > 4) {
      confidence *= 0.9; // Reduce confidence for old knowledge
      issues.push('Knowledge may be outdated');
    }
    
    // Check source reliability
    if (!item.source || item.source === 'unknown') {
      confidence *= 0.8;
      issues.push('Knowledge source is unreliable');
    }
    
    // Check validation status
    if (!item.isValidated) {
      confidence *= 0.7;
      issues.push('Knowledge has not been validated');
    }
    
    return {
      isValid: confidence >= 0.6 && issues.length === 0,
      confidence: Math.max(0, Math.min(1, confidence)),
      issues,
    };
  }
  
  private static checkConsistency(item: KnowledgeItem, experiences: Experience[]): number {
    // Simple consistency check based on content similarity
    const itemContent = JSON.stringify(item.content);
    let consistentExperiences = 0;
    
    for (const experience of experiences) {
      const experienceContent = JSON.stringify({
        context: experience.context,
        action: experience.action,
        outcome: experience.outcome,
      });
      
      // Calculate similarity (simplified)
      const similarity = this.calculateSimilarity(itemContent, experienceContent);
      if (similarity > 0.7 && experience.success) {
        consistentExperiences++;
      }
    }
    
    return experiences.length > 0 ? consistentExperiences / experiences.length : 0.5;
  }
  
  private static calculateSimilarity(str1: string, str2: string): number {
    // Simple Jaccard similarity
    const set1 = new Set(str1.toLowerCase().split(/\W+/));
    const set2 = new Set(str2.toLowerCase().split(/\W+/));
    
    const intersection = new Set([...set1].filter(x => set2.has(x)));
    const union = new Set([...set1, ...set2]);
    
    return intersection.size / union.size;
  }
}

export class LearningEngine extends EventEmitter {
  private experiences: Experience[] = [];
  private patterns: Map<string, Pattern> = new Map();
  private knowledgeGraph: KnowledgeGraph = { nodes: [], edges: [] };
  private learningInterval?: NodeJS.Timeout;
  
  constructor() {
    super();
    this.startLearningCycle();
  }
  
  async captureExperience(experience: Omit<Experience, 'id' | 'timestamp'>): Promise<Experience> {
    const fullExperience: Experience = {
      ...experience,
      id: crypto.randomUUID(),
      timestamp: new Date(),
    };
    
    this.experiences.push(fullExperience);
    
    // Keep only recent experiences in memory
    const maxExperiences = 10000;
    if (this.experiences.length > maxExperiences) {
      this.experiences = this.experiences.slice(-maxExperiences);
    }
    
    StructuredLogger.info('Experience captured', {
      experienceId: fullExperience.id,
      type: fullExperience.type,
      success: fullExperience.success,
      agentId: fullExperience.agentId,
    });
    
    this.emit('experience:captured', fullExperience);
    return fullExperience;
  }
  
  async discoverPatterns(): Promise<Pattern[]> {
    const recentExperiences = this.experiences.filter(
      e => Date.now() - e.timestamp.getTime() < 24 * 60 * 60 * 1000 // Last 24 hours
    );
    
    const newPatterns = await PatternDiscovery.analyzeExperiences(recentExperiences);
    
    for (const pattern of newPatterns) {
      // Check if pattern already exists
      const existingPattern = Array.from(this.patterns.values()).find(
        p => this.arePatternsimilar(p, pattern)
      );
      
      if (existingPattern) {
        // Update existing pattern
        existingPattern.frequency += pattern.frequency;
        existingPattern.confidence = (existingPattern.confidence + pattern.confidence) / 2;
        existingPattern.lastSeen = pattern.lastSeen;
      } else {
        // Add new pattern
        this.patterns.set(pattern.id, pattern);
        
        // Save to database
        await storage.createLearningPattern({
          name: pattern.name,
          description: `Discovered pattern with ${pattern.frequency} occurrences`,
          pattern: pattern,
          confidence: pattern.confidence.toString(),
          frequency: pattern.frequency,
          lastSeen: pattern.lastSeen,
        });
        
        StructuredLogger.info('New pattern discovered', {
          patternId: pattern.id,
          name: pattern.name,
          type: pattern.type,
          confidence: pattern.confidence,
          frequency: pattern.frequency,
        });
        
        this.emit('pattern:discovered', pattern);
      }
    }
    
    return newPatterns;
  }
  
  private arePatternsimilar(pattern1: Pattern, pattern2: Pattern): boolean {
    if (pattern1.type !== pattern2.type) return false;
    
    const similarity = this.calculateObjectSimilarity(pattern1.conditions, pattern2.conditions);
    return similarity > 0.8;
  }
  
  private calculateObjectSimilarity(obj1: Record<string, any>, obj2: Record<string, any>): number {
    const keys1 = new Set(Object.keys(obj1));
    const keys2 = new Set(Object.keys(obj2));
    const allKeys = new Set([...keys1, ...keys2]);
    
    let matches = 0;
    for (const key of allKeys) {
      if (obj1[key] === obj2[key]) {
        matches++;
      }
    }
    
    return matches / allKeys.size;
  }
  
  async createKnowledge(knowledge: Omit<InsertKnowledgeItem, 'createdAt' | 'updatedAt'>): Promise<KnowledgeItem> {
    const item = await storage.createKnowledgeItem(knowledge);
    
    // Update knowledge graph
    await this.updateKnowledgeGraph(item);
    
    StructuredLogger.info('Knowledge item created', {
      knowledgeId: item.id,
      type: item.type,
      source: item.source,
      confidence: item.confidence,
    });
    
    this.emit('knowledge:created', item);
    return item;
  }
  
  async validateKnowledge(knowledgeId: string): Promise<void> {
    const item = await storage.getKnowledgeItemById(knowledgeId);
    if (!item) {
      throw new AppError('Knowledge item not found', 404);
    }
    
    // Get related experiences
    const relatedExperiences = this.experiences.filter(
      e => e.type === item.type || this.isRelatedContent(e, item)
    );
    
    const validation = await KnowledgeValidator.validateKnowledge(item, relatedExperiences);
    
    // Update knowledge item
    await storage.updateKnowledgeItem(knowledgeId, {
      confidence: validation.confidence.toString(),
      isValidated: validation.isValid,
      relevanceScore: validation.confidence.toString(),
    });
    
    StructuredLogger.info('Knowledge validation completed', {
      knowledgeId,
      isValid: validation.isValid,
      confidence: validation.confidence,
      issues: validation.issues,
    });
    
    this.emit('knowledge:validated', { item, validation });
  }
  
  private isRelatedContent(experience: Experience, knowledge: KnowledgeItem): boolean {
    // Simple content similarity check
    const expContent = JSON.stringify([experience.context, experience.action, experience.outcome]);
    const knContent = JSON.stringify(knowledge.content);
    
    return this.calculateStringSimilarity(expContent, knContent) > 0.5;
  }
  
  private calculateStringSimilarity(str1: string, str2: string): number {
    const words1 = new Set(str1.toLowerCase().split(/\W+/));
    const words2 = new Set(str2.toLowerCase().split(/\W+/));
    
    const intersection = new Set([...words1].filter(x => words2.has(x)));
    const union = new Set([...words1, ...words2]);
    
    return intersection.size / union.size;
  }
  
  private async updateKnowledgeGraph(item: KnowledgeItem): Promise<void> {
    // Add node for this knowledge item
    const node = {
      id: item.id,
      type: item.type,
      label: item.type,
      properties: {
        confidence: item.confidence,
        source: item.source,
        tags: item.tags,
      },
    };
    
    this.knowledgeGraph.nodes.push(node);
    
    // Find relationships with existing knowledge
    for (const existingNode of this.knowledgeGraph.nodes) {
      if (existingNode.id === item.id) continue;
      
      const existingItem = await storage.getKnowledgeItemById(existingNode.id);
      if (!existingItem) continue;
      
      const similarity = this.calculateObjectSimilarity(
        item.content as Record<string, any>,
        existingItem.content as Record<string, any>
      );
      
      if (similarity > 0.3) {
        this.knowledgeGraph.edges.push({
          source: item.id,
          target: existingNode.id,
          relationship: 'related',
          weight: similarity,
        });
      }
    }
  }
  
  async getOptimizationSuggestions(agentId?: string): Promise<Array<{
    type: 'performance' | 'efficiency' | 'accuracy' | 'reliability';
    suggestion: string;
    confidence: number;
    impact: 'low' | 'medium' | 'high';
    implementation: Record<string, any>;
  }>> {
    const suggestions: Array<{
      type: 'performance' | 'efficiency' | 'accuracy' | 'reliability';
      suggestion: string;
      confidence: number;
      impact: 'low' | 'medium' | 'high';
      implementation: Record<string, any>;
    }> = [];
    
    // Analyze patterns for optimization opportunities
    const optimizationPatterns = Array.from(this.patterns.values()).filter(
      p => p.type === 'optimization' && p.confidence > 0.7
    );
    
    for (const pattern of optimizationPatterns) {
      // Analyze pattern for specific suggestions
      if (pattern.conditions.responseTime && pattern.conditions.responseTime.avg > 1000) {
        suggestions.push({
          type: 'performance',
          suggestion: 'Optimize response time by implementing caching strategy',
          confidence: pattern.confidence,
          impact: 'high',
          implementation: {
            action: 'implement_caching',
            parameters: pattern.actions,
          },
        });
      }
      
      if (pattern.conditions.errorRate && pattern.conditions.errorRate > 0.05) {
        suggestions.push({
          type: 'reliability',
          suggestion: 'Improve error handling and retry mechanisms',
          confidence: pattern.confidence,
          impact: 'medium',
          implementation: {
            action: 'improve_error_handling',
            parameters: pattern.actions,
          },
        });
      }
    }
    
    // Filter suggestions by agent if specified
    if (agentId) {
      const agentExperiences = this.experiences.filter(e => e.agentId === agentId);
      // Additional agent-specific analysis could be added here
    }
    
    return suggestions.sort((a, b) => b.confidence - a.confidence);
  }
  
  private startLearningCycle(): void {
    if (this.learningInterval) {
      clearInterval(this.learningInterval);
    }
    
    this.learningInterval = setInterval(async () => {
      try {
        await this.runLearningCycle();
      } catch (error) {
        StructuredLogger.error('Learning cycle error', error);
      }
    }, 5 * 60 * 1000); // Run every 5 minutes
  }
  
  private async runLearningCycle(): Promise<void> {
    StructuredLogger.debug('Running learning cycle');
    
    // Discover new patterns
    await this.discoverPatterns();
    
    // Validate recent knowledge
    const recentKnowledge = await storage.getRecentKnowledge(10);
    for (const item of recentKnowledge) {
      if (!item.isValidated) {
        await this.validateKnowledge(item.id);
      }
    }
    
    // Clean up old patterns
    await this.cleanupOldPatterns();
    
    this.emit('learning:cycle-complete');
  }
  
  private async cleanupOldPatterns(): Promise<void> {
    const cutoffDate = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000); // 30 days
    
    for (const [id, pattern] of this.patterns.entries()) {
      if (pattern.lastSeen < cutoffDate && pattern.frequency < 5) {
        this.patterns.delete(id);
        await storage.updateLearningPattern(id, { isActive: false });
      }
    }
  }
  
  async getMetrics(): Promise<LearningMetrics> {
    const totalExperiences = this.experiences.length;
    const patternsDiscovered = this.patterns.size;
    const knowledgeStats = await storage.getKnowledgeStats();
    
    const recentExperiences = this.experiences.filter(
      e => Date.now() - e.timestamp.getTime() < 24 * 60 * 60 * 1000
    );
    
    const learningVelocity = recentExperiences.length;
    const avgConfidence = Array.from(this.patterns.values())
      .reduce((sum, p) => sum + p.confidence, 0) / this.patterns.size || 0;
    
    return {
      totalExperiences,
      patternsDiscovered,
      knowledgeValidated: knowledgeStats.validated,
      improvementSuggestions: (await this.getOptimizationSuggestions()).length,
      confidenceScore: avgConfidence,
      learningVelocity,
    };
  }
  
  getKnowledgeGraph(): KnowledgeGraph {
    return this.knowledgeGraph;
  }
  
  async getPatterns(type?: Pattern['type']): Promise<Pattern[]> {
    const patterns = Array.from(this.patterns.values());
    return type ? patterns.filter(p => p.type === type) : patterns;
  }
  
  destroy(): void {
    if (this.learningInterval) {
      clearInterval(this.learningInterval);
    }
    
    this.experiences = [];
    this.patterns.clear();
    this.knowledgeGraph = { nodes: [], edges: [] };
    this.removeAllListeners();
    
    StructuredLogger.info('Learning engine destroyed');
  }
}

// Singleton instance
export const learningEngine = new LearningEngine();
