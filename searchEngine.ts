import { storage } from '../storage';
import type { File } from '@shared/schema';

interface SearchResult {
  file: File;
  score: number;
  highlights?: string[];
}

export class SearchEngine {
  async searchFiles(query: string, userId: string): Promise<SearchResult[]> {
    // Basic text search implementation
    const files = await storage.searchFiles(query, userId);
    
    return files.map(file => ({
      file,
      score: this.calculateRelevanceScore(file, query),
      highlights: this.extractHighlights(file, query)
    }));
  }

  private calculateRelevanceScore(file: File, query: string): number {
    let score = 0;
    const queryLower = query.toLowerCase();
    
    // Name match (highest priority)
    if (file.originalName.toLowerCase().includes(queryLower)) {
      score += 10;
    }
    
    // Exact filename match gets highest score
    if (file.originalName.toLowerCase() === queryLower) {
      score += 20;
    }
    
    return Math.min(score, 100);
  }

  private extractHighlights(file: File, query: string): string[] {
    const highlights: string[] = [];
    const queryLower = query.toLowerCase();
    
    // Add filename highlights
    if (file.originalName.toLowerCase().includes(queryLower)) {
      highlights.push(file.originalName);
    }
    
    return highlights.slice(0, 3); // Limit to 3 highlights
  }

  async indexFile(fileId: string, content: string): Promise<void> {
    await storage.createSearchIndex({
      fileId,
      content,
      rankingScore: content.length > 0 ? 1.0 : 0.0,
    });
  }
}

export const searchEngine = new SearchEngine();