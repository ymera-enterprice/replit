import * as fs from 'fs/promises';
import * as path from 'path';
import * as crypto from 'crypto';
import { exec } from 'child_process';
import { promisify } from 'util';
import { storage } from '../storage';
import type { File, NewFileMetadata } from '@shared/schema';

const execAsync = promisify(exec);

export interface ProcessingResult {
  extractedText?: string;
  metadata: Record<string, any>;
  thumbnailPath?: string;
  pageCount?: number;
  wordCount?: number;
  language?: string;
  tags: string[];
}

export class FileProcessor {
  private uploadDir = process.env.UPLOAD_DIR || './uploads';
  private thumbnailDir = process.env.THUMBNAIL_DIR || './thumbnails';

  constructor() {
    this.ensureDirectories();
  }

  private async ensureDirectories() {
    await fs.mkdir(this.uploadDir, { recursive: true });
    await fs.mkdir(this.thumbnailDir, { recursive: true });
  }

  async processFile(file: File): Promise<ProcessingResult> {
    console.log(`Processing file: ${file.name} (${file.mimeType})`);
    
    try {
      // Log processing start
      await storage.createProcessingLog({
        fileId: file.id,
        stage: 'extract',
        status: 'started',
        message: `Starting processing for ${file.mimeType}`,
      });

      let result: ProcessingResult;

      switch (file.mimeType) {
        case 'application/pdf':
          result = await this.processPDF(file);
          break;
        case 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
          result = await this.processDOCX(file);
          break;
        case 'text/plain':
          result = await this.processText(file);
          break;
        case 'image/jpeg':
        case 'image/png':
        case 'image/gif':
        case 'image/webp':
          result = await this.processImage(file);
          break;
        default:
          if (file.mimeType.startsWith('text/') || this.isCodeFile(file.name)) {
            result = await this.processCode(file);
          } else {
            result = await this.processGeneric(file);
          }
      }

      // Create metadata record
      const metadata: NewFileMetadata = {
        fileId: file.id,
        extractedText: result.extractedText,
        metadata: result.metadata,
        thumbnailPath: result.thumbnailPath,
        pageCount: result.pageCount,
        wordCount: result.wordCount,
        language: result.language,
        tags: result.tags,
      };

      await storage.createFileMetadata(metadata);

      // Create search index
      if (result.extractedText) {
        await storage.createSearchIndex({
          fileId: file.id,
          content: result.extractedText,
          rankingScore: this.calculateRankingScore(result),
        });
      }

      // Log processing completion
      await storage.createProcessingLog({
        fileId: file.id,
        stage: 'complete',
        status: 'completed',
        message: 'File processing completed successfully',
      });

      // Update file status
      await storage.updateFileStatus(file.id, 'ready');

      return result;
    } catch (error) {
      console.error(`Error processing file ${file.id}:`, error);
      
      await storage.createProcessingLog({
        fileId: file.id,
        stage: 'extract',
        status: 'failed',
        message: `Processing failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
      });

      await storage.updateFileStatus(file.id, 'error');
      throw error;
    }
  }

  private async processPDF(file: File): Promise<ProcessingResult> {
    const filePath = file.path;
    
    try {
      // Extract text using pdftotext (requires poppler-utils)
      const { stdout: text } = await execAsync(`pdftotext "${filePath}" -`);
      
      // Get page count
      const { stdout: pageInfo } = await execAsync(`pdfinfo "${filePath}"`);
      const pageMatch = pageInfo.match(/Pages:\s+(\d+)/);
      const pageCount = pageMatch ? parseInt(pageMatch[1]) : undefined;

      // Generate thumbnail
      const thumbnailPath = await this.generatePDFThumbnail(filePath, file.id);

      // Extract metadata
      const metadata = {
        creator: this.extractMetadataField(pageInfo, 'Creator'),
        producer: this.extractMetadataField(pageInfo, 'Producer'),
        title: this.extractMetadataField(pageInfo, 'Title'),
        subject: this.extractMetadataField(pageInfo, 'Subject'),
        keywords: this.extractMetadataField(pageInfo, 'Keywords'),
      };

      return {
        extractedText: text,
        metadata,
        thumbnailPath,
        pageCount,
        wordCount: this.countWords(text),
        language: this.detectLanguage(text),
        tags: this.extractTags(text, file.name),
      };
    } catch (error) {
      console.error('PDF processing error:', error);
      return this.processGeneric(file);
    }
  }

  private async processDOCX(file: File): Promise<ProcessingResult> {
    const filePath = file.path;
    
    try {
      // Extract text using pandoc or unzip approach
      const { stdout: text } = await execAsync(`pandoc "${filePath}" -t plain`);

      const metadata = {
        format: 'DOCX',
        hasImages: text.includes('[IMAGE'),
        hasTables: text.includes('|'),
      };

      return {
        extractedText: text,
        metadata,
        wordCount: this.countWords(text),
        language: this.detectLanguage(text),
        tags: this.extractTags(text, file.name),
      };
    } catch (error) {
      console.error('DOCX processing error:', error);
      return this.processGeneric(file);
    }
  }

  private async processText(file: File): Promise<ProcessingResult> {
    const content = await fs.readFile(file.path, 'utf-8');
    
    const metadata = {
      encoding: 'UTF-8',
      lineCount: content.split('\n').length,
      hasCode: this.detectCodePatterns(content),
    };

    return {
      extractedText: content,
      metadata,
      wordCount: this.countWords(content),
      language: this.detectLanguage(content),
      tags: this.extractTags(content, file.name),
    };
  }

  private async processImage(file: File): Promise<ProcessingResult> {
    try {
      // Get image info using ImageMagick
      const { stdout: info } = await execAsync(`identify "${file.path}"`);
      const infoMatch = info.match(/(\d+)x(\d+)/);
      const dimensions = infoMatch ? { width: parseInt(infoMatch[1]), height: parseInt(infoMatch[2]) } : null;

      // Generate thumbnail
      const thumbnailPath = await this.generateImageThumbnail(file.path, file.id);

      // Try OCR if enabled
      let extractedText = '';
      try {
        const { stdout: ocrText } = await execAsync(`tesseract "${file.path}" stdout`);
        extractedText = ocrText.trim();
      } catch (ocrError) {
        console.log('OCR not available or failed, skipping text extraction');
      }

      const metadata = {
        dimensions,
        format: path.extname(file.name).toUpperCase().slice(1),
        hasText: extractedText.length > 0,
      };

      return {
        extractedText: extractedText || undefined,
        metadata,
        thumbnailPath,
        wordCount: extractedText ? this.countWords(extractedText) : undefined,
        language: extractedText ? this.detectLanguage(extractedText) : undefined,
        tags: this.extractImageTags(file.name, metadata),
      };
    } catch (error) {
      console.error('Image processing error:', error);
      return this.processGeneric(file);
    }
  }

  private async processCode(file: File): Promise<ProcessingResult> {
    const content = await fs.readFile(file.path, 'utf-8');
    const extension = path.extname(file.name).toLowerCase();
    
    const metadata = {
      language: this.getCodeLanguage(extension),
      lineCount: content.split('\n').length,
      hasComments: this.detectComments(content, extension),
      hasImports: this.detectImports(content, extension),
      complexity: this.estimateComplexity(content),
    };

    return {
      extractedText: content,
      metadata,
      wordCount: this.countWords(content),
      tags: this.extractCodeTags(content, extension),
    };
  }

  private async processGeneric(file: File): Promise<ProcessingResult> {
    const stats = await fs.stat(file.path);
    
    const metadata = {
      fileType: file.mimeType,
      processed: false,
      reason: 'Unsupported file type for content extraction',
    };

    return {
      metadata,
      tags: [file.mimeType.split('/')[0], path.extname(file.name).slice(1)].filter(Boolean),
    };
  }

  private async generatePDFThumbnail(filePath: string, fileId: string): Promise<string> {
    const thumbnailPath = path.join(this.thumbnailDir, `${fileId}.png`);
    
    try {
      await execAsync(`pdftoppm "${filePath}" -png -singlefile -scale-to-x 300 -scale-to-y -1 "${thumbnailPath.replace('.png', '')}"`);
      return thumbnailPath;
    } catch (error) {
      console.error('Thumbnail generation failed:', error);
      return "";
    }
  }

  private async generateImageThumbnail(originalPath: string, fileId: string): Promise<string> {
    const thumbnailPath = path.join(this.thumbnailDir, `${fileId}.png`);
    
    try {
      await execAsync(`convert "${originalPath}" -thumbnail 300x300^ -gravity center -extent 300x300 "${thumbnailPath}"`);
      return thumbnailPath;
    } catch (error) {
      console.error('Image thumbnail generation failed:', error);
      return "";
    }
  }

  private extractMetadataField(info: string, field: string): string | undefined {
    const match = info.match(new RegExp(`${field}:\\s*(.+)`));
    return match ? match[1].trim() : undefined;
  }

  private countWords(text: string): number {
    return text.trim().split(/\s+/).filter(word => word.length > 0).length;
  }

  private detectLanguage(text: string): string {
    // Simple language detection based on common words
    const englishWords = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'];
    const words = text.toLowerCase().split(/\s+/);
    const englishCount = words.filter(word => englishWords.includes(word)).length;
    
    if (englishCount > words.length * 0.1) {
      return 'en';
    }
    
    return 'unknown';
  }

  private extractTags(text: string, filename: string): string[] {
    const tags = new Set<string>();
    
    // Add file extension
    const ext = path.extname(filename).slice(1).toLowerCase();
    if (ext) tags.add(ext);
    
    // Extract common keywords
    const keywords = text.toLowerCase().match(/\b[a-z]{3,}\b/g) || [];
    const commonKeywords = ['report', 'analysis', 'project', 'document', 'data', 'information'];
    
    commonKeywords.forEach(keyword => {
      if (keywords.includes(keyword)) {
        tags.add(keyword);
      }
    });
    
    return Array.from(tags).slice(0, 10);
  }

  private extractImageTags(filename: string, metadata: any): string[] {
    const tags = ['image'];
    
    if (metadata.format) {
      tags.push(metadata.format.toLowerCase());
    }
    
    if (metadata.hasText) {
      tags.push('has-text');
    }
    
    // Add tags based on filename
    const name = filename.toLowerCase();
    const imageTypes = ['screenshot', 'photo', 'diagram', 'chart', 'logo', 'icon'];
    imageTypes.forEach(type => {
      if (name.includes(type)) {
        tags.push(type);
      }
    });
    
    return tags;
  }

  private extractCodeTags(content: string, extension: string): string[] {
    const tags = ['code'];
    
    if (extension) {
      tags.push(extension.slice(1));
    }
    
    // Detect frameworks/libraries
    const frameworks = {
      'react': /import.*react/i,
      'vue': /import.*vue/i,
      'angular': /import.*angular/i,
      'express': /import.*express/i,
      'flask': /from flask/i,
      'django': /from django/i,
    };
    
    Object.entries(frameworks).forEach(([name, pattern]) => {
      if (pattern.test(content)) {
        tags.push(name);
      }
    });
    
    return tags;
  }

  private isCodeFile(filename: string): boolean {
    const codeExtensions = ['.js', '.ts', '.jsx', '.tsx', '.py', '.java', '.cpp', '.c', '.h', '.css', '.scss', '.html', '.xml', '.json', '.yaml', '.yml'];
    return codeExtensions.some(ext => filename.toLowerCase().endsWith(ext));
  }

  private getCodeLanguage(extension: string): string {
    const languageMap: Record<string, string> = {
      '.js': 'javascript',
      '.ts': 'typescript',
      '.jsx': 'javascript',
      '.tsx': 'typescript',
      '.py': 'python',
      '.java': 'java',
      '.cpp': 'cpp',
      '.c': 'c',
      '.h': 'c',
      '.css': 'css',
      '.scss': 'scss',
      '.html': 'html',
      '.xml': 'xml',
      '.json': 'json',
      '.yaml': 'yaml',
      '.yml': 'yaml',
    };
    
    return languageMap[extension] || 'unknown';
  }

  private detectCodePatterns(content: string): boolean {
    const codePatterns = [
      /function\s+\w+\s*\(/,
      /class\s+\w+/,
      /import\s+.*from/,
      /def\s+\w+\s*\(/,
      /public\s+class/,
      /#include\s*</,
    ];
    
    return codePatterns.some(pattern => pattern.test(content));
  }

  private detectComments(content: string, extension: string): boolean {
    const commentPatterns: Record<string, RegExp[]> = {
      '.js': [/\/\//, /\/\*.*\*\//],
      '.ts': [/\/\//, /\/\*.*\*\//],
      '.py': [/#/],
      '.java': [/\/\//, /\/\*.*\*\//],
      '.cpp': [/\/\//, /\/\*.*\*\//],
      '.c': [/\/\//, /\/\*.*\*\//],
    };
    
    const patterns = commentPatterns[extension] || [];
    return patterns.some(pattern => pattern.test(content));
  }

  private detectImports(content: string, extension: string): boolean {
    const importPatterns: Record<string, RegExp[]> = {
      '.js': [/import\s+.*from/, /require\s*\(/],
      '.ts': [/import\s+.*from/, /require\s*\(/],
      '.py': [/import\s+/, /from\s+.*import/],
      '.java': [/import\s+/],
      '.cpp': [/#include\s*</],
      '.c': [/#include\s*</],
    };
    
    const patterns = importPatterns[extension] || [];
    return patterns.some(pattern => pattern.test(content));
  }

  private estimateComplexity(content: string): number {
    const complexityIndicators = [
      /if\s*\(/g,
      /for\s*\(/g,
      /while\s*\(/g,
      /switch\s*\(/g,
      /function\s+\w+/g,
      /class\s+\w+/g,
    ];
    
    let complexity = 0;
    complexityIndicators.forEach(pattern => {
      const matches = content.match(pattern);
      complexity += matches ? matches.length : 0;
    });
    
    return complexity;
  }

  private calculateRankingScore(result: ProcessingResult): number {
    let score = 0;
    
    // Base score for having extractable text
    if (result.extractedText) {
      score += 10;
    }
    
    // Bonus for word count
    if (result.wordCount) {
      score += Math.min(result.wordCount / 100, 5);
    }
    
    // Bonus for metadata richness
    score += Object.keys(result.metadata).length;
    
    // Bonus for having tags
    score += result.tags.length;
    
    return Math.round(score);
  }
}

export const fileProcessor = new FileProcessor();
