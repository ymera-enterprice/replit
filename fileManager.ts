import fs from 'fs/promises';
import path from 'path';
import crypto from 'crypto';
import { pipeline } from 'stream/promises';
import { createReadStream, createWriteStream } from 'fs';
import { Request } from 'express';
import { storage } from '../storage';
import { File, InsertFile } from '@shared/schema';
import { StructuredLogger } from '../middleware/logging';
import { AppError, ValidationError } from '../middleware/errorHandler';

export interface FileUploadOptions {
  maxSize?: number;
  allowedMimeTypes?: string[];
  allowedExtensions?: string[];
  generateThumbnails?: boolean;
  virusScan?: boolean;
  encrypt?: boolean;
}

export interface FileMetadata {
  originalName: string;
  mimeType: string;
  size: number;
  hash: string;
  uploadedBy?: string;
  tags?: string[];
  [key: string]: any;
}

export interface VirusScanResult {
  isClean: boolean;
  threats?: string[];
  scanner: string;
  scanTime: Date;
}

class FileValidator {
  private static readonly DANGEROUS_EXTENSIONS = [
    '.exe', '.bat', '.cmd', '.scr', '.pif', '.com', '.cpl', '.dll',
    '.msi', '.msp', '.mst', '.vbs', '.vbe', '.js', '.jse', '.wsf',
    '.wsh', '.ps1', '.ps1xml', '.ps2', '.ps2xml', '.psc1', '.psc2'
  ];

  private static readonly MAGIC_NUMBERS: { [key: string]: number[] } = {
    'image/jpeg': [0xFF, 0xD8, 0xFF],
    'image/png': [0x89, 0x50, 0x4E, 0x47],
    'image/gif': [0x47, 0x49, 0x46],
    'application/pdf': [0x25, 0x50, 0x44, 0x46],
    'application/zip': [0x50, 0x4B, 0x03, 0x04],
    'text/plain': [], // No magic number for text files
  };

  static async validateFile(filePath: string, metadata: FileMetadata, options: FileUploadOptions): Promise<void> {
    // Check file size
    if (options.maxSize && metadata.size > options.maxSize) {
      throw new ValidationError(`File size exceeds maximum allowed size of ${options.maxSize} bytes`);
    }

    // Check file extension
    const ext = path.extname(metadata.originalName).toLowerCase();
    
    if (this.DANGEROUS_EXTENSIONS.includes(ext)) {
      throw new ValidationError(`File extension '${ext}' is not allowed for security reasons`);
    }

    if (options.allowedExtensions && !options.allowedExtensions.includes(ext)) {
      throw new ValidationError(`File extension '${ext}' is not allowed`);
    }

    // Check MIME type
    if (options.allowedMimeTypes && !options.allowedMimeTypes.includes(metadata.mimeType)) {
      throw new ValidationError(`MIME type '${metadata.mimeType}' is not allowed`);
    }

    // Validate file content matches MIME type
    await this.validateMagicNumbers(filePath, metadata.mimeType);

    // Check for embedded scripts in images
    if (metadata.mimeType.startsWith('image/')) {
      await this.scanForEmbeddedThreats(filePath);
    }
  }

  private static async validateMagicNumbers(filePath: string, mimeType: string): Promise<void> {
    const expectedMagicNumbers = this.MAGIC_NUMBERS[mimeType];
    if (!expectedMagicNumbers || expectedMagicNumbers.length === 0) {
      return; // Skip validation for files without magic numbers
    }

    try {
      const buffer = Buffer.alloc(expectedMagicNumbers.length);
      const file = await fs.open(filePath, 'r');
      await file.read(buffer, 0, expectedMagicNumbers.length, 0);
      await file.close();

      for (let i = 0; i < expectedMagicNumbers.length; i++) {
        if (buffer[i] !== expectedMagicNumbers[i]) {
          throw new ValidationError('File content does not match declared MIME type');
        }
      }
    } catch (error) {
      if (error instanceof ValidationError) throw error;
      throw new ValidationError('Unable to validate file content');
    }
  }

  private static async scanForEmbeddedThreats(filePath: string): Promise<void> {
    try {
      const content = await fs.readFile(filePath);
      const contentString = content.toString('binary');

      // Look for suspicious patterns
      const suspiciousPatterns = [
        /<script/i,
        /javascript:/i,
        /eval\(/i,
        /document\.write/i,
        /window\.location/i,
        /%3Cscript/i, // URL encoded script tags
      ];

      for (const pattern of suspiciousPatterns) {
        if (pattern.test(contentString)) {
          throw new ValidationError('File contains potentially malicious content');
        }
      }
    } catch (error) {
      if (error instanceof ValidationError) throw error;
      StructuredLogger.warn('Could not scan file for embedded threats', { filePath, error: error.message });
    }
  }
}

class VirusScanner {
  static async scanFile(filePath: string): Promise<VirusScanResult> {
    // In production, this would integrate with ClamAV, Windows Defender, or other antivirus solutions
    // For now, we'll simulate a virus scan with basic heuristics
    
    const startTime = new Date();
    
    try {
      const stats = await fs.stat(filePath);
      const content = await fs.readFile(filePath);
      
      // Simple heuristic checks
      const threats: string[] = [];
      
      // Check for suspicious file patterns
      if (this.containsSuspiciousContent(content)) {
        threats.push('Heuristic.Suspicious.Content');
      }
      
      // Check file size anomalies
      if (stats.size > 100 * 1024 * 1024) { // 100MB
        threats.push('Heuristic.Large.File');
      }
      
      const result: VirusScanResult = {
        isClean: threats.length === 0,
        threats: threats.length > 0 ? threats : undefined,
        scanner: 'YMERA-Heuristic-Scanner',
        scanTime: new Date(),
      };
      
      StructuredLogger.info('Virus scan completed', {
        filePath,
        isClean: result.isClean,
        threats: result.threats,
        duration: new Date().getTime() - startTime.getTime(),
      });
      
      return result;
    } catch (error) {
      StructuredLogger.error('Virus scan failed', error, { filePath });
      
      return {
        isClean: false,
        threats: ['Scanner.Error'],
        scanner: 'YMERA-Heuristic-Scanner',
        scanTime: new Date(),
      };
    }
  }
  
  private static containsSuspiciousContent(content: Buffer): boolean {
    const suspiciousSignatures = [
      Buffer.from('X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*'), // EICAR test string
      Buffer.from('eval('), // JavaScript eval
      Buffer.from('exec('), // Python exec
      Buffer.from('system('), // System commands
    ];
    
    for (const signature of suspiciousSignatures) {
      if (content.includes(signature)) {
        return true;
      }
    }
    
    return false;
  }
}

export class FileManager {
  private readonly uploadDir: string;
  private readonly tempDir: string;
  
  constructor() {
    this.uploadDir = process.env.UPLOAD_DIR || path.join(process.cwd(), 'uploads');
    this.tempDir = process.env.TEMP_DIR || path.join(process.cwd(), 'temp');
    this.ensureDirectories();
  }
  
  private async ensureDirectories(): Promise<void> {
    try {
      await fs.mkdir(this.uploadDir, { recursive: true });
      await fs.mkdir(this.tempDir, { recursive: true });
    } catch (error) {
      StructuredLogger.error('Failed to create upload directories', error);
      throw new AppError('File system initialization failed');
    }
  }
  
  async uploadFile(
    fileData: Buffer | NodeJS.ReadableStream,
    metadata: FileMetadata,
    options: FileUploadOptions = {}
  ): Promise<File> {
    const uploadId = crypto.randomUUID();
    const tempPath = path.join(this.tempDir, `upload_${uploadId}_${Date.now()}`);
    
    try {
      // Write file to temporary location
      if (Buffer.isBuffer(fileData)) {
        await fs.writeFile(tempPath, fileData);
      } else {
        await pipeline(fileData, createWriteStream(tempPath));
      }
      
      // Update metadata with actual file size and hash
      const stats = await fs.stat(tempPath);
      metadata.size = stats.size;
      metadata.hash = await this.calculateFileHash(tempPath);
      
      // Validate file
      await FileValidator.validateFile(tempPath, metadata, options);
      
      // Virus scan if enabled
      let virusScanResult: VirusScanResult | undefined;
      if (options.virusScan !== false) {
        virusScanResult = await VirusScanner.scanFile(tempPath);
        if (!virusScanResult.isClean) {
          throw new ValidationError(`File failed virus scan: ${virusScanResult.threats?.join(', ')}`);
        }
      }
      
      // Generate final filename and path
      const ext = path.extname(metadata.originalName);
      const filename = `${uploadId}${ext}`;
      const finalPath = path.join(this.uploadDir, filename);
      
      // Move file to final location
      await fs.rename(tempPath, finalPath);
      
      // Encrypt file if requested
      if (options.encrypt) {
        await this.encryptFile(finalPath);
      }
      
      // Save file record to database
      const fileRecord = await storage.createFile({
        filename,
        originalName: metadata.originalName,
        mimeType: metadata.mimeType,
        size: metadata.size,
        path: finalPath,
        hash: metadata.hash,
        uploadedBy: metadata.uploadedBy,
        metadata: {
          ...metadata,
          uploadOptions: options,
        },
        tags: metadata.tags || [],
        virusScanStatus: virusScanResult ? 'clean' : 'skipped',
        virusScanResult,
      });
      
      StructuredLogger.info('File uploaded successfully', {
        fileId: fileRecord.id,
        filename: fileRecord.filename,
        originalName: fileRecord.originalName,
        size: fileRecord.size,
        uploadedBy: fileRecord.uploadedBy,
      });
      
      return fileRecord;
      
    } catch (error) {
      // Clean up temporary file
      try {
        await fs.unlink(tempPath);
      } catch (cleanupError) {
        StructuredLogger.warn('Failed to clean up temporary file', { tempPath });
      }
      
      throw error;
    }
  }
  
  async downloadFile(fileId: string, userId?: string): Promise<NodeJS.ReadableStream> {
    const fileRecord = await storage.getFileById(fileId);
    if (!fileRecord) {
      throw new AppError('File not found', 404);
    }
    
    // Check permissions
    if (!fileRecord.isPublic && fileRecord.uploadedBy !== userId) {
      throw new AppError('Access denied', 403);
    }
    
    // Check if file exists on disk
    try {
      await fs.access(fileRecord.path);
    } catch (error) {
      StructuredLogger.error('File not found on disk', error, { fileId, path: fileRecord.path });
      throw new AppError('File not available', 404);
    }
    
    // Increment download count
    await storage.updateFile(fileId, {
      downloadCount: fileRecord.downloadCount + 1,
    });
    
    StructuredLogger.info('File download started', {
      fileId,
      filename: fileRecord.filename,
      downloadedBy: userId,
      downloadCount: fileRecord.downloadCount + 1,
    });
    
    // Decrypt file if encrypted
    if (fileRecord.metadata?.encrypted) {
      return this.createDecryptedStream(fileRecord.path);
    }
    
    return createReadStream(fileRecord.path);
  }
  
  async deleteFile(fileId: string, userId?: string): Promise<void> {
    const fileRecord = await storage.getFileById(fileId);
    if (!fileRecord) {
      throw new AppError('File not found', 404);
    }
    
    // Check permissions
    if (fileRecord.uploadedBy !== userId) {
      throw new AppError('Access denied', 403);
    }
    
    try {
      // Delete file from disk
      await fs.unlink(fileRecord.path);
    } catch (error) {
      StructuredLogger.warn('Failed to delete file from disk', { fileId, path: fileRecord.path });
    }
    
    // Delete file record from database
    await storage.deleteFile(fileId);
    
    StructuredLogger.info('File deleted successfully', {
      fileId,
      filename: fileRecord.filename,
      deletedBy: userId,
    });
  }
  
  async getFileMetadata(fileId: string, userId?: string): Promise<File> {
    const fileRecord = await storage.getFileById(fileId);
    if (!fileRecord) {
      throw new AppError('File not found', 404);
    }
    
    // Check permissions for private files
    if (!fileRecord.isPublic && fileRecord.uploadedBy !== userId) {
      throw new AppError('Access denied', 403);
    }
    
    return fileRecord;
  }
  
  async searchFiles(query: {
    tags?: string[];
    mimeType?: string;
    uploadedBy?: string;
    filename?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ files: File[]; total: number }> {
    return storage.searchFiles(query);
  }
  
  private async calculateFileHash(filePath: string): Promise<string> {
    const hash = crypto.createHash('sha256');
    const stream = createReadStream(filePath);
    
    for await (const chunk of stream) {
      hash.update(chunk);
    }
    
    return hash.digest('hex');
  }
  
  private async encryptFile(filePath: string): Promise<void> {
    // Simple encryption implementation
    // In production, use proper encryption libraries and key management
    const key = crypto.randomBytes(32);
    const iv = crypto.randomBytes(16);
    
    const input = createReadStream(filePath);
    const output = createWriteStream(`${filePath}.encrypted`);
    const cipher = crypto.createCipher('aes-256-cbc', key);
    
    await pipeline(input, cipher, output);
    
    // Replace original file with encrypted version
    await fs.unlink(filePath);
    await fs.rename(`${filePath}.encrypted`, filePath);
    
    // Store encryption metadata (in production, use secure key storage)
    const metadata = { key: key.toString('hex'), iv: iv.toString('hex') };
    await fs.writeFile(`${filePath}.meta`, JSON.stringify(metadata));
  }
  
  private createDecryptedStream(filePath: string): NodeJS.ReadableStream {
    // In production, implement proper decryption
    // For now, return the file as-is
    return createReadStream(filePath);
  }
  
  async getStorageStats(): Promise<{
    totalFiles: number;
    totalSize: number;
    byMimeType: Record<string, { count: number; size: number }>;
  }> {
    return storage.getFileStorageStats();
  }
  
  async cleanupExpiredFiles(): Promise<void> {
    // Clean up temporary files older than 24 hours
    const tempFiles = await fs.readdir(this.tempDir);
    const now = Date.now();
    
    for (const file of tempFiles) {
      const filePath = path.join(this.tempDir, file);
      try {
        const stats = await fs.stat(filePath);
        const age = now - stats.mtime.getTime();
        
        if (age > 24 * 60 * 60 * 1000) { // 24 hours
          await fs.unlink(filePath);
          StructuredLogger.debug('Cleaned up expired temp file', { filePath });
        }
      } catch (error) {
        StructuredLogger.warn('Failed to clean up temp file', { filePath, error: error.message });
      }
    }
  }
}
