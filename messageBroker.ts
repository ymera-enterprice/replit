import { EventEmitter } from 'events';
import crypto from 'crypto';
import { storage } from '../storage';
import { Message, InsertMessage, Agent } from '@shared/schema';
import { StructuredLogger } from '../middleware/logging';
import { AppError } from '../middleware/errorHandler';

export interface MessagePayload {
  type: string;
  data: any;
  correlationId?: string;
  replyTo?: string;
  timestamp?: Date;
}

export interface QueueConfig {
  name: string;
  maxSize?: number;
  deadLetterQueue?: string;
  retryDelay?: number;
  maxRetries?: number;
  priority?: boolean;
}

export interface MessageOptions {
  priority?: number;
  delay?: number;
  ttl?: number;
  persistent?: boolean;
  correlationId?: string;
  replyTo?: string;
}

interface QueueStats {
  name: string;
  size: number;
  processing: number;
  completed: number;
  failed: number;
  deadLetters: number;
}

export class MessageBroker extends EventEmitter {
  private queues: Map<string, QueueConfig> = new Map();
  private consumers: Map<string, Array<{
    handler: (message: Message) => Promise<void>;
    agentId?: string;
    concurrency: number;
    processing: Set<string>;
  }>> = new Map();
  private processingInterval?: NodeJS.Timeout;
  private readonly encryptionKey: Buffer;
  
  constructor() {
    super();
    this.encryptionKey = this.getOrCreateEncryptionKey();
    this.startMessageProcessor();
    
    // Setup default queues
    this.createQueue('default');
    this.createQueue('high-priority', { priority: true, maxRetries: 5 });
    this.createQueue('dead-letter', { maxRetries: 0 });
  }
  
  private getOrCreateEncryptionKey(): Buffer {
    const key = process.env.MESSAGE_ENCRYPTION_KEY;
    if (key) {
      return Buffer.from(key, 'hex');
    }
    
    // Generate new key (in production, store this securely)
    const newKey = crypto.randomBytes(32);
    StructuredLogger.warn('Generated new message encryption key - store this securely', {
      key: newKey.toString('hex')
    });
    return newKey;
  }
  
  createQueue(name: string, config: Partial<QueueConfig> = {}): void {
    const queueConfig: QueueConfig = {
      name,
      maxSize: 10000,
      retryDelay: 60000, // 1 minute
      maxRetries: 3,
      priority: false,
      ...config,
    };
    
    this.queues.set(name, queueConfig);
    this.consumers.set(name, []);
    
    StructuredLogger.info('Queue created', { 
      name, 
      config: queueConfig 
    });
  }
  
  async publish(
    queue: string, 
    payload: MessagePayload, 
    options: MessageOptions = {}
  ): Promise<string> {
    const queueConfig = this.queues.get(queue);
    if (!queueConfig) {
      throw new AppError(`Queue '${queue}' not found`, 404);
    }
    
    // Encrypt payload
    const encryptedPayload = this.encryptPayload(payload);
    
    // Calculate scheduled time
    const scheduledAt = options.delay 
      ? new Date(Date.now() + options.delay)
      : new Date();
    
    // Create message
    const message = await storage.createMessage({
      queue,
      payload: encryptedPayload,
      priority: options.priority || 1,
      maxAttempts: queueConfig.maxRetries || 3,
      scheduledAt,
    });
    
    StructuredLogger.info('Message published', {
      messageId: message.id,
      queue,
      priority: message.priority,
      scheduledAt: message.scheduledAt,
      correlationId: payload.correlationId,
    });
    
    this.emit('message:published', { message, queue });
    return message.id;
  }
  
  async consume(
    queue: string,
    handler: (message: Message) => Promise<void>,
    options: { 
      agentId?: string; 
      concurrency?: number 
    } = {}
  ): Promise<void> {
    const queueConfig = this.queues.get(queue);
    if (!queueConfig) {
      throw new AppError(`Queue '${queue}' not found`, 404);
    }
    
    const consumers = this.consumers.get(queue)!;
    const consumer = {
      handler,
      agentId: options.agentId,
      concurrency: options.concurrency || 1,
      processing: new Set<string>(),
    };
    
    consumers.push(consumer);
    
    StructuredLogger.info('Consumer registered', {
      queue,
      agentId: options.agentId,
      concurrency: consumer.concurrency,
    });
  }
  
  private startMessageProcessor(): void {
    if (this.processingInterval) {
      clearInterval(this.processingInterval);
    }
    
    this.processingInterval = setInterval(async () => {
      await this.processMessages();
    }, 1000); // Process every second
  }
  
  private async processMessages(): Promise<void> {
    for (const [queueName, queueConfig] of this.queues.entries()) {
      const consumers = this.consumers.get(queueName) || [];
      
      for (const consumer of consumers) {
        if (consumer.processing.size >= consumer.concurrency) {
          continue; // Consumer at capacity
        }
        
        const availableSlots = consumer.concurrency - consumer.processing.size;
        const messages = await storage.getPendingMessages(queueName, availableSlots);
        
        for (const message of messages) {
          // Mark message as processing
          await storage.updateMessage(message.id, {
            status: 'processing',
            processedBy: consumer.agentId,
            processedAt: new Date(),
          });
          
          consumer.processing.add(message.id);
          
          // Process message asynchronously
          this.processMessage(message, consumer, queueConfig)
            .finally(() => {
              consumer.processing.delete(message.id);
            });
        }
      }
    }
  }
  
  private async processMessage(
    message: Message,
    consumer: {
      handler: (message: Message) => Promise<void>;
      agentId?: string;
    },
    queueConfig: QueueConfig
  ): Promise<void> {
    try {
      // Decrypt payload
      const decryptedMessage = {
        ...message,
        payload: this.decryptPayload(message.payload),
      };
      
      // Execute handler
      await consumer.handler(decryptedMessage);
      
      // Mark as completed
      await storage.updateMessage(message.id, {
        status: 'completed',
        processedAt: new Date(),
      });
      
      StructuredLogger.info('Message processed successfully', {
        messageId: message.id,
        queue: message.queue,
        agentId: consumer.agentId,
      });
      
      this.emit('message:completed', { message, agentId: consumer.agentId });
      
    } catch (error) {
      StructuredLogger.error('Message processing failed', error, {
        messageId: message.id,
        queue: message.queue,
        agentId: consumer.agentId,
        attempts: message.attempts,
      });
      
      await this.handleFailedMessage(message, queueConfig, error);
    }
  }
  
  private async handleFailedMessage(
    message: Message,
    queueConfig: QueueConfig,
    error: any
  ): Promise<void> {
    const newAttempts = message.attempts + 1;
    
    if (newAttempts >= message.maxAttempts) {
      // Move to dead letter queue
      const deadLetterQueue = queueConfig.deadLetterQueue || 'dead-letter';
      
      await storage.updateMessage(message.id, {
        status: 'failed',
        queue: deadLetterQueue,
        attempts: newAttempts,
        error: error.message,
        failedAt: new Date(),
      });
      
      StructuredLogger.warn('Message moved to dead letter queue', {
        messageId: message.id,
        originalQueue: message.queue,
        deadLetterQueue,
        error: error.message,
      });
      
      this.emit('message:dead-letter', { message, error });
    } else {
      // Schedule retry
      const retryDelay = queueConfig.retryDelay || 60000;
      const scheduledAt = new Date(Date.now() + (retryDelay * newAttempts));
      
      await storage.updateMessage(message.id, {
        status: 'pending',
        attempts: newAttempts,
        scheduledAt,
        error: error.message,
      });
      
      StructuredLogger.info('Message scheduled for retry', {
        messageId: message.id,
        queue: message.queue,
        attempt: newAttempts,
        scheduledAt,
      });
      
      this.emit('message:retry', { message, attempt: newAttempts });
    }
  }
  
  private encryptPayload(payload: MessagePayload): any {
    try {
      const iv = crypto.randomBytes(16);
      const cipher = crypto.createCipher('aes-256-gcm', this.encryptionKey);
      
      let encrypted = cipher.update(JSON.stringify(payload), 'utf8', 'hex');
      encrypted += cipher.final('hex');
      
      const authTag = cipher.getAuthTag();
      
      return {
        encrypted,
        iv: iv.toString('hex'),
        authTag: authTag.toString('hex'),
        algorithm: 'aes-256-gcm',
      };
    } catch (error) {
      StructuredLogger.error('Failed to encrypt message payload', error);
      throw new AppError('Message encryption failed');
    }
  }
  
  private decryptPayload(encryptedData: any): MessagePayload {
    try {
      if (!encryptedData.encrypted) {
        // Payload is not encrypted (legacy messages)
        return encryptedData;
      }
      
      const decipher = crypto.createDecipher('aes-256-gcm', this.encryptionKey);
      decipher.setAuthTag(Buffer.from(encryptedData.authTag, 'hex'));
      
      let decrypted = decipher.update(encryptedData.encrypted, 'hex', 'utf8');
      decrypted += decipher.final('utf8');
      
      return JSON.parse(decrypted);
    } catch (error) {
      StructuredLogger.error('Failed to decrypt message payload', error);
      throw new AppError('Message decryption failed');
    }
  }
  
  async getQueueStats(queueName?: string): Promise<QueueStats[]> {
    const queues = queueName ? [queueName] : Array.from(this.queues.keys());
    const stats: QueueStats[] = [];
    
    for (const queue of queues) {
      const queueStats = await storage.getQueueStats(queue);
      stats.push({
        name: queue,
        size: queueStats.pending,
        processing: queueStats.processing,
        completed: queueStats.completed,
        failed: queueStats.failed,
        deadLetters: queueStats.deadLetters,
      });
    }
    
    return stats;
  }
  
  async purgeQueue(queueName: string): Promise<number> {
    const queueConfig = this.queues.get(queueName);
    if (!queueConfig) {
      throw new AppError(`Queue '${queueName}' not found`, 404);
    }
    
    const purgedCount = await storage.purgeQueue(queueName);
    
    StructuredLogger.info('Queue purged', {
      queue: queueName,
      purgedCount,
    });
    
    this.emit('queue:purged', { queue: queueName, count: purgedCount });
    return purgedCount;
  }
  
  async reprocessDeadLetters(fromQueue: string, toQueue: string): Promise<number> {
    const deadLetters = await storage.getDeadLetterMessages(fromQueue);
    let reprocessedCount = 0;
    
    for (const message of deadLetters) {
      try {
        // Reset message for reprocessing
        await storage.updateMessage(message.id, {
          queue: toQueue,
          status: 'pending',
          attempts: 0,
          error: null,
          failedAt: null,
          scheduledAt: new Date(),
        });
        
        reprocessedCount++;
      } catch (error) {
        StructuredLogger.error('Failed to reprocess dead letter', error, {
          messageId: message.id,
        });
      }
    }
    
    StructuredLogger.info('Dead letters reprocessed', {
      fromQueue,
      toQueue,
      count: reprocessedCount,
    });
    
    return reprocessedCount;
  }
  
  async sendResponse(
    originalMessage: Message,
    responsePayload: any
  ): Promise<void> {
    const originalPayload = this.decryptPayload(originalMessage.payload);
    
    if (originalPayload.replyTo) {
      await this.publish(originalPayload.replyTo, {
        type: 'response',
        data: responsePayload,
        correlationId: originalPayload.correlationId,
        timestamp: new Date(),
      });
    }
  }
  
  async sendRequest(
    queue: string,
    payload: any,
    replyQueue: string,
    timeout: number = 30000
  ): Promise<any> {
    const correlationId = crypto.randomUUID();
    
    // Setup response handler
    const responsePromise = new Promise((resolve, reject) => {
      let responseHandler: (message: Message) => Promise<void>;
      let timeoutId: NodeJS.Timeout;
      
      const cleanup = () => {
        clearTimeout(timeoutId);
        const consumers = this.consumers.get(replyQueue) || [];
        const index = consumers.findIndex(c => c.handler === responseHandler);
        if (index >= 0) {
          consumers.splice(index, 1);
        }
      };
      
      responseHandler = async (message: Message) => {
        const decryptedPayload = this.decryptPayload(message.payload);
        if (decryptedPayload.correlationId === correlationId) {
          cleanup();
          resolve(decryptedPayload.data);
        }
      };
      
      timeoutId = setTimeout(() => {
        cleanup();
        reject(new AppError('Request timeout', 408));
      }, timeout);
      
      this.consume(replyQueue, responseHandler);
    });
    
    // Send request
    await this.publish(queue, {
      type: 'request',
      data: payload,
      correlationId,
      replyTo: replyQueue,
      timestamp: new Date(),
    });
    
    return responsePromise;
  }
  
  destroy(): void {
    if (this.processingInterval) {
      clearInterval(this.processingInterval);
    }
    
    this.queues.clear();
    this.consumers.clear();
    this.removeAllListeners();
    
    StructuredLogger.info('Message broker destroyed');
  }
}

// Singleton instance
export const messageBroker = new MessageBroker();
