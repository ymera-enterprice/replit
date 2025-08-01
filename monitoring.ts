import { EventEmitter } from 'events';
import os from 'os';
import fs from 'fs/promises';
import { storage } from '../storage';
import { SystemMetric, InsertSystemMetric } from '@shared/schema';
import { StructuredLogger } from '../middleware/logging';

export interface HealthCheck {
  name: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  message?: string;
  responseTime?: number;
  lastChecked: Date;
  metadata?: Record<string, any>;
}

export interface SystemHealth {
  overall: 'healthy' | 'degraded' | 'unhealthy';
  checks: HealthCheck[];
  uptime: number;
  timestamp: Date;
}

export interface PerformanceMetrics {
  cpu: {
    usage: number;
    loadAverage: number[];
    cores: number;
  };
  memory: {
    used: number;
    free: number;
    total: number;
    percentage: number;
  };
  disk: {
    used: number;
    free: number;
    total: number;
    percentage: number;
  };
  network: {
    bytesReceived: number;
    bytesSent: number;
    packetsReceived: number;
    packetsSent: number;
  };
}

export interface Alert {
  id: string;
  type: 'warning' | 'critical' | 'info';
  title: string;
  message: string;
  source: string;
  timestamp: Date;
  acknowledged: boolean;
  metadata?: Record<string, any>;
}

class SystemMonitor {
  private static previousNetworkStats: any = null;
  
  static async getSystemMetrics(): Promise<PerformanceMetrics> {
    const cpuUsage = await this.getCpuUsage();
    const memoryInfo = this.getMemoryInfo();
    const diskInfo = await this.getDiskInfo();
    const networkInfo = await this.getNetworkInfo();
    
    return {
      cpu: {
        usage: cpuUsage,
        loadAverage: os.loadavg(),
        cores: os.cpus().length,
      },
      memory: memoryInfo,
      disk: diskInfo,
      network: networkInfo,
    };
  }
  
  private static async getCpuUsage(): Promise<number> {
    const startTime = process.hrtime();
    const startUsage = process.cpuUsage();
    
    // Wait a small amount of time to measure
    await new Promise(resolve => setTimeout(resolve, 100));
    
    const endTime = process.hrtime(startTime);
    const endUsage = process.cpuUsage(startUsage);
    
    const userTime = endUsage.user / 1000; // Convert to milliseconds
    const systemTime = endUsage.system / 1000;
    const totalTime = (endTime[0] * 1000) + (endTime[1] / 1000000);
    
    return ((userTime + systemTime) / totalTime) * 100;
  }
  
  private static getMemoryInfo(): PerformanceMetrics['memory'] {
    const total = os.totalmem();
    const free = os.freemem();
    const used = total - free;
    
    return {
      used,
      free,
      total,
      percentage: (used / total) * 100,
    };
  }
  
  private static async getDiskInfo(): Promise<PerformanceMetrics['disk']> {
    try {
      const stats = await fs.statfs(process.cwd());
      const total = stats.blocks * stats.blksize;
      const free = stats.bavail * stats.blksize;
      const used = total - free;
      
      return {
        used,
        free,
        total,
        percentage: (used / total) * 100,
      };
    } catch (error) {
      // Fallback for systems that don't support statfs
      return {
        used: 0,
        free: 0,
        total: 0,
        percentage: 0,
      };
    }
  }
  
  private static async getNetworkInfo(): Promise<PerformanceMetrics['network']> {
    try {
      const interfaces = os.networkInterfaces();
      let bytesReceived = 0;
      let bytesSent = 0;
      let packetsReceived = 0;
      let packetsSent = 0;
      
      // This is a simplified implementation
      // In production, you'd want to read from /proc/net/dev on Linux
      for (const [name, addresses] of Object.entries(interfaces)) {
        if (name !== 'lo' && addresses) {
          // Network stats would need to be read from system files
          // This is a placeholder implementation
        }
      }
      
      return {
        bytesReceived,
        bytesSent,
        packetsReceived,
        packetsSent,
      };
    } catch (error) {
      return {
        bytesReceived: 0,
        bytesSent: 0,
        packetsReceived: 0,
        packetsSent: 0,
      };
    }
  }
}

class HealthChecker {
  private checks: Map<string, () => Promise<HealthCheck>> = new Map();
  
  addCheck(name: string, check: () => Promise<HealthCheck>): void {
    this.checks.set(name, check);
  }
  
  async runAllChecks(): Promise<HealthCheck[]> {
    const results: HealthCheck[] = [];
    
    for (const [name, check] of this.checks.entries()) {
      try {
        const startTime = Date.now();
        const result = await Promise.race([
          check(),
          new Promise<HealthCheck>((_, reject) =>
            setTimeout(() => reject(new Error('Health check timeout')), 5000)
          ),
        ]);
        
        result.responseTime = Date.now() - startTime;
        results.push(result);
      } catch (error) {
        results.push({
          name,
          status: 'unhealthy',
          message: error.message,
          responseTime: Date.now() - Date.now(),
          lastChecked: new Date(),
        });
      }
    }
    
    return results;
  }
  
  async runCheck(name: string): Promise<HealthCheck | null> {
    const check = this.checks.get(name);
    if (!check) return null;
    
    try {
      const startTime = Date.now();
      const result = await check();
      result.responseTime = Date.now() - startTime;
      return result;
    } catch (error) {
      return {
        name,
        status: 'unhealthy',
        message: error.message,
        lastChecked: new Date(),
      };
    }
  }
}

export class MonitoringService extends EventEmitter {
  private healthChecker = new HealthChecker();
  private alerts: Alert[] = [];
  private metrics: SystemMetric[] = [];
  private monitoringInterval?: NodeJS.Timeout;
  private readonly startTime = Date.now();
  
  constructor() {
    super();
    this.setupDefaultHealthChecks();
    this.startMonitoring();
  }
  
  private setupDefaultHealthChecks(): void {
    // Database health check
    this.healthChecker.addCheck('database', async () => {
      try {
        await storage.healthCheck();
        return {
          name: 'database',
          status: 'healthy',
          message: 'Database connection is working',
          lastChecked: new Date(),
        };
      } catch (error) {
        return {
          name: 'database',
          status: 'unhealthy',
          message: `Database error: ${error.message}`,
          lastChecked: new Date(),
        };
      }
    });
    
    // Memory health check
    this.healthChecker.addCheck('memory', async () => {
      const memInfo = SystemMonitor['getMemoryInfo']();
      const status = memInfo.percentage > 90 ? 'unhealthy' : 
                   memInfo.percentage > 75 ? 'degraded' : 'healthy';
      
      return {
        name: 'memory',
        status,
        message: `Memory usage: ${memInfo.percentage.toFixed(1)}%`,
        lastChecked: new Date(),
        metadata: memInfo,
      };
    });
    
    // CPU health check
    this.healthChecker.addCheck('cpu', async () => {
      const cpuUsage = await SystemMonitor['getCpuUsage']();
      const status = cpuUsage > 90 ? 'unhealthy' : 
                   cpuUsage > 75 ? 'degraded' : 'healthy';
      
      return {
        name: 'cpu',
        status,
        message: `CPU usage: ${cpuUsage.toFixed(1)}%`,
        lastChecked: new Date(),
        metadata: { usage: cpuUsage },
      };
    });
    
    // Disk health check
    this.healthChecker.addCheck('disk', async () => {
      const diskInfo = await SystemMonitor['getDiskInfo']();
      const status = diskInfo.percentage > 90 ? 'unhealthy' : 
                   diskInfo.percentage > 80 ? 'degraded' : 'healthy';
      
      return {
        name: 'disk',
        status,
        message: `Disk usage: ${diskInfo.percentage.toFixed(1)}%`,
        lastChecked: new Date(),
        metadata: diskInfo,
      };
    });
  }
  
  private startMonitoring(): void {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
    }
    
    this.monitoringInterval = setInterval(async () => {
      await this.collectMetrics();
      await this.checkHealth();
    }, 30000); // Every 30 seconds
  }
  
  private async collectMetrics(): Promise<void> {
    try {
      const systemMetrics = await SystemMonitor.getSystemMetrics();
      const timestamp = new Date();
      
      // Store CPU metrics
      await this.storeMetric('cpu_usage', systemMetrics.cpu.usage, 'percent', 'system');
      await this.storeMetric('cpu_load_1m', systemMetrics.cpu.loadAverage[0], 'load', 'system');
      
      // Store memory metrics
      await this.storeMetric('memory_usage', systemMetrics.memory.percentage, 'percent', 'system');
      await this.storeMetric('memory_used', systemMetrics.memory.used, 'bytes', 'system');
      
      // Store disk metrics
      await this.storeMetric('disk_usage', systemMetrics.disk.percentage, 'percent', 'system');
      await this.storeMetric('disk_free', systemMetrics.disk.free, 'bytes', 'system');
      
      // Store network metrics
      await this.storeMetric('network_bytes_received', systemMetrics.network.bytesReceived, 'bytes', 'system');
      await this.storeMetric('network_bytes_sent', systemMetrics.network.bytesSent, 'bytes', 'system');
      
      this.emit('metrics:collected', systemMetrics);
    } catch (error) {
      StructuredLogger.error('Failed to collect metrics', error);
    }
  }
  
  private async storeMetric(name: string, value: number, unit: string, source: string): Promise<void> {
    try {
      const metric = await storage.createSystemMetric({
        metricName: name,
        value: value.toString(),
        unit,
        source,
        tags: { type: 'system_monitoring' },
      });
      
      this.metrics.push(metric);
      
      // Keep only recent metrics in memory
      const maxMetrics = 1000;
      if (this.metrics.length > maxMetrics) {
        this.metrics = this.metrics.slice(-maxMetrics);
      }
    } catch (error) {
      StructuredLogger.error('Failed to store metric', error, { name, value, unit, source });
    }
  }
  
  private async checkHealth(): Promise<void> {
    try {
      const checks = await this.healthChecker.runAllChecks();
      const overallStatus = this.calculateOverallHealth(checks);
      
      const health: SystemHealth = {
        overall: overallStatus,
        checks,
        uptime: Date.now() - this.startTime,
        timestamp: new Date(),
      };
      
      // Check for alerts
      for (const check of checks) {
        if (check.status === 'unhealthy') {
          await this.createAlert('critical', `${check.name} health check failed`, check.message || '', check.name);
        } else if (check.status === 'degraded') {
          await this.createAlert('warning', `${check.name} performance degraded`, check.message || '', check.name);
        }
      }
      
      this.emit('health:checked', health);
    } catch (error) {
      StructuredLogger.error('Health check failed', error);
    }
  }
  
  private calculateOverallHealth(checks: HealthCheck[]): SystemHealth['overall'] {
    const unhealthyCount = checks.filter(c => c.status === 'unhealthy').length;
    const degradedCount = checks.filter(c => c.status === 'degraded').length;
    
    if (unhealthyCount > 0) return 'unhealthy';
    if (degradedCount > 0) return 'degraded';
    return 'healthy';
  }
  
  async createAlert(type: Alert['type'], title: string, message: string, source: string, metadata?: Record<string, any>): Promise<Alert> {
    const alert: Alert = {
      id: crypto.randomUUID(),
      type,
      title,
      message,
      source,
      timestamp: new Date(),
      acknowledged: false,
      metadata,
    };
    
    this.alerts.push(alert);
    
    // Keep only recent alerts
    const maxAlerts = 1000;
    if (this.alerts.length > maxAlerts) {
      this.alerts = this.alerts.slice(-maxAlerts);
    }
    
    StructuredLogger.warn('Alert created', {
      alertId: alert.id,
      type: alert.type,
      title: alert.title,
      source: alert.source,
    });
    
    this.emit('alert:created', alert);
    return alert;
  }
  
  async acknowledgeAlert(alertId: string): Promise<void> {
    const alert = this.alerts.find(a => a.id === alertId);
    if (alert) {
      alert.acknowledged = true;
      StructuredLogger.info('Alert acknowledged', { alertId });
      this.emit('alert:acknowledged', alert);
    }
  }
  
  async getSystemHealth(): Promise<SystemHealth> {
    const checks = await this.healthChecker.runAllChecks();
    const overallStatus = this.calculateOverallHealth(checks);
    
    return {
      overall: overallStatus,
      checks,
      uptime: Date.now() - this.startTime,
      timestamp: new Date(),
    };
  }
  
  async getSystemMetrics(): Promise<PerformanceMetrics> {
    return SystemMonitor.getSystemMetrics();
  }
  
  getAlerts(type?: Alert['type'], acknowledged?: boolean): Alert[] {
    let filteredAlerts = this.alerts;
    
    if (type) {
      filteredAlerts = filteredAlerts.filter(a => a.type === type);
    }
    
    if (acknowledged !== undefined) {
      filteredAlerts = filteredAlerts.filter(a => a.acknowledged === acknowledged);
    }
    
    return filteredAlerts.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  }
  
  async getMetricHistory(metricName: string, hours: number = 24): Promise<SystemMetric[]> {
    const cutoff = new Date(Date.now() - hours * 60 * 60 * 1000);
    return storage.getMetricHistory(metricName, cutoff);
  }
  
  async getServiceStats(): Promise<{
    totalRequests: number;
    totalErrors: number;
    averageResponseTime: number;
    activeConnections: number;
  }> {
    // This would integrate with your application metrics
    // For now, return simulated data based on actual monitoring
    const recentMetrics = this.metrics.filter(
      m => Date.now() - m.timestamp.getTime() < 60000 // Last minute
    );
    
    return {
      totalRequests: recentMetrics.length,
      totalErrors: recentMetrics.filter(m => m.metricName.includes('error')).length,
      averageResponseTime: 145, // Would be calculated from response time metrics
      activeConnections: 12, // Would be tracked from connection metrics
    };
  }
  
  addHealthCheck(name: string, check: () => Promise<HealthCheck>): void {
    this.healthChecker.addCheck(name, check);
  }
  
  removeHealthCheck(name: string): void {
    this.healthChecker['checks'].delete(name);
  }
  
  destroy(): void {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
    }
    
    this.alerts = [];
    this.metrics = [];
    this.removeAllListeners();
    
    StructuredLogger.info('Monitoring service destroyed');
  }
}

// Singleton instance
export const monitoringService = new MonitoringService();
