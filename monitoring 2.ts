import { storage } from '../storage';
import { 
  SystemMetrics, 
  InsertSystemMetrics,
  ActivityLog,
  InsertActivityLog,
  ErrorLog,
  InsertErrorLog 
} from '@shared/schema';

export class MonitoringService {
  async getSystemMetrics(limit?: number): Promise<SystemMetrics[]> {
    return await storage.getSystemMetrics(limit);
  }

  async getLatestSystemMetrics(): Promise<SystemMetrics | null> {
    return await storage.getLatestSystemMetrics();
  }

  async createSystemMetrics(metrics: InsertSystemMetrics): Promise<SystemMetrics> {
    return await storage.createSystemMetrics(metrics);
  }

  async getActivityLogs(limit?: number, agentId?: string): Promise<ActivityLog[]> {
    return await storage.getActivityLogs(limit, agentId);
  }

  async createActivityLog(log: InsertActivityLog): Promise<ActivityLog> {
    return await storage.createActivityLog(log);
  }

  async getErrorLogs(limit?: number, resolved?: boolean): Promise<ErrorLog[]> {
    return await storage.getErrorLogs(limit, resolved);
  }

  async createErrorLog(error: InsertErrorLog): Promise<ErrorLog> {
    return await storage.createErrorLog(error);
  }

  async resolveError(id: string, resolutionNotes?: string): Promise<ErrorLog | null> {
    return await storage.resolveError(id, resolutionNotes);
  }

  async logAgentActivity(agentId: string, agentType: string, message: string, level: 'info' | 'warning' | 'error' | 'success' = 'info'): Promise<ActivityLog> {
    return await this.createActivityLog({
      agent_id: agentId,
      agent_type: agentType as any,
      message,
      level
    });
  }

  async logSystemError(errorType: string, message: string, severity: 'low' | 'medium' | 'high' | 'critical', agentId?: string, stackTrace?: string): Promise<ErrorLog> {
    return await this.createErrorLog({
      agent_id: agentId,
      error_type: errorType,
      message,
      severity,
      resolved: false,
      stack_trace: stackTrace
    });
  }

  async getSystemHealth(): Promise<{
    overall_health: 'excellent' | 'good' | 'fair' | 'poor';
    cpu_status: 'normal' | 'high' | 'critical';
    memory_status: 'normal' | 'high' | 'critical';
    error_rate: 'low' | 'medium' | 'high';
    active_agents: number;
  }> {
    const latestMetrics = await this.getLatestSystemMetrics();
    const errorLogs = await this.getErrorLogs(100, false); // Get unresolved errors
    
    if (!latestMetrics) {
      return {
        overall_health: 'poor',
        cpu_status: 'normal',
        memory_status: 'normal',
        error_rate: 'low',
        active_agents: 0
      };
    }

    // Determine CPU status
    let cpu_status: 'normal' | 'high' | 'critical' = 'normal';
    if (latestMetrics.cpu_usage > 80) cpu_status = 'critical';
    else if (latestMetrics.cpu_usage > 60) cpu_status = 'high';

    // Determine memory status
    let memory_status: 'normal' | 'high' | 'critical' = 'normal';
    if (latestMetrics.memory_usage > 85) memory_status = 'critical';
    else if (latestMetrics.memory_usage > 70) memory_status = 'high';

    // Determine error rate
    const criticalErrors = errorLogs.filter(e => e.severity === 'critical').length;
    const highErrors = errorLogs.filter(e => e.severity === 'high').length;
    let error_rate: 'low' | 'medium' | 'high' = 'low';
    if (criticalErrors > 0 || highErrors > 5) error_rate = 'high';
    else if (highErrors > 2 || errorLogs.length > 10) error_rate = 'medium';

    // Determine overall health
    let overall_health: 'excellent' | 'good' | 'fair' | 'poor' = 'excellent';
    if (cpu_status === 'critical' || memory_status === 'critical' || error_rate === 'high') {
      overall_health = 'poor';
    } else if (cpu_status === 'high' || memory_status === 'high' || error_rate === 'medium') {
      overall_health = 'fair';
    } else if (cpu_status === 'high' || memory_status === 'high' || errorLogs.length > 5) {
      overall_health = 'good';
    }

    return {
      overall_health,
      cpu_status,
      memory_status,
      error_rate,
      active_agents: latestMetrics.active_agents
    };
  }

  async generateSystemReport(): Promise<{
    uptime_percentage: number;
    total_tasks_completed: number;
    error_summary: {
      critical: number;
      high: number;
      medium: number;
      low: number;
    };
    performance_metrics: {
      avg_cpu_usage: number;
      avg_memory_usage: number;
      avg_network_latency: number;
    };
  }> {
    const metrics = await this.getSystemMetrics(24); // Last 24 hours
    const errorLogs = await this.getErrorLogs();
    
    const performanceMetrics = metrics.length > 0 ? {
      avg_cpu_usage: Math.round(metrics.reduce((sum, m) => sum + m.cpu_usage, 0) / metrics.length * 10) / 10,
      avg_memory_usage: Math.round(metrics.reduce((sum, m) => sum + m.memory_usage, 0) / metrics.length * 10) / 10,
      avg_network_latency: Math.round(metrics.reduce((sum, m) => sum + m.network_latency, 0) / metrics.length * 10) / 10
    } : {
      avg_cpu_usage: 0,
      avg_memory_usage: 0,
      avg_network_latency: 0
    };

    const errorSummary = {
      critical: errorLogs.filter(e => e.severity === 'critical').length,
      high: errorLogs.filter(e => e.severity === 'high').length,
      medium: errorLogs.filter(e => e.severity === 'medium').length,
      low: errorLogs.filter(e => e.severity === 'low').length
    };

    // Calculate uptime (simplified - assume high availability)
    const uptime_percentage = 99.9;
    
    const latestMetrics = await this.getLatestSystemMetrics();
    const total_tasks_completed = latestMetrics?.tasks_completed || 0;

    return {
      uptime_percentage,
      total_tasks_completed,
      error_summary: errorSummary,
      performance_metrics: performanceMetrics
    };
  }
}

export const monitoringService = new MonitoringService();
