import { Activity, Database, Wifi, Shield } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';

interface SystemMetricsProps {
  metrics?: {
    cpu?: number;
    memory?: number;
    network?: number;
    dbConnections?: number;
    maxConnections?: number;
    queryTime?: number;
    requestsPerMin?: number;
    responseTime?: number;
    errorRate?: number;
    activeSessions?: number;
  };
}

export default function SystemMetrics({ metrics = {} }: SystemMetricsProps) {
  const {
    cpu = 34,
    memory = 67,
    network = 23,
    dbConnections = 47,
    maxConnections = 100,
    queryTime = 23,
    requestsPerMin = 1247,
    responseTime = 142,
    errorRate = 0.02,
    activeSessions = 34
  } = metrics;

  const getStatusColor = (value: number, thresholds: { warning: number; critical: number }) => {
    if (value >= thresholds.critical) return 'text-error';
    if (value >= thresholds.warning) return 'text-warning';
    return 'text-success';
  };

  const getProgressColor = (value: number, thresholds: { warning: number; critical: number }) => {
    if (value >= thresholds.critical) return 'bg-error';
    if (value >= thresholds.warning) return 'bg-warning';
    return 'bg-success';
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
      {/* Performance Metrics */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Activity className="h-5 w-5 text-primary" />
            <span>Performance</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-dark-300">CPU Usage</span>
              <span className={`text-sm font-medium ${getStatusColor(cpu, { warning: 70, critical: 90 })}`}>
                {cpu}%
              </span>
            </div>
            <Progress 
              value={cpu} 
              className="h-2"
              indicatorClassName={getProgressColor(cpu, { warning: 70, critical: 90 })}
            />
          </div>
          
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-dark-300">Memory</span>
              <span className={`text-sm font-medium ${getStatusColor(memory, { warning: 80, critical: 95 })}`}>
                {memory}%
              </span>
            </div>
            <Progress 
              value={memory} 
              className="h-2"
              indicatorClassName={getProgressColor(memory, { warning: 80, critical: 95 })}
            />
          </div>
          
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-dark-300">Network</span>
              <span className={`text-sm font-medium ${getStatusColor(network, { warning: 75, critical: 90 })}`}>
                {network}%
              </span>
            </div>
            <Progress 
              value={network} 
              className="h-2"
              indicatorClassName={getProgressColor(network, { warning: 75, critical: 90 })}
            />
          </div>

          {/* Additional Metrics */}
          <div className="pt-2 border-t border-dark-600">
            <div className="grid grid-cols-2 gap-2 text-center">
              <div>
                <div className="text-xs text-dark-400">Load Avg</div>
                <div className="text-sm font-medium text-dark-200">1.24</div>
              </div>
              <div>
                <div className="text-xs text-dark-400">Uptime</div>
                <div className="text-sm font-medium text-dark-200">99.8%</div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* Database Status */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Database className="h-5 w-5 text-primary" />
            <span>Database</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-dark-300">PostgreSQL</span>
            <Badge className="bg-success/20 text-success border-success/30">
              Online
            </Badge>
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-sm text-dark-300">Redis Cache</span>
            <Badge className="bg-success/20 text-success border-success/30">
              Online
            </Badge>
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-sm text-dark-300">Connections</span>
            <span className="text-sm font-medium text-dark-200">
              {dbConnections}/{maxConnections}
            </span>
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-sm text-dark-300">Query Time</span>
            <span className={`text-sm font-medium ${getStatusColor(queryTime, { warning: 100, critical: 500 })}`}>
              {queryTime}ms avg
            </span>
          </div>

          {/* Connection Usage Progress */}
          <div>
            <div className="text-xs text-dark-400 mb-1">Connection Usage</div>
            <Progress 
              value={(dbConnections / maxConnections) * 100} 
              className="h-1"
              indicatorClassName={getProgressColor((dbConnections / maxConnections) * 100, { warning: 70, critical: 90 })}
            />
          </div>

          {/* Database Performance */}
          <div className="pt-2 border-t border-dark-600">
            <div className="grid grid-cols-2 gap-2 text-center">
              <div>
                <div className="text-xs text-dark-400">TPS</div>
                <div className="text-sm font-medium text-dark-200">2.4K</div>
              </div>
              <div>
                <div className="text-xs text-dark-400">Cache Hit</div>
                <div className="text-sm font-medium text-success">98.7%</div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* API Status */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Wifi className="h-5 w-5 text-primary" />
            <span>API Gateway</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-dark-300">Requests/min</span>
            <span className="text-sm font-medium text-primary">
              {requestsPerMin.toLocaleString()}
            </span>
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-sm text-dark-300">Response Time</span>
            <span className={`text-sm font-medium ${getStatusColor(responseTime, { warning: 500, critical: 1000 })}`}>
              {responseTime}ms
            </span>
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-sm text-dark-300">Error Rate</span>
            <span className={`text-sm font-medium ${getStatusColor(errorRate, { warning: 1, critical: 5 })}`}>
              {errorRate}%
            </span>
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-sm text-dark-300">Rate Limits</span>
            <Badge className="bg-warning/20 text-warning border-warning/30">
              3 active
            </Badge>
          </div>

          {/* API Health */}
          <div className="pt-2 border-t border-dark-600">
            <div className="grid grid-cols-2 gap-2 text-center">
              <div>
                <div className="text-xs text-dark-400">Success Rate</div>
                <div className="text-sm font-medium text-success">99.98%</div>
              </div>
              <div>
                <div className="text-xs text-dark-400">Avg Load</div>
                <div className="text-sm font-medium text-dark-200">54ms</div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* Security Status */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Shield className="h-5 w-5 text-primary" />
            <span>Security</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-dark-300">Authentication</span>
            <Badge className="bg-success/20 text-success border-success/30">
              Active
            </Badge>
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-sm text-dark-300">Failed Logins</span>
            <span className="text-sm font-medium text-success">
              2 today
            </span>
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-sm text-dark-300">Active Sessions</span>
            <span className="text-sm font-medium text-primary">
              {activeSessions}
            </span>
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-sm text-dark-300">SSL/TLS</span>
            <Badge className="bg-success/20 text-success border-success/30">
              Valid
            </Badge>
          </div>

          {/* Security Metrics */}
          <div className="pt-2 border-t border-dark-600">
            <div className="grid grid-cols-2 gap-2 text-center">
              <div>
                <div className="text-xs text-dark-400">Threats</div>
                <div className="text-sm font-medium text-success">0</div>
              </div>
              <div>
                <div className="text-xs text-dark-400">Blocked IPs</div>
                <div className="text-sm font-medium text-warning">7</div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
