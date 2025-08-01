import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { Progress } from '@/components/ui/progress';
import ActivityFeed from '@/components/monitoring/activity-feed';
import ErrorTracker from '@/components/monitoring/error-tracker';
import { 
  Activity, 
  Cpu, 
  HardDrive, 
  Wifi, 
  Server, 
  AlertTriangle,
  CheckCircle,
  Clock,
  Zap
} from 'lucide-react';

export default function Monitoring() {
  const { data: systemMetrics, isLoading: metricsLoading } = useQuery({
    queryKey: ['/api/monitoring/system-metrics/latest'],
    refetchInterval: 5000,
  });

  const { data: systemHealth, isLoading: healthLoading } = useQuery({
    queryKey: ['/api/monitoring/health'],
    refetchInterval: 10000,
  });

  const { data: activityLogs, isLoading: logsLoading } = useQuery({
    queryKey: ['/api/monitoring/activity-logs', { limit: 50 }],
    refetchInterval: 5000,
  });

  const { data: errorLogs, isLoading: errorsLoading } = useQuery({
    queryKey: ['/api/monitoring/error-logs', { limit: 20, resolved: false }],
    refetchInterval: 10000,
  });

  const { data: systemReport } = useQuery({
    queryKey: ['/api/monitoring/report'],
    refetchInterval: 60000,
  });

  if (metricsLoading || healthLoading) {
    return (
      <div className="min-h-screen p-6">
        <div className="max-w-7xl mx-auto">
          <div className="mb-8">
            <Skeleton className="h-8 w-64 mb-4" />
            <Skeleton className="h-6 w-96" />
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-32" />
            ))}
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <Skeleton className="h-96" />
            <Skeleton className="h-96" />
          </div>
        </div>
      </div>
    );
  }

  const metrics = systemMetrics?.data;
  const health = systemHealth?.data;
  const activities = activityLogs?.data || [];
  const errors = errorLogs?.data || [];
  const report = systemReport?.data;

  const getHealthColor = (status: string) => {
    switch (status) {
      case 'excellent': return 'text-green-400';
      case 'good': return 'text-blue-400';
      case 'fair': return 'text-yellow-400';
      case 'poor': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'normal': return 'text-green-400';
      case 'high': return 'text-yellow-400';
      case 'critical': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const formatStorage = (bytes: number) => {
    const tb = bytes / (1024 ** 4);
    return `${tb.toFixed(1)}TB`;
  };

  return (
    <div className="min-h-screen p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">System Monitoring</h1>
          <p className="text-muted-foreground">
            Real-time system health, performance metrics, and activity monitoring
          </p>
        </div>

        {/* System Health Overview */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card className="glass-card">
            <CardContent className="p-6 text-center">
              <div className="flex items-center justify-center mb-3">
                <Cpu className="w-8 h-8 text-blue-400" />
              </div>
              <div className="text-3xl font-bold text-green-400 mb-2">
                {metrics?.cpu_usage || 0}%
              </div>
              <div className="text-sm text-muted-foreground mb-3">CPU Usage</div>
              <Progress 
                value={metrics?.cpu_usage || 0} 
                className="w-full h-2"
              />
              <div className={`text-xs mt-2 ${getStatusColor(health?.cpu_status || 'normal')}`}>
                {health?.cpu_status?.toUpperCase() || 'NORMAL'}
              </div>
            </CardContent>
          </Card>
          
          <Card className="glass-card">
            <CardContent className="p-6 text-center">
              <div className="flex items-center justify-center mb-3">
                <Server className="w-8 h-8 text-purple-400" />
              </div>
              <div className="text-3xl font-bold text-blue-400 mb-2">
                {metrics?.memory_usage || 0}%
              </div>
              <div className="text-sm text-muted-foreground mb-3">Memory Usage</div>
              <Progress 
                value={metrics?.memory_usage || 0} 
                className="w-full h-2"
              />
              <div className={`text-xs mt-2 ${getStatusColor(health?.memory_status || 'normal')}`}>
                {health?.memory_status?.toUpperCase() || 'NORMAL'}
              </div>
            </CardContent>
          </Card>
          
          <Card className="glass-card">
            <CardContent className="p-6 text-center">
              <div className="flex items-center justify-center mb-3">
                <Wifi className="w-8 h-8 text-green-400" />
              </div>
              <div className="text-3xl font-bold text-purple-400 mb-2">
                {metrics?.network_latency || 0}ms
              </div>
              <div className="text-sm text-muted-foreground mb-3">Network Latency</div>
              <Progress 
                value={Math.min(100, (metrics?.network_latency || 0) * 2)} 
                className="w-full h-2"
              />
              <div className="text-xs mt-2 text-green-400">OPTIMAL</div>
            </CardContent>
          </Card>
          
          <Card className="glass-card">
            <CardContent className="p-6 text-center">
              <div className="flex items-center justify-center mb-3">
                <HardDrive className="w-8 h-8 text-yellow-400" />
              </div>
              <div className="text-3xl font-bold text-yellow-400 mb-2">
                {formatStorage(metrics?.storage_used || 0)}
              </div>
              <div className="text-sm text-muted-foreground mb-3">Storage Used</div>
              <Progress 
                value={metrics ? (metrics.storage_used / metrics.storage_total) * 100 : 0} 
                className="w-full h-2"
              />
              <div className="text-xs mt-2 text-green-400">HEALTHY</div>
            </CardContent>
          </Card>
        </div>

        {/* System Status Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <Card className="glass-card">
            <CardContent className="p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <div className={`w-3 h-3 rounded-full ${
                    health?.overall_health === 'excellent' ? 'bg-green-400' :
                    health?.overall_health === 'good' ? 'bg-blue-400' :
                    health?.overall_health === 'fair' ? 'bg-yellow-400' : 'bg-red-400'
                  } pulse-dot`} />
                  <span className="font-semibold">System Health</span>
                </div>
                <Badge variant={health?.overall_health === 'excellent' ? 'default' : 'destructive'}>
                  {health?.overall_health?.toUpperCase() || 'UNKNOWN'}
                </Badge>
              </div>
              <div className="text-2xl font-bold mb-2">
                {report?.uptime_percentage || 99.9}%
              </div>
              <div className="text-sm text-muted-foreground">Uptime</div>
            </CardContent>
          </Card>

          <Card className="glass-card">
            <CardContent className="p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <Activity className="w-5 h-5 text-secondary" />
                  <span className="font-semibold">Active Agents</span>
                </div>
                <Badge variant="outline">
                  {health?.active_agents || 0} / 8
                </Badge>
              </div>
              <div className="text-2xl font-bold mb-2">
                {metrics?.tasks_completed?.toLocaleString() || '0'}
              </div>
              <div className="text-sm text-muted-foreground">Tasks Completed</div>
            </CardContent>
          </Card>

          <Card className="glass-card">
            <CardContent className="p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <AlertTriangle className="w-5 h-5 text-accent" />
                  <span className="font-semibold">Error Rate</span>
                </div>
                <Badge variant={health?.error_rate === 'low' ? 'default' : 'destructive'}>
                  {health?.error_rate?.toUpperCase() || 'LOW'}
                </Badge>
              </div>
              <div className="flex space-x-4">
                <div>
                  <div className="text-xl font-bold text-red-400">
                    {metrics?.error_count || 0}
                  </div>
                  <div className="text-xs text-muted-foreground">Errors</div>
                </div>
                <div>
                  <div className="text-xl font-bold text-yellow-400">
                    {metrics?.warning_count || 0}
                  </div>
                  <div className="text-xs text-muted-foreground">Warnings</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Live Activity Feed */}
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center">
                <Zap className="w-5 h-5 mr-2 text-green-400" />
                Live Activity Feed
              </CardTitle>
            </CardHeader>
            <CardContent>
              {logsLoading ? (
                <div className="space-y-4">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <Skeleton key={i} className="h-16" />
                  ))}
                </div>
              ) : (
                <ActivityFeed activities={activities} limit={10} showTimestamp />
              )}
            </CardContent>
          </Card>

          {/* Error Tracking */}
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center">
                <AlertTriangle className="w-5 h-5 mr-2 text-red-400" />
                Error Tracking
              </CardTitle>
            </CardHeader>
            <CardContent>
              {errorsLoading ? (
                <div className="space-y-4">
                  {Array.from({ length: 3 }).map((_, i) => (
                    <Skeleton key={i} className="h-20" />
                  ))}
                </div>
              ) : (
                <ErrorTracker errors={errors} />
              )}
            </CardContent>
          </Card>
        </div>

        {/* Performance Metrics Summary */}
        {report && (
          <Card className="glass-card mt-8">
            <CardHeader>
              <CardTitle className="flex items-center">
                <Activity className="w-5 h-5 mr-2 text-secondary" />
                Performance Summary (24h)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-400 mb-1">
                    {report.performance_metrics?.avg_cpu_usage?.toFixed(1) || '0.0'}%
                  </div>
                  <div className="text-sm text-muted-foreground">Avg CPU Usage</div>
                </div>
                
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-400 mb-1">
                    {report.performance_metrics?.avg_memory_usage?.toFixed(1) || '0.0'}%
                  </div>
                  <div className="text-sm text-muted-foreground">Avg Memory Usage</div>
                </div>
                
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-400 mb-1">
                    {report.performance_metrics?.avg_network_latency?.toFixed(1) || '0.0'}ms
                  </div>
                  <div className="text-sm text-muted-foreground">Avg Latency</div>
                </div>
                
                <div className="text-center">
                  <div className="text-2xl font-bold text-primary mb-1">
                    {report.total_tasks_completed?.toLocaleString() || '0'}
                  </div>
                  <div className="text-sm text-muted-foreground">Tasks Completed</div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
