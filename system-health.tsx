import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { SystemMetrics } from "@shared/schema";

interface SystemHealthProps {
  metrics?: SystemMetrics;
}

export function SystemHealth({ metrics }: SystemHealthProps) {
  if (!metrics) {
    return (
      <Card className="mb-8">
        <CardHeader>
          <CardTitle>System Health & Performance</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[1, 2, 3].map((i) => (
              <div key={i} className="space-y-4 animate-pulse">
                <div className="h-4 bg-muted rounded w-3/4" />
                <div className="h-2 bg-muted rounded" />
                <div className="flex justify-between">
                  <div className="h-3 bg-muted rounded w-16" />
                  <div className="h-3 bg-muted rounded w-12" />
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  // Calculate performance percentages
  const dbPerformance = Math.min(100, Math.max(0, 100 - (parseInt(metrics.dbLatency) / 50) * 100));
  const memoryPercentage = parseInt(metrics.memoryUsage) / parseInt(metrics.memoryTotal) * 100;
  const wsPerformance = Math.min(100, Math.max(0, 100 - (parseInt(metrics.wsLatency) / 100) * 100));

  return (
    <Card className="mb-8">
      <CardHeader>
        <CardTitle>System Health & Performance</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Database Performance */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-foreground">Database Queries</span>
              <span className="text-sm text-success">{metrics.dbLatency} avg</span>
            </div>
            <Progress value={dbPerformance} className="h-2" />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Excellent</span>
              <span>{metrics.dbQueries} q/s</span>
            </div>
          </div>

          {/* Memory Usage */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-foreground">Memory Usage</span>
              <span className="text-sm text-warning">{metrics.memoryUsage} / {metrics.memoryTotal}</span>
            </div>
            <Progress value={memoryPercentage} className="h-2" />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Within limits</span>
              <span>{parseInt(metrics.memoryTotal) - parseInt(metrics.memoryUsage)}MB free</span>
            </div>
          </div>

          {/* WebSocket Connections */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-foreground">WebSocket Latency</span>
              <span className="text-sm text-success">{metrics.wsLatency}</span>
            </div>
            <Progress value={wsPerformance} className="h-2" />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Optimal</span>
              <span>{metrics.wsConnections} active</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
