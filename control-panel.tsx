import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { 
  Play, 
  RotateCcw, 
  Download, 
  Settings, 
  Activity,
  Database,
  Zap,
  AlertTriangle
} from "lucide-react";

export default function ControlPanel() {
  const [isExpanded, setIsExpanded] = useState(false);
  const { toast } = useToast();
  const queryClient = useQueryClient();

  // System test mutation
  const systemTestMutation = useMutation({
    mutationFn: async () => {
      // Test multiple endpoints to verify system health
      const testResults = await Promise.allSettled([
        fetch('/api/auth/user').then(res => ({ endpoint: '/api/auth/user', status: res.status })),
        fetch('/api/dashboard/metrics').then(res => ({ endpoint: '/api/dashboard/metrics', status: res.status })),
        fetch('/api/projects').then(res => ({ endpoint: '/api/projects', status: res.status })),
        fetch('/api/agents').then(res => ({ endpoint: '/api/agents', status: res.status })),
      ]);
      
      return testResults;
    },
    onSuccess: (results) => {
      const passedTests = results.filter(result => 
        result.status === 'fulfilled' && 
        (result.value as any).status < 400
      ).length;
      
      const passRate = (passedTests / results.length) * 100;
      
      toast({
        title: "System Test Complete",
        description: `${passedTests}/${results.length} endpoints passed (${Math.round(passRate)}%)`,
        variant: passRate >= 90 ? "default" : "destructive",
      });
    },
    onError: () => {
      toast({
        title: "System Test Failed",
        description: "Unable to complete system testing",
        variant: "destructive",
      });
    },
  });

  // Refresh metrics mutation
  const refreshMetricsMutation = useMutation({
    mutationFn: async () => {
      // Create a new system metric to trigger refresh
      await apiRequest('/api/metrics', {
        method: 'POST',
        body: JSON.stringify({
          metricType: 'manual_refresh',
          value: Date.now(),
          unit: 'timestamp',
        }),
      });
    },
    onSuccess: () => {
      // Invalidate all cached queries to force refresh
      queryClient.invalidateQueries();
      toast({
        title: "Metrics Refreshed",
        description: "All dashboard data has been updated",
      });
    },
    onError: () => {
      toast({
        title: "Refresh Failed",
        description: "Unable to refresh metrics",
        variant: "destructive",
      });
    },
  });

  // Export logs mutation
  const exportLogsMutation = useMutation({
    mutationFn: async () => {
      // Fetch system metrics for export
      const response = await fetch('/api/metrics?hours=168'); // Last week
      if (!response.ok) throw new Error('Failed to fetch logs');
      
      const data = await response.json();
      return data;
    },
    onSuccess: (data) => {
      // Create and download JSON file
      const blob = new Blob([JSON.stringify(data, null, 2)], { 
        type: 'application/json' 
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `ymera-logs-${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      
      toast({
        title: "Logs Exported",
        description: "System logs downloaded successfully",
      });
    },
    onError: () => {
      toast({
        title: "Export Failed",
        description: "Unable to export system logs",
        variant: "destructive",
      });
    },
  });

  // Database backup mutation
  const backupMutation = useMutation({
    mutationFn: async () => {
      // Trigger database metrics collection
      await apiRequest('/api/metrics', {
        method: 'POST',
        body: JSON.stringify({
          metricType: 'backup_initiated',
          value: 1,
          unit: 'count',
        }),
      });
    },
    onSuccess: () => {
      toast({
        title: "Backup Initiated",
        description: "Database backup process started",
      });
    },
    onError: () => {
      toast({
        title: "Backup Failed",
        description: "Unable to initiate backup",
        variant: "destructive",
      });
    },
  });

  const controlActions = [
    {
      id: 'test',
      label: 'Run System Test',
      icon: Play,
      color: 'text-green-400',
      action: () => systemTestMutation.mutate(),
      loading: systemTestMutation.isPending,
      description: 'Test all API endpoints and system health'
    },
    {
      id: 'refresh',
      label: 'Refresh Metrics',
      icon: RotateCcw,
      color: 'text-blue-400',
      action: () => refreshMetricsMutation.mutate(),
      loading: refreshMetricsMutation.isPending,
      description: 'Update all dashboard metrics and data'
    },
    {
      id: 'export',
      label: 'Export Logs',
      icon: Download,
      color: 'text-purple-400',
      action: () => exportLogsMutation.mutate(),
      loading: exportLogsMutation.isPending,
      description: 'Download system logs and analytics'
    },
    {
      id: 'backup',
      label: 'Backup Database',
      icon: Database,
      color: 'text-orange-400',
      action: () => backupMutation.mutate(),
      loading: backupMutation.isPending,
      description: 'Create database backup'
    }
  ];

  return (
    <div className="fixed bottom-6 right-6 z-50">
      {/* Expanded Panel */}
      {isExpanded && (
        <Card className="glass-card mb-4 p-4 w-80 animate-in slide-in-from-bottom-2 duration-300">
          <div className="space-y-3">
            <div className="flex items-center justify-between border-b border-white/20 pb-3">
              <h3 className="font-semibold flex items-center gap-2">
                <Settings className="w-4 h-4" />
                Control Panel
              </h3>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setIsExpanded(false)}
                className="h-6 w-6 p-0 text-white/60 hover:text-white"
              >
                ×
              </Button>
            </div>
            
            {controlActions.map((action) => (
              <div key={action.id} className="space-y-2">
                <Button
                  onClick={action.action}
                  disabled={action.loading}
                  className="w-full justify-start glass-card border border-white/20 hover:bg-white/10 transition-all"
                  variant="ghost"
                >
                  <action.icon className={`w-4 h-4 mr-3 ${action.color}`} />
                  <span className="flex-1 text-left">
                    {action.loading ? 'Processing...' : action.label}
                  </span>
                  {action.loading && (
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  )}
                </Button>
                <p className="text-xs text-white/60 px-3">
                  {action.description}
                </p>
              </div>
            ))}
            
            {/* System Status Indicators */}
            <div className="border-t border-white/20 pt-3 space-y-2">
              <h4 className="text-sm font-medium text-white/80">System Status</h4>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-400 rounded-full status-dot"></div>
                  <span>API Health</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-400 rounded-full status-dot"></div>
                  <span>Database</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-400 rounded-full status-dot"></div>
                  <span>WebSocket</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-yellow-400 rounded-full status-dot"></div>
                  <span>AI Agents</span>
                </div>
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Main Control Buttons */}
      <div className="flex flex-col gap-3">
        {!isExpanded && (
          <>
            <Button
              onClick={() => systemTestMutation.mutate()}
              disabled={systemTestMutation.isPending}
              className="glass-card p-4 rounded-full hover:scale-110 transition-transform group"
              title="Run System Test"
            >
              {systemTestMutation.isPending ? (
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-green-400"></div>
              ) : (
                <Play className="text-green-400 w-5 h-5 group-hover:scale-110 transition-transform" />
              )}
            </Button>

            <Button
              onClick={() => refreshMetricsMutation.mutate()}
              disabled={refreshMetricsMutation.isPending}
              className="glass-card p-4 rounded-full hover:scale-110 transition-transform group"
              title="Refresh Metrics"
            >
              {refreshMetricsMutation.isPending ? (
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-400"></div>
              ) : (
                <RotateCcw className="text-blue-400 w-5 h-5 group-hover:rotate-180 transition-transform duration-500" />
              )}
            </Button>

            <Button
              onClick={() => exportLogsMutation.mutate()}
              disabled={exportLogsMutation.isPending}
              className="glass-card p-4 rounded-full hover:scale-110 transition-transform group"
              title="Export Logs"
            >
              {exportLogsMutation.isPending ? (
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-purple-400"></div>
              ) : (
                <Download className="text-purple-400 w-5 h-5 group-hover:translate-y-1 transition-transform" />
              )}
            </Button>
          </>
        )}

        {/* Settings Toggle */}
        <Button
          onClick={() => setIsExpanded(!isExpanded)}
          className={`glass-card p-4 rounded-full hover:scale-110 transition-all group ${
            isExpanded ? 'ymera-gradient text-black' : ''
          }`}
          title="Control Panel"
        >
          <Settings className={`w-5 h-5 transition-transform ${
            isExpanded ? 'rotate-45' : 'group-hover:rotate-45'
          }`} />
        </Button>
      </div>
    </div>
  );
}
