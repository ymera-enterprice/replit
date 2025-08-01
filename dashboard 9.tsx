import { useQuery } from "@tanstack/react-query";
import SystemStats from "@/components/dashboard/system-stats";
import AgentVisualization from "@/components/dashboard/agent-visualization";
import ActivityFeed from "@/components/dashboard/activity-feed";
import FilesPreview from "@/components/dashboard/files-preview";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function Dashboard() {
  const { data: stats } = useQuery({
    queryKey: ["/api/stats"],
  });

  const { data: activity } = useQuery({
    queryKey: ["/api/activity"],
  });

  return (
    <div className="h-full p-6 overflow-y-auto scroll-container">
      {/* System Status Cards */}
      <SystemStats stats={stats} />
      
      {/* Main Dashboard Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-8">
        
        {/* Multi-Agent Learning System */}
        <div className="lg:col-span-2">
          <Card className="glass-card h-96">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg font-semibold text-foreground">
                  Multi-Agent Learning System
                </CardTitle>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-success rounded-full pulse-dot"></div>
                  <span className="text-sm text-success">Active Learning</span>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <AgentVisualization agents={stats?.agents} />
            </CardContent>
          </Card>
        </div>
        
        {/* Real-time Activity Feed */}
        <div>
          <Card className="glass-card h-96">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg font-semibold text-foreground">
                  Real-time Activity
                </CardTitle>
                <div className="w-2 h-2 bg-success rounded-full pulse-dot"></div>
              </div>
            </CardHeader>
            <CardContent className="p-0">
              <ActivityFeed activities={activity} />
            </CardContent>
          </Card>
        </div>
      </div>
      
      {/* File Management Preview */}
      <div className="mt-8">
        <FilesPreview />
      </div>
    </div>
  );
}
