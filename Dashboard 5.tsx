import { StatsCard } from "./StatsCard";
import { ThreeVisualization } from "./ThreeVisualization";
import { useSystemStats } from "../hooks/useSystemStats";
import { useQuery } from "@tanstack/react-query";

interface DashboardStats {
  activeProjects: number;
  totalFiles: number;
  totalUsers: number;
  totalApiRequests: number;
  metrics: {
    cpuUsage: number;
    memoryUsage: number;
    diskUsage: number;
    networkLatency: number;
    activeConnections: number;
  };
}

interface ApiEndpoint {
  method: string;
  path: string;
  description: string;
  example: string;
}

interface ActivityItem {
  id: string;
  action: string;
  description: string;
  timestamp: string;
  userId?: string;
}

export function Dashboard() {
  const { stats, isLoading: statsLoading } = useSystemStats();

  const { data: apiDocs } = useQuery<{ endpoints: ApiEndpoint[] }>({
    queryKey: ["/api/docs"],
  });

  const { data: activity } = useQuery<ActivityItem[]>({
    queryKey: ["/api/activity"],
    refetchInterval: 10000, // Refresh every 10 seconds
  });

  if (statsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-yellow-500"></div>
      </div>
    );
  }

  const getActivityIcon = (action: string) => {
    switch (action) {
      case 'project_created':
        return { icon: 'plus', color: 'bg-blue-500/20 text-blue-500' };
      case 'file_uploaded':
        return { icon: 'upload', color: 'bg-yellow-500/20 text-yellow-500' };
      case 'user_login':
        return { icon: 'sign-in-alt', color: 'bg-green-500/20 text-green-500' };
      default:
        return { icon: 'check', color: 'bg-green-500/20 text-green-500' };
    }
  };

  const formatTimeAgo = (timestamp: string) => {
    const now = new Date();
    const time = new Date(timestamp);
    const diffInMinutes = Math.floor((now.getTime() - time.getTime()) / (1000 * 60));
    
    if (diffInMinutes < 1) return 'Just now';
    if (diffInMinutes < 60) return `${diffInMinutes} minutes ago`;
    
    const diffInHours = Math.floor(diffInMinutes / 60);
    if (diffInHours < 24) return `${diffInHours} hours ago`;
    
    const diffInDays = Math.floor(diffInHours / 24);
    return `${diffInDays} days ago`;
  };

  return (
    <div className="h-full overflow-y-auto p-6">
      {/* Stats Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatsCard
          title="Active Projects"
          value={stats?.activeProjects || 0}
          change="+12%"
          changeType="positive"
          description="Active Projects"
          icon={<i className="fas fa-project-diagram text-blue-500 text-xl"></i>}
          iconBgColor="bg-blue-500/20"
        />
        
        <StatsCard
          title="API Requests"
          value={stats?.totalApiRequests?.toLocaleString() || 0}
          change="+8%"
          changeType="positive"
          description="API Requests"
          icon={<i className="fas fa-exchange-alt text-yellow-500 text-xl"></i>}
          iconBgColor="bg-yellow-500/20"
        />
        
        <StatsCard
          title="System Uptime"
          value="99.9%"
          change="100%"
          changeType="positive"
          description="System Uptime"
          icon={<i className="fas fa-clock text-green-500 text-xl"></i>}
          iconBgColor="bg-green-500/20"
        />
        
        <StatsCard
          title="Storage Used"
          value={`${((stats?.totalFiles || 0) * 2.3).toFixed(1)} GB`}
          change="+2%"
          changeType="neutral"
          description="Storage Used"
          icon={<i className="fas fa-hdd text-red-500 text-xl"></i>}
          iconBgColor="bg-red-500/20"
        />
      </div>
      
      {/* Main Dashboard Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        
        {/* 3D Visualization Panel */}
        <div className="xl:col-span-2 glass-card rounded-xl p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold">System Architecture Visualization</h3>
            <div className="flex space-x-2">
              <button className="px-3 py-1 bg-blue-500/20 text-blue-500 rounded-lg text-sm hover:bg-blue-500/30 transition-colors">
                3D View
              </button>
              <button className="px-3 py-1 bg-white/10 rounded-lg text-sm hover:bg-white/20 transition-colors">
                2D View
              </button>
            </div>
          </div>
          
          <ThreeVisualization />
          
          {/* Performance Metrics */}
          <div className="grid grid-cols-3 gap-4 mt-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-500">
                {stats?.metrics?.cpuUsage?.toFixed(0) || 0}%
              </div>
              <div className="text-xs text-white/70">CPU Usage</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-500">
                {stats?.metrics?.memoryUsage?.toFixed(0) || 0}%
              </div>
              <div className="text-xs text-white/70">Memory</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-500">
                {stats?.metrics?.networkLatency?.toFixed(0) || 0}ms
              </div>
              <div className="text-xs text-white/70">Avg Latency</div>
            </div>
          </div>
        </div>
        
        {/* API Documentation & Health */}
        <div className="space-y-6">
          
          {/* API Documentation Panel */}
          <div className="glass-card rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <i className="fas fa-book mr-2 text-blue-500"></i>
              API Documentation
            </h3>
            
            <div className="space-y-3">
              {apiDocs?.endpoints?.slice(0, 4).map((endpoint, index) => (
                <div 
                  key={index}
                  className="flex items-center justify-between p-3 bg-black/30 rounded-lg hover:bg-black/40 transition-colors cursor-pointer"
                >
                  <div>
                    <div className={`font-mono text-sm ${
                      endpoint.method === 'GET' ? 'text-green-400' : 
                      endpoint.method === 'POST' ? 'text-yellow-500' : 'text-red-500'
                    }`}>
                      {endpoint.method} {endpoint.path}
                    </div>
                    <div className="text-xs text-white/70">{endpoint.description}</div>
                  </div>
                  <i className="fas fa-external-link-alt text-white/50"></i>
                </div>
              ))}
            </div>
            
            <button className="w-full mt-4 px-4 py-2 bg-blue-500/20 border border-blue-500/30 text-blue-500 rounded-lg hover:bg-blue-500/30 transition-colors hover-glow">
              View Full Documentation
            </button>
          </div>
          
          {/* Health Monitor Panel */}
          <div className="glass-card rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <i className="fas fa-heartbeat mr-2 text-green-500"></i>
              System Health
            </h3>
            
            <div className="space-y-4">
              {/* Server Status */}
              <div className="flex items-center justify-between">
                <span className="text-sm">Express Server</span>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-green-400 text-sm">Running</span>
                </div>
              </div>
              
              {/* Database Status */}
              <div className="flex items-center justify-between">
                <span className="text-sm">PostgreSQL</span>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-green-400 text-sm">Connected</span>
                </div>
              </div>
              
              {/* WebSocket Status */}
              <div className="flex items-center justify-between">
                <span className="text-sm">WebSocket</span>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-green-400 text-sm">Online</span>
                </div>
              </div>
              
              {/* API Status */}
              <div className="flex items-center justify-between">
                <span className="text-sm">API Gateway</span>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-green-400 text-sm">Operational</span>
                </div>
              </div>
            </div>
            
            {/* Response Time Chart */}
            <div className="mt-4 p-3 bg-black/30 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-white/70">Response Time</span>
                <span className="text-xs text-green-400">
                  {stats?.metrics?.networkLatency?.toFixed(0) || 12}ms avg
                </span>
              </div>
              <div className="w-full bg-black/50 rounded-full h-2">
                <div 
                  className="bg-gradient-to-r from-green-500 to-green-400 h-2 rounded-full" 
                  style={{ width: '85%' }}
                ></div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Recent Activity */}
      <div className="mt-6 glass-card rounded-xl p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
          <i className="fas fa-history mr-2 text-white/70"></i>
          Recent Activity
        </h3>
        
        <div className="space-y-3">
          {activity?.length ? (
            activity.slice(0, 5).map((item) => {
              const { icon, color } = getActivityIcon(item.action);
              return (
                <div 
                  key={item.id}
                  className="flex items-center space-x-4 p-3 hover:bg-white/5 rounded-lg transition-colors"
                >
                  <div className={`w-8 h-8 ${color} rounded-full flex items-center justify-center`}>
                    <i className={`fas fa-${icon} text-xs`}></i>
                  </div>
                  <div className="flex-1">
                    <div className="text-sm font-medium">{item.description}</div>
                    <div className="text-xs text-white/50">{formatTimeAgo(item.timestamp)}</div>
                  </div>
                </div>
              );
            })
          ) : (
            <div className="text-center py-8 text-white/50">
              <i className="fas fa-history text-3xl mb-2"></i>
              <p>No recent activity</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
