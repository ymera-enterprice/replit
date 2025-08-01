import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useMetrics } from "@/hooks/useMetrics";
import { Skeleton } from "@/components/ui/skeleton";
import { 
  Activity, 
  Users, 
  BarChart3, 
  Bot, 
  FolderRoot, 
  Rocket, 
  Brain,
  Shield,
  FolderOpen,
  MessageSquare
} from "lucide-react";

export default function OverviewDashboard() {
  const { data: metrics, isLoading, error } = useMetrics();

  if (isLoading) {
    return (
      <div className="space-y-8">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-32 glass-card" />
          ))}
        </div>
        <Skeleton className="h-64 glass-card" />
      </div>
    );
  }

  if (error) {
    return (
      <Card className="glass-card border-destructive/50">
        <CardContent className="p-6 text-center">
          <div className="text-destructive text-xl mb-2">Error Loading Metrics</div>
          <p className="text-muted-foreground">Failed to load dashboard data. Please try again.</p>
        </CardContent>
      </Card>
    );
  }

  const stats = metrics?.data || {
    totalUsers: 0,
    activeUsers: 0,
    totalProjects: 0,
    activeAgents: 0,
    totalFiles: 0,
    totalMessages: 0,
    activeConnections: 0,
  };

  return (
    <div className="space-y-8">
      {/* System Status Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="glass-card interactive-card">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">System Status</h3>
              <div className="w-3 h-3 bg-green-400 rounded-full status-dot"></div>
            </div>
            <div className="text-3xl font-bold gradient-text">ONLINE</div>
            <p className="text-white/70 text-sm">All systems operational</p>
          </CardContent>
        </Card>
        
        <Card className="glass-card interactive-card">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">Active Users</h3>
              <Users className="w-6 h-6 text-secondary" />
            </div>
            <div className="text-3xl font-bold gradient-text">{stats.activeUsers.toLocaleString()}</div>
            <p className="text-white/70 text-sm">
              {stats.totalUsers > 0 
                ? `${Math.round((stats.activeUsers / stats.totalUsers) * 100)}% of total`
                : 'No users yet'
              }
            </p>
          </CardContent>
        </Card>
        
        <Card className="glass-card interactive-card">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">API Requests</h3>
              <BarChart3 className="w-6 h-6 text-primary" />
            </div>
            <div className="text-3xl font-bold gradient-text">
              {stats.activeConnections.toLocaleString()}
            </div>
            <p className="text-white/70 text-sm">Active connections</p>
          </CardContent>
        </Card>
        
        <Card className="glass-card interactive-card">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">AI Agents</h3>
              <Bot className="w-6 h-6 text-accent" />
            </div>
            <div className="text-3xl font-bold gradient-text">{stats.activeAgents}</div>
            <p className="text-white/70 text-sm">Running tasks</p>
          </CardContent>
        </Card>
      </div>

      {/* Real-time Activity Chart */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="w-5 h-5" />
            Real-time Activity
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-end justify-between h-32 gap-2">
            {Array.from({ length: 8 }).map((_, i) => {
              const height = Math.random() * 80 + 20;
              const colors = ['bg-primary/70', 'bg-secondary/70', 'bg-accent/70'];
              const color = colors[i % colors.length];
              
              return (
                <div 
                  key={i}
                  className={`chart-bar ${color} w-8 rounded-t cursor-pointer hover:opacity-80`}
                  style={{ height: `${height}%` }}
                  title={`Activity ${i + 1}: ${Math.round(height)}%`}
                />
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Phase Status Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="glass-card">
          <CardContent className="p-6">
            <h4 className="text-lg font-semibold mb-4 flex items-center">
              <FolderRoot className="w-5 h-5 mr-3 text-primary" />
              Phase 1: Core FolderRoot
            </h4>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="flex items-center gap-2">
                  <Shield className="w-4 h-4" />
                  Authentication
                </span>
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              </div>
              <div className="flex justify-between items-center">
                <span className="flex items-center gap-2">
                  <FolderOpen className="w-4 h-4" />
                  Project Management
                </span>
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              </div>
              <div className="flex justify-between items-center">
                <span className="flex items-center gap-2">
                  <BarChart3 className="w-4 h-4" />
                  File Operations
                </span>
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card className="glass-card">
          <CardContent className="p-6">
            <h4 className="text-lg font-semibold mb-4 flex items-center">
              <Rocket className="w-5 h-5 mr-3 text-secondary" />
              Phase 2: Enhanced Features
            </h4>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="flex items-center gap-2">
                  <MessageSquare className="w-4 h-4" />
                  Real-time Communication
                </span>
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              </div>
              <div className="flex justify-between items-center">
                <span className="flex items-center gap-2">
                  <Activity className="w-4 h-4" />
                  Monitoring & Metrics
                </span>
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              </div>
              <div className="flex justify-between items-center">
                <span className="flex items-center gap-2">
                  <Users className="w-4 h-4" />
                  Collaboration Tools
                </span>
                <div className="w-2 h-2 bg-yellow-400 rounded-full"></div>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card className="glass-card">
          <CardContent className="p-6">
            <h4 className="text-lg font-semibold mb-4 flex items-center">
              <Brain className="w-5 h-5 mr-3 text-accent" />
              Phase 3: AI Integration
            </h4>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="flex items-center gap-2">
                  <Bot className="w-4 h-4" />
                  AI Agents System
                </span>
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              </div>
              <div className="flex justify-between items-center">
                <span className="flex items-center gap-2">
                  <Brain className="w-4 h-4" />
                  Learning Engine
                </span>
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              </div>
              <div className="flex justify-between items-center">
                <span className="flex items-center gap-2">
                  <Activity className="w-4 h-4" />
                  Knowledge Graph
                </span>
                <div className="w-2 h-2 bg-yellow-400 rounded-full"></div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Performance Summary */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle>Performance Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-primary mb-1">
                {stats.totalProjects}
              </div>
              <div className="text-sm text-muted-foreground">Total Projects</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-secondary mb-1">
                {(stats.totalFileSize / (1024 * 1024 * 1024)).toFixed(1)}GB
              </div>
              <div className="text-sm text-muted-foreground">Storage Used</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-accent mb-1">
                {stats.totalMessages.toLocaleString()}
              </div>
              <div className="text-sm text-muted-foreground">Messages Sent</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-green-400 mb-1">
                99.8%
              </div>
              <div className="text-sm text-muted-foreground">System Uptime</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
