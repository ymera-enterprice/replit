import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { 
  Activity, 
  Bot, 
  Brain, 
  Settings, 
  Users, 
  BarChart3,
  Clock,
  Shield,
  Zap,
  AlertCircle,
  CheckCircle2,
  XCircle
} from "lucide-react";
import { useAuth } from "@/lib/auth";
import { useToast } from "@/hooks/use-toast";

interface DashboardStats {
  totalAgents: number;
  activeAgents: number;
  totalLearningData: number;
  recentActivities: number;
}

interface Agent {
  id: string;
  agentId: string;
  name: string;
  agentType: string;
  status: string;
  enabled: boolean;
  lastHeartbeat?: string;
  capabilities: string[];
}

interface Activity {
  id: string;
  activityType: string;
  activityDetails: any;
  createdAt: string;
  success: boolean;
  resourceType?: string;
}

export default function Dashboard() {
  const { user, logout } = useAuth();
  const { toast } = useToast();

  // Fetch dashboard statistics
  const { data: stats, isLoading: statsLoading } = useQuery<DashboardStats>({
    queryKey: ['/api/system/stats'],
    queryFn: async () => {
      // Mock stats for now - in production this would be a real API call
      return {
        totalAgents: 0,
        activeAgents: 0,
        totalLearningData: 0,
        recentActivities: 0
      };
    }
  });

  // Fetch user's agents
  const { data: agentsData, isLoading: agentsLoading } = useQuery<{ agents: Agent[]; total: number }>({
    queryKey: ['/api/agents'],
    queryFn: async () => {
      const response = await fetch('/api/agents', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
        }
      });
      if (!response.ok) {
        throw new Error('Failed to fetch agents');
      }
      return response.json();
    }
  });

  // Fetch recent activities
  const { data: activitiesData, isLoading: activitiesLoading } = useQuery<{ activities: Activity[]; total: number }>({
    queryKey: ['/api/users/activities'],
    queryFn: async () => {
      const response = await fetch('/api/users/activities?limit=10', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
        }
      });
      if (!response.ok) {
        throw new Error('Failed to fetch activities');
      }
      return response.json();
    }
  });

  const handleLogout = async () => {
    try {
      await logout();
      toast({
        title: "Logged out successfully",
        description: "You have been logged out of your account.",
      });
    } catch (error) {
      toast({
        title: "Logout failed",
        description: "There was a problem logging you out.",
        variant: "destructive",
      });
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case 'active':
        return <CheckCircle2 className="h-4 w-4 text-green-500" />;
      case 'inactive':
        return <XCircle className="h-4 w-4 text-gray-500" />;
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      default:
        return <Clock className="h-4 w-4 text-yellow-500" />;
    }
  };

  const getStatusBadgeVariant = (status: string) => {
    switch (status.toLowerCase()) {
      case 'active':
        return 'default';
      case 'inactive':
        return 'secondary';
      case 'error':
        return 'destructive';
      default:
        return 'outline';
    }
  };

  const formatActivityType = (type: string) => {
    return type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  const formatTimeAgo = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInMinutes = Math.floor((now.getTime() - date.getTime()) / (1000 * 60));
    
    if (diffInMinutes < 1) return 'Just now';
    if (diffInMinutes < 60) return `${diffInMinutes}m ago`;
    if (diffInMinutes < 1440) return `${Math.floor(diffInMinutes / 60)}h ago`;
    return `${Math.floor(diffInMinutes / 1440)}d ago`;
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b">
        <div className="flex h-16 items-center px-4 md:px-6">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Bot className="h-6 w-6 text-primary" />
              <span className="font-bold text-xl">YMERA</span>
            </div>
          </div>
          
          <div className="ml-auto flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Avatar className="h-8 w-8">
                <AvatarImage src={user?.avatarUrl} alt={user?.displayName || user?.username} />
                <AvatarFallback>
                  {user?.displayName?.charAt(0) || user?.username?.charAt(0) || 'U'}
                </AvatarFallback>
              </Avatar>
              <div className="flex flex-col">
                <span className="text-sm font-medium">{user?.displayName || user?.username}</span>
                <span className="text-xs text-muted-foreground">{user?.email}</span>
              </div>
            </div>
            <Button variant="outline" size="sm" onClick={handleLogout}>
              Logout
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 space-y-4 p-4 md:p-6">
        {/* Welcome Section */}
        <div className="flex items-center justify-between space-y-2">
          <div>
            <h2 className="text-3xl font-bold tracking-tight">Welcome back, {user?.displayName || user?.username}!</h2>
            <p className="text-muted-foreground">
              Here's what's happening with your YMERA platform today.
            </p>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Agents</CardTitle>
              <Bot className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {statsLoading ? '...' : stats?.totalAgents || 0}
              </div>
              <p className="text-xs text-muted-foreground">
                AI agents in your fleet
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Active Agents</CardTitle>
              <Zap className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {statsLoading ? '...' : stats?.activeAgents || 0}
              </div>
              <p className="text-xs text-muted-foreground">
                Currently running
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Learning Data</CardTitle>
              <Brain className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {statsLoading ? '...' : stats?.totalLearningData || 0}
              </div>
              <p className="text-xs text-muted-foreground">
                Knowledge entries
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Recent Activity</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {statsLoading ? '...' : stats?.recentActivities || 0}
              </div>
              <p className="text-xs text-muted-foreground">
                Actions today
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Main Content Tabs */}
        <Tabs defaultValue="agents" className="space-y-4">
          <TabsList>
            <TabsTrigger value="agents">Agents</TabsTrigger>
            <TabsTrigger value="activity">Activity</TabsTrigger>
            <TabsTrigger value="analytics">Analytics</TabsTrigger>
          </TabsList>

          <TabsContent value="agents" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Your Agents</CardTitle>
                <CardDescription>
                  Manage and monitor your AI agents
                </CardDescription>
              </CardHeader>
              <CardContent>
                {agentsLoading ? (
                  <div className="flex items-center justify-center py-8">
                    <div className="text-muted-foreground">Loading agents...</div>
                  </div>
                ) : agentsData?.agents.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-8 space-y-4">
                    <Bot className="h-12 w-12 text-muted-foreground" />
                    <div className="text-center">
                      <h3 className="text-lg font-medium">No agents yet</h3>
                      <p className="text-muted-foreground">
                        Create your first AI agent to get started
                      </p>
                    </div>
                    <Button>Create Agent</Button>
                  </div>
                ) : (
                  <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                    {agentsData?.agents.map((agent) => (
                      <Card key={agent.id}>
                        <CardHeader className="pb-3">
                          <div className="flex items-center justify-between">
                            <CardTitle className="text-base">{agent.name}</CardTitle>
                            {getStatusIcon(agent.status)}
                          </div>
                          <CardDescription>
                            {agent.agentType} • {agent.agentId}
                          </CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-3">
                          <div className="flex items-center justify-between">
                            <Badge variant={getStatusBadgeVariant(agent.status)}>
                              {agent.status}
                            </Badge>
                            <Badge variant={agent.enabled ? "default" : "secondary"}>
                              {agent.enabled ? "Enabled" : "Disabled"}
                            </Badge>
                          </div>
                          
                          {agent.capabilities.length > 0 && (
                            <div>
                              <p className="text-sm font-medium mb-1">Capabilities:</p>
                              <div className="flex flex-wrap gap-1">
                                {agent.capabilities.slice(0, 3).map((capability, index) => (
                                  <Badge key={index} variant="outline" className="text-xs">
                                    {capability}
                                  </Badge>
                                ))}
                                {agent.capabilities.length > 3 && (
                                  <Badge variant="outline" className="text-xs">
                                    +{agent.capabilities.length - 3} more
                                  </Badge>
                                )}
                              </div>
                            </div>
                          )}
                          
                          {agent.lastHeartbeat && (
                            <p className="text-xs text-muted-foreground">
                              Last seen: {formatTimeAgo(agent.lastHeartbeat)}
                            </p>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="activity" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Recent Activity</CardTitle>
                <CardDescription>
                  Your recent actions and system events
                </CardDescription>
              </CardHeader>
              <CardContent>
                {activitiesLoading ? (
                  <div className="flex items-center justify-center py-8">
                    <div className="text-muted-foreground">Loading activity...</div>
                  </div>
                ) : activitiesData?.activities.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-8 space-y-4">
                    <Activity className="h-12 w-12 text-muted-foreground" />
                    <div className="text-center">
                      <h3 className="text-lg font-medium">No activity yet</h3>
                      <p className="text-muted-foreground">
                        Your actions will appear here
                      </p>
                    </div>
                  </div>
                ) : (
                  <ScrollArea className="h-96">
                    <div className="space-y-4">
                      {activitiesData?.activities.map((activity, index) => (
                        <div key={activity.id}>
                          <div className="flex items-start space-x-3">
                            <div className="flex-shrink-0">
                              {activity.success ? (
                                <CheckCircle2 className="h-5 w-5 text-green-500" />
                              ) : (
                                <XCircle className="h-5 w-5 text-red-500" />
                              )}
                            </div>
                            <div className="flex-1 min-w-0">
                              <p className="text-sm font-medium">
                                {formatActivityType(activity.activityType)}
                              </p>
                              {activity.resourceType && (
                                <p className="text-xs text-muted-foreground">
                                  {activity.resourceType}
                                </p>
                              )}
                              <p className="text-xs text-muted-foreground">
                                {formatTimeAgo(activity.createdAt)}
                              </p>
                            </div>
                          </div>
                          {index < activitiesData.activities.length - 1 && (
                            <Separator className="my-3" />
                          )}
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="analytics" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>System Analytics</CardTitle>
                <CardDescription>
                  Performance metrics and insights
                </CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col items-center justify-center py-8 space-y-4">
                <BarChart3 className="h-12 w-12 text-muted-foreground" />
                <div className="text-center">
                  <h3 className="text-lg font-medium">Analytics Coming Soon</h3>
                  <p className="text-muted-foreground">
                    Detailed analytics and insights will be available here
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
