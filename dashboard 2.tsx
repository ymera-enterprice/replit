import { useState, useEffect } from "react";
import { useLocation } from "wouter";
import { useQuery } from "@tanstack/react-query";
import { useAuth } from "@/lib/auth";
import { useWebSocket } from "@/hooks/use-websocket";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { useToast } from "@/hooks/use-toast";
import {
  Bot,
  Search,
  Bell,
  FileText,
  Users,
  Zap,
  CheckCircle,
  Upload,
  BarChart3,
  Settings,
  Shield,
  UserPlus,
  Activity,
  Clock,
  AlertTriangle,
} from "lucide-react";

interface DashboardStats {
  totalFiles: number;
  activeUsers: number;
  totalOperations: number;
  systemStatus: string;
}

interface WebSocketStats {
  connections: number;
  messagesPerSec: number;
  latency: number;
}

interface FileOperation {
  id: string;
  operation: string;
  status: string;
  createdAt: string;
  user?: {
    id: string;
    username: string;
    displayName?: string;
  };
  file?: {
    id: string;
    filename: string;
    originalName: string;
    size: number;
  };
}

interface ActivityLog {
  id: string;
  action: string;
  createdAt: string;
  user?: {
    id: string;
    username: string;
    displayName?: string;
  };
  details?: any;
}

export default function DashboardPage() {
  const [location, setLocation] = useLocation();
  const { user, logout, isLoading } = useAuth();
  const { toast } = useToast();
  const [searchQuery, setSearchQuery] = useState("");

  // WebSocket connection
  const { isConnected, lastMessage } = useWebSocket();

  // Redirect if not authenticated
  useEffect(() => {
    if (!user && !isLoading) {
      navigate("/login", { replace: true });
    }
  }, [user, isLoading, navigate]);

  // Fetch dashboard stats
  const { data: stats } = useQuery<DashboardStats>({
    queryKey: ["/api/dashboard/stats"],
    enabled: !!user,
  });

  // Fetch WebSocket stats
  const { data: wsStats } = useQuery<WebSocketStats>({
    queryKey: ["/api/dashboard/websocket-stats"],
    enabled: !!user,
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  // Fetch file operations
  const { data: fileOperations } = useQuery<FileOperation[]>({
    queryKey: ["/api/file-operations"],
    enabled: !!user,
  });

  // Fetch activity logs
  const { data: activityLogs } = useQuery<ActivityLog[]>({
    queryKey: ["/api/activity"],
    enabled: !!user,
  });

  // Handle logout
  const handleLogout = async () => {
    try {
      await logout();
      toast({
        title: "Logged out",
        description: "You have been successfully logged out.",
      });
      navigate("/login", { replace: true });
    } catch (error: any) {
      toast({
        title: "Logout failed",
        description: error.message,
        variant: "destructive",
      });
    }
  };

  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      try {
        const message = JSON.parse(lastMessage);
        if (message.type === 'user_activity' || message.type === 'file_activity') {
          toast({
            title: "Real-time Update",
            description: `${message.user} ${message.action}${message.filename ? ` ${message.filename}` : ''}`,
          });
        }
      } catch (error) {
        console.error("Error parsing WebSocket message:", error);
      }
    }
  }, [lastMessage, toast]);

  if (isLoading || !user) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="text-center">
          <Bot className="h-12 w-12 mx-auto mb-4 text-primary animate-pulse" />
          <p className="text-muted-foreground">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  const getInitials = (name: string) => {
    return name
      .split(" ")
      .map((n) => n[0])
      .join("")
      .toUpperCase()
      .slice(0, 2);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  };

  const getFileTypeColor = (filename: string) => {
    const ext = filename.split(".").pop()?.toLowerCase();
    switch (ext) {
      case "pdf":
        return "file-type-pdf";
      case "xlsx":
      case "xls":
        return "file-type-xlsx";
      case "pptx":
      case "ppt":
        return "file-type-pptx";
      case "docx":
      case "doc":
        return "file-type-docx";
      default:
        return "bg-muted text-muted-foreground";
    }
  };

  const formatTimeAgo = (timestamp: string) => {
    const now = new Date();
    const time = new Date(timestamp);
    const diffInSeconds = Math.floor((now.getTime() - time.getTime()) / 1000);

    if (diffInSeconds < 60) return "Just now";
    if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)} min ago`;
    if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)} hr ago`;
    return `${Math.floor(diffInSeconds / 86400)} days ago`;
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Top Navigation */}
      <nav className="bg-card border-b border-border px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="w-8 h-8 gradient-bg rounded-lg flex items-center justify-center">
              <span className="text-white font-bold">Y</span>
            </div>
            <h1 className="text-xl font-bold text-foreground">YMERA Enterprise</h1>
            <div className="hidden md:flex items-center space-x-1 ml-8">
              <Badge variant="secondary" className="bg-success/10 text-success">
                <div className="w-2 h-2 bg-success rounded-full mr-1 animate-pulse" />
                System Healthy
              </Badge>
              <Badge variant="secondary" className="bg-info/10 text-info">
                <div className="w-2 h-2 bg-info rounded-full mr-1" />
                DB Connected
              </Badge>
              <Badge variant="secondary" className="bg-secondary/10 text-secondary">
                <div className="w-2 h-2 bg-secondary rounded-full mr-1" />
                {isConnected ? "WebSocket Active" : "WebSocket Disconnected"}
              </Badge>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            {/* Search Bar */}
            <div className="relative">
              <Input
                type="text"
                placeholder="Search files, users, projects..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-80 pl-10"
              />
              <Search className="w-5 h-5 text-muted-foreground absolute left-3 top-2.5" />
            </div>

            {/* Notifications */}
            <Button variant="ghost" size="icon" className="relative">
              <Bell className="w-5 h-5" />
              <span className="absolute -top-1 -right-1 bg-destructive text-destructive-foreground text-xs rounded-full w-5 h-5 flex items-center justify-center">
                3
              </span>
            </Button>

            {/* User Menu */}
            <div className="flex items-center space-x-3">
              <Avatar className="w-8 h-8">
                <AvatarFallback className="bg-primary text-primary-foreground">
                  {getInitials(user.displayName || user.username)}
                </AvatarFallback>
              </Avatar>
              <div className="hidden md:block">
                <p className="text-sm font-medium text-foreground">
                  {user.displayName || user.username}
                </p>
                <p className="text-xs text-muted-foreground">System Administrator</p>
              </div>
              <Button variant="ghost" onClick={handleLogout}>
                Logout
              </Button>
            </div>
          </div>
        </div>
      </nav>

      <div className="flex">
        {/* Sidebar */}
        <aside className="w-64 bg-card border-r border-border min-h-screen">
          <nav className="p-4 space-y-2">
            <Button variant="secondary" className="w-full justify-start">
              <BarChart3 className="w-5 h-5 mr-3" />
              Dashboard
            </Button>
            <Button variant="ghost" className="w-full justify-start">
              <FileText className="w-5 h-5 mr-3" />
              File Management
            </Button>
            <Button variant="ghost" className="w-full justify-start">
              <Users className="w-5 h-5 mr-3" />
              Collaboration
            </Button>
            <Button variant="ghost" className="w-full justify-start">
              <Search className="w-5 h-5 mr-3" />
              Advanced Search
            </Button>
            <Button variant="ghost" className="w-full justify-start">
              <Activity className="w-5 h-5 mr-3" />
              Analytics
            </Button>

            <div className="pt-4 mt-4 border-t border-border">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">
                Administration
              </p>
              <Button variant="ghost" className="w-full justify-start">
                <UserPlus className="w-5 h-5 mr-3" />
                User Management
              </Button>
              <Button variant="ghost" className="w-full justify-start">
                <Settings className="w-5 h-5 mr-3" />
                System Settings
              </Button>
              <Button variant="ghost" className="w-full justify-start">
                <Shield className="w-5 h-5 mr-3" />
                Security
              </Button>
            </div>
          </nav>
        </aside>

        {/* Main Content */}
        <main className="flex-1 p-6">
          {/* Dashboard Header */}
          <div className="mb-8">
            <h2 className="text-3xl font-bold text-foreground mb-2">Enterprise Dashboard</h2>
            <p className="text-muted-foreground">
              Complete system integration with real-time monitoring and file management
            </p>
          </div>

          {/* System Status Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="w-12 h-12 bg-success/10 rounded-lg flex items-center justify-center">
                    <CheckCircle className="w-6 h-6 text-success" />
                  </div>
                  <Badge className="bg-success/10 text-success">Healthy</Badge>
                </div>
                <h3 className="text-sm font-medium text-muted-foreground mb-1">System Status</h3>
                <p className="text-2xl font-bold text-foreground">100%</p>
                <p className="text-xs text-muted-foreground mt-1">All services operational</p>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="w-12 h-12 bg-info/10 rounded-lg flex items-center justify-center">
                    <FileText className="w-6 h-6 text-info" />
                  </div>
                  <Badge className="bg-info/10 text-info">Active</Badge>
                </div>
                <h3 className="text-sm font-medium text-muted-foreground mb-1">Files Processed</h3>
                <p className="text-2xl font-bold text-foreground">{stats?.totalFiles || 0}</p>
                <p className="text-xs text-muted-foreground mt-1">+12% from last week</p>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="w-12 h-12 bg-secondary/10 rounded-lg flex items-center justify-center">
                    <Users className="w-6 h-6 text-secondary" />
                  </div>
                  <Badge className="bg-secondary/10 text-secondary">Online</Badge>
                </div>
                <h3 className="text-sm font-medium text-muted-foreground mb-1">Active Users</h3>
                <p className="text-2xl font-bold text-foreground">{stats?.activeUsers || 0}</p>
                <p className="text-xs text-muted-foreground mt-1">Across departments</p>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="w-12 h-12 bg-warning/10 rounded-lg flex items-center justify-center">
                    <Zap className="w-6 h-6 text-warning" />
                  </div>
                  <Badge className="bg-warning/10 text-warning">Optimized</Badge>
                </div>
                <h3 className="text-sm font-medium text-muted-foreground mb-1">Performance</h3>
                <p className="text-2xl font-bold text-foreground">{wsStats?.latency || 0}ms</p>
                <p className="text-xs text-muted-foreground mt-1">Average response time</p>
              </CardContent>
            </Card>
          </div>

          {/* Main Content Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* File Management Panel */}
            <Card className="lg:col-span-2">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>Recent File Operations</CardTitle>
                  <Button variant="outline" size="sm">
                    View All
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <div className="overflow-hidden">
                  <div className="space-y-4">
                    {fileOperations?.slice(0, 5).map((operation) => (
                      <div
                        key={operation.id}
                        className="flex items-center space-x-3 py-3 border-b border-border last:border-b-0"
                      >
                        <div className="w-8 h-8 bg-info/10 rounded-lg flex items-center justify-center">
                          <FileText className="w-4 h-4 text-info" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-foreground truncate">
                            {operation.file?.originalName || "Unknown file"}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {operation.file?.size ? formatFileSize(operation.file.size) : "Unknown size"}
                          </p>
                        </div>
                        <Badge
                          className={`${getFileTypeColor(
                            operation.file?.originalName || ""
                          )} text-xs`}
                        >
                          {operation.file?.originalName?.split(".").pop()?.toUpperCase() || "FILE"}
                        </Badge>
                        <div className="text-right">
                          <p className="text-sm text-foreground">
                            {operation.user?.displayName || operation.user?.username}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {formatTimeAgo(operation.createdAt)}
                          </p>
                        </div>
                        <Badge
                          variant={operation.status === "completed" ? "secondary" : "outline"}
                          className={
                            operation.status === "completed"
                              ? "bg-success/10 text-success"
                              : operation.status === "processing"
                              ? "bg-warning/10 text-warning"
                              : "bg-destructive/10 text-destructive"
                          }
                        >
                          <div className="w-1.5 h-1.5 rounded-full mr-1 bg-current" />
                          {operation.status}
                        </Badge>
                      </div>
                    )) || (
                      <div className="text-center py-8 text-muted-foreground">
                        No file operations found
                      </div>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Real-time Activity Panel */}
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>Live Activity</CardTitle>
                  <div className="w-2 h-2 bg-success rounded-full animate-pulse" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {activityLogs?.slice(0, 6).map((log) => (
                    <div key={log.id} className="flex items-start space-x-3">
                      <div className="w-8 h-8 bg-info/10 rounded-full flex items-center justify-center flex-shrink-0">
                        {log.action === "login" ? (
                          <UserPlus className="w-4 h-4 text-info" />
                        ) : log.action === "file_upload" ? (
                          <Upload className="w-4 h-4 text-success" />
                        ) : log.action === "logout" ? (
                          <Activity className="w-4 h-4 text-muted-foreground" />
                        ) : (
                          <AlertTriangle className="w-4 h-4 text-warning" />
                        )}
                      </div>
                      <div className="min-w-0 flex-1">
                        <p className="text-sm text-foreground">
                          <span className="font-medium">
                            {log.user?.displayName || log.user?.username || "System"}
                          </span>{" "}
                          {log.action === "login"
                            ? "logged in"
                            : log.action === "logout"
                            ? "logged out"
                            : log.action === "file_upload"
                            ? "uploaded a file"
                            : log.action}
                        </p>
                        <p className="text-xs text-muted-foreground mt-1">
                          {formatTimeAgo(log.createdAt)}
                        </p>
                      </div>
                    </div>
                  )) || (
                    <div className="text-center py-8 text-muted-foreground">
                      No recent activity
                    </div>
                  )}
                </div>

                {/* Quick Actions */}
                <div className="mt-6 pt-6 border-t border-border">
                  <h4 className="text-sm font-medium text-foreground mb-3">Quick Actions</h4>
                  <div className="space-y-2">
                    <Button variant="ghost" className="w-full justify-start" size="sm">
                      <Upload className="w-4 h-4 mr-2" />
                      Upload Files
                    </Button>
                    <Button variant="ghost" className="w-full justify-start" size="sm">
                      <Users className="w-4 h-4 mr-2" />
                      Start Collaboration
                    </Button>
                    <Button variant="ghost" className="w-full justify-start" size="sm">
                      <BarChart3 className="w-4 h-4 mr-2" />
                      Generate Report
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Advanced Features Section */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
            {/* WebSocket Status Panel */}
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>WebSocket Status</CardTitle>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-success rounded-full animate-pulse" />
                    <span className="text-xs text-success">
                      {isConnected ? "Connected" : "Disconnected"}
                    </span>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Active Connections</span>
                    <span className="text-sm font-medium text-foreground">
                      {wsStats?.connections || 0}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Messages/sec</span>
                    <span className="text-sm font-medium text-foreground">
                      {wsStats?.messagesPerSec || 0}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Latency</span>
                    <span className="text-sm font-medium text-foreground">
                      {wsStats?.latency || 0}ms
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Search & Indexing Status */}
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>Search & Indexing</CardTitle>
                  <Badge className="bg-info/10 text-info">Active</Badge>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Indexed Documents</span>
                    <span className="text-sm font-medium text-foreground">
                      {stats?.totalFiles || 0}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Search Queries Today</span>
                    <span className="text-sm font-medium text-foreground">1,293</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Index Size</span>
                    <span className="text-sm font-medium text-foreground">2.4 GB</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </main>
      </div>
    </div>
  );
}
