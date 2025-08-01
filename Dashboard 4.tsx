import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { 
  Search, 
  Bell, 
  Moon, 
  Sun, 
  ChevronDown,
  Filter,
  Grid3X3,
  List,
  Settings,
  Share2,
  Clock,
  Star,
  Users,
  Home,
  Folder,
  Edit,
  MessageSquareDashed,
  Code,
  Shield,
  Plus
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { YmeraLogo } from '@/components/YmeraLogo';
import { FileUpload } from '@/components/FileUpload';
import { FileGrid } from '@/components/FileGrid';
import { CollaborationPanel } from '@/components/CollaborationPanel';
import { ProcessingQueue } from '@/components/ProcessingQueue';
import { useWebSocket } from '@/lib/websocket';
import { useTheme } from '@/lib/theme';

export default function Dashboard() {
  const [selectedFile, setSelectedFile] = useState<any>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const { theme, setTheme } = useTheme();
  const { connectionStatus, activeUsers, processingFiles } = useWebSocket();

  // Fetch user's files
  const { data: filesData, isLoading: filesLoading, refetch: refetchFiles } = useQuery({
    queryKey: ['/api/files'],
    queryFn: async () => {
      const response = await fetch('/api/files', {
        headers: {
          'X-User-Id': 'user123',
          'X-User-Name': 'John Doe',
        },
      });
      if (!response.ok) throw new Error('Failed to fetch files');
      return response.json();
    },
  });

  // Fetch user activity
  const { data: activityData } = useQuery({
    queryKey: ['/api/activity'],
    queryFn: async () => {
      const response = await fetch('/api/activity?limit=10', {
        headers: {
          'X-User-Id': 'user123',
          'X-User-Name': 'John Doe',
        },
      });
      if (!response.ok) throw new Error('Failed to fetch activity');
      return response.json();
    },
  });

  const handleFileUploadComplete = () => {
    refetchFiles();
  };

  const handleFileSelect = (file: any) => {
    setSelectedFile(file);
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;
    
    // Implement search functionality
    console.log('Searching for:', searchQuery);
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Navigation Header */}
      <header className="bg-white dark:bg-gray-800 shadow-lg border-b border-gray-200 dark:border-gray-700 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Logo */}
            <YmeraLogo />

            {/* Global Search */}
            <div className="flex-1 max-w-2xl mx-8">
              <form onSubmit={handleSearch} className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Search className="h-5 w-5 text-gray-400" />
                </div>
                <Input
                  type="text"
                  placeholder="Search files, content, collaborators..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-20 py-2 bg-gray-100 dark:bg-gray-700 border-gray-300 dark:border-gray-600 focus:ring-2 focus:ring-blue-500"
                />
                <div className="absolute inset-y-0 right-0 pr-3 flex items-center">
                  <kbd className="hidden sm:inline-block px-2 py-1 text-xs font-medium text-gray-500 bg-gray-200 dark:bg-gray-600 rounded">
                    ⌘K
                  </kbd>
                </div>
              </form>
            </div>

            {/* User Controls */}
            <div className="flex items-center space-x-4">
              {/* Notifications */}
              <div className="relative">
                <Button variant="ghost" size="sm">
                  <Bell className="h-5 w-5" />
                  <span className="absolute -top-1 -right-1 w-3 h-3 bg-orange-500 rounded-full animate-bounce-gentle"></span>
                </Button>
              </div>

              {/* Dark Mode Toggle */}
              <Button variant="ghost" size="sm" onClick={toggleTheme}>
                {theme === 'dark' ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
              </Button>

              {/* User Profile */}
              <div className="flex items-center space-x-2">
                <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-orange-500 rounded-full flex items-center justify-center">
                  <span className="text-white text-sm font-medium">JD</span>
                </div>
                <span className="hidden md:block text-sm font-medium">John Doe</span>
                <ChevronDown className="h-4 w-4 text-gray-400" />
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="flex h-[calc(100vh-4rem)]">
        {/* Sidebar Navigation */}
        <aside className="w-64 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col">
          {/* Quick Actions */}
          <div className="p-4 border-b border-gray-200 dark:border-gray-700">
            <Button className="w-full bg-gradient-to-r from-blue-600 to-blue-800 hover:from-blue-700 hover:to-blue-900 text-white shadow-lg">
              <Plus className="w-4 h-4 mr-2" />
              Upload Files
            </Button>
          </div>

          {/* Navigation Menu */}
          <nav className="flex-1 p-4 space-y-2 overflow-y-auto custom-scrollbar">
            <div className="space-y-1">
              <div className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3">
                File Management
              </div>
              
              <a href="#" className="flex items-center px-3 py-2 text-sm font-medium text-blue-600 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <Home className="w-5 h-5 mr-3" />
                Dashboard
              </a>
              
              <a href="#" className="flex items-center px-3 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors">
                <Folder className="w-5 h-5 mr-3" />
                My Files
                <Badge variant="secondary" className="ml-auto">
                  {filesData?.files?.length || 0}
                </Badge>
              </a>
              
              <a href="#" className="flex items-center px-3 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors">
                <Users className="w-5 h-5 mr-3" />
                Shared Files
                <Badge className="ml-auto bg-orange-500">23</Badge>
              </a>
              
              <a href="#" className="flex items-center px-3 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors">
                <Clock className="w-5 h-5 mr-3" />
                Recent Activity
              </a>
              
              <a href="#" className="flex items-center px-3 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors">
                <Star className="w-5 h-5 mr-3" />
                Favorites
              </a>
            </div>

            <div className="space-y-1 pt-4">
              <div className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3">
                Collaboration
              </div>
              
              <a href="#" className="flex items-center px-3 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors">
                <Edit className="w-5 h-5 mr-3" />
                Live Editing
                <div className="ml-auto flex -space-x-1">
                  <div className="w-4 h-4 bg-green-500 rounded-full border-2 border-white animate-pulse"></div>
                  <div className="w-4 h-4 bg-blue-500 rounded-full border-2 border-white"></div>
                </div>
              </a>
              
              <a href="#" className="flex items-center px-3 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors">
                <MessageSquareDashed className="w-5 h-5 mr-3" />
                MessageSquareDashed & Reviews
              </a>
              
              <a href="#" className="flex items-center px-3 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors">
                <Share2 className="w-5 h-5 mr-3" />
                Share Management
              </a>
            </div>

            <div className="space-y-1 pt-4">
              <div className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3">
                Advanced
              </div>
              
              <a href="#" className="flex items-center px-3 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors">
                <Code className="w-5 h-5 mr-3" />
                Version History
              </a>
              
              <a href="#" className="flex items-center px-3 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors">
                <Shield className="w-5 h-5 mr-3" />
                Security & Audit
              </a>
              
              <a href="#" className="flex items-center px-3 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors">
                <Settings className="w-5 h-5 mr-3" />
                Settings
              </a>
            </div>
          </nav>

          {/* Storage Usage */}
          <div className="p-4 border-t border-gray-200 dark:border-gray-700">
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-600 dark:text-gray-400">Storage Used</span>
                <span className="font-medium text-gray-900 dark:text-gray-100">68.5 GB of 100 GB</span>
              </div>
              <Progress value={68.5} className="h-2" />
            </div>
          </div>
        </aside>

        {/* Main Content Area */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {/* File Operations Toolbar */}
          <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">File Dashboard</h1>
                <div className="flex items-center text-sm text-gray-500 dark:text-gray-400">
                  <Folder className="w-4 h-4 mr-1" />
                  <span>/Documents/Projects/YMERA-Phase2</span>
                </div>
              </div>
              
              <div className="flex items-center space-x-3">
                {/* View Controls */}
                <div className="flex items-center bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
                  <Button
                    variant={viewMode === 'grid' ? 'default' : 'ghost'}
                    size="sm"
                    onClick={() => setViewMode('grid')}
                    className="h-8 w-8 p-0"
                  >
                    <Grid3X3 className="w-4 h-4" />
                  </Button>
                  <Button
                    variant={viewMode === 'list' ? 'default' : 'ghost'}
                    size="sm"
                    onClick={() => setViewMode('list')}
                    className="h-8 w-8 p-0"
                  >
                    <List className="w-4 h-4" />
                  </Button>
                </div>
                
                {/* Sort & Filter */}
                <Select defaultValue="modified">
                  <SelectTrigger className="w-40">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="modified">Sort by: Modified</SelectItem>
                    <SelectItem value="name">Sort by: Name</SelectItem>
                    <SelectItem value="size">Sort by: Size</SelectItem>
                    <SelectItem value="type">Sort by: Type</SelectItem>
                  </SelectContent>
                </Select>
                
                <Button variant="outline">
                  <Filter className="w-4 h-4 mr-2" />
                  Filters
                </Button>
              </div>
            </div>
          </div>

          {/* File Content */}
          <div className="flex-1 overflow-y-auto p-6 custom-scrollbar">
            {/* File Upload Area */}
            <div className="mb-6">
              <FileUpload onUploadComplete={handleFileUploadComplete} />
            </div>

            {/* Processing Queue */}
            <ProcessingQueue />

            {/* File Grid */}
            <div className="mb-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Recent Files</h2>
                <a href="#" className="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 text-sm font-medium">
                  View All
                </a>
              </div>
              
              {filesLoading ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                  {[...Array(8)].map((_, i) => (
                    <Card key={i} className="animate-pulse">
                      <CardContent className="p-4">
                        <div className="aspect-square bg-gray-200 dark:bg-gray-700 rounded-lg mb-3"></div>
                        <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded mb-2"></div>
                        <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-2/3"></div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              ) : (
                <FileGrid
                  files={filesData?.files || []}
                  viewMode={viewMode}
                  onFileSelect={handleFileSelect}
                />
              )}
            </div>

            {/* Analytics Dashboard */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Processing Statistics */}
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Processing Stats</CardTitle>
                  <Code className="w-4 h-4 text-blue-600" />
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Files Processed Today</span>
                      <span className="font-bold text-2xl text-blue-600">247</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Success Rate</span>
                      <span className="font-bold text-2xl text-green-600">98.5%</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Avg Processing Time</span>
                      <span className="font-bold text-2xl text-orange-600">2.3s</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Collaboration Activity */}
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Live Activity</CardTitle>
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    <span className="text-xs text-green-600 font-medium">Live</span>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {activityData?.activity?.slice(0, 3).map((activity: any, index: number) => (
                      <div key={index} className="flex items-center space-x-3">
                        <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center text-white text-xs font-medium">
                          {activity.userName.charAt(0)}
                        </div>
                        <div className="flex-1">
                          <p className="text-sm text-gray-900 dark:text-gray-100">
                            {activity.userName} {activity.action}ed a file
                          </p>
                          <p className="text-xs text-gray-500 dark:text-gray-400">
                            {new Date(activity.createdAt).toLocaleTimeString()}
                          </p>
                        </div>
                      </div>
                    )) || (
                      <p className="text-sm text-gray-500 dark:text-gray-400">No recent activity</p>
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Security Status */}
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Security Status</CardTitle>
                  <Shield className="w-4 h-4 text-green-600" />
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Files Encrypted</span>
                      <span className="font-bold text-2xl text-green-600">100%</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Malware Scanned</span>
                      <span className="font-bold text-2xl text-green-600">247/247</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Access Violations</span>
                      <span className="font-bold text-2xl text-red-600">0</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </main>

        {/* Right Sidebar - File Details & Collaboration */}
        {selectedFile && (
          <CollaborationPanel 
            file={selectedFile} 
            onClose={() => setSelectedFile(null)} 
          />
        )}
      </div>

      {/* WebSocket Status Indicator */}
      <div className="fixed bottom-6 right-6 z-50">
        <Card className="min-w-[200px]">
          <CardContent className="p-4">
            <div className="flex items-center space-x-3">
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${
                  connectionStatus === 'connected' ? 'bg-green-500 animate-pulse' :
                  connectionStatus === 'connecting' ? 'bg-orange-500 animate-pulse' :
                  'bg-red-500'
                }`}></div>
                <span className="text-sm font-medium">
                  WebSocket {connectionStatus === 'connected' ? 'Connected' : 'Disconnected'}
                </span>
              </div>
            </div>
            <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
              <div className="flex justify-between">
                <span>Active Users:</span>
                <span className="font-medium text-blue-600">{activeUsers}</span>
              </div>
              <div className="flex justify-between">
                <span>Files Processing:</span>
                <span className="font-medium text-orange-600">{processingFiles}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
