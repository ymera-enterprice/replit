import { useState } from 'react';
import { Link, useLocation } from 'wouter';
import { 
  Home, 
  Bot, 
  FolderOpen, 
  Brain, 
  Users, 
  FileText, 
  BarChart3, 
  Shield, 
  Settings,
  TrendingUp,
  Zap,
  Target
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';

interface NavItem {
  icon: React.ElementType;
  label: string;
  path: string;
  badge?: string | number;
  badgeVariant?: 'default' | 'secondary' | 'success' | 'warning' | 'error';
}

const mainNavItems: NavItem[] = [
  { icon: Home, label: 'Dashboard', path: '/' },
  { icon: Bot, label: 'Intelligent Agents', path: '/agents', badge: 7 },
  { icon: FolderOpen, label: 'Projects', path: '/projects', badge: 12, badgeVariant: 'secondary' },
  { icon: Brain, label: 'Knowledge Graph', path: '/knowledge' },
  { icon: Users, label: 'Collaboration', path: '/collaboration' },
];

const systemNavItems: NavItem[] = [
  { icon: FileText, label: 'File Management', path: '/files' },
  { icon: BarChart3, label: 'Monitoring', path: '/monitoring' },
  { icon: Shield, label: 'Security', path: '/security' },
  { icon: Settings, label: 'Settings', path: '/settings' },
];

export default function Sidebar() {
  const [location] = useLocation();

  const isActiveRoute = (path: string) => {
    if (path === '/') {
      return location === '/';
    }
    return location.startsWith(path);
  };

  const getBadgeColor = (variant?: string) => {
    switch (variant) {
      case 'secondary':
        return 'bg-secondary text-dark-900';
      case 'success':
        return 'bg-success text-white';
      case 'warning':
        return 'bg-warning text-dark-900';
      case 'error':
        return 'bg-error text-white';
      default:
        return 'bg-primary text-white';
    }
  };

  return (
    <aside className="w-64 bg-dark-800 border-r border-dark-700 flex flex-col h-full">
      <nav className="flex-1 px-4 py-6 space-y-2">
        {/* Main Navigation */}
        <div className="space-y-1">
          <div className="text-xs font-medium text-dark-400 uppercase tracking-wider px-3 py-2">
            Platform
          </div>
          
          {mainNavItems.map((item) => {
            const Icon = item.icon;
            const isActive = isActiveRoute(item.path);
            
            return (
              <Link key={item.path} href={item.path}>
                <Button
                  variant="ghost"
                  className={`w-full justify-start px-3 py-2 h-auto ${
                    isActive
                      ? 'bg-primary/20 text-primary border border-primary/30 glow-primary'
                      : 'hover:bg-dark-700 text-dark-300 hover:text-dark-100'
                  } transition-all duration-200`}
                >
                  <Icon className="w-5 h-5 mr-3" />
                  <span className="flex-1 text-left">{item.label}</span>
                  {item.badge && (
                    <Badge 
                      className={`ml-auto h-5 px-2 text-xs ${getBadgeColor(item.badgeVariant)}`}
                    >
                      {item.badge}
                    </Badge>
                  )}
                </Button>
              </Link>
            );
          })}
        </div>
        
        {/* System Management */}
        <div className="space-y-1 pt-4">
          <div className="text-xs font-medium text-dark-400 uppercase tracking-wider px-3 py-2">
            System
          </div>
          
          {systemNavItems.map((item) => {
            const Icon = item.icon;
            const isActive = isActiveRoute(item.path);
            
            return (
              <Link key={item.path} href={item.path}>
                <Button
                  variant="ghost"
                  className={`w-full justify-start px-3 py-2 h-auto ${
                    isActive
                      ? 'bg-primary/20 text-primary border border-primary/30 glow-primary'
                      : 'hover:bg-dark-700 text-dark-300 hover:text-dark-100'
                  } transition-all duration-200`}
                >
                  <Icon className="w-5 h-5 mr-3" />
                  <span className="flex-1 text-left">{item.label}</span>
                  {item.badge && (
                    <Badge 
                      className={`ml-auto h-5 px-2 text-xs ${getBadgeColor(item.badgeVariant)}`}
                    >
                      {item.badge}
                    </Badge>
                  )}
                </Button>
              </Link>
            );
          })}
        </div>
      </nav>
      
      {/* Learning Status Panel */}
      <div className="p-4">
        <Card className="glass-effect border-dark-600">
          <CardContent className="p-4">
            <div className="text-xs font-medium text-dark-400 uppercase tracking-wider mb-3">
              Learning Status
            </div>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-dark-300">Active Learning</span>
                <div className="flex items-center space-x-1">
                  <TrendingUp className="w-3 h-3 text-success" />
                  <div className="w-2 h-2 rounded-full bg-success animate-pulse"></div>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-dark-300">Knowledge Sync</span>
                <div className="flex items-center space-x-1">
                  <Zap className="w-3 h-3 text-warning" />
                  <div className="w-2 h-2 rounded-full bg-warning animate-pulse"></div>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-dark-300">Pattern Discovery</span>
                <div className="flex items-center space-x-1">
                  <Target className="w-3 h-3 text-primary" />
                  <div className="w-2 h-2 rounded-full bg-primary animate-pulse"></div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </aside>
  );
}
