import { Link, useLocation } from 'wouter';
import { Users, Brain, Activity, Settings, Home } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { useQuery } from '@tanstack/react-query';

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: Home },
  { name: 'Agents', href: '/agents', icon: Users },
  { name: 'Learning', href: '/learning', icon: Brain },
  { name: 'Monitoring', href: '/monitoring', icon: Activity },
];

export default function Header() {
  const [location] = useLocation();

  const { data: systemHealth } = useQuery({
    queryKey: ['/api/monitoring/health'],
    refetchInterval: 30000,
  });

  const isOnline = systemHealth?.data?.overall_health !== 'poor';

  return (
    <header className="glass-header sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Link href="/">
              <div className="font-great-vibes text-4xl ymera-text-gradient cursor-pointer">
                Ymera
              </div>
            </Link>
            <div className="text-sm text-muted-foreground">
              Enterprise Platform v4.0
            </div>
          </div>

          <nav className="hidden md:flex items-center space-x-6">
            {navigation.map((item) => {
              const Icon = item.icon;
              const isActive = location === item.href || (item.href === '/dashboard' && location === '/');
              
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-colors ${
                    isActive
                      ? 'bg-primary/20 text-primary'
                      : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{item.name}</span>
                </Link>
              );
            })}
          </nav>

          <div className="flex items-center space-x-4">
            <Badge 
              variant={isOnline ? "default" : "destructive"}
              className="glass-card px-3 py-2 rounded-full"
            >
              <div className="flex items-center space-x-2">
                <div 
                  className={`w-2 h-2 rounded-full pulse-dot ${
                    isOnline ? 'bg-green-400' : 'bg-red-400'
                  }`}
                />
                <span className="text-sm">
                  {isOnline ? 'System Online' : 'System Issues'}
                </span>
              </div>
            </Badge>
            
            <button className="glass-card p-2 rounded-full hover:bg-white/20 transition-all">
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
