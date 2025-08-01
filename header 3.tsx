import { Bell, Settings, User, LogOut } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { useWebSocket } from '@/hooks/use-websocket';

export default function Header() {
  const { isConnected } = useWebSocket();

  const playLogoAnimation = () => {
    console.log('Logo animation triggered');
  };

  return (
    <header className="bg-dark-800 border-b border-dark-700 px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Logo Section */}
        <div className="flex items-center space-x-4">
          <div 
            className="flex items-center space-x-3 cursor-pointer"
            onClick={playLogoAnimation}
          >
            <div className="font-script text-4xl bg-gradient-to-r from-secondary via-primary to-accent bg-clip-text text-transparent">
              Ymera
            </div>
            <div className="text-xs text-dark-400 font-script opacity-70">
              by Mohamed Mansour
            </div>
          </div>
          <div className="h-8 w-px bg-dark-600"></div>
          <div className="text-lg font-semibold text-dark-200">
            Multi-Agent Learning Platform
          </div>
        </div>
        
        {/* Real-time Status & User Actions */}
        <div className="flex items-center space-x-6">
          {/* System Status */}
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-success animate-pulse' : 'bg-error'}`}></div>
            <span className="text-sm text-dark-300">
              System {isConnected ? 'Online' : 'Offline'}
            </span>
          </div>
          
          {/* Active Agents Count */}
          <div className="glass-effect px-3 py-1 rounded-lg">
            <span className="text-sm text-dark-300">Active Agents: </span>
            <span className="text-primary font-semibold">7</span>
          </div>
          
          {/* Notifications */}
          <Button
            variant="ghost" 
            size="icon"
            className="relative hover:bg-dark-700"
          >
            <Bell className="h-5 w-5 text-dark-300" />
            <Badge className="absolute -top-1 -right-1 h-5 w-5 p-0 text-xs bg-error">
              3
            </Badge>
          </Button>
          
          {/* Settings */}
          <Button
            variant="ghost"
            size="icon"
            className="hover:bg-dark-700"
          >
            <Settings className="h-5 w-5 text-dark-300" />
          </Button>
          
          {/* User Profile */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" className="flex items-center space-x-3 hover:bg-dark-700">
                <div className="relative">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-r from-primary to-secondary flex items-center justify-center">
                    <User className="h-4 w-4 text-white" />
                  </div>
                  <div className="absolute -top-1 -right-1 w-3 h-3 rounded-full bg-success border-2 border-dark-800"></div>
                </div>
                <div className="text-left">
                  <div className="text-sm font-medium text-dark-100">Admin User</div>
                  <div className="text-xs text-dark-400">System Administrator</div>
                </div>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="bg-dark-800 border-dark-700">
              <DropdownMenuItem className="text-dark-200 hover:bg-dark-700">
                <User className="mr-2 h-4 w-4" />
                Profile
              </DropdownMenuItem>
              <DropdownMenuItem className="text-dark-200 hover:bg-dark-700">
                <Settings className="mr-2 h-4 w-4" />
                Settings
              </DropdownMenuItem>
              <DropdownMenuSeparator className="bg-dark-700" />
              <DropdownMenuItem className="text-error hover:bg-dark-700">
                <LogOut className="mr-2 h-4 w-4" />
                Logout
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </header>
  );
}
