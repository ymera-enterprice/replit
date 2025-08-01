import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/hooks/useAuth";

const navigation = [
  { name: 'Dashboard', href: '/', icon: 'fas fa-chart-line' },
  { name: 'Multi-Agents', href: '/agents', icon: 'fas fa-robot' },
  { name: 'File Manager', href: '/files', icon: 'fas fa-folder' },
  { name: 'Collaboration', href: '/collaboration', icon: 'fas fa-users' },
  { name: 'Projects', href: '/projects', icon: 'fas fa-project-diagram' },
  { name: 'Analytics', href: '/analytics', icon: 'fas fa-chart-bar' },
  { name: 'Security', href: '/security', icon: 'fas fa-shield-alt' },
];

export default function Sidebar() {
  const [location, setLocation] = useLocation();
  const { user } = useAuth();

  const handleLogout = () => {
    window.location.href = "/api/logout";
  };

  return (
    <div className="w-64 glass-effect border-r border-border flex flex-col lg:flex hidden">
      {/* Logo Section */}
      <div className="p-6 border-b border-border">
        <div className="font-luxury text-3xl gradient-text text-center">Ymera</div>
        <div className="text-xs text-center text-muted-foreground mt-1">Enterprise Platform</div>
      </div>
      
      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2">
        {navigation.map((item) => {
          const isActive = location === item.href;
          return (
            <button
              key={item.name}
              onClick={() => setLocation(item.href)}
              className={`w-full flex items-center space-x-3 p-3 rounded-lg transition-all text-left ${
                isActive
                  ? 'bg-primary/20 text-primary border-r-2 border-primary'
                  : 'text-muted-foreground hover:bg-white/10 hover:text-foreground'
              }`}
            >
              <i className={`${item.icon} w-4 text-center`}></i>
              <span>{item.name}</span>
            </button>
          );
        })}
      </nav>
      
      {/* User Profile */}
      <div className="p-4 border-t border-border">
        <div className="flex items-center space-x-3 mb-3">
          <img 
            src={user?.profileImageUrl || "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=80&h=80"} 
            alt="User profile" 
            className="w-10 h-10 rounded-full object-cover" 
          />
          <div className="flex-1 min-w-0">
            <div className="text-sm font-medium text-foreground truncate">
              {user?.firstName} {user?.lastName}
            </div>
            <div className="text-xs text-muted-foreground truncate">Enterprise Admin</div>
          </div>
          <div className="w-2 h-2 bg-success rounded-full status-indicator"></div>
        </div>
        
        <Button 
          onClick={handleLogout}
          variant="outline" 
          size="sm" 
          className="w-full glass-effect"
        >
          <i className="fas fa-sign-out-alt mr-2"></i>
          Sign Out
        </Button>
      </div>
    </div>
  );
}
