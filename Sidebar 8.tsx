import { YmeraLogo } from "./YmeraLogo";
import { useWebSocket } from "../hooks/useWebSocket";

interface SidebarProps {
  activeSection: string;
  onSectionChange: (section: string) => void;
}

export function Sidebar({ activeSection, onSectionChange }: SidebarProps) {
  const { isConnected } = useWebSocket();

  const menuItems = [
    { id: 'dashboard', label: 'Dashboard', icon: 'chart-line' },
    { id: 'projects', label: 'Projects', icon: 'folder' },
    { id: 'monitoring', label: 'Monitoring', icon: 'heartbeat' },
    { id: 'docs', label: 'API Docs', icon: 'book' },
    { id: 'files', label: 'File Manager', icon: 'cloud-upload-alt' },
    { id: 'agents', label: 'AI Agents', icon: 'robot' },
  ];

  return (
    <aside className="w-64 bg-black/30 border-r border-white/10 p-6 flex flex-col">
      {/* YMERA Logo Component */}
      <div className="mb-8">
        <YmeraLogo />
      </div>
      
      {/* Navigation Menu */}
      <nav className="space-y-2 flex-1">
        {menuItems.map((item) => (
          <button
            key={item.id}
            onClick={() => onSectionChange(item.id)}
            className={`w-full flex items-center px-4 py-3 rounded-lg transition-colors ${
              activeSection === item.id
                ? 'bg-white/10 text-yellow-500 border border-yellow-500/30'
                : 'hover:bg-white/5'
            }`}
          >
            <i className={`fas fa-${item.icon} mr-3`}></i>
            {item.label}
          </button>
        ))}
      </nav>
      
      {/* Connection Status */}
      <div className="border-t border-white/10 pt-4 space-y-3">
        <div className="flex items-center justify-between text-sm">
          <span>WebSocket</span>
          <div className="flex items-center">
            <div className={`w-2 h-2 rounded-full mr-2 ${
              isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'
            }`}></div>
            <span className={isConnected ? 'text-green-400' : 'text-red-400'}>
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>
        <div className="flex items-center justify-between text-sm">
          <span>Server Health</span>
          <div className="flex items-center">
            <div className="w-2 h-2 rounded-full bg-green-500 mr-2"></div>
            <span className="text-green-400">Operational</span>
          </div>
        </div>
        <div className="flex items-center justify-between text-sm">
          <span>Database</span>
          <div className="flex items-center">
            <div className="w-2 h-2 rounded-full bg-green-500 mr-2"></div>
            <span className="text-green-400">Online</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
