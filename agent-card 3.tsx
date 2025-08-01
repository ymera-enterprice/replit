import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { 
  Brain, 
  Zap, 
  Activity, 
  Play, 
  Pause, 
  Eye,
  MoreVertical,
  Settings
} from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import type { Agent } from '@/types/agent';

interface AgentCardProps {
  agent: Agent;
  showDetails?: boolean;
  onView?: (agent: Agent) => void;
  onStart?: (agent: Agent) => void;
  onPause?: (agent: Agent) => void;
}

const getAgentIcon = (type: string) => {
  switch (type) {
    case 'task_executor':
      return Activity;
    case 'data_processor':
      return Zap;
    case 'knowledge_manager':
      return Brain;
    case 'coordinator':
      return Activity;
    case 'analyzer':
      return Brain;
    case 'communicator':
      return Activity;
    default:
      return Brain;
  }
};

const getAgentIconColor = (type: string) => {
  switch (type) {
    case 'task_executor':
      return 'bg-blue-500/20';
    case 'data_processor':
      return 'bg-purple-500/20';
    case 'knowledge_manager':
      return 'bg-green-500/20';
    case 'coordinator':
      return 'bg-orange-500/20';
    case 'analyzer':
      return 'bg-indigo-500/20';
    case 'communicator':
      return 'bg-pink-500/20';
    default:
      return 'bg-gray-500/20';
  }
};

const getStatusColor = (status: string) => {
  switch (status) {
    case 'active':
      return 'bg-success';
    case 'learning':
      return 'bg-secondary';
    case 'idle':
      return 'bg-muted-foreground';
    case 'error':
      return 'bg-error';
    case 'initializing':
      return 'bg-primary animate-pulse';
    default:
      return 'bg-muted-foreground';
  }
};

const getStatusText = (status: string) => {
  switch (status) {
    case 'active':
      return 'Active';
    case 'learning':
      return 'Learning';
    case 'idle':
      return 'Idle';
    case 'error':
      return 'Error';
    case 'initializing':
      return 'Initializing';
    default:
      return status;
  }
};

export default function AgentCard({ 
  agent, 
  showDetails = false, 
  onView, 
  onStart, 
  onPause 
}: AgentCardProps) {
  const AgentIcon = getAgentIcon(agent.type);

  return (
    <Card className="agent-card">
      <CardContent className="p-4">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className={`p-3 rounded-lg ${getAgentIconColor(agent.type)}`}>
            <AgentIcon className="h-5 w-5 text-foreground" />
          </div>
          
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full pulse-dot ${getStatusColor(agent.status)}`} />
            
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="sm" className="opacity-0 group-hover:opacity-100 transition-opacity">
                  <MoreVertical className="w-4 h-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="glass-effect border-border">
                {agent.status === 'active' || agent.status === 'learning' ? (
                  <DropdownMenuItem onClick={() => onPause?.(agent)}>
                    <Pause className="w-4 h-4 mr-2" />
                    Pause Agent
                  </DropdownMenuItem>
                ) : (
                  <DropdownMenuItem onClick={() => onStart?.(agent)}>
                    <Play className="w-4 h-4 mr-2" />
                    Start Agent
                  </DropdownMenuItem>
                )}
                <DropdownMenuItem onClick={() => onView?.(agent)}>
                  <Eye className="w-4 h-4 mr-2" />
                  View Details
                </DropdownMenuItem>
                <DropdownMenuItem>
                  <Settings className="w-4 h-4 mr-2" />
                  Configure
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>

        {/* Agent Info */}
        <div className="mb-4">
          <h3 className="text-lg font-semibold mb-2 text-foreground">{agent.name}</h3>
          <p className="text-sm text-muted-foreground mb-3 line-clamp-2">
            {agent.description || 'No description provided'}
          </p>
        </div>

        {/* Status and Metrics */}
        <div className="space-y-3">
          <div className="flex items-center justify-between text-sm">
            <Badge variant="outline" className="capitalize">
              {getStatusText(agent.status)}
            </Badge>
            <span className="text-muted-foreground">
              CPU: {agent.cpuUsage || 0}%
            </span>
          </div>

          {showDetails && (
            <>
              {/* Performance Metrics */}
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="bg-black/20 p-2 rounded">
                  <div className="text-muted-foreground">Health</div>
                  <div className="font-semibold text-success">{agent.healthScore || 100}%</div>
                </div>
                <div className="bg-black/20 p-2 rounded">
                  <div className="text-muted-foreground">Memory</div>
                  <div className="font-semibold text-secondary">{agent.memoryUsage || 0}%</div>
                </div>
                <div className="bg-black/20 p-2 rounded">
                  <div className="text-muted-foreground">Tasks</div>
                  <div className="font-semibold text-accent">{agent.taskCount || 0}</div>
                </div>
                <div className="bg-black/20 p-2 rounded">
                  <div className="text-muted-foreground">Success</div>
                  <div className="font-semibold text-primary">{agent.successRate || 0}%</div>
                </div>
              </div>

              {/* Resource Usage */}
              <div className="space-y-2">
                <div>
                  <div className="flex justify-between text-xs text-muted-foreground mb-1">
                    <span>CPU Usage</span>
                    <span>{agent.cpuUsage || 0}%</span>
                  </div>
                  <Progress value={agent.cpuUsage || 0} className="h-1" />
                </div>
                <div>
                  <div className="flex justify-between text-xs text-muted-foreground mb-1">
                    <span>Memory Usage</span>
                    <span>{agent.memoryUsage || 0}%</span>
                  </div>
                  <Progress value={agent.memoryUsage || 0} className="h-1" />
                </div>
              </div>

              {/* Last Activity */}
              <div className="text-xs text-muted-foreground">
                Last activity: {agent.lastActivity 
                  ? new Date(agent.lastActivity).toLocaleTimeString()
                  : 'Never'
                }
              </div>
            </>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex items-center justify-between mt-4">
          <div className="flex space-x-2">
            {onView && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => onView(agent)}
                className="hover:bg-white/10"
              >
                <Eye className="h-4 w-4" />
              </Button>
            )}
            
            {agent.status === 'active' || agent.status === 'learning' ? (
              onPause && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onPause(agent)}
                  className="hover:bg-white/10"
                >
                  <Pause className="h-4 w-4" />
                </Button>
              )
            ) : (
              onStart && agent.status !== 'error' && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onStart(agent)}
                  className="hover:bg-white/10"
                >
                  <Play className="h-4 w-4" />
                </Button>
              )
            )}
          </div>

          <div className="text-right">
            <div className="text-sm font-medium text-foreground">{agent.healthScore || 100}%</div>
            <div className="text-xs text-muted-foreground">Performance</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
