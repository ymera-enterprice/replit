import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  Activity, 
  AlertCircle, 
  CheckCircle, 
  Info, 
  Zap,
  Clock
} from 'lucide-react';

interface ActivityFeedProps {
  activities: Array<{
    id: string;
    agent_id?: string;
    agent_type?: string;
    message: string;
    level: 'info' | 'warning' | 'error' | 'success';
    created_at: string;
    details?: Record<string, any>;
  }>;
  limit?: number;
  showTimestamp?: boolean;
}

const getLevelIcon = (level: string) => {
  switch (level) {
    case 'success':
      return <CheckCircle className="w-4 h-4 text-green-400" />;
    case 'warning':
      return <AlertCircle className="w-4 h-4 text-yellow-400" />;
    case 'error':
      return <AlertCircle className="w-4 h-4 text-red-400" />;
    default:
      return <Info className="w-4 h-4 text-blue-400" />;
  }
};

const getLevelColor = (level: string) => {
  switch (level) {
    case 'success':
      return 'text-green-400';
    case 'warning':
      return 'text-yellow-400';
    case 'error':
      return 'text-red-400';
    default:
      return 'text-blue-400';
  }
};

const getAgentTypeColor = (type?: string) => {
  switch (type) {
    case 'editing':
      return 'text-blue-400';
    case 'enhancement':
      return 'text-purple-400';
    case 'monitoring':
      return 'text-green-400';
    case 'orchestration':
      return 'text-orange-400';
    case 'project':
      return 'text-indigo-400';
    case 'validation':
      return 'text-red-400';
    case 'examination':
      return 'text-yellow-400';
    case 'learning_engine':
      return 'text-pink-400';
    default:
      return 'text-gray-400';
  }
};

const formatTime = (timestamp: string) => {
  const now = new Date();
  const time = new Date(timestamp);
  const diffMs = now.getTime() - time.getTime();
  const diffMins = Math.floor(diffMs / (1000 * 60));
  
  if (diffMins < 1) return 'now';
  if (diffMins < 60) return `${diffMins}m`;
  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h`;
  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays}d`;
};

export default function ActivityFeed({ 
  activities, 
  limit = 10, 
  showTimestamp = false 
}: ActivityFeedProps) {
  const displayedActivities = activities.slice(0, limit);

  if (displayedActivities.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        <Activity className="w-8 h-8 mx-auto mb-2 opacity-50" />
        <div>No activity logs available</div>
      </div>
    );
  }

  return (
    <ScrollArea className="max-h-80">
      <div className="space-y-3 font-jetbrains text-sm">
        {displayedActivities.map((activity) => (
          <div key={activity.id} className="flex items-start space-x-3 bg-black/20 p-3 rounded-lg">
            <div className="flex-shrink-0 mt-0.5">
              {getLevelIcon(activity.level)}
            </div>
            
            <div className="flex-1 min-w-0">
              <div className="flex items-start justify-between">
                <div className={`font-medium ${getLevelColor(activity.level)}`}>
                  {showTimestamp && (
                    <span className="text-green-400 mr-2">
                      [{new Date(activity.created_at).toLocaleTimeString()}]
                    </span>
                  )}
                  {activity.agent_type && (
                    <span className={`${getAgentTypeColor(activity.agent_type)} mr-2`}>
                      {activity.agent_type.toUpperCase()}
                    </span>
                  )}
                  {activity.message}
                </div>
                
                <div className="flex items-center space-x-2 ml-2">
                  <div className="flex items-center text-xs text-muted-foreground">
                    <Clock className="w-3 h-3 mr-1" />
                    {formatTime(activity.created_at)}
                  </div>
                </div>
              </div>
              
              {activity.details && Object.keys(activity.details).length > 0 && (
                <div className="mt-2 text-xs text-muted-foreground">
                  {Object.entries(activity.details).map(([key, value]) => (
                    <Badge key={key} variant="outline" className="mr-1 text-xs">
                      {key}: {String(value)}
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </ScrollArea>
  );
}
