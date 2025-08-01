import { Brain, TrendingUp, Zap, Target, Clock, Award } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';

interface LearningEvent {
  id: string;
  description: string;
  timestamp: string;
  type: 'success' | 'warning' | 'info';
}

interface LearningStatusProps {
  activeSessions?: number;
  totalSessions?: number;
  consolidationProgress?: number;
  patternDiscovery?: number;
  recentEvents?: LearningEvent[];
}

export default function LearningStatus({
  activeSessions = 5,
  totalSessions = 7,
  consolidationProgress = 89,
  patternDiscovery = 67,
  recentEvents = []
}: LearningStatusProps) {
  const defaultEvents: LearningEvent[] = [
    {
      id: '1',
      description: 'New pattern discovered in customer behavior data',
      timestamp: '2 minutes ago',
      type: 'success'
    },
    {
      id: '2',
      description: 'Knowledge transfer completed between Alpha and Beta agents',
      timestamp: '5 minutes ago',
      type: 'warning'
    },
    {
      id: '3',
      description: 'Memory consolidation session started',
      timestamp: '12 minutes ago',
      type: 'info'
    }
  ];

  const events = recentEvents.length > 0 ? recentEvents : defaultEvents;

  const getEventIcon = (type: string) => {
    switch (type) {
      case 'success':
        return <div className="w-2 h-2 rounded-full bg-success mt-2 animate-pulse" />;
      case 'warning':
        return <div className="w-2 h-2 rounded-full bg-warning mt-2" />;
      case 'info':
        return <div className="w-2 h-2 rounded-full bg-primary mt-2" />;
      default:
        return <div className="w-2 h-2 rounded-full bg-dark-500 mt-2" />;
    }
  };

  return (
    <Card className="glass-card">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Brain className="h-5 w-5 text-primary" />
          <span>Learning Status</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Learning Progress */}
        <div className="space-y-4">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-dark-300 flex items-center space-x-2">
                <TrendingUp className="h-4 w-4" />
                <span>Active Learning Sessions</span>
              </span>
              <span className="text-sm font-medium text-primary">
                {activeSessions}/{totalSessions}
              </span>
            </div>
            <Progress 
              value={(activeSessions / totalSessions) * 100} 
              className="h-2 progress-gradient-primary" 
            />
          </div>
          
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-dark-300 flex items-center space-x-2">
                <Zap className="h-4 w-4" />
                <span>Knowledge Consolidation</span>
              </span>
              <span className="text-sm font-medium text-secondary">
                {consolidationProgress}%
              </span>
            </div>
            <Progress 
              value={consolidationProgress} 
              className="h-2 progress-gradient-secondary" 
            />
          </div>
          
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-dark-300 flex items-center space-x-2">
                <Target className="h-4 w-4" />
                <span>Pattern Discovery</span>
              </span>
              <span className="text-sm font-medium text-success">
                {patternDiscovery}%
              </span>
            </div>
            <Progress 
              value={patternDiscovery} 
              className="h-2 progress-gradient-success" 
            />
          </div>
        </div>
        
        {/* Recent Learning Events */}
        <div>
          <h4 className="text-sm font-medium text-dark-200 mb-4 flex items-center space-x-2">
            <Clock className="h-4 w-4" />
            <span>Recent Learning Events</span>
          </h4>
          <div className="space-y-3">
            {events.map((event) => (
              <div key={event.id} className="flex items-start space-x-3">
                {getEventIcon(event.type)}
                <div className="flex-1">
                  <p className="text-sm text-dark-200">{event.description}</p>
                  <p className="text-xs text-dark-400">{event.timestamp}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
        
        {/* Quick Stats */}
        <div className="grid grid-cols-2 gap-4">
          <div className="text-center p-3 bg-dark-700/50 rounded-lg">
            <div className="text-lg font-bold text-primary">245</div>
            <div className="text-xs text-dark-400">Sessions Today</div>
          </div>
          <div className="text-center p-3 bg-dark-700/50 rounded-lg">
            <div className="text-lg font-bold text-secondary">92%</div>
            <div className="text-xs text-dark-400">Avg Efficiency</div>
          </div>
        </div>
        
        {/* Learning Analytics Button */}
        <Button className="w-full btn-ghost-primary">
          <Award className="h-4 w-4 mr-2" />
          View Learning Analytics
        </Button>
      </CardContent>
    </Card>
  );
}
