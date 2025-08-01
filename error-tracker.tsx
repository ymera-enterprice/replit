import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  AlertTriangle, 
  XCircle, 
  AlertCircle, 
  Info,
  Check,
  Clock,
  ExternalLink
} from 'lucide-react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useToast } from '@/hooks/use-toast';

interface ErrorTrackerProps {
  errors: Array<{
    id: string;
    agent_id?: string;
    error_type: string;
    message: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
    resolved: boolean;
    stack_trace?: string;
    resolution_notes?: string;
    created_at: string;
    resolved_at?: string;
  }>;
}

const getSeverityIcon = (severity: string) => {
  switch (severity) {
    case 'critical':
      return <XCircle className="w-4 h-4 text-red-500" />;
    case 'high':
      return <AlertTriangle className="w-4 h-4 text-red-400" />;
    case 'medium':
      return <AlertCircle className="w-4 h-4 text-yellow-400" />;
    case 'low':
      return <Info className="w-4 h-4 text-blue-400" />;
    default:
      return <AlertCircle className="w-4 h-4 text-gray-400" />;
  }
};

const getSeverityColor = (severity: string) => {
  switch (severity) {
    case 'critical':
      return 'border-red-500/20 bg-red-500/10';
    case 'high':
      return 'border-red-500/20 bg-red-500/10';
    case 'medium':
      return 'border-yellow-500/20 bg-yellow-500/10';
    case 'low':
      return 'border-blue-500/20 bg-blue-500/10';
    default:
      return 'border-gray-500/20 bg-gray-500/10';
  }
};

const getSeverityBadgeColor = (severity: string) => {
  switch (severity) {
    case 'critical':
      return 'text-red-400';
    case 'high':
      return 'text-red-400';
    case 'medium':
      return 'text-yellow-400';
    case 'low':
      return 'text-blue-400';
    default:
      return 'text-gray-400';
  }
};

const formatTime = (timestamp: string) => {
  const now = new Date();
  const time = new Date(timestamp);
  const diffMs = now.getTime() - time.getTime();
  const diffMins = Math.floor(diffMs / (1000 * 60));
  
  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins} mins ago`;
  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours} hours ago`;
  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays} days ago`;
};

export default function ErrorTracker({ errors }: ErrorTrackerProps) {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const resolveErrorMutation = useMutation({
    mutationFn: async (errorId: string) => {
      const response = await fetch(`/api/monitoring/error-logs/${errorId}/resolve`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          resolutionNotes: 'Manually resolved by user'
        }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to resolve error');
      }
      
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/monitoring/error-logs'] });
      toast({
        title: 'Error Resolved',
        description: 'The error has been marked as resolved.',
      });
    },
    onError: () => {
      toast({
        title: 'Failed to Resolve Error',
        description: 'There was an error resolving the error log.',
        variant: 'destructive',
      });
    },
  });

  if (errors.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        <Check className="w-8 h-8 mx-auto mb-2 text-green-400" />
        <div className="text-green-400 font-medium">No Active Errors</div>
        <div className="text-sm">All systems operating normally</div>
      </div>
    );
  }

  const unresolvedErrors = errors.filter(error => !error.resolved);
  const criticalCount = unresolvedErrors.filter(e => e.severity === 'critical').length;
  const highCount = unresolvedErrors.filter(e => e.severity === 'high').length;

  return (
    <div className="space-y-4">
      {/* Error Summary */}
      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-1">
            <span className="text-red-400 font-medium">{criticalCount}</span>
            <span className="text-muted-foreground">critical</span>
          </div>
          <div className="flex items-center space-x-1">
            <span className="text-yellow-400 font-medium">{highCount}</span>
            <span className="text-muted-foreground">high</span>
          </div>
          <div className="flex items-center space-x-1">
            <span className="text-muted-foreground font-medium">{unresolvedErrors.length}</span>
            <span className="text-muted-foreground">total active</span>
          </div>
        </div>
        
        <Button variant="ghost" size="sm" className="text-primary hover:text-primary/80">
          View All Logs
          <ExternalLink className="w-3 h-3 ml-1" />
        </Button>
      </div>

      {/* Error List */}
      <ScrollArea className="max-h-64">
        <div className="space-y-3">
          {unresolvedErrors.slice(0, 5).map((error) => (
            <div 
              key={error.id} 
              className={`border p-4 rounded-lg ${getSeverityColor(error.severity)}`}
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center space-x-2">
                  {getSeverityIcon(error.severity)}
                  <div className="font-medium text-foreground">
                    {error.error_type}
                  </div>
                  <Badge variant="outline" className={getSeverityBadgeColor(error.severity)}>
                    {error.severity.toUpperCase()}
                  </Badge>
                </div>
                
                <div className="flex items-center space-x-2">
                  <div className="text-xs text-muted-foreground flex items-center">
                    <Clock className="w-3 h-3 mr-1" />
                    {formatTime(error.created_at)}
                  </div>
                  
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => resolveErrorMutation.mutate(error.id)}
                    disabled={resolveErrorMutation.isPending}
                    className="text-xs"
                  >
                    {resolveErrorMutation.isPending ? 'Resolving...' : 'Resolve'}
                  </Button>
                </div>
              </div>
              
              <div className="text-sm text-muted-foreground mb-2">
                {error.message}
              </div>
              
              {error.agent_id && (
                <div className="text-xs text-muted-foreground">
                  Agent ID: {error.agent_id}
                </div>
              )}
              
              {error.stack_trace && (
                <details className="mt-2">
                  <summary className="text-xs text-muted-foreground cursor-pointer hover:text-foreground">
                    Stack Trace
                  </summary>
                  <pre className="text-xs text-muted-foreground mt-2 p-2 bg-black/30 rounded overflow-x-auto">
                    {error.stack_trace}
                  </pre>
                </details>
              )}
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}
