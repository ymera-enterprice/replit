import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface StatusCardProps {
  title: string;
  icon: string;
  status: 'active' | 'inactive' | 'warning' | 'error';
  metrics: Array<{
    label: string;
    value: string | number;
  }>;
  description?: string;
  animated?: boolean;
}

export default function StatusCard({
  title,
  icon,
  status,
  metrics,
  description,
  animated = false
}: StatusCardProps) {
  const statusConfig = {
    active: {
      badge: 'ACTIVE',
      color: 'bg-green-500/20 text-green-400',
      badgeVariant: 'secondary' as const,
      iconColor: 'text-ymera-blue'
    },
    inactive: {
      badge: 'INACTIVE',
      color: 'bg-gray-500/20 text-gray-400',
      badgeVariant: 'secondary' as const,
      iconColor: 'text-gray-400'
    },
    warning: {
      badge: 'WARNING',
      color: 'bg-yellow-500/20 text-yellow-400',
      badgeVariant: 'secondary' as const,
      iconColor: 'text-ymera-gold'
    },
    error: {
      badge: 'ERROR',
      color: 'bg-red-500/20 text-red-400',
      badgeVariant: 'destructive' as const,
      iconColor: 'text-ymera-red'
    }
  };

  const config = statusConfig[status];

  return (
    <Card className={`bg-dark-secondary border-dark-tertiary ${animated ? 'animate-fade-in hover-lift' : 'hover-lift'}`}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg font-semibold flex items-center">
            <i className={`${icon} ${config.iconColor} mr-3`}></i>
            {title}
          </CardTitle>
          <Badge 
            variant={config.badgeVariant}
            className={`px-3 py-1 ${config.color} rounded-full text-xs font-medium`}
          >
            {config.badge}
          </Badge>
        </div>
      </CardHeader>
      
      <CardContent>
        <div className="space-y-4">
          {metrics.map((metric, index) => (
            <div key={index} className="flex justify-between items-center">
              <span className="text-sm text-gray-400">{metric.label}</span>
              <span className="text-white font-medium">{metric.value}</span>
            </div>
          ))}
        </div>
        
        {description && (
          <div className="mt-6 p-3 bg-dark-tertiary rounded-lg">
            <div className="text-xs text-gray-400 mb-2">Recent Activity</div>
            <div className="text-sm">{description}</div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
