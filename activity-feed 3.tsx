import { formatTimeAgo } from "@/lib/utils";

interface Activity {
  id: string;
  type: string;
  description: string;
  timestamp: string;
  user: string;
  icon: string;
}

interface ActivityFeedProps {
  activities?: Activity[];
}

export default function ActivityFeed({ activities = [] }: ActivityFeedProps) {
  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'file_upload':
        return 'fas fa-upload text-success';
      case 'agent_update':
        return 'fas fa-brain text-secondary';
      case 'collaboration':
        return 'fas fa-users text-primary';
      case 'security':
        return 'fas fa-exclamation-triangle text-warning';
      case 'deployment':
        return 'fas fa-rocket text-accent';
      default:
        return 'fas fa-info-circle text-muted-foreground';
    }
  };

  if (activities.length === 0) {
    return (
      <div className="p-6 text-center">
        <div className="w-12 h-12 bg-muted/20 rounded-full flex items-center justify-center mx-auto mb-3">
          <i className="fas fa-activity text-muted-foreground"></i>
        </div>
        <p className="text-sm text-muted-foreground">No recent activity</p>
      </div>
    );
  }

  return (
    <div className="space-y-4 h-72 overflow-y-auto scroll-container p-4">
      {activities.map((activity) => (
        <div key={activity.id} className="flex items-start space-x-3 p-3 bg-black/20 rounded-lg">
          <div className="w-8 h-8 bg-white/10 rounded-full flex items-center justify-center flex-shrink-0">
            <i className={`${getActivityIcon(activity.type)} text-xs`}></i>
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-sm text-foreground">{activity.description}</div>
            <div className="text-xs text-muted-foreground">
              {formatTimeAgo(activity.timestamp)}
            </div>
            <div className="text-xs text-muted-foreground/80">by {activity.user}</div>
          </div>
        </div>
      ))}
    </div>
  );
}
