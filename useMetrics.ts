import { useQuery } from '@tanstack/react-query';
import { useToast } from '@/hooks/use-toast';
import { isUnauthorizedError } from '@/lib/authUtils';

interface DashboardMetrics {
  totalUsers: number;
  activeUsers: number;
  totalProjects: number;
  activeProjects: number;
  totalFiles: number;
  totalFileSize: number;
  totalMessages: number;
  activeAgents: number;
  knowledgeNodes: number;
  activeConnections: number;
}

interface MetricsResponse {
  data: DashboardMetrics;
}

export function useMetrics() {
  const { toast } = useToast();

  return useQuery<MetricsResponse>({
    queryKey: ['/api/dashboard/metrics'],
    refetchInterval: 30000, // Refetch every 30 seconds for real-time updates
    staleTime: 15000, // Consider data stale after 15 seconds
    retry: (failureCount, error) => {
      if (isUnauthorizedError(error as Error)) {
        toast({
          title: "Unauthorized",
          description: "You are logged out. Logging in again...",
          variant: "destructive",
        });
        setTimeout(() => {
          window.location.href = "/api/login";
        }, 500);
        return false;
      }
      return failureCount < 3;
    },
    meta: {
      errorMessage: "Failed to fetch dashboard metrics",
    },
  });
}

export function useSystemMetrics(type?: string, hours: number = 24) {
  const { toast } = useToast();

  return useQuery({
    queryKey: ['/api/metrics', type, hours],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (type) params.append('type', type);
      params.append('hours', hours.toString());
      
      const response = await fetch(`/api/metrics?${params}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch metrics: ${response.statusText}`);
      }
      return response.json();
    },
    refetchInterval: 60000, // Refetch every minute
    staleTime: 30000, // Consider data stale after 30 seconds
    retry: (failureCount, error) => {
      if (isUnauthorizedError(error as Error)) {
        toast({
          title: "Unauthorized",
          description: "You are logged out. Logging in again...",
          variant: "destructive",
        });
        setTimeout(() => {
          window.location.href = "/api/login";
        }, 500);
        return false;
      }
      return failureCount < 3;
    },
  });
}

export function useRealtimeMetrics() {
  const { toast } = useToast();

  return useQuery({
    queryKey: ['/api/websocket/connections'],
    refetchInterval: 15000, // Refetch every 15 seconds for connection status
    staleTime: 10000, // Consider data stale after 10 seconds
    retry: (failureCount, error) => {
      if (isUnauthorizedError(error as Error)) {
        toast({
          title: "Unauthorized",
          description: "You are logged out. Logging in again...",
          variant: "destructive",
        });
        setTimeout(() => {
          window.location.href = "/api/login";
        }, 500);
        return false;
      }
      return failureCount < 3;
    },
  });
}

// Custom hook for creating new metrics
export function useCreateMetric() {
  const { toast } = useToast();
  
  return async (metricData: {
    metricType: string;
    value: number;
    unit?: string;
    metadata?: any;
  }) => {
    try {
      const response = await fetch('/api/metrics', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(metricData),
      });

      if (!response.ok) {
        throw new Error(`Failed to create metric: ${response.statusText}`);
      }

      const result = await response.json();
      return result;
    } catch (error) {
      if (isUnauthorizedError(error as Error)) {
        toast({
          title: "Unauthorized",
          description: "You are logged out. Logging in again...",
          variant: "destructive",
        });
        setTimeout(() => {
          window.location.href = "/api/login";
        }, 500);
        throw error;
      }
      
      toast({
        title: "Error",
        description: "Failed to create metric",
        variant: "destructive",
      });
      throw error;
    }
  };
}
