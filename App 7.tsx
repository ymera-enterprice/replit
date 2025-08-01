import { Route, Switch } from 'wouter';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from '@/components/ui/toaster';
import Header from '@/components/layout/header';
import Dashboard from '@/pages/dashboard';
import Agents from '@/pages/agents';
import Learning from '@/pages/learning';
import Monitoring from '@/pages/monitoring';
import NotFound from '@/pages/not-found';
import { useRealTimeData } from '@/hooks/use-real-time-data';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30000, // 30 seconds
      refetchInterval: 60000, // 1 minute
    },
  },
});

function AppContent() {
  // Initialize real-time data connection
  useRealTimeData();

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Header />
      <main>
        <Switch>
          <Route path="/" component={Dashboard} />
          <Route path="/dashboard" component={Dashboard} />
          <Route path="/agents" component={Agents} />
          <Route path="/learning" component={Learning} />
          <Route path="/monitoring" component={Monitoring} />
          <Route component={NotFound} />
        </Switch>
      </main>
      <Toaster />
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppContent />
    </QueryClientProvider>
  );
}

export default App;
