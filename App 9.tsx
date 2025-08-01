import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Router, Route, Switch } from 'wouter';
import { Toaster } from '@/components/ui/toaster';
import Header from '@/components/layout/header';
import Sidebar from '@/components/layout/sidebar';
import Dashboard from '@/pages/dashboard';
import Agents from '@/pages/agents';
import Projects from '@/pages/projects';
import Knowledge from '@/pages/knowledge';
import Collaboration from '@/pages/collaboration';
import Monitoring from '@/pages/monitoring';
import NotFound from '@/pages/not-found';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="h-screen bg-background text-foreground overflow-hidden">
          <Header />
          <div className="flex h-[calc(100vh-80px)]">
            <Sidebar />
            <main className="flex-1 overflow-auto">
              <Switch>
                <Route path="/" component={Dashboard} />
                <Route path="/agents" component={Agents} />
                <Route path="/projects" component={Projects} />
                <Route path="/knowledge" component={Knowledge} />
                <Route path="/collaboration" component={Collaboration} />
                <Route path="/monitoring" component={Monitoring} />
                <Route path="/files">
                  <div className="p-6">
                    <h1 className="text-3xl font-bold mb-4">File Management</h1>
                    <p className="text-muted-foreground">File management functionality coming soon...</p>
                  </div>
                </Route>
                <Route path="/security">
                  <div className="p-6">
                    <h1 className="text-3xl font-bold mb-4">Security Center</h1>
                    <p className="text-muted-foreground">Security management functionality coming soon...</p>
                  </div>
                </Route>
                <Route path="/settings">
                  <div className="p-6">
                    <h1 className="text-3xl font-bold mb-4">System Settings</h1>
                    <p className="text-muted-foreground">Settings configuration coming soon...</p>
                  </div>
                </Route>
                <Route component={NotFound} />
              </Switch>
            </main>
          </div>
          <Toaster />
        </div>
      </Router>
    </QueryClientProvider>
  );
}

export default App;
