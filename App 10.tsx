import { Switch, Route } from "wouter";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { useAuth } from "@/hooks/useAuth";

// Pages
import Landing from "@/pages/landing";
import Dashboard from "@/pages/dashboard";
import Agents from "@/pages/agents";
import Collaboration from "@/pages/collaboration";
import Files from "@/pages/files";
import Projects from "@/pages/projects";
import Analytics from "@/pages/analytics";
import Security from "@/pages/security";
import NotFound from "@/pages/not-found";

// Layout components
import Sidebar from "@/components/layout/sidebar";
import TopBar from "@/components/layout/topbar";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function AuthenticatedLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <TopBar />
        <main className="flex-1 overflow-hidden">
          {children}
        </main>
      </div>
    </div>
  );
}

function Router() {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="loading-spinner w-8 h-8 border-4 border-primary/20 border-t-primary rounded-full"></div>
      </div>
    );
  }

  return (
    <Switch>
      {!isAuthenticated ? (
        <Route path="/" component={Landing} />
      ) : (
        <>
          <Route path="/">
            <AuthenticatedLayout>
              <Dashboard />
            </AuthenticatedLayout>
          </Route>
          
          <Route path="/agents">
            <AuthenticatedLayout>
              <Agents />
            </AuthenticatedLayout>
          </Route>
          
          <Route path="/collaboration">
            <AuthenticatedLayout>
              <Collaboration />
            </AuthenticatedLayout>
          </Route>
          
          <Route path="/files">
            <AuthenticatedLayout>
              <Files />
            </AuthenticatedLayout>
          </Route>
          
          <Route path="/projects">
            <AuthenticatedLayout>
              <Projects />
            </AuthenticatedLayout>
          </Route>
          
          <Route path="/analytics">
            <AuthenticatedLayout>
              <Analytics />
            </AuthenticatedLayout>
          </Route>
          
          <Route path="/security">
            <AuthenticatedLayout>
              <Security />
            </AuthenticatedLayout>
          </Route>
        </>
      )}
      
      <Route component={NotFound} />
    </Switch>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router />
      <Toaster />
    </QueryClientProvider>
  );
}
