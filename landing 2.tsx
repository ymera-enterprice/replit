import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

export default function Landing() {
  const handleLogin = () => {
    window.location.href = "/api/login";
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <Card className="glass-card max-w-2xl w-full">
        <CardContent className="p-12 text-center">
          {/* Logo */}
          <div className="mb-8">
            <div className="font-luxury text-6xl gradient-text mb-2">Ymera</div>
            <div className="text-lg text-muted-foreground">Enterprise Multi-Agent Platform</div>
          </div>
          
          {/* Description */}
          <div className="mb-8 space-y-4">
            <h1 className="text-3xl font-bold text-foreground">
              Welcome to the Future of Enterprise Automation
            </h1>
            <p className="text-lg text-muted-foreground leading-relaxed">
              Experience next-generation multi-agent learning systems with real-time collaboration, 
              AI-powered insights, and production-ready infrastructure designed for enterprise-scale operations.
            </p>
          </div>
          
          {/* Features */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div className="p-4 glass-effect rounded-lg">
              <div className="w-12 h-12 bg-primary/20 rounded-lg flex items-center justify-center mx-auto mb-3">
                <i className="fas fa-robot text-primary text-xl"></i>
              </div>
              <h3 className="font-semibold text-foreground mb-2">Multi-Agent Learning</h3>
              <p className="text-sm text-muted-foreground">
                Advanced AI agents that learn, adapt, and collaborate in real-time
              </p>
            </div>
            
            <div className="p-4 glass-effect rounded-lg">
              <div className="w-12 h-12 bg-secondary/20 rounded-lg flex items-center justify-center mx-auto mb-3">
                <i className="fas fa-users text-secondary text-xl"></i>
              </div>
              <h3 className="font-semibold text-foreground mb-2">Real-time Collaboration</h3>
              <p className="text-sm text-muted-foreground">
                Seamless team collaboration with live code editing and communication
              </p>
            </div>
            
            <div className="p-4 glass-effect rounded-lg">
              <div className="w-12 h-12 bg-accent/20 rounded-lg flex items-center justify-center mx-auto mb-3">
                <i className="fas fa-shield-alt text-accent text-xl"></i>
              </div>
              <h3 className="font-semibold text-foreground mb-2">Enterprise Security</h3>
              <p className="text-sm text-muted-foreground">
                Production-ready security with encryption and comprehensive audit logging
              </p>
            </div>
          </div>
          
          {/* Call to Action */}
          <div className="space-y-4">
            <Button 
              onClick={handleLogin}
              className="btn-gradient px-8 py-3 text-lg font-semibold"
              size="lg"
            >
              <i className="fas fa-sign-in-alt mr-2"></i>
              Sign In to Continue
            </Button>
            
            <p className="text-sm text-muted-foreground">
              Secure authentication powered by enterprise-grade infrastructure
            </p>
          </div>
          
          {/* Status Indicator */}
          <div className="mt-8 flex items-center justify-center space-x-2 text-sm">
            <div className="w-2 h-2 bg-success rounded-full pulse-dot status-indicator"></div>
            <span className="text-success">All systems operational</span>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
