import { useEffect, useState } from "react";

interface Agent {
  id: string;
  name: string;
  status: string;
  healthScore: number;
  cpuUsage: number;
  memoryUsage: number;
}

interface AgentVisualizationProps {
  agents?: Agent[];
}

export default function AgentVisualization({ agents = [] }: AgentVisualizationProps) {
  const [learningMetrics, setLearningMetrics] = useState({
    accuracy: 94.2,
    throughput: 2.1,
    efficiency: 89.5
  });

  useEffect(() => {
    // Simulate real-time learning metrics updates
    const interval = setInterval(() => {
      setLearningMetrics(prev => ({
        accuracy: Math.max(90, Math.min(99, prev.accuracy + (Math.random() - 0.5) * 0.8)),
        throughput: Math.max(1.5, Math.min(3.0, prev.throughput + (Math.random() - 0.5) * 0.3)),
        efficiency: Math.max(85, Math.min(95, prev.efficiency + (Math.random() - 0.5) * 1.0))
      }));
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-4">
      {/* Agent Network Visualization */}
      <div className="relative h-64 bg-black/20 rounded-lg p-4 overflow-hidden">
        <svg className="w-full h-full">
          {/* Gradient Definition */}
          <defs>
            <linearGradient id="connectionGradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" style={{ stopColor: 'hsl(51, 100%, 50%)', stopOpacity: 0.6 }} />
              <stop offset="50%" style={{ stopColor: 'hsl(217, 91%, 60%)', stopOpacity: 0.6 }} />
              <stop offset="100%" style={{ stopColor: 'hsl(0, 78%, 30%)', stopOpacity: 0.6 }} />
            </linearGradient>
          </defs>
          
          {/* Agent Connections */}
          <line x1="60" y1="80" x2="140" y2="120" stroke="url(#connectionGradient)" strokeWidth="2" opacity="0.7"/>
          <line x1="140" y1="120" x2="220" y2="80" stroke="url(#connectionGradient)" strokeWidth="2" opacity="0.7"/>
          <line x1="60" y1="80" x2="220" y2="80" stroke="url(#connectionGradient)" strokeWidth="1" opacity="0.5"/>
          <line x1="140" y1="40" x2="140" y2="120" stroke="url(#connectionGradient)" strokeWidth="2" opacity="0.7"/>
          <line x1="140" y1="40" x2="60" y2="80" stroke="url(#connectionGradient)" strokeWidth="1" opacity="0.5"/>
          <line x1="140" y1="40" x2="220" y2="80" stroke="url(#connectionGradient)" strokeWidth="1" opacity="0.5"/>
          
          {/* Agent Nodes */}
          <circle cx="140" cy="40" r="12" fill="hsl(51, 100%, 50%)" opacity="0.8">
            <animate attributeName="r" values="12;14;12" dur="2s" repeatCount="indefinite"/>
          </circle>
          <circle cx="60" cy="80" r="10" fill="hsl(217, 91%, 60%)" opacity="0.8"/>
          <circle cx="220" cy="80" r="10" fill="hsl(0, 78%, 30%)" opacity="0.8"/>
          <circle cx="140" cy="120" r="10" fill="hsl(142, 76%, 36%)" opacity="0.8"/>
          
          {/* Agent Labels */}
          <text x="140" y="30" textAnchor="middle" fill="white" fontSize="10" fontFamily="Inter">Master Agent</text>
          <text x="60" y="100" textAnchor="middle" fill="white" fontSize="8" fontFamily="Inter">Learning</text>
          <text x="220" y="100" textAnchor="middle" fill="white" fontSize="8" fontFamily="Inter">Analysis</text>
          <text x="140" y="140" textAnchor="middle" fill="white" fontSize="8" fontFamily="Inter">Processing</text>
        </svg>
        
        {/* Agent Status Overlay */}
        <div className="absolute top-4 right-4 space-y-2">
          <div className="flex items-center space-x-2 text-xs">
            <div className="w-2 h-2 bg-primary rounded-full pulse-dot"></div>
            <span className="text-muted-foreground">Pattern Recognition: Active</span>
          </div>
          <div className="flex items-center space-x-2 text-xs">
            <div className="w-2 h-2 bg-secondary rounded-full"></div>
            <span className="text-muted-foreground">Knowledge Synthesis: 87%</span>
          </div>
          <div className="flex items-center space-x-2 text-xs">
            <div className="w-2 h-2 bg-success rounded-full"></div>
            <span className="text-muted-foreground">Task Distribution: Optimal</span>
          </div>
        </div>
      </div>
      
      {/* Learning Metrics */}
      <div className="grid grid-cols-3 gap-4">
        <div className="text-center">
          <div className="text-lg font-semibold text-primary">
            {learningMetrics.accuracy.toFixed(1)}%
          </div>
          <div className="text-xs text-muted-foreground">Accuracy</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-semibold text-secondary">
            {learningMetrics.throughput.toFixed(1)}k/s
          </div>
          <div className="text-xs text-muted-foreground">Throughput</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-semibold text-success">
            {learningMetrics.efficiency.toFixed(1)}%
          </div>
          <div className="text-xs text-muted-foreground">Efficiency</div>
        </div>
      </div>
    </div>
  );
}
