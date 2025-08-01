import { useEffect, useRef } from 'react';
import { useQuery } from '@tanstack/react-query';

interface Node {
  id: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  color: string;
  label: string;
}

interface Edge {
  source: string;
  target: string;
  strength: number;
}

export default function KnowledgeGraph() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const nodesRef = useRef<Node[]>([]);
  const edgesRef = useRef<Edge[]>([]);

  const { data: knowledgeNodes } = useQuery({
    queryKey: ['/api/knowledge/nodes'],
    refetchInterval: 60000,
  });

  const { data: knowledgeRelationships } = useQuery({
    queryKey: ['/api/knowledge/relationships'],
    refetchInterval: 60000,
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    // Initialize nodes with sample data (since we don't have real knowledge graph data)
    const colors = ['#FBBF24', '#2563EB', '#991B1B', '#10B981', '#8B5CF6', '#F59E0B'];
    const nodes: Node[] = [];
    const edges: Edge[] = [];

    // Create nodes
    for (let i = 0; i < 12; i++) {
      nodes.push({
        id: `node-${i}`,
        x: Math.random() * rect.width,
        y: Math.random() * rect.height,
        vx: (Math.random() - 0.5) * 2,
        vy: (Math.random() - 0.5) * 2,
        radius: Math.random() * 8 + 6,
        color: colors[Math.floor(Math.random() * colors.length)],
        label: `N${i + 1}`
      });
    }

    // Create edges
    for (let i = 0; i < nodes.length; i++) {
      const connectionCount = Math.floor(Math.random() * 3) + 1;
      for (let j = 0; j < connectionCount; j++) {
        const targetIndex = Math.floor(Math.random() * nodes.length);
        if (targetIndex !== i) {
          edges.push({
            source: nodes[i].id,
            target: nodes[targetIndex].id,
            strength: Math.random()
          });
        }
      }
    }

    nodesRef.current = nodes;
    edgesRef.current = edges;

    // Physics simulation
    const simulate = () => {
      const nodes = nodesRef.current;
      const edges = edgesRef.current;

      // Apply forces
      nodes.forEach(node => {
        // Gravity towards center
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;
        const dx = centerX - node.x;
        const dy = centerY - node.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance > 0) {
          node.vx += (dx / distance) * 0.01;
          node.vy += (dy / distance) * 0.01;
        }

        // Repulsion between nodes
        nodes.forEach(other => {
          if (other !== node) {
            const dx = other.x - node.x;
            const dy = other.y - node.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance > 0 && distance < 100) {
              const force = -50 / (distance * distance);
              node.vx += (dx / distance) * force;
              node.vy += (dy / distance) * force;
            }
          }
        });

        // Spring forces for connected nodes
        edges.forEach(edge => {
          if (edge.source === node.id) {
            const target = nodes.find(n => n.id === edge.target);
            if (target) {
              const dx = target.x - node.x;
              const dy = target.y - node.y;
              const distance = Math.sqrt(dx * dx + dy * dy);
              const idealDistance = 80;
              
              if (distance > 0) {
                const force = (distance - idealDistance) * 0.1 * edge.strength;
                node.vx += (dx / distance) * force;
                node.vy += (dy / distance) * force;
              }
            }
          }
        });

        // Apply velocity and damping
        node.vx *= 0.85;
        node.vy *= 0.85;
        node.x += node.vx;
        node.y += node.vy;

        // Boundary conditions
        if (node.x < node.radius) {
          node.x = node.radius;
          node.vx *= -0.5;
        }
        if (node.x > rect.width - node.radius) {
          node.x = rect.width - node.radius;
          node.vx *= -0.5;
        }
        if (node.y < node.radius) {
          node.y = node.radius;
          node.vy *= -0.5;
        }
        if (node.y > rect.height - node.radius) {
          node.y = rect.height - node.radius;
          node.vy *= -0.5;
        }
      });
    };

    // Render function
    const render = () => {
      // Clear canvas
      ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
      ctx.fillRect(0, 0, rect.width, rect.height);

      const nodes = nodesRef.current;
      const edges = edgesRef.current;

      // Draw edges
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
      ctx.lineWidth = 1;
      
      edges.forEach(edge => {
        const source = nodes.find(n => n.id === edge.source);
        const target = nodes.find(n => n.id === edge.target);
        
        if (source && target) {
          ctx.globalAlpha = edge.strength * 0.6 + 0.2;
          ctx.beginPath();
          ctx.moveTo(source.x, source.y);
          ctx.lineTo(target.x, target.y);
          ctx.stroke();
        }
      });

      ctx.globalAlpha = 1;

      // Draw nodes
      nodes.forEach(node => {
        // Node glow
        const gradient = ctx.createRadialGradient(
          node.x, node.y, 0,
          node.x, node.y, node.radius * 2
        );
        gradient.addColorStop(0, node.color + '80');
        gradient.addColorStop(1, node.color + '00');
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius * 2, 0, 2 * Math.PI);
        ctx.fill();

        // Node body
        ctx.fillStyle = node.color;
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, 2 * Math.PI);
        ctx.fill();

        // Node border
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Pulse effect for central nodes
        if (node.radius > 10) {
          const pulseRadius = node.radius + Math.sin(Date.now() * 0.005) * 3;
          ctx.strokeStyle = node.color + '40';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(node.x, node.y, pulseRadius, 0, 2 * Math.PI);
          ctx.stroke();
        }
      });
    };

    // Animation loop
    const animate = () => {
      simulate();
      render();
      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [knowledgeNodes, knowledgeRelationships]);

  return (
    <div className="h-64 w-full relative">
      <canvas
        ref={canvasRef}
        className="w-full h-full rounded-lg"
        style={{ width: '100%', height: '100%' }}
      />
      
      {/* Legend */}
      <div className="absolute bottom-2 right-2 text-xs text-muted-foreground">
        <div className="flex items-center space-x-1">
          <div className="w-2 h-2 bg-primary rounded-full animate-pulse"></div>
          <span>Knowledge Nodes</span>
        </div>
      </div>
    </div>
  );
}
