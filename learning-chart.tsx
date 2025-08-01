import { useEffect, useRef } from 'react';
import { useQuery } from '@tanstack/react-query';

export default function LearningChart() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  const { data: learningOverview } = useQuery({
    queryKey: ['/api/learning/overview'],
    refetchInterval: 30000,
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

    // Clear canvas
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.fillRect(0, 0, rect.width, rect.height);

    // Generate sample learning data points
    const data = [];
    const baseAccuracy = learningOverview?.data?.total_pattern_accuracy || 85;
    
    for (let i = 0; i < 24; i++) {
      const variation = (Math.sin(i * 0.5) + Math.random() * 0.5 - 0.25) * 5;
      data.push(Math.max(70, Math.min(100, baseAccuracy + variation)));
    }

    // Draw grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    
    // Horizontal grid lines
    for (let i = 0; i <= 4; i++) {
      const y = (rect.height * i) / 4;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(rect.width, y);
      ctx.stroke();
    }

    // Vertical grid lines
    for (let i = 0; i <= 6; i++) {
      const x = (rect.width * i) / 6;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, rect.height);
      ctx.stroke();
    }

    // Draw pattern accuracy line
    ctx.strokeStyle = '#FBBF24'; // ymera-gold
    ctx.lineWidth = 3;
    ctx.beginPath();

    data.forEach((value, index) => {
      const x = (rect.width * index) / (data.length - 1);
      const y = rect.height - ((value - 70) / 30) * rect.height;
      
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    ctx.stroke();

    // Draw knowledge growth line
    const growthData = data.map(val => val * 0.8 + Math.random() * 10);
    ctx.strokeStyle = '#2563EB'; // ymera-blue
    ctx.lineWidth = 2;
    ctx.beginPath();

    growthData.forEach((value, index) => {
      const x = (rect.width * index) / (growthData.length - 1);
      const y = rect.height - ((value - 70) / 30) * rect.height;
      
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    ctx.stroke();

    // Draw data points
    data.forEach((value, index) => {
      const x = (rect.width * index) / (data.length - 1);
      const y = rect.height - ((value - 70) / 30) * rect.height;
      
      ctx.fillStyle = '#FBBF24';
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
    });

    // Add gradient fill under the main line
    const gradient = ctx.createLinearGradient(0, 0, 0, rect.height);
    gradient.addColorStop(0, 'rgba(251, 191, 36, 0.3)');
    gradient.addColorStop(1, 'rgba(251, 191, 36, 0.05)');
    
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.moveTo(0, rect.height);
    
    data.forEach((value, index) => {
      const x = (rect.width * index) / (data.length - 1);
      const y = rect.height - ((value - 70) / 30) * rect.height;
      ctx.lineTo(x, y);
    });
    
    ctx.lineTo(rect.width, rect.height);
    ctx.closePath();
    ctx.fill();

  }, [learningOverview]);

  return (
    <div className="h-64 w-full relative">
      <canvas
        ref={canvasRef}
        className="w-full h-full rounded-lg"
        style={{ width: '100%', height: '100%' }}
      />
      
      {/* Legend */}
      <div className="absolute bottom-2 left-2 flex items-center space-x-4 text-xs">
        <div className="flex items-center space-x-1">
          <div className="w-3 h-0.5 bg-primary"></div>
          <span className="text-muted-foreground">Pattern Accuracy</span>
        </div>
        <div className="flex items-center space-x-1">
          <div className="w-3 h-0.5 bg-secondary"></div>
          <span className="text-muted-foreground">Knowledge Growth</span>
        </div>
      </div>
    </div>
  );
}
