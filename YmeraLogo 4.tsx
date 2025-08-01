import { useState } from "react";

interface YmeraLogoProps {
  size?: number;
  showText?: boolean;
  className?: string;
  onAnimationComplete?: () => void;
}

export function YmeraLogo({ 
  size = 40, 
  showText = true, 
  className = "",
  onAnimationComplete 
}: YmeraLogoProps) {
  const [isAnimating, setIsAnimating] = useState(false);

  const playAnimation = () => {
    setIsAnimating(true);
    setTimeout(() => {
      setIsAnimating(false);
      onAnimationComplete?.();
    }, 4000);
  };

  return (
    <div className={`flex items-center cursor-pointer ${className}`} onClick={playAnimation}>
      <div className="relative mr-3">
        <svg 
          width={size} 
          height={size} 
          viewBox="0 0 100 100" 
          className="neon-glow"
        >
          <defs>
            <linearGradient id="infinityGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" style={{ stopColor: '#FBBF24', stopOpacity: 1 }} />
              <stop offset="50%" style={{ stopColor: '#2563EB', stopOpacity: 1 }} />
              <stop offset="100%" style={{ stopColor: '#991B1B', stopOpacity: 1 }} />
            </linearGradient>
          </defs>
          <path 
            className={`infinity-path ${isAnimating ? 'animate-infinity-draw' : ''}`}
            d="M 20 50 C 20 30, 40 30, 50 50 C 50 70, 40 70, 50 50 C 50 30, 70 30, 80 50 C 80 70, 60 70, 50 50" 
            style={{
              strokeDasharray: 500,
              strokeDashoffset: isAnimating ? 0 : 500,
            }}
          />
        </svg>
      </div>
      {showText && (
        <div>
          <h1 className="font-great-vibes text-2xl gradient-text">YMERA</h1>
          <p className="text-xs text-white/70">by Mohamed Mansour</p>
        </div>
      )}
    </div>
  );
}
