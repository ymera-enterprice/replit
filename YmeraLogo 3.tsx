interface YmeraLogoProps {
  size?: 'small' | 'medium' | 'large';
  animated?: boolean;
  showText?: boolean;
}

export default function YmeraLogo({ 
  size = 'medium', 
  animated = false, 
  showText = true 
}: YmeraLogoProps) {
  const sizeClasses = {
    small: 'w-8 h-8',
    medium: 'w-12 h-12',
    large: 'w-16 h-16'
  };

  const textSizeClasses = {
    small: 'text-lg',
    medium: 'text-2xl',
    large: 'text-4xl'
  };

  return (
    <div className={`flex items-center space-x-3 ${animated ? 'animate-float' : ''}`}>
      <div className="relative">
        <svg 
          className={sizeClasses[size]} 
          viewBox="0 0 64 64"
          fill="none"
        >
          <defs>
            <linearGradient id="logoGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="hsl(45, 93%, 58%)" />
              <stop offset="50%" stopColor="hsl(217, 91%, 60%)" />
              <stop offset="100%" stopColor="hsl(0, 84%, 35%)" />
            </linearGradient>
          </defs>
          <path 
            stroke="url(#logoGradient)" 
            strokeWidth="3" 
            fill="none" 
            d="M 8 32 C 8 16, 24 16, 32 32 C 32 48, 24 48, 32 32 C 32 16, 48 16, 56 32 C 56 48, 40 48, 32 32"
            className={animated ? 'animate-infinity-draw' : ''}
            strokeDasharray={animated ? '500' : undefined}
            strokeDashoffset={animated ? '1000' : undefined}
          />
        </svg>
      </div>
      {showText && (
        <div>
          <h2 className={`font-great-vibes ${textSizeClasses[size]} gradient-text`}>
            Ymera
          </h2>
          {size === 'large' && (
            <p className="text-xs text-gray-400 -mt-1">Enterprise AI Platform</p>
          )}
        </div>
      )}
    </div>
  );
}
