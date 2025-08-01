import YmeraLogo from './YmeraLogo';

export default function LoadingScreen() {
  return (
    <div className="fixed inset-0 bg-dark-primary z-50 flex items-center justify-center">
      <div className="text-center">
        <div className="mb-8">
          <YmeraLogo size="large" animated />
        </div>
        <div className="text-ymera-gold/60 text-sm mb-4">
          Enterprise System Integration
        </div>
        <div className="w-64 bg-dark-secondary rounded-full h-1">
          <div className="bg-gradient-to-r from-ymera-gold to-ymera-blue h-1 rounded-full animate-loading-progress"></div>
        </div>
      </div>
    </div>
  );
}
