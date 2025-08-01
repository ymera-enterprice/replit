import { Link, useLocation } from 'wouter';
import { 
  Home, 
  Users, 
  Brain, 
  Activity, 
  Settings, 
  Menu,
  X
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Sheet, SheetContent, SheetTrigger } from '@/components/ui/sheet';
import { useState } from 'react';

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: Home },
  { name: 'Agents', href: '/agents', icon: Users },
  { name: 'Learning', href: '/learning', icon: Brain },
  { name: 'Monitoring', href: '/monitoring', icon: Activity },
];

export default function Navigation() {
  const [location] = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const isActive = (href: string) => {
    return location === href || (href === '/dashboard' && location === '/');
  };

  const NavLink = ({ item, mobile = false }: { item: typeof navigation[0]; mobile?: boolean }) => {
    const Icon = item.icon;
    const active = isActive(item.href);
    
    return (
      <Link
        href={item.href}
        className={`flex items-center space-x-3 px-4 py-3 rounded-lg transition-all ${
          active
            ? 'bg-primary/20 text-primary border border-primary/30'
            : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
        } ${mobile ? 'w-full' : ''}`}
        onClick={() => mobile && setMobileMenuOpen(false)}
      >
        <Icon className="w-5 h-5" />
        <span className="font-medium">{item.name}</span>
      </Link>
    );
  };

  return (
    <>
      {/* Desktop Navigation */}
      <nav className="hidden md:flex items-center space-x-2">
        {navigation.map((item) => (
          <NavLink key={item.name} item={item} />
        ))}
      </nav>

      {/* Mobile Navigation */}
      <div className="md:hidden">
        <Sheet open={mobileMenuOpen} onOpenChange={setMobileMenuOpen}>
          <SheetTrigger asChild>
            <Button variant="ghost" size="sm" className="p-2">
              <Menu className="w-5 h-5" />
            </Button>
          </SheetTrigger>
          <SheetContent side="right" className="glass-card border-border">
            <div className="flex flex-col space-y-4 mt-8">
              <div className="flex items-center justify-between mb-6">
                <div className="font-great-vibes text-2xl ymera-text-gradient">
                  Ymera
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setMobileMenuOpen(false)}
                  className="p-2"
                >
                  <X className="w-5 h-5" />
                </Button>
              </div>
              
              {navigation.map((item) => (
                <NavLink key={item.name} item={item} mobile />
              ))}
              
              <div className="pt-6 border-t border-border">
                <NavLink 
                  item={{ name: 'Settings', href: '/settings', icon: Settings }} 
                  mobile 
                />
              </div>
            </div>
          </SheetContent>
        </Sheet>
      </div>
    </>
  );
}
