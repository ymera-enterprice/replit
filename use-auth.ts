import { useState, useEffect, useCallback } from 'react';
import { User } from '@shared/schema';
import { authService } from '@/lib/auth';
import { AuthContextType } from '@/types';

export function useAuth(): AuthContextType {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const initAuth = async () => {
      try {
        console.log('Initializing authentication...');
        const currentUser = await authService.getCurrentUser();
        console.log('Current user:', currentUser);
        setUser(currentUser);
      } catch (error) {
        console.error('Auth initialization error:', error);
      } finally {
        setIsLoading(false);
      }
    };

    initAuth();
  }, []);

  const login = useCallback(async (username: string, password: string, rememberMe = false): Promise<boolean> => {
    setIsLoading(true);
    try {
      console.log('Attempting login for:', username);
      const result = await authService.login({ username, password, rememberMe });
      console.log('Login result:', result);
      
      if (result.success && result.user) {
        setUser(result.user);
        console.log('User authenticated successfully:', result.user);
        // Force a page refresh to ensure the app re-renders with authenticated state
        setTimeout(() => window.location.reload(), 100);
        return true;
      }
      
      return false;
    } catch (error) {
      console.error('Login error:', error);
      return false;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const logout = useCallback(async () => {
    setIsLoading(true);
    try {
      await authService.logout();
      setUser(null);
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    user,
    login,
    logout,
    isLoading,
    isAuthenticated: !!user,
  };
}
