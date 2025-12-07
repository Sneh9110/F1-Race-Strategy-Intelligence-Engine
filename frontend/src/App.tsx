import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';
import { useEffect } from 'react';

// Layout
import DashboardLayout from '@/components/layout/DashboardLayout';
import ProtectedRoute from '@/components/common/ProtectedRoute';

// Pages
import LoginPage from '@/pages/LoginPage';
import LiveStrategyConsole from '@/pages/LiveStrategyConsole';
import LapByLapMonitor from '@/pages/LapByLapMonitor';
import StrategyTreeVisualizer from '@/pages/StrategyTreeVisualizer';
import CompetitorComparison from '@/pages/CompetitorComparison';
import WeatherTrackEvolution from '@/pages/WeatherTrackEvolution';
import NotFoundPage from '@/pages/NotFoundPage';

// Store
import { useAuthStore } from '@/stores/authStore';

// Create QueryClient instance
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5000, // 5s for race state
      gcTime: 300000, // 5min cache (formerly cacheTime)
      refetchOnWindowFocus: true,
      retry: 2,
    },
  },
});

function App() {
  const { checkAuth } = useAuthStore();

  // Check auth status on app load
  useEffect(() => {
    checkAuth();
  }, [checkAuth]);

  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <DashboardLayout />
              </ProtectedRoute>
            }
          >
            <Route index element={<Navigate to="/strategy" replace />} />
            <Route path="strategy" element={<LiveStrategyConsole />} />
            <Route path="lap-monitor" element={<LapByLapMonitor />} />
            <Route path="strategy-tree" element={<StrategyTreeVisualizer />} />
            <Route path="competitors" element={<CompetitorComparison />} />
            <Route path="weather" element={<WeatherTrackEvolution />} />
          </Route>

          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </Router>

      {/* Toast notifications */}
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#1F1F2E',
            color: '#F8F9FA',
            border: '1px solid #4A4A5E',
          },
          success: {
            iconTheme: {
              primary: '#10B981',
              secondary: '#F8F9FA',
            },
          },
          error: {
            iconTheme: {
              primary: '#EF4444',
              secondary: '#F8F9FA',
            },
          },
        }}
      />
    </QueryClientProvider>
  );
}

export default App;
