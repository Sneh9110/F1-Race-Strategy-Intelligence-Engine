import { Outlet, Link, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  Activity,
  GitBranch,
  Users,
  CloudRain,
  Menu,
  X,
  LogOut,
  User,
} from 'lucide-react';
import { useState } from 'react';
import { useAuthStore } from '@/stores/authStore';
import { useRaceStore } from '@/stores/raceStore';
import { formatDateTime } from '@/utils/formatters';

export default function DashboardLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const location = useLocation();
  const { user, logout } = useAuthStore();
  const { raceState, lastUpdate } = useRaceStore();

  const navigation = [
    { name: 'Live Strategy', href: '/strategy', icon: LayoutDashboard },
    { name: 'Lap Monitor', href: '/lap-monitor', icon: Activity },
    { name: 'Strategy Tree', href: '/strategy-tree', icon: GitBranch },
    { name: 'Competitors', href: '/competitors', icon: Users },
    { name: 'Weather & Track', href: '/weather', icon: CloudRain },
  ];

  const isActive = (path: string) => location.pathname === path;

  return (
    <div className="flex h-screen bg-racing-black text-white font-formula">
      {/* Sidebar */}
      <aside
        className={`${
          sidebarOpen ? 'w-64' : 'w-20'
        } bg-racing-darkgray border-r border-racing-gray transition-all duration-300 ease-in-out flex flex-col`}
      >
        {/* Logo */}
        <div className="h-16 flex items-center justify-between px-4 border-b border-racing-gray">
          {sidebarOpen ? (
            <h1 className="text-xl font-bold text-redbull">F1 Strategy</h1>
          ) : (
            <div className="text-2xl font-bold text-redbull">F1</div>
          )}
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 rounded-lg hover:bg-racing-gray transition-colors"
            aria-label="Toggle sidebar"
          >
            {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-2">
          {navigation.map((item) => {
            const Icon = item.icon;
            return (
              <Link
                key={item.name}
                to={item.href}
                className={`flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
                  isActive(item.href)
                    ? 'bg-racing-gray text-white border-l-4 border-redbull'
                    : 'text-gray-300 hover:bg-racing-gray hover:text-white'
                }`}
              >
                <Icon size={20} />
                {sidebarOpen && <span>{item.name}</span>}
              </Link>
            );
          })}
        </nav>

        {/* Quick Stats */}
        {sidebarOpen && raceState && (
          <div className="p-4 border-t border-racing-gray">
            <div className="text-xs text-gray-400 mb-2">Quick Stats</div>
            <div className="space-y-1 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Lap:</span>
                <span className="font-semibold">
                  {raceState.current_lap}/{raceState.total_laps}
                </span>
              </div>
              {raceState.safety_car_active && (
                <div className="flex items-center space-x-2 text-yellow-500">
                  <div className="w-2 h-2 rounded-full bg-yellow-500 animate-pulse"></div>
                  <span>Safety Car</span>
                </div>
              )}
              <div className="flex justify-between">
                <span className="text-gray-400">Weather:</span>
                <span className="font-semibold">{raceState.weather_condition}</span>
              </div>
            </div>
          </div>
        )}

        {/* User Menu */}
        <div className="p-4 border-t border-racing-gray">
          {sidebarOpen ? (
            <div className="space-y-2">
              <div className="flex items-center space-x-2 text-sm">
                <User size={16} className="text-gray-400" />
                <span className="text-gray-300">{user?.username || 'User'}</span>
              </div>
              <button
                onClick={logout}
                className="flex items-center space-x-2 text-sm text-gray-400 hover:text-white transition-colors w-full"
              >
                <LogOut size={16} />
                <span>Logout</span>
              </button>
            </div>
          ) : (
            <button
              onClick={logout}
              className="p-2 rounded-lg hover:bg-racing-gray transition-colors w-full flex justify-center"
              aria-label="Logout"
            >
              <LogOut size={20} className="text-gray-400" />
            </button>
          )}
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="h-16 bg-racing-darkgray border-b border-racing-gray flex items-center justify-between px-6">
          <div className="flex items-center space-x-4">
            {raceState && (
              <>
                <h2 className="text-xl font-semibold">{raceState.circuit_name}</h2>
                <span className="text-gray-400">•</span>
                <span className="text-gray-400">{raceState.session_type}</span>
              </>
            )}
          </div>
          <div className="flex items-center space-x-4">
            {/* Live Indicator */}
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
              <span className="text-sm text-gray-400">Live</span>
            </div>
            {/* Last Update */}
            {lastUpdate && (
              <div className="text-sm text-gray-400">
                Updated: {formatDateTime(lastUpdate, 'HH:mm:ss')}
              </div>
            )}
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 overflow-auto custom-scrollbar bg-racing-black p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
