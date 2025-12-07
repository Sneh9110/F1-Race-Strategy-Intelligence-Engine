import { useState } from 'react';
import { useRaceStore } from '@/stores/raceStore';
import { AlertCircle, TrendingUp, Clock, Gauge } from 'lucide-react';
import LoadingSpinner from '@/components/common/LoadingSpinner';

export default function LiveStrategyConsole() {
  const { raceState, selectedDriverNumber, setSelectedDriver } = useRaceStore();
  const [driverNumber, setDriverNumber] = useState(selectedDriverNumber || 44);

  const selectedDriver = raceState?.drivers.find((d) => d.driver_number === driverNumber);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Live Strategy Console</h1>
        {!raceState && (
          <div className="text-yellow-500 flex items-center space-x-2">
            <AlertCircle size={20} />
            <span>No active race session</span>
          </div>
        )}
      </div>

      {!raceState ? (
        <div className="card flex flex-col items-center justify-center py-20">
          <LoadingSpinner size="lg" />
          <p className="text-gray-400 mt-4">Waiting for race data...</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column: Driver Selector & Current State */}
          <div className="space-y-6">
            {/* Driver Selector */}
            <div className="card">
              <div className="card-header">
                <span>Select Driver</span>
              </div>
              <select
                value={driverNumber}
                onChange={(e) => {
                  const num = Number(e.target.value);
                  setDriverNumber(num);
                  setSelectedDriver(num);
                }}
                className="select-field w-full"
              >
                {raceState.drivers.map((driver) => (
                  <option key={driver.driver_number} value={driver.driver_number}>
                    #{driver.driver_number} {driver.driver_name} - P{driver.current_position}
                  </option>
                ))}
              </select>
            </div>

            {/* Current State */}
            {selectedDriver && (
              <div className="card">
                <div className="card-header">
                  <span>Current State</span>
                </div>
                <div className="space-y-3">
                  <div className="stat-card">
                    <div className="stat-label">Position</div>
                    <div className="stat-value">P{selectedDriver.current_position}</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Tire Compound</div>
                    <div className="stat-value">{selectedDriver.tire_compound}</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Tire Age</div>
                    <div className="stat-value">{selectedDriver.tire_age} laps</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Pit Stops</div>
                    <div className="stat-value">{selectedDriver.pit_stops}</div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Center Column: Recommendations */}
          <div className="space-y-6">
            <div className="card">
              <div className="card-header">
                <span>Strategy Recommendation</span>
                <div className="traffic-light-green"></div>
              </div>
              <div className="space-y-4">
                <div className="bg-racing-gray rounded-lg p-4">
                  <div className="text-xl font-semibold mb-2">STAY OUT</div>
                  <div className="text-sm text-gray-400">
                    Continue current stint. Tire degradation within acceptable limits.
                  </div>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Confidence:</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-32 bg-racing-gray rounded-full h-2">
                      <div className="bg-green-500 h-2 rounded-full" style={{ width: '85%' }}></div>
                    </div>
                    <span className="font-semibold">85%</span>
                  </div>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Risk Level:</span>
                  <span className="text-green-500 font-semibold">LOW</span>
                </div>
              </div>
            </div>

            {/* Alternatives */}
            <div className="card">
              <div className="card-header">
                <span>Alternative Options</span>
              </div>
              <div className="space-y-2">
                <div className="bg-racing-gray rounded-lg p-3 hover:bg-racing-lightgray cursor-pointer transition-colors">
                  <div className="font-semibold">PIT for HARD</div>
                  <div className="text-xs text-gray-400 mt-1">Expected time loss: +22.5s</div>
                </div>
                <div className="bg-racing-gray rounded-lg p-3 hover:bg-racing-lightgray cursor-pointer transition-colors">
                  <div className="font-semibold">EXTEND to Lap 45</div>
                  <div className="text-xs text-gray-400 mt-1">Risk: Tire cliff after Lap 42</div>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: SC Probability & Pit Windows */}
          <div className="space-y-6">
            {/* Safety Car Probability */}
            <div className="card">
              <div className="card-header">
                <span>Safety Car Probability</span>
              </div>
              <div className="flex flex-col items-center py-6">
                <div className="relative w-32 h-32">
                  <svg className="w-full h-full" viewBox="0 0 100 100">
                    <circle
                      cx="50"
                      cy="50"
                      r="45"
                      fill="none"
                      stroke="#2D2D3F"
                      strokeWidth="10"
                    />
                    <circle
                      cx="50"
                      cy="50"
                      r="45"
                      fill="none"
                      stroke="#10B981"
                      strokeWidth="10"
                      strokeDasharray="282.7"
                      strokeDashoffset="226.2"
                      strokeLinecap="round"
                      transform="rotate(-90 50 50)"
                    />
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-3xl font-bold">20%</span>
                  </div>
                </div>
                <div className="mt-4 text-center">
                  <div className="text-green-500 font-semibold">LOW RISK</div>
                  <div className="text-xs text-gray-400 mt-1">Based on circuit history</div>
                </div>
              </div>
            </div>

            {/* Live Metrics */}
            <div className="card">
              <div className="card-header">
                <span>Live Metrics</span>
              </div>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Clock size={16} className="text-gray-400" />
                    <span className="text-sm text-gray-400">Gap to Leader</span>
                  </div>
                  <span className="font-semibold">
                    {selectedDriver?.gap_to_leader.toFixed(1)}s
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <TrendingUp size={16} className="text-gray-400" />
                    <span className="text-sm text-gray-400">Gap to Next</span>
                  </div>
                  <span className="font-semibold">
                    {selectedDriver?.gap_to_next.toFixed(1)}s
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Gauge size={16} className="text-gray-400" />
                    <span className="text-sm text-gray-400">Last Lap</span>
                  </div>
                  <span className="font-semibold">
                    {selectedDriver?.last_lap_time.toFixed(3)}s
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
