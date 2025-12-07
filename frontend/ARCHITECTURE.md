# F1 Strategy Dashboard Architecture

## System Overview

The F1 Strategy Dashboard is a React-based single-page application (SPA) that provides real-time race strategy insights. It follows a modern frontend architecture with clear separation of concerns.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface (React)                   │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Pages     │  │  Components  │  │   Layouts    │       │
│  └─────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  State Management (Zustand)                  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Auth Store  │  │  Race Store  │  │   UI Store   │       │
│  └─────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Layer (TanStack Query + WS)                │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ HTTP Client │  │  WebSocket   │  │    Cache     │       │
│  └─────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Backend + Redis                     │
└─────────────────────────────────────────────────────────────┘
```

## Component Hierarchy

```
App.tsx
├── Router
│   ├── LoginPage
│   └── DashboardLayout
│       ├── Header
│       ├── Sidebar
│       └── Outlet
│           ├── LiveStrategyConsole
│           │   ├── DriverSelector
│           │   ├── RecommendationCard
│           │   ├── PitWindowChart
│           │   └── SCProbabilityMeter
│           ├── LapByLapMonitor
│           │   ├── LapTimeChart
│           │   ├── DegradationChart
│           │   └── StintTable
│           ├── StrategyTreeVisualizer
│           │   ├── D3TreeCanvas
│           │   ├── ComparisonPanel
│           │   └── SimulationControls
│           ├── CompetitorComparison
│           │   ├── DriverCard[]
│           │   ├── PaceChart
│           │   └── UndercutCalculator
│           └── WeatherTrackEvolution
│               ├── WeatherForecast
│               ├── TrackHeatmap
│               └── GripEvolutionChart
```

## Data Flow

### 1. HTTP Requests (REST API)

```
User Action
    ↓
React Component
    ↓
TanStack Query Hook (useLapTimePrediction)
    ↓
Service Layer (predictionService.predictLapTime)
    ↓
API Client (axios + interceptors)
    ↓
FastAPI Backend
    ↓
Response → Cache → Component Re-render
```

### 2. WebSocket Updates (Real-time)

```
Race Event (Lap Completed)
    ↓
FastAPI Backend → WebSocket Message
    ↓
WebSocket Manager (receives message)
    ↓
Race Store (handleWebSocketMessage)
    ↓
State Update → Component Re-render
```

### 3. Authentication Flow

```
Login Form Submit
    ↓
authService.login(username, password)
    ↓
POST /api/v1/auth/token
    ↓
JWT Token → localStorage
    ↓
authStore.setUser(user)
    ↓
Navigate to /strategy
```

## State Management

### Zustand Stores

#### 1. Auth Store (`authStore.ts`)

**Purpose**: Manage authentication state

**State**:
- `user`: UserInfo | null
- `token`: string | null
- `isAuthenticated`: boolean
- `isLoading`: boolean

**Actions**:
- `login(username, password)`
- `logout()`
- `checkAuth()`

**Persistence**: localStorage (`f1-auth-storage`)

#### 2. Race Store (`raceStore.ts`)

**Purpose**: Manage live race data

**State**:
- `raceState`: RaceState | null
- `selectedDriverNumber`: number | null
- `isLive`: boolean
- `lastUpdate`: Date | null

**Actions**:
- `setRaceState(state)`
- `updateDriverState(driverNumber, update)`
- `handleWebSocketMessage(message)`

**Updates**: WebSocket + Polling

#### 3. UI Store (`uiStore.ts`)

**Purpose**: UI preferences and settings

**State**:
- `theme`: 'light' | 'dark'
- `sidebarCollapsed`: boolean
- `refreshInterval`: number
- `notificationsEnabled`: boolean

**Persistence**: localStorage (`f1-ui-storage`)

## API Integration

### Services Architecture

```
Service Layer
├── apiClient.ts          # Axios instance with interceptors
├── authService.ts        # Authentication endpoints
├── predictionService.ts  # Prediction endpoints
├── simulationService.ts  # Simulation endpoints
├── strategyService.ts    # Strategy endpoints
└── raceStateService.ts   # Race state CRUD
```

### Request Interceptor Flow

```
Request
    ↓
Add JWT token (Authorization: Bearer)
    ↓
Add API version header (X-API-Version: v1)
    ↓
Generate request ID (X-Request-ID: uuid)
    ↓
Send to backend
```

### Response Interceptor Flow

```
Response
    ↓
Check status code
    ├── 401 → Logout + Redirect to login
    ├── 429 → Show rate limit toast
    ├── 5xx → Retry with exponential backoff (max 3)
    └── 2xx → Extract data from APIResponse wrapper
    ↓
Return to component
```

## WebSocket Manager

### Connection Lifecycle

```
connect(sessionId, token)
    ↓
new WebSocket(`ws://localhost:8000/ws/race/${sessionId}?token=${token}`)
    ↓
onopen → startHeartbeat() (ping every 30s)
    ↓
onmessage → Parse JSON → Notify handlers
    ↓
onclose → handleReconnect() (exponential backoff)
    ↓
Retry: 1s → 2s → 4s → 8s → max 30s
```

### Message Types

- `RACE_STATE_UPDATE`: Full race state
- `LAP_COMPLETED`: Driver completed lap
- `PIT_STOP`: Driver pit stop event
- `SAFETY_CAR`: Safety car deployment
- `STRATEGY_RECOMMENDATION`: New recommendation

## Caching Strategy

### TanStack Query Configuration

```typescript
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5000,        // 5s default
      gcTime: 300000,         // 5min cache
      refetchOnWindowFocus: true,
      retry: 2,
    },
  },
});
```

### Cache Keys

- `['raceState', sessionId]` - Race state
- `['prediction', 'laptime', request]` - Lap time predictions
- `['prediction', 'degradation', request]` - Tire degradation
- `['prediction', 'safety-car', request]` - SC probability
- `['strategy', 'recommendation', request]` - Strategy rec
- `['strategy', 'modules']` - Decision modules list

### Cache Invalidation

```typescript
// On WebSocket message
if (message.type === 'LAP_COMPLETED') {
  queryClient.invalidateQueries(['raceState', sessionId]);
}

// On mutation success
mutationSuccess: () => {
  queryClient.invalidateQueries(['raceState', sessionId]);
}
```

## Performance Optimizations

### 1. Code Splitting

```typescript
const LiveStrategyConsole = lazy(() => import('./pages/LiveStrategyConsole'));
```

### 2. Component Memoization

```typescript
export default React.memo(DriverCard, (prev, next) => {
  return prev.driver.driver_number === next.driver.driver_number &&
         prev.driver.current_position === next.driver.current_position;
});
```

### 3. Virtualization

Long lists (20 drivers, 70 laps) use `react-window` for performance.

### 4. Debouncing

```typescript
const debouncedUpdate = useMemo(
  () => debounce(updateChart, 300),
  []
);
```

## Security

### 1. Authentication

- JWT tokens stored in localStorage
- Tokens included in all API requests
- Auto-refresh on expiry
- Secure logout clears all tokens

### 2. CORS

Backend must include frontend URL in CORS config:

```python
allow_origins=["http://localhost:5173", "http://localhost:3000"]
```

### 3. XSS Prevention

- React escapes all user input
- CSP headers in production (nginx)
- No `dangerouslySetInnerHTML`

## Error Handling

### Error Boundary

```typescript
<ErrorBoundary fallback={<ErrorFallback />}>
  <Page />
</ErrorBoundary>
```

### API Errors

- 401: Redirect to login
- 429: Show rate limit toast
- 5xx: Retry with exponential backoff
- Network error: Show connection lost banner

### Loading States

- Skeleton loaders for initial load
- Spinners for button actions
- Progress bars for long operations

## Deployment

### Development

```bash
npm run dev  # Vite dev server on :5173
```

### Production

```bash
npm run build  # Build to dist/
docker build -t f1-dashboard .
docker run -p 3000:80 f1-dashboard
```

### Nginx

- Serves static files from `/usr/share/nginx/html`
- Proxies `/api` to backend `:8000`
- Proxies `/ws` with WebSocket upgrade
- SPA routing: All routes serve `index.html`

## Monitoring

### Performance Marks

```typescript
performance.mark('chart-render-start');
// Render chart
performance.mark('chart-render-end');
performance.measure('chart-render', 'chart-render-start', 'chart-render-end');
```

### Error Logging

```typescript
console.error('[API Error]', error);
// In production: Send to Sentry
```

### User Analytics

- Page views (React Router)
- Button clicks
- API latency
- WebSocket connection status

## Future Enhancements

1. **Offline Support**: Service Worker + IndexedDB
2. **PWA**: Installable app with manifest
3. **Multi-language**: i18n support
4. **Theme Customization**: Team color schemes
5. **Advanced Charting**: 3D visualizations
6. **AI Insights**: GPT-powered analysis
7. **Mobile App**: React Native version
