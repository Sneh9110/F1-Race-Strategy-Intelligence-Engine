# F1 Strategy Dashboard Frontend

Production-grade React + TypeScript dashboard for the F1 Race Strategy Intelligence Engine. Provides real-time strategy recommendations, lap-by-lap monitoring, simulation capabilities, and competitor analysis.

## Features

- 🏎️ **Live Strategy Console**: Real-time strategy recommendations with traffic light system
- 📊 **Lap-by-Lap Monitor**: Track lap times, tire degradation, and stint performance
- 🌳 **Strategy Tree Visualizer**: Interactive D3.js visualization of race strategies
- 🏁 **Competitor Comparison**: Analyze pace deltas and undercut/overcut opportunities
- 🌤️ **Weather & Track Evolution**: Real-time weather updates and grip evolution

## Tech Stack

- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite
- **Routing**: React Router v6
- **Data Fetching**: TanStack Query (React Query)
- **State Management**: Zustand
- **Styling**: Tailwind CSS
- **Charts**: Recharts + D3.js
- **Real-time**: WebSocket
- **Testing**: Playwright (E2E)

## Prerequisites

- Node.js 18+ and npm 9+
- Backend API running on `http://localhost:8000`

## Getting Started

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Environment Setup

Copy the example environment file and configure:

```bash
cp .env.example .env.local
```

Edit `.env.local`:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
VITE_API_VERSION=v1
VITE_REFRESH_INTERVAL_RACE_STATE=5000
VITE_REFRESH_INTERVAL_PREDICTIONS=10000
VITE_REFRESH_INTERVAL_SIMULATIONS=30000
```

### 3. Development

```bash
npm run dev
```

The app will be available at `http://localhost:5173`.

### 4. Build for Production

```bash
npm run build
```

Built files will be in the `dist/` directory.

### 5. Preview Production Build

```bash
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/          # Reusable UI components
│   │   ├── common/         # Buttons, Cards, Modals
│   │   ├── charts/         # Chart wrappers
│   │   └── layout/         # Header, Sidebar
│   ├── pages/              # Page components
│   │   ├── LoginPage.tsx
│   │   ├── LiveStrategyConsole.tsx
│   │   ├── LapByLapMonitor.tsx
│   │   ├── StrategyTreeVisualizer.tsx
│   │   ├── CompetitorComparison.tsx
│   │   └── WeatherTrackEvolution.tsx
│   ├── services/           # API clients
│   │   ├── apiClient.ts
│   │   ├── authService.ts
│   │   ├── predictionService.ts
│   │   ├── simulationService.ts
│   │   └── strategyService.ts
│   ├── hooks/              # Custom React hooks
│   ├── stores/             # Zustand stores
│   ├── types/              # TypeScript interfaces
│   ├── utils/              # Helper functions
│   ├── config/             # Configuration
│   ├── App.tsx             # Root component
│   └── main.tsx            # Entry point
├── public/                 # Static assets
├── tests/                  # E2E tests
├── Dockerfile              # Production Docker image
├── nginx.conf              # Nginx configuration
├── package.json
├── vite.config.ts
├── tailwind.config.js
└── tsconfig.json
```

## Authentication

Default credentials for demo:
- **Username**: `admin`
- **Password**: `admin123`

The app uses JWT tokens stored in `localStorage`. Tokens are automatically refreshed on expiry.

## API Integration

The dashboard connects to the FastAPI backend via:

1. **REST API**: HTTP requests for predictions, simulations, strategy
2. **WebSocket**: Real-time race state updates at `/ws/race/{session_id}`

All API requests include:
- JWT token in `Authorization: Bearer <token>` header
- API version in `X-API-Version: v1` header
- Request ID in `X-Request-ID` header

### Caching Strategy

- **Race State**: 5s stale time, 5s refetch interval
- **Predictions**: 60s stale time (matches API TTL)
- **Simulations**: User-triggered, no auto-refetch
- **Weather**: 5min stale time

## Real-Time Updates

### WebSocket

The dashboard uses WebSocket for instant updates:

```typescript
// Connect to session
wsManager.connect(sessionId, token);

// Subscribe to messages
wsManager.onMessage((message) => {
  // Handle LAP_COMPLETED, PIT_STOP, SAFETY_CAR, etc.
});
```

### Polling Fallback

If WebSocket disconnects, the app falls back to polling:

- **Race State**: Every 5s
- **Predictions**: Every 10s
- **SC Probability**: Every 10s

## Testing

### E2E Tests

Run Playwright tests:

```bash
npm run test:e2e
```

Run in UI mode:

```bash
npm run test:e2e:ui
```

Test files:
- `tests/auth.spec.ts` - Login flow
- `tests/live-strategy.spec.ts` - Strategy console
- `tests/lap-monitor.spec.ts` - Lap-by-lap monitor
- `tests/strategy-tree.spec.ts` - Strategy visualizer
- `tests/competitors.spec.ts` - Competitor comparison
- `tests/weather.spec.ts` - Weather & track

## Docker Deployment

### Build Image

```bash
docker build -t f1-strategy-dashboard .
```

### Run Container

```bash
docker run -p 3000:80 \
  -e VITE_API_BASE_URL=http://api:8000 \
  f1-strategy-dashboard
```

### Docker Compose

The frontend is included in the root `docker-compose.yml`:

```yaml
services:
  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - api
```

## Performance

### Metrics

- First Contentful Paint (FCP): < 1.5s
- Time to Interactive (TTI): < 3s
- Largest Contentful Paint (LCP): < 2.5s
- Chart render time: < 500ms

### Optimizations

- Code splitting by route (lazy loading)
- Memoization of expensive components
- Virtualized long lists
- Image lazy loading
- Service Worker for offline support

## Accessibility

The dashboard follows WCAG 2.1 AA guidelines:

- Keyboard navigation for all interactive elements
- ARIA labels and roles
- Screen reader support
- Minimum contrast ratios (4.5:1)
- Focus indicators

## Troubleshooting

### API Connection Issues

If the API is unreachable:

1. Check backend is running: `http://localhost:8000/api/v1/health`
2. Verify CORS is enabled in backend
3. Check environment variables in `.env.local`

### WebSocket Connection Fails

1. Check WebSocket URL: `ws://localhost:8000/ws/race/{sessionId}`
2. Verify JWT token is valid
3. Check browser console for errors

### Build Errors

```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Clear Vite cache
rm -rf .vite
npm run build
```

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers (iOS Safari, Chrome Mobile)

## Contributing

See `CONTRIBUTING.md` for development guidelines.

## License

MIT License - see `LICENSE` file for details.

## Support

For issues or questions:
- GitHub Issues: [F1 Strategy Engine Issues](https://github.com/your-repo/issues)
- Documentation: See `ARCHITECTURE.md`
