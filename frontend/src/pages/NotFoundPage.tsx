import { Link } from 'react-router-dom';

export default function NotFoundPage() {
  return (
    <div className="min-h-screen bg-racing-black flex items-center justify-center p-4">
      <div className="text-center">
        <h1 className="text-6xl font-bold text-redbull mb-4">404</h1>
        <h2 className="text-2xl font-semibold mb-4">Page Not Found</h2>
        <p className="text-gray-400 mb-8">
          The page you're looking for doesn't exist or has been moved.
        </p>
        <Link to="/strategy" className="button-primary inline-block">
          Go to Dashboard
        </Link>
      </div>
    </div>
  );
}
