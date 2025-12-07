// Utility functions for formatting data

import { format } from 'date-fns';

/**
 * Format lap time in MM:SS.mmm format
 */
export function formatLapTime(seconds: number): string {
  if (!seconds || seconds < 0) return '--:--.---';

  const minutes = Math.floor(seconds / 60);
  const secs = seconds % 60;

  return `${minutes}:${secs.toFixed(3).padStart(6, '0')}`;
}

/**
 * Format time difference with +/- sign
 */
export function formatTimeDelta(seconds: number): string {
  if (!seconds || !isFinite(seconds)) return '--';

  const sign = seconds >= 0 ? '+' : '';
  return `${sign}${seconds.toFixed(3)}s`;
}

/**
 * Format gap time
 */
export function formatGap(seconds: number): string {
  if (!seconds || !isFinite(seconds)) return '--';
  if (seconds === 0) return 'LEADER';

  return `+${seconds.toFixed(1)}s`;
}

/**
 * Format percentage
 */
export function formatPercentage(value: number, decimals = 1): string {
  if (!isFinite(value)) return '--';
  return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Format number with thousands separator
 */
export function formatNumber(value: number, decimals = 0): string {
  if (!isFinite(value)) return '--';
  return value.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

/**
 * Format temperature
 */
export function formatTemperature(celsius: number): string {
  if (!isFinite(celsius)) return '--°C';
  return `${celsius.toFixed(1)}°C`;
}

/**
 * Format speed
 */
export function formatSpeed(kmh: number): string {
  if (!isFinite(kmh)) return '-- km/h';
  return `${kmh.toFixed(0)} km/h`;
}

/**
 * Format date/time
 */
export function formatDateTime(date: Date | string, formatStr = 'PPpp'): string {
  try {
    const d = typeof date === 'string' ? new Date(date) : date;
    return format(d, formatStr);
  } catch {
    return '--';
  }
}

/**
 * Format timestamp as relative time
 */
export function formatRelativeTime(date: Date | string): string {
  try {
    const d = typeof date === 'string' ? new Date(date) : date;
    const now = new Date();
    const diffMs = now.getTime() - d.getTime();
    const diffSec = Math.floor(diffMs / 1000);

    if (diffSec < 60) return `${diffSec}s ago`;
    if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
    if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
    return `${Math.floor(diffSec / 86400)}d ago`;
  } catch {
    return '--';
  }
}

/**
 * Format position (1st, 2nd, 3rd, etc.)
 */
export function formatPosition(position: number): string {
  if (!position || position < 1) return '--';

  const suffixes = ['th', 'st', 'nd', 'rd'];
  const lastDigit = position % 10;
  const lastTwoDigits = position % 100;

  if (lastTwoDigits >= 11 && lastTwoDigits <= 13) {
    return `${position}th`;
  }

  return `${position}${suffixes[lastDigit] || suffixes[0]}`;
}

/**
 * Format stint notation (e.g., "S-15" for 15 laps on Soft)
 */
export function formatStint(compound: string, laps: number): string {
  const shortCode = compound.charAt(0).toUpperCase();
  return `${shortCode}-${laps}`;
}

/**
 * Format degradation rate
 */
export function formatDegradationRate(rate: number): string {
  if (!isFinite(rate)) return '--';
  return `${rate >= 0 ? '+' : ''}${rate.toFixed(3)}s/lap`;
}

/**
 * Truncate string with ellipsis
 */
export function truncate(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength - 3)}...`;
}

/**
 * Format file size
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
}
