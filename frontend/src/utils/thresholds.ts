// Warning thresholds and color coding utilities

import { TRAFFIC_LIGHT_COLORS } from '@/config/constants';
import { TrafficLight } from '@/types/strategy.types';

/**
 * Get tire life warning level
 */
export function getTireWarningLevel(
  tireAge: number,
  predictedLife: number
): 'safe' | 'warning' | 'critical' {
  if (!predictedLife || predictedLife <= 0) return 'safe';

  const usage = tireAge / predictedLife;

  if (usage > 0.95) return 'critical'; // Red - need to pit NOW
  if (usage > 0.80) return 'warning'; // Yellow - plan pit soon
  return 'safe'; // Green - tire is fine
}

/**
 * Get tire warning color
 */
export function getTireWarningColor(tireAge: number, predictedLife: number): string {
  const level = getTireWarningLevel(tireAge, predictedLife);

  switch (level) {
    case 'critical':
      return 'text-red-600 bg-red-50 border-red-300';
    case 'warning':
      return 'text-yellow-600 bg-yellow-50 border-yellow-300';
    case 'safe':
      return 'text-green-600 bg-green-50 border-green-300';
  }
}

/**
 * Get safety car risk level
 */
export function getSCRiskLevel(probability: number): 'low' | 'medium' | 'high' {
  if (probability > 0.7) return 'high'; // Red - high probability
  if (probability > 0.4) return 'medium'; // Yellow - moderate probability
  return 'low'; // Green - low probability
}

/**
 * Get safety car risk color
 */
export function getSCRiskColor(probability: number): string {
  const level = getSCRiskLevel(probability);

  switch (level) {
    case 'high':
      return 'text-red-600';
    case 'medium':
      return 'text-yellow-600';
    case 'low':
      return 'text-green-600';
  }
}

/**
 * Get confidence level color
 */
export function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.8) return 'text-green-600';
  if (confidence >= 0.5) return 'text-yellow-600';
  return 'text-red-600';
}

/**
 * Get confidence level label
 */
export function getConfidenceLabel(confidence: number): string {
  if (confidence >= 0.8) return 'High';
  if (confidence >= 0.5) return 'Medium';
  return 'Low';
}

/**
 * Get traffic light color
 */
export function getTrafficLightColor(trafficLight: TrafficLight): string {
  switch (trafficLight) {
    case TrafficLight.GREEN:
      return TRAFFIC_LIGHT_COLORS.GREEN;
    case TrafficLight.AMBER:
      return TRAFFIC_LIGHT_COLORS.AMBER;
    case TrafficLight.RED:
      return TRAFFIC_LIGHT_COLORS.RED;
  }
}

/**
 * Get traffic light background class
 */
export function getTrafficLightClass(trafficLight: TrafficLight): string {
  switch (trafficLight) {
    case TrafficLight.GREEN:
      return 'bg-green-500';
    case TrafficLight.AMBER:
      return 'bg-yellow-500';
    case TrafficLight.RED:
      return 'bg-red-500';
  }
}

/**
 * Get risk level color
 */
export function getRiskColor(risk: number): string {
  if (risk > 0.7) return 'text-red-600';
  if (risk > 0.3) return 'text-yellow-600';
  return 'text-green-600';
}

/**
 * Get position change color
 */
export function getPositionChangeColor(change: number): string {
  if (change > 0) return 'text-green-600'; // Gained positions
  if (change < 0) return 'text-red-600'; // Lost positions
  return 'text-gray-600'; // No change
}

/**
 * Get lap time delta color
 */
export function getLapTimeDeltaColor(delta: number): string {
  if (delta < -0.1) return 'text-green-600'; // Faster
  if (delta > 0.1) return 'text-red-600'; // Slower
  return 'text-gray-600'; // Similar
}

/**
 * Get fuel load warning level
 */
export function getFuelWarningLevel(fuelLoad: number, lapsRemaining: number): 'safe' | 'warning' | 'critical' {
  const fuelPerLap = 1.5; // Approximate kg per lap
  const requiredFuel = lapsRemaining * fuelPerLap;

  if (fuelLoad < requiredFuel * 0.95) return 'critical';
  if (fuelLoad < requiredFuel * 1.05) return 'warning';
  return 'safe';
}

/**
 * Get weather severity color
 */
export function getWeatherSeverityColor(rainProbability: number): string {
  if (rainProbability > 0.7) return 'text-red-600';
  if (rainProbability > 0.3) return 'text-yellow-600';
  return 'text-green-600';
}

/**
 * Get gap status color
 */
export function getGapStatusColor(gap: number, threshold = 20): string {
  if (gap < threshold) return 'text-yellow-600'; // Under pressure
  if (gap > threshold * 2) return 'text-green-600'; // Comfortable
  return 'text-gray-600'; // Normal
}

/**
 * Get performance indicator color (compared to teammate)
 */
export function getPerformanceColor(delta: number): string {
  if (delta < -0.2) return 'text-green-600 font-semibold'; // Significantly faster
  if (delta < -0.05) return 'text-green-600'; // Faster
  if (delta > 0.2) return 'text-red-600 font-semibold'; // Significantly slower
  if (delta > 0.05) return 'text-red-600'; // Slower
  return 'text-gray-600'; // Similar pace
}
