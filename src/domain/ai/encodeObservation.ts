import { clamp } from '../geometry';
import { DEFAULT_SIM_CONFIG } from '../simulation';
import type { AgentObservation, SimConfig } from '../types';

export const STATE_VECTOR_VERSION = 2;
export const STATE_VECTOR_SIZE = 18;
export const MAX_YAW_RATE = 6;

export function encodeObservation(observation: AgentObservation, config: SimConfig = DEFAULT_SIM_CONFIG): number[] {
  const lapAngle = observation.lapProgress * Math.PI * 2;
  return [
    ...observation.rays.map((ray) => clamp(ray / config.rayMaxDistance, 0, 1)),
    clamp(observation.speed / config.maxSpeed, 0, 1),
    clamp(observation.headingError / Math.PI, -1, 1),
    clamp(observation.lateralOffset / 2, -1, 1),
    Math.sin(lapAngle),
    Math.cos(lapAngle),
    observation.offTrack ? 1 : 0,
    observation.collision ? 1 : 0,
    clamp(observation.yawRate / MAX_YAW_RATE, -1, 1),
    clamp(observation.bodySlipAngle / (Math.PI / 2), -1, 1),
  ];
}
