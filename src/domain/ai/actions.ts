import type { AgentAction } from '../types';

export const ACTION_SPACE_VERSION = 1;

export const AI_ACTIONS: AgentAction[] = [
  { throttle: 0.85, brake: 0, steer: 0 },
  { throttle: 0.75, brake: 0, steer: -0.45 },
  { throttle: 0.75, brake: 0, steer: 0.45 },
  { throttle: 0.35, brake: 0, steer: -0.8 },
  { throttle: 0.35, brake: 0, steer: 0.8 },
  { throttle: 0, brake: 0.65, steer: 0 },
  { throttle: 0.12, brake: 0.25, steer: 0 },
];

export function indexToAction(index: number): AgentAction {
  const clamped = Math.max(0, Math.min(AI_ACTIONS.length - 1, Math.round(index)));
  return AI_ACTIONS[clamped];
}
