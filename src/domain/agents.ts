import { clamp } from './geometry';
import type { Agent, AgentAction, AgentContext, AgentObservation } from './types';

export function manualActionFromInput(keys: ReadonlySet<string>, heldControls: ReadonlySet<string>): AgentAction {
  const pressed = (code: string) => keys.has(code) || heldControls.has(code);
  return {
    throttle: pressed('ArrowUp') || pressed('KeyW') ? 1 : 0,
    brake: pressed('ArrowDown') || pressed('KeyS') ? 1 : 0,
    steer: (pressed('ArrowRight') || pressed('KeyD') ? 1 : 0) - (pressed('ArrowLeft') || pressed('KeyA') ? 1 : 0)
  };
}

export class ReferenceLineAgent implements Agent {
  id = 'reference-line';
  label = 'Reference';
  private cautiousness = 1;

  reset(_context: AgentContext): void {
    this.cautiousness = 1;
  }

  step(observation: AgentObservation, _dt: number): AgentAction {
    const frontRay = observation.rays[Math.floor(observation.rays.length / 2)] ?? 0;
    const leftRay = observation.rays[1] ?? frontRay;
    const rightRay = observation.rays[observation.rays.length - 2] ?? frontRay;
    const headingTerm = observation.headingError * 1.45;
    const offsetTerm = -observation.lateralOffset * 0.7;
    const rayBias = clamp((rightRay - leftRay) / 260, -0.35, 0.35);
    const steer = clamp(headingTerm + offsetTerm + rayBias, -1, 1);
    const cornering = Math.abs(observation.headingError) + Math.abs(observation.lateralOffset) * 0.45 + Math.abs(steer) * 0.35;
    const targetSpeed = clamp(430 - cornering * 210, 150, 430) * this.cautiousness;
    const blocked = frontRay < 66;

    if (observation.collision || observation.offTrack) {
      this.cautiousness = clamp(this.cautiousness - 0.04, 0.72, 1);
    } else {
      this.cautiousness = clamp(this.cautiousness + 0.004, 0.72, 1);
    }

    return {
      throttle: blocked ? 0.12 : observation.speed < targetSpeed ? 0.86 : 0.2,
      brake: blocked || observation.speed > targetSpeed + 70 ? 0.65 : 0,
      steer
    };
  }
}

export function getAgentLabel(agentId: string): string {
  return agentId === 'manual' ? 'Manual' : 'Reference';
}
