import { clamp } from './geometry';
import { DEFAULT_SIM_CONFIG } from './simulation';
import type { Agent, AgentAction, AgentContext, AgentObservation } from './types';

type RecoveryState = 'normal' | 'try-left' | 'try-right';

const RECOVERY_BLOCKED_RAY = 66;
const RECOVERY_TRIGGER_RAY = 135;
const RECOVERY_SUDDEN_DROP = 48;
const RECOVERY_SAMPLE_TIME = 0.42;
const RECOVERY_IMPROVEMENT = 34;
const RECOVERY_CLEAR_RAY = 126;
const RECOVERY_SIDE_SWITCH_MARGIN = 28;
const RECOVERY_COOLDOWN_TIME = 0.22;
const RECOVERY_STEER = 0.72;
const RECOVERY_THROTTLE = 0.08;

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

  private recoveryState: RecoveryState = 'normal';
  private recoveryTimer = 0;
  private previousFrontRay = DEFAULT_SIM_CONFIG.rayMaxDistance;
  private recoveryStartRay = DEFAULT_SIM_CONFIG.rayMaxDistance;
  private bestRecoveryRay = 0;
  private recoveryCooldown = 0;

  reset(_context: AgentContext): void {
    this.cautiousness = 1;
    this.recoveryState = 'normal';
    this.recoveryTimer = 0;
    this.previousFrontRay = DEFAULT_SIM_CONFIG.rayMaxDistance;
    this.recoveryStartRay = DEFAULT_SIM_CONFIG.rayMaxDistance;
    this.bestRecoveryRay = 0;
    this.recoveryCooldown = 0;
  }

  step(observation: AgentObservation, dt: number): AgentAction {
    const { frontRay, leftRay, rightRay } = getRecoveryRays(observation);
    this.recoveryCooldown = Math.max(0, this.recoveryCooldown - dt);

    const blocked = frontRay < RECOVERY_BLOCKED_RAY;
    const suddenDrop = frontRay < this.previousFrontRay - RECOVERY_SUDDEN_DROP && frontRay < RECOVERY_TRIGGER_RAY;
    const needsRecovery = blocked || suddenDrop || observation.collision || observation.offTrack;

    if (this.recoveryState === 'normal' && this.recoveryCooldown <= 0 && needsRecovery) {
      this.enterRecovery(leftRay > rightRay ? 'try-left' : 'try-right', frontRay);
    }

    if (this.recoveryState !== 'normal') {
      const action = this.stepRecovery(frontRay, leftRay, rightRay, dt);
      this.previousFrontRay = frontRay;
      return action;
    }

    const action = this.stepReferenceLine(observation, frontRay, blocked);
    this.previousFrontRay = frontRay;
    return action;
  }

  private stepReferenceLine(observation: AgentObservation, frontRay: number, blocked: boolean): AgentAction {
    const leftRay = observation.rays[1] ?? frontRay;
    const rightRay = observation.rays[observation.rays.length - 2] ?? frontRay;

    const headingTerm = observation.headingError * 1.45;
    const offsetTerm = -observation.lateralOffset * 0.7;
    const rayBias = clamp((rightRay - leftRay) / 260, -0.35, 0.35);
    const steer = clamp(headingTerm + offsetTerm + rayBias, -1, 1);
    const cornering = Math.abs(observation.headingError) + Math.abs(observation.lateralOffset) * 0.45 + Math.abs(steer) * 0.35;
    const targetSpeed = clamp(430 - cornering * 210, 150, 430) * this.cautiousness;

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

  private stepRecovery(frontRay: number, leftRay: number, rightRay: number, dt: number): AgentAction {
    this.recoveryTimer += dt;
    const improved =
      frontRay >= this.recoveryStartRay + RECOVERY_IMPROVEMENT ||
      frontRay >= RECOVERY_CLEAR_RAY ||
      this.bestRecoveryRay >= this.recoveryStartRay + RECOVERY_IMPROVEMENT;
    const sampledLongEnough = this.recoveryTimer >= RECOVERY_SAMPLE_TIME;
    this.bestRecoveryRay = Math.max(this.bestRecoveryRay, frontRay);
    const currentDirection = this.recoveryState === 'try-left' ? -1 : 1;
    const currentSideRay = this.recoveryState === 'try-left' ? leftRay : rightRay;
    const oppositeSideRay = this.recoveryState === 'try-left' ? rightRay : leftRay;
    const oppositeLooksBetter = oppositeSideRay > currentSideRay + RECOVERY_SIDE_SWITCH_MARGIN;

    if (this.recoveryState === 'try-left') {
      if (sampledLongEnough || oppositeLooksBetter) {
        if (improved) {
          this.exitRecovery();
        } else {
          this.enterRecovery('try-right', frontRay);
        }
      }

      return {
        throttle: recoveryThrottle(frontRay),
        brake: recoveryBrake(frontRay),
        steer: -RECOVERY_STEER
      };
    }

    if (sampledLongEnough || improved) {
      this.exitRecovery();
    }

    return {
      throttle: recoveryThrottle(frontRay),
      brake: recoveryBrake(frontRay),
      steer: currentDirection * RECOVERY_STEER
    };
  }

  private enterRecovery(state: Exclude<RecoveryState, 'normal'>, frontRay: number): void {
    this.recoveryState = state;
    this.recoveryTimer = 0;
    this.recoveryStartRay = frontRay;
    this.bestRecoveryRay = frontRay;
  }

  private exitRecovery(): void {
    this.recoveryState = 'normal';
    this.recoveryTimer = 0;
    this.recoveryCooldown = RECOVERY_COOLDOWN_TIME;
  }
}

function getRecoveryRays(observation: AgentObservation) {
  const centerIndex = Math.floor(observation.rays.length / 2);
  const frontRay = observation.rays[centerIndex] ?? 0;
  return {
    frontRay,
    leftRay: observation.rays[Math.max(0, centerIndex - 2)] ?? frontRay,
    rightRay: observation.rays[Math.min(observation.rays.length - 1, centerIndex + 2)] ?? frontRay
  };
}

function recoveryThrottle(frontRay: number): number {
  return frontRay < RECOVERY_BLOCKED_RAY ? RECOVERY_THROTTLE : 0.16;
}

function recoveryBrake(frontRay: number): number {
  if (frontRay < RECOVERY_BLOCKED_RAY) return 0.38;
  return frontRay < RECOVERY_CLEAR_RAY ? 0.12 : 0;
}

export function getAgentLabel(agentId: string): string {
  return agentId === 'manual' ? 'Manual' : 'Reference';
}
