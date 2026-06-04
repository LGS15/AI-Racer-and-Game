import { angleDifference, clamp, crossedForward, dot, forwardDelta, fromAngle, scale } from './geometry';
import { closestPointOnTrack, fromProgressDistance, getCheckpointTargets, sampleTrackAt, toProgressDistance } from './track';
import type { AgentAction, AgentObservation, CarState, CompiledTrack, SimConfig, SimulationEvent, SimulationState, Vec2 } from './types';

export const DEFAULT_SIM_CONFIG: SimConfig = {
  fixedDt: 1 / 60,
  carLength: 34,
  carWidth: 18,
  carRadius: 15,
  engineForce: 1100,
  brakeForce: 1450,
  reverseBrakeForce: 520,
  forwardDrag: 0.08,
  offTrackDrag: 4.2,
  maxSpeed: 520,
  maxReverseSpeed: 120,
  cgToFront: 12,
  cgToRear: 12,
  maxSteerAngle: 0.55,
  corneringStiffnessFront: 1800,
  corneringStiffnessRear: 1500,
  tireGripFront: 950,
  tireGripRear: 800,
  yawInertia: 250,
  rayAngles: [-Math.PI / 2, -Math.PI / 3, -Math.PI / 6, -Math.PI / 12, 0, Math.PI / 12, Math.PI / 6, Math.PI / 3, Math.PI / 2],
  rayMaxDistance: 250,
  rayStep: 6
};

export function clampAction(action: AgentAction): AgentAction {
  return {
    throttle: clamp(Number.isFinite(action.throttle) ? action.throttle : 0, 0, 1),
    brake: clamp(Number.isFinite(action.brake) ? action.brake : 0, 0, 1),
    steer: clamp(Number.isFinite(action.steer) ? action.steer : 0, -1, 1)
  };
}

export function createInitialState(compiled: CompiledTrack, config: SimConfig = DEFAULT_SIM_CONFIG, seed = 1): SimulationState {
  const start = sampleTrackAt(compiled, compiled.startDistance);
  const back = fromProgressDistance(compiled, -config.carLength * 1.4);
  const staged = sampleTrackAt(compiled, back);
  const car: CarState = {
    x: staged.point.x,
    y: staged.point.y,
    heading: start.heading,
    vx: 0,
    vy: 0,
    yawRate: 0,
    lap: 0,
    lapTime: 0,
    progressDistance: toProgressDistance(compiled, back),
    nextCheckpoint: 0,
    collisions: 0,
    offTrack: false
  };

  return {
    car,
    elapsed: 0,
    seed,
    lapStartTime: 0,
    events: []
  };
}

export function stepSimulation(
  state: SimulationState,
  rawAction: AgentAction,
  compiled: CompiledTrack,
  dt: number,
  config: SimConfig = DEFAULT_SIM_CONFIG
): SimulationState {
  const action = clampAction(rawAction);
  const car = { ...state.car };
  const events: SimulationEvent[] = [];

  const cosH = Math.cos(car.heading);
  const sinH = Math.sin(car.heading);
  const vbx = car.vx * cosH + car.vy * sinH;
  const vby = -car.vx * sinH + car.vy * cosH;
  const yaw = car.yawRate;

  const delta = action.steer * config.maxSteerAngle;
  const slipDenom = Math.max(Math.abs(vbx), 5);
  const slipFront = delta - (vby + config.cgToFront * yaw) / slipDenom;
  const slipRear = -(vby - config.cgToRear * yaw) / slipDenom;

  const throttleTaper = vbx > 0 ? Math.max(0, 1 - vbx / config.maxSpeed) : 1;
  const engineLong = action.throttle * config.engineForce * throttleTaper;
  const braking =
    action.brake > 0 && Math.abs(vbx) < 24 && action.throttle < 0.05
      ? -config.reverseBrakeForce
      : -Math.sign(vbx || 1) * config.brakeForce;
  const brakeLong = action.brake * braking;
  const dragLong = -vbx * config.forwardDrag;

  const rearLongUsed = Math.min(Math.abs(engineLong), config.tireGripRear);
  const rearLatBudget = Math.sqrt(Math.max(0, config.tireGripRear * config.tireGripRear - rearLongUsed * rearLongUsed));
  const speedFactor = Math.tanh(Math.hypot(vbx, vby) / 30);
  const fyFrontTire =
    config.tireGripFront *
    Math.tanh((config.corneringStiffnessFront * slipFront) / config.tireGripFront) *
    speedFactor;
  const rearDenom = Math.max(rearLatBudget, 1);
  const fyRearTire =
    rearLatBudget * Math.tanh((config.corneringStiffnessRear * slipRear) / rearDenom) * speedFactor;

  const cosD = Math.cos(delta);
  const sinD = Math.sin(delta);
  const fxFrontBody = -fyFrontTire * sinD;
  const fyFrontBody = fyFrontTire * cosD;

  const fbx = engineLong + brakeLong + dragLong + fxFrontBody;
  const fby = fyFrontBody + fyRearTire;
  const moment = config.cgToFront * fyFrontBody - config.cgToRear * fyRearTire;

  const fwx = fbx * cosH - fby * sinH;
  const fwy = fbx * sinH + fby * cosH;

  car.vx += fwx * dt;
  car.vy += fwy * dt;
  car.yawRate = yaw + (moment / config.yawInertia) * dt;
  car.heading += car.yawRate * dt;

  const speed = Math.hypot(car.vx, car.vy);
  const maxSpeed = vbx < -5 ? config.maxReverseSpeed : config.maxSpeed;
  if (speed > maxSpeed) {
    car.vx = (car.vx / speed) * maxSpeed;
    car.vy = (car.vy / speed) * maxSpeed;
  }

  car.x += car.vx * dt;
  car.y += car.vy * dt;

  const closest = closestPointOnTrack(compiled, car);
  const trackEdge = Math.max(4, closest.width / 2 - config.carRadius);
  let collision = false;

  if (closest.distanceToCenter > trackEdge) {
    const side = Math.sign(closest.lateralOffset) || 1;
    const outward = closest.distanceToCenter > 0.000001 ? closest.offset : scale(closest.normal, side);
    const correction = closest.distanceToCenter - trackEdge;
    car.x -= outward.x * correction;
    car.y -= outward.y * correction;
    const velocity: Vec2 = { x: car.vx, y: car.vy };
    const outwardVelocity = dot(velocity, outward);
    if (outwardVelocity > 0) {
      car.vx -= outward.x * outwardVelocity * 1.35;
      car.vy -= outward.y * outwardVelocity * 1.35;
    }
    car.vx *= 1 - clamp(config.offTrackDrag * dt, 0, 0.85);
    car.vy *= 1 - clamp(config.offTrackDrag * dt, 0, 0.85);
    collision = !state.car.offTrack || Math.hypot(car.vx, car.vy) > 85;
  }

  const correctedClosest = closestPointOnTrack(compiled, car);
  const previousProgress = car.progressDistance;
  car.progressDistance = correctedClosest.progressDistance;
  car.offTrack = correctedClosest.distanceToCenter > trackEdge;

  const elapsed = state.elapsed + dt;
  car.lapTime = elapsed - state.lapStartTime;

  const progressDelta = forwardDelta(previousProgress, car.progressDistance, compiled.totalLength);
  const checkpoints = getCheckpointTargets(compiled);
  const checkpointCount = checkpoints.length;
  let nextCheckpoint = car.nextCheckpoint % checkpointCount;
  let lapStartTime = state.lapStartTime;

  for (
    let guard = 0;
    guard < checkpointCount && crossedForward(previousProgress, progressDelta, checkpoints[nextCheckpoint]?.progress ?? 0, compiled.totalLength);
    guard += 1
  ) {
    if (nextCheckpoint === 0 && car.lap === 0 && state.lapStartTime === 0) {
      lapStartTime = elapsed;
      car.lapTime = 0;
      events.push({ type: 'checkpoint', time: elapsed, checkpointIndex: nextCheckpoint });
    } else if (nextCheckpoint === 0) {
      const lapTime = elapsed - state.lapStartTime;
      car.lap += 1;
      car.lastLapTime = lapTime;
      car.bestLapTime = car.bestLapTime === undefined ? lapTime : Math.min(car.bestLapTime, lapTime);
      lapStartTime = elapsed;
      car.lapTime = 0;
      events.push({ type: 'lap', time: elapsed, lap: car.lap, lapTime });
    } else {
      events.push({ type: 'checkpoint', time: elapsed, checkpointIndex: nextCheckpoint });
    }
    nextCheckpoint = (nextCheckpoint + 1) % checkpointCount;
  }

  car.nextCheckpoint = nextCheckpoint;

  if (collision) {
    car.collisions += 1;
    events.push({ type: 'collision', time: elapsed, x: car.x, y: car.y });
  }

  return {
    car,
    elapsed,
    seed: state.seed,
    lapStartTime,
    events
  };
}

export function buildObservation(
  car: CarState,
  compiled: CompiledTrack,
  config: SimConfig = DEFAULT_SIM_CONFIG,
  collision = false
): AgentObservation {
  const closest = closestPointOnTrack(compiled, car);
  const tangentHeading = closest.heading;
  const speed = Math.hypot(car.vx, car.vy);
  const cosH = Math.cos(car.heading);
  const sinH = Math.sin(car.heading);
  const vbx = car.vx * cosH + car.vy * sinH;
  const vby = -car.vx * sinH + car.vy * cosH;
  const bodySlipAngle = speed > 0.0001 ? Math.atan2(vby, vbx) : 0;
  const rays = config.rayAngles.map((rayAngle) => castRay(car, car.heading + rayAngle, compiled, config));
  return {
    rays,
    rayAngles: config.rayAngles,
    speed,
    yawRate: car.yawRate,
    bodySlipAngle,
    headingError: angleDifference(tangentHeading, car.heading),
    lateralOffset: clamp(closest.lateralOffset / Math.max(1, closest.width / 2), -2, 2),
    checkpointProgress: car.nextCheckpoint / Math.max(1, getCheckpointTargets(compiled).length),
    lapProgress: car.progressDistance / compiled.totalLength,
    offTrack: car.offTrack,
    collision
  };
}

function castRay(car: CarState, angle: number, compiled: CompiledTrack, config: SimConfig): number {
  const direction = fromAngle(angle);
  for (let distanceValue = 0; distanceValue <= config.rayMaxDistance; distanceValue += config.rayStep) {
    const point = {
      x: car.x + direction.x * distanceValue,
      y: car.y + direction.y * distanceValue
    };
    const closest = closestPointOnTrack(compiled, point);
    if (closest.distanceToCenter > closest.width / 2 - config.carRadius * 0.45) {
      return distanceValue;
    }
  }
  return config.rayMaxDistance;
}
