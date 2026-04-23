import { angleDifference, clamp, crossedForward, dot, forwardDelta, fromAngle, scale } from './geometry';
import { closestPointOnTrack, fromProgressDistance, sampleTrackAt, toProgressDistance } from './track';
import type { AgentAction, AgentObservation, CarState, CompiledTrack, SimConfig, SimulationEvent, SimulationState, Vec2 } from './types';

export const DEFAULT_SIM_CONFIG: SimConfig = {
  fixedDt: 1 / 60,
  carLength: 34,
  carWidth: 18,
  carRadius: 15,
  engineForce: 980,
  brakeForce: 1450,
  reverseBrakeForce: 520,
  forwardDrag: 1.45,
  lateralFriction: 8.8,
  offTrackDrag: 4.2,
  maxSpeed: 520,
  maxReverseSpeed: 120,
  turnRate: 3.25,
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
  const forward = fromAngle(car.heading);
  const right = { x: -forward.y, y: forward.x };
  const forwardSpeed = dot({ x: car.vx, y: car.vy }, forward);
  const sideSpeed = dot({ x: car.vx, y: car.vy }, right);

  const steeringAuthority = clamp(Math.abs(forwardSpeed) / 220, 0.18, 1);
  const travelDirection = forwardSpeed < -5 ? -1 : 1;
  car.heading += action.steer * config.turnRate * steeringAuthority * travelDirection * dt;

  const updatedForward = fromAngle(car.heading);
  const updatedRight = { x: -updatedForward.y, y: updatedForward.x };
  const braking =
    action.brake > 0 && Math.abs(forwardSpeed) < 24 && action.throttle < 0.05
      ? -config.reverseBrakeForce
      : -Math.sign(forwardSpeed || 1) * config.brakeForce;
  const longitudinal = action.throttle * config.engineForce + action.brake * braking - forwardSpeed * config.forwardDrag;
  const lateral = -sideSpeed * config.lateralFriction;

  car.vx += (updatedForward.x * longitudinal + updatedRight.x * lateral) * dt;
  car.vy += (updatedForward.y * longitudinal + updatedRight.y * lateral) * dt;

  const speed = Math.hypot(car.vx, car.vy);
  const maxSpeed = forwardSpeed < -5 ? config.maxReverseSpeed : config.maxSpeed;
  if (speed > maxSpeed) {
    car.vx = (car.vx / speed) * maxSpeed;
    car.vy = (car.vy / speed) * maxSpeed;
  }

  car.x += car.vx * dt;
  car.y += car.vy * dt;

  const closest = closestPointOnTrack(compiled, car);
  const trackEdge = Math.max(4, closest.width / 2 - config.carRadius);
  let collision = false;

  if (Math.abs(closest.lateralOffset) > trackEdge) {
    const side = Math.sign(closest.lateralOffset) || 1;
    const correction = Math.abs(closest.lateralOffset) - trackEdge;
    car.x -= closest.normal.x * side * correction;
    car.y -= closest.normal.y * side * correction;
    const velocity: Vec2 = { x: car.vx, y: car.vy };
    const outwardVelocity = dot(velocity, scale(closest.normal, side));
    if (outwardVelocity > 0) {
      car.vx -= closest.normal.x * side * outwardVelocity * 1.35;
      car.vy -= closest.normal.y * side * outwardVelocity * 1.35;
    }
    car.vx *= 1 - clamp(config.offTrackDrag * dt, 0, 0.85);
    car.vy *= 1 - clamp(config.offTrackDrag * dt, 0, 0.85);
    collision = !state.car.offTrack || Math.hypot(car.vx, car.vy) > 85;
  }

  const correctedClosest = closestPointOnTrack(compiled, car);
  const previousProgress = car.progressDistance;
  car.progressDistance = correctedClosest.progressDistance;
  car.offTrack = Math.abs(correctedClosest.lateralOffset) > trackEdge;

  const elapsed = state.elapsed + dt;
  car.lapTime = elapsed - state.lapStartTime;

  const progressDelta = forwardDelta(previousProgress, car.progressDistance, compiled.totalLength);
  const checkpointCount = Math.max(2, compiled.source.checkpointCount);
  const spacing = compiled.totalLength / checkpointCount;
  let nextCheckpoint = car.nextCheckpoint;
  let lapStartTime = state.lapStartTime;

  for (let guard = 0; guard < checkpointCount && crossedForward(previousProgress, progressDelta, nextCheckpoint * spacing, compiled.totalLength); guard += 1) {
    if (nextCheckpoint === 0 && state.elapsed > 1) {
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
  const rays = config.rayAngles.map((rayAngle) => castRay(car, car.heading + rayAngle, compiled, config));
  return {
    rays,
    rayAngles: config.rayAngles,
    speed,
    headingError: angleDifference(tangentHeading, car.heading),
    lateralOffset: clamp(closest.lateralOffset / Math.max(1, closest.width / 2), -2, 2),
    checkpointProgress: car.nextCheckpoint / Math.max(1, compiled.source.checkpointCount),
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
    if (Math.abs(closest.lateralOffset) > closest.width / 2 - config.carRadius * 0.45) {
      return distanceValue;
    }
  }
  return config.rayMaxDistance;
}
