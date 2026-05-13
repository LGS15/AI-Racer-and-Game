export type Mode = 'drive' | 'edit' | 'analyze';

export interface Vec2 {
  x: number;
  y: number;
}

export interface TrackPoint extends Vec2 {
  id: string;
  width?: number;
}

export interface TrackCheckpoint {
  id: string;
  progress: number;
}

export interface TrackJson {
  version: 1;
  metadata: {
    name: string;
    createdAt: string;
    description?: string;
  };
  centerline: TrackPoint[];
  globalWidth: number;
  start: {
    pointIndex: number;
    direction: 1 | -1;
  };
  checkpointCount: number;
  checkpoints?: TrackCheckpoint[];
}

export interface ValidationResult<T> {
  ok: boolean;
  value?: T;
  warnings: string[];
  errors: string[];
}

export interface TrackSegment {
  index: number;
  a: TrackPoint;
  b: TrackPoint;
  length: number;
  cumulative: number;
  heading: number;
  widthA: number;
  widthB: number;
}

export interface CompiledTrack {
  source: TrackJson;
  segments: TrackSegment[];
  totalLength: number;
  startDistance: number;
}

export interface TrackSample {
  point: Vec2;
  tangent: Vec2;
  normal: Vec2;
  heading: number;
  width: number;
  absoluteDistance: number;
  progressDistance: number;
}

export interface ClosestTrackPoint extends TrackSample {
  segmentIndex: number;
  segmentT: number;
  offset: Vec2;
  lateralOffset: number;
  distanceToCenter: number;
  onTrack: boolean;
}

export interface AgentAction {
  throttle: number;
  brake: number;
  steer: number;
}

export interface AgentObservation {
  rays: number[];
  rayAngles: number[];
  speed: number;
  yawRate: number;
  bodySlipAngle: number;
  headingError: number;
  lateralOffset: number;
  checkpointProgress: number;
  lapProgress: number;
  offTrack: boolean;
  collision: boolean;
}

export interface AgentContext {
  track: TrackJson;
  seed: number;
  checkpointCount: number;
  rayAngles: number[];
}

export interface Agent {
  id: string;
  label: string;
  reset(context: AgentContext): void;
  step(observation: AgentObservation, dt: number): AgentAction;
}

export interface SimConfig {
  fixedDt: number;
  carLength: number;
  carWidth: number;
  carRadius: number;
  engineForce: number;
  brakeForce: number;
  reverseBrakeForce: number;
  forwardDrag: number;
  offTrackDrag: number;
  maxSpeed: number;
  maxReverseSpeed: number;
  cgToFront: number;
  cgToRear: number;
  maxSteerAngle: number;
  corneringStiffnessFront: number;
  corneringStiffnessRear: number;
  tireGripFront: number;
  tireGripRear: number;
  yawInertia: number;
  rayAngles: number[];
  rayMaxDistance: number;
  rayStep: number;
}

export interface CarState {
  x: number;
  y: number;
  heading: number;
  vx: number;
  vy: number;
  yawRate: number;
  lap: number;
  lapTime: number;
  lastLapTime?: number;
  bestLapTime?: number;
  progressDistance: number;
  nextCheckpoint: number;
  collisions: number;
  offTrack: boolean;
}

export type SimulationEvent =
  | { type: 'checkpoint'; time: number; checkpointIndex: number }
  | { type: 'lap'; time: number; lap: number; lapTime: number }
  | { type: 'collision'; time: number; x: number; y: number };

export interface SimulationState {
  car: CarState;
  elapsed: number;
  seed: number;
  lapStartTime: number;
  events: SimulationEvent[];
}

export interface ReplayFrame {
  t: number;
  x: number;
  y: number;
  heading: number;
  speed: number;
  lap: number;
  action: AgentAction;
  observation: Pick<AgentObservation, 'rays' | 'headingError' | 'lateralOffset' | 'lapProgress' | 'offTrack' | 'collision'>;
}

export interface ReplayLap {
  lap: number;
  time: number;
  completedAt: number;
}

export interface ReplayJson {
  version: 1;
  trackVersion: 1;
  trackHash: string;
  seed: number;
  createdAt: string;
  duration: number;
  agent: {
    id: string;
    label: string;
  };
  simConfig: {
    fixedDt: number;
    rayAngles: number[];
    rayMaxDistance: number;
  };
  frames: ReplayFrame[];
  laps: ReplayLap[];
  events: SimulationEvent[];
}
