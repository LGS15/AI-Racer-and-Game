import { hashTrack } from './track';
import type {
  AgentAction,
  AgentObservation,
  ReplayFrame,
  ReplayJson,
  ReplayLap,
  SimConfig,
  SimulationEvent,
  SimulationState,
  TrackJson,
  ValidationResult
} from './types';

interface ReplayBuilder {
  replay: ReplayJson;
  lastRecordedAt: number;
}

export function createReplayBuilder(track: TrackJson, agent: { id: string; label: string }, seed: number, config: SimConfig): ReplayBuilder {
  return {
    lastRecordedAt: -Infinity,
    replay: {
      version: 1,
      trackVersion: track.version,
      trackHash: hashTrack(track),
      seed,
      createdAt: new Date().toISOString(),
      duration: 0,
      agent,
      simConfig: {
        fixedDt: config.fixedDt,
        rayAngles: config.rayAngles,
        rayMaxDistance: config.rayMaxDistance
      },
      frames: [],
      laps: [],
      events: []
    }
  };
}

export function recordReplayFrame(
  builder: ReplayBuilder,
  state: SimulationState,
  action: AgentAction,
  observation: AgentObservation,
  sampleEvery = 1 / 30
): void {
  if (state.elapsed - builder.lastRecordedAt < sampleEvery && state.events.length === 0) return;
  const frame: ReplayFrame = {
    t: round(state.elapsed),
    x: round(state.car.x),
    y: round(state.car.y),
    heading: round(state.car.heading),
    speed: round(observation.speed),
    lap: state.car.lap,
    action: {
      throttle: round(action.throttle),
      brake: round(action.brake),
      steer: round(action.steer)
    },
    observation: {
      rays: observation.rays.map(round),
      headingError: round(observation.headingError),
      lateralOffset: round(observation.lateralOffset),
      lapProgress: round(observation.lapProgress),
      offTrack: observation.offTrack,
      collision: observation.collision
    }
  };
  builder.replay.frames.push(frame);
  builder.lastRecordedAt = state.elapsed;
  builder.replay.duration = round(state.elapsed);

  for (const event of state.events) {
    builder.replay.events.push(roundEvent(event));
    if (event.type === 'lap') {
      builder.replay.laps.push({ lap: event.lap, time: round(event.lapTime), completedAt: round(event.time) });
    }
  }
}

export function finalizeReplay(builder: ReplayBuilder, state: SimulationState): ReplayJson {
  builder.replay.duration = round(state.elapsed);
  return JSON.parse(JSON.stringify(builder.replay)) as ReplayJson;
}

export function validateReplayJson(input: unknown): ValidationResult<ReplayJson> {
  const warnings: string[] = [];
  const errors: string[] = [];
  const value = input as Partial<ReplayJson> | null;
  if (!value || typeof value !== 'object') {
    return { ok: false, warnings, errors: ['Replay JSON must be an object.'] };
  }
  if (value.version !== 1) errors.push('Replay version must be 1.');
  if (value.trackVersion !== 1) errors.push('Replay trackVersion must be 1.');
  if (typeof value.trackHash !== 'string') errors.push('Replay must include a trackHash.');
  if (!Array.isArray(value.frames) || value.frames.length === 0) errors.push('Replay must include at least one frame.');
  if (!value.agent || typeof value.agent.id !== 'string') errors.push('Replay must include agent metadata.');

  const replay = value as ReplayJson;
  return { ok: errors.length === 0, value: errors.length === 0 ? replay : undefined, warnings, errors };
}

export function sampleReplayFrame(replay: ReplayJson, time: number): ReplayFrame | undefined {
  if (!replay.frames.length) return undefined;
  const t = Math.max(0, Math.min(time, replay.duration));
  let low = 0;
  let high = replay.frames.length - 1;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (replay.frames[mid].t < t) low = mid + 1;
    else high = mid;
  }
  return replay.frames[low] ?? replay.frames[replay.frames.length - 1];
}

export function replaySpeedSeries(replay?: ReplayJson): number[] {
  return replay?.frames.map((frame) => frame.speed) ?? [];
}

export function replayActionSeries(replay?: ReplayJson): { throttle: number[]; brake: number[]; steer: number[] } {
  return {
    throttle: replay?.frames.map((frame) => frame.action.throttle) ?? [],
    brake: replay?.frames.map((frame) => frame.action.brake) ?? [],
    steer: replay?.frames.map((frame) => frame.action.steer) ?? []
  };
}

export function replayLapSeries(replay?: ReplayJson): ReplayLap[] {
  return replay?.laps ?? [];
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function roundEvent(event: SimulationEvent): SimulationEvent {
  if (event.type === 'collision') return { ...event, time: round(event.time), x: round(event.x), y: round(event.y) };
  if (event.type === 'lap') return { ...event, time: round(event.time), lapTime: round(event.lapTime) };
  return { ...event, time: round(event.time) };
}
