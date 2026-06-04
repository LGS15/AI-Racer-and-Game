import { compileTrack, hashTrack } from '../track';
import {
  createInitialState, stepSimulation, buildObservation, DEFAULT_SIM_CONFIG,
} from '../simulation';
import { encodeObservation, STATE_VECTOR_SIZE, STATE_VECTOR_VERSION } from './encodeObservation';
import { indexToAction, AI_ACTIONS, ACTION_SPACE_VERSION } from './actions';
import type { LapTraceJson, LapTracePoint, SimulationState, TrackJson } from '../types';

// Resolvers for in-flight action_request messages, keyed by step. Unlike the
// live ExternalAgent (which fires-and-buffers and relies on the rAF loop to
// pump the socket), a tight headless loop must *await* each reply — so we
// correlate request and response by step number here.
type Pending = Map<number, (actionIndex: number) => void>;

export interface EpisodeResult {
  lapTraces: LapTraceJson[];
}

// Listens for the policy's messages: resolves a pending action_request when its
// action_response arrives, and signals readiness on session_ready. Returns a
// detach function so the listener doesn't outlive the episode.
function attachResponseHandler(ws: WebSocket, pending: Pending, onReady: () => void): () => void {
  const handler = (event: MessageEvent) => {
    let msg: Record<string, unknown>;
    try {
      msg = JSON.parse(event.data as string);
    } catch {
      return; // ignore unparseable frames
    }
    if (msg.type === 'session_ready') {
      onReady();
    } else if (msg.type === 'action_response' && typeof msg.step === 'number') {
      const resolve = pending.get(msg.step);
      if (resolve) {
        pending.delete(msg.step);
        resolve(typeof msg.actionIndex === 'number' ? msg.actionIndex : 0);
      }
    }
  };
  ws.addEventListener('message', handler);
  return () => ws.removeEventListener('message', handler);
}

// Sends one action_request and resolves with the action index the policy picks.
// (action_request carries no env: no policy reads it on requests — reward is
// computed from the transition's env below.)
function requestAction(ws: WebSocket, pending: Pending, sessionId: string, step: number, state: number[]): Promise<number> {
  return new Promise((resolve) => {
    pending.set(step, resolve);
    ws.send(JSON.stringify({ type: 'action_request', sessionId, step, state }));
  });
}

// One episode, as fast as the CPU allows, over an already-open socket. Talks the
// same protocol (§08) to the same Python policy as the live bridge — the policy
// can't tell the difference, but there's no canvas and no 60 Hz cap.
export async function runEpisode(track: TrackJson, ws: WebSocket, maxSteps = 3000): Promise<EpisodeResult> {
  const compiled = compileTrack(track);
  const sessionId = `headless-${Date.now()}`;
  const pending: Pending = new Map();
  const lapTraces: LapTraceJson[] = [];
  let currentLapPoints: LapTracePoint[] = [];
  let lastTraceSampleAt = -Infinity;

  let markReady: () => void = () => {};
  const ready = new Promise<void>((resolve) => { markReady = resolve; });
  const detach = attachResponseHandler(ws, pending, markReady);

  try {
    // 1. Handshake — the same session_start the bridge sends; wait for the policy to accept.
    ws.send(JSON.stringify({
      type: 'session_start',
      sessionId,
      trackHash: hashTrack(track),
      stateVectorVersion: STATE_VECTOR_VERSION,
      actionSpaceVersion: ACTION_SPACE_VERSION,
      fixedDt: DEFAULT_SIM_CONFIG.fixedDt,
      stateSize: STATE_VECTOR_SIZE,
      actionCount: AI_ACTIONS.length,
    }));
    await ready;

    // 2. Step loop.
    let state = createInitialState(compiled, DEFAULT_SIM_CONFIG, 1);
    for (let step = 0; step < maxSteps; step++) {
      const obs = buildObservation(state.car, compiled, DEFAULT_SIM_CONFIG);
      const vector = encodeObservation(obs, DEFAULT_SIM_CONFIG);

      const actionIndex = await requestAction(ws, pending, sessionId, step, vector);
      const next = stepSimulation(
        state, indexToAction(actionIndex), compiled, DEFAULT_SIM_CONFIG.fixedDt, DEFAULT_SIM_CONFIG,
      );
      const collided = next.events.some((e) => e.type === 'collision');
      const checkpointCompleted = next.events.some((e) => e.type === 'checkpoint');
      const lapEvent = next.events.find((e) => e.type === 'lap');
      const nextObs = buildObservation(next.car, compiled, DEFAULT_SIM_CONFIG, collided);
      const tracePoint = toTracePoint(next, compiled.totalLength);
      const startsTimedLap = next.events.some((e) => e.type === 'checkpoint' && e.checkpointIndex === 0);
      const shouldSampleTrace = tracePoint.t - lastTraceSampleAt >= 1 / 20 || next.events.length > 0;

      if (startsTimedLap) {
        currentLapPoints = [tracePoint];
        lastTraceSampleAt = tracePoint.t;
      } else if (currentLapPoints.length > 0 && shouldSampleTrace) {
        currentLapPoints.push(tracePoint);
        lastTraceSampleAt = tracePoint.t;
      }

      if (lapEvent?.type === 'lap') {
        if (currentLapPoints[currentLapPoints.length - 1] !== tracePoint) {
          currentLapPoints.push(tracePoint);
        }
        lapTraces.push({
          version: 1,
          trackVersion: track.version,
          trackHash: hashTrack(track),
          seed: 1,
          createdAt: new Date().toISOString(),
          agent: { id: 'external', label: 'External AI' },
          source: { sessionId },
          lap: lapEvent.lap,
          lapTime: round(lapEvent.lapTime),
          completedAt: round(lapEvent.time),
          points: currentLapPoints.map((point) => ({ ...point })),
        });
        currentLapPoints = [tracePoint];
        lastTraceSampleAt = tracePoint.t;
      }

      ws.send(JSON.stringify({
        type: 'transition',
        sessionId,
        step,
        state: vector,
        actionIndex,
        nextState: encodeObservation(nextObs, DEFAULT_SIM_CONFIG),
        done: false,
        env: {
          previousProgressDistance: state.car.progressDistance,
          nextProgressDistance: next.car.progressDistance,
          totalLength: compiled.totalLength,
          speed: Math.hypot(next.car.vx, next.car.vy),
          collision: collided,
          offTrack: next.car.offTrack,
          checkpointCompleted,
          lapCompleted: Boolean(lapEvent),
          lapTime: lapEvent?.type === 'lap' ? lapEvent.lapTime : undefined,
          elapsed: next.elapsed,
        },
      }));

      state = next;
    }
    return { lapTraces };
  } finally {
    detach();
  }
}

function toTracePoint(state: SimulationState, totalLength: number): LapTracePoint {
  return {
    t: round(state.elapsed),
    x: round(state.car.x),
    y: round(state.car.y),
    speed: round(Math.hypot(state.car.vx, state.car.vy)),
    progressDistance: round(state.car.progressDistance % totalLength),
    offTrack: state.car.offTrack,
  };
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}
