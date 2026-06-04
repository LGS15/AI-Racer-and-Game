import { describe, expect, it } from 'vitest';
import { ReferenceLineAgent, manualActionFromInput } from './agents';
import {
  addCheckpoint,
  closestPointOnTrack,
  compileTrack,
  createDefaultTrack,
  getCheckpointTargets,
  getCheckpoints,
  getTrackPointBarriers,
  hashTrack,
  moveCheckpoint,
  updateTrackPointWidth,
  validateTrackJson
} from './track';
import { DEFAULT_SIM_CONFIG, buildObservation, clampAction, createInitialState, stepSimulation } from './simulation';
import { createReplayBuilder, finalizeReplay, recordReplayFrame, sampleReplayFrame, validateReplayJson } from './replay';
import type { AgentAction } from './types';

describe('track generation', () => {
  it('compiles a closed centerline track and generates checkpoints', () => {
    const track = createDefaultTrack();
    const compiled = compileTrack(track);

    expect(compiled.segments).toHaveLength(track.centerline.length);
    expect(compiled.totalLength).toBeGreaterThan(1200);
    expect(getCheckpoints(compiled)).toHaveLength(track.checkpointCount);
    expect(hashTrack(track)).toMatch(/^[a-f0-9]{8}$/);
  });

  it('validates imported track JSON with recoverable point ids', () => {
    const track = createDefaultTrack();
    const input = {
      ...track,
      centerline: track.centerline.map(({ x, y }) => ({ x, y }))
    };
    const result = validateTrackJson(input);

    expect(result.ok).toBe(true);
    expect(result.value?.centerline.every((point) => point.id)).toBe(true);
  });

  it('uses manual checkpoints when they are present', () => {
    const track = {
      ...createDefaultTrack(),
      checkpointCount: 2,
      checkpoints: [
        { id: 'start', progress: 0 },
        { id: 'hairpin', progress: 320 }
      ]
    };
    const checkpoints = getCheckpointTargets(compileTrack(track));

    expect(checkpoints).toHaveLength(2);
    expect(checkpoints[1].id).toBe('hairpin');
    expect(checkpoints[1].progress).toBeCloseTo(320);
  });

  it('edits checkpoint positions and local track limits', () => {
    const track = createDefaultTrack();
    const compiled = compileTrack(track);
    const withCheckpoint = addCheckpoint(track, compiled, { x: 575, y: 165 });
    const moved = moveCheckpoint(withCheckpoint, compileTrack(withCheckpoint), 'c0', { x: 985, y: 405 });
    const widened = updateTrackPointWidth(moved, moved.centerline[0].id, 188);

    expect(withCheckpoint.checkpoints).toHaveLength(track.checkpointCount + 1);
    expect(moved.checkpoints?.[0].progress).not.toBe(0);
    expect(widened.centerline[0].width).toBe(188);
  });

  it('reports exact barrier endpoints for track points', () => {
    const track = updateTrackPointWidth(createDefaultTrack(), 'p0', 188);
    const [barrier] = getTrackPointBarriers(track);

    expect(barrier.width).toBe(188);
    expect(distanceBetween(barrier.left, barrier.right)).toBeCloseTo(188, 5);
    expect(barrier.usesGlobalWidth).toBe(false);
  });

  it('does not treat points beyond a segment end as inside the track', () => {
    const track = {
      ...createDefaultTrack(),
      globalWidth: 40,
      centerline: [
        { id: 'a', x: 0, y: 0 },
        { id: 'b', x: 100, y: 0 },
        { id: 'c', x: 100, y: 100 },
        { id: 'd', x: 0, y: 100 }
      ]
    };
    const closest = closestPointOnTrack(compileTrack(track), { x: 180, y: 0 });

    expect(closest.distanceToCenter).toBeGreaterThan(track.globalWidth / 2);
    expect(closest.onTrack).toBe(false);
  });
});

describe('simulation', () => {
  it('clamps invalid agent actions', () => {
    expect(clampAction({ throttle: 9, brake: -1, steer: 3 })).toEqual({ throttle: 1, brake: 0, steer: 1 });
    expect(clampAction({ throttle: Number.NaN, brake: Number.POSITIVE_INFINITY, steer: Number.NaN })).toEqual({
      throttle: 0,
      brake: 0,
      steer: 0
    });
  });

  it('detects wall contact when the car is outside the road', () => {
    const compiled = compileTrack(createDefaultTrack());
    const state = createInitialState(compiled);
    state.car.x = 10;
    state.car.y = 10;

    const next = stepSimulation(state, { throttle: 1, brake: 0, steer: 0 }, compiled, DEFAULT_SIM_CONFIG.fixedDt);

    expect(next.car.collisions).toBeGreaterThan(0);
    expect(next.events.some((event) => event.type === 'collision')).toBe(true);
  });

  it('starts timing at the first start-line crossing instead of counting it as a lap', () => {
    const compiled = compileTrack(createDefaultTrack());
    const config = { ...DEFAULT_SIM_CONFIG, engineForce: 100 };
    let state = createInitialState(compiled, config);

    for (let index = 0; index < 120; index += 1) {
      state = stepSimulation(state, { throttle: 1, brake: 0, steer: 0 }, compiled, config.fixedDt, config);
      if (state.events.some((event) => event.type === 'checkpoint' || event.type === 'lap')) break;
    }

    expect(state.elapsed).toBeGreaterThan(1);
    expect(state.events).toContainEqual({ type: 'checkpoint', time: state.elapsed, checkpointIndex: 0 });
    expect(state.events.some((event) => event.type === 'lap')).toBe(false);
    expect(state.car.lap).toBe(0);
    expect(state.lapStartTime).toBe(state.elapsed);
    expect(state.car.lapTime).toBe(0);
    expect(state.car.nextCheckpoint).toBe(1);
  });

  it('builds ray observations and progress values', () => {
    const compiled = compileTrack(createDefaultTrack());
    const state = createInitialState(compiled);
    const observation = buildObservation(state.car, compiled);

    expect(observation.rays).toHaveLength(DEFAULT_SIM_CONFIG.rayAngles.length);
    expect(observation.rays.every((ray) => ray >= 0 && ray <= DEFAULT_SIM_CONFIG.rayMaxDistance)).toBe(true);
    expect(observation.lapProgress).toBeGreaterThanOrEqual(0);
    expect(observation.lapProgress).toBeLessThanOrEqual(1);
  });

  it('is deterministic for the same action sequence', () => {
    const actions: AgentAction[] = Array.from({ length: 180 }, (_, index) => ({
      throttle: 1,
      brake: 0,
      steer: index < 80 ? 0.08 : -0.04
    }));
    const a = runSequence(actions);
    const b = runSequence(actions);

    expect(a.car.x).toBeCloseTo(b.car.x, 8);
    expect(a.car.y).toBeCloseTo(b.car.y, 8);
    expect(a.car.heading).toBeCloseTo(b.car.heading, 8);
  });
});

describe('agents and replay', () => {
  it('turns keyboard input into manual actions', () => {
    const keys = new Set(['KeyW', 'ArrowLeft']);
    const held = new Set<string>();

    expect(manualActionFromInput(keys, held)).toEqual({ throttle: 1, brake: 0, steer: -1 });
  });

  it('keeps the reference agent inside the action contract', () => {
    const compiled = compileTrack(createDefaultTrack());
    const state = createInitialState(compiled);
    const agent = new ReferenceLineAgent();
    const action = agent.step(buildObservation(state.car, compiled), DEFAULT_SIM_CONFIG.fixedDt);

    expect(action.throttle).toBeGreaterThanOrEqual(0);
    expect(action.throttle).toBeLessThanOrEqual(1);
    expect(action.brake).toBeGreaterThanOrEqual(0);
    expect(action.brake).toBeLessThanOrEqual(1);
    expect(action.steer).toBeGreaterThanOrEqual(-1);
    expect(action.steer).toBeLessThanOrEqual(1);
  });

  it('tries the opposite side when recovery does not improve the front ray', () => {
    const compiled = compileTrack(createDefaultTrack());
    const state = createInitialState(compiled);
    const observation = buildObservation(state.car, compiled);
    const agent = new ReferenceLineAgent();
    const blockedObservation = {
      ...observation,
      rays: observation.rays.map((ray, index) => (index === Math.floor(observation.rays.length / 2) ? 40 : ray))
    };

    const firstAction = agent.step(blockedObservation, DEFAULT_SIM_CONFIG.fixedDt);
    for (let index = 0; index < 26; index += 1) {
      agent.step(blockedObservation, DEFAULT_SIM_CONFIG.fixedDt);
    }
    const fallbackAction = agent.step(blockedObservation, DEFAULT_SIM_CONFIG.fixedDt);

    expect(firstAction.steer).toBeLessThan(0);
    expect(firstAction.throttle).toBeLessThan(0.2);
    expect(fallbackAction.steer).toBeGreaterThan(0);
  });

  it('starts recovery toward the clearer side', () => {
    const compiled = compileTrack(createDefaultTrack());
    const state = createInitialState(compiled);
    const observation = buildObservation(state.car, compiled);
    const centerIndex = Math.floor(observation.rays.length / 2);
    const agent = new ReferenceLineAgent();
    const blockedObservation = {
      ...observation,
      rays: observation.rays.map((ray, index) => {
        if (index === centerIndex) return 40;
        if (index === centerIndex - 2) return 32;
        if (index === centerIndex + 2) return 180;
        return ray;
      })
    };

    const action = agent.step(blockedObservation, DEFAULT_SIM_CONFIG.fixedDt);

    expect(action.steer).toBeGreaterThan(0);
  });

  it('does not enter recovery for a distant front ray drop', () => {
    const compiled = compileTrack(createDefaultTrack());
    const state = createInitialState(compiled);
    const observation = buildObservation(state.car, compiled);
    const centerIndex = Math.floor(observation.rays.length / 2);
    const agent = new ReferenceLineAgent();
    const openDropObservation = {
      ...observation,
      rays: observation.rays.map((ray, index) => (index === centerIndex ? 180 : ray))
    };

    const action = agent.step(openDropObservation, DEFAULT_SIM_CONFIG.fixedDt);

    expect(action.throttle).toBeGreaterThan(0.2);
  });

  it('records and validates replay JSON', () => {
    const track = createDefaultTrack();
    const compiled = compileTrack(track);
    let state = createInitialState(compiled);
    const builder = createReplayBuilder(track, { id: 'manual', label: 'Manual' }, 1, DEFAULT_SIM_CONFIG);

    for (let index = 0; index < 20; index += 1) {
      const action = { throttle: 1, brake: 0, steer: 0 };
      state = stepSimulation(state, action, compiled, DEFAULT_SIM_CONFIG.fixedDt);
      recordReplayFrame(builder, state, action, buildObservation(state.car, compiled));
    }

    const replay = finalizeReplay(builder, state);
    const validation = validateReplayJson(replay);

    expect(validation.ok).toBe(true);
    expect(replay.frames.length).toBeGreaterThan(0);
    expect(sampleReplayFrame(replay, replay.duration / 2)).toBeDefined();
  });
});

function runSequence(actions: AgentAction[]) {
  const compiled = compileTrack(createDefaultTrack());
  let state = createInitialState(compiled);
  for (const action of actions) {
    state = stepSimulation(state, action, compiled, DEFAULT_SIM_CONFIG.fixedDt);
  }
  return state;
}

function distanceBetween(a: { x: number; y: number }, b: { x: number; y: number }) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}
