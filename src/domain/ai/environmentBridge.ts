import { DEFAULT_SIM_CONFIG } from '../simulation';
import { hashTrack } from '../track';
import type { Agent, AgentAction, AgentContext, AgentObservation, CompiledTrack, SimulationState } from '../types';
import { ACTION_SPACE_VERSION, AI_ACTIONS } from './actions';
import { STATE_VECTOR_SIZE, STATE_VECTOR_VERSION, encodeObservation } from './encodeObservation';
import type { ToServerMessage } from './protocol';

export type BridgeStatus = 'disconnected' | 'connecting' | 'connected' | 'ready';

const IDLE_ACTION: AgentAction = { throttle: 0, brake: 0, steer: 0 };

export class ExternalAgent implements Agent {
  id = 'external';
  label = 'External AI';

  status: BridgeStatus = 'disconnected';
  onStatusChange?: (status: BridgeStatus) => void;
  onResetRequest?: (reason: string) => void;

  private ws: WebSocket | null = null;
  private sessionId = '';
  private stepCount = 0;
  private episode = 0;
  private compiled: CompiledTrack | null = null;
  private context: AgentContext | null = null;

  // The action buffered from Python's last response — returned on the next step()
  private pendingAction: AgentAction = IDLE_ACTION;
  private pendingActionIndex = 0;

  // Action that step() actually returned on the previous call — used in onTransition()
  private appliedActionIndex = 0;

  // State vector sent with the most recent action_request — used in onTransition()
  private lastRequestState: number[] = [];

  connect(url: string): void {
    this.disconnect();
    this._setStatus('connecting');
    let ws: WebSocket;
    try {
      ws = new WebSocket(url);
    } catch {
      this._setStatus('disconnected');
      return;
    }
    this.ws = ws;

    ws.onopen = () => {
      this._setStatus('connected');
      if (this.context && this.compiled) {
        this._sendSessionStart();
      }
    };

    ws.onmessage = (event: MessageEvent) => {
      try {
        this._handleMessage(JSON.parse(event.data as string));
      } catch {
        // ignore unparseable messages
      }
    };

    ws.onclose = () => {
      this.ws = null;
      this._setStatus('disconnected');
    };
    ws.onerror = () => {
      this.ws = null;
      this._setStatus('disconnected');
    };
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.onerror = null;
      this.ws.close();
      this.ws = null;
    }
    this._setStatus('disconnected');
  }

  setCompiledTrack(compiled: CompiledTrack): void {
    this.compiled = compiled;
  }

  // Called by App.tsx on reset — sends session_start if already connected
  reset(context: AgentContext): void {
    this.context = context;
    this.stepCount = 0;
    this.pendingAction = IDLE_ACTION;
    this.pendingActionIndex = 0;
    this.appliedActionIndex = 0;
    this.lastRequestState = [];

    if (this.status === 'connected' || this.status === 'ready') {
      this.episode += 1;
      this._sendSessionStart();
    }
  }

  step(observation: AgentObservation, _dt: number): AgentAction {
    if (this.status !== 'ready') return IDLE_ACTION;

    const state = encodeObservation(observation, DEFAULT_SIM_CONFIG);
    this.lastRequestState = state;

    // Capture what we're about to return so onTransition() knows which action was applied
    this.appliedActionIndex = this.pendingActionIndex;
    const actionToReturn = this.pendingAction;

    // Request Python's decision for the *next* step while we apply the current buffered one
    this._send({
      type: 'action_request',
      sessionId: this.sessionId,
      step: this.stepCount,
      state,
      env: {
        elapsed: this.stepCount * DEFAULT_SIM_CONFIG.fixedDt,
        lap: 0,
        progressDistance: observation.lapProgress * (this.compiled?.totalLength ?? 0),
        speed: observation.speed,
        collisions: 0,
        offTrack: observation.offTrack,
      },
    });

    this.stepCount += 1;
    return actionToReturn;
  }

  // Called by App.tsx after each physics step with both the before and after states.
  // Sends the full transition metadata Python needs to compute rewards.
  onTransition(prev: SimulationState, next: SimulationState, nextObs: AgentObservation, compiled: CompiledTrack): void {
    if (this.status !== 'ready' || this.lastRequestState.length === 0) return;

    const nextState = encodeObservation(nextObs, DEFAULT_SIM_CONFIG);
    const checkpointCompleted = next.events.some((e) => e.type === 'checkpoint');
    const lapEvent = next.events.find((e) => e.type === 'lap');
    const lapCompleted = Boolean(lapEvent);
    const collision = next.events.some((e) => e.type === 'collision');

    this._send({
      type: 'transition',
      sessionId: this.sessionId,
      step: this.stepCount - 1,
      state: this.lastRequestState,
      actionIndex: this.appliedActionIndex,
      nextState,
      done: false,
      env: {
        previousProgressDistance: prev.car.progressDistance,
        nextProgressDistance: next.car.progressDistance,
        totalLength: compiled.totalLength,
        speed: Math.hypot(next.car.vx, next.car.vy),
        collision,
        offTrack: next.car.offTrack,
        checkpointCompleted,
        lapCompleted,
        lapTime: lapEvent?.type === 'lap' ? lapEvent.lapTime : undefined,
        elapsed: next.elapsed,
      },
    });

    this.lastRequestState = nextState;
  }

  private _sendSessionStart(): void {
    if (!this.context || !this.compiled) return;
    this.sessionId = `session-${Date.now()}-ep${this.episode}`;
    this._setStatus('connected'); // reset to connected until Python responds with session_ready
    this._send({
      type: 'session_start',
      sessionId: this.sessionId,
      trackHash: hashTrack(this.context.track),
      stateVectorVersion: STATE_VECTOR_VERSION,
      actionSpaceVersion: ACTION_SPACE_VERSION,
      fixedDt: DEFAULT_SIM_CONFIG.fixedDt,
      stateSize: STATE_VECTOR_SIZE,
      actionCount: AI_ACTIONS.length,
    });
  }

  private _handleMessage(msg: Record<string, unknown>): void {
    switch (msg.type) {
      case 'session_ready':
        this._setStatus('ready');
        break;
      case 'action_response': {
        const idx = typeof msg.actionIndex === 'number' ? msg.actionIndex : 0;
        this.pendingActionIndex = Math.max(0, Math.min(AI_ACTIONS.length - 1, Math.round(idx)));
        this.pendingAction = AI_ACTIONS[this.pendingActionIndex];
        break;
      }
      case 'reset_request':
        this.onResetRequest?.(typeof msg.reason === 'string' ? msg.reason : 'unknown');
        break;
    }
  }

  private _send(msg: ToServerMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  private _setStatus(status: BridgeStatus): void {
    this.status = status;
    this.onStatusChange?.(status);
  }
}
