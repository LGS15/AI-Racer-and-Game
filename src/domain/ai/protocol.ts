// Messages sent from TypeScript → Python
export type SessionStartMessage = {
  type: 'session_start';
  sessionId: string;
  trackHash: string;
  stateVectorVersion: number;
  actionSpaceVersion: number;
  fixedDt: number;
  stateSize: number;
  actionCount: number;
};

export type ActionRequestMessage = {
  type: 'action_request';
  sessionId: string;
  step: number;
  state: number[];
  env: {
    elapsed: number;
    lap: number;
    progressDistance: number;
    speed: number;
    collisions: number;
    offTrack: boolean;
  };
};

export type TransitionMessage = {
  type: 'transition';
  sessionId: string;
  step: number;
  state: number[];
  actionIndex: number;
  nextState: number[];
  done: boolean;
  env: {
    previousProgressDistance: number;
    nextProgressDistance: number;
    totalLength: number;
    speed: number;
    collision: boolean;
    offTrack: boolean;
    lapCompleted: boolean;
    elapsed: number;
  };
};

export type EpisodeStartMessage = {
  type: 'episode_start';
  sessionId: string;
  episode: number;
};

// Messages sent from Python → TypeScript
export type SessionReadyMessage = {
  type: 'session_ready';
  algorithm: string;
  training: boolean;
};

export type ActionResponseMessage = {
  type: 'action_response';
  sessionId: string;
  step: number;
  actionIndex: number;
};

export type ResetRequestMessage = {
  type: 'reset_request';
  sessionId: string;
  reason: string;
};

export type ToServerMessage = SessionStartMessage | ActionRequestMessage | TransitionMessage | EpisodeStartMessage;
export type FromServerMessage = SessionReadyMessage | ActionResponseMessage | ResetRequestMessage;
