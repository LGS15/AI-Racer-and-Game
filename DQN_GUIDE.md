# Starting With DQN

This guide assumes the racing simulator stays in TypeScript and the AI logic runs outside it, most likely in Python. The TypeScript side should expose a small environment bridge: reset the simulator, encode observations, apply action indices, and stream transition metadata. DQN should be one Python-side approach behind that bridge, not something hard-coded into the game.

## Architecture Goal

Keep the boundary narrow:

```text
TypeScript simulator
  owns tracks, physics, observations, action execution, rendering

Protocol boundary
  owns stable message shapes, action indices, state vector versioning

Python AI runtime
  owns DQN, replay buffer, reward shaping, exploration, logging, model saving
```

That lets you later swap DQN for PPO, SAC, imitation learning, a rule-based search policy, or a remote model without rewriting the simulator.

## Minimal TypeScript Responsibilities

The TS side should do only the environment work that must stay coupled to the game:

- Compile and load tracks.
- Reset `SimulationState`.
- Build `AgentObservation` from the current car state.
- Encode observations into a stable numeric state vector.
- Map an `actionIndex` to an `AgentAction`.
- Step physics with `stepSimulation`.
- Stream enough raw transition metadata for Python to compute rewards.
- Render or record the resulting run.

Avoid putting neural-network training, replay buffers, optimizer logic, or algorithm-specific reward classes in TS.

## Python Responsibilities

Python should own the AI layer:

- Choose actions during live driving.
- Compute reward from transition metadata.
- Store transitions in replay buffers.
- Train DQN or any future algorithm.
- Manage exploration schedules.
- Save and load models.
- Produce evaluation metrics and training logs.

The Python side should not need to know how car physics, ray casting, checkpoint generation, or track editing works.

## Shared Action Space

DQN needs discrete actions. Keep the action table in a small shared contract so every AI approach receives and returns action indices consistently.

First action table:

```ts
export const AI_ACTIONS: AgentAction[] = [
  { throttle: 0.85, brake: 0, steer: 0 },
  { throttle: 0.75, brake: 0, steer: -0.45 },
  { throttle: 0.75, brake: 0, steer: 0.45 },
  { throttle: 0.35, brake: 0, steer: -0.8 },
  { throttle: 0.35, brake: 0, steer: 0.8 },
  { throttle: 0, brake: 0.65, steer: 0 },
  { throttle: 0.12, brake: 0.25, steer: 0 }
];
```

Python should receive the same table in metadata, or load it from a matching JSON file:

```json
{
  "actionSpaceVersion": 1,
  "actions": [
    { "throttle": 0.85, "brake": 0, "steer": 0 },
    { "throttle": 0.75, "brake": 0, "steer": -0.45 }
  ]
}
```

Do not let Python return raw throttle/brake/steer for the first DQN. Returning action indices keeps the protocol simple and makes algorithms comparable.

## State Vector

The state vector can stay TS-owned because it is derived directly from `AgentObservation`. Treat it as a versioned protocol artifact, not as DQN-specific code.

The current default observation has 9 rays, so this first vector has 16 values:

```ts
[
  ...raysNormalized,          // 9 values, ray / rayMaxDistance
  speedNormalized,            // speed / maxSpeed
  headingErrorNormalized,     // headingError / Math.PI
  lateralOffsetNormalized,    // lateralOffset / 2
  Math.sin(lapProgressTau),
  Math.cos(lapProgressTau),
  offTrack ? 1 : 0,
  collision ? 1 : 0
]
```

Implementation sketch:

```ts
function encodeObservation(observation: AgentObservation, config = DEFAULT_SIM_CONFIG): number[] {
  const tau = Math.PI * 2;
  const lapAngle = observation.lapProgress * tau;

  return [
    ...observation.rays.map((ray) => clamp(ray / config.rayMaxDistance, 0, 1)),
    clamp(observation.speed / config.maxSpeed, 0, 1),
    clamp(observation.headingError / Math.PI, -1, 1),
    clamp(observation.lateralOffset / 2, -1, 1),
    Math.sin(lapAngle),
    Math.cos(lapAngle),
    observation.offTrack ? 1 : 0,
    observation.collision ? 1 : 0
  ];
}
```

Include `stateVectorVersion` in every session handshake. If this encoding changes, old models should be rejected or explicitly migrated.

## Protocol Shape

Use WebSocket for live driving because the game needs frequent action requests. Local HTTP can work for slower debugging, but WebSocket is a better default.

### Session Start

TS sends:

```json
{
  "type": "session_start",
  "sessionId": "run-001",
  "trackHash": "abc12345",
  "stateVectorVersion": 1,
  "actionSpaceVersion": 1,
  "fixedDt": 0.0166666667,
  "stateSize": 16,
  "actionCount": 7
}
```

Python responds:

```json
{
  "type": "session_ready",
  "algorithm": "dqn",
  "training": true
}
```

### Action Request

TS sends the current state:

```json
{
  "type": "action_request",
  "sessionId": "run-001",
  "step": 128,
  "state": [0.8, 0.74, 0.41],
  "env": {
    "elapsed": 2.133,
    "lap": 0,
    "progressDistance": 312.4,
    "speed": 188.2,
    "collisions": 0,
    "offTrack": false
  }
}
```

Python responds:

```json
{
  "type": "action_response",
  "sessionId": "run-001",
  "step": 128,
  "actionIndex": 2
}
```

### Transition Event

After TS applies the action and steps physics, it sends the transition:

```json
{
  "type": "transition",
  "sessionId": "run-001",
  "step": 128,
  "state": [0.8, 0.74, 0.41],
  "actionIndex": 2,
  "nextState": [0.82, 0.7, 0.39],
  "done": false,
  "env": {
    "previousProgressDistance": 312.4,
    "nextProgressDistance": 319.8,
    "totalLength": 1840.2,
    "speed": 196.6,
    "collision": false,
    "offTrack": false,
    "lapCompleted": false,
    "elapsed": 2.15
  }
}
```

Python computes reward from this payload and stores the transition.

## Reward In Python

Reward belongs in Python if you want multiple AI approaches and faster experimentation. The key rule is that Python must receive authoritative simulator transition data, not just the normalized state vector.

Start with:

```py
progress_reward = progress_delta_px * 0.02
speed_reward = speed * 0.0008
steering_penalty = abs(action["steer"]) * 0.01
collision_penalty = 2.0 if collision else 0.0
off_track_penalty = 0.12 if off_track else 0.0
stuck_penalty = 0.02 if speed < 12 else 0.0
lap_bonus = 8.0 if lap_completed else 0.0

reward = (
    progress_reward
    + speed_reward
    - steering_penalty
    - collision_penalty
    - off_track_penalty
    - stuck_penalty
    + lap_bonus
)
```

Python can compute `progress_delta_px` from `previousProgressDistance`, `nextProgressDistance`, and `totalLength`. Handle wraparound when the car crosses the start line.

## Episode Rules

Episode termination can be decided in Python, but TS still needs to execute the reset when Python requests it.

Reasonable first limits:

```py
max_episode_seconds = 45
max_collisions = 8
max_no_progress_seconds = 4
target_laps = 1
```

Python can send:

```json
{
  "type": "reset_request",
  "sessionId": "run-001",
  "reason": "no_progress"
}
```

TS resets with:

```ts
createInitialState(compiled, DEFAULT_SIM_CONFIG, seed);
```

Then TS starts a new episode and sends another `session_start` or `episode_start` message.

## DQN Training Loop In Python

DQN is only one consumer of the protocol:

```py
state = message["state"]
action_index = epsilon_greedy(q_network, state, epsilon)
send_action(action_index)

transition = wait_for_transition()
reward = compute_reward(transition)
replay_buffer.push(
    transition["state"],
    transition["actionIndex"],
    reward,
    transition["nextState"],
    transition["done"],
)

if ready_to_train:
    train_step()
```

Bellman target:

```py
target_q = reward + (0 if done else gamma * max(target_network(next_state)))
```

Start hyperparameters:

```py
gamma = 0.99
learning_rate = 0.00025
batch_size = 64
replay_capacity = 50_000
warmup_steps = 2_000
train_every_steps = 4
target_update_every_steps = 1_000
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay_steps = 40_000
```

## Model Shape

First DQN network:

```text
input: 16
dense: 64, relu
dense: 64, relu
output: actionCount
```

Use PyTorch first unless there is a strong reason not to. Export later only if the browser needs local inference without Python running.

## Suggested File Layout

Keep TS protocol/environment files separate from AI algorithms:

```text
src/domain/ai/
  actions.ts
  encodeObservation.ts
  protocol.ts
  environmentBridge.ts
```

Python side:

```text
ai_service/
  server.py
  protocol.py
  rewards.py
  algorithms/
    dqn.py
    random_policy.py
    reference_clone.py
  replay_buffer.py
  train.py
  models/
```

The important part is that `algorithms/` can grow without changing the TypeScript simulator contract.

## Live Mode Vs Training Mode

Use two clients against the same Python service:

Live mode:

- React app renders the game.
- TS sends one action request per simulation step or every few fixed steps.
- Python returns actions.
- Good for watching behavior and debugging.

Training mode:

- A headless TS runner steps the simulator as fast as possible.
- It streams the same protocol messages to Python.
- Rendering is disabled.
- Good for collecting enough experience for DQN.

Do not rely on the visible React loop for serious training. It is useful for inspection, not throughput.

## Milestone Plan

1. Define `AI_ACTIONS`, `stateVectorVersion`, and protocol message types.
2. Add `encodeObservation` and tests for shape, ranges, and finite values.
3. Add a tiny Python WebSocket service with a random policy.
4. Connect the React drive loop to the Python random policy.
5. Add transition messages with progress, collision, off-track, lap, and elapsed metadata.
6. Move reward computation into Python and log reward components.
7. Add a headless TS environment runner using the same protocol.
8. Implement DQN in Python.
9. Train on the default track until progress improves.
10. Add other Python algorithms behind the same action API.

## Debugging Checklist

If behavior looks wrong:

- Verify TS and Python agree on `stateVectorVersion`.
- Verify TS and Python agree on `actionSpaceVersion`.
- Log every `actionIndex` and decoded `AgentAction`.
- Check that all state values are finite.
- Check that `fixedDt` is constant.
- Confirm Python reward uses wrapped progress distance correctly.
- Run a random Python policy first; it should move the car and produce transitions.
- Compare live-mode transitions with headless-runner transitions.

## Recommended First Definition Of Done

The first external DQN path is good enough when:

- The React app can drive from actions returned by Python.
- Python receives full transition metadata and computes reward itself.
- A random Python policy can run complete episodes.
- A headless TS runner can generate faster-than-real-time transitions.
- DQN training improves average progress over random actions.
- The protocol is generic enough that a second Python policy can be added without editing simulator physics.
