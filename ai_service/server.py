import asyncio
import json
import os
import random

import websockets

from checkpoints import BestPerformanceTracker
from dqn import DQNAgent
from replay_buffer import ReplayBuffer
from rewards import compute_rewards

# MUST match src/domain/ai/actions.ts exactly (so the same order and also vvalues ) 
AI_ACTIONS = [
    {"throttle": 0.85, "brake": 0.0,  "steer":  0.0},
    {"throttle": 0.75, "brake": 0.0,  "steer": -0.45},
    {"throttle": 0.75, "brake": 0.0,  "steer":  0.45},
    {"throttle": 0.35, "brake": 0.0,  "steer": -0.8},
    {"throttle": 0.35, "brake": 0.0,  "steer":  0.8},
    {"throttle": 0.0,  "brake": 0.65, "steer":  0.0},
    {"throttle": 0.12, "brake": 0.25, "steer":  0.0},
]
STATE_SIZE = 18
ACTION_COUNT = len(AI_ACTIONS)

BATCH_SIZE          = 64
REPLAY_CAPACITY     = 50_000
WARMUP_STEPS        = 2_000     # collect experience before the first train_step
TRAIN_EVERY         = 4
TARGET_UPDATE_EVERY = 1_000
LOG_EVERY           = 1_000
EPSILON_START       = 1.0
EPSILON_END         = 0.05
EPSILON_DECAY_STEPS = 40_000

# Checkpoint lives next to this file (ai_service/models/dqn.pt) regardless of cwd.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "dqn.pt")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def epsilon_at(step):
    frac = min(1.0, step / EPSILON_DECAY_STEPS)
    return EPSILON_START + frac * (EPSILON_END - EPSILON_START)

class Trainer: 
    def __init__(self):
        self.agent = DQNAgent(STATE_SIZE, ACTION_COUNT)
        self.buffer = ReplayBuffer(REPLAY_CAPACITY)
        if os.path.exists(MODEL_PATH):
            checkpoint = self.agent.load(MODEL_PATH)
            self.step = checkpoint["step"]
            self.performance = BestPerformanceTracker(
                checkpoint.get("best_lap_time"),
                checkpoint.get("best_reward"),
            )
            print(f"resumed from {MODEL_PATH} at step {self.step}", flush=True)
        else:
            self.step = 0           # global, persists across episodes
            self.performance = BestPerformanceTracker()
        self.last_loss = None       # most recent train_step loss (None until warmup)
        self.reward_sum = 0.0       # reward accumulated since the last log line

    def start_session(self, session_id):
        self._save_if_needed(self.performance.start_session(session_id))

    def choose_action(self, state):
        return self.agent.act(state, epsilon_at(self.step))
    
    def observe(self, msg):
        action = AI_ACTIONS[msg["actionIndex"]]
        reward = compute_rewards(msg["env"], action)
        self.buffer.push(msg["state"], msg["actionIndex"], reward, 
                         msg["nextState"], 1.0 if msg["done"] else 0.0
                         )
        self.step += 1
        self.reward_sum += reward
        save_decision = self.performance.observe(reward, msg["env"])
        if len(self.buffer) >= WARMUP_STEPS and self.step % TRAIN_EVERY == 0:
            self.last_loss = self.agent.train_step(self.buffer.sample(BATCH_SIZE))
        if self.step % TARGET_UPDATE_EVERY == 0:
            self.agent.sync_target()
        if self.step % LOG_EVERY == 0:
            mean_reward = self.reward_sum / LOG_EVERY
            loss = f"{self.last_loss:.4f}" if self.last_loss is not None else "warmup"
            print(
                f"step {self.step:>7}  eps {epsilon_at(self.step):.3f}  "
                f"buffer {len(self.buffer):>6}  mean_reward {mean_reward:+.4f}  loss {loss}",
                flush=True,
            )
            self.reward_sum = 0.0
        self._save_if_needed(save_decision)

    def finalize(self, reason):
        self._save_if_needed(self.performance.finalize_session(reason))

    def _save_if_needed(self, decision):
        if decision is None:
            return
        self.agent.save(
            MODEL_PATH,
            self.step,
            self.performance.best_lap_time,
            self.performance.best_reward,
        )
        print(
            f"saved {MODEL_PATH} at step {self.step} "
            f"({decision.reason}: {decision.metric}={decision.value:.3f})",
            flush=True,
        )

trainer = Trainer()






async def handle_client(websocket):
    print("Game connected.")
    async for raw in websocket:
        msg = json.loads(raw)
        kind = msg["type"]

        if kind == "session_start":
            assert msg["stateSize"] == STATE_SIZE,     "state vector size mismatch"
            assert msg["actionCount"] == ACTION_COUNT, "action count mismatch"
            trainer.start_session(msg["sessionId"])
            await websocket.send(json.dumps({
                "type": "session_ready", "algorithm": "dqn", "training": True,
            }))

        elif kind == "action_request":
            await websocket.send(json.dumps({
                "type": "action_response",
                "sessionId": msg["sessionId"],
                "step": msg["step"],
                "actionIndex": trainer.choose_action(msg["state"]),
            }))

        elif kind == "transition":
            trainer.observe(msg)


async def main():
    print("DQN service on ws://localhost:8765 — start the game, select External.")
    async with websockets.serve(handle_client, "localhost", 8765):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        trainer.finalize("exit")
