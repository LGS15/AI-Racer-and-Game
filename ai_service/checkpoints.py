import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SaveDecision:
    reason: str
    metric: str
    value: float


class BestPerformanceTracker:
    def __init__(self, best_lap_time=None, best_reward=None):
        self.best_lap_time = _finite_positive(best_lap_time)
        self.best_reward = _finite_number(best_reward)
        self.session_id = None
        self.session_reward = 0.0
        self.session_steps = 0
        self.session_completed_lap = False
        self.lap_start_elapsed = 0.0

    def start_session(self, session_id):
        decision = self.finalize_session("new session")
        self.session_id = session_id
        self.session_reward = 0.0
        self.session_steps = 0
        self.session_completed_lap = False
        self.lap_start_elapsed = 0.0
        return decision

    def observe(self, reward, env):
        self.session_reward += float(reward)
        self.session_steps += 1

        if not env.get("lapCompleted"):
            return None

        elapsed = _finite_number(env.get("elapsed"))
        lap_time = _finite_positive(env.get("lapTime"))
        if lap_time is None and elapsed is not None:
            lap_time = _finite_positive(elapsed - self.lap_start_elapsed)
        if elapsed is not None:
            self.lap_start_elapsed = elapsed

        if lap_time is None:
            return None

        self.session_completed_lap = True
        if self.best_lap_time is not None and lap_time >= self.best_lap_time:
            return None

        self.best_lap_time = lap_time
        return SaveDecision("new best lap time", "lap_time", lap_time)

    def finalize_session(self, reason):
        if self.session_steps == 0:
            return None
        if self.best_lap_time is not None or self.session_completed_lap:
            return None
        if self.best_reward is not None and self.session_reward <= self.best_reward:
            return None

        self.best_reward = self.session_reward
        return SaveDecision(reason, "reward", self.session_reward)


def _finite_number(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_positive(value):
    number = _finite_number(value)
    if number is None or number <= 0:
        return None
    return number
