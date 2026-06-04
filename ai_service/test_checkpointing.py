import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from checkpoints import BestPerformanceTracker


def test_first_completed_lap_saves():
    tracker = BestPerformanceTracker()
    tracker.start_session("a")

    decision = tracker.observe(1.0, {"lapCompleted": True, "lapTime": 12.5, "elapsed": 12.5})

    assert decision is not None
    assert decision.metric == "lap_time"
    assert tracker.best_lap_time == 12.5


def test_slower_lap_does_not_save_then_faster_lap_saves():
    tracker = BestPerformanceTracker(best_lap_time=10.0)
    tracker.start_session("a")

    assert tracker.observe(1.0, {"lapCompleted": True, "lapTime": 11.0, "elapsed": 11.0}) is None
    decision = tracker.observe(1.0, {"lapCompleted": True, "lapTime": 9.0, "elapsed": 20.0})

    assert decision is not None
    assert decision.value == 9.0
    assert tracker.best_lap_time == 9.0


def test_reward_fallback_saves_best_no_lap_session_only():
    tracker = BestPerformanceTracker()
    tracker.start_session("a")
    tracker.observe(3.0, {"lapCompleted": False, "elapsed": 1.0})

    first = tracker.start_session("b")
    tracker.observe(2.0, {"lapCompleted": False, "elapsed": 1.0})
    second = tracker.start_session("c")
    tracker.observe(4.0, {"lapCompleted": False, "elapsed": 1.0})
    third = tracker.finalize_session("exit")

    assert first is not None
    assert first.metric == "reward"
    assert second is None
    assert third is not None
    assert third.value == 4.0
    assert tracker.best_reward == 4.0


def test_reward_fallback_stops_after_lap_time_exists():
    tracker = BestPerformanceTracker()
    tracker.start_session("a")
    tracker.observe(1.0, {"lapCompleted": True, "lapTime": 10.0, "elapsed": 10.0})

    tracker.start_session("b")
    tracker.observe(100.0, {"lapCompleted": False, "elapsed": 1.0})

    assert tracker.finalize_session("exit") is None
    assert tracker.best_lap_time == 10.0


def test_finalize_does_not_overwrite_better_reward_checkpoint():
    tracker = BestPerformanceTracker(best_reward=20.0)
    tracker.start_session("a")
    tracker.observe(10.0, {"lapCompleted": False, "elapsed": 1.0})

    assert tracker.finalize_session("exit") is None
    assert tracker.best_reward == 20.0


def test_lap_time_falls_back_to_elapsed_since_lap_start():
    tracker = BestPerformanceTracker()
    tracker.start_session("a")
    tracker.observe(1.0, {"lapCompleted": True, "elapsed": 8.0})
    decision = tracker.observe(1.0, {"lapCompleted": True, "elapsed": 15.5})

    assert decision is not None
    assert decision.value == 7.5
    assert tracker.best_lap_time == 7.5
