import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from rewards import compute_rewards, progress_delta


ACTION = {"throttle": 0.0, "brake": 0.0, "steer": 0.0}


def env(**overrides):
    base = {
        "previousProgressDistance": 100.0,
        "nextProgressDistance": 100.0,
        "totalLength": 1000.0,
        "speed": 100.0,
        "collision": False,
        "offTrack": False,
        "checkpointCompleted": False,
        "lapCompleted": False,
    }
    base.update(overrides)
    return base


def test_speed_reward_requires_forward_progress():
    stopped_progress = compute_rewards(env(), ACTION)
    backward = compute_rewards(env(nextProgressDistance=90.0), ACTION)

    assert stopped_progress == pytest.approx(-0.003)
    assert backward == pytest.approx(-0.303)


def test_checkpoint_bonus_rewards_ordered_checkpoint_progress():
    reward = compute_rewards(env(nextProgressDistance=110.0, checkpointCompleted=True), ACTION)

    assert reward == pytest.approx(1.067)


def test_lap_bonus_still_applies_on_completed_lap():
    reward = compute_rewards(env(nextProgressDistance=110.0, lapCompleted=True), ACTION)

    assert reward == pytest.approx(12.317)


def test_off_track_does_not_change_reward():
    on_track = compute_rewards(env(nextProgressDistance=110.0, offTrack=False), ACTION)
    off_track = compute_rewards(env(nextProgressDistance=110.0, offTrack=True), ACTION)

    assert off_track == pytest.approx(on_track)


def test_progress_delta_wraps_backward_crossing_start_line():
    delta = progress_delta(env(previousProgressDistance=10.0, nextProgressDistance=990.0))

    assert delta == pytest.approx(-20.0)
