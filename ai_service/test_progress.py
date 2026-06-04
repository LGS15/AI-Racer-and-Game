import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from checkpoints import SaveDecision
from progress import TrainingProgressLogger


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_interval_log_is_durable_for_notebook_progress_charts(tmp_path):
    path = tmp_path / "training_progress.csv"
    logger = TrainingProgressLogger(str(path))

    logger.log_interval(
        step=1000,
        epsilon=0.5,
        buffer_size=1000,
        mean_reward=1.25,
        loss=None,
        best_lap_time=12.5,
        best_reward=None,
    )

    rows = read_rows(path)
    assert len(rows) == 1
    assert rows[0]["event"] == "interval"
    assert rows[0]["step"] == "1000"
    assert rows[0]["mean_reward"] == "1.25"
    assert rows[0]["loss"] == ""
    assert rows[0]["best_lap_time"] == "12.5"


def test_checkpoint_log_preserves_why_the_model_was_saved(tmp_path):
    path = tmp_path / "training_progress.csv"
    logger = TrainingProgressLogger(str(path))

    logger.log_checkpoint(
        SaveDecision("new best lap time", "lap_time", 9.75),
        step=2000,
        epsilon=0.25,
        buffer_size=2000,
        loss=0.125,
        best_lap_time=9.75,
        best_reward=None,
    )

    rows = read_rows(path)
    assert len(rows) == 1
    assert rows[0]["event"] == "checkpoint"
    assert rows[0]["reason"] == "new best lap time"
    assert rows[0]["metric"] == "lap_time"
    assert rows[0]["metric_value"] == "9.75"


def test_lap_log_keeps_all_lap_times_and_marks_new_fastest(tmp_path):
    path = tmp_path / "training_progress.csv"
    logger = TrainingProgressLogger(str(path))

    logger.log_lap(
        step=3000,
        epsilon=0.1,
        buffer_size=3000,
        loss=0.25,
        lap_time=8.5,
        best_lap_time=8.5,
        best_reward=None,
        is_new_best=True,
    )
    logger.log_lap(
        step=4000,
        epsilon=0.1,
        buffer_size=4000,
        loss=0.2,
        lap_time=8.9,
        best_lap_time=8.5,
        best_reward=None,
        is_new_best=False,
    )

    rows = read_rows(path)
    assert [row["event"] for row in rows] == ["lap", "lap"]
    assert [row["metric_value"] for row in rows] == ["8.5", "8.9"]
    assert rows[0]["reason"] == "new fastest lap"
    assert rows[1]["reason"] == ""
