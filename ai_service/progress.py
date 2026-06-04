import csv
import os


FIELDNAMES = [
    "event",
    "step",
    "epsilon",
    "buffer_size",
    "mean_reward",
    "loss",
    "best_lap_time",
    "best_reward",
    "reason",
    "metric",
    "metric_value",
]


class TrainingProgressLogger:
    def __init__(self, path):
        self.path = path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._ensure_header()

    def log_interval(self, step, epsilon, buffer_size, mean_reward, loss, best_lap_time, best_reward):
        self._append({
            "event": "interval",
            "step": step,
            "epsilon": epsilon,
            "buffer_size": buffer_size,
            "mean_reward": mean_reward,
            "loss": loss,
            "best_lap_time": best_lap_time,
            "best_reward": best_reward,
        })

    def log_checkpoint(self, decision, step, epsilon, buffer_size, loss, best_lap_time, best_reward):
        self._append({
            "event": "checkpoint",
            "step": step,
            "epsilon": epsilon,
            "buffer_size": buffer_size,
            "loss": loss,
            "best_lap_time": best_lap_time,
            "best_reward": best_reward,
            "reason": decision.reason,
            "metric": decision.metric,
            "metric_value": decision.value,
        })

    def log_lap(self, step, epsilon, buffer_size, loss, lap_time, best_lap_time, best_reward, is_new_best):
        self._append({
            "event": "lap",
            "step": step,
            "epsilon": epsilon,
            "buffer_size": buffer_size,
            "loss": loss,
            "best_lap_time": best_lap_time,
            "best_reward": best_reward,
            "reason": "new fastest lap" if is_new_best else None,
            "metric": "lap_time",
            "metric_value": lap_time,
        })

    def _ensure_header(self):
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            return
        with open(self.path, "w", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, FIELDNAMES).writeheader()

    def _append(self, row):
        with open(self.path, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, FIELDNAMES)
            writer.writerow({name: _format(row.get(name)) for name in FIELDNAMES})


def _format(value):
    return "" if value is None else str(value)
