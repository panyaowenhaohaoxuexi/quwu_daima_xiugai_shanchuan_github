import csv
import json
from pathlib import Path

import torch


def _events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_training_logger_writes_serializable_summary_and_union_of_event_columns(tmp_path):
    from training.observability import TrainingLogger

    logger = TrainingLogger(tmp_path, resume=False)
    logger.write_run_summary({"device": "cpu", "output_dir": Path(tmp_path), "parameter_count": torch.tensor(12)})
    logger.log_event("train_step", epoch=1, step=1, loss=torch.tensor(0.25), learning_rate=1e-4)
    logger.log_event("validation", epoch=1, psnr=20.0, is_best=True)

    summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert summary == {"device": "cpu", "output_dir": str(tmp_path), "parameter_count": 12}
    assert [event["event"] for event in _events(tmp_path / "metrics.jsonl")] == ["train_step", "validation"]
    with (tmp_path / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {"event", "loss", "learning_rate", "psnr", "is_best"}.issubset(rows[0])
    assert rows[0]["loss"] == "0.25"
    assert rows[1]["is_best"] == "True"


def test_training_logger_overwrites_new_runs_and_appends_resumed_runs(tmp_path):
    from training.observability import TrainingLogger

    TrainingLogger(tmp_path, resume=False).log_event("train_step", global_step=1)
    TrainingLogger(tmp_path, resume=True).log_event("train_step", global_step=2)
    assert [event["global_step"] for event in _events(tmp_path / "metrics.jsonl")] == [1, 2]

    TrainingLogger(tmp_path, resume=False).log_event("train_step", global_step=3)
    assert [event["global_step"] for event in _events(tmp_path / "metrics.jsonl")] == [3]
