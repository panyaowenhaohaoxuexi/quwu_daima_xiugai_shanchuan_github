"""Small, dependency-free persistence helpers for training observability."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def _json_value(value):
    """Convert common training values into stable JSON/CSV scalar values."""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numel") and value.numel() == 1:
        return value.item()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except ValueError:
            pass
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


class TrainingLogger:
    """Write a machine-readable run summary and append-only training events."""

    def __init__(self, output_dir, *, resume=False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "metrics.jsonl"
        self.csv_path = self.output_dir / "metrics.csv"
        if not resume:
            self.jsonl_path.write_text("", encoding="utf-8")
            self.csv_path.write_text("", encoding="utf-8")

    def write_run_summary(self, summary):
        with (self.output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(_json_value(summary), handle, ensure_ascii=False, indent=2, sort_keys=True)

    def log_event(self, event, **fields):
        record = {
            "event": event,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **{name: _json_value(value) for name, value in fields.items()},
        }
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._rewrite_csv()

    def _rewrite_csv(self):
        records = []
        if self.jsonl_path.is_file():
            with self.jsonl_path.open(encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
        fieldnames = sorted({name for record in records for name in record})
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            if not fieldnames:
                return
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
