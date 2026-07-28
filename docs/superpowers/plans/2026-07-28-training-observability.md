# Training Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist startup diagnostics and Teacher/EMA training and validation metrics in JSONL and CSV while preserving all existing training semantics.

**Architecture:** Add a standard-library `TrainingLogger` responsible only for run metadata and event persistence. Teacher and EMA build a summary after data/model initialization, write each logical training event, and write each validation event; console messages format the same values without becoming a data source.

**Tech Stack:** Python standard library (`csv`, `json`, `datetime`, `pathlib`), PyTorch training entry points, pytest.

---

### Task 1: Define and test the persistent event writer

**Files:**
- Create: `training/observability.py`
- Test: `tests/test_training_observability.py`

- [ ] **Step 1: Write failing tests for new and resumed log files.**

```python
logger = TrainingLogger(tmp_path, resume=False)
logger.write_run_summary({"device": "cpu", "parameter_count": 12})
logger.log_event("train_step", epoch=1, step=1, loss=0.25)
logger.log_event("validation", epoch=1, psnr=20.0, is_best=True)
assert [json.loads(line)["event"] for line in (tmp_path / "metrics.jsonl").read_text().splitlines()] == ["train_step", "validation"]
assert "loss" in (tmp_path / "metrics.csv").read_text()
assert json.loads((tmp_path / "run_summary.json").read_text())["parameter_count"] == 12
```

- [ ] **Step 2: Run the test and verify it fails because `TrainingLogger` does not exist.**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests/test_training_observability.py -q`

- [ ] **Step 3: Implement a minimal standard-library event writer.**

```python
class TrainingLogger:
    def __init__(self, output_dir, *, resume): ...
    def write_run_summary(self, summary): ...
    def log_event(self, event, **fields): ...
```

It normalizes `Path`, tensors and NumPy scalars to JSON-compatible values; rewrites CSV with the union of event columns so later loss fields are retained.

- [ ] **Step 4: Run the test and verify it passes.**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests/test_training_observability.py -q`

### Task 2: Instrument Teacher startup, step progress and validation

**Files:**
- Modify: `Teacher.py`
- Modify: `tests/test_teacher_end_to_end.py`

- [ ] **Step 1: Add a failing end-to-end assertion for `run_summary.json`, `metrics.jsonl`, and validation/train events.**

```python
events = [json.loads(line) for line in (checkpoint_dir.parent / "metrics.jsonl").read_text().splitlines()]
assert {event["event"] for event in events} == {"train_step", "validation"}
assert json.loads((checkpoint_dir.parent / "run_summary.json").read_text())["stage"] == "source"
```

- [ ] **Step 2: Run the targeted Teacher test and verify the new assertions fail.**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests/test_teacher_end_to_end.py -q`

- [ ] **Step 3: Instantiate `TrainingLogger` once after model/data startup and write source events.**

The summary contains stage, device, seed, dataset lengths, batch/worker configuration, logical epoch/iteration counts, RGB/TIR/total parameter counts, Res2Net checkpoint path and `res2net_pretrained_loaded=True`, plus output directories. Each step writes all scalar terms returned by `source_training_step`, LR and duration. Validation writes PSNR, SSIM, best flag and LR.

- [ ] **Step 4: Run the targeted Teacher test and verify it passes.**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests/test_teacher_end_to_end.py -q`

### Task 3: Instrument EMA startup, step progress and validation

**Files:**
- Modify: `EMA.py`
- Modify: `tests/test_ema_end_to_end.py`

- [ ] **Step 1: Add failing end-to-end assertions for EMA summary and event contents.**

```python
events = [json.loads(line) for line in (checkpoint_dir.parent / "metrics.jsonl").read_text().splitlines()]
assert any(event["event"] == "train_step" and "L_clip" in event for event in events)
summary = json.loads((checkpoint_dir.parent / "run_summary.json").read_text())
assert summary["stage"] == "ema"
assert summary["clip_resources"]["rn101_loaded"] is True
```

- [ ] **Step 2: Run the targeted EMA test and verify the new assertions fail.**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests/test_ema_end_to_end.py -q`

- [ ] **Step 3: Instantiate `TrainingLogger` after CLIP/data/model startup and write EMA events.**

The summary includes source/real/validation lengths, student/teacher parameter counts, checkpoint source, and the four initialized resources: Res2Net, ViT-B/32, RN101 and haze prompt. Each step records real, anchor and CLIP loss components, total loss, LR and duration. Validation records student validation metrics and best flag.

- [ ] **Step 4: Run the targeted EMA test and verify it passes.**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests/test_ema_end_to_end.py -q`

### Task 4: Verify regression safety

**Files:**
- Test: `tests/`

- [ ] **Step 1: Run the full test suite.**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest -q`

- [ ] **Step 2: Inspect generated test artifacts only through test temporary directories and report the passing count.**
