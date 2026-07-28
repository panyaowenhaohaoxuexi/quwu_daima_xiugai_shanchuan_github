# CoA Training Schedule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Source and EMA stages use CoA-equivalent fixed-step schedules, Adam optimization, cosine learning rates, and EMA update cadence.

**Architecture:** Add schedule options and pure schedule helpers to the existing option modules.  Drive the existing Source and EMA loss computations from cycling data iterators for configured logical steps.  Keep checkpoints and model semantics intact.

**Tech Stack:** Python 3.10, PyTorch, pytest.

---

### Task 1: Schedule option and helper contract

**Files:**
- Modify: `option/Teacher.py`
- Modify: `option/EMA.py`
- Test: `tests/test_coa_training_schedule.py`

- [ ] **Step 1: Write failing tests**

```python
def test_coa_source_and_ema_schedule_defaults():
    assert (source.epochs, source.iters_per_epoch) == (20, 5000)
    assert (source.start_lr, source.end_lr) == (1e-4, 1e-6)
    assert (ema.epochs, ema.iters_per_epoch) == (20, 1000)
    assert (ema.start_lr, ema.end_lr, ema.ema_decay) == (1e-7, 1e-8, 0.95)

def test_cosine_schedule_reaches_coa_start_and_end_rates():
    assert cosine_lr(1, 100000, 1e-4, 1e-6) < 1e-4
    assert cosine_lr(100000, 100000, 1e-4, 1e-6) == pytest.approx(1e-6)
```

- [ ] **Step 2: Run the tests and observe missing options/helper failures.**
- [ ] **Step 3: Add validated options and the shared pure cosine helper.**
- [ ] **Step 4: Re-run the tests and confirm they pass.**

### Task 2: Source fixed-step training

**Files:**
- Modify: `Teacher.py`
- Test: `tests/test_coa_training_schedule.py`

- [ ] **Step 1: Write failing tests for cycling a short loader to `iters_per_epoch` and applying Adam/cosine LR.**
- [ ] **Step 2: Run the tests and observe the current full-loader epoch behavior fail.**
- [ ] **Step 3: Replace AdamW with CoA Adam and run exactly `iters_per_epoch` cycling batches per logical epoch.**
- [ ] **Step 4: Re-run Source schedule tests.**

### Task 3: EMA fixed-step training and cadence

**Files:**
- Modify: `EMA.py`
- Test: `tests/test_coa_training_schedule.py`

- [ ] **Step 1: Write failing tests that the real and source iterators cycle, the LR reaches the CoA endpoint, and teacher parameters update only once per logical epoch.**
- [ ] **Step 2: Run tests and observe per-batch teacher update/current optimizer behavior fail.**
- [ ] **Step 3: Use CoA Adam, fixed real/source steps, cosine LR, and one `ema_decay=0.95` update after each logical epoch.**
- [ ] **Step 4: Re-run EMA schedule tests.**

### Task 4: Regression verification

**Files:**
- Test: `tests/test_teacher_end_to_end.py`
- Test: `tests/test_ema_end_to_end.py`

- [ ] **Step 1: Update smoke invocations to explicitly use one logical epoch with one step.**
- [ ] **Step 2: Run the complete test suite with `D:\\anaconda\\envs\\CoA\\python.exe -m pytest -q`.**
- [ ] **Step 3: Confirm Source and EMA checkpoint/resume tests preserve the existing checkpoint format.**
