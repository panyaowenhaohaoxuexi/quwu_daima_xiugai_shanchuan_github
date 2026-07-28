# CoA Training Schedule Design

## Goal

Port CoA's optimizer, fixed-step epoch schedule, cosine learning-rate schedule, and EMA update cadence into this repository without changing the RGB--TIR model, loss definitions, datasets, checkpoint schema, or the CoA repository.

## CoA reference schedule

| Stage | Logical epochs | Steps per epoch | Optimizer | Learning rate | Teacher update |
|---|---:|---:|---|---|---|
| Source | 20 | 5000 | Adam (`betas=(0.9, 0.999)`, `eps=1e-8`) | cosine `1e-4` to `1e-6` | none |
| EMA | 20 | 1000 | Adam (`betas=(0.9, 0.999)`, `eps=1e-8`) | cosine `1e-7` to `1e-8` | once per logical epoch, `alpha=0.95` |

## Design

Each project stage gets explicit `iters_per_epoch`, `start_lr`, `end_lr`, and `no_lr_sche` options.  The training loops run the configured number of logical steps rather than treating one complete traversal of the physical loader as one epoch.  Iterators restart on exhaustion, so the local FLIR and M3FD datasets can support CoA's fixed schedules regardless of their sample counts.

The learning rate is set before each optimizer step by CoA's cosine formula using the one-based global logical step and `epochs * iters_per_epoch` as the horizon.  `no_lr_sche` retains a constant `start_lr` for controlled ablations.  Source and EMA checkpoints keep their existing schema and contain the optimizer state; resume preserves the completed logical epoch and therefore continues from the correct global schedule position.

EMA retains the current real-domain loss, source anchor loss, CLIP loss, geometry consistency, and teacher freezing.  The teacher is updated once after all `iters_per_epoch` student steps with the CoA coefficient `0.95`, instead of once per batch.

## Non-goals

Do not port CoA's single-image architecture, evaluation loop, loss weights, test dataset assumptions, DataParallel wrapper, or batch size of 24.  The current RGB--TIR model and its memory requirements keep the repository's existing batch-size defaults.

