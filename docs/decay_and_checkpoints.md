# Decay, Checkpoints, and Cross-System Training

## Problem

Two features partly contradict each other:

### 1. Intermediate checkpoints

When running on a mix of machines with very different speeds (1s vs 100s for
the same config), it's useful to:
- Record intermediate checkpoints at power-of-2 steps (when elapsed time > 1s)
- Use these to avoid re-running slow configs on slow machines
- `_select_untimed` in search picks good configs from one system and tries
  them on another, skipping if a lower-step variant already took too long

### 2. LR decay

Currently: decay is defined as the ratio between final and initial LR, stored
as 1/2^n. Exponential decay is applied over the total number of steps.

**The conflict:** An intermediate checkpoint from a decayed run doesn't
correspond to a valid configuration, because the effective decay ratio at step
s < S is R^(s/S), which generally isn't 1/2^n.

## Options considered

### Option 1: Drop `_select_untimed` (CHOSEN)
Don't try to run configs cross-system based on timing estimates. Each machine
explores independently. Multiple machines still help (shared DB, shared search
state). No timeouts, no checkpoints, no partial-run validity issues.

Decay can be defined however we want.

### Option 2: Per-step decay
Define decay as (1 - 1/2^n) per step instead of per run. Any partial run is
valid. But: no natural way to represent "no decay" (decay=1), and search over
decay values can diverge (always improving as n grows).

### Option 3: Train-time estimation model
Build a model that predicts training time for a given (spec, system, steps)
triple. Avoids needing checkpoints — just predict whether a config will be too
slow. Can be added later as a separate concern.

### Option 4: No decay at all
Simplest, but we've seen decay improve results.

### Option 5: Decay-aware checkpoints
Only record checkpoints at steps where the effective decay ratio is valid
(e.g., at step S/2 of a run with ratio 1/4, effective ratio is 1/2). Works
for power-of-2 steps and ratios, but constrains which checkpoints are useful.

## Decision

Option 1 for now. Option 3 can be added later if cross-system optimization
becomes important again.
