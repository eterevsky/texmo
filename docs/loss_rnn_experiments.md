# RNN loss-predictor experiments

Log of the hyperparameter / architecture sweep for the RNN loss
predictor ([`texmo/predict/loss_rnn.py`](../texmo/predict/loss_rnn.py)).

All numbers are **val L1 (log2-space)**, with `~X%` = typical relative
error `2^L1 - 1`. Each row is the median across 5 seeds; `range` is
`max - min` across those seeds.

## Reference points

On the DB at the time of the sweep:

| Predictor                 | val L1 | ~typical |
|---------------------------|--------|----------|
| constant (train median)   | 0.1194 | 8.6%     |
| RF (2 feat)               | 0.0739 | 5.3%     |
| HistGBR (big)             | 0.0504 | 3.6%     |
| **RNN best**              | 0.0488 | 3.4%     |
| oracle (per-conf median)  | 0.0305 | 2.1%     |

## Best configuration

**`rnn(tanh, h=32, lr=0.02 cosine, steps=8000)`** — wired in as the
default RnnPredictor in `loss_cli.py`.

## Experiments

Unless noted: tanh cell, elman cell type, globals only at init,
mini-batch=1024, optax adamw, 5 seeds.

| # | Config | val L1 | ~typ | range | note |
|---|--------|--------|------|-------|------|
| 1 | baseline h=8 lr=0.01 steps=2000 const          | 0.0611 | 4.3% | 0.010 | reference |
| 2 | h=16 lr=0.01 steps=2000 const                  | 0.0585 | 4.1% | 0.008 | bigger hidden helps |
| 3 | h=32 lr=0.01 steps=2000 const                  | 0.0583 | 4.1% | 0.005 | diminishes past 16 |
| 4 | h=16 lr=0.005 steps=2000 const                 | 0.0573 | 4.1% | 0.012 | |
| 5 | h=16 lr=0.003 steps=2000 const                 | 0.0568 | 4.0% | 0.004 | |
| 6 | h=16 lr=0.001 steps=2000 const                 | 0.0675 | 4.8% | 0.005 | underfit |
| 7 | h=16 lr=0.003 steps=4000 const                 | 0.0542 | 3.8% | 0.005 | more steps help |
| 8 | h=16 lr=0.003 steps=8000 const                 | 0.0538 | 3.8% | 0.004 | diminishes past 4000 |
| 9 | h=32 lr=0.003 steps=4000 const                 | 0.0529 | 3.7% | 0.002 | matches HistGBR |
| 10| h=32 lr=0.01 steps=4000 cosine                 | 0.0505 | 3.6% | 0.002 | **cosine is the big win** |
| 11| h=32 lr=0.01 steps=8000 cosine                 | 0.0496 | 3.5% | 0.001 | |
| 12| **h=32 lr=0.02 steps=8000 cosine**             | **0.0488** | **3.4%** | 0.001 | **best** |
| 13| h=64 lr=0.02 steps=8000 cosine                 | 0.0487 | 3.4% | 0.000 | saturated |
| 14| + feat_proj=16                                 | 0.0488 | 3.4% | 0.000 | no gain |
| 15| + rnn_sub_steps=2                              | 0.0486 | 3.4% | 0.001 | no gain (noise) |
| 16| gelu cell                                      | 0.0492 | 3.5% | 0.001 | slightly worse |
| 17| GRU cell (+3× params, same shape)              | 0.0487 | 3.4% | 0.001 | no gain |
| 18| pooling=mean over valid h_t                    | 0.0504 | 3.6% | 0.001 | worse — dilutes last-layer signal |

## What moved the needle

1. Cosine LR decay — biggest single improvement.
2. Hidden size 8 → 32.
3. Steps 2000 → 8000.

## Null results (probably no need to retry until data distribution shifts)

- `feat_proj` (pre-RNN dense on layer features).
- `rnn_sub_steps=2` (two updates per layer with different weights).
- GRU cell.
- Mean pooling over all `h_t` (vs using only `h_final`).
- Activations other than tanh for the cell.
- `h > 32`.

## Open ideas to revisit

- **Stacked RNN layers** — not tried. `sub_steps=2` (similar compute
  pattern) saturates, so stacked is expected to plateau too, but
  unproven.
- `rnn_sub_steps=2` showed a hint of gain (0.0486 vs 0.0488) that's
  within seed noise today. Worth revisiting once the search covers a
  wider variety of model shapes where nonlinear per-layer capacity
  matters more.
- **Richer per-layer features**: current features = layer type +
  dims. Missing: layer position (first/last), neighbor context.
  Should matter more for longer models.
- **Iterative HistGBR** (option 3 from the earlier discussion) —
  still a fallback if RNN progress stalls.

## Round 2 — 2026-04 (276k labeled runs)

Re-tune at ~2.5× the data and broader architecture coverage from
the predicted-2nd/3rd-neighbor and (retired) hill-climb strategies.
Three knobs revisited: an extra dense head before the output, the
already-existing `feat_proj` (was a null result before), and the
mini-batch size (was a hardcoded module constant).

90/10 conf-id split as before; 5 seeds per RNN config, median +
range across seeds; HistGBR is deterministic so a single run.

### Reference points (this dataset)

| Predictor | val L1 | ~typical |
|---|---|---|
| HistGBR (big)             | 0.0527 | 3.7% |
| HistGBR (big + 2nd layer) | 0.0525 | 3.7% |
| RNN baseline (h=32 lr=0.02 cos 8k bs=1024) | 0.0511 | 3.6% |
| **RNN best (head=32.gelu + feat_proj=32 + bs=2048)** | **0.0493** | **3.5%** |

### Stage 1 — single-knob sweep

| # | Config (added on top of baseline) | val L1 | Δ | range |
|---|---|---|---|---|
| 1 | head=16.gelu | 0.0504 | -0.0007 | 0.0010 |
| 2 | head=16.tanh | 0.0509 | -0.0002 | 0.0004 |
| 3 | **head=32.gelu** | **0.0499** | **-0.0012** | 0.0003 |
| 4 | head=32.tanh | 0.0500 | -0.0011 | 0.0005 |
| 5 | feat_proj=16 | 0.0509 | -0.0002 | 0.0014 |
| 6 | feat_proj=32 | 0.0503 | -0.0008 | 0.0015 |
| 7 | feat_proj=64 | 0.0502 | -0.0009 | 0.0009 |
| 8 | bs=256 | 0.0521 | +0.0010 | 0.0009 |
| 9 | bs=512 | 0.0513 | +0.0002 | 0.0013 |
| 10 | bs=2048 | 0.0508 | -0.0003 | 0.0005 |
| 11 | bs=4096 | 0.0505 | -0.0006 | 0.0006 |

Take-aways:
- Output head with hidden width 32 is the biggest single win
  (Δ-0.0012, tight range). gelu vs tanh is a wash; activation
  picked gelu for symmetry with the existing global-init layer.
- `feat_proj` (a null result in Round 1) is now mildly useful at
  width 32–64 (Δ-0.0008/-0.0009). The bigger, more diverse
  dataset must have made the projection earn its keep.
- Small batches hurt (256 → +0.0010); larger batches help slightly,
  with diminishing returns past 2048.

### Stage 2 — combine the wins

| # | Config | val L1 | Δ from baseline | range |
|---|---|---|---|---|
| 12 | head=32.gelu | 0.0500 | -0.0007 | 0.0004 |
| 13 | + bs=2048 | 0.0497 | -0.0010 | 0.0004 |
| 14 | + feat_proj=32 | 0.0495 | -0.0012 | 0.0005 |
| 15 | + feat_proj=64 | 0.0498 | -0.0009 | 0.0016 |
| 16 | **+ feat_proj=32 + bs=2048** | **0.0494** | **-0.0013** | 0.0005 |
| 17 | + feat_proj=64 + bs=2048 | 0.0493 | -0.0014 | 0.0005 |

The three improvements stack roughly additively. feat_proj=32 vs 64
is within seed noise; picked 32 to match the head width and the
hidden state — keeping all dimensions the same makes the model
slightly easier to reason about.

### Production config

```
hidden=32, cell_activation='tanh',
lr=0.02, lr_schedule='cosine', steps=8000,
feat_proj=32, out_hidden=32, out_activation='gelu',
batch_size=2048,
```

Wired into `train_loss_model`. Persisted models from the previous
config still load — `out_hidden=0` is the dataclass default and
disables the new head.
