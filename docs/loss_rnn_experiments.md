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

Unless noted: tanh cell, vanilla cell type, globals only at init,
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
