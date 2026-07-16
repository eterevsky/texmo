# Loss prediction

For a Configuration (architecture spec + metaparameters), the loss
predictor estimates the **final eval loss** the configuration would
reach if trained. This lets the search rank candidates without
spending compute on them.

Code lives in [`texmo/predict/loss_rnn.py`](../texmo/predict/loss_rnn.py).
CLI: `texmo loss` runs a small zoo of predictors against a held-out
split and reports val L1.

## What it predicts

The target is `log2(loss)` — log2 of the final eval loss in bits per
byte, clipped to `[MIN_LOSS, MAX_LOSS]`. We predict in log-space
because losses span 1–2 orders of magnitude and a 0.1 vs 0.05 b/B
miss is much more interesting than a 5.0 vs 4.95 miss.

Reported metric: **val L1 in log2 space**. A val L1 of `0.05` means
the typical prediction is off by a factor of `2^0.05 ≈ 1.035`, i.e.
~3.5% relative error.

## Architecture

A tiny RNN whose hidden state is initialized from the configuration's
globals, then updated once per hidden layer. A pre-RNN projection
adapts the per-layer features and a two-layer head reads off the
final hidden state.

```
       globals = [log2(out_size), log2(batch), log2(length),
                  log2(steps),    log2(lr),    log2(decay)]   (6 dims)

       h_0 = gelu(W_glob · globals + b_glob)                  (h dims)

       for each hidden layer i:
           x_i = gelu(W_proj · feat_i + b_proj)               (proj dims)
           h_i = tanh(W_h · h_{i-1} + W_x · x_i + b_rnn)

       o   = gelu(W_pre_out · h_final + b_pre_out)            (out_h dims)
       y   = W_out · o + b_out                                (1 dim, log2-loss)
```

- **Hidden width** `h = 32`. Bigger sizes saturated in the sweep.
- **Pre-RNN feature projection** `feat_proj = 32`. A null result in
  Round 1; useful at the larger Round 2 dataset.
- **Output head** an extra `gelu`-activated dense `h → 32 → 1`
  (Round 2 win, ~Δ-0.001 on its own).
- **Cell** Elman-style with tanh activation. GRU added no signal.
- **Pooling** last (use `h_final`); mean pooling diluted the late-layer
  signal.
- **Padding for variable layer counts** Each batch is padded to the
  max layer count seen in training; `jax.lax.scan` carries `h_i = h_{i-1}`
  on padded steps via `jnp.where(mask, new_h, h)`.

### Per-layer feature vector

For each hidden layer:

```
[ log2(num_weights), log2(in_size), log2(out_size),
  one-hot bool per "simple" type (dense.relu, gru, lstm, ...),
  log2(suffix_length), log2(latent_reps), log2(lrnn_reps),
  skip_add_dist, skip_cat_dist ]
```

A "simple" layer is one fully described by its in/out sizes plus its
type. Layers carrying an extra parameter (suffix length, latent/lrnn
reps, skip distance) get a dedicated raw feature instead of a one-hot
bit; the non-zero raw feature implicitly identifies the type. Reps
and suffix length span orders of magnitude so they go in as `log2`;
skip distances are always small (1–4), so they go in raw.

### Model2 / splits

For a Model2 spec the layer tree is flattened into the same flat
sequence a skip produced: a split becomes a marker step followed by
its main branch (branch 0) inline; the other/gate branch folds into
the marker (its weights land in the marker's `log2(num_weights)`, so a
`pass` residual reads 0 and a dense gate reads its parameter count).

`split.add` / `split.cat` **reuse** the `skip_add_dist` /
`skip_cat_dist` slots: `skip.D` parses to a `D`-layer split, so the two
representations of one architecture produce identical features and all
skip-labeled data transfers. `split.mul` (no skip analog) gets its own
`split_mul_dist` slot. The faithful per-branch (forking) treatment is
deferred to a v2 branching predictor.

Note: changing the feature schema (this added the `split_mul_dist`
slot) invalidates any persisted `LossModel` — per the no-back-compat
policy, delete the saved `'loss'` row and let the next refit rebuild.

## Training

- **Loss** mean L1 between predicted and target log2 loss.
- **Mini-batch** 2048 (configs sampled with replacement per step).
- **Optimizer** `optax.adamw`, **lr=0.02 with cosine decay**, **8000 steps**.
- **Init** Glorot-ish for dense weights; biases zero.
- **Data** every labeled run in the DB (one example per run; the same
  conf appears multiple times if it has multiple runs, weighting it
  proportionally).

Cosine LR decay was the single biggest win in the original sweep;
the output head was the biggest win on retune. See
[`loss_rnn_experiments.md`](loss_rnn_experiments.md) for the full
experiment log.

## Performance

2026-07 snapshot (908k labeled runs, 499k confs — post tied-codec
rollout):

| Predictor                          | val L1 | ~typical | at 276k |
|------------------------------------|--------|----------|---------|
| constant (train median)            | 0.264  | 20.1%    | 0.133   |
| RandomForest (log_weights, log_data) | 0.079 | 5.7%     | 0.070   |
| HistGradientBoosting (big features)| 0.064  | 4.5%     | 0.053   |
| **RNN (production config)**        | **0.0596** | **4.2%** | 0.0494 |
| oracle (per-conf val median)       | 0.0228 | 1.6%     | 0.030   |

The oracle predicts each conf's val median from its own runs — the
typical run-to-run deviation around the per-conf median, i.e. the
irreducible noise floor for any predictor that doesn't see
training-time randomness. Note the drift between snapshots: every
predictor got worse as the conf space diversified (the constant
doubled — the loss distribution itself is much wider now), while the
oracle *improved* — runs got more repeatable. The predictor↔oracle
gap roughly doubled, which is the motivation for the Round-3 feature
work. The run-level spread has a heavy tail: median deviation is only
~0.6%, but ~2% of runs diverge (loss clipped at MAX_LOSS), and p99
deviation is ~0.48.

## Where the error lives (2026-07 decomposition)

Per-run |err| on the val split, model vs oracle, by bucket:

| bucket | share of runs | model L1 | oracle L1 | share of total L1 |
|---|---|---|---|---|
| diverged (loss ≥ MAX)  | 2.1%  | 1.06  | 0.18  | **38%** |
| loss ∈ [4, MAX)        | 79.3% | 0.042 | 0.022 | 57% |
| loss ∈ [2, 4)          | 18.6% | 0.015 | 0.010 | 5% |
| weights < 100          | 66.9% | 0.050 | 0.025 | 57% |
| weights ∈ [100, 1k)    | 20.6% | 0.041 | 0.011 | 15% |
| weights ∈ [1k, 10k)    | 11.3% | **0.127** | 0.028 | 25% |
| weights ≥ 10k          | 1.2%  | **0.191** | 0.023 | 4% |

Splitting each weight bucket by divergence settles where the gap
lives (share = fraction of total val L1):

| bucket | n | model L1 | share |
|---|---|---|---|
| w < 100, converged        | 59,872 | 0.046 | 50.8% |
| w ∈ [100, 1k), converged  | 18,401 | 0.020 | 7.0%  |
| w ∈ [1k, 10k), converged  | 9,580  | **0.020** | 3.6% |
| w ≥ 10k, converged        | 979    | 0.054 | 1.0%  |
| diverged (all weights)    | 1,912  | 0.45–1.7 | **37.5%** |

Structural findings (corrected after the ≥2-run-conf re-analysis —
the raw oracle is flattered by single-run confs, 32% of val runs,
where it predicts its own value for free):

- **On ≥2-run confs the model is ~1.4× the honest floor** (0.046 vs
  oracle 0.034) — about the same ratio as the 276k snapshot. The
  headline 0.059-vs-0.023 exaggerates the gap.
- **Converged models predict near the floor at every size** — [1k,
  10k) converged is 0.020, *better* than tiny models. The apparent
  big-model gap was divergence concentration in disguise
  (upweighting big-model training examples was correspondingly a
  null — see loss_rnn_experiments.md).
- **The divergence tail is mostly bimodal, i.e. irreducible.** Among
  diverged runs in ≥2-run confs, minority-divergence runs (some runs
  of the conf converge, some don't; model AND oracle both pay ~1)
  carry 9.4% of total L1; consistently-diverging confs the model
  genuinely misses are only 1.9%. The dominant bad motif is
  `bits.4.oh+bp|rnn.N.gelu` chains at high step counts.
- **A divergence head improves L1 bookkeeping, not medians** — now
  measured, not just bounded: a supervised classifier separates
  divergence extremely well (AUC 0.974), yet composing it with the
  model (`p̂ ≥ t → clip`) moves run-level L1 0.0587 → 0.0571 while
  the conf-level `|pred − true median|` stays flat at every
  threshold. The genuine value of p̂ is as a separate stability
  signal for search ranking (a good median with 40% divergence odds
  is information the median hides). See loss_rnn_experiments.md for
  the full three-step chase (gated head → classifier → override).
- The rest of the L1 is diffuse small errors on sub-100-weight
  models — the closest-to-saturated regime.

## Use in search

Two strategies in [`texmo/search.py`](../texmo/search.py) call the
predictor:

- **`predicted_2nd_neighbor`** — BFS depth 2 (~100 candidates) from
  the current top conf; rank by compound `median(predicted_loss,
  observed_losses...)`; walk the top 9 by run count.
- **`predicted_3rd_neighbor`** — same, BFS depth 3 (~1000 candidates,
  bigger jump from the seed).

Both adjust each candidate's `steps` to the current time budget
before scoring, since `steps` is the cheapest dimension to vary.

## Persistence and refit

The fitted `LossModel` (a small dataclass holding params, simple-type
list, and max_layers) is pickled to `<db_dir>/loss_model.pickle`
(`predict/persist.py`; rotation keeps 3 timestamped backups) so the
server doesn't have to retrain on restart. On load the server
sanity-probes the model with a trivial predict and refits on any
error, so feature-schema changes just cost one refit. The
model-training thread refits whenever the total run count crosses
`_LOSS_REFIT_EVERY = 200`.

The training takes ~30 s on a modern CPU and ~6 min on a Raspberry Pi 4.

## CLI

```
uv run texmo.py loss [--db <path>]
```

Loads all labeled runs, splits 90/10 train/val by `conf_id` (so all
runs of a conf land on the same side), trains every predictor in a
small zoo (constant, RF, HistGBR variants, RNN, oracle), and prints
val L1 + typical-error per predictor.
