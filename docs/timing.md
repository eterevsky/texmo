# Training-run time prediction

For a given `(system, precision)` pair, we fit a small model that
predicts the **total wall-clock time** of a training run as a function
of the model spec, sample length, batch size, and step count. Search
uses these predictions to pick the largest power-of-two step count
that fits a time budget.

Code lives in [`texmo/predict/timing.py`](../texmo/predict/timing.py).
CLI: `texmo time`.

## Model

The JAX backend trains in chunks of `CHUNK_SIZE = 256` steps each
batched into a `lax.scan` (see [`backends.md`](backends.md) and
[`texmo/manager_jax.py`](../texmo/manager_jax.py)). A run with `S`
total steps does

    N_full  = S // CHUNK_SIZE
    N_short = S  % CHUNK_SIZE

full-size scans plus, if `N_short > 0`, one trailing short scan. JIT
costs are paid once per unique scan shape, so the first full and the
short trailing scan each carry a constant overhead in addition to the
per-step compute.

Total time is modelled as a non-negative linear combination over four
blocks:

    T_total = T_init  +  N_full · T_scan_full
              +  has_short · T_scan_short
              +  S · T_step

where each `T_x` is itself a non-negative linear combination of
features:

    T_init       = Σ_c <w_init[t_c],  f_c>     # per-layer features
    T_step       = Σ_c <w_step[t_c],  f_c>     # per-layer features
    T_scan_full  = <w_scan_full[t_in],  g_full>     # input-only
    T_scan_short = <w_scan_short[t_in], g_short>    # input-only + N_short

with `c` ranging over the components (input, hidden layers, output)
of the spec, `t_in` the input-layer type, and per-layer feature
vectors `f_c` chosen for the layer's cost structure (see below).

`T_scan_full` / `T_scan_short` are keyed by the input layer alone
because the cost we're modelling there is JAX dispatch / scan
bookkeeping over the int8 token tensor — what's *inside* the scan
shows up in `T_step` and the JIT cost in `T_init`.

## Fitting

A single joint NNLS (`scipy.optimize.nnls`) solve over all four blocks
of weights. Non-negativity is enforced directly (no `exp(θ)`
reparameterization), and the loss is unweighted MSE in seconds².

We optimise absolute error, not relative error: search makes
time-budget decisions where the cost of a 10× miss on a 1000s run is
far worse than a 10× miss on a 1s run. NNLS naturally puts more
weight on long runs, which matches that priority.

Runs are filtered before fitting:

- `train_time` is `None` or non-positive → skip.
- `steps < MIN_STEPS = 4` → skip.
- `train_time < MIN_TIME = 0.1 s` → skip (noise floor).

Each `(system, precision)` pair is fit independently; pairs with
fewer than `MIN_SAMPLES = 20` are skipped entirely.

## Component types

Granular enough to separate kernels with distinct cost structure:

- **Inputs** — `bytes`, `bits.1`, `bits.1+bp`, `bits.2.oh`, `bits.2.oh+bp`,
  `bits.4.oh`, `bits.4.oh+bp`, `bits.8`. One type per spec string.
- **Hidden layers** — `dense.relu`, `dense.gelu`, `dense.tanh`,
  `dense` (bare, no activation — only as a `split.mul` gate path),
  `rnn.relu`, `rnn.gelu`, `rnn.tanh`, `gru`, `mgru`, `mingru`,
  `lstm`, `latent`, `lrnn`, `suffix`, `norm`, `skip.add`, `skip.cat`,
  `split.add`, `split.mul`, `split.cat`. Activation is part of the
  type because different activations use different kernels. For a
  Model2 spec the featurizer recurses into split branches: a split
  contributes one `split.{op}` merge component plus the components of
  every layer in its branches (sum-over-branches), so the branch
  compute is accounted for the same way the old flat skip span was.
- **Output** — single `output` type, always a dense projection.

Each type carries its own pair of weight vectors (init and step
blocks share the feature schema but learn separate coefficients), so
no cross-type coupling.

## Feature vectors

### Per-layer (init / step blocks)

Same feature definitions as the previous per-step model — they vary
by layer type:

#### Base (9 features) — `norm` and similar
    [1, L, B·L, IS, IS·L, IS·B·L, OS, OS·L, OS·B·L]

#### Dense matmul (12 features) — `dense.*`, `rnn.*`, `gru`, `mgru`,
`mingru`, `lstm`, `latent`, `lrnn`, `output`

Base + three matmul terms:

    [..., IS·OS, IS·OS·L, IS·OS·B·L]

#### Suffix (10 features)

Base + the suffix length:

    [..., suffix_length]

#### Skip (4 features) — `skip.add` / `skip.cat`

    [1, IS, IS·L, IS·B·L]

#### Split (4 features) — `split.add` / `split.mul` / `split.cat`

Same cheap shape as skip, on the merged output width `OS`. Only the
combine cost — the branch compute is captured by the branches' own
per-layer components.

    [1, OS, OS·L, OS·B·L]

#### Input (3 features)

    [1, L, B·L]

### Scan-dispatch (input-only)

#### `T_scan_full` (3 features)

    [1, B, B·L]

The dispatch cost depends on the data tensor shape, not on what's
inside the scan. Keyed by input type so different bit widths can have
slightly different overhead.

#### `T_scan_short` (4 features)

Same as scan-full plus `N_short` itself, since a short scan's
overhead can grow with the number of iterations:

    [1, B, B·L, N_short]

## Unseen types

If a spec contains a type that wasn't in the training data — e.g. the
first time we see `lstm.X` on a new system — its feature contribution
is zero (the type has no entry in any of the four weight dicts).
Predictions for such configs underestimate (we silently drop the
unknown contribution), but the scheduler will then *try* such configs
rather than reject them as "too slow", and one run of training data
is enough to get a proper fit at the next refit.

## Inversion (search)

Given a configuration and a time budget `t`, search calls
`predict_max_steps(system, conf, t)` to get the largest power-of-two
`S` whose predicted total time fits in `t` (or 0 if even `S=2`
doesn't fit, or `None` if no model is fit for this pair).

The implementation rounds an analytic upper bound from
`(t − T_init) / (T_step + T_scan_full / CHUNK_SIZE)` down to a
power of two, then walks down power-of-two candidates checking the
exact total-time prediction at each.

## Data source

Runs from `DbReader`. Old per-step (PyTorch / `--no-scan` JAX) data
is not compatible with the chunked-scan decomposition; before fitting
the model on a system that previously ran in per-step mode, the
caller should rename the system in the DB so the old runs land in a
separate bucket (e.g. `whitebox` → `pre-scan-whitebox`).

## CLI

```
uv run texmo.py time \
    --system <name> \
    --spec <spec> \
    -p <precision> \
    -b <batch> \
    -l <length> \
    --steps <n>
```

Fits all `(system, precision)` pairs from the DB, reports per-pair
RMSE in seconds, and (if `--spec` is given) prints the predicted
per-step time and predicted total time for the query. If any actual
runs in the DB match the exact query, also prints their median/min/max
per-step for comparison.
