# Training-step time prediction

For a given `(system, precision)` pair, we fit a small model that
predicts per-step training time as a function of the model spec,
sample length, and batch size. This feeds future scheduling decisions
(e.g. skipping configurations likely to time out on a given system).

Code lives in [`texmo/predict/timing.py`](../texmo/predict/timing.py).
CLI: `texmo time`.

## Model

Predicted time is an **additive, linear-in-features** sum over the
components of a model (input, each hidden layer, output):

    T_pred = Σ_c  <exp(θ_{c.type}), f_c>

where
- `c` ranges over the components of the spec,
- `c.type` is the component's type identifier (see below),
- `θ_t` is a weight vector specific to that type,
- `f_c` is a feature vector computed from this component's shape
  (input size, output size, batch, length).

The `exp(·)` ensures every contribution is non-negative (we never
predict negative time).

### Why is this "linear in features" when a component's time has
multiplicative structure (e.g. `IS · OS · B · L`)?

Because the nonlinearities are **baked into the features themselves**.
A dense layer's feature vector already contains the precomputed
product `IS · OS · B · L`; the model just learns a weight for it. Two
dense layers in a row contribute two separate `IS·OS·B·L` values,
summed in the packed feature array, and then weighted once. This is
algebraically equivalent to summing per-layer time and matches the
physical intuition that total time ≈ sum of per-layer times.

## Component types

Types are granular enough to separate kernels with distinct cost
structure:

- **Inputs** — `bytes`, `bits.1`, `bits.1+bp`, `bits.2.oh`, `bits.2.oh+bp`,
  `bits.4.oh`, `bits.4.oh+bp`, `bits.8` — one type per spec string.
- **Hidden layers** — `dense.relu`, `dense.gelu`, `dense.tanh`,
  `rnn.relu`, `rnn.gelu`, `rnn.tanh`, `gru`, `mgru`, `mingru`,
  `lstm`, `latent`, `lrnn`, `suffix`, `norm`, `skip.add`, `skip.cat`.
  Activation is part of the type because different activations use
  different kernels (cuDNN vs Python loop, in particular).
- **Output** — a single `output` type, always a dense projection with
  no activation.

Each type carries its own weight vector, so no cross-type coupling.

## Feature vectors

Each type has a feature set chosen for its cost structure:

### Base (9 features): used for `norm` and similar

    [1, L, B·L, IS, IS·L, IS·B·L, OS, OS·L, OS·B·L]

where IS = input size, OS = output size, B = batch, L = length.

### Dense matmul (12 features): for `dense.*`, `rnn.*`, `gru`,
`mgru`, `mingru`, `lstm`, `latent`, `lrnn`, and `output`

Base + three matmul terms:

    [..., IS·OS, IS·OS·L, IS·OS·B·L]

### Suffix (10 features)

Base + the suffix length itself as a raw feature (useful because
suffix is a memory-rearrangement operation, not a matmul):

    [..., suffix_length]

### Skip (4 features): for `skip.add` / `skip.cat`

Minimal set — skip is a near-free merge op:

    [1, IS, IS·L, IS·B·L]

### Input (3 features)

No meaningful IS for an input layer, and OS is fixed per type (it's
determined by the spec string, e.g. `bits.1+bp` always has OS=4):

    [1, L, B·L]

## Fitting

- **Loss**: mean squared log-relative error,
  `mean((log T_pred − log T_actual)²)`.
  Log-space puts a 20 ms vs 10 ms prediction error on the same scale
  as a 2 s vs 1 s error, which is what we want for a model that
  spans orders of magnitude.
- **Parameterization**: `w = exp(θ)` with unconstrained θ. Avoids the
  sign-symmetry of `w²`.
- **Optimizer**: `optax.adamw` with `lr=0.05`, 2000 steps, no weight
  decay.
- **Init**: θ = −14 for all features, so initial contributions are
  ~1e-6. Small but nonzero, so the optimizer can grow any feature
  that turns out to matter.

### Self-regularization

You might expect features that don't carry signal to stay at init.
They don't — they get pushed to very-negative θ (observed values
around -100 after fitting on real data) because including them would
systematically over-predict. The log-space MSE has a gradient that
pulls them down, because the pred is ALL the features' contributions
summed, and any feature that adds without explaining variance makes
the prediction too large.

This applies to features *within* a seen type. Types that don't
appear in training data at all get zero gradient — those θ stay at
init. See the next section.

## Unseen types

If a spec references a component type that wasn't in the training
data for this `(system, precision)` — e.g. the first time we see
`lstm.X` on a new system — its gradient during fit is identically
zero (features are zero for every sample), so θ stays at init. A
nonzero-but-small `exp(θ)` multiplied by a large `IS·OS·B·L` can yield
a nonsense prediction.

**Behavior**: `predict_time` contributes **0** for types not present
in the weights dict. This biases predictions slightly low (we
underestimate configs containing an unknown type) but has the
desirable effect that the scheduler will try such configs rather
than reject them as "too slow". One run is enough to get real data
and a proper fit at the next refit.

## Data source

Runs from `ResultDB`. For each `(conf, run)` pair:

- Skip if `run.train_time` is None or `conf.steps < 4`.
- `per_step = run.train_time / (conf.steps - 1)` — the cumulative
  `train_time` excludes the JIT-compile-dominated first step, so
  dividing by `steps - 1` gives the steady-state per-step time.
- Skip if `per_step < 1 ms` (noise floor).

Group by `(run.system, conf.precision)`; skip pairs with fewer than
20 samples.

## Fit quality

On the current DB (mix of systems and precisions, up to ~15k samples
per `(system, precision)` pair), log-MSE is typically 0.03–0.12,
corresponding to ~20–40% RMS relative error. The fit is better for
large datasets and worse for smaller ones.

This is a coarse but useful model: for a spec whose type mix is well
represented in the training data, predictions land within a factor
of ~1.1–1.5 of the median actual time. Improving it further likely
needs more features (per-precision hardware utilization, memory-
bandwidth terms, etc.) or a nonlinear model.

## CLI

```
uv run texmo.py time \
    --system <name> \
    --spec <spec> \
    -p <precision> \
    -b <batch> \
    -l <length>
```

Fits all `(system, precision)` pairs from the DB, reports per-pair
fit loss, and (if `--spec` is given) prints the predicted per-step
time for the query. If any actual runs in the DB match the exact
query, also prints their median/min/max for comparison.
