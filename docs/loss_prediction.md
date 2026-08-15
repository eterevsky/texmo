# Loss prediction

For a Configuration (architecture spec + metaparameters), the loss
predictor estimates the **final eval loss** the configuration would
reach if trained. This lets the search rank candidates without
spending compute on them.

The predictor is the **structure-mirroring "tree" model**
([`texmo/predict/loss_tree.py`](../texmo/predict/loss_tree.py)),
designed 2026-07 (Oleg) and validated against the previous flat RNN.
**Status**: in production — the typed-step tree is what the server
publishes. Refits run either on a worker (the default) or in-process
on the model thread, selected by `config.LOSS_REFIT`; see
"Persistence and refit". The flat RNN
([`texmo/predict/loss_rnn.py`](../texmo/predict/loss_rnn.py)) stays
as the fast baseline and the source of the shared feature
extraction.

## What it predicts

The target is `log2(loss)` — log2 of the final eval loss in bits per
byte, clipped to `[MIN_LOSS, MAX_LOSS]`. We predict in log-space
because losses span 1–2 orders of magnitude and a 0.1 vs 0.05 b/B
miss is much more interesting than a 5.0 vs 4.95 miss.

Reported metric: **val L1 in log2 space** on a 90/10 split by
`conf_id % 10` (all runs of a conf land on the same side). A val L1
of `0.05` means the typical prediction is off by a factor of
`2^0.05 ≈ 1.035`, i.e. ~3.5% relative error.

## Architecture

The predictor's compute graph mirrors the evaluated model's layer
DAG. Each conf is compiled to an **instruction tape over a register
file** of fixed-width activation vectors (`D_ACT = 32`): a plain
layer reads its register and overwrites it, a split *forks* (each
branch runs as its own chain from the fork register), and a merge
reads both branch registers. A batch executes as one `jax.lax.scan`
over the padded tapes — forks are two reads of one register, so no
per-branch control flow exists at runtime, and the substrate would
execute arbitrary DAGs if the model grammar ever grows them.

Per instruction:

    p       = gelu(W_p · features)                    # shared, 33→16
    act_new = tanh(W_step[type] · [p ⊕ act_src1 ⊕ act_src2])  # 80→32

- `features` are the same per-layer vectors the flat RNN used
  (log-weights, in/out sizes, type one-hots, extra-dim slots), plus
  a merge flag; `act_src2` is zeroed for plain layers.
- **The step is typed**: `W_step` is a weight bank with one slot per
  layer type — every simple type, every extra-dim type
  (suffix/latent/lrnn/msr/lmgu/conv) individually, and each merge op
  (`add`/`cat`/`mul`) separately — plus a fallback. Slots are
  assigned from `layer_type_id` at compile time.
- A `split.mul` gate branch is a real processed chain (the flat
  model folded it into a scalar weight count); branch order in the
  merge concatenation carries the value/gate roles.
- **Head skip (load-bearing)**: the readout sees
  `[act_final ⊕ act0]` where `act0 = gelu(W_glob · globals)` is the
  input/codec node — the codec owns both ends of the evaluated
  model, and the predictor mirrors that. Without this skip the tree
  is *worse* than the flat model (globals wash out over deep paths).
- Head: `gelu(dense 64→32) → dense → scalar`.
- Globals (12): output size, batch, length, steps, lr, decay, cosine
  flag, is_tied, codec IO budget, log2(total num_weights), and two
  tokenset features resolved via the tokenizer registry: the fold
  residual charge (0 for lossless sets — the additive b/B offset a
  fold conf's target carries) and log2(bytes per token) (the token
  granularity; byte context = length × bpt, log-additive with the
  length global). An earlier nbits feature was a null, but only
  because bits/bytes sets tie granularity to output size; fold sets
  (32 tokens, 1 byte each) break that correspondence.

## Training

L1 on the clipped log2 target; adamw, lr 0.02 with cosine decay,
8000 steps, mini-batch 2048; one example per labeled run. The whole
fit is a single jitted `lax.scan`; tape compilation is a one-off
Python pass (~1–2 min at 900k runs).

**Cost**: the per-instruction weight-bank gather dominates, and the
workload is GPU-shaped — a current desktop NVIDIA GPU runs the
production fit in ~1.5–2 min, an Apple-silicon laptop in ~9–20 min,
a small ARM SBC in ~2.5 h (all at ~900k runs, measured 2026-07-19
across the fleet). Measured end to end on the server host
(2026-08-10, fresh process, 1.16M labeled runs): raw DB read 5.5 s +
(spec, precision)-deduped parse 14.1 s + fit 93.3 s ≈ **2 min**. The
dedup matters: parsing every row separately costs ~3× that load. The
`per_type_fwd=False` variant (one shared step) fits several times
faster at val L1 0.0570 and is the fallback where refit cost
matters.

## Performance

2026-07 snapshot (908k labeled runs, 499k confs):

| model | val L1 | ~typical |
|---|---|---|
| constant (train median)     | 0.264  | 20.1% |
| HistGBR (big features)      | 0.064  | 4.5%  |
| flat RNN (previous default) | 0.0587 | 4.2%  |
| tree, shared step           | 0.0570 | 4.0%  |
| **tree, typed step**        | **0.0550** | **3.9%** |
| tree, typed step (confs with ≥2 runs only) | 0.0437 | 3.1% |
| oracle (confs with ≥2 runs only)       | 0.0335 | 2.3% |
| oracle (per-conf val median, all runs) | 0.0228 | 1.6% |

The all-runs oracle is flattered by single-run confs (32% of val
runs, where "the conf's median" is the run's own value — zero error
by construction); the ≥2-run rows are the honest comparison. There
the tree runs at 1.30× the noise floor (0.0437 vs 0.0335) against
the flat RNN's 1.38× (0.0461, measured a few days earlier on the
then-smaller live DB) — roughly 20% of the honest remaining gap
closed. (An earlier "roughly a third" claim mixed populations.)

**Mutation ranking** — the search-shaped metric: over 2.2k val conf
pairs differing by exactly one model mutation, sign accuracy of the
predicted loss delta is **0.851 vs the flat RNN's 0.828** (+2.2pp,
positive in 5/5 paired seeds; Spearman on deltas ~0.74). The
structure-mirroring design is better at exactly the single-mutation
deltas the search consumes.

## How we got here (ablation summary, 2026-07)

Every axis measured at the production budget, multi-seed
(scratch drivers; full log in the loss-predictor branch history):

- **Topology and the head skip pay only together**: tree without
  skip0 0.0591 (worse than flat 0.0587); flat + skip0 0.0582; tree +
  skip0 0.0570. True fork/merge is worth ~0.0012 *on top of* the
  skip, verified against a flat-tape control at architecture parity.
- **Typed step is the biggest single win** (0.0570 → 0.0551).
  Typed `Layer_params` alone gave −0.0002 and is fully subsumed by
  the typed step. The fine-grained bank (per-extra-dim-type and
  per-merge-op slots, replacing the accidental shared slots) is
  neutral on L1 (0.0550) and kept for correctness.
- **Nulls, all measured**: width d=64 (0.0553/0.0563, ~4.8× cost);
  per-instruction depth — stack2 null, blind tied-latent null,
  and the input-re-injecting latent step (K=4 + untied separator)
  clearly negative (0.0599 median; its *train* loss says the inner
  recurrence hurts optimization). `latent_step` stays in the code
  as a reference implementation.
- **Globals**: log2(total weights) wired in (−0.0008 on the *flat*
  model; the scan can't reconstruct a sum of exponentials from
  per-layer logs); bits-per-token and log2(num_mults) were nulls.
  Caveat (Oleg, 2026-07-18): the log-total is dominated by the
  largest layer, so a huge layer feeding a narrow bottleneck reads
  like a balanced model in this global. Ablated on the tree
  (2026-07-18): a complete null — with-tw 0.0562 vs no-tw 0.0562 on
  identical data (3 seeds), so the structure already carries the
  information and the bottleneck concern is moot there. Kept because
  the feature set is shared with the flat baseline, which does use
  it. Flat-model sweep history lives in
  [`loss_rnn_experiments.md`](loss_rnn_experiments.md).

## Where the error lives

Per-run |err| against the oracle, by bucket (flat-RNN analysis;
carries over — the tree improves diffusely, not by fixing one
bucket):

- **Converged models predict near the floor at every size** — e.g.
  weights ∈ [1k, 10k) converged sits at L1 0.020. An apparent
  big-model gap was divergence concentration in disguise
  (upweighting big-model training examples: null).
- **The divergence tail (2.1% of runs, ~38% of total L1) is mostly
  bimodal** — some runs of the same conf converge and some don't,
  so the L1-optimal point prediction (the conditional median) pays
  for the diverged half no matter what. A supervised classifier
  separates divergence well (AUC 0.97), but composing it in moves
  only run-level L1, not the conf-level median — divergence-
  awareness is L1 bookkeeping, not a better median (measured to a
  close 2026-07-16). p̂'s real value would be as a *separate
  stability signal* in search ranking.
- Plain L1 already targets the mixture median (for divergence
  probability p < 0.5 that's the 0.5/(1−p) quantile of the
  convergent distribution; above, the clip), so no loss-function
  correction is needed for median consistency.

## Use in search

Two strategies in [`texmo/search.py`](../texmo/search.py) call the
predictor:

- **`predicted_2nd_neighbor`** — BFS depth 2 (~100 candidates) from
  the current top conf; rank by compound `median(predicted_loss,
  observed_losses...)`; walk the top 9 by run count.
- **`predicted_3rd_neighbor`** — same, BFS depth 3 (~1000
  candidates, bigger jump from the seed).

Both adjust each candidate's `steps` to the current time budget
before scoring, since `steps` is the cheapest dimension to vary.

## Persistence and refit

The fitted model (`TreeLossModel`: params, simple-type list, flags;
the flat `LossModel` analogously) is pickled to
`<db_dir>/loss_model.pickle` (`predict/persist.py`; rotation keeps 3
timestamped backups). On load the server sanity-probes it with a
trivial predict and refits on any error, so feature-schema changes
just cost one refit.

**Two refit modes, one switch.** `config.LOSS_REFIT` (overridable
with `--loss-refit`) selects where the fit runs:

- `'workers'` (default) — the distributed flow. Every
  `_LOSS_REFIT_EVERY = 500` labeled runs the server hands ONE refit
  job to the next `/select` from a client advertising `refit=1`
  (worker-side eligibility: `config.REFIT` / `--no-refit`, so slow
  machines don't volunteer) and forgets it: no reservation, no
  timeout. A dead worker just means the next boundary issues a fresh
  job, and overlapping fits resolve by `run_count` at submission. The
  client downloads `GET /training_data` (gzipped CSV streamed off a
  raw cursor, parsed client-side with `parse_model2` deduped by
  (spec, precision)) and POSTs the pickle to `/submit_loss_model`.
- `'server'` — ModelThread fits in-process on the same cadence,
  loading via `loss_tree.load_training_data` (the same dedup, applied
  straight to `iter_labeled_runs_raw`), then persists and swaps the
  holder.

The mode gates only the **trigger**. Both endpoints, the submit
acceptance path, and the client-side job exist in either mode, so a
submission that arrives while ModelThread owns the cadence is still
accepted when it is valid and newer — which keeps switching modes on
a live fleet safe. `RunAdded` is posted once per run with a loss, so
both cadences count exactly labeled runs. The `LossRefit` message
(startup with no usable persisted model) fits in-process in both
modes; it predates the distributed flow.

Acceptance for a submitted model: it must survive a probe-predict
(guarding schema drift from a client on older code) *and* carry a
`run_count` — the training rows the fitter counted, monotonic over
the DB's life — greater than the published model's. The publish log
records fit time, grant-to-publish turnaround, and the interval since
the previous model: the numbers that say whether the cadence is
keeping up.

**Choosing.** In `'server'` mode the fit blocks ModelThread for
~2 min at 1.2M runs, so timing refits and estimate refreshes queue
behind it — that is the cost of the cadence staying at 500. It also
runs on whatever JAX platform the server process was started with
(`texmo.py` applies `config.JAX_PLATFORMS` before anything touches a
device), so the server host needs a GPU platform there for the ~2 min
figure; on CPU the same fit is minutes to tens of minutes, which is
why `'workers'` is the default. The fit peaks around 3.4 GiB of
device memory, and a standalone fit on a busy worker OOMs — hence the
client job runs inside the worker's own pool. Each handout, in turn,
costs an 11–13 s `/training_data` payload build on a request thread,
a 118 MiB download, and a client training slot.

*History*: the distributed flow landed 2026-07-18, when the server
host had CPU-only JAX and a local fit was a ~16 min stall. Native
CUDA there killed that premise and the flow was removed 2026-08-10 in
favour of the in-process fit; it came back a few days later as a
mode alongside it, since the JAX platform the server runs on is a
per-host decision rather than a permanent one.

## CLI

```
uv run texmo.py loss [--db <path>]
```

Loads all labeled runs, splits by `conf_id % 10`, trains the zoo —
constant, RF, HistGBR variants, the flat RNN (previous production
config), `TreePredictor` in both step variants — and prints val L1 +
typical error per predictor, plus the oracle bound.
