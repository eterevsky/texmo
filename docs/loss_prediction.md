# Loss prediction

For a Configuration (architecture spec + metaparameters), the loss
predictor estimates the **final eval loss** the configuration would
reach if trained. This lets the search rank candidates without
spending compute on them.

The predictor is the **structure-mirroring "tree" model**
([`texmo/predict/loss_tree.py`](../texmo/predict/loss_tree.py)),
designed 2026-07 (Oleg) and validated against the previous flat RNN.
**Status**: in production — `loss_tree.train_loss_model` (typed
step) is what the server's model thread fits and publishes. Refits
currently run in-process on the server (~16 min on whitebox CPU, an
accepted interim cost); moving them to a client-side job is the next
infrastructure task (roadmap). The flat RNN
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

**Cost** (~900k runs): the per-instruction weight-bank gather
dominates, and the workload is GPU-shaped — measured in production
client-refit jobs (2026-07-19): **4090 2m12s**, m5 8m36s, mini
19m28s, whitebox in-process ~16m, pi5 2h27m (keep `REFIT = False`
there — a 2.5h turnaround guarantees its submission arrives stale
and the fit is wasted). Download is ~10s and the client's
(spec, precision)-deduped parse 3–17s, vs ~2m50s for the old
server-side load. The `per_type_fwd=False` variant (one shared step)
fits in ~4 min on whitebox at val L1 0.0570 and is the fallback
where refit cost matters.

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

**Refits run on clients** (2026-07-18). Every
`server._LOSS_REFIT_EVERY = 1000` labeled runs, ONE refit job is
handed to the next `/select` from a client advertising `refit=1`
(worker-side eligibility: `config.REFIT` / `--no-refit`; note the
default is True even for config.py files that predate the flag, so
slow machines like the Pi must opt out explicitly — the criterion is
that a worker's grant-to-publish turnaround must stay well under the
typical grant interval, or its submissions lose the run-count race
and the fit is wasted) and forgotten — no reservation, no
timeout: a dead worker just means the next 1000-run boundary issues
a fresh job (run-count cadence is the natural staleness clock), and
overlapping fits resolve by run count at submission. The client
downloads `GET /training_data` (gzipped CSV streamed off a raw
cursor — no server-side parsing; the client dedups `parse_model2` by
(spec, precision)), fits the production config, and POSTs the pickle
to `/submit_loss_model` tagged with the snapshot's run count. The
server accepts it only if newer than the published model *and* it
survives a probe-predict (guarding schema drift from clients on
older code), then persists + atomically swaps the holder. The
publish log records fit time, grant-to-publish turnaround, interval
since the previous model, and runs-behind — the numbers that say
whether refit cadence is keeping up. The in-process fit remains only
as the startup fallback when no persisted model exists.

## CLI

```
uv run texmo.py loss [--db <path>]
```

Loads all labeled runs, splits by `conf_id % 10`, trains the zoo —
constant, RF, HistGBR variants, the flat RNN (previous production
config), `TreePredictor` in both step variants — and prints val L1 +
typical error per predictor, plus the oracle bound.
