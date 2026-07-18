# Loss predictor v2: structure-mirroring RNN (design, 2026-07-16)

Oleg's proposal. Status: prototyped in
`scratch/loss_tree_20260716.py`; first ablations below say the idea
works — val L1 0.0568 vs the production 0.0587 at untuned first-shot
hyperparameters. Nothing in production yet.

## First ablations (2026-07-16, 2 seeds each, production budget)

The prototype compiles each conf to a register-machine tape (see
sketch below) and toggles the three axes independently. `flat` uses
the same architecture on the production flattened sequence — the
topology-parity control.

| variant                | flat tapes | tree tapes |
|------------------------|-----------|------------|
| base                   | 0.0587 (production model) | 0.0591 |
| + skip0 (act0 → head)  | 0.0582    | **0.0570** |
| + skip0 + per-type params | 0.0580 | **0.0568** |

Reading:

- **The win is real and decomposes**: ~0.0007 from the codec-mirror
  head skip alone (it helps even on flat tapes) and a further
  ~0.0012 from true fork/merge topology — but only in combination:
  tree topology *without* the head skip is slightly worse than flat
  (0.0591), presumably because branchy paths dilute the globals that
  the head needs, and the skip restores them.
- Per-type `Layer_params` adds a small, very consistent extra
  (0.0568/0.0568 across seeds).
- Total −0.0019 vs production at d=32/dp=16 with zero tuning — the
  largest single improvement found in Round 3 (features topped out
  at −0.0008).

**5-seed confirmation of tree+skip0 (everything shared)** — the
promotion candidate after deciding to drop per-type params (keeps
~90% of the win at ~old-model+20% refit cost): val L1
0.0564/0.0566/0.0570/0.0572/0.0572, median **0.0570**, range 0.0008.

**Mutation-pair ranking** (2,232 val conf pairs differing by one
model mutation, both ends with ≥2 runs; scored on the 1,545 pairs
with |Δmedian| ≥ 0.02): inconclusive. First run: production 0.824,
tree(per-type) 0.835. Re-run a day later (live DB had grown, fresh
fits): production 0.837, tree(shared) 0.829; Spearman ~0.72–0.74 for
both throughout. Production's own score moved 1.3pp between runs, so
the metric's run-to-run noise exceeds any model difference — needs a
multi-seed paired design to resolve. The pair miner and eval live in
`scratch/loss_pairs_20260716.py`.

**Per-type `Layer_forward`** (typed weights on the step itself, the
truest form of the per-type idea; 2 seeds each):

| `Layer_params` \ `Layer_forward` | shared | per-type |
|---|---|---|
| shared   | 0.0570 (5 seeds) | **0.0554 / 0.0552** |
| per-type | 0.0568 | 0.0565 / 0.0555 |

Typed step + shared params is the clear winner: **~0.0553**, total
−0.0034 vs production. Typed params on top of the typed step add
nothing — the step subsumes them. Cost: the in-scan weight-bank
gather makes fits ~4.5× slower (~950s vs ~210s on whitebox CPU);
mitigations, in rising effort: sort instructions by type (per-type
batched matmuls instead of gathered einsum), GPU/client refits (see
roadmap — this workload is GPU-shaped, unlike the flat RNN), or fall
back to tree+skip0 (0.0570) where refit cost matters.

**Step depth — null to negative** (on the shared-step base, 2 seeds
each): stack2 (second untied dense) 0.0571/0.0563 — inside the
base's seed spread; latent2 (tied inner steps + untied separator)
0.0572/0.0573 — null; latent4 0.0584/0.0604 — clearly worse. Echoes
the flat model's `rnn_sub_steps=2` null: per-instruction depth isn't
where the capacity is needed — the typed step is. (Depth on top of
the typed step is untested; low prior given both single-axis
results.)

**Typed-step confirmation (5 seeds)**: 0.0548–0.0557, median
**0.0551**, range 0.0009 — the −0.0036 vs production holds.
**d=64 width probe**: 0.0553/0.0563 (2 seeds) at ~4.8× the fit
cost — no gain over d=32's 0.0551, width is saturated at 32, as it
always was for the flat model.

## Bottom line (2026-07-18)

| model | val L1 | refit cost (whitebox CPU) |
|---|---|---|
| production flat RNN | 0.0587 | ~3.5 min |
| tree + skip0 (all shared) | 0.0570 | ~4 min |
| **tree + skip0 + typed step** | **0.0551** | ~16 min |
| ≥2-run-conf oracle floor | ~0.034 | — |

Promoted to `texmo/predict/loss_tree.py` (register-machine compiler,
`per_type_fwd` flag, tests) and wired into the `texmo loss` zoo as
`TreePredictor` — production `train_loss_model` untouched pending
review. **Deployment decision (Oleg, 2026-07-18): the typed-step
refit cost is acceptable, but refits move to a client machine** —
see the roadmap's predictor-refit-job item, now the next infra task.

Known coarseness worth a follow-up: the step weight bank gives each
simple type its own slot but folds all extra-dim types
(suffix/conv/latent/lrnn/msr/lmgu) into one shared slot and all
three merge ops into another (op identity only via features).

**Multi-seed paired mutation-pair eval (2026-07-18)** — 5 seeds of
both models on identical data, scored on the same freshly-mined pair
set, per-seed paired deltas:

| | production | tree (typed step) | paired Δ |
|---|---|---|---|
| median | 0.828 | **0.851** | **+0.024** |
| range  | 0.003 | 0.009 | +0.016 … +0.024 |

The tree wins mutation-ranking in **all 5 paired seeds** (+2.2pp
median sign accuracy; 5/5 alone is p≈0.03). The earlier single-seed
ambiguity was noise plus the weaker (non-typed-step) variant. This
is the search-shaped metric — the structure-mirroring predictor is
better at exactly the single-mutation deltas the search consumes,
and by proportionally more than its L1 gain.

## The idea

The predictor's computation graph mirrors the evaluated model's
layer DAG. One fixed-width activation vector (8/16/32) per position
represents "the state of knowledge about the state of that layer" in
the original model; the predictor layers all share one shape but
carry (possibly) per-layer-type weights.

- **Input node** — two types, one-hot and emb. Consumes the codec
  dimensions and the training metaparameters, produces `act[0]`
  (one or two dense layers).
- **Per hidden layer** — two transformations:

      p        = Layer_params(size, length, ...)      # per-type
      act[i+1] = Layer_forward(p ⊕ act[i])            # per-type or shared

  `Layer_params` digests the layer's own dimensions; `Layer_forward`
  advances the state. Either both are per-type, or only
  `Layer_params` is per-type and `Layer_forward` is shared.
- **Splits fork for real** — the activation is duplicated into two
  channels, each branch processed independently, then:

      act[i+1] = Merge_forward(act_a ⊕ act_b)         # per-op, no params

  (The merge itself is parameter-less in the real model, so probably
  no `Layer_params` on the merge.)
- **Head** — concat the final activations with `act[0]` (the codec
  owns both ends of the real model, so the IO representation feeds
  the head directly) and apply a last dense.
- `Layer_forward` could be a plain dense or latent-recurrent
  (several internal steps) for more complex per-layer transformations.

Why it's attractive: (a) it matches the logical structure of the
thing being predicted; (b) it should be sensitive to one-or-two-layer
mutations — exactly the deltas the search asks about.

## Relation to the current model

The current loss RNN is a degenerate case: flatten the tree
(marker + main branch inline, gate branch folded into a scalar
weight feature), shared Elman cell, layer type as a one-hot inside
the feature vector, `feat_proj` as a shared `Layer_params`. The v2
generalizes along three independent axes:

1. **True topology** — real fork/merge; a `split.mul` gate branch
   becomes a first-class chain instead of a folded weight count.
2. **Per-type weights** — a per-type expert instead of one cell
   conditioned on a one-hot. Risk: data fragmentation for rare types
   (a per-type `Layer_params` with a shared `Layer_forward` is the
   cheaper middle ground — Oleg's lean too).
3. **Codec mirror at the head** — `act[0]` skip-connected to the
   output, mirroring the tied codec owning both model ends.

Each axis can be toggled independently against the current model —
that's the experiment design.

## Execution / batching sketch

Ragged trees batch poorly, but a register-machine compilation makes
this a plain scan (Oleg: batch per-layer with a switch over types;
this is one concrete form of it):

- Compile each conf to an instruction tape:
  `(type_id, params_features, src1, src2 | -1, dst)`; a fork is just
  two instructions reading the same source register, a merge reads
  two registers. Register count = max branch nesting + 1 (small).
- Pad tapes to the batch max; `jax.lax.scan` over instruction index
  with a `[B, R, d]` register file; masked steps carry through.
- With a **shared** `Layer_forward` + type one-hot inside `p`, no
  switch is needed at all — every step computes the same dense.
  Per-type weights later = an einsum-selected weight bank.

## Evaluation plan

- Primary: val L1 on the standard `conf_id % 10` split vs the
  production RNN (0.0587) and the ≥2-run-conf floor (~0.034).
- Sliced: split-containing confs (the gate-branch upgrade should
  show here first), deep chains, weight buckets.
- Mutation sensitivity (motivation (b)): collect val conf pairs that
  differ by a single mutation and measure sign/rank agreement of
  predicted vs observed loss deltas — closer to what the search
  consumes than global L1.

## Context: where the headroom is (from the Round-3 analysis)

Honest ceiling on ≥2-run confs is ~0.034 vs the model's 0.046; the
divergence tail is mostly bimodal (irreducible) with a small
supervised-classifier-recoverable part (see loss_prediction.md).
So v2 is playing for ~0.01 of L1 plus better mutation-level ranking,
which L1 only proxies.
