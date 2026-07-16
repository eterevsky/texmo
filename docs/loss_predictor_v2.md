# Loss predictor v2: structure-mirroring RNN (design, 2026-07-16)

Oleg's proposal, recorded before prototyping. Status: idea + first
experiments planned; nothing in production.

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
