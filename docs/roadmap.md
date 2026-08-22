# Roadmap

Open work only. Anything finished (or settled without being built)
moves to [`done.md`](done.md).

## Grand goal

Given a parameter budget N and compute budget T, answer:
1. What is the best model architecture?
2. How should it be trained? — i.e. which metaparameters, what LR schedule,
   and potentially whether to use incremental training (adding layers
   gradually) instead of training from scratch.

## Architectures to add

* **DeltaNet** (sequencing open 2026-08-16 — not deprioritized, just
  deciding which blocks land before the chatbot program and which
  after). Linear-attention with delta-rule updates:
  `S_t = S_{t-1}·(I − β k_t k_t^T) + β k_t v_t^T`, with `β_t` learned
  per token. Content-dependent forgetting in key-space; equivalent
  to online L2 regression of values against keys (i.e. a closed-form
  flavour of test-time training). Parallel-trainable via the
  Yang-et-al 2024 scan. Tests the "MSR's fixed γ is what hurt"
  hypothesis directly.

* **S4D.** Simplest structured-state-space model: diagonal linear
  recurrence with complex eigenvalues, HiPPO initialisation.
  Different axis from MSR/DeltaNet — long-range memory via
  polynomial decay rather than gated/matrix state. Weight count is
  `O(d_state)` per layer; at `d_state=4` it's very competitive on
  parameter budget. Worth comparing against lrnn at 1000 weights.

* **TTT-MLP** (speculative; decide after DeltaNet). Hidden state =
  weights of a small MLP, updated each step by an inner SGD step on
  a self-supervised loss. Strictly more expressive than DeltaNet,
  but requires backprop through inner SGD at training time. At
  ~1000-weight budgets, in-place adaptation may matter more than at
  scale.

* **`bits.8.gen.X` — bitwise generative IO** (2026-08-08 idea;
  deferred 2026-08-17 to overlap with the chatbot program — a good
  fit for the stretch while synthetic dialog data generates. Design
  settled 2026-08-16: the generator is a dense-shaped cell
  `state_n = tanh(W·[h, bit_{n-1}, state_{n-1}] + b)` with the logit
  read out linearly — no activation on the logit, which would cap
  bit confidence — `h` fed at every bit, state zero-initialized;
  still open: the IO-ladder neighbor bridge). Not a
  hidden layer but a fourth IO kind (see [`io.md`](io.md)), replacing
  both ends of the model at once:
  - **Input**: the byte as 8 raw bit values — width 8, no one-hot, no
    embedding table.
  - **Output**: the byte is generated bit by bit by a small recurrent
    cell instead of a 256-way head:
    `logit_n, state_n = output_layer(bit_{n-1}, state_{n-1}, h)`,
    where `h` is the chain's last activation, `bit_{-1}` is 0.5 (or
    the last bit of the previous byte), and `state_{-1}` is zeros.

  The 8 conditional bit probabilities chain-rule into an exact
  distribution over all 256 byte values, so this stays a lossless
  whole-byte model: no residual charge, 1.0 bytes/token, none of the
  sub-byte phase bookkeeping (`+bp`) that the bits.N kinds need.

  The point is cost: both ends become `O(X)` instead of `O(256·X)`.
  The head is the single largest parameter block in a small model —
  the reason sub-byte inputs exist at all (io.md, "why sub-byte") —
  and this removes it while keeping byte-level granularity. A
  `bytes|dense.1.tanh` model spends 768 of its 769 weights on IO;
  this kind should spend a small constant.

  **X is the size of the generator's recurrent state** (`state_n`) —
  the kind's only metaparameter, and a search dimension like any
  other width.

  Open questions: X's neighbor edges (2x/half, and what it bridges
  to); whether the generator sees `h` at
  every bit or only at bit 0; step-mode cost (8 sequential mini-steps
  per byte — cheap in weights, not in dispatches, which is exactly
  where GPUs hurt us); and how it prices against the tied codec at
  equal weight count, since that is the current answer to the same
  problem.

* **HGRN2** (Qin et al. 2024, "Hierarchically Gated Recurrent
  Network"). Adds depth-dependent gating — lower layers have
  lower forget rates (longer memory), upper layers shorter. The
  gain comes mostly in multi-layer models; less obviously useful
  at our typical 1-2 hidden-layer regime, so deferred until we
  have a baseline that benefits from depth.
* **LRU** (Orvieto et al. 2023, "Resurrecting Recurrent Neural
  Networks for Long Sequences", DeepMind). Linear recurrence with
  complex eigenvalues + input-dependent gating — essentially S4D-
  with-input-dependent-gates. Revisit after S4D so we have a
  baseline structured-recurrence to compare against.

## Tiny chatbot program (2026-07 ideas)

Context: with the tied codec in search, embedded models win over
one-hot starting around w~100 (searched up to w~200). Long-term
goal: the smallest model that is minimally coherent as a chatbot —
answers "Hi / Who are you? / Do you prefer dogs or cats?" style
exchanges sensibly; stretch goal: copy-from-context ("I'm Oleg.
What's my name?" -> "Oleg"), which likely needs attention/suffix
copying and is probably the capability that decides the weight
budget. Dream target: 32-64K weights.

**Milestone order (2026-08-16):** after the architecture tweaks
(`bits.8.gen.X`):

1. **Conversation harness**: two models talk (not necessarily on
   texmo — Gemma-class and others via their own runtimes), each with
   its own system prompt; record, process, save as synthetic
   training data. The harness itself is built (2026-08-18, see
   [`done.md`](done.md)); still open for this milestone: pick the
   generator model (see the open-model survey entry), generate real dialogs, and write the processing step
   that turns them into a training corpus.
   - Optionally **force the models' hand**: constrain generation to
     a small vocabulary by masking output logits. Subword plan
     (2026-08-16, details deferred until the bridge is reached):
     allow every token that is a *substring* of valid text — cheap
     and prefix-safe by construction — then either clean up what
     comes out or drop whole responses/dialogs marked invalid.
     Grammar-constrained decoding (GBNF / outlines-style) remains
     the heavyweight alternative if leakage is worse than expected.
     The harness already reserves the slot: each participant's
     `extra` dict is merged verbatim into the request body (last, so
     it can override the sampling params), which is where
     `logit_bias` or llama.cpp's `grammar` goes — and it is recorded
     with every dialog, so a tranche always says which constraint
     produced it.
2. **Nonsense-rate eval** (2026-08-21): a big model (Gemma-class via
   the dialog harness) makes smalltalk with a texmo model; a judge
   pass marks each small-model response nonsense / not. Working goal:
   the smallest model with nonsense < 90% on simple smalltalk. Needs
   a texmo-side OpenAI-compatible endpoint so the harness can seat a
   texmo model unchanged. This metric gates the data experiments in
   milestone 3 — "does dumbed-down data help" is unanswerable
   without it. This is the middle rung of the coherency-measurement
   ladder (absorbed the former standalone "Coherency eval" entry):
   cross-entropy on the dialog distribution as the cheap proxy →
   judge-scored nonsense rate → a fixed script of probe dialogs with
   exact/fuzzy answer matching if the judge proves too coarse.
3. **Tranches** of synthetic data under different instructions and
   vocabulary limits. (Skepticism on record 2026-08-21: training may
   already extract the simplest patterns from natural SODA, and the
   TinyStories-style precedent operates at 100x our weight budgets —
   test one constrained-vocab tranche against equal-bytes SODA on the
   nonsense metric before investing further.)
4. **Loss reconnaissance**: train search-sized models on the dialog
   tranches, compare against books3 losses. NOTE: this is the
   cross-corpus eval where per-corpus residual accounting goes live
   — lossy tokensets' residual constants were computed from books3
   frequencies, and `chunk_residual_bits` (see tokenset loose ends)
   exists for exactly this.
5. **Best possible model at ~16k weights** on the chosen tranche.
6. *(optional)* **Web demo**: pure-JavaScript in-page chat with the
   model. At 16k weights this is a few tens of KB of weights and
   hand-rolled matmuls — trivially feasible; rides on the
   model-store JSON export (first entry below).

* **Save weights of Pareto-optimal models** (deprioritized
  2026-08-21: search runs on books3, chat models train on SODA — not
  transferable). When a run's eval score is Pareto-optimal for its
  weight count (the server already computes `changed_winner` at
  add_run), export the model as a model_store JSON — a few KB inline
  at search sizes. Especially for w>1000 where retraining costs
  minutes. Also enables inspecting learned weights (embedding scale,
  table geometry), warm starts, and chat demos.

* **Chat format in the model JSON.** `texmo.py chat` hardcodes the
  `Name: utterance` turn format; move it into an optional `"chat"`
  section of the model manifest, so customized formats (Gemma
  few-shot prompting — note our conversion is the base model, not
  instruct) ride with the model.

* **Synthetic dialog data from a big open model.** Generate the
  narrow dialog training distribution with the best locally-runnable
  open-weights model (30B-class, quantized, llama.cpp/vLLM — outside
  texmo infra; a texmo path is a bonus). Output-permissive licenses
  (Gemma-style terms) keep the data provenance clean, unlike
  API-generated text with no-train clauses. Tiny models want narrow,
  heavily repeated data — the generation budget is small.

* **Open-model survey (do before the synthetic-data step).** Review
  the top open-weights models beyond Gemma — DeepSeek, GLM, current
  HF leaderboard — for (a) the synthetic-data generator pick, (b)
  architecture worth porting down (DeepSeek's latent attention /
  KV compression; the SmolLM/MobileLLM small-model tricks like layer
  sharing), (c) the next full-fidelity port candidate (Llama-class
  needs SiLU + a GQA knob + full rotary).

## Analysis

* **Re-check logit capping under bf16** (2026-08-21, waits on run
  mass). The cap was removed after the fp32 3-way study found it
  convergence-neutral (io.md records the decision), but bf16 — just
  enabled in the search template — has an 8-bit mantissa and a very
  different rounding/overflow regime, so a soft-cap could matter
  there where it did not in fp32. Once enough bf16 runs (and bf16
  divergences) accumulate, re-run the nocap-vs-cap comparison on
  bf16 divergent confs before considering the question closed for
  that precision. The fp32 harness pattern lives in the machine-local
  scratch/cap_study (untracked; rebuildable from the io.md
  description if lost).

* **Layer audit: what works and what doesn't.** Mine the results DB
  for which layer types and motifs actually show up on or near the
  Pareto frontier (per weight bucket), which only ever ride along,
  and which never win at all. Outcomes: retire the losers (the
  relu / bits.1+bm precedent — parse-but-invalid with migration
  edges), promote the winners in append/swap priors, and settle
  pending conditionals (matGRU if matLSTM earns it).


## Infrastructure

* **Spec-only "potential loss" predictor** (2026-08-21 idea, not
  started). Predict the best loss a *spec* could reach, marginalizing
  out the metaparameters, so selection can run in two stages under a
  weight budget W: first pick the most promising architecture, then
  separately optimize (lr, batch, length, steps) for it. Motivation:
  saturation time keeps growing (top confs at ~5 min of training,
  10-min limit), so spending runs on metaparameter wiggling of
  unpromising specs is the expensive part. v0 needs no new model:
  the tree predictor already conditions on metaparams, so "potential
  loss" is a min over a small metaparam grid of its predictions —
  distill into a true spec-only model later only if the grid is too
  slow or noisy at select time.

* **Exclusion regexes for sub-searches** (2026-08-15 idea;
  deprioritized 2026-08-16 — the top confs have since diversified on
  their own; revisit if one shape re-colonizes the leaderboard). A
  lot of the top confs cluster around one structure — a byproduct of
  neighbor search: whatever leads the frontier gets mutated the most,
  so its shape colonizes the leaderboard whether or not the lead is
  structural. The mirror of the positive sub-searches: give a % of
  runs an *exclusion* regex, so that share of the search explores
  only confs that do NOT match the currently dominant shape.

  The sub-search machinery already fits — a `TemplateEntry` carries a
  regex, a share, and a default; an include/exclude toggle per entry
  (one flag on the entry, one checkbox column in the web form) turns
  the same plumbing into "70% unrestricted, 10% transformers, 10%
  anything-but-`mingru`". Things to settle: coverage keying for a
  negated entry (per-entry coverage should just work, the matching
  conf set is well-defined); the entry default must itself not match
  the excluded regex (the existing default-vs-regex validation,
  inverted); and neighbor generation stays unfiltered — only
  selection is gated, same as positive entries today.

* **Split `is_valid` into invalid vs search-ineligible.** Today one
  flag conflates "this model cannot run / makes no sense" with "this
  model runs fine but search shouldn't propose it" (off-whitelist
  bits variants, non-power-of-2 embedding widths like the load-only
  `tokens.256000.gemma.emb.2560`). Two predicates — `is_valid` and
  something like `in_search_space` — would let reports/eval treat
  load-only models as first-class while keeping the search grammar
  tight.

## Training procedures

* **Progressive layer-wise training.** For an N-hidden-layer model,
  the schedule is `2N − 1` phases: N "growth" phases (add l_i with
  prior layers frozen, fresh output projection) interleaved with
  `N − 1` "joint" phases (unfreeze everything trained so far, train
  jointly). Total step count: `(2N − 1) · 2^K` for tunable K.
  Encoded as one bit (`progressive: bool`) on `Configuration`.
  Search neighbor: progressive at `(2N − 1) · 2^K` steps is a
  neighbor of non-progressive at `2^M` steps where
  `2^M ≤ (2N − 1) · 2^K < 2^{M+1}`. An older version of this
  manager implemented something similar with stacked GRUs and
  reached lower loss than joint training, supporting the bet that
  small-model joint optimisation gets stuck in local minima that
  layer-wise training avoids.
