# Roadmap

## Grand goal

Given a parameter budget N and compute budget T, answer:
1. What is the best model architecture?
2. How should it be trained? — i.e. which metaparameters, what LR schedule,
   and potentially whether to use incremental training (adding layers
   gradually) instead of training from scratch.

## Architectures to add

* **DeltaNet.** Linear-attention with delta-rule updates:
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

* **`bits.8.gen.X` — bitwise generative IO** (2026-08-08 idea). Not a
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

* **Swish / SiLU as a third activation** (2026-08-10 idea).
  `x * sigmoid(x)` — the activation Llama-class models use, and the
  `S` in SwiGLU. The search currently offers exactly two:
  `_JAX_ACTIVATIONS` also holds `relu`, but it is retired (2026-07,
  almost strictly worse than gelu in the DB), so `is_valid` admits
  only `tanh` and `gelu` and the mutation cycle proposes only those.

  Cheap to try — `jax.nn.silu` into `_JAX_ACTIVATIONS`, the name into
  the `is_valid` whitelist, and the activation-swap cycle picks it up
  for free. Two reasons it might earn its slot rather than just
  widening the search: it is smooth and non-monotonic like gelu but
  materially cheaper (one sigmoid vs gelu's erf/tanh approximation),
  which matters because our models are dispatch- and elementwise-
  bound rather than matmul-bound; and it is the natural partner for
  the gated form the split machinery already expresses
  (`split.mul(dense.X.silu, dense.X)` is SwiGLU exactly, where today
  that reads `.gelu` and is GeGLU).

  Watch for: a third activation multiplies the per-size mutation
  fan-out, so check it does not dilute the search more than it
  contributes — the relu retirement is the precedent for cutting one
  back out. Settle it the same way, from the DB (see the layer-audit
  bullet under Analysis).

* **NoPE attention — no position encoding** (2026-08-16 idea, not
  urgent). `attn` always applies rotary embeddings; a NoPE variant
  drops them entirely. Causal LMs demonstrably work without explicit
  position encoding (the causal mask leaks position; Haviv et al.
  2022, Kazemnejad et al. 2023 — the latter finds NoPE length-
  generalizes *better*), and at our window sizes (4–16) the rotation
  may be pure overhead — we are dispatch- and elementwise-bound, and
  RoPE is weightless, so dropping it changes cost but not
  `num_weights`. To settle: the spelling (a variant flag on the attn
  spec vs a separate type) and its neighbor edge (swap with plain
  attn at equal shape), then let the search price it.

### Candidates to revisit

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

* **Save weights of Pareto-optimal models.** When a run's eval score
  is Pareto-optimal for its weight count (the server already
  computes `changed_winner` at add_run), export the model as a
  model_store JSON — a few KB inline at search sizes. Especially for
  w>1000 where retraining costs minutes. Also enables inspecting
  learned weights (embedding scale, table geometry), warm starts,
  and chat demos.

* **Rudimentary chat command.** REPL over the step path: "User: " /
  "Bot: " line prefixes, stop on newline or a character cap,
  temperature knob. The chat format lives in the model JSON (an
  optional "chat" section) so customized formats (Gemma few-shot
  prompting — note our conversion is the base model, not instruct)
  ride with the model.

* **Tokenset loose ends.** The families themselves are built and
  documented in [`io.md`](io.md); what is left:
  - **Decode policy for lossy generation**: sample within a group vs
    print the canonical head character (currently head character).
  - **Per-eval-slice residual accounting**: decided against for now
    (the corpus-average constant is exact in expectation);
    `chunk_residual_bits` is ready if a cross-corpus eval appears.
  - **A Unicode-char-level set** (chars, not bytes, behind a shift /
    escape), sketched 2026-07-20. The 128 rung is no longer empty
    (fold-128 and hexbpe-128 both exist), so it has to earn its
    place against them.

* **Coherency eval.** Working assumption to start: cross-entropy
  over the dialog training distribution is a workable proxy for
  coherency (possibly too optimistic). If it proves insufficient,
  escalate to a fixed script of probe dialogs scored automatically
  (exact/fuzzy answer match) so search/training can see progress
  toward the chatbot goal directly.

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

* **Investigate LittleLearner** (2026-08-16 idea;
  https://littlelearner-ll.github.io/). A 0.6B / 1.3B / 5B family
  trained from scratch on "LittleCurriculum", an 88B-token corpus
  distilled from FineWeb-Edu to K–5 reading level, with Base / GRPO /
  "Chatty" variants. Two angles: (a) **earmarked as the
  synthetic-dialog generator for the nano chatbot** — a model whose
  entire distribution is elementary-school English emits exactly the
  narrow, simple text a 32–64K-weight model could actually learn,
  likely a better-matched teacher than a quantized 30B generalist
  (see the synthetic-dialog entry above); (b) if the architecture
  turns out close to something texmo runs (the page doesn't say —
  check the checkpoints), a conversion à la RecurrentGemma for
  local eval/generation, but only if it doesn't require too many
  changes. First step either way: find the actual weights/license
  and the architecture config.

## Analysis

* **Layer audit: what works and what doesn't.** Mine the results DB
  for which layer types and motifs actually show up on or near the
  Pareto frontier (per weight bucket), which only ever ride along,
  and which never win at all. Outcomes: retire the losers (the
  relu / bits.1+bm precedent — parse-but-invalid with migration
  edges), promote the winners in append/swap priors, and settle
  pending conditionals (matGRU if matLSTM earns it).


## Infrastructure

* **select_conf latency.** The numbers below are a **2026-07-18
  report and are now stale** — they predate the server work that
  followed (incremental timing refresh, chunked upserts, dropping
  stale `Select` messages, refit cadence 100 -> 400). Re-measure
  before acting on them: avg 6.5 s at 76% of server wall time;
  `SearchServer.select` at 171% (overlapping requests queue ~8 s on
  top); worst inner offender `_select_uncovered_top` (avg 4.4 s, max
  79 s), then `_select_predicted_best` / `_select_top_neighbor`
  (~1.5-3 s each); `timing._refresh_estimates.iter_confs` ~1 min per
  refresh.

  Two things that have changed the picture since: a `top_confs_global`
  call still costs ~3.6 s unrestricted, and a spec regex adds ~33% on
  top (4.8 s measured 2026-08-08) because `REGEXP` is a Python
  callback per row — and multi-template search now issues those
  per-entry, multiplying the distinct query shapes any cache would
  have to hold. Revisit with caching / incremental candidate sets
  when it starts hurting client utilization.

* **step/forward parity for consuming->stateful chains.** forward
  drops the transient outputs of consuming layers (conv/suffix,
  valid-trim), but step ticks all layers synchronously, so during the
  total_padding warm-up any downstream STATEFUL layer ingests the
  transients into its state (suffix.2-gru.4: |step-forward| ~0.33 at
  position 0, decaying over the sample). Every conv/suffix->recurrent
  model therefore trains (forward) on a slightly different function
  than it evals (forward_recurrent = step path).

  **This is the only known failing behaviour in the tree** — it is
  pinned as the suite's only two xfails, and clearing them is the
  deliverable:

      model2_test.py::test_step_matches_forward_all_families
        [bits.4.oh+bp|dense.8-conv.2-rglru.2]
        [bits.1+bp|suffix.2-gru.4]

  They are strict xfails, so whichever fix lands flips them to
  passing on its own and the `xfail` markers come out with it. A
  green-with-no-xfails suite is the acceptance test. Candidate fixes,
  decision pending:
  (a) make conv/suffix causal-pad internally (Gemma-style, no
      consumption, no total_padding extras) -- a semantics change
      for existing models;
  (b) drop the synthetic no-beginning convention entirely (Oleg,
      2026-07-13): train and eval on fragments that HAVE a first
      token, and for from-scratch generation prompt with "\n" or
      "\n\n" -- removes the max-entropy padding altogether, making
      step == forward trivial and the model contract match how
      pretrained LMs (BOS) already work.

* **Per-entry bounds for sub-searches** (2026-08-10, out of the
  bits.1 scaling investigation; deprioritized 2026-08-16 — niche
  until another region-targeted measurement is actually wanted; the
  analysis below stays valid). A sub-search entry carries only a
  spec filter; `max_weights` and `train_time` stay shared with the
  main search, so an entry aimed at a *region* rather than a shape
  cannot reach it.

  Worked example: 10% of the search on `bits\.1.*` to measure scaling
  past 4000 weights produced 53 runs there out of ~10k, and one
  frontier conf. Not a run-count gate — the 4000-8000 frontier holds
  one conf at `min_num_runs=1` as well. The heavy confs are dominated
  on merit because they train short: they beat the 3.033 bar from
  4096 steps and reach **2.848** at 32768 (144 s, well inside a 480 s
  budget), but only 7 runs exist at that step count. The cause is
  compounding log-uniform draws — P(t big enough) ~19% x P(weight
  ceiling > 4000) ~15% x the 10% share ~= 0.3% of selects.

  Letting an entry override `max_weights` and `train_time` (that
  entry at `-w 4000-8000 -t 120-480`) points its whole share at the
  cell being measured — roughly 30x the sampling rate there, with no
  effect on the main search. `TemplateEntry` already owns a
  `Template`, so this is two optional bound overrides plus two form
  columns.

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
