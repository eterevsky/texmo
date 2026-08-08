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

  Open questions: X (the generator's state width) as a search
  dimension and its neighbor edges; whether the generator sees `h` at
  every bit or only at bit 0; step-mode cost (8 sequential mini-steps
  per byte — cheap in weights, not in dispatches, which is exactly
  where GPUs hurt us); and how it prices against the tied codec at
  equal weight count, since that is the current answer to the same
  problem.

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

* **32-token ambiguous letter set with honest entropy accounting.**
  DONE 2026-07-19 as the *fold* tokenset kind — see the "Fold
  tokensets" section in [`io.md`](io.md) and
  `tokens/tokens32_raw_fold.json`. Key upgrade over the sketch here:
  the within-group knowledge is priced as **model weights** (one
  stored head-character frequency per token, +32), making the b/B
  comparison to bits/bytes exact rather than merely additive.
  Search integration DONE 2026-07-19: neighbor bridges
  `bits.4.oh+bp <-> tokens.32.raw_fold.oh` and
  `bits.4.emb.X <-> tokens.32.raw_fold.emb.X` (the oh<->emb swap was
  already automatic), plus two loss-predictor globals (the tokenset's
  residual charge and log2 bytes-per-token; timing needs nothing —
  unseen input types contribute zero until the regular refit learns
  them).
  Remaining: per-eval-slice residual accounting (decided against for
  now — the corpus-average constant is exact in expectation;
  `chunk_residual_bits` is ready if a cross-corpus eval appears).
  Decode policy for generation TBD (sample within group vs
  canonical head character — currently head character).

* **64-token set.** DONE 2026-07-20 as `tokens64_shift` — went
  with in-band marker tokens (raw processing) rather than the
  capswords sibling sketched here: lowercase + digits + shift
  marker (caps, tab, six shifted-symbol pairs) + 26 top bytes + a
  hex escape for everything else. Fully lossless: residual 0, zero
  extra weights, 1.038 tokens/byte, handled by the generic DP
  tokenizer (no fold machinery). See the fold section in
  [`io.md`](io.md). Search-reachable 2026-07-20 as
  `tokens.32.raw_fold <-> tokens.64.shift` neighbors (oh and emb, at
  the same table width); predictors need nothing new — the loss
  globals resolve residual 0 / log2(bpt) from the tokenset, the
  timing model learns the input key at its next refit.
  Unicode-char-level sibling (128 tokens, chars not bytes behind the
  same shift/escape) discussed 2026-07-20, revisit later.

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

## Analysis

* **Layer audit: what works and what doesn't.** Mine the results DB
  for which layer types and motifs actually show up on or near the
  Pareto frontier (per weight bucket), which only ever ride along,
  and which never win at all. Outcomes: retire the losers (the
  relu / bits.1+bm precedent — parse-but-invalid with migration
  edges), promote the winners in append/swap priors, and settle
  pending conditionals (matGRU if matLSTM earns it).

* **Embedding scale spread.** Every tied-codec run logs
  (x, y, scale=exp(y), loss, spec) to results/emb_scale.jsonl on its
  machine. Collect the files from the fleet and analyze the spread
  of the learned input/output scale: does exp(y) stay near the
  sqrt(X) init, does it depend on domain (bits vs bytes vs tokens),
  width, or chain depth — and should the init (or a prior) change
  accordingly.

## Infrastructure

* **select_conf latency** (2026-07-18 report, pre-restart): avg
  6.5 s at 76% of server wall time; `SearchServer.select` at 171%
  (overlapping requests queue ~8 s on top). Worst inner offender:
  `_select_uncovered_top` (avg 4.4 s, max 79 s), then
  `_select_predicted_best` / `_select_top_neighbor` (~1.5–3 s each).
  Also `timing._refresh_estimates.iter_confs` at ~1 min per refresh.
  No action yet — revisit with caching / incremental candidate sets
  when it starts hurting client utilization.

* **Predictor training off the server's critical path** — DONE
  2026-07-18 for the loss model (option (b): refit jobs handed to
  clients via /select + /training_data + /submit_loss_model; see
  loss_prediction.md). Follow-ups dropped 2026-07-19 (Oleg):
  distributed loss training is enough; no GPU refit benchmark, no
  timing-model outsourcing.

* **Revisit: move loss + timing model training back to the server**
  (2026-08-08). The decision above was made when the server host had
  CPU-only JAX and a refit was a serious stall. That premise is gone:
  native CUDA on Windows (winjax) makes local refits cheap, while the
  distributed path costs a client slot per refit, a training-data
  round trip, and a more complicated flow to reason about. Measure a
  local refit first, then decide for the loss model and the timing
  model independently.

* **`texmo.py loss` is broken** (observed 2026-08-03). The offline
  loss-predictor evaluation harness dies inside sklearn's
  HistGradientBoosting binning: `ValueError: window shape cannot be
  larger than input array shape`, thrown from numpy's sliding-window
  machinery — i.e. numpy/sklearn version drift, not anything texmo
  changed (it reproduces identically on a clean env and predates
  winjax). Scope is limited to that harness and its sklearn
  baselines; the server's distributed tree-predictor refits go
  through `predict/loss_rnn.py` and are unaffected. Fix, or drop the
  sklearn baselines if they have outlived their usefulness as a
  reference point.

* **Separate fp32 from tf32 in `Precision`** (measured 2026-08-08 on
  the native-CUDA 5090). XLA defaults fp32 matmuls to **TF32** tensor
  cores on Ampere+, so a run labelled `fp32` computes with a 10-bit
  mantissa on CUDA and a 23-bit one on CPU. On bench_jax's GRU.256 at
  a stable LR the two backends land 0.11% apart in loss (CUDA 4.8017
  vs CPU 4.8072) and CUDA is not bit-reproducible run to run; forcing
  `jax_default_matmul_precision=highest` reproduces the CPU value
  exactly (4.8072, twice) at no measurable cost — 42.9 vs ~42
  ms/step, because these models are dispatch-bound rather than
  matmul-throughput-bound.

  Two questions, in order. (1) Is there a *systematic* per-backend
  bias in the results DB? Testable directly: confs that ran on both a
  CPU system (pi5/m5/mini) and a CUDA one (4090/5060ti). 0.11% is
  well under the 1-2% differences the search adjudicates, but it is a
  bias, not noise. (2) If so — add a `tf32` member to `Precision` so
  the DB records what actually ran, or force `highest` fleet-wide and
  keep a single honest `fp32`? The first forks conf identity across
  every existing row; the second is free at today's model sizes but
  may stop being free as they grow.

* **step/forward parity for consuming->stateful chains.** forward
  drops the transient outputs of consuming layers (conv/suffix,
  valid-trim), but step ticks all layers synchronously, so during the
  total_padding warm-up any downstream STATEFUL layer ingests the
  transients into its state (suffix.2-gru.4: |step-forward| ~0.33 at
  position 0, decaying over the sample). Every conv/suffix->recurrent
  model therefore trains (forward) on a slightly different function
  than it evals (forward_recurrent = step path). Pinned as strict
  xfails in model2_test's step==forward sweep. Candidate fixes,
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

* **Weighted multi-template search** (2026-08-08 idea). Run several
  templates concurrently, each with a share of the run budget — e.g.
  10% to transformer-shaped confs, 10% to a newly implemented layer
  so it gets measured properly, the rest to the unrestricted main
  search. Today the server holds exactly one `Template` and every
  `select_conf` draws from it, so a narrow interest can only be
  pursued by narrowing the whole search.

  Needs: a weighted template list (config + CLI, ideally editable
  live like the current template is); a draw at selection time;
  accounting so the shares hold *over time* rather than per request
  (a 10% template must not starve when its confs are slow, nor
  monopolize when they are fast); and web-interface work — per-
  template frontier views and share display/editing.

  Motivation: a new layer currently has to compete from cold against
  a mature general population, so it can be starved out before it is
  ever fairly measured — the search is efficient at exploitation and
  we keep paying for that at exploration time.

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

(A former "run an off-the-shelf open-source LLM" section lived here;
accomplished 2026-07 — RecurrentGemma-2B runs end-to-end on texmo:
sliding-window attention with partial RoPE, RG-LRU, dependency-free
BPE tokenizer conversion, safetensors-referencing model manifests,
bench/eval/chat-ready generation. Next-port candidates are covered
by the open-model survey bullet above.)
