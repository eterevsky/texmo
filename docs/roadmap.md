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
  Remaining follow-ups: per-eval-slice residual accounting (the
  corpus-average constant is used today; `chunk_residual_bits` is
  ready), loss-predictor treatment for fold confs (needs a schema
  extension — coordinate with client-refit rollout), and seeding the
  search with fold confs (no ladder edges exist from bits/bytes).
  Decode policy for generation TBD (sample within group vs
  canonical head character — currently head character).

* **64-token set.** Sibling of the above with capswords processing
  to reduce (but still support) capitals: all letters + digits +
  punctuation. Lower risk, reuses existing processing.

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
  loss_prediction.md). Remaining follow-ups: benchmark refit time on
  the GPU clients (the typed-step einsum bank is GPU-shaped; if the
  5090 fits in minutes, reconsider the pi5 denylist economics), and
  consider the same treatment for timing-model refits (the other
  ~40% of model-thread time — harder, since `_refresh_estimates`
  writes predicted rows through the writer).

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
