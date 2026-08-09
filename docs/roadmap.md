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

* **Custom tokensets — DONE, and then some.** What started here as
  "a 32-token ambiguous letter set with honest entropy accounting"
  turned into a whole family, built 2026-07-19..2026-08-03:
  `tokens.32.fold` (lossy folding, head-frequency accounting),
  `tokens.{32,64}.shift` (shift marker at tokenization),
  `tokens.{32,64,128,256}.hexbpe` (nibble fallback + BPE merges) and
  `tokens.128.fold` (127 top bytes + uniform catch-all), all
  search-reachable and all priced in weights by the same
  one-per-stored-choice rule. **[`io.md`](io.md) is the live record**
  — sizes, residuals, weight charges and the neighbor topology all
  moved since, so trust it rather than any summary here.

  Genuinely still open from this thread:
  - **Decode policy for lossy generation**: sample within a group vs
    print the canonical head character (currently head character).
  - **Per-eval-slice residual accounting**: decided against for now
    (the corpus-average constant is exact in expectation);
    `chunk_residual_bits` is ready if a cross-corpus eval appears.
  - **A Unicode-char-level set** (chars, not bytes, behind a shift /
    escape) — sketched 2026-07-20 and still unbuilt. Note the 128
    rung is no longer empty (fold-128 and hexbpe-128 both exist), so
    this would have to earn its place against them.

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

* **fp32 vs tf32 — investigated 2026-08-08, no action needed.**
  XLA defaults fp32 matmuls to **TF32** tensor
  cores on Ampere+, so a run labelled `fp32` computes with a 10-bit
  mantissa on CUDA and a 23-bit one on CPU. On bench_jax's GRU.256 at
  a stable LR the two backends land 0.11% apart in loss (CUDA 4.8017
  vs CPU 4.8072) and CUDA is not bit-reproducible run to run; forcing
  `jax_default_matmul_precision=highest` reproduces the CPU value
  exactly (4.8072, twice).

  **The verdict: the DB shows no TF32 bias worth acting on — don't
  split `Precision`, and don't force `highest` either.** First, TF32
  is confirmed as the default on *every* CUDA machine in the fleet,
  not just the new one (`scratch/tf32_probe.py`:
  4090/Linux 2.92e-4 and 5090/Windows 2.93e-4 relative error vs
  HIGHEST; CPU exactly 0). So every 4090/5060ti row in the DB is TF32
  and every pi5/m5/mini row is true fp32, all labelled `fp32`.

  Then the paired test over confs that ran in both groups (35,464
  pairs, per-conf medians, diverged runs excluded):

  | comparison | arithmetic | median shift | sign test |
  | --- | --- | ---: | ---: |
  | CUDA vs CPU | tf32 vs fp32 | +0.028% | +4.8σ |
  | 4090 vs 5060ti | **both tf32** | +0.379% | +15.4σ |
  | macs vs pi5 | **both fp32** | +0.023% | +5.4σ |
  | m5 vs mini | **both fp32** | +0.004% | +0.9σ |

  Two same-arithmetic CPU machines reproduce a +0.02%/+5σ signature
  of their own, and the 5060ti (also TF32) sits 0.38% from the 4090 —
  13x the CPU/CUDA gap — so pooling CUDA machines is not safe. With
  the 4090 alone the picture is cleaner: it sits +0.054/+0.078/+0.077%
  above pi5/m5/mini respectively (7.5-10σ each) while the two Macs
  agree to +0.004% (0.9σ).

  **But calendar-era effects dwarf all of it.** Same conf, same
  machine, later runs score 0.3-0.6% worse on *every* machine
  (`scratch/time_drift.py`) — because the search re-runs confs that
  scored well, so a conf's first run is selected for being lucky and
  every later run regresses toward the mean. Controlling for era
  halves the 4090 effect to **+0.033%**, and it survives in only one
  era of four (+0.006%, +0.029%, −0.008%, +0.068%). That residue is
  unexplained and is *not* claimed as a TF32 measurement.

  TF32 also does not destabilize training — CPU diverged slightly
  *more* (2.89% vs 2.43% of runs, McNemar z = −9.1). Everything here
  is 1-2 orders below the 1-2% differences the search adjudicates and
  ~100x below the per-conf seed noise (sd 3.5%).

  And forcing `highest` is not free anyway. Measured on the 5090
  (2026-08-08): **+4.1%** geomean over the benchmark suite at search
  sizes (5-3000 weights, 60 entries — slower on only 28 of them, so
  near noise), but **+25-40%** on bench_jax's GRU.256 (ram 36.5 ->
  49.5, gpu 36.9 -> 51.7 ms/step). Exactly the
  dispatch-bound/matmul-bound split: tiny models never feed the
  tensor cores, so TF32 buys them nothing and costs them nothing —
  but the moment models get big enough for TF32 to matter for loss,
  it is also big enough to matter for speed. Revisit only if the
  chatbot program pushes training past ~100k weights.

  Scripts: `scratch/{tf32_probe,group_bias,era_bias,time_drift}.py`.

* **First-run bias: measured at 0.3-0.6%** (2026-08-08, a by-product
  of the TF32 investigation above). Same conf, same machine, later
  runs score 0.3-0.6% worse than earlier ones across the whole fleet
  (+2.8σ to +13.2σ, `scratch/time_drift.py`). This is regression to
  the mean and it is **by design, already mitigated**: the search
  re-runs top confs precisely to correct the selection, and
  `top_confs_*` requires `min_num_runs=2` so a single lucky run
  cannot reach a leaderboard.

  Recorded because the *magnitude* is worth knowing: the first-run
  bias is well under the 1-2% differences the search adjudicates,
  which says the existing two-run floor is adequately sized rather
  than merely directionally right. If a future change makes ranking
  more sensitive (finer frontier resolution, automated promotion on
  fewer runs), this is the number to re-check against.

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

* **Weighted multi-template search — DONE 2026-08-09.** Sub-searches
  with budget shares; specs live only in the sub-search rows, every
  other bound stays shared. Coverage is per (system, entry), the
  layer cap is filtered by each entry's own minimum layer count, and
  the default fallback returns the entry's seed so a new sub-space
  bootstraps instead of draining into the general search. See
  [`search.md`](search.md).

  Deliberately left ephemeral (revisit only if it bites): realized
  shares are in-memory counters that reset on restart, and no
  `run`-table column records which entry produced a run — per-entry
  views re-derive it by applying the regex at query time.

  One consequence worth remembering: shares are of *exploration*
  selects. `pick_me` and the warmup ladder are drawn before the
  entry and are no longer spec-restricted at all, since the base
  template now carries no spec.

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
