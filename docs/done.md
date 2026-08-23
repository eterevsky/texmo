# Done

Completed roadmap items, newest first, grouped by month.
[`roadmap.md`](roadmap.md) carries only open work; when something
lands (or is settled without being built), its entry moves here as a
short record with pointers — not a second copy of the design docs.

## 2026-08

* **Frontier seeding for sub-searches** (2026-08-23). A per-entry
  **Seed** checkbox that fixes the propagation problem `bits.4.pair.*`
  exposed the week before: a new family lands with its own share of
  the budget and is still visited three times in its first day,
  because the only way into it is a handful of bridge edges the
  neighbor walk has to stumble onto at random. While Seed is on, the
  entry is served from a queue instead of its strategy lottery: the
  unrestricted Pareto frontier, expanded by `conf_neighbors` under the
  entry's template, filtered to what the entry admits and has **zero
  recorded runs** — each run once, ahead of the coverage walk, with
  the source's metaparams riding along (so a family's head arrives
  pre-tuned, not at the sub-space's cold default) and `steps` reduced
  per requesting system by the timing model. No budget knob: the
  entry's share governs the drain rate and run-once bounds the spend.
  Recorded as strategy `frontier_seed`.

  The queue is cached — the sweep is one `conf_neighbors` call per
  frontier conf — and invalidated by the frontier moving, by draining
  once, or by an `/update`. "The frontier moved" needed a new
  cross-thread signal: `changed_winner` is computed inside the
  writer's transaction and `DbWriterProxy` drops it, so `WriterThread`
  now bumps a shared `FrontierVersion` counter the search thread
  reads. A counter rather than a queue message deliberately — the
  value coalesces, so a burst of flips costs one rebuild rather than
  one message each, and `search` imports `db`, not the reverse. See
  [`search.md`](search.md#frontier-seeding) and
  [`threads.md`](threads.md).

  The UI wrinkle worth remembering: an unchecked checkbox posts
  nothing, so `entry_seed` could not join the positionally-zipped
  `entry_preset` / `entry_share` fields — one ticked row three down
  would have landed on row 0. Each box carries its own row index as
  its value instead, kept current by the `updateShares` walk the table
  already ran on load / Add / Remove.

* **Bit/byte router — dropped for now** (2026-08-22, settled without
  being built). The MoE-style "tokenization as routed compute" idea
  (2026-08-16): a router deciding per position whether the next
  bit/byte comes from a cheap local generator or a full model pass.
  Dropped because the input side has no good construction — the
  output function is makeable, but consuming the model's own
  variable-length bit/byte emissions would need something like a
  second RNN collecting input bits, which is far-fetched. To be
  reintroduced if a better idea appears; the full entry lives in
  this file's git history.

* **`bits.4.pair.K` — the multiplicative hex-pair arm** (2026-08-22).
  The sibling the `.add` rename anticipated, built the day after it.
  Same input side and same composed-256-logit contract; the low
  nibble's conditional becomes `U @ ((V@h + b_v) ⊙ A[:, hi]) + b_u`
  — k shared channels gated per high nibble, no `W2`/`b2`/`D` — so
  `P(lo | hi)` finally depends on the hidden state. The whole
  (16 hi, 16 lo) grid is one einsum, no per-hi loop and no second
  dispatch. `(16+k)X + 33k + 32` weights; any k >= 1 parses, runs and
  is valid (k = 0 is the additive arm, not a degenerate k), with the
  power-of-two grid coming from the ladder edges rather than from a
  validity rule. Edges: `bits.4.oh+bp ↔
  .16`, the arm toggle `.add ↔ .16` at the weight-comparable k, and
  a doubling/halving k ladder down to k = 1, plus
  `tokens.32.hexbpe.oh ↔` both arms (32-wide IO on both sides, and
  hexbpe's encoding is itself a mix of whole bytes and hex digits).
  An off-grid k gets outgoing-only edges to its nearest rungs, so a
  hand-built population can migrate onto the grid.
  `.add` is byte-identical — the arms share one class and differ in
  one method. [`io.md`](io.md) kind 4 is now the family section.

  The timing model was fixed to match (2026-08-22): its single
  `output` key charged every codec's head as one `isize*osize`
  matmul, so both pair arms were billed as a 256-way dense — 5-6x
  their real cost at the widths search uses. Over-predicted step
  time feeds `predict_max_steps`, which would have quietly
  shortchanged the family's step budgets on its first frontier
  outing. `_output_component` now takes the head's mult count from
  the codec (`_head_mults`); the dense and tied heads keep exactly
  `isize*osize`, so their features — and the fitted weights that
  depend on them — are bit-for-bit unchanged.

  Follow-ups cleared from the roadmap 2026-08-22 — the open
  questions (which arm wins and where; where the k ladder settles;
  richer wiring: `bytes`, `tokens.128.fold.oh`, an emb-mode analog)
  are the search's to answer now, with Oleg watching its frontier
  debut. Registered forecast (Oleg, 2026-08-22): `.K` will almost
  always beat `.add` for some k.

* **`bits.4.pair.add` — hex-pair IO, a fourth IO kind** (2026-08-21).
  Built the same day it was proposed as "hex generative IO". One
  position per byte; input is two concatenated 16-value one-hots
  (width 32, parameter-free); output is two 16-way heads plus a
  static (16, 16) coupling matrix, composed by the codec into
  ordinary 256-way byte log-probabilities so the model contract, the
  loss, sampling and the step path are all untouched.
  `32X + 288` weights against a byte head's `256X + 256`.
  `layers/pair_codec.py`; design, cost table and the accepted
  context-free-conditional limitation in [`io.md`](io.md) (kind 4).
  Search-reachable through one toggle bridge with `bits.4.oh+bp`.
  Named `.add` (2026-08-22) for the ADDITIVE arm of the pair family
  — the bare family name `bits.4.pair` is a parse error — ahead of
  the multiplicative sibling `bits.4.pair.K` (see the entry above,
  which carries the family's follow-ups).

* **Cross-corpus rank transferability study** (2026-08-21). 35 confs
  x 3 fresh SODA runs against the DB's books3 run distributions.
  Result — books3 selection holds on SODA, with the gap-transfer
  rule and the >= 64-token caveat — recorded in
  [`findings.md`](findings.md); raw data in the machine-local
  scratch/transfer_study/ (untracked).

* **Roadmap triage — three entries dropped** (2026-08-21, settled
  without being built):
  - **Per-entry bounds for sub-searches** — niche until another
    region-targeted measurement is wanted; the worked example
    (bits.1 scaling past 4000 weights, ~0.3% of selects reaching the
    cell) stays in this entry's history via git if ever needed.
  - **Tokenset loose ends** (lossy decode policy, per-eval-slice
    residual accounting, the Unicode-char-level set) — dropped with
    the lossy-tokenset deprioritization; `chunk_residual_bits`
    remains in the code if a cross-corpus lossy eval ever appears.
  - **Investigate LittleLearner** — no standalone plans; the model
    stays a synthetic-dialog generator candidate alongside Gemma and
    Qwen (its Qwen3-architecture delta list for a hypothetical port
    lives in this entry's git history).

* **`texmo.py chat` — REPL over the step path** (2026-08-21, commits
  `eff858312`, `4a3396947`). Closes chatbot milestone 2 and the
  standalone "Rudimentary chat command" entry. Dialog format is
  `Name: utterance` with blank-line turn boundaries (`--user-name`,
  `--bot-name`, `--preamble-file`, `--temperature`); the reply streams
  token by token, holding back any partial decode a later token could
  rewrite; every session appends a JSONL transcript under
  `data/chat_logs/`. The sampling loop moved out of `continue_prefix`
  into a `sample_tokens` generator shared by `generate` and `chat`,
  and each turn re-tokenizes the whole dialog, since tokenization is
  context-dependent for multi-byte tokensets. Not done: the manifest
  `"chat"` section from the old entry — the turn format is still
  hardcoded, so that part stays open on the roadmap.

* **`select_conf` latency — closed without action** (2026-08-21).
  The entry's figures were a 2026-07-18 report already marked stale in
  its own text, and latency is currently fine, so nothing was measured
  or optimized. What did change in the meantime: the predicted-best
  BFS cap went 4096 -> 10k with sub-cap levels shuffled (2026-08-18,
  commit `c3c5adf56`), measured at ~170 us per conf end to end and
  ~1.7 s per predicted-best select. Reopen with fresh numbers if
  select latency starts hurting client utilization.

* **Folding decided against for the chatbot corpora** (2026-08-20).
  Measured that case and specials cost a contextual model only about
  0.02-0.04 b/B, which does not pay for lossy preprocessing — dialog
  corpora stay raw. `scripts/fold_corpus.py` (commit `d733bc2be`)
  folds a corpus to a fold-tokenset alphabet and stays for
  fold-tokenset work. Fold accounting itself is in [`io.md`](io.md),
  "Fold tokensets: lossy folding with honest accounting"; the
  transferability entry on the roadmap keeps the conclusion as
  context.

* **Logit soft-cap removed** (2026-08-19, commit `213c35a2e`). Logits
  are the raw head output; the `cap_logits` flag is gone, and the
  RecurrentGemma port carries Gemma's `final_softcap=30` explicitly.
  Full record — the fp32 3-way study, the convergence-neutral result
  and the miscompilation argument — in [`io.md`](io.md), "Removed: the
  logit soft-cap (2026-08-19)". Re-checking the question under bf16 is
  a separate open roadmap entry.

* **Conversation harness — chatbot milestone 1, build** (2026-08-18,
  commits `37c9220c0`, `f4881cda2`). `scripts/dialog_harness.py`, with
  `scripts/dialog_sample_conf.json` (external endpoints) and
  `scripts/dialog_sample_local_conf.json` (harness-managed servers) as
  runnable example configs, and tests in
  `texmo/dialog_harness_test.py` (there, not in `scripts/`, because
  `testpaths = ["texmo"]`). Two participants, one JSON config, dialogs
  appended as JSONL under `data/dialogs/`, one record per dialog
  carrying both participants verbatim for provenance; stop rule is a
  `min_dialog_bytes` budget with a `max_turns` cap. One command goes
  from config to dialogs: a participant gives either a `base_url` (an
  endpoint someone else runs, untouched) or a `model_path`, and the
  harness starts `llama-server` on that GGUF itself, on a free port,
  waits for /health, and always tears it down again — two seats on one
  file share a single server. Generators still run outside texmo: it
  speaks OpenAI chat-completions and imports no inference library.
  `scripts/dialog_text.py` renders a JSONL tranche as readable text.
  The milestone stays open for the rest: pick the generator model,
  generate real dialogs, and write the corpus-processing step.

## Earlier

Completions before 2026-08-10 were pruned from the roadmap rather than
logged here (commit `a24c1a9c8` did that sweep). Where one carried a
finding, the finding moved into the doc that owns the subject:
precision (fp32 vs tf32) in [`backends.md`](backends.md), first-run
loss bias in [`search.md`](search.md), the tokenset families in
[`io.md`](io.md), the tied codec in [`tied_io.md`](tied_io.md), and
the Split/Model2 representation in [`split.md`](split.md).
