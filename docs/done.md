# Done

Completed roadmap items, newest first, grouped by month.
[`roadmap.md`](roadmap.md) carries only open work; when something
lands (or is settled without being built), its entry moves here as a
short record with pointers — not a second copy of the design docs.

## 2026-08

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
  the multiplicative sibling `bits.4.pair.K`; the follow-ups stay on
  the roadmap.

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
