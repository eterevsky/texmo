# Findings

Important empirical results that keep informing decisions — the
knowledge, not the work log ([`done.md`](done.md) records the work;
entries here are the results worth re-reading a year later). Dated,
newest first. Full data locations are noted per entry; scratch/ paths
are machine-local and untracked.

## Chatbot eval baseline: 8k is far below the bar (2026-08-27)

First full scripted-examiner eval (1000 seeds x T 0.3/0.5/0.7,
15,000 graded answers, anchors clean, zero judge errors).
`hb32-8k-1ub` (hexbpe-32, 8k weights, User/Bot SODA): a ~ 1.2-1.8%,
b ~ 0.6-1.2%, c ~ 0.1-0.2% against targets 90/90/50. T=0.5 best on
b (1.2%, a 2-3 s.e. lead) — matches the earlier hand impression;
repetition ~ 0 (the model rambles, it does not loop). The sharpest
structure: position 1 outscores positions 2-5 by 3-10x (b 4.5% ->
0.1% at T=0.5) — the context compounds the model's own degenerate
output and it never recovers, pointing at longer-context training
(the length/entropy hypothesis) and a teacher-forced-history eval
variant to separate "can't use context" from "poisons its own
context". Reference artifacts (machine-local, untracked):
data/eval/baseline-hb32-8k-1ub*.jsonl / -report.md.

**Architecture selection on books3 is trustworthy for SODA training.**
Measured over 35 confs (frontier + random, mixed families) x 3 fresh
SODA runs against the DB's existing books3 run distributions, as the
pairwise win function W(C1, C2) = P(loss(R1) < loss(R2)) over run
pairs:

- W barely moves: mean |W_soda − W_books3| = 0.029 over 595 conf
  pairs; 1.5% reverse direction; Spearman 0.996 on median losses.
- Confident books3 verdicts (W ≥ 0.8) agree on SODA ~99% of the
  time — statistically at the ceiling set by 3-vs-3 run-sampling
  noise, i.e. no measurable corpus effect beyond sampling.
- The practical rule reads off the books3 **loss gap**, not W:

  | books3 gap (b/B) | SODA agreement |
  |---|---|
  | < 0.02 | ~50% (coin flip — this band is run noise on books3 itself) |
  | 0.02 – 0.10 | ~95% |
  | > 0.10 | 100% in sample |

- The cell that matters most is same-weight pairs — cross-weight
  comparisons are decided by capacity, the search's real decisions
  happen between models of similar size (sample: 353–11,425 weights,
  median 1,361). Split by weight ratio:

  | weight ratio | pairs | confident (books3) | SODA agreement | mean \|ΔW\| |
  |---|---|---|---|---|
  | ≤ 1.25x | 74 | 59 | **0.966** | 0.103 |
  | 1.25–2x | 135 | 133 | 0.992 | 0.044 |
  | > 2x | 386 | 372 | 1.000 | 0.017 |

  Even in the hardest cell — same weight class, median books3 gap
  0.078 b/B — transfer holds at ~97%.
- Mechanism: the corpus change **stretches** loss differences
  affinely instead of reordering them — soda ≈ 1.438·books3 − 2.074
  (r = 0.996), pairwise gaps ~1.4x wider on SODA. Order-preserving
  by construction.
- `bits.*` pairs (corpus-independent tokenization): 78/78 confident
  pairs transfer, all metrics — pure architecture ranking transfers
  perfectly at these sizes.
- The one soft cell: pairs touching a **≥ 64-token tokenized conf**
  agree at 0.981 (vs 0.998 rest) and own all three reversals with a
  real books3 gap (hexbpe64, shift64). Supports the standing plan:
  per-corpus tokenset builds become necessary from tokens.64.bpe up;
  below that, tokens.32.* and bits.* are universal.
- Accounting caveat: books3-fitted vs SODA-refitted residual
  constants flip 10 cross-family verdicts involving lossy tokensets
  (fold32 residual 0.348 → 0.314 on SODA, shift32 0.151 → 0.109) —
  refit residuals per corpus for any cross-tokenset comparison on a
  new corpus.

Data: machine-local `scratch/transfer_study/` (report.md,
runs.jsonl); DB books3 side is reproducible from the run table.

## Case-folding buys nothing (2026-08-20)

**Removing the upper/lowercase distinction (and specials) simplifies
the text by far less than it costs.** On information-equalized data
(SODA folded to the 32-char fold alphabet), hexbpe-32 models the
folded stream at 1.88 b/B vs 1.89 raw-on-raw — i.e. case, digits,
and punctuation cost a contextual model only **~0.015–0.04 b/B**,
because case is nearly determined by context. The fold tokenset's
static reconstruction charge is 0.348 b/B — an order of magnitude
more than the modeling it saves. DB confirmation at every scale:
fold-32 never leads hexbpe-32/shift-32 on honest terms from 500 to
12000 weights. Consequence: no lossy preprocessing for chatbot
corpora; lossless tokensets carry the program. Details and protocol:
[`io.md`](io.md) (fold accounting), `scripts/fold_corpus.py`.

## The logit soft-cap was convergence-neutral in fp32 (2026-08-19)

**The cap neither helped nor hurt where runs converged, and its
apparent stabilization was within noise on an unbiased sample.**
3-way study (no cap / correct cap / miscompiled cap): on a
run-weighted sample of historically-diverged runs the correct cap
rescued 2 and caused 2 of 8 blowups (net zero); loss deltas on
converged runs ±0.1%. The earlier "meh" verdict had unknowingly
measured a miscompiled formula: XLA:GPU folded the /30 into cuBLASLt
alpha while fusing the head bias unscaled (upstream-fixed 2026-07-24;
that bug cost days and is itself the strongest argument recorded
against keeping low-value components with unusual compute patterns).
The cap was removed; a bf16 re-check is an open roadmap item since
the rounding regime differs. Full record: [`io.md`](io.md),
"Removed: the logit soft-cap".
