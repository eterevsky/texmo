# Findings

Important empirical results that keep informing decisions — the
knowledge, not the work log ([`done.md`](done.md) records the work;
entries here are the results worth re-reading a year later). Dated,
newest first. Full data locations are noted per entry; scratch/ paths
are machine-local and untracked.

## The eval can see speech acts now: the style-mixed examiner (2026-09-03)

The scripted-examiner eval was blind to everything s5 added (see
2026-08-30 below). It now mixes the examiner's own side, per seed
index and from seeded streams: 30% of dialogs entirely lower case
without closing full stops, 40% opened by a forced bare greeting (the
seed opener moves to the examiner's second turn; still 10 turns and 5
answers), 15% closed by a forced farewell, with a few never-trained
forms in both phrase tables. Re-baselining eight students at 100
seeds: the three s5 models answer greetings 88-97% and farewells
77-100% in kind and mirror lower case on 97-100% of their answers;
the five s4/s3/ELIZA students score a flat **0% on all three**, and
the phrase-bot nulls 0-9% ok / 10-22% mirror (the luck floor -- a bag
drawn from an s5 model's own output contains lowercase phrases).
Unseen greeting forms are answered 50-100% against 96-100% for
trained ones: the act generalizes; the table is not merely memorized.

Two judge rulings landed with it, calibrated old-vs-new on the same
answers: pass A ignores capitalization and a missing final period (it
had been costing lower-case answers ~2.5 points), and pass C counts a
minimal direct answer that resolves the question as substantive (bare
polar answers to polar questions 15% -> 40% credited; deflections and
the good anchor unmoved; the pass-C *parser* was exonerated -- 26,030
stored replies re-parsed with zero disagreements, so the old "Yes."
inconsistency was the judge's own). Under the amended c, seven of
eight students beat their own phrase bot on substance (+0.6 to +7.8)
where none did before; **mg12k-s5 (12.4k weights, the search's best
12k conf trained on s5) is the only model positive on both b (46.0
vs null 41.0) and c (11.8 vs 4.0)** -- the first model above its own
floor on responsiveness and substance at once. The s4/s5 ranking
flipped with the instrument: mg12k-s4, the best model under the old
examiner (and the first ever to beat its null overall, b 41.8 vs
35.8 at T=0.4), answers 0% of greetings and goes 5.6 points of b
negative once 40% of dialogs open with one.

Side-effects to remember: b shifted -5.6 to +7.6 purely from the
easier turn mix, so **pre- and post-2026-09-03 a/b/c are not
comparable** (legacy numbers below stand as history); lower-case
input *helps* case-trained models by 2-11 points of b while costing
mg12k-s4 6.4 and ELIZA 5.4 (its rules key on capitalized words), and
hb32-8k-s3 drops to c = 0.0% there. The judge itself carries ~2.5
points of run-to-run noise at T=0 (llama.cpp batch nondeterminism).
Thin spot: the unseen-farewell cell is n = 2 per run. Artifacts:
`evals/style100-*`, `scratch/eval_style/` (machine-local).

## Synthesized speech acts and case style are learned outright: s5 (2026-08-30)

SODA starts and ends mid-conversation, so the simplified corpora had
no greetings, farewells or thanks, and a chat opening with "Hi!" got a
random trivial phrase. **s5** = s4 + two seeded post-steps on the
rendered turns: a greeting pair prefixed to 50% of dialogs, thanks
(20%) and a farewell pair (40%) appended, alternation preserved
(23.9% of Bot turns become speech acts); and 50% of dialogs rewritten
whole -- both sides, after the insertion -- in lower case without the
closing full stop. Same 30k dialogs and polar log as s4.

Judge-free probe (20 samples per opener, T = 0.5): correct speech-act
replies went from 0-1% (s4 models) to **60% (hb32) / 80% (rl32)**,
rl32 answering every capitalized greeting/farewell/thanks probe 100%;
case mirroring on lower-case probes from 0% to **94-97%**. Cost: the
scripted-examiner eval is blind to all of it (its openers carry
content in the same turn and none of its dialogs say goodbye), and b
drifts down within noise (hb32 40.0 -> 35.6, rl32 33.6 -> 31.4,
z <= 1.4) because the cues compete for the same weights: hb32's
polar-cue margin over its null fell from z 4.0 to z 1.0. Corpus loss
is not comparable across corpora (s5 1.183 vs s4 1.212 for hb32 --
the corpus got easier, not the model better). Two notes for the next
round: bare unpunctuated lower-case openers ("hi") are still out of
distribution because the synthesized lower-case greetings keep their
"!"; and each new corpus regularity needs its own probe, since the
examiner eval measures only what SODA-style openers exercise.
Artifacts: `models/{hb32,rl32}-8k-s5.json`, `data/eval/eval100-*s5*`,
`scratch/s5/` (machine-local).

## Longer context and 4x steps move the loss, not the chat metrics (2026-08-29)

Two ~8k specs (the hb32 baseline and the rglru+lstm `rl32`) trained on
s3 in five arms -- base 256x128 / 16k steps, l256 (128x256), l512
(64x512), 4x steps, l512-4x -- and scored with one instrument (same
sampled bytes for every model; recurrent and parallel losses agree to
4 decimals at these batch sizes). Loss at 1024-byte samples, b/B:

| arm | hb32 | rl32 |
|---|---|---|
| base | 1.274 | 1.289 |
| l256 | 1.277 | 1.265 |
| l512 | 1.274 | 1.266 |
| 4x | 1.258 | 1.263 |
| l512-4x | **1.242** | 1.261 |

Steps move hb32 and length moves rl32, and only hb32 compounds the two
(-0.032). All single runs; same-conf spread in the DB is 0.02-0.05
b/B, so treat the pattern as suggestive and leave the question to the
search, which mutates steps/batch/length and averages over runs. The
loss predictor, asked the same question, ranks rl32 as the less
saturated model at 16k steps (slope -0.016 vs -0.013 b/B per octave)
but with a 1.2x ratio where the observation was 2.4x -- it is a prior
on saturation, not a measurement (`steps` is one global log2 feature).

The chatbot eval does not move at all: across the ten models b sits in
24.4-29.8% and c in 0.8-2.8% (n = 500, SE +-2 points), rank
correlation of loss with b -0.27, and **no model beats its own phrase
bot** (b premium -8.6 to +1.0, difference SE +-2.8). The best-loss
model is not the best chatbot; a bag of `rl32-l512-4x`'s own 20 most
frequent answers drawn blind scores b 36.0, the highest of the sweep.
Corpus-loss gains at this size land entirely below the resolution of
the responsiveness metric; what moved it was the data (next entries).
Artifacts: `models/{hb32,rl32}-8k-s3-*.json`, `data/eval/eval100-*`,
`scratch/len_sweep/` (machine-local).

## A tiny model can learn a structural cue: s4 (2026-08-29)

The s3 8k baseline sat exactly on its null-model floor for
responsiveness (b 25.8 vs 26.8) because its corpus had almost no polar
answers -- 487 bare yes/no in 104k Bot turns; the simplification prompt
had asked for *varied* phrases. **s4** restores them: an aux-verb-
initial User question is detected structurally, a Qwen3-8B pass labels
what the original Bot reply meant as an answer ("granted or refused",
read broadly -- requests and offers count, since a tiny model cannot
tell them from polar questions either; 81% of cued turns get a label
vs 20% under the strict reading), and the corpus writes bare
"Yes."/"No." there (6.1% of Bot turns). Same 30k dialogs as s3.

Trained on s4, the same spec scores a 90.2 / b 40.0 / c 4.2 against
its own phrase bot's 100 / 44.0 / 3.0 -- still below the floor
overall, because a generically agreeable "Yes." raises the floor too
(the null model is not a constant; re-derive it per corpus). But on
the turns that follow an aux-verb-initial question it scores **b 62.2
/ c 11.2** against the null's 34.7 / 6.1 (z = 4.0) and s3's 15.9 / 1.1
(z = 7.4), and only there: it emits a polar answer 56% of the time on
cued turns and 6.7% elsewhere (8.4x enrichment) while the null model
is at chance. It generalizes past the detector ("So light is made up
of colors?" -> "Yes."). First evidence that ~8k weights condition an
answer on a structural property of the input; the route past the
phrase-bot floor is more cue->answer regularities in the corpus, not
more data. Hand impression (Oleg): noticeably more coherent than any
earlier 8k model, with "I am good." as the attractor once it is in the
context twice. Artifacts: `models/{hb32,rl32}-8k-s4.json`,
`data/eval/eval100-hb32-8k-s4*`, `data/simplified/polar-log.jsonl`.

## ELIZA is the (b) bar, and reflection is not what buys it (2026-08-29)

Weizenbaum's DOCTOR (`scripts/eliza/`, 0 weights) through the same
eval: **a 79.2 / b 37.0 / c 3.2** -- the highest b of any zero-weight
student, +11 over the s3 model, c on the floor: the "responsive but
empty" shape the eval should be able to name. But ELIZA's *own* phrase
bot -- its 20 most frequent answers drawn blind -- scores b 34.2, so
reading the question is worth +2.8 points and the rest is vocabulary:
content-free questions fit almost any turn ("In what way?" b 77.5 when
drawn at random, "Can you elaborate on that?" 91.7), where a
SODA-trained model's declaratives ("I am good.") fit few. So (b) alone
cannot separate comprehension from a well-chosen bag of
interrogatives, and the bar a small model must clear on b before it
has shown it conditions on anything is ELIZA's 37, not the phrase
bot's 27 -- and clearing it without moving c means it learned
reflection, not content. ELIZA's low a is 1966 meeting 2026 input:
short canned replies a 94 / b 70, long reflected-clause reassemblies a
57 / b 11. Artifacts: `data/eval/eval100-eliza*`,
`eval100-phrasebot-eliza*`.

## XLA:GPU adds a fused bias on the wrong axis (2026-08-29)

**A `dot` whose output layout is `{0,1}` plus a bias add is fused by
XLA:GPU into a cuBLASLt `epilogue:"BIAS"` that computes
`out[m,n] += b[m]` instead of `out[m,n] += b[n]`.** Silent whenever
M == N (the wrong broadcast is shape-compatible; XLA declines the
fusion when M != N, so only square dots corrupt). Second miscompile of
a cuBLASLt bias epilogue on this stack after the soft-cap one below —
same class, different mechanism, and again found only because two
formulations of the same quantity disagreed.

Reached in texmo from `Model2Jax.forward_recurrent` (vmap of `step`
over the batch inside `lax.scan`): each layer's `[batch, width]`
dot+bias goes feature-major downstream of a `split.cat` merge, so
`batch == width` hits it. `models/hb32-8k-s3.json` on
`data/soda/soda_train_ub_s3.txt` read **10.84 b/B at batch 32 and
2.09 at batch 16 against a true 1.36**; every other batch size, the
parallel forward, and CPU were correct, at every sample length. It
does not need trained weights, only a nonzero bias — at init the
error is just small.

Only the recurrent eval path (`ManagerJax.eval`, `texmo.py eval`) is
structurally exposed; the parallel forward, the training step, and
unbatched generation never produce a square `{0,1}` dot. Search
results are safe because the eval batch is 1024 and no searched layer
is 1024 wide — the exposure is `texmo.py eval --chunk W` and any
hand-run eval whose batch equals a layer width. Fixed by
`--xla_gpu_cublas_fallback=false`, injected at `import texmo`
(`texmo/xla_flags.py`); perf-neutral to favorable on this repo's
shapes. `--xla_gpu_enable_cublaslt=false` does NOT help (inert in this
build). Seen on jax/jaxlib 0.11.0 + winjax CUDA 13, sm_120; other
CUDA machines are unverified. 20-line standalone repro:
`scratch/recurrent_bug/repro.py`.

## Simplified data moves an 8k model from ~1% to ~80% grammatical (2026-08-28)

Same spec and hyperparameters as the baseline, retrained on 30k
SODA dialogs in four renderings of ONE simplification pass (scripts/
simplify_corpus.py): s0 natural, s1 Bot A2-simplified, s2 Bot reduced
to a trivial phrase, s3 both sides reduced (User A2, Bot trivial).
100-seed scripted-examiner eval at T=0.5 (500 graded answers; the
examiner speaks natural English throughout):

| model | a | b | c |
|---|---|---|---|
| baseline / s0 / s1 | ~1-2% | ~0-1% | ~0% |
| s2 | 59% | 34% | 11% |
| s3 | **79%** | **44%** | 15% |

Reading: (1) steering the answer distribution toward trivial phrases
buys grammar and responsiveness wholesale at 8k -- the model can
learn "which short phrase fits" where it cannot learn sentences.
(2) s3 >> s2: simplifying the User side -- what the model must READ
-- matters as much as what it produces, and the gain survives a
natural-English examiner. (3) c stays low (deflection rate b-c ~ 29%):
the substance target is the real fight, as designed. (4) The
per-position curve is flat for s2/s3 (no context poisoning: replies
are too short to poison), unlike the baseline's 3-10x decay.
(5) Corpus size is not the constraint: s0 (30k dialogs, ~25x
repeated) shows no overfitting (train 1.717 vs valid 1.688 b/B,
parallel-forward numbers) and matches the full 814 MB baseline.
Training-corpus loss (parallel forward): s0 1.717, s1 1.646, s2
1.589, s3 1.280 b/B. Artifacts: data/eval/eval100-hb32-8k-*.jsonl,
models/hb32-8k-s{0,1,2,3}.json (machine-local).

**Null-model correction (same day, Oleg's check):** a phrase bot
drawing s3's own 20 most frequent answers at random with s3's
frequencies (63% of its output; `chat_eval.py phrases` +
`--student-phrases`) scores a 91.2 / b 46.8 / c 10.4 on the same 100
seeds. s3's b (43.6) is *below* the bot's (z = -1.0): on
responsiveness s3 is indistinguishable from random phrase emission
-- everything it scores comes from the marginal fit of casual stock
phrases to casual turns (~47% is the floor for ANY phrase emitter on
this eval). The only crack is c (+4.2, z = +2.0, one of three
comparisons). s3 does learn turn *type* (it asks back after 45% of
statements vs 22% of questions, ~5 sigma) but not turn *content*.
Two methodological consequences: (1) "coherent" must mean beating
the model's own phrase-bot on b/c, so the null model becomes a
standard control column; (2) the judge leaks context into criterion
(a): ~9% of perfectly-formed phrases fail it when the conversation
is bad, so a's ceiling under this judge is ~91% -- the a >= 90
target sits on the ceiling; grade (a) with the context hidden.

**Three-pass judge (same day):** (a) judged on the utterance alone,
(b) on the pair, (c) on the whole dialog; llama-server prefix
caching makes the extra passes cheap (pass A evaluates ~69 of ~368
prompt tokens; 25 answers/s at 16 slots, no knee yet). Re-judged
numbers (a/b/c): phrase bot **100**/26.8/1.6, s3 92.0/25.8/1.6, s2
76.6/23.8/1.2, baseline (T=0.5, 100 seeds) 8.0/2.2/0.0. So under a
clean instrument: s3 CLEARS the a >= 90 target, its responsiveness
sits exactly on the null-model floor (b premium -1 point), and
substance is ~0 for every 8k model while the good anchor scores
86.7 on c -- the strictness is signal. The old single-prompt judge
had been inflating b (context lent stock phrases plausibility) and
crushing a (context leaked into a reply-only criterion), for the
baseline too (a 1.6 -> 8-14%). Definition going forward: coherent =
beats its own phrase-bot on b and c.

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

## Books3 rankings transfer to SODA (2026-08-21)

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
