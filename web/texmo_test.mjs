/**
 * Pins web/texmo.js to the Python engine.
 *
 *     node web/texmo_test.mjs
 *
 * Loads the real tokenset and the real model manifests from disk,
 * replays every prompt in web/test/vectors.json through the
 * JavaScript port and asserts:
 *
 *   - token ids are byte-exact with the Python DP tokenizer;
 *   - untokenize() round-trips to the same text;
 *   - the per-position logits agree within TOLERANCES (relative for
 *     magnitudes above 1, absolute below);
 *   - the 32-token greedy continuation is identical.
 *
 * Regenerate the fixture with `uv run python web/export_vectors.py`.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import {
  Model, ReplyCollector, TokenSet, Tokenizer, argmax,
} from './texmo.js';

const ROOT = join(import.meta.dirname, '..');

const DEFAULT_TOLERANCE = 1e-4;

/**
 * Per-model logit tolerance, where the *reference* is too noisy for the
 * default.
 *
 * The fixture is fp32, which is what the engine actually runs, but fp32
 * rounding is amplified differently by different architectures, and the
 * tolerance can never be tighter than the reference's own uncertainty.
 * Re-exporting at fp64 and diffing measures that (see
 * `scratch/web/cmp_fp64.mjs` for the recipe):
 *
 *              JS vs fp64   fp32 vs fp64
 *   rl32-8k      4.2e-6       2.8e-5
 *   hb32-8k      4.1e-6       3.1e-5
 *   mg12k        5.8e-6       3.5e-4
 *
 * The JS agrees with the fp64 truth to ~5e-6 for all three -- it is
 * uniformly accurate, and its matmuls accumulate in doubles, so it sits
 * *nearer* fp64 than fp32 XLA does. mg12k's wider band is the fp32
 * reference drifting, not the port: its long-memory chain (mingru.32
 * into a full-block rglru.1 into gru.16) amplifies fp32 rounding about
 * ten times harder than the 8k models' do. Greedy continuations are
 * still asserted token-for-token, which is the behavioural pin.
 *
 * Reproduce the fp64 column with:
 *   JAX_ENABLE_X64=1 uv run python web/export_vectors.py \
 *     --precision fp64 --places 9 -o scratch/web/vectors_fp64.json
 */
const TOLERANCES = {
  'models/mg-12k-s5.json': 5e-4,
};

function readJson(...parts) {
  return JSON.parse(readFileSync(join(ROOT, ...parts), 'utf-8'));
}

/** Deviation of one logit row against the reference row. */
function deviation(got, want) {
  let maxAbs = 0;
  let maxRel = 0;
  for (let i = 0; i < want.length; i++) {
    const abs = Math.abs(got[i] - want[i]);
    if (abs > maxAbs) maxAbs = abs;
    const rel = abs / Math.max(Math.abs(want[i]), 1);
    if (rel > maxRel) maxRel = rel;
  }
  return { maxAbs, maxRel };
}

let failures = 0;

function check(ok, message) {
  if (!ok) {
    failures++;
    console.error(`  FAIL  ${message}`);
  }
  return ok;
}

const vectors = readJson('web', 'test', 'vectors.json');
console.log(
  `fixture generated ${vectors.generated}, `
  + `${vectors.max_positions} positions/prompt, `
  + `${vectors.greedy_tokens} greedy tokens`);

let worstAbs = 0;
let worstRel = 0;

for (const modelRef of vectors.models) {
  const tokenset = new TokenSet(
    readJson('tokens', `${modelRef.tokens_name}.json`));
  const tokenizer = new Tokenizer(tokenset);
  const manifest = readJson(...modelRef.manifest.split('/'));
  const model = new Model(manifest, tokenset);
  const tolerance = TOLERANCES[modelRef.manifest] ?? DEFAULT_TOLERANCE;

  console.log(`\n${modelRef.manifest}`);
  console.log(`  ${model.spec}`);
  check(model.spec === modelRef.spec, 'spec mismatch');
  check(
    model.numWeights === modelRef.num_weights,
    `num_weights: JS ${model.numWeights} != ${modelRef.num_weights}`);
  check(model.ntokens === modelRef.ntokens, 'ntokens mismatch');
  check(
    tokenset.ntokens === modelRef.ntokens,
    'tokenset size mismatch');

  let modelAbs = 0;
  let modelRel = 0;

  for (const testCase of modelRef.cases) {
    const ids = tokenizer.tokenize(testCase.prompt);
    const idsOk = check(
      ids.length === testCase.tokens.length
      && ids.every((v, i) => v === testCase.tokens[i]),
      `${testCase.name}: token ids differ\n`
      + `        JS: ${JSON.stringify(ids)}\n`
      + `        PY: ${JSON.stringify(testCase.tokens)}`);
    check(
      tokenizer.untokenizeText(ids) === testCase.untokenized,
      `${testCase.name}: untokenize round-trip differs`);
    if (!idsOk) continue;

    // Prefill exactly as export_vectors.py does: the position-0
    // logits from reset(), then one row per prompt token.
    let logits = model.reset();
    let dev = deviation(logits, testCase.logits0);
    check(
      dev.maxRel < tolerance,
      `${testCase.name}: logits0 deviation ${dev.maxRel.toExponential(2)}`);
    modelAbs = Math.max(modelAbs, dev.maxAbs);
    modelRel = Math.max(modelRel, dev.maxRel);

    for (let i = 0; i < ids.length; i++) {
      logits = model.step(ids[i]);
      if (i >= testCase.logits.length) continue;
      dev = deviation(logits, testCase.logits[i]);
      if (dev.maxRel >= tolerance) {
        check(false,
          `${testCase.name}: position ${i} deviation `
          + `${dev.maxRel.toExponential(2)}`);
      }
      modelAbs = Math.max(modelAbs, dev.maxAbs);
      modelRel = Math.max(modelRel, dev.maxRel);
    }

    const greedy = [];
    let cur = argmax(logits);
    for (let n = 0; n < vectors.greedy_tokens; n++) {
      greedy.push(cur);
      cur = argmax(model.step(cur));
    }
    check(
      greedy.length === testCase.greedy.length
      && greedy.every((v, i) => v === testCase.greedy[i]),
      `${testCase.name}: greedy continuation differs\n`
      + `        JS: ${JSON.stringify(greedy)}\n`
      + `        PY: ${JSON.stringify(testCase.greedy)}`);
    check(
      tokenizer.untokenizeText(greedy) === testCase.greedy_text,
      `${testCase.name}: greedy text differs`);

    // Partial decodes: a reply can stop mid character, and the
    // streaming UI renders every one of these prefixes.
    for (let k = 1; k <= greedy.length; k++) {
      const partial = tokenizer.untokenizeText(greedy.slice(0, k));
      if (partial !== testCase.greedy_partials[k - 1]) {
        check(false,
          `${testCase.name}: partial decode of ${k} tokens differs:\n`
          + `        JS: ${JSON.stringify(partial)}\n`
          + `        PY: ${JSON.stringify(testCase.greedy_partials[k - 1])}`);
        break;
      }
    }

    // The reply collector, replayed over the same greedy stream so
    // it is deterministic (chat.collect_reply on the Python side).
    let streamed = '';
    const collector = new ReplyCollector(
      tokenizer, vectors.greedy_tokens, (delta) => { streamed += delta; });
    for (const token of greedy) {
      if (collector.push(token)) break;
    }
    const collected = collector.finish();
    check(
      collected.text === testCase.collected_text,
      `${testCase.name}: collected reply differs:\n`
      + `        JS: ${JSON.stringify(collected.text)}\n`
      + `        PY: ${JSON.stringify(testCase.collected_text)}`);
    check(
      collected.ntokens === testCase.collected_tokens,
      `${testCase.name}: collected token count `
      + `${collected.ntokens} != ${testCase.collected_tokens}`);
    check(
      streamed === testCase.streamed,
      `${testCase.name}: streamed text differs:\n`
      + `        JS: ${JSON.stringify(streamed)}\n`
      + `        PY: ${JSON.stringify(testCase.streamed)}`);
  }

  console.log(
    `  ${modelRef.cases.length} prompts OK; max logit deviation: `
    + `${modelRel.toExponential(2)} relative, `
    + `${modelAbs.toExponential(2)} absolute `
    + `(tolerance ${tolerance.toExponential(0)})`);
  worstAbs = Math.max(worstAbs, modelAbs);
  worstRel = Math.max(worstRel, modelRel);
}

console.log(
  `\nworst deviation across ${vectors.models.length} models: `
  + `${worstRel.toExponential(2)} relative, `
  + `${worstAbs.toExponential(2)} absolute`);

if (failures) {
  console.error(`\n${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('all checks passed');
