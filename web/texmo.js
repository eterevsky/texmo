/**
 * texmo.js -- a dependency-free JavaScript port of the Texmo
 * inference path: tokenizer, model step and dialog logic.
 *
 * It is a line-by-line port of the Python engine, deliberately kept
 * close to the original so the two can be diffed:
 *
 *   texmo/tokens/processing.py      -> capswords2 process/unprocess
 *   texmo/tokens/tokenizer.py       -> DP tokenizer + Decoder
 *   texmo/spec_parser.py            -> parseSpec
 *   texmo/layers/{rnn,dense,lstm,gru,rglru,rmsnorm,split,seq}.py
 *   texmo/layers/one_hot_codec.py   -> OneHotCodec
 *   texmo/model2_jax.py             -> Model
 *   texmo/chat.py                   -> buildPrompt/appendReply/ReplyCollector
 *
 * Deviations from the Python (all deliberate, all verified by
 * web/texmo_test.mjs against exported reference vectors):
 *
 *  - Weights and layer outputs are Float32Array (JAX's fp32); the
 *    arithmetic inside a matmul accumulates in JS doubles and is
 *    rounded to fp32 on store. That is slightly *more* accurate than
 *    XLA, not less.
 *  - A Model owns its recurrent state instead of threading it
 *    functionally: one Model instance is one running sequence.
 *    `reset()` starts a new one.
 *  - Sampling uses mulberry32 rather than JAX's PRNG, so sampled
 *    text is not reproducible across the two (greedy decoding is).
 */

// ---------------------------------------------------------------------
// bytes and strings
// ---------------------------------------------------------------------

const TEXT_ENCODER = new TextEncoder();
const TEXT_DECODER_STRICT = new TextDecoder('utf-8', { fatal: true });
const TEXT_DECODER_LOSSY = new TextDecoder('utf-8');

export function encodeUtf8(s) {
  return TEXT_ENCODER.encode(s);
}

/** Throws on malformed UTF-8, like Python's bytes.decode('utf-8'). */
export function decodeUtf8(bytes) {
  return TEXT_DECODER_STRICT.decode(bytes);
}

/** Python's bytes.decode('utf-8', errors='replace'). */
export function decodeUtf8Lossy(bytes) {
  return TEXT_DECODER_LOSSY.decode(bytes);
}

/** Map a byte array to a latin-1 string usable as a Map key. */
function byteKey(bytes) {
  let s = '';
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  return s;
}

function concatBytes(chunks) {
  let n = 0;
  for (const c of chunks) n += c.length;
  const out = new Uint8Array(n);
  let at = 0;
  for (const c of chunks) {
    out.set(c, at);
    at += c.length;
  }
  return out;
}

// ---------------------------------------------------------------------
// Text pre-processing: `capswords2` (texmo/tokens/processing.py)
// ---------------------------------------------------------------------

export const CAPITALIZED_MARKER = String.fromCharCode(0x14);
export const ALLCAPS_MARKER = String.fromCharCode(0x15);
export const WORD_MARKER = String.fromCharCode(0x16);
export const UPPER_MARKER = String.fromCharCode(0x17);

const WORD_START_MARKERS = [
  CAPITALIZED_MARKER, ALLCAPS_MARKER, UPPER_MARKER];

// Python's str.isalpha() is categories Lu/Ll/Lt/Lm/Lo == \p{L};
// str.isupper() is Lu + Other_Uppercase == \p{Uppercase}; str.islower()
// is Ll + Other_Lowercase == \p{Lowercase}; titlecase is Lt.
const RE_ALPHA = /^\p{L}$/u;
const RE_UPPER = /^\p{Uppercase}$/u;
const RE_LOWER = /^\p{Lowercase}$/u;
const RE_TITLE = /^\p{Lt}$/u;

const isAlphaChar = (ch) => RE_ALPHA.test(ch);
const isUpperChar = (ch) => RE_UPPER.test(ch);
const isLowerChar = (ch) => RE_LOWER.test(ch);
const isTitleChar = (ch) => RE_TITLE.test(ch);

/** Python str.islower(): no upper/title char, at least one lower. */
function strIsLower(s) {
  let cased = false;
  for (const ch of s) {
    if (isUpperChar(ch) || isTitleChar(ch)) return false;
    if (isLowerChar(ch)) cased = true;
  }
  return cased;
}

/** Python str.isupper(): no lower/title char, at least one upper. */
function strIsUpper(s) {
  let cased = false;
  for (const ch of s) {
    if (isLowerChar(ch) || isTitleChar(ch)) return false;
    if (isUpperChar(ch)) cased = true;
  }
  return cased;
}

function addWord2(out, word) {
  const chars = Array.from(word);
  const first = chars[0];
  const rest = chars.slice(1).join('');
  if (isUpperChar(first) && (rest === '' || strIsLower(rest))) {
    out.push(CAPITALIZED_MARKER);
    out.push(word.toLowerCase());
  } else if (isUpperChar(first) && rest !== '' && strIsUpper(rest)) {
    out.push(ALLCAPS_MARKER);
    out.push(word.toLowerCase());
  } else if (chars.some(isUpperChar)) {
    for (const ch of chars) {
      if (isUpperChar(ch)) {
        out.push(UPPER_MARKER);
        out.push(ch.toLowerCase());
      } else {
        out.push(ch);
      }
    }
  } else {
    out.push(word);
  }
  out.push(WORD_MARKER);
}

/** processing.process2 -- the capswords2 forward pass. */
export function process2(text) {
  const out = [];
  let word = [];
  // 'word' | 'space_after_word' | 'nonword'
  let state = 'nonword';

  for (const ch of text) {
    if (isAlphaChar(ch)) {
      if (state === 'word') {
        word.push(ch);
      } else {
        // A pending space before a word is elided: WORD_MARKER on the
        // previous word already implies the boundary.
        word = [ch];
        state = 'word';
      }
    } else if (ch === ' ') {
      if (state === 'word') {
        addWord2(out, word.join(''));
        word = [];
        state = 'space_after_word';
      } else if (state === 'space_after_word') {
        out.push(' ');
        out.push(ch);
        state = 'nonword';
      } else {
        out.push(ch);
      }
    } else {
      if (state === 'word') {
        addWord2(out, word.join(''));
        word = [];
      } else if (state === 'space_after_word') {
        out.push(' ');
      }
      out.push(ch);
      state = 'nonword';
    }
  }

  if (state === 'word') addWord2(out, word.join(''));
  else if (state === 'space_after_word') out.push(' ');

  return out.join('');
}

/** processing.unprocess2 -- the capswords2 inverse. */
export function unprocess2(text) {
  const s = Array.from(text);
  const out = [];
  let upperNext = false;   // UPPER_MARKER: exactly one letter
  let upperWord = false;   // ALLCAPS_MARKER: until WORD_MARKER

  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (ch === CAPITALIZED_MARKER || ch === UPPER_MARKER) {
      upperNext = true;
    } else if (ch === ALLCAPS_MARKER) {
      upperWord = true;
    } else if (ch === WORD_MARKER) {
      upperWord = false;
      upperNext = false;
      // The space between two words was dropped by process2, so put
      // it back when another word starts here.
      const nxt = i + 1 < s.length ? s[i + 1] : '';
      if (nxt && (isAlphaChar(nxt) || WORD_START_MARKERS.includes(nxt))) {
        out.push(' ');
      }
    } else if (upperNext || upperWord) {
      out.push(ch.toUpperCase());
      upperNext = false;
    } else {
      out.push(ch);
    }
  }
  return out.join('');
}

const PROCESSORS = {
  capswords2: [process2, unprocess2],
};

/** Text -> processed bytes (`apply_processing`). */
function applyProcessing(name, text) {
  const fns = PROCESSORS[name];
  if (name && name !== 'raw' && !fns) {
    throw new Error(`unsupported tokenset processing: ${name}`);
  }
  return encodeUtf8(fns ? fns[0](text) : text);
}

/**
 * Processed bytes -> text bytes (`undo_processing`).
 *
 * Mirrors the Python exactly, including its defensive behaviour: a
 * partial decode that is not valid UTF-8 is returned unchanged (with
 * the markers still in it) rather than raising.
 */
function undoProcessing(name, bytes) {
  const fns = PROCESSORS[name];
  if (!fns) return bytes;
  let text;
  try {
    text = decodeUtf8(bytes);
  } catch {
    return bytes;
  }
  return encodeUtf8(fns[1](text));
}

// ---------------------------------------------------------------------
// Tokenset + DP tokenizer (texmo/tokens/{tokenset,tokenizer}.py)
// ---------------------------------------------------------------------

/** A JSON `tokens`/`sequences` entry -> bytes, or an ext-token id. */
function parseTokenEntry(entry) {
  if (typeof entry === 'string') return encodeUtf8(entry);
  if (Array.isArray(entry)) return Uint8Array.from(entry);
  if (typeof entry === 'number') return entry;   // numbered ("ext") token
  throw new Error(`bad token entry: ${JSON.stringify(entry)}`);
}

export class TokenSet {
  constructor(json) {
    this.type = json.type;
    this.processing = json.processing;
    this.stats = json.stats || {};
    this.algorithm = json.algorithm || 'dp';
    // tokens[i] = {id, bytes|null}
    this.tokens = [];
    const byString = new Map();     // byteKey -> first id with it
    for (const entry of json.tokens) {
      const parsed = parseTokenEntry(entry);
      const id = this.tokens.length;
      if (typeof parsed === 'number') {
        this.tokens.push({ id, bytes: null });
      } else {
        this.tokens.push({ id, bytes: parsed });
        const k = byteKey(parsed);
        if (!byString.has(k)) byString.set(k, id);
      }
    }
    // sequences: byteKey -> {bytes, ids}
    this.sequences = new Map();
    for (const seq of json.sequences || []) {
      const bytes = parseTokenEntry(seq.string);
      const ids = seq.tokens.map((t) => {
        const parsed = parseTokenEntry(t);
        // Sequence members are spelled either as ext-token ids (bare
        // ints -> our nibble tokens) or as literal pieces.
        if (typeof parsed === 'number') return parsed;
        const id = byString.get(byteKey(parsed));
        if (id === undefined) {
          throw new Error(`sequence refers to unknown token`);
        }
        return id;
      });
      this.sequences.set(byteKey(bytes), { bytes, ids });
    }
  }

  get ntokens() {
    return this.tokens.length;
  }

  /** Corpus knowledge charged as model weights (docs/io.md). */
  get extraWeights() {
    return Math.trunc(this.stats.extra_weights || 0);
  }
}

/**
 * The generic dynamic-programming tokenizer over tokens + sequences.
 *
 * This is what the Python registry hands out for `hexbpe` sets at
 * sampling time (`registry.get_tokenizer`) -- NOT the builder-exact
 * merge loop -- so this port reproduces what the model was sampled
 * with.
 */
export class Tokenizer {
  constructor(tokenset) {
    this.tokenset = tokenset;

    // -- spans: every byte string with a fixed token spelling.
    this.spans = new Map();
    const empty = { bytes: new Uint8Array(0), len: 0, revIds: [], cost: 0 };
    this.spans.set('', empty);
    for (const token of tokenset.tokens) {
      if (token.bytes === null) continue;
      const ids = [token.id];
      this.spans.set(byteKey(token.bytes), {
        bytes: token.bytes,
        len: token.bytes.length,
        revIds: ids.slice().reverse(),
        cost: ids.length,
      });
    }
    for (const { bytes, ids } of tokenset.sequences.values()) {
      this.spans.set(byteKey(bytes), {
        bytes,
        len: bytes.length,
        revIds: ids.slice().reverse(),
        cost: ids.length,
      });
    }
    for (const span of this.spans.values()) {
      if (span.len > 0) {
        span.suffixSpan = this._findSuffixSpan(span.bytes.subarray(1));
      }
    }

    this._buildSuffixStates();
    this._buildDecoder();
  }

  /** Longest span that is a suffix of `bytes` (the whole of it counts). */
  _findSuffixSpan(bytes) {
    for (let start = 0; start <= bytes.length; start++) {
      const span = this.spans.get(byteKey(bytes.subarray(start)));
      if (span !== undefined) return span;
    }
    throw new Error('unreachable: no suffix span');
  }

  /**
   * The finite-state machine over "interesting" suffixes: one state
   * per prefix of any span, each with the longest span that is a
   * suffix of that state's string plus a 256-way next table.
   */
  _buildSuffixStates() {
    const states = new Map();
    states.set('', { bytes: new Uint8Array(0), span: this.spans.get('') });
    for (const span of this.spans.values()) {
      for (let end = 1; end <= span.len; end++) {
        const prefix = span.bytes.subarray(0, end);
        const key = byteKey(prefix);
        if (!states.has(key)) {
          states.set(key, { bytes: prefix, span: this._findSuffixSpan(prefix) });
        }
      }
    }
    for (const state of states.values()) {
      state.next = new Array(256).fill(null);
      const extended = new Uint8Array(state.bytes.length + 1);
      extended.set(state.bytes);
      for (let byte = 0; byte < 256; byte++) {
        extended[state.bytes.length] = byte;
        for (let start = 0; start <= extended.length; start++) {
          const next = states.get(byteKey(extended.subarray(start)));
          if (next !== undefined) {
            state.next[byte] = next;
            break;
          }
        }
      }
    }
    this._states = states;
    this._root = states.get('');
  }

  _buildDecoder() {
    // Token-id tuples (joined with ',') -> the bytes they spell.
    this._tokensToString = new Map();
    this._hasLonger = new Set();
    for (const token of this.tokenset.tokens) {
      if (token.bytes !== null) {
        this._tokensToString.set(String(token.id), token.bytes);
      }
    }
    for (const { bytes, ids } of this.tokenset.sequences.values()) {
      this._tokensToString.set(ids.join(','), bytes);
      for (let end = 1; end < ids.length; end++) {
        this._hasLonger.add(ids.slice(0, end).join(','));
      }
    }
  }

  /** Text -> token ids (processing + optimal DP segmentation). */
  tokenize(text) {
    return this.tokenizeProcessed(
      applyProcessing(this.tokenset.processing, text));
  }

  /**
   * Optimal (fewest-token) segmentation of an already-processed byte
   * chunk. The whole chunk is segmented at once: tokenization is
   * context-dependent, which is why the dialog is re-tokenized from
   * scratch every turn instead of appended to incrementally.
   */
  tokenizeProcessed(chunk) {
    let state = this._root;
    const cost = [0];
    const spans = [this.spans.get('')];

    for (let pos = 0; pos < chunk.length; pos++) {
      state = state.next[chunk[pos]];
      let span = state.span;
      let bestSpan = span;
      let bestCost = cost[cost.length - span.len] + span.cost;

      span = span.suffixSpan;
      while (span && span.len > 0) {
        const newCost = cost[cost.length - span.len] + span.cost;
        if (newCost < bestCost) {
          bestCost = newCost;
          bestSpan = span;
        }
        span = span.suffixSpan;
      }

      cost.push(bestCost);
      spans.push(bestSpan);
    }

    const tokens = [];
    let pos = spans.length - 1;
    while (pos > 0) {
      const span = spans[pos];
      for (const id of span.revIds) tokens.push(id);
      pos -= span.len;
    }
    tokens.reverse();
    return tokens;
  }

  /**
   * Token ids -> bytes. Defensive, like the Python `Decoder`: a
   * *sampled* stream can misuse a marker or escape, so when nothing
   * decodable starts at the current position the head token is
   * dropped and the rest reprocessed.
   */
  decode(ids) {
    const out = [];
    let fragment = [];
    for (const id of ids) {
      fragment.push(id);
      if (this._hasLonger.has(fragment.join(','))) continue;
      const tail = [];
      while (fragment.length && !this._tokensToString.has(fragment.join(','))) {
        tail.push(fragment.pop());
      }
      if (fragment.length) {
        out.push(this._tokensToString.get(fragment.join(',')));
      } else {
        tail.pop();
      }
      fragment = tail;
      fragment.reverse();
    }
    if (fragment.length) {
      while (fragment.length && !this._tokensToString.has(fragment.join(','))) {
        fragment.pop();
      }
      if (fragment.length) {
        out.push(this._tokensToString.get(fragment.join(',')));
      }
    }
    return concatBytes(out);
  }

  /** Token ids -> text bytes (decode + inverse processing). */
  untokenize(ids) {
    return undoProcessing(this.tokenset.processing, this.decode(ids));
  }

  /** Token ids -> text, with U+FFFD for a trailing partial character. */
  untokenizeText(ids) {
    return decodeUtf8Lossy(this.untokenize(ids));
  }
}

// ---------------------------------------------------------------------
// Tensors
// ---------------------------------------------------------------------

/** A nested-list weight -> {rows, cols, data} in row-major fp32. */
function mat(nested) {
  const rows = nested.length;
  const cols = nested[0].length;
  const data = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    const row = nested[i];
    if (row.length !== cols) throw new Error('ragged weight matrix');
    for (let j = 0; j < cols; j++) data[i * cols + j] = row[j];
  }
  return { rows, cols, data };
}

function vec(list) {
  return Float32Array.from(list);
}

/** out[i] = sum_j M[i][j] * x[j] + b[i]; accumulated in doubles. */
function matVecAdd(M, x, b, out) {
  const { rows, cols, data } = M;
  for (let i = 0; i < rows; i++) {
    let s = b ? b[i] : 0;
    const off = i * cols;
    for (let j = 0; j < cols; j++) s += data[off + j] * x[j];
    out[i] = s;
  }
  return out;
}

/** out[i] += sum_j M[i][j] * x[j]. */
function matVecAccum(M, x, out) {
  const { rows, cols, data } = M;
  for (let i = 0; i < rows; i++) {
    let s = out[i];
    const off = i * cols;
    for (let j = 0; j < cols; j++) s += data[off + j] * x[j];
    out[i] = s;
  }
  return out;
}

// ---------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------

const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);

/** jax.nn.gelu's default: the tanh approximation. */
function gelu(x) {
  const cdf = 0.5 * (1 + Math.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)));
  return x * cdf;
}

function sigmoid(x) {
  return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
}

/** jax.nn.softplus == logaddexp(x, 0), computed stably. */
function softplus(x) {
  return x > 0 ? x + Math.log1p(Math.exp(-x)) : Math.log1p(Math.exp(x));
}

const ACTIVATIONS = {
  relu: (x) => (x > 0 ? x : 0),
  tanh: Math.tanh,
  gelu,
  silu: (x) => x * sigmoid(x),
};

function applyActivation(fn, buf) {
  if (!fn) return buf;
  for (let i = 0; i < buf.length; i++) buf[i] = fn(buf[i]);
  return buf;
}

// ---------------------------------------------------------------------
// Spec parsing (texmo/spec_parser.py)
// ---------------------------------------------------------------------

/** Split on `delim` characters that are not inside parentheses. */
function splitAtDepth0(s, delim) {
  const parts = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (c === '(') depth++;
    else if (c === ')') {
      depth--;
      if (depth < 0) throw new Error(`unbalanced ')' in spec: ${s}`);
    } else if (c === delim && depth === 0) {
      parts.push(s.slice(start, i));
      start = i + 1;
    }
  }
  if (depth !== 0) throw new Error(`unbalanced '(' in spec: ${s}`);
  parts.push(s.slice(start));
  return parts;
}

/**
 * `spec` -> {codec, layers}. The layer defs carry input/output widths
 * and a `build(weights)` that returns the runtime layer.
 */
export function parseSpec(spec) {
  const parts = spec.split('|');
  if (parts.length > 2) throw new Error("model spec can't contain more than one |");
  const inputSpec = parts.length === 2 ? parts[0] : '';
  const layersSpec = parts.length === 2 ? parts[1] : parts[0];
  const codec = parseCodec(inputSpec);
  const layers = parseLayerList(layersSpec, codec.size);
  const lastWidth = layers.length ? layers[layers.length - 1].size : codec.size;
  codec.setHeadWidth(lastWidth);
  return { codec, layers };
}

function parseLayerList(layersSpec, inputSize) {
  if (!layersSpec.trim()) return [];
  const layers = [];
  let shape = inputSize;
  for (let piece of splitAtDepth0(layersSpec, '-')) {
    piece = piece.trim();
    if (!piece) throw new Error(`empty layer in spec: ${layersSpec}`);
    const layer = piece.startsWith('split.')
      ? parseSplit(piece, shape)
      : buildLayerDef(piece, shape);
    layers.push(layer);
    shape = layer.size;
  }
  return layers;
}

function parseSplit(spec, inputSize) {
  const openIdx = spec.indexOf('(');
  if (openIdx < 0) throw new Error(`split missing '(': ${spec}`);
  if (!spec.endsWith(')')) throw new Error(`split missing ')': ${spec}`);
  const head = spec.slice(0, openIdx).split('.');
  if (head.length !== 2 || head[0] !== 'split') {
    throw new Error(`invalid split head: ${spec}`);
  }
  const op = head[1];
  if (!['add', 'mul', 'cat'].includes(op)) {
    throw new Error(`unknown split op: ${op}`);
  }
  const body = spec.slice(openIdx + 1, -1);
  const branches = splitAtDepth0(body, ',').map((branchSpec) => {
    branchSpec = branchSpec.trim();
    if (branchSpec === 'pass') return [];
    if (!branchSpec) throw new Error(`empty branch in split: ${spec}`);
    return parseLayerList(branchSpec, inputSize);
  });
  return new SplitDef(op, branches, inputSize);
}

function buildLayerDef(spec, inputSize) {
  const parts = spec.split('.');
  const name = parts[0];
  switch (name) {
    case 'dense':
      return new DenseDef(parseInt(parts[1], 10), inputSize, parts[2] || null);
    case 'rnn':
      return new RnnDef(parseInt(parts[1], 10), inputSize, parts[2] || null);
    case 'lstm':
      return new LstmDef(parseInt(parts[1], 10), inputSize);
    case 'gru':
      return new GruDef(parseInt(parts[1], 10), inputSize);
    case 'mgru':
      return new MgruDef(parseInt(parts[1], 10), inputSize);
    case 'mingru':
      return new MinGruDef(parseInt(parts[1], 10), inputSize);
    case 'rglru':
      return new RglruDef(parseInt(parts[1], 10), inputSize);
    case 'rmsnorm':
      return new RmsNormDef(inputSize);
    case 'norm':
      return new NormDef(inputSize);
    default:
      throw new Error(`layer type not ported to JS: '${spec}'`);
  }
}

// ---------------------------------------------------------------------
// Layers. Each Def parses/sizes; each build(weights) returns a runtime
// object with `size`, `reset()` and `step(x) -> Float32Array`.
// ---------------------------------------------------------------------

class LayerDef {
  constructor(inputSize, size) {
    this.inputSize = inputSize;
    this.size = size;
    // Input positions consumed per output position. Everything ported
    // here is 1; see Model's total-padding check.
    this.length = 1;
  }
}

class DenseDef extends LayerDef {
  constructor(size, inputSize, activation) {
    super(inputSize, size);
    this.name = 'dense';
    this.activation = activation;
    if (activation && !(activation in ACTIVATIONS)) {
      throw new Error(`unknown activation: ${activation}`);
    }
    this.numWeights = size * inputSize + size;
  }
  build(w) {
    return new DenseLayer(this, w);
  }
}

class DenseLayer {
  constructor(def, w) {
    this.size = def.size;
    this.W = mat(w.w);
    this.b = vec(w.b);
    this.act = ACTIVATIONS[def.activation] || null;
    this.out = new Float32Array(def.size);
  }
  reset() {}
  step(x) {
    return applyActivation(this.act, matVecAdd(this.W, x, this.b, this.out));
  }
}

class RnnDef extends LayerDef {
  constructor(size, inputSize, activation) {
    super(inputSize, size);
    this.name = 'rnn';
    this.activation = activation;
    if (activation && !(activation in ACTIVATIONS)) {
      throw new Error(`unknown activation: ${activation}`);
    }
    this.numWeights = size * inputSize + size * size + size;
  }
  build(w) {
    return new RnnLayer(this, w);
  }
}

/** Elman RNN: h = act(W_ih x + W_hh h + b). */
class RnnLayer {
  constructor(def, w) {
    this.size = def.size;
    this.wIh = mat(w.w_ih);
    this.wHh = mat(w.w_hh);
    this.b = vec(w.b);
    this.act = ACTIVATIONS[def.activation];
    this.h = new Float32Array(def.size);
    this.scratch = new Float64Array(def.size);
  }
  reset() {
    this.h.fill(0);
  }
  step(x) {
    matVecAdd(this.wIh, x, this.b, this.scratch);
    matVecAccum(this.wHh, this.h, this.scratch);
    for (let i = 0; i < this.size; i++) this.h[i] = this.act(this.scratch[i]);
    return this.h;
  }
}

class LstmDef extends LayerDef {
  constructor(size, inputSize) {
    super(inputSize, size);
    this.name = 'lstm';
    this.numWeights = 4 * size * (inputSize + size) + 4 * size;
  }
  build(w) {
    return new LstmLayer(this, w);
  }
}

/** Stacked-gate LSTM, gate order [f, i, o, g]. */
class LstmLayer {
  constructor(def, w) {
    this.size = def.size;
    this.wIh = mat(w.w_ih);
    this.wHh = mat(w.w_hh);
    this.b = vec(w.b);
    this.h = new Float32Array(def.size);
    this.c = new Float32Array(def.size);
    this.gates = new Float64Array(4 * def.size);
  }
  reset() {
    this.h.fill(0);
    this.c.fill(0);
  }
  step(x) {
    const s = this.size;
    const g = this.gates;
    matVecAdd(this.wIh, x, this.b, g);
    matVecAccum(this.wHh, this.h, g);
    for (let i = 0; i < s; i++) {
      const f = sigmoid(g[i]);
      const inp = sigmoid(g[s + i]);
      const o = sigmoid(g[2 * s + i]);
      const cand = Math.tanh(g[3 * s + i]);
      const cNew = f * this.c[i] + inp * cand;
      this.c[i] = cNew;
      this.h[i] = o * Math.tanh(this.c[i]);
    }
    return this.h;
  }
}

class GruDef extends LayerDef {
  constructor(size, inputSize) {
    super(inputSize, size);
    this.name = 'gru';
    this.numWeights = 3 * size * (inputSize + size) + 3 * size;
  }
  build(w) {
    return new GruLayer(this, w);
  }
}

/** Standard GRU, gate order [r, z, n]. */
class GruLayer {
  constructor(def, w) {
    this.size = def.size;
    this.wIh = mat(w.w_ih);
    this.wHrz = mat(w.w_hrz);
    this.wHn = mat(w.w_hn);
    this.b = vec(w.b);
    this.h = new Float32Array(def.size);
    this.gi = new Float64Array(3 * def.size);
    this.hrz = new Float64Array(2 * def.size);
    this.hn = new Float64Array(def.size);
  }
  reset() {
    this.h.fill(0);
  }
  step(x) {
    const s = this.size;
    matVecAdd(this.wIh, x, this.b, this.gi);
    matVecAdd(this.wHrz, this.h, null, this.hrz);
    matVecAdd(this.wHn, this.h, null, this.hn);
    for (let i = 0; i < s; i++) {
      const r = sigmoid(this.gi[i] + this.hrz[i]);
      const z = sigmoid(this.gi[s + i] + this.hrz[s + i]);
      const n = Math.tanh(this.gi[2 * s + i] + r * this.hn[i]);
      this.h[i] = (1 - z) * n + z * this.h[i];
    }
    return this.h;
  }
}

class MgruDef extends LayerDef {
  constructor(size, inputSize) {
    super(inputSize, size);
    this.name = 'mgru';
    this.numWeights = 2 * (size * (inputSize + size) + size);
  }
  build(w) {
    return new MgruLayer(this, w);
  }
}

/**
 * Single-gate GRU variant (update = 1 - forget):
 *   f  = sigmoid(w_fx x + w_fh h + b_f)
 *   hc = tanh(w_hx x + w_hh (f*h) + b_h)
 *   h' = (1-f) * h + f * hc
 */
class MgruLayer {
  constructor(def, w) {
    this.size = def.size;
    this.wIh = mat(w.w_ih);
    this.wFh = mat(w.w_fh);
    this.wHh = mat(w.w_hh);
    this.b = vec(w.b);
    this.h = new Float32Array(def.size);
    this.gi = new Float64Array(2 * def.size);
    this.fh = new Float64Array(def.size);
    this.gated = new Float32Array(def.size);
    this.hh = new Float64Array(def.size);
    this.f = new Float64Array(def.size);
  }
  reset() {
    this.h.fill(0);
  }
  step(x) {
    const s = this.size;
    const f = this.f;
    matVecAdd(this.wIh, x, this.b, this.gi);
    matVecAdd(this.wFh, this.h, null, this.fh);
    for (let i = 0; i < s; i++) {
      f[i] = sigmoid(this.gi[i] + this.fh[i]);
      this.gated[i] = f[i] * this.h[i];
    }
    matVecAdd(this.wHh, this.gated, null, this.hh);
    for (let i = 0; i < s; i++) {
      const hc = Math.tanh(this.gi[s + i] + this.hh[i]);
      this.h[i] = (1 - f[i]) * this.h[i] + f[i] * hc;
    }
    return this.h;
  }
}

class MinGruDef extends LayerDef {
  constructor(size, inputSize) {
    super(inputSize, size);
    this.name = 'mingru';
    this.numWeights = 2 * size * inputSize + 2 * size;
  }
  build(w) {
    return new MinGruLayer(this, w);
  }
}

/** Input-only gate and candidate: h' = (1-z) h + z (W_h x + b_h). */
class MinGruLayer {
  constructor(def, w) {
    this.size = def.size;
    this.wIh = mat(w.w_ih);
    this.b = vec(w.b);
    this.h = new Float32Array(def.size);
    this.gates = new Float64Array(2 * def.size);
  }
  reset() {
    this.h.fill(0);
  }
  step(x) {
    const s = this.size;
    matVecAdd(this.wIh, x, this.b, this.gates);
    for (let i = 0; i < s; i++) {
      const z = sigmoid(this.gates[i]);
      this.h[i] = (1 - z) * this.h[i] + z * this.gates[s + i];
    }
    return this.h;
  }
}

const RGLRU_C = 8.0;   // recurrence-gate temperature (Griffin eq. 3)

class RglruDef extends LayerDef {
  constructor(blocks, inputSize) {
    super(inputSize, inputSize);   // dimension-preserving
    this.name = 'rglru';
    this.blocks = blocks;
    if (inputSize % blocks !== 0) {
      throw new Error(`rglru.${blocks} does not divide width ${inputSize}`);
    }
    this.blockWidth = inputSize / blocks;
    const gate = blocks * this.blockWidth * (this.blockWidth + 1);
    this.numWeights = this.size + 2 * gate;
  }
  build(w) {
    return new RglruLayer(this, w);
  }
}

/**
 * RG-LRU (Griffin / RecurrentGemma) with block-diagonal gates:
 *   r = sigmoid(W_a x + b_a),  i = sigmoid(W_x x + b_x)
 *   a = exp(-8 * r * softplus(lambda))
 *   h = a * h + sqrt(1 - a^2) * (i * x)
 * The very first step of a sequence is a reset: multiplier 1.
 */
class RglruLayer {
  constructor(def, w) {
    this.size = def.size;
    this.blocks = def.blocks;
    this.blockWidth = def.blockWidth;
    this.lam = vec(w.lam);
    // (blocks, bw, bw) and (blocks, bw), flattened.
    this.wIg = Float32Array.from(w.w_ig.flat(2));
    this.wRg = Float32Array.from(w.w_rg.flat(2));
    this.bIg = Float32Array.from(w.b_ig.flat());
    this.bRg = Float32Array.from(w.b_rg.flat());
    this.h = new Float32Array(def.size);
    this.out = new Float32Array(def.size);
    this.first = true;
  }
  reset() {
    this.h.fill(0);
    this.first = true;
  }
  step(x) {
    const bw = this.blockWidth;
    for (let k = 0; k < this.blocks; k++) {
      const base = k * bw;
      const wOff = k * bw * bw;
      for (let d = 0; d < bw; d++) {
        let ig = this.bIg[base + d];
        let rg = this.bRg[base + d];
        for (let c = 0; c < bw; c++) {
          // einsum '...kc,kcd->...kd'
          ig += x[base + c] * this.wIg[wOff + c * bw + d];
          rg += x[base + c] * this.wRg[wOff + c * bw + d];
        }
        const i = sigmoid(ig);
        const r = sigmoid(rg);
        const logA = -RGLRU_C * r * softplus(this.lam[base + d]);
        const a = Math.exp(logA);
        let mult = 1.0;
        if (!this.first) {
          const aSq = Math.exp(2 * logA);
          mult = Math.sqrt(Math.min(Math.max(1 - aSq, 1e-12), 1.0));
        }
        this.h[base + d] = a * this.h[base + d] + x[base + d] * i * mult;
      }
    }
    this.first = false;
    this.out.set(this.h);
    return this.out;
  }
}

class RmsNormDef extends LayerDef {
  constructor(inputSize) {
    super(inputSize, inputSize);
    this.name = 'rmsnorm';
    this.numWeights = inputSize;
  }
  build(w) {
    return new RmsNormLayer(this, w);
  }
}

/** Gemma-style RMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + gamma). */
class RmsNormLayer {
  constructor(def, w) {
    this.size = def.size;
    this.gamma = vec(w.gamma);
    this.out = new Float32Array(def.size);
  }
  reset() {}
  step(x) {
    let ms = 0;
    for (let i = 0; i < this.size; i++) ms += x[i] * x[i];
    ms /= this.size;
    const scale = 1 / Math.sqrt(ms + 1e-6);
    for (let i = 0; i < this.size; i++) {
      this.out[i] = x[i] * scale * (1 + this.gamma[i]);
    }
    return this.out;
  }
}

class NormDef extends LayerDef {
  constructor(inputSize) {
    super(inputSize, inputSize);
    this.name = 'norm';
    this.numWeights = 0;
  }
  build() {
    return new NormLayer(this);
  }
}

/** Parameter-free L2 normalisation (texmo/layers/norm.py). */
class NormLayer {
  constructor(def) {
    this.size = def.size;
    this.out = new Float32Array(def.size);
  }
  reset() {}
  step(x) {
    let ss = 0;
    for (let i = 0; i < this.size; i++) ss += x[i] * x[i];
    const scale = 1 / Math.sqrt(ss + 1e-12);
    for (let i = 0; i < this.size; i++) this.out[i] = x[i] * scale;
    return this.out;
  }
}

class SplitDef extends LayerDef {
  constructor(op, branches, inputSize) {
    const sizes = branches.map(
      (b) => (b.length ? b[b.length - 1].size : inputSize));
    const size = op === 'cat'
      ? sizes.reduce((a, b) => a + b, 0)
      : Math.max(...sizes);
    super(inputSize, size);
    this.name = 'split';
    if (branches.length !== 2) {
      // SplitDef.is_valid enforces 2-way in Python too; the merge
      // fold below writes into one buffer, so N > 2 would alias.
      throw new Error(`split needs exactly 2 branches, got ${branches.length}`);
    }
    this.op = op;
    this.branches = branches;
    this.branchSizes = sizes;
    this.length = branches.length
      ? Math.max(...branches.map(seqLength))
      : 1;
    this.numWeights = branches.reduce(
      (acc, b) => acc + b.reduce((a, l) => a + l.numWeights, 0), 0);
  }
  build(w) {
    return new SplitLayer(this, w);
  }
}

function seqLength(layers) {
  return 1 + layers.reduce((acc, l) => acc + l.length - 1, 0);
}

/**
 * Fork-and-merge. `add` / `mul` are element-wise on the overlap with
 * the longer side carried through unchanged; `cat` concatenates
 * (docs/split.md).
 */
class SplitLayer {
  constructor(def, weights) {
    this.size = def.size;
    this.op = def.op;
    this.branches = def.branches.map(
      (b, i) => b.map((layer, j) => layer.build(weights[i][j])));
    this.out = new Float32Array(def.size);
  }
  reset() {
    for (const branch of this.branches) for (const l of branch) l.reset();
  }
  step(x) {
    const outs = this.branches.map((branch) => {
      let v = x;
      for (const layer of branch) v = layer.step(v);
      return v;
    });
    let acc = outs[0];
    for (let i = 1; i < outs.length; i++) acc = this._merge(acc, outs[i]);
    return acc;
  }
  _merge(v, source) {
    const out = this.out;
    if (this.op === 'cat') {
      out.set(v, 0);
      out.set(source, v.length);
      return out;
    }
    const dv = v.length;
    const ds = source.length;
    const overlap = Math.min(dv, ds);
    for (let i = 0; i < overlap; i++) {
      out[i] = this.op === 'add' ? v[i] + source[i] : v[i] * source[i];
    }
    const longer = dv >= ds ? v : source;
    for (let i = overlap; i < longer.length; i++) out[i] = longer[i];
    return out;
  }
}

// ---------------------------------------------------------------------
// Codec (texmo/layers/one_hot_codec.py)
// ---------------------------------------------------------------------

/**
 * The fixed-codebook codec, tokenized-input arm: a one-hot over the
 * tokenset in, an independent learnable dense head out. The bit-chunk
 * arms (`bytes`, `bits.N[.oh][+bp]`) and the tied / hex-pair codecs
 * are not ported.
 */
class OneHotCodecDef {
  constructor(ntokens, variation) {
    this.ntokens = ntokens;
    this.variation = variation;
    this.size = ntokens;
    this.tokensName = `tokens.${ntokens}.${variation}`;
    this.head = null;
  }
  setHeadWidth(lastWidth) {
    this.head = new DenseDef(
      this.ntokens > 2 ? this.ntokens : 1, lastWidth, null);
  }
  get numWeights() {
    // The tokenset's own `extra_weights` surcharge is added by Model,
    // which is the only place that has the tokenset in hand.
    return this.head.numWeights;
  }
  toString() {
    return `tokens.${this.ntokens}.${this.variation}.oh`;
  }
  build(headWeights) {
    return new OneHotCodec(this, headWeights);
  }
}

function parseCodec(spec) {
  if (!spec.startsWith('tokens.')) {
    throw new Error(
      `input spec '${spec}' is not ported to JS: only ` +
      `tokens.N.variation.oh models run in the browser`);
  }
  const parts = spec.split('.');
  if (parts.length !== 4 || parts[3] !== 'oh') {
    throw new Error(`bad input spec: '${spec}'`);
  }
  return new OneHotCodecDef(parseInt(parts[1], 10), parts[2]);
}

class OneHotCodec {
  constructor(def, headWeights) {
    this.ntokens = def.ntokens;
    this.size = def.size;
    this.head = def.head.build(headWeights);
    this.padOutput = def.ntokens <= 2;
    this.vector = new Float32Array(def.size);
    this.uniform = new Float32Array(def.size).fill(1 / def.ntokens);
    this.logitsBuf = this.padOutput ? new Float32Array(2) : null;
  }
  /** Input vector for "no token observed yet" (max entropy). */
  initialVector() {
    return this.uniform;
  }
  encodeStep(token) {
    this.vector.fill(0);
    this.vector[token] = 1;
    return this.vector;
  }
  logitsStep(h) {
    const out = this.head.step(h);
    if (!this.padOutput) return out;
    this.logitsBuf[0] = out[0];
    this.logitsBuf[1] = 0;
    return this.logitsBuf;
  }
}

// ---------------------------------------------------------------------
// Model (texmo/model2_jax.py)
// ---------------------------------------------------------------------

/**
 * A running model: `reset()` starts a new sequence and returns the
 * position-0 logits; `step(tokenId)` consumes one token and returns
 * the logits for the next one.
 *
 * The state lives inside the layer objects, so one Model instance is
 * one sequence at a time. Building a Model is cheap (it only copies
 * the weights into typed arrays), and prefilling a few hundred tokens
 * of dialog costs a few milliseconds, so the chat re-prefills from
 * scratch each turn exactly as `texmo.chat` does.
 */
export class Model {
  /**
   * @param {object} manifest  the model JSON: {spec, precision, weights}
   * @param {TokenSet} [tokenset]  only used for the weight count
   */
  constructor(manifest, tokenset = null) {
    this.spec = manifest.spec;
    this.precision = manifest.precision;
    const { codec, layers } = parseSpec(manifest.spec);
    this.codecDef = codec;
    this.layerDefs = layers;
    this.tokensName = codec.tokensName;
    this.ntokens = codec.ntokens;

    // `total_padding` -- 1 + sum(length - 1). Every layer ported here
    // has length 1, so the prefix `initial_step` walks is a single
    // max-entropy position and prefill collapses to one `step` from
    // zero state. A length > 1 layer would need the real prefill.
    this.totalPadding = seqLength(layers);
    if (this.totalPadding !== 1) {
      throw new Error(
        `spec needs prefill over ${this.totalPadding} positions, ` +
        `which the JS port does not implement`);
    }

    const weights = manifest.weights;
    this.layers = layers.map((def, i) => def.build(weights[1][i]));
    this.codec = codec.build(weights[2]);

    this.numWeights = codec.numWeights
      + layers.reduce((a, l) => a + l.numWeights, 0)
      + (tokenset ? tokenset.extraWeights : 0);

    this.reset();
  }

  /** Start a new sequence; returns the logits before any input. */
  reset() {
    for (const layer of this.layers) layer.reset();
    this.codec.head.reset();
    return this._run(this.codec.initialVector());
  }

  /** Consume one token; returns the logits for the next position. */
  step(token) {
    return this._run(this.codec.encodeStep(token));
  }

  _run(v) {
    for (const layer of this.layers) v = layer.step(v);
    return this.codec.logitsStep(v);
  }
}

// ---------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------

/** mulberry32: a small, seedable PRNG returning floats in [0, 1). */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function random() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function argmax(logits) {
  let best = 0;
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > logits[best]) best = i;
  }
  return best;
}

/** Temperature-softmax sample over the logits. */
export function sampleLogits(logits, temperature, random) {
  if (temperature <= 0) return argmax(logits);
  const n = logits.length;
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = logits[i] / temperature;
    if (v > max) max = v;
  }
  const probs = new Float64Array(n);
  let total = 0;
  for (let i = 0; i < n; i++) {
    const p = Math.exp(logits[i] / temperature - max);
    probs[i] = p;
    total += p;
  }
  let target = random() * total;
  for (let i = 0; i < n; i++) {
    target -= probs[i];
    if (target <= 0) return i;
  }
  return n - 1;
}

// ---------------------------------------------------------------------
// Dialog (texmo/chat.py)
// ---------------------------------------------------------------------

export const TURN_SEPARATOR = '\n\n';
/** What a byte that is not (yet) a character decodes to. */
export const REPLACEMENT = String.fromCharCode(0xfffd);

/** The dialog with the user's turn appended and the bot's opened. */
export function buildPrompt(dialog, utterance, userName = 'User', botName = 'Bot') {
  return `${dialog}${userName}: ${utterance}${TURN_SEPARATOR}${botName}: `;
}

/** The dialog history after the bot's reply, normalized. */
export function appendReply(prompt, reply) {
  return prompt + reply.trim() + TURN_SEPARATOR;
}

function commonPrefixLen(a, b) {
  let n = 0;
  while (n < a.length && n < b.length && a[n] === b[n]) n++;
  return n;
}

/** Strip trailing U+FFFD and '\n' characters (Python's rstrip set). */
function rstripPartial(s) {
  let end = s.length;
  while (end > 0 && (s[end - 1] === '\n' || s[end - 1] === REPLACEMENT)) end--;
  return s.slice(0, end);
}

/**
 * Consume sampled tokens up to the turn boundary or the cap --
 * `chat.collect_reply`, turned inside out so a UI can feed it one
 * token at a time and still stream.
 *
 * Streamed text can never be taken back, so anything the next token
 * could still change is held back: a trailing replacement character
 * (the head of a multi-byte character), a trailing newline (which may
 * turn out to be the boundary's first half), and any delta that does
 * not extend what was already streamed. Whatever is held back is
 * flushed by `finish()`.
 */
export class ReplyCollector {
  constructor(tokenizer, maxTokens, onDelta = null) {
    this.tokenizer = tokenizer;
    this.maxTokens = maxTokens;
    this.onDelta = onDelta;
    this.ids = [];
    this.text = '';
    this.shown = '';
  }

  /** Feed one sampled token; returns true when the reply is complete. */
  push(token) {
    this.ids.push(token);
    try {
      this.text = this.tokenizer.untokenizeText(this.ids);
    } catch {
      // A tokenizer that raises mid group: the previous text stands.
    }
    if (this.onDelta) {
      const visible = rstripPartial(this.text.split(TURN_SEPARATOR)[0]);
      if (visible !== this.shown && visible.startsWith(this.shown)) {
        this.onDelta(visible.slice(this.shown.length));
        this.shown = visible;
      }
    }
    return this.text.includes(TURN_SEPARATOR)
      || this.ids.length >= this.maxTokens;
  }

  /** Flush the held-back tail; returns {text, ntokens}. */
  finish() {
    const visible = this.text.split(TURN_SEPARATOR)[0];
    if (this.onDelta) {
      const tail = visible.slice(commonPrefixLen(this.shown, visible));
      if (tail) this.onDelta(tail);
    }
    return { text: this.text, reply: visible, ntokens: this.ids.length };
  }
}

/**
 * Prefill `prompt` into `model` and sample a reply. Synchronous --
 * the browser UI drives `model.step` itself so it can yield to the
 * event loop; this is the convenience path for scripts and tests.
 */
export function generateReply(
  model, tokenizer, prompt, { maxTokens = 256, temperature = 0.4,
    random = Math.random, onDelta = null } = {},
) {
  const ids = tokenizer.tokenize(prompt);
  model.reset();
  for (let i = 0; i < ids.length - 1; i++) model.step(ids[i]);
  let c = ids[ids.length - 1];
  const collector = new ReplyCollector(tokenizer, maxTokens, onDelta);
  for (;;) {
    const logits = model.step(c);
    c = sampleLogits(logits, temperature, random);
    if (collector.push(c)) break;
  }
  return collector.finish();
}
