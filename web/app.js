/**
 * The chat page: fetches a model manifest and its tokenset, then runs
 * the dialog loop from `texmo.js` against them.
 *
 * All model logic lives in `texmo.js`; this file is the UI. It drives
 * `model.step` itself (rather than calling `generateReply`) so it can
 * hand the event loop back between chunks -- the model is tiny, but a
 * few hundred prefill steps in one synchronous burst would still show
 * up as a dropped frame.
 */
import {
  Model, ReplyCollector, TokenSet, Tokenizer,
  appendReply, buildPrompt, mulberry32, parseSpec, sampleLogits,
} from './texmo.js';

// Data paths are relative to the page and carry no prefix; `fetchData`
// works out where `models/` and `tokens/` actually sit (see below).
const MODELS = {
  mg12k: {
    label: 'mg-12k-s5',
    manifest: 'models/mg-12k-s5.json',
    temperature: 0.4,
    figure: 'fig-mg12k',
  },
  hb32: {
    label: 'hb32-8k-s5',
    manifest: 'models/hb32-8k-s5.json',
    temperature: 0.3,
    figure: 'fig-hb32',
  },
  rl32: {
    label: 'rl32-8k-s5',
    manifest: 'models/rl32-8k-s5.json',
    temperature: 0.4,
    figure: 'fig-rl32',
  },
};

const MAX_REPLY_TOKENS = 256;
// Steps between yields to the browser. Prefill runs as fast as it can;
// generation is paced so the reply visibly types out.
const PREFILL_CHUNK = 64;
const STREAM_CHUNK = 4;

const el = (id) => document.getElementById(id);
const ui = {
  model: el('model'),
  temperature: el('temperature'),
  newChat: el('new-chat'),
  transcript: el('transcript'),
  composer: el('composer'),
  utterance: el('utterance'),
  send: el('send'),
  status: el('status'),
  footerModel: el('footer-model'),
  footerWeights: el('footer-weights'),
};

/**
 * Hand the event loop back for one paint. Races a timer against the
 * animation frame: rAF alone stalls wherever frames are not being
 * produced (a background tab, a headless run), and the reply would
 * hang there instead of merely rendering late.
 */
function frame() {
  return new Promise((resolve) => {
    let done = false;
    const go = () => {
      if (done) return;
      done = true;
      resolve();
    };
    requestAnimationFrame(go);
    setTimeout(go, 32);
  });
}

const state = {
  key: null,
  model: null,
  tokenizer: null,
  dialog: '',
  busy: false,
  random: mulberry32((Math.random() * 2 ** 32) >>> 0),
};

const modelCache = new Map();
const tokenizerCache = new Map();

// ---------------------------------------------------------------------
// loading
// ---------------------------------------------------------------------

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url}: HTTP ${response.status}`);
  }
  return response.json();
}

// Two layouts, no build step: published, the page sits at the site
// root with `models/` and `tokens/` beside it; in the repo it sits in
// `web/` with the same two directories one level up. Try the deployed
// layout first, fall back to the repo one, and remember which
// answered -- so a page load costs at most one stray 404.
const DATA_PREFIXES = ['', '../'];
let dataPrefix = null;

async function fetchData(path) {
  if (dataPrefix !== null) return fetchJson(dataPrefix + path);
  let last = null;
  for (const prefix of DATA_PREFIXES) {
    try {
      const json = await fetchJson(prefix + path);
      dataPrefix = prefix;
      return json;
    } catch (error) {
      last = error;
    }
  }
  throw last;
}

async function loadTokenizer(name) {
  if (!tokenizerCache.has(name)) {
    const json = await fetchData(`tokens/${name}.json`);
    tokenizerCache.set(name, new Tokenizer(new TokenSet(json)));
  }
  return tokenizerCache.get(name);
}

async function loadModel(key) {
  if (modelCache.has(key)) return modelCache.get(key);
  const config = MODELS[key];
  const manifest = await fetchData(config.manifest);
  // The spec names the tokenset, so a different manifest just works.
  const tokensName = parseSpec(manifest.spec).codec.tokensName;
  const tokenizer = await loadTokenizer(tokensName);
  const entry = {
    config,
    tokenizer,
    model: new Model(manifest, tokenizer.tokenset),
  };
  modelCache.set(key, entry);
  return entry;
}

async function activate(key) {
  const config = MODELS[key];
  setBusy(true);
  setStatus(`Loading ${config.label}…`);
  try {
    const entry = await loadModel(key);
    state.key = key;
    state.model = entry.model;
    state.tokenizer = entry.tokenizer;
    ui.temperature.value = String(config.temperature);
    for (const [otherKey, other] of Object.entries(MODELS)) {
      el(other.figure).hidden = otherKey !== key;
    }
    ui.footerModel.textContent = config.label;
    ui.footerWeights.textContent = entry.model.numWeights.toLocaleString('en-US');
    newChat();
    setStatus(
      `${config.label}: ${entry.model.numWeights.toLocaleString('en-US')} `
      + `weights, ${entry.model.ntokens}-token vocabulary. Ready.`);
  } catch (error) {
    setStatus(loadHint(error), true);
    return;
  } finally {
    setBusy(false);
  }
}

function loadHint(error) {
  if (location.protocol === 'file:') {
    return 'Cannot load the model over file://. Serve the repository '
      + 'over HTTP instead: `python -m http.server` in the repo root, '
      + 'then open /web/.';
  }
  return `Could not load the model (${error.message}).`;
}

// ---------------------------------------------------------------------
// transcript
// ---------------------------------------------------------------------

function addTurn(who, cls, text) {
  const turn = document.createElement('div');
  turn.className = `turn ${cls}`;
  const label = document.createElement('span');
  label.className = 'who';
  label.textContent = who;
  const said = document.createElement('span');
  said.className = 'said';
  said.textContent = text;
  turn.append(label, said);
  ui.transcript.append(turn);
  scrollDown();
  return { turn, said };
}

function scrollDown() {
  ui.transcript.scrollTop = ui.transcript.scrollHeight;
}

function newChat() {
  state.dialog = '';
  ui.transcript.replaceChildren();
  const hint = document.createElement('p');
  hint.className = 'empty';
  // Called once before the first model finishes loading, then again on
  // every activate() and New chat, when the count is known.
  const size = state.model
    ? `${state.model.numWeights.toLocaleString('en-US')} weights`
    : 'a few thousand weights';
  hint.textContent =
    `Say hello. This model has ${size} and was trained on short, `
    + 'simplified dialogs, so it does best with greetings, small talk '
    + 'and very short questions.';
  ui.transcript.append(hint);
}

function setStatus(text, isError = false) {
  ui.status.textContent = text;
  ui.status.classList.toggle('error', isError);
}

function setBusy(busy) {
  state.busy = busy;
  const ready = !busy && state.model !== null;
  ui.utterance.disabled = !ready;
  ui.send.disabled = !ready;
  ui.model.disabled = busy;
  ui.newChat.disabled = busy;
}

// ---------------------------------------------------------------------
// the turn
// ---------------------------------------------------------------------

function temperature() {
  const value = Number.parseFloat(ui.temperature.value);
  if (!Number.isFinite(value) || value <= 0) {
    return MODELS[state.key].temperature;
  }
  return Math.min(Math.max(value, 0.05), 3);
}

async function respond(utterance) {
  const { model, tokenizer } = state;
  const hint = ui.transcript.querySelector('.empty');
  if (hint) hint.remove();
  addTurn('You', 'user', utterance);

  const prompt = buildPrompt(state.dialog, utterance);
  const ids = tokenizer.tokenize(prompt);
  const { turn, said } = addTurn('Bot', 'bot pending', '');

  // -- prefill: the whole dialog, re-tokenized from scratch. The
  // segmentation is context-dependent, so an incremental feed could
  // split a token across the append boundary (see texmo/chat.py).
  setStatus(`Reading ${ids.length} tokens of dialog…`);
  let compute = 0;
  let logits = model.reset();
  for (let i = 0; i < ids.length; i += PREFILL_CHUNK) {
    const start = performance.now();
    const end = Math.min(i + PREFILL_CHUNK, ids.length);
    for (let j = i; j < end; j++) logits = model.step(ids[j]);
    compute += performance.now() - start;
    await frame();
  }
  const prefillMs = compute;

  // -- sample the reply, stopping at the next turn boundary.
  const collector = new ReplyCollector(
    tokenizer, MAX_REPLY_TOKENS, (delta) => {
      said.textContent += delta;
      scrollDown();
    });
  const t = temperature();
  let token = sampleLogits(logits, t, state.random);
  let generated = 0;
  compute = 0;
  for (;;) {
    if (collector.push(token)) break;
    const start = performance.now();
    logits = model.step(token);
    token = sampleLogits(logits, t, state.random);
    compute += performance.now() - start;
    generated++;
    if (generated % STREAM_CHUNK === 0) await frame();
  }
  const result = collector.finish();
  turn.classList.remove('pending');
  if (!said.textContent) said.textContent = '(nothing)';

  state.dialog = appendReply(prompt, result.text);

  const steps = Math.max(ids.length + generated, 1);
  const totalMs = prefillMs + compute;
  setStatus(
    `${ids.length} prompt tokens + ${result.ntokens} generated `
    + `at T = ${t}; ${totalMs.toFixed(totalMs < 10 ? 1 : 0)} ms of `
    + `compute, ${(totalMs / steps).toFixed(3)} ms/token.`);
}

// ---------------------------------------------------------------------
// wiring
// ---------------------------------------------------------------------

ui.composer.addEventListener('submit', async (event) => {
  event.preventDefault();
  const utterance = ui.utterance.value.trim();
  if (!utterance || state.busy || !state.model) return;
  ui.utterance.value = '';
  setBusy(true);
  try {
    await respond(utterance);
  } catch (error) {
    setStatus(`Generation failed: ${error.message}`, true);
    throw error;
  } finally {
    setBusy(false);
    ui.utterance.focus();
  }
});

ui.model.addEventListener('change', () => activate(ui.model.value));

ui.newChat.addEventListener('click', () => {
  if (state.busy) return;
  newChat();
  setStatus('New chat.');
  ui.utterance.focus();
});

newChat();
activate(ui.model.value);
