# web/ — the in-browser chat page

A static page that talks to a Texmo chat model with **inference
implemented in JavaScript, in the browser**. No server, no build step,
no external resources: the page fetches a model manifest from
`models/` and a tokenset from `tokens/` over relative paths and runs
the whole thing — tokenizer, recurrent step, sampling — locally.

It is published with GitHub Pages, where the contents of `web/` land
at the site *root* and `models/` and `tokens/` sit beside them. In the
repository the same two directories are one level up instead, so the
loader tries `models/…` first and falls back to `../models/…`,
remembering which answered. Both layouts work with no build step and
no configuration; the only cost is one 404 in the server log per page
load in the repository layout.

## Serving it locally

`fetch` does not work from `file://`, so open the page over HTTP:

```sh
python -m http.server 8000        # from the repository ROOT, not web/
```

then <http://localhost:8000/web/>. Any static server works; the only
requirement is that `models/` and `tokens/` are served from the same
tree.

## Files

| file | what it is |
| --- | --- |
| `index.html` | the page: chat column plus a hand-authored SVG schematic per model |
| `style.css` | one accent colour, light/dark from `prefers-color-scheme` |
| `app.js` | the UI: loading, the turn loop, streaming, the model switch |
| `texmo.js` | the port itself — tokenizer, spec parser, layers, model, dialog logic. No dependencies, no DOM |
| `export_vectors.py` | regenerates `test/vectors.json` from the Python engine |
| `texmo_test.mjs` | the parity test (`node web/texmo_test.mjs`) |
| `test/vectors.json` | reference token ids, logits and greedy continuations |
| `package.json` | a `"type": "module"` marker so Node treats `texmo.js` as ESM. Nothing is installed from it |

## Models

Two chat models, switchable from the page's dropdown. Both use the
`tokens.32.hexbpe` tokenset and the `User:` / `Bot:` turn format, and
both were trained on the `s5` corpus (LLM-simplified SODA dialogs plus
greetings, farewells and thanks).

| model | weights | spec | recommended T |
| --- | --- | --- | --- |
| `models/rl32-8k-s5.json` | 8,017 | `rnn.32.gelu` + `rglru.16` + `lstm.16` | 0.4 |
| `models/hb32-8k-s5.json` | 8,001 | `rnn.32.gelu` + `rglru.16` + `mgru.16` | 0.3 |

Selecting a model loads its JSON on first use, resets the chat and
sets the temperature field to that model's recommendation.

**Both manifests must be committed** for Pages to serve them —
`models/` is otherwise mostly untracked.

## Publishing

`.github/workflows/pages.yml` deploys on every push to `master` that
touches `web/`, `models/` or `tokens/` (or the workflow itself). It
copies `web/` — minus the dev-only files — to the root of a `_site/`
tree, adds `models/` and `tokens/` as subdirectories, and uploads that
as the Pages artifact.

One manual step, once: in the repository's **Settings → Pages**, set
the source to **GitHub Actions**. The site is then
<https://eterevsky.github.io/texmo/>; a custom domain pointed at the
repository later serves the same tree at its own root, with no change
here.

## Verifying the port

`texmo.js` reimplements the engine, so it is pinned to it by
reference vectors rather than by review:

```sh
uv run python web/export_vectors.py     # regenerate test/vectors.json
node web/texmo_test.mjs                 # assert the JS matches
```

`export_vectors.py` loads each manifest exactly as `texmo.py chat`
does (`model_store.load_model` → `Model2Jax.initial_step` / `step`)
and, for six prompts — a multi-turn dialog, running lowercase text
with no sentence punctuation, dense punctuation, mixed-case and
ALL-CAPS words, non-ASCII characters, and plain prose — records the
token ids, the position-0 logits, one logit row per prompt position
(first 64) and a 32-token greedy continuation.

`texmo_test.mjs` replays them through `texmo.js` and asserts token ids
are byte-exact, `untokenize` round-trips identically, logits agree
within `1e-4` (relative above magnitude 1, absolute below) and the
greedy continuations match token for token. Rerun both after touching
anything in `texmo.js`, and rerun the export after touching a layer,
the codec or the tokenizer on the Python side.

## What the JS port covers

Enough for the published models, not the whole engine:

- **Tokenizer**: the `capswords2` text preprocessing and the generic
  dynamic-programming tokenizer over tokens + sequences — which is
  what `registry.get_tokenizer` hands out for `hexbpe` sets at
  sampling time. `fold`, `shift`, `shift_bucket`, the SentencePiece
  BPE loop and the `bits.N` chunkers are not ported.
- **Codec**: `tokens.N.variation.oh` only. The tied-embedding and
  hex-pair codecs and the bit-chunk inputs are not ported.
- **Layers**: `dense`, `rnn`, `lstm`, `gru`, `mgru`, `mingru`,
  `rglru`, `rmsnorm`, `norm`, `split.{add,cat,mul}` and layer
  sequences. Anything else raises at parse time with the offending
  spec fragment.
- **Prefill**: only specs whose `total_padding` is 1, i.e. no
  multi-position layer (`conv`, `suffix`, `attn`, `msr`). Such a spec
  is rejected rather than run approximately.

A manifest outside that envelope fails loudly when the page loads it;
it does not silently compute something else.

## Deliberate deviations from the Python

Recorded here because they are the places the two could drift:

- Weights and layer outputs are `Float32Array` (JAX's fp32), but the
  arithmetic inside a matmul accumulates in JS doubles and rounds once
  on store — slightly *more* accurate than XLA, not less. Measured
  worst deviation across both models and all reference positions:
  ~3e-5 relative.
- A `Model` owns its recurrent state instead of threading it
  functionally; one instance is one running sequence, and `reset()`
  starts a new one. Every turn re-tokenizes and re-prefills the whole
  dialog, exactly as `texmo/chat.py` does — tokenization is
  context-dependent, so an incremental feed could split a token across
  the append boundary.
- Sampling uses `mulberry32` rather than JAX's PRNG, so sampled text
  is not reproducible between the two. Greedy decoding is, which is
  what the test checks.
