# TexMo

TexMo searches for good small language-model architectures and
metaparameters, starting from the smallest possible models (a handful
of weights) and growing upward. Instead of fixing a single
architecture and scaling it, it explores the joint space of
architectures and hyperparameters, training a large number of small
configurations and using their results to predict where to look next.

The implementation is primarily targeting **JAX** (a PyTorch backend
also exists; see [`docs/backends.md`](docs/backends.md)).

## What "good" means here

For every weight budget the search tracks the best cross-entropy loss
achieved on held-out text from an English-language dataset, measured
per input *byte* — the model reads raw UTF-8, no tokenizer. The plot
below summarises ~560k training runs across ~290k distinct
configurations collected so far.

![Loss vs. weights](docs/images/graph.png)

Best models found so far at a few sample weight budgets — see
[`docs/best_models.md`](docs/best_models.md) for the full list:

| Weights | Loss   | Spec                                                                                          |
|--------:|-------:|-----------------------------------------------------------------------------------------------|
| 50      | 4.7796 | `bits.1+bp\|skip.2.add-mgru.1-skip.2.cat-dense.4.tanh-norm-skip.2.cat-suffix.2-dense.1.tanh`  |
| 97      | 4.4743 | `bits.1+bp\|mgru.4-dense.4.gelu`                                                              |
| 200     | 4.1061 | `bits.4.oh+bp\|latent.2.2-suffix.2-mingru.4-norm-skip.2.add-mingru.4-norm`                    |
| 392     | 3.8028 | `bits.4.oh+bp\|dense.4.tanh-skip.5.add-suffix.2-mgru.4-skip.2.add-suffix.2-skip.1.add-dense.8.gelu` |

A spec has the form `<input>|<layer>-<layer>-...`. The part before `|`
describes how an input byte is fed to the network: `bits.1` feeds the
byte one bit at a time, `bits.4.oh` feeds it in two 4-bit chunks
one-hot-encoded, and the `+bp` suffix adds a positional encoding for
the bit position within the byte. The part after `|` is the sequence
of layers applied at each step (see [`docs/layers.md`](docs/layers.md)
for the full set). `skip.X.add` / `skip.X.cat` are residual
connections that fold the output of `X` layers back into the stream by
addition or concatenation (see [`docs/skip.md`](docs/skip.md)).

## Why no tokens

TexMo trains directly on bytes (and bit-prefixes), not on a learned
tokenization, for two reasons:

1. **Tokens encode language knowledge.** A BPE vocabulary trained on a
   large corpus already captures word frequencies and common
   morphology. Counting those bytes-of-text per byte-of-model is
   misleading: part of the language model lives in the tokenizer.
   Training on raw bytes forces the model itself to learn everything
   it knows.
2. **Multiple tokenizations would blow up the search space.** The
   architecture search already covers many degrees of freedom; adding
   "which tokenizer" as another axis would multiply the work without
   informing the comparison between architectures.

The tokenization code in the repo still works and is documented in
[`docs/tokens.md`](docs/tokens.md); it is currently parked.

## Architectures don't scale equally

Per-system throughput at the same near-best loss target:

![Throughput across systems](docs/images/throughput.png)

For these tiny weight budgets the GPU (5060ti) is **outright slower**
than the Apple-silicon laptop (m5, running on the CPU — not the GPU
via Metal) and even the Raspberry Pi 5 — except for fixed-suffix
convolution-like models (`suffix.N` + dense), which the GPU handles
well. The reason is that the majority of best-performing
configurations at this scale are **recurrent** (RNN / GRU / mGRU /
minGRU / LSTM / latent-state), and recurrent layers parallelise poorly
across the sequence dimension. CPUs lose less to a small recurrent
step than GPUs lose to under-used SMs.

## Implemented layers

The search composes models from these building blocks (see
[`docs/layers.md`](docs/layers.md) for equations and parameter counts):

- `dense.<size>.<activation>` — feed-forward (tanh, gelu activations)
- `rnn.<size>.<activation>` — vanilla recurrent
- `gru.<size>`, `mgru.<size>`, `mingru.<size>` — gated recurrent variants
- `lstm.<size>`
- `latent.<size>.<reps>`, `lrnn.<size>.<reps>` — latent-state recurrent, `lrnn`
  includes two dimensions of recurrence
- `suffix.<length>` — stack the last few positions
- `norm` — layer normalisation
- `skip.<X>.<add|cat>` — residual connections (see
  [`docs/skip.md`](docs/skip.md))

For the full pipeline (input encodings, layer stacking, two backends),
see [`docs/architecture.md`](docs/architecture.md).

## How the search picks candidates

The space of architectures and metaparameters is far too large to
enumerate, so TexMo uses **local search**: it picks a known-good
configuration, generates its *neighbours* (layer-size mutations,
type swaps, append/remove a layer, precision tweaks — see
[`docs/search.md`](docs/search.md)), and trains promising ones. Two
auxiliary regression models decide which neighbours are promising
before paying for a real training run:

- A **timing model** that predicts per-step training time from
  architecture features and the target system. See
  [`docs/timing.md`](docs/timing.md).
- A **loss-prediction model** (an RNN trained on the result database
  itself) that estimates final loss for unseen configurations. See
  [`docs/loss_prediction.md`](docs/loss_prediction.md) and the
  experiment notes in
  [`docs/loss_rnn_experiments.md`](docs/loss_rnn_experiments.md).

The full distributed loop — server, workers, neighbour generation,
coverage walks across systems — is described in
[`docs/search.md`](docs/search.md).

## Metaparameters

The search currently optimises over:

- **Model architecture** — layer sequence and per-layer sizes
- **Batch size**
- **Sample length** (context window)
- **Steps**
- **Learning rate**

These are implemented and accepted by the trainer but **not actively
varied by the search**:

- **LR decay schedule** (`none` / `exp` / `cosine`) — defaults to
  cosine; see
  [`docs/decay_and_checkpoints.md`](docs/decay_and_checkpoints.md)
- **Precision** (`fp32` / `fp16` / `bf16` / `fp64`) — defaults to fp32

## Setup

Before running anything, copy `config_sample.py` to `config.py` and
edit the paths (training data location, DB path, system name, server
address, backend). `config.py` is gitignored — every host gets its
own.

## Training a single model

To train one configuration end-to-end and see its loss curve:

```
uv run texmo.py train -s 'bits.1+bp|mgru.4-dense.4.gelu' -b 32 --length 128 --steps 1024 --lr 0.01
```

`-s` is the model spec (see the [conventions](#what-good-means-here)
above), `-b` is the batch size, `--length` is the context window in
bytes, and `--lr` is the (initial) learning rate. See
[`docs/architecture.md`](docs/architecture.md) for the full pipeline.

## Running the search

Server on one machine:

```
uv run texmo.py search -t 1-120 -w 5-800
```

Worker(s) on the same or other machines:

```
uv run texmo.py client
```

Configure system name and server address in `config.py`. The web UI on
port 5000 exposes filters, the Pareto plot, the fastest-near-best
report, and per-system throughput.

## Status

This is a research playground, not a product. Roadmap entries live in
[`docs/roadmap.md`](docs/roadmap.md).
