# Training backends: PyTorch vs JAX

Training runs on **JAX**, and JAX is now the only backend in the tree.

**The PyTorch backend was REMOVED on 2026-08-04.** It went in two
stages:

- 2026-07 — the full-model torch runtime was retired along with the
  legacy `Model`/`ModelDef` flat representation. `ManagerTorch` became
  a parked stub raising `NotImplementedError`, but the per-layer
  `{Name}Module` (`nn.Module`) implementations stayed, maintained and
  tested.
- 2026-08-04 — the `*Module` classes, `LayerModule`, `build_module()`,
  `ManagerTorch`, `Precision.dtype`, and the `torch` dependency itself
  were all deleted.

Two things decided it. First, the torch side only ever covered the
**legacy layer subset** (dense, rnn, gru/mgru/mingru, lstm, latent/lrnn,
msr, norm, suffix, and the bit/byte inputs). Every layer added since —
mingru's newer cousins, conv, rglru, matlstm, mullstm, slstm, lmgu,
attn, split, rmsnorm, and both codecs — is JAX-only, so the torch path
could not run a modern search conf even in principle. Second, torch's
bundled cudnn DLLs collide with the winjax CUDA plugin on Windows, and
2–4 GB of CUDA torch per machine bought nothing.

The full history is in git. What follows is the record of *why* JAX won
for this workload — and what a revived torch backend would be up
against. `scripts/bench_torch.py` survives as a standalone
cross-framework reference (run it with `uv run --with torch ...`); it
is what produced the `PyTorch + ...` rows in `bench.txt`.

## Why two backends?

PyTorch and JAX have complementary strengths for this project's workload:
architecture search over small recurrent models on heterogeneous hardware.

### PyTorch strengths

- **Hardware support**: CUDA on Windows, Metal on Mac, CPU everywhere.
- **Fused kernels**: `nn.RNN`, `nn.GRU`, `nn.LSTM` use cuDNN/oneDNN
  fused implementations that are much faster than Python-level loops.
- **Ecosystem**: mature tooling, debugging, profiling.

### PyTorch weaknesses

- **Custom recurrent layers are slow.** Layers not covered by a built-in
  fused kernel (GELU-activated RNN, mGRU, minGRU) had to be implemented
  as Python `for` loops over the sequence length. Each timestep dispatches
  separate kernels, so overhead dominates for small hidden sizes. This
  is the decisive one: every layer added after the initial set was a
  custom cell with no fused kernel behind it.
- **`torch.compile` doesn't help.** The inductor backend unrolls the
  Python loop, producing a huge graph that takes minutes to compile and
  still doesn't fuse into a single kernel. Recompiles on every new
  sequence length (e.g. eval vs train).
- **`torch.scan` is broken.** The experimental `torch._higher_order_ops.scan`
  works in eager mode (but is 10x *slower* than a plain loop due to
  tracing overhead) and crashes under `torch.compile` (as of PyTorch 2.11).
- **`torch.jit.script`** compiles the loop as a real loop (no unrolling)
  but doesn't fuse per-step ops into a single kernel. Measured ~1.3x
  speedup — not enough to close the gap.

### JAX strengths

- **`lax.scan` + `jit` just works.** The scan body is compiled once into
  a single fused kernel with an internal loop. No unrolling, no
  recompilation on shape changes, near-optimal performance for any
  custom recurrent cell.
- **Custom layers are fast.** A hand-written mGRU in JAX (scan + jit)
  runs at comparable speed to PyTorch's fused `nn.GRU`.

### JAX weaknesses (as of the original comparison)

- **No Windows GPU support.** Upstream JAX has no CUDA wheels for
  Windows. Since closed locally by
  [winjax](https://github.com/eterevsky/winjax), a PJRT plugin loader
  that registers CUDA on import.
- **No Metal support.** Upstream JAX doesn't run on Apple GPUs. Since
  closed by [metaljax](https://github.com/eterevsky/metaljax), an
  MLX-based backend.

Both gaps closing is the other half of why the torch backend stopped
earning its keep: the hardware-coverage argument for keeping it around
had evaporated.

## Benchmark: loop overhead

Same model (`bits.1+bp|rnn.1.tanh`, 8 weights, batch 64, seq_len 256,
1024 steps, fp32, CPU):

| Implementation              | Time    |
|-----------------------------|---------|
| PyTorch `nn.RNN` (fused)    | ~6 s    |
| PyTorch loop + jit.script   | ~16 s   |
| PyTorch loop (eager)        | ~48 s   |

The 8x gap between fused and eager is almost entirely kernel dispatch
overhead — the actual compute per step is negligible at hidden size 1.

For mGRU (no fused kernel available):

| Implementation              | Time     |
|-----------------------------|----------|
| PyTorch `nn.GRU` (fused)    | ~10 s    |
| PyTorch mGRU loop (eager)   | ~2 min   |

## What this means now

- **Architecture search over small models (<1M params):** JAX, which is
  what the project does. Custom recurrent layers are the main
  exploration axis, and JAX makes them all equally fast.
- **Larger experiments on GPU:** the fused-kernel argument for PyTorch
  still holds in the abstract, but it only covers the standard cells
  (GRU, LSTM, RNN), which are not where this project spends its time.
  Reviving it would mean writing a Model2-shaped runtime and torch
  implementations for ~15 layers, not resurrecting what was deleted.
- **Cross-verification:** was a real benefit of two backends and is
  gone. `scripts/bench_torch.py` is the remaining cross-framework
  sanity point, at the whole-model level rather than per layer.
