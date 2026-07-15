# Training backends: PyTorch vs JAX

Training runs on **JAX**. The PyTorch full-model backend was retired
with the legacy `Model`/`ModelDef` representation (2026-07):
`--backend torch` now hits a parked `ManagerTorch` stub, though the
per-layer `nn.Module` implementations are still maintained and tested.
The trade-offs below are the record of why JAX won for this workload
— and what a revived torch backend would be up against.

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
  fused kernel (GELU-activated RNN, mGRU, minGRU) must be implemented
  as Python `for` loops over the sequence length. Each timestep dispatches
  separate kernels, so overhead dominates for small hidden sizes.
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

### JAX weaknesses

- **No Windows GPU support.** JAX doesn't support CUDA on Windows.
- **No Metal support.** JAX doesn't run on Apple GPUs.
- **CPU-only on Windows and Mac**, which is acceptable for small-model
  search but limits larger experiments.

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

## Recommended usage

- **Architecture search over small models (<1M params):** use JAX.
  Custom recurrent layers are the project's main exploration axis,
  and JAX makes them all equally fast.
- **Larger experiments on GPU:** use PyTorch. Fused kernels for
  standard layers (GRU, LSTM, RNN) are hard to beat, and GPU memory
  management is more mature.
- **Cross-verification:** run the same configuration on both backends
  to verify correctness and compare loss curves.
