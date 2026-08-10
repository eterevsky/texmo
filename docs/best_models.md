# Current best models

Pareto-best configurations by weight count: for each weight count, the
conf with the lowest median loss that also beats every lighter conf.
Loss is per-byte cross-entropy on the held-out set; time is the median
wall-clock training time on the system that trains it fastest. Every LR
schedule the search runs is included -- `a↘0` is cosine-to-zero, `a→b`
exponential decay from `a` to `b`, a bare `a` a constant rate.

**A dated snapshot of a moving target.** The search adds runs
continuously, so this table is stale the moment it is written; it is
kept in the repo as a readable checkpoint, not as the source of truth.
Snapshot of 179 rows taken on 2026-08-09 (confs with at least
2 runs, up to 3000 weights). Regenerate with:

    uv run python scripts/make_best_models.py

Spec conventions are explained in the [README](../README.md#what-good-means-here);
the layer types are documented in [`layers.md`](layers.md) and the input
encodings in [`io.md`](io.md).

| Weights | Loss | Runs | Time | Batch×Len | LR | Steps | P | Spec |
|--------:|-----:|-----:|-----:|----------:|---:|------:|--:|------|
| 5 | 6.7485 | 6 | 10.6 s | 64×64 | 1/8↘0 | 32768 | bf16 | `bits.1+bp\|` |
| 7 | 6.2863 | 8 | 1.53 s | 2×32768 | 1/4↘0 | 16384 | fp32 | `bits.1+bp\|dense.1.gelu` |
| 8 | 5.7800 | 4 | 1.59 s | 16×64 | 1/4→1/32 | 128 | bf16 | `bits.1+bp\|dense.1.tanh-suffix.2` |
| 10 | 5.6564 | 7 | 3.22 s | 2×512 | 1/4→1/256 | 2048 | fp32 | `bits.1+bp\|dense.1.tanh-suffix.4` |
| 11 | 5.6496 | 8 | 21.1 s | 1×512 | 1/32↘0 | 1048576 | fp32 | `bits.1+bp\|rnn.1.tanh-suffix.4` |
| 12 | 5.5856 | 9 | 3.49 s | 4×2048 | 1/256↘0 | 32768 | fp32 | `bits.1+bp\|dense.1.tanh-suffix.4-dense.1.tanh` |
| 13 | 5.5720 | 10 | 6.84 s | 16×256 | 1/32↘0 | 65536 | fp32 | `bits.1+bp\|rnn.1.tanh-suffix.4-dense.1.tanh` |
| 14 | 5.4715 | 10 | 25.8 s | 1024×64 | 1/8↘0 | 16384 | fp32 | `bits.1+bp\|mgru.1` |
| 15 | 5.3849 | 10 | 1.73 s | 8×512 | 1/16↘0 | 16384 | fp32 | `bits.1+bp\|split.cat(rnn.1.tanh, pass)-norm-dense.1.tanh-suffix.2` |
| 16 | 5.3432 | 5 | 51.9 s | 4×1024 | 1/32↘0 | 65536 | fp32 | `bits.1+bp\|split.add(split.add(split.add(rnn.1.gelu, pass)-norm, pass), pass)-norm-dense.1.tanh-split.add(suffix.2-dense.1.tanh, pass)` |
| 17 | 5.2983 | 3 | 13.1 s | 16×512 | 1/8→1/32 | 8192 | fp32 | `bits.1+bp\|split.cat(rnn.1.tanh, pass)-norm-dense.1.tanh-split.add(suffix.2-dense.1.tanh, pass)` |
| 18 | 5.2311 | 2 | 2 m 9 s | 4×256 | 1/16↘0 | 4194304 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-dense.1.tanh-suffix.2` |
| 20 | 5.1530 | 11 | 7.98 s | 32×512 | 1/8↘0 | 16384 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-dense.1.tanh-split.add(suffix.2-dense.1.tanh, pass)` |
| 22 | 5.1438 | 8 | 19.3 s | 1×1024 | 1/8↘0 | 524288 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-dense.1.tanh-split.add(suffix.4-dense.1.tanh, pass)` |
| 23 | 5.1338 | 6 | 33.7 s | 1×2048 | 1/8↘0 | 524288 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-dense.1.tanh-split.cat(suffix.4-dense.1.tanh, pass)` |
| 24 | 5.1260 | 6 | 16.9 s | 1×2048 | 1/8↘0 | 262144 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-dense.1.tanh-suffix.2-dense.2.tanh` |
| 25 | 5.0871 | 8 | 4.57 s | 2×512 | 1/32↘0 | 16384 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-split.mul(dense.1.tanh, dense.1)-split.add(suffix.2-dense.1.tanh, pass)` |
| 26 | 5.0860 | 5 | 4.17 s | 2×512 | 1/32↘0 | 16384 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-split.mul(dense.1, dense.1)-split.cat(suffix.2-dense.1.tanh, pass)` |
| 27 | 5.0569 | 4 | 16.8 s | 1×4096 | 1/16↘0 | 131072 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-dense.2.tanh-norm-suffix.2-dense.1.tanh` |
| 28 | 5.0212 | 8 | 2.44 s | 1×2048 | 1/32↘0 | 32768 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-dense.2.tanh-norm-split.add(suffix.2-dense.1.tanh, pass)` |
| 29 | 5.0161 | 2 | 3 m 20 s | 1×2048 | 1/16↘0 | 2097152 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-latent.2.2-suffix.2` |
| 31 | 4.9831 | 4 | 41.7 s | 4×2048 | 1/32↘0 | 131072 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-latent.2.2-suffix.2-dense.1.tanh` |
| 32 | 4.9523 | 6 | 30.2 s | 4×512 | 1/8↘0 | 262144 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-latent.2.2-suffix.2-rnn.1.tanh` |
| 34 | 4.9218 | 7 | 31.0 s | 2×1024 | 1/16↘0 | 262144 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-latent.2.2-suffix.2-split.add(dense.1.tanh, pass)-norm` |
| 35 | 4.8864 | 5 | 1 m 9 s | 4×1024 | 1/8↘0 | 65536 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-latent.2.2-split.add(suffix.2-split.cat(dense.1.tanh, pass), pass)-norm` |
| 42 | 4.8412 | 8 | 34.4 s | 1×2048 | 1/32↘0 | 262144 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-latent.2.2-split.cat(suffix.4-dense.1.tanh, pass)-norm-suffix.2-dense.1.tanh` |
| 44 | 4.8201 | 4 | 2 m 10 s | 1×2048 | 1/32↘0 | 1048576 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-latent.2.2-split.cat(suffix.2-dense.1.tanh, pass)-norm-suffix.2-split.cat(dense.1.tanh, pass)` |
| 47 | 4.8120 | 3 | 1 m 8 s | 1×4096 | 1/32↘0 | 65536 | fp32 | `bits.1+bp\|rnn.4.gelu-norm-suffix.2-dense.1.tanh` |
| 49 | 4.7840 | 4 | 50.9 s | 2×1024 | 1/32↘0 | 262144 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-latent.2.2-split.cat(suffix.4-rnn.1.tanh, pass)-norm-split.cat(suffix.2-dense.1.tanh, pass)-norm-rnn.1.tanh` |
| 52 | 4.7615 | 3 | 2 m 5 s | 1×2048 | 1/32↘0 | 131072 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-split.mul(latent.2.2, dense.2)-split.cat(suffix.4-dense.1.gelu, pass)-norm-suffix.2-dense.1.tanh` |
| 55 | 4.7327 | 6 | 16.0 s | 2×512 | 1/32↘0 | 262144 | fp32 | `bits.1+bp\|rnn.4.tanh-split.add(mingru.1, pass)-norm-suffix.2` |
| 57 | 4.6920 | 4 | 1 m 32 s | 4×1024 | 1/32↘0 | 131072 | fp32 | `bits.1+bp\|rnn.4.tanh-split.cat(mingru.1, pass)-norm-suffix.2` |
| 59 | 4.6791 | 3 | 6 m 10 s | 4×1024 | 1/128↘0 | 524288 | fp32 | `bits.1+bp\|rnn.4.tanh-split.cat(mgru.1, pass)-norm-suffix.2` |
| 65 | 4.6571 | 2 | 1 m 58 s | 4×1024 | 1/32↘0 | 131072 | fp32 | `bits.1+bp\|rnn.4.tanh-split.cat(gru.1, pass)-suffix.2` |
| 69 | 4.6208 | 4 | 1 m 31 s | 4×1024 | 1/32↘0 | 131072 | fp32 | `bits.1+bp\|rnn.4.tanh-split.cat(mingru.2, pass)-norm-suffix.2` |
| 73 | 4.6156 | 3 | 2 m 33 s | 2×1024 | 1/32↘0 | 1048576 | fp32 | `bits.1+bp\|rnn.4.tanh-norm-suffix.2-split.cat(mingru.1, pass)-norm-suffix.2` |
| 75 | 4.6101 | 5 | 1 m 37 s | 4×1024 | 1/32↘0 | 131072 | fp32 | `bits.1+bp\|rnn.4.tanh-split.cat(mingru.2-split.mul(pass, dense.2), pass)-norm-suffix.2` |
| 77 | 4.6012 | 3 | 5 m 28 s | 4×1024 | 1/32↘0 | 262144 | fp32 | `bits.1+bp\|rnn.4.tanh-rnn.4.tanh` |
| 79 | 4.5957 | 8 | 1 m 38 s | 4×1024 | 1/32↘0 | 131072 | fp32 | `bits.1+bp\|rnn.4.tanh-norm-split.add(split.add(mingru.2-split.mul(pass, dense.2.tanh), pass), pass)-norm-suffix.4` |
| 80 | 4.5449 | 3 | 5 m 51 s | 4×512 | 1/16↘0 | 524288 | fp32 | `bits.1+bp\|mgru.4-norm-rnn.1.tanh` |
| 84 | 4.5407 | 5 | 1 m 18 s | 1×2048 | 1/16↘0 | 262144 | fp32 | `bits.1+bp\|split.add(mgru.2, pass)-mingru.4-norm-split.cat(suffix.2-rnn.1.tanh, pass)` |
| 85 | 4.5032 | 5 | 44.3 s | 1×2048 | 1/16↘0 | 262144 | fp32 | `bits.1+bp\|rnn.4.tanh-mingru.4-norm-suffix.2` |
| 87 | 4.4959 | 4 | 45.1 s | 1×2048 | 1/16↘0 | 262144 | fp32 | `bits.1+bp\|rnn.4.tanh-mingru.4-norm-suffix.2-dense.1.tanh` |
| 93 | 4.4740 | 3 | 2 m 51 s | 4×1024 | 1/32↘0 | 65536 | fp32 | `bits.1+bp\|split.mul(rnn.4.tanh, dense.4)-norm-split.add(mingru.2, pass)-norm-suffix.4` |
| 97 | 4.4564 | 3 | 3 m 7 s | 32×512 | 1/32↘0 | 131072 | fp32 | `bits.1+bp\|mgru.4-dense.4.gelu-norm` |
| 101 | 4.4424 | 6 | 1.32 s | 1×512 | 1/8↘0 | 32768 | fp32 | `bits.4.emb.4\|conv.2-norm-conv.2-split.add(split.add(rmsnorm, pass), pass)` |
| 104 | 4.4387 | 4 | 1 m 19 s | 64×256 | 1/16↘0 | 65536 | fp32 | `bits.1+bp\|mgru.4-norm-split.cat(dense.4.gelu, pass)-norm-rnn.1.tanh` |
| 105 | 4.4051 | 8 | 8.88 s | 2×256 | 1/64↘0 | 131072 | fp32 | `bits.4.emb.4\|rglru.4-norm-split.add(norm, pass)-conv.2` |
| 113 | 4.3715 | 7 | 7.86 s | 4×256 | 1/32↘0 | 65536 | fp32 | `bits.4.emb.4\|rglru.2-norm-conv.2` |
| 117 | 4.3520 | 4 | 1 m 14 s | 4×256 | 1/32↘0 | 131072 | fp32 | `bits.4.emb.4\|rglru.2-split.add(rmsnorm, pass)-conv.2` |
| 121 | 4.3450 | 6 | 6.71 s | 2×256 | 1/32↘0 | 131072 | fp32 | `bits.4.emb.4\|split.add(split.add(split.add(mingru.4-rmsnorm, pass), pass), pass)-rmsnorm` |
| 124 | 4.3436 | 4 | 4 m 20 s | 8×512 | 1/16↘0 | 524288 | fp32 | `bits.1+bp\|rnn.8.tanh-split.cat(rnn.1.tanh, pass)` |
| 125 | 4.3111 | 4 | 3.78 s | 1×256 | 1/16↘0 | 16384 | fp32 | `bits.4.emb.4\|split.add(split.add(conv.2-norm, pass)-norm, pass)-norm-split.add(split.add(norm, pass), pass)-mingru.4` |
| 129 | 4.3099 | 3 | 1 m 17 s | 4×128 | 1/16↘0 | 262144 | fp32 | `bits.4.emb.4\|rglru.1-norm-conv.2` |
| 133 | 4.2742 | 8 | 3.07 s | 4×128 | 1/32↘0 | 32768 | fp32 | `bits.4.emb.4\|rglru.2-split.add(norm, pass)-conv.2-split.mul(pass, dense.4)` |
| 141 | 4.2620 | 9 | 1 m 16 s | 4×512 | 1/16↘0 | 262144 | fp32 | `bits.4.emb.4\|rglru.2-conv.4-split.mul(pass, dense.4)` |
| 145 | 4.2528 | 7 | 5.40 s | 64×32 | 1/16↘0 | 16384 | fp32 | `bits.4.emb.4\|mingru.4-norm-conv.2-split.mul(pass, dense.4)` |
| 149 | 4.2358 | 9 | 1.40 s | 1×256 | 1/32↘0 | 32768 | fp32 | `bits.4.emb.4\|split.add(conv.2-split.add(split.add(rmsnorm, pass), pass)-mingru.4, pass)-split.mul(pass, dense.4)` |
| 153 | 4.2264 | 7 | 3.94 s | 1×1024 | 1/32↘0 | 32768 | fp32 | `bits.4.emb.4\|split.add(dense.4.tanh-split.add(dense.4.gelu-conv.4-split.add(split.add(norm, pass), pass), pass)-split.mul(pass, dense.4), pass)` |
| 159 | 4.2158 | 5 | 4.58 s | 1×1024 | 1/32↘0 | 32768 | fp32 | `bits.4.emb.4\|split.add(dense.2.tanh-split.add(dense.4.gelu-conv.4-split.add(split.add(rmsnorm, pass), pass), pass)-split.mul(pass, dense.4), pass)-split.mul(pass, dense.4)` |
| 161 | 4.1908 | 6 | 32.2 s | 4×128 | 1/32↘0 | 65536 | fp32 | `bits.4.emb.4\|rglru.2-norm-conv.4-split.mul(pass, dense.4)-split.mul(pass, dense.4.tanh)` |
| 169 | 4.1891 | 7 | 2.08 s | 2×256 | 1/32↘0 | 32768 | fp32 | `bits.4.emb.4\|split.add(conv.2-split.add(split.add(dense.4.tanh-rmsnorm, pass), pass)-mingru.4, pass)-split.mul(pass, dense.4)` |
| 176 | 4.1626 | 7 | 1 m 49 s | 16×2048 | 1/16↘0 | 65536 | fp32 | `bits.4.oh+bp\|dense.2.tanh-suffix.2-split.add(mingru.4, pass)-norm-split.add(dense.4.gelu, pass)` |
| 185 | 4.1438 | 4 | 29.4 s | 2×512 | 1/16↘0 | 65536 | fp32 | `bits.4.emb.4\|suffix.2-mingru.4-split.mul(pass, dense.4)-split.mul(pass, dense.4.tanh)` |
| 193 | 4.1163 | 6 | 21.1 s | 4×512 | 1/32↘0 | 65536 | fp32 | `bits.4.emb.4\|rglru.2-norm-suffix.2-mingru.4-split.add(split.mul(pass, dense.4), pass)` |
| 196 | 4.1026 | 3 | 3 m 11 s | 4×1024 | 1/16↘0 | 524288 | fp32 | `bits.4.oh+bp\|dense.2.tanh-suffix.2-mingru.4-norm-split.add(dense.4.gelu-split.add(dense.4.tanh, pass), pass)` |
| 204 | 4.0970 | 6 | 13.7 s | 32×128 | 1/16↘0 | 32768 | fp32 | `bits.4.oh+bp\|rnn.4.tanh-suffix.2-dense.4.gelu` |
| 207 | 4.0796 | 4 | 14.4 s | 8×256 | 1/32↘0 | 32768 | fp32 | `tokens.32.fold.emb.4\|split.add(rnn.2.tanh-split.cat(norm, pass), pass)` |
| 211 | 4.0641 | 5 | 24.0 s | 4×128 | 1/16↘0 | 65536 | fp32 | `tokens.32.fold.emb.4\|split.add(split.add(split.add(rnn.2.tanh, pass)-rmsnorm-split.add(split.add(norm, pass), pass), pass), pass)` |
| 215 | 4.0599 | 7 | 11.2 s | 4×256 | 1/16↘0 | 32768 | fp32 | `tokens.32.fold.emb.4\|split.add(mingru.2-split.add(split.cat(rmsnorm, pass), pass), pass)` |
| 217 | 4.0569 | 6 | 21.4 s | 8×128 | 1/16↘0 | 65536 | fp32 | `tokens.32.fold.emb.4\|split.add(split.add(mingru.2, pass)-rmsnorm, pass)` |
| 221 | 4.0507 | 5 | 22.3 s | 8×256 | 1/32↘0 | 32768 | fp32 | `tokens.32.fold.emb.4\|split.add(mgru.2-split.cat(norm, pass), pass)` |
| 223 | 4.0398 | 9 | 15.0 s | 8×256 | 1/32↘0 | 32768 | fp32 | `tokens.32.fold.emb.4\|split.add(mgru.2-rmsnorm, pass)` |
| 227 | 4.0349 | 5 | 12.0 s | 1×256 | 1/16↘0 | 32768 | fp32 | `tokens.32.fold.emb.4\|split.add(split.add(rnn.2.tanh, pass)-norm-dense.4.tanh, pass)` |
| 229 | 4.0284 | 5 | 2.18 s | 2×256 | 1/8↘0 | 16384 | fp32 | `tokens.32.fold.emb.4\|split.add(split.add(dense.4.tanh-split.add(conv.2-norm, pass)-rmsnorm, pass)-split.add(norm-split.add(norm, pass)-split.add(norm, pass), pass), pass)` |
| 231 | 4.0161 | 5 | 4.30 s | 32×128 | 1/4↘0 | 8192 | fp32 | `tokens.32.shift.emb.4\|mingru.4-split.add(rmsnorm, pass)` |
| 233 | 4.0053 | 4 | 12.1 s | 16×64 | 1/16↘0 | 32768 | fp32 | `tokens.32.fold.emb.4\|split.add(split.mul(pass, dense.4.tanh)-mingru.2, pass)` |
| 237 | 4.0028 | 4 | 43.9 s | 16×128 | 1/16↘0 | 131072 | fp32 | `tokens.32.fold.emb.4\|split.add(mingru.4-rmsnorm, pass)` |
| 239 | 3.9835 | 4 | 11.8 s | 64×64 | 1/8↘0 | 16384 | fp32 | `tokens.32.shift.emb.4\|lrnn.4.2-split.add(split.add(split.add(norm, pass), pass), pass)` |
| 243 | 3.9685 | 5 | 25.0 s | 64×64 | 1/8↘0 | 32768 | fp32 | `tokens.32.shift.emb.4\|split.add(lrnn.4.2-split.add(rmsnorm, pass), pass)` |
| 247 | 3.9632 | 5 | 1 m 3 s | 32×64 | 1/32↘0 | 65536 | fp32 | `tokens.32.shift.emb.4\|split.add(split.mul(pass, dense.4)-mingru.4, pass)` |
| 253 | 3.9591 | 5 | 23.8 s | 32×128 | 1/16↘0 | 65536 | fp32 | `tokens.32.fold.emb.4\|split.add(split.mul(pass, dense.4)-mingru.4, pass)` |
| 257 | 3.9554 | 5 | 1 m 47 s | 32×512 | 1/32↘0 | 65536 | fp32 | `bits.1+bp\|mgru.8-norm-split.cat(dense.4.gelu, pass)` |
| 263 | 3.9198 | 4 | 31.0 s | 4×256 | 1/16↘0 | 32768 | fp32 | `tokens.32.shift.emb.4\|split.add(split.add(split.add(conv.2-norm, pass)-split.mul(norm, dense.4), pass)-split.add(split.add(norm, pass)-split.add(rmsnorm, pass), pass)-mingru.4, pass)` |
| 273 | 3.9196 | 5 | 19.3 s | 2×512 | 1/16↘0 | 32768 | fp32 | `tokens.32.fold.emb.4\|split.add(split.add(split.add(conv.2-rmsnorm, pass)-split.mul(norm, dense.4), pass)-split.add(split.add(norm, pass)-rmsnorm, pass)-mingru.4, pass)` |
| 275 | 3.9133 | 4 | 1 m 6 s | 2×512 | 1/32↘0 | 65536 | fp32 | `tokens.32.shift.emb.4\|split.add(split.add(split.add(conv.4-rmsnorm, pass)-split.mul(norm, dense.4), pass)-split.add(split.add(norm, pass)-rmsnorm, pass)-mingru.4, pass)` |
| 283 | 3.8985 | 2 | 34.4 s | 4×256 | 1/16↘0 | 32768 | fp32 | `tokens.32.shift.emb.4\|split.add(split.add(dense.4.tanh-split.add(conv.2-norm, pass)-split.mul(norm, dense.4), pass)-split.add(split.add(norm, pass)-split.add(rmsnorm, pass), pass)-mingru.4, pass)` |
| 287 | 3.8921 | 4 | 44.4 s | 2×512 | 1/16↘0 | 65536 | fp32 | `tokens.32.shift.emb.4\|split.add(split.add(split.add(conv.2-rmsnorm, pass)-split.mul(norm, dense.4), pass)-split.add(split.add(norm, pass)-rmsnorm, pass)-mingru.4-dense.4, pass)` |
| 295 | 3.8880 | 2 | 25.1 s | 4×512 | 1/16↘0 | 32768 | fp32 | `tokens.32.shift.emb.4\|split.add(split.add(split.add(conv.4-rmsnorm, pass)-split.mul(norm, dense.4), pass)-split.add(dense.4.gelu-split.add(norm, pass)-rmsnorm, pass)-mingru.4, pass)` |
| 303 | 3.8718 | 3 | 1 m 9 s | 2×512 | 1/16↘0 | 65536 | fp32 | `tokens.32.shift.emb.4\|split.add(conv.2-norm-split.mul(dense.4.tanh, dense.4), pass)-split.add(split.add(rmsnorm-split.mul(pass, dense.4.tanh), pass)-mingru.4, pass)` |
| 315 | 3.8627 | 2 | 1 m 22 s | 4×512 | 1/16↘0 | 65536 | fp32 | `tokens.32.shift.emb.4\|split.add(split.add(split.add(conv.4-rmsnorm, pass)-split.mul(norm, dense.4), pass)-split.add(dense.4.gelu-split.add(norm, pass)-rmsnorm-dense.4.tanh, pass)-mingru.4, pass)` |
| 327 | 3.8508 | 6 | 23.8 s | 1×512 | 1/16↘0 | 32768 | fp32 | `tokens.32.shift.emb.4\|split.add(split.add(split.add(conv.2-rmsnorm, pass)-split.mul(norm, dense.4), pass)-split.mul(pass, dense.4)-split.add(split.add(split.add(norm-split.mul(dense.4, dense.4), pass)-rmsnorm, pass)-mingru.4, pass), pass)` |
| 337 | 3.8479 | 4 | 15.7 s | 32×64 | 1/8↘0 | 16384 | fp32 | `tokens.32.shift.emb.4\|split.add(split.mul(split.add(lrnn.4.2-norm-dense.2, pass)-rmsnorm, dense.4)-rmsnorm-split.add(mingru.4-split.mul(pass, dense.4.tanh), pass), pass)` |
| 339 | 3.8300 | 4 | 42.9 s | 32×64 | 1/8↘0 | 16384 | fp32 | `tokens.32.shift.emb.4\|split.mul(split.add(lrnn.4.4-dense.4, pass), dense.4)-split.add(mingru.4-split.mul(pass, dense.4.tanh), pass)` |
| 363 | 3.8226 | 4 | 1 m 8 s | 32×64 | 1/8↘0 | 32768 | fp32 | `tokens.32.shift.emb.4\|split.add(split.mul(split.add(lrnn.4.4-dense.4, pass)-split.mul(rmsnorm, dense.4), dense.4)-split.add(mingru.4-split.mul(pass, dense.4.tanh), pass), pass)` |
| 379 | 3.8199 | 6 | 15.4 s | 32×64 | 1/8↘0 | 16384 | fp32 | `tokens.32.shift.emb.4\|split.mul(split.add(suffix.2-lrnn.4.2-dense.4, pass)-split.mul(rmsnorm, dense.4), dense.4)-split.add(mingru.4-split.mul(pass, dense.4.tanh), pass)` |
| 380 | 3.8075 | 3 | 5 m 15 s | 8×2048 | 1/16↘0 | 131072 | fp32 | `bits.4.oh+bp\|dense.4.tanh-split.add(suffix.2-rnn.4.tanh-dense.8.gelu-norm, pass)-split.add(dense.8.gelu, pass)` |
| 383 | 3.8072 | 4 | 1 m 27 s | 32×64 | 1/8↘0 | 32768 | fp32 | `tokens.32.shift.emb.4\|split.add(split.mul(split.add(lrnn.4.4-dense.4, pass)-split.mul(rmsnorm-dense.4, dense.4), dense.4)-split.add(mingru.4-split.mul(pass, dense.4.tanh), pass), pass)` |
| 387 | 3.7978 | 4 | 41.1 s | 16×128 | 1/16↘0 | 65536 | fp32 | `tokens.32.shift.emb.8\|rnn.4.tanh-split.cat(dense.4.tanh, pass)` |
| 395 | 3.7950 | 4 | 33.1 s | 64×128 | 1/8↘0 | 16384 | fp32 | `tokens.32.shift.emb.8\|split.add(split.add(mingru.4, pass), pass)-split.add(split.add(rmsnorm, pass), pass)` |
| 400 | 3.7907 | 5 | 1 m 20 s | 16×1024 | 1/16↘0 | 65536 | fp32 | `bits.4.oh+bp\|dense.4.tanh-split.add(suffix.2-mingru.4-dense.8.gelu-norm, pass)-split.add(dense.8.gelu, pass)` |
| 405 | 3.7907 | 5 | 10.7 s | 4×256 | 1/16↘0 | 65536 | fp32 | `bits.4.emb.8\|split.mul(mingru.4, dense.4.tanh)-suffix.2-dense.8.gelu-split.add(dense.8.gelu-rmsnorm, pass)` |
| 407 | 3.7716 | 5 | 24.8 s | 16×128 | 1/16↘0 | 65536 | fp32 | `tokens.32.shift.emb.8\|mingru.4-split.cat(dense.4.tanh, pass)` |
| 415 | 3.7651 | 4 | 35.2 s | 64×128 | 1/8↘0 | 16384 | fp32 | `tokens.32.shift.emb.8\|split.add(rnn.4.gelu-split.add(split.add(norm, pass)-split.add(dense.8.gelu-rmsnorm, pass), pass), pass)` |
| 423 | 3.7562 | 4 | 28.4 s | 32×128 | 1/8↘0 | 65536 | fp32 | `tokens.32.shift.emb.8\|split.add(mingru.2, pass)-dense.8.tanh` |
| 432 | 3.7511 | 5 | 1 m 22 s | 16×1024 | 1/16↘0 | 65536 | fp32 | `bits.4.oh+bp\|dense.4.tanh-split.add(suffix.2-mingru.4-suffix.2-dense.8.gelu-norm, pass)-split.add(dense.8.gelu, pass)` |
| 447 | 3.7349 | 3 | 17.6 s | 64×256 | 1/8↘0 | 16384 | fp32 | `tokens.32.shift.emb.8\|split.add(rnn.4.tanh, pass)-split.add(dense.8.tanh, pass)-split.add(rmsnorm, pass)` |
| 451 | 3.7302 | 5 | 53.6 s | 256×64 | 1/8↘0 | 8192 | fp32 | `tokens.32.shift.emb.8\|rnn.8.tanh` |
| 455 | 3.7175 | 5 | 19.1 s | 16×128 | 1/16↘0 | 16384 | fp32 | `tokens.32.shift.emb.8\|split.add(lrnn.4.2, pass)-norm-dense.8.tanh` |
| 459 | 3.7129 | 3 | 14.4 s | 16×256 | 1/8↘0 | 16384 | fp32 | `tokens.32.shift.emb.8\|split.add(mingru.4, pass)-dense.8.tanh` |
| 472 | 3.7077 | 2 | 3 m 30 s | 16×256 | 1/16↘0 | 131072 | fp32 | `bits.4.oh+bp\|lrnn.4.2-suffix.2-mingru.8-rmsnorm-dense.8.gelu-split.add(norm, pass)` |
| 474 | 3.7047 | 4 | 2 m 4 s | 64×128 | 1/8↘0 | 32768 | fp32 | `tokens.32.hexbpe.emb.8\|lrnn.8.2` |
| 487 | 3.6979 | 4 | 13.2 s | 16×128 | 1/8↘0 | 32768 | fp32 | `tokens.32.shift.emb.8\|split.add(split.add(mingru.2, pass), pass)-suffix.2-dense.8.tanh` |
| 491 | 3.6864 | 4 | 12.4 s | 16×128 | 1/8↘0 | 16384 | fp32 | `tokens.32.shift.emb.8\|split.add(mgru.4, pass)-dense.8.tanh` |
| 510 | 3.6863 | 5 | 2 m 11 s | 32×128 | 1/8↘0 | 16384 | fp32 | `tokens.32.hexbpe.emb.8\|lrnn.8.4-split.add(split.add(dense.4.tanh, pass), pass)` |
| 511 | 3.6769 | 3 | 19.4 s | 16×128 | 1/8↘0 | 16384 | fp32 | `tokens.32.shift.emb.8\|split.add(mgru.4-split.mul(pass, dense.4), pass)-dense.8.tanh` |
| 515 | 3.6466 | 6 | 1 m 7 s | 64×64 | 1/4↘0 | 16384 | fp32 | `tokens.32.shift.emb.8\|lrnn.8.4` |
| 551 | 3.6256 | 3 | 3 m 5 s | 32×64 | 1/16↘0 | 32768 | fp32 | `tokens.32.shift.emb.8\|lrnn.8.4-split.add(split.add(split.add(dense.4.gelu, pass), pass), pass)` |
| 567 | 3.6211 | 4 | 3 m 10 s | 32×64 | 1/16↘0 | 32768 | fp32 | `tokens.32.shift.emb.8\|lrnn.8.4-split.add(split.add(split.add(latent.4.2, pass), pass), pass)` |
| 579 | 3.6208 | 4 | 1 m 12 s | 128×64 | 1/4↘0 | 16384 | fp32 | `tokens.32.shift.emb.8\|suffix.2-lrnn.8.4` |
| 583 | 3.6186 | 3 | 32.7 s | 16×128 | 1/8↘0 | 16384 | fp32 | `tokens.32.shift.emb.8\|split.add(mgru.4-split.mul(mgru.4, dense.4), pass)-dense.8.tanh` |
| 587 | 3.5906 | 2 | 2 m 33 s | 128×32 | 1/4↘0 | 65536 | fp32 | `tokens.32.shift.emb.8\|lrnn.8.4-split.mul(pass, dense.8)` |
| 626 | 3.5860 | 4 | 22.0 s | 16×128 | 1/8↘0 | 32768 | fp32 | `tokens.32.hexbpe.emb.8\|rnn.8.tanh-split.add(dense.8.tanh-split.mul(pass, dense.8), pass)-split.add(split.mul(pass, dense.8), pass)` |
| 659 | 3.5813 | 2 | 2 m 30 s | 8×256 | 1/8↘0 | 16384 | fp32 | `tokens.32.shift.emb.8\|lrnn.8.2-split.add(mingru.8, pass)` |
| 667 | 3.5734 | 4 | 15.9 s | 16×128 | 1/8↘0 | 32768 | fp32 | `tokens.32.shift.emb.8\|mingru.8-split.mul(latent.8.2, dense.8)` |
| 675 | 3.5686 | 5 | 6.78 s | 64×32 | 1/16↘0 | 8192 | fp32 | `tokens.32.shift.emb.8\|split.add(rnn.4.tanh-dense.4.tanh, pass)-rnn.8.tanh-split.add(split.add(split.mul(dense.8-rmsnorm, dense.8), pass), pass)` |
| 690 | 3.5673 | 5 | 4.64 s | 16×128 | 1/8↘0 | 8192 | fp32 | `tokens.32.hexbpe.emb.8\|mgru.8-split.add(dense.8.tanh, pass)-split.add(split.mul(pass, dense.8), pass)` |
| 695 | 3.5525 | 4 | 1 m 13 s | 32×32 | 1/16↘0 | 32768 | fp32 | `tokens.32.shift.emb.8\|lrnn.8.4-split.add(split.add(split.add(dense.4.tanh, pass)-mingru.8, pass), pass)` |
| 715 | 3.5427 | 5 | 1 m 18 s | 32×32 | 1/16↘0 | 32768 | fp32 | `tokens.32.shift.emb.8\|lrnn.8.4-split.add(split.add(split.add(dense.4.tanh-rglru.4, pass)-mingru.8, pass), pass)` |
| 723 | 3.5385 | 2 | 1 m 11 s | 32×128 | 1/8↘0 | 65536 | fp32 | `tokens.32.shift.emb.8\|mgru.8-split.add(split.add(latent.8.2, pass), pass)` |
| 762 | 3.5127 | 5 | 8.26 s | 16×256 | 1/8↘0 | 8192 | fp32 | `tokens.32.hexbpe.emb.8\|mgru.8-split.add(dense.8.tanh, pass)-split.add(split.mul(dense.8, dense.8), pass)` |
| 824 | 3.5056 | 2 | 3 m 43 s | 128×128 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.16.gelu-dense.8.gelu-norm` |
| 832 | 3.5041 | 4 | 56.1 s | 128×128 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.16.gelu-dense.8.gelu-rmsnorm` |
| 867 | 3.4882 | 4 | 20.8 s | 16×128 | 1/8↘0 | 32768 | fp32 | `tokens.32.shift.emb.8\|mgru.8-split.add(split.add(latent.8.2-split.mul(pass, dense.8.tanh), pass)-split.mul(pass, dense.8), pass)` |
| 875 | 3.4739 | 3 | 1 m 3 s | 64×64 | 1/16↘0 | 32768 | fp32 | `tokens.32.shift.emb.8\|mgru.8-split.add(dense.8-split.add(split.mul(split.add(dense.8, pass), dense.8)-split.mul(pass, dense.8), pass), pass)` |
| 906 | 3.4661 | 5 | 8.58 s | 16×256 | 1/8↘0 | 8192 | fp32 | `tokens.32.hexbpe.emb.8\|dense.8.tanh-mgru.8-split.add(dense.8.gelu-dense.8.tanh, pass)-split.add(split.mul(dense.8, dense.8), pass)` |
| 939 | 3.4621 | 3 | 1 m 12 s | 64×64 | 1/16↘0 | 32768 | fp32 | `tokens.32.shift.emb.8\|mgru.8-split.add(rnn.8.tanh, pass)-split.add(split.add(dense.8, pass)-split.mul(pass, dense.8)-split.mul(pass, dense.8), pass)` |
| 944 | 3.4507 | 3 | 4 m 29 s | 16×512 | 1/64↘0 | 65536 | fp32 | `bits.4.oh+bp\|lrnn.8.2-norm-rnn.16.gelu` |
| 962 | 3.4429 | 5 | 25.9 s | 16×256 | 1/8↘0 | 16384 | fp32 | `tokens.32.hexbpe.emb.8\|gru.8-split.add(split.add(latent.8.2-split.mul(pass, dense.8.tanh), pass)-split.mul(pass, dense.8), pass)` |
| 1003 | 3.4428 | 3 | 10.5 s | 64×128 | 1/8↘0 | 4096 | fp32 | `tokens.32.shift.emb.8\|gru.8-split.add(latent.8.2-split.mul(pass, dense.8), pass)-split.mul(pass, dense.8)` |
| 1024 | 3.4387 | 4 | 3 m 25 s | 64×128 | 1/16↘0 | 131072 | fp32 | `bits.4.oh+bp\|rnn.8.tanh-mgru.8-suffix.2-dense.16.gelu-norm` |
| 1034 | 3.4350 | 4 | 12.4 s | 16×128 | 1/8↘0 | 16384 | fp32 | `tokens.32.hexbpe.emb.8\|gru.8-split.add(latent.8.2-split.mul(pass, dense.8), pass)-split.mul(dense.8.gelu, dense.8)` |
| 1042 | 3.4017 | 5 | 22.7 s | 16×128 | 1/8↘0 | 32768 | fp32 | `tokens.32.hexbpe.emb.8\|gru.8-split.add(split.add(dense.8.tanh-split.mul(pass, dense.8), pass)-split.mul(split.mul(pass, dense.8.tanh)-dense.8.tanh, dense.8), pass)` |
| 1120 | 3.3920 | 3 | 30.8 s | 64×128 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|split.add(rnn.16.gelu, pass)-dense.16.gelu-rmsnorm` |
| 1152 | 3.3799 | 4 | 52.3 s | 64×128 | 1/32↘0 | 32768 | fp32 | `bits.4.oh+bp\|rnn.16.gelu-norm-dense.16.gelu-split.add(conv.2-rmsnorm, pass)` |
| 1216 | 3.3693 | 4 | 2 m 35 s | 16×256 | 1/64↘0 | 65536 | fp32 | `bits.4.oh+bp\|lrnn.8.2-rnn.16.gelu-split.add(dense.16.gelu, pass)` |
| 1225 | 3.3641 | 3 | 1 m 52 s | 64×128 | 1/32↘0 | 65536 | fp32 | `tokens.32.hexbpe.oh\|rnn.16.gelu-dense.8.gelu` |
| 1233 | 3.3639 | 3 | 1 m 26 s | 64×128 | 1/32↘0 | 32768 | fp32 | `tokens.32.hexbpe.oh\|rnn.16.gelu-dense.8.gelu-rmsnorm` |
| 1288 | 3.3611 | 5 | 1 m 56 s | 16×512 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.16.gelu-rnn.8.tanh-suffix.2-dense.16.gelu` |
| 1304 | 3.3547 | 3 | 1 m 16 s | 64×128 | 1/16↘0 | 32768 | fp32 | `tokens.64.shift.emb.8\|split.cat(rglru.1, pass)-dense.16.gelu-mingru.8` |
| 1314 | 3.3457 | 2 | 4 m 45 s | 128×128 | 1/8↘0 | 16384 | fp32 | `tokens.32.hexbpe.emb.16\|lrnn.16.4` |
| 1344 | 3.3241 | 2 | 1 m 59 s | 64×128 | 1/64↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.16.gelu-rnn.16.gelu-norm` |
| 1360 | 3.2979 | 2 | 3 m 55 s | 64×128 | 1/64↘0 | 131072 | fp32 | `bits.4.oh+bp\|rnn.16.gelu-rnn.16.gelu-split.add(rmsnorm, pass)` |
| 1552 | 3.2605 | 2 | 2 m 51 s | 32×512 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.8.tanh-suffix.2-dense.16.gelu-norm-rnn.16.tanh-split.add(dense.16.gelu, pass)` |
| 1569 | 3.2552 | 3 | 47.1 s | 32×128 | 1/32↘0 | 32768 | fp32 | `tokens.32.hexbpe.oh\|rnn.8.gelu-norm-rnn.16.gelu-norm-dense.8.tanh-norm-dense.16.gelu` |
| 1696 | 3.2545 | 5 | 1 m 23 s | 32×256 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.16.gelu-rnn.16.gelu-norm-split.add(dense.16.tanh-rglru.16, pass)-norm` |
| 1697 | 3.2420 | 4 | 30.6 s | 32×128 | 1/32↘0 | 32768 | fp32 | `tokens.32.hexbpe.oh\|rnn.8.gelu-norm-rnn.16.gelu-norm-dense.8.tanh-norm-suffix.2-dense.16.gelu` |
| 1844 | 3.2343 | 4 | 1 m 30 s | 64×512 | 1/32↘0 | 32768 | fp32 | `bits.2.oh+bp\|rnn.32.gelu-dense.16.gelu` |
| 1858 | 3.2238 | 2 | 2 m 53 s | 32×128 | 1/16↘0 | 131072 | fp32 | `tokens.32.hexbpe.emb.16\|mgru.16-dense.16.gelu` |
| 1888 | 3.2186 | 2 | 1 m 47 s | 32×256 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.16.gelu-rnn.16.gelu-norm-split.add(dense.16.gelu-dense.16.tanh, pass)-norm` |
| 1904 | 3.2167 | 5 | 44.3 s | 32×128 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.16.gelu-rnn.16.gelu-rmsnorm-mingru.16` |
| 1920 | 3.2091 | 3 | 1 m 14 s | 32×128 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.16.gelu-rnn.16.gelu-split.add(rmsnorm-split.add(mingru.16, pass)-rmsnorm, pass)` |
| 2016 | 3.2063 | 4 | 2 m 12 s | 128×256 | 1/16↘0 | 32768 | fp32 | `bits.4.oh+bp\|dense.8.gelu-mgru.16-dense.32.gelu-norm` |
| 2056 | 3.2018 | 5 | 1 m 57 s | 16×256 | 1/32↘0 | 131072 | fp32 | `bits.4.oh+bp\|dense.4.tanh-rnn.32.gelu-dense.16.gelu` |
| 2080 | 3.1892 | 3 | 30.8 s | 32×256 | 1/32↘0 | 32768 | fp32 | `bits.4.oh+bp\|rnn.8.tanh-norm-rnn.32.gelu-rmsnorm` |
| 2128 | 3.1758 | 3 | 1 m 45 s | 16×512 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.32.gelu` |
| 2165 | 3.1727 | 3 | 25.4 s | 8×128 | 1/32↘0 | 32768 | fp32 | `tokens.64.hexbpe.emb.16\|rnn.16.gelu-split.add(split.mul(dense.16, dense.16), pass)` |
| 2245 | 3.1649 | 3 | 1 m 20 s | 64×128 | 1/16↘0 | 32768 | fp32 | `tokens.64.hexbpe.emb.16\|mullstm.8-dense.16.gelu-rmsnorm` |
| 2256 | 3.1546 | 2 | 59.5 s | 64×128 | 1/32↘0 | 32768 | fp32 | `bits.4.oh+bp\|dense.8-norm-rnn.32.gelu-dense.16.gelu` |
| 2272 | 3.1439 | 4 | 55.0 s | 32×128 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|dense.8.tanh-rnn.32.gelu-norm-dense.16.gelu-split.add(rmsnorm, pass)` |
| 2336 | 3.1347 | 4 | 2 m 21 s | 32×128 | 1/32↘0 | 131072 | fp32 | `bits.4.oh+bp\|latent.8.2-rnn.32.gelu-dense.16.gelu-split.add(rmsnorm, pass)` |
| 2400 | 3.1291 | 5 | 5 m 23 s | 64×64 | 1/32↘0 | 262144 | fp32 | `bits.4.oh+bp\|rnn.32.gelu-dense.16.gelu` |
| 2416 | 3.1004 | 3 | 3 m 28 s | 64×128 | 1/32↘0 | 131072 | fp32 | `bits.4.oh+bp\|rnn.32.gelu-dense.16.gelu-rmsnorm` |
| 2656 | 3.0920 | 5 | 1 m 5 s | 32×256 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.32.gelu-rnn.16.gelu-norm` |
| 2745 | 3.0848 | 3 | 1 m 60 s | 32×128 | 1/32↘0 | 131072 | fp32 | `tokens.32.hexbpe.oh\|latent.8.2-rnn.32.gelu-dense.16.gelu-split.add(rmsnorm, pass)` |
| 2784 | 3.0831 | 5 | 2 m 13 s | 32×128 | 1/32↘0 | 131072 | fp32 | `bits.4.oh+bp\|dense.8.gelu-rnn.32.gelu-rnn.16.gelu-norm-dense.16.tanh` |
| 2928 | 3.0694 | 6 | 59.4 s | 32×128 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.32.gelu-rnn.16.gelu-split.add(dense.16.gelu, pass)` |
| 2944 | 3.0650 | 2 | 1 m 60 s | 32×256 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.32.gelu-rnn.16.gelu-split.add(dense.16.gelu-rmsnorm, pass)` |
| 2960 | 3.0569 | 2 | 2 m 9 s | 32×256 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.32.gelu-rnn.16.gelu-split.add(dense.16.gelu-rmsnorm, pass)-rmsnorm` |
