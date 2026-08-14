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
Snapshot of 211 rows taken on 2026-08-12 (confs with at least
2 runs, up to 10000 weights). Regenerate with:

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
| 14 | 5.4763 | 13 | 25.8 s | 1024×64 | 1/8↘0 | 16384 | fp32 | `bits.1+bp\|mgru.1` |
| 15 | 5.3849 | 10 | 1.73 s | 8×512 | 1/16↘0 | 16384 | fp32 | `bits.1+bp\|split.cat(rnn.1.tanh, pass)-norm-dense.1.tanh-suffix.2` |
| 16 | 5.3432 | 5 | 51.9 s | 4×1024 | 1/32↘0 | 65536 | fp32 | `bits.1+bp\|split.add(split.add(split.add(rnn.1.gelu, pass)-norm, pass), pass)-norm-dense.1.tanh-split.add(suffix.2-dense.1.tanh, pass)` |
| 17 | 5.2983 | 3 | 13.1 s | 16×512 | 1/8→1/32 | 8192 | fp32 | `bits.1+bp\|split.cat(rnn.1.tanh, pass)-norm-dense.1.tanh-split.add(suffix.2-dense.1.tanh, pass)` |
| 18 | 5.2311 | 2 | 2 m 9 s | 4×256 | 1/16↘0 | 4194304 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-dense.1.tanh-suffix.2` |
| 20 | 5.1530 | 11 | 7.98 s | 32×512 | 1/8↘0 | 16384 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-dense.1.tanh-split.add(suffix.2-dense.1.tanh, pass)` |
| 23 | 5.1338 | 6 | 33.7 s | 1×2048 | 1/8↘0 | 524288 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-dense.1.tanh-split.cat(suffix.4-dense.1.tanh, pass)` |
| 24 | 5.1244 | 7 | 16.9 s | 1×2048 | 1/8↘0 | 262144 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-dense.1.tanh-suffix.2-dense.2.tanh` |
| 25 | 5.0871 | 8 | 4.57 s | 2×512 | 1/32↘0 | 16384 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-split.mul(dense.1.tanh, dense.1)-split.add(suffix.2-dense.1.tanh, pass)` |
| 26 | 5.0860 | 5 | 4.17 s | 2×512 | 1/32↘0 | 16384 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-split.mul(dense.1, dense.1)-split.cat(suffix.2-dense.1.tanh, pass)` |
| 27 | 5.0542 | 5 | 16.8 s | 1×4096 | 1/16↘0 | 131072 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-dense.2.tanh-norm-suffix.2-dense.1.tanh` |
| 28 | 5.0212 | 8 | 2.44 s | 1×2048 | 1/32↘0 | 32768 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-dense.2.tanh-norm-split.add(suffix.2-dense.1.tanh, pass)` |
| 29 | 5.0161 | 2 | 3 m 20 s | 1×2048 | 1/16↘0 | 2097152 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-latent.2.2-suffix.2` |
| 31 | 4.9831 | 4 | 41.7 s | 4×2048 | 1/32↘0 | 131072 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-latent.2.2-suffix.2-dense.1.tanh` |
| 32 | 4.9683 | 7 | 30.2 s | 4×512 | 1/8↘0 | 262144 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-latent.2.2-suffix.2-rnn.1.tanh` |
| 33 | 4.9538 | 5 | 28.2 s | 4×512 | 1/8↘0 | 262144 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-latent.2.2-split.add(suffix.2-rnn.1.tanh, pass)` |
| 34 | 4.9218 | 7 | 31.0 s | 2×1024 | 1/16↘0 | 262144 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-latent.2.2-suffix.2-split.add(dense.1.tanh, pass)-norm` |
| 35 | 4.9035 | 2 | 3 m 25 s | 4×2048 | 1/16↘0 | 524288 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-latent.2.2-suffix.2-split.cat(dense.1.tanh, pass)-norm` |
| 40 | 4.8995 | 4 | 16.9 s | 1×1024 | 1/32↘0 | 262144 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-latent.2.2-norm-split.cat(suffix.2-dense.1.tanh, pass)-norm-suffix.2-dense.1.tanh-dense.1.tanh` |
| 41 | 4.8912 | 8 | 30.4 s | 2×1024 | 1/16↘0 | 262144 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-latent.2.2-suffix.2-split.add(dense.2.tanh, pass)-norm-dense.1.tanh` |
| 42 | 4.8451 | 9 | 34.4 s | 1×2048 | 1/32↘0 | 262144 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-latent.2.2-split.cat(suffix.4-dense.1.tanh, pass)-norm-suffix.2-dense.1.tanh` |
| 44 | 4.8201 | 4 | 2 m 10 s | 1×2048 | 1/32↘0 | 1048576 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-latent.2.2-split.cat(suffix.2-dense.1.tanh, pass)-norm-suffix.2-split.cat(dense.1.tanh, pass)` |
| 47 | 4.8082 | 4 | 1 m 8 s | 1×4096 | 1/32↘0 | 65536 | fp32 | `bits.1+bp\|rnn.4.gelu-norm-suffix.2-dense.1.tanh` |
| 49 | 4.7840 | 4 | 50.9 s | 2×1024 | 1/32↘0 | 262144 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-latent.2.2-split.cat(suffix.4-rnn.1.tanh, pass)-norm-split.cat(suffix.2-dense.1.tanh, pass)-norm-rnn.1.tanh` |
| 52 | 4.7615 | 3 | 2 m 5 s | 1×2048 | 1/32↘0 | 131072 | fp32 | `bits.1+bp\|split.add(mingru.1, pass)-norm-split.mul(latent.2.2, dense.2)-split.cat(suffix.4-dense.1.gelu, pass)-norm-suffix.2-dense.1.tanh` |
| 55 | 4.7441 | 7 | 16.0 s | 2×512 | 1/32↘0 | 262144 | fp32 | `bits.1+bp\|rnn.4.tanh-split.add(mingru.1, pass)-norm-suffix.2` |
| 57 | 4.6969 | 5 | 1 m 32 s | 4×1024 | 1/32↘0 | 131072 | fp32 | `bits.1+bp\|rnn.4.tanh-split.cat(mingru.1, pass)-norm-suffix.2` |
| 59 | 4.6791 | 3 | 6 m 10 s | 4×1024 | 1/128↘0 | 524288 | fp32 | `bits.1+bp\|rnn.4.tanh-split.cat(mgru.1, pass)-norm-suffix.2` |
| 65 | 4.6571 | 2 | 1 m 58 s | 4×1024 | 1/32↘0 | 131072 | fp32 | `bits.1+bp\|rnn.4.tanh-split.cat(gru.1, pass)-suffix.2` |
| 69 | 4.6157 | 5 | 1 m 31 s | 4×1024 | 1/32↘0 | 131072 | fp32 | `bits.1+bp\|rnn.4.tanh-split.cat(mingru.2, pass)-norm-suffix.2` |
| 73 | 4.6156 | 3 | 2 m 33 s | 2×1024 | 1/32↘0 | 1048576 | fp32 | `bits.1+bp\|rnn.4.tanh-norm-suffix.2-split.cat(mingru.1, pass)-norm-suffix.2` |
| 77 | 4.6012 | 3 | 5 m 28 s | 4×1024 | 1/32↘0 | 262144 | fp32 | `bits.1+bp\|rnn.4.tanh-rnn.4.tanh` |
| 79 | 4.5924 | 9 | 1 m 38 s | 4×1024 | 1/32↘0 | 131072 | fp32 | `bits.1+bp\|rnn.4.tanh-norm-split.add(split.add(mingru.2-split.mul(pass, dense.2.tanh), pass), pass)-norm-suffix.4` |
| 80 | 4.5449 | 3 | 5 m 51 s | 4×512 | 1/16↘0 | 524288 | fp32 | `bits.1+bp\|mgru.4-norm-rnn.1.tanh` |
| 84 | 4.5407 | 5 | 1 m 18 s | 1×2048 | 1/16↘0 | 262144 | fp32 | `bits.1+bp\|split.add(mgru.2, pass)-mingru.4-norm-split.cat(suffix.2-rnn.1.tanh, pass)` |
| 85 | 4.5032 | 5 | 44.3 s | 1×2048 | 1/16↘0 | 262144 | fp32 | `bits.1+bp\|rnn.4.tanh-mingru.4-norm-suffix.2` |
| 87 | 4.4959 | 4 | 45.1 s | 1×2048 | 1/16↘0 | 262144 | fp32 | `bits.1+bp\|rnn.4.tanh-mingru.4-norm-suffix.2-dense.1.tanh` |
| 93 | 4.4740 | 3 | 2 m 51 s | 4×1024 | 1/32↘0 | 65536 | fp32 | `bits.1+bp\|split.mul(rnn.4.tanh, dense.4)-norm-split.add(mingru.2, pass)-norm-suffix.4` |
| 97 | 4.4564 | 3 | 3 m 7 s | 32×512 | 1/32↘0 | 131072 | fp32 | `bits.1+bp\|mgru.4-dense.4.gelu-norm` |
| 101 | 4.4424 | 6 | 1.32 s | 1×512 | 1/8↘0 | 32768 | fp32 | `bits.4.emb.4\|conv.2-norm-conv.2-split.add(split.add(rmsnorm, pass), pass)` |
| 104 | 4.4387 | 4 | 1 m 19 s | 64×256 | 1/16↘0 | 65536 | fp32 | `bits.1+bp\|mgru.4-norm-split.cat(dense.4.gelu, pass)-norm-rnn.1.tanh` |
| 105 | 4.4070 | 9 | 8.88 s | 2×256 | 1/64↘0 | 131072 | fp32 | `bits.4.emb.4\|rglru.4-norm-split.add(norm, pass)-conv.2` |
| 113 | 4.3715 | 7 | 7.86 s | 4×256 | 1/32↘0 | 65536 | fp32 | `bits.4.emb.4\|rglru.2-norm-conv.2` |
| 117 | 4.3498 | 5 | 1 m 14 s | 4×256 | 1/32↘0 | 131072 | fp32 | `bits.4.emb.4\|rglru.2-split.add(rmsnorm, pass)-conv.2` |
| 121 | 4.3488 | 7 | 6.71 s | 2×256 | 1/32↘0 | 131072 | fp32 | `bits.4.emb.4\|split.add(split.add(split.add(mingru.4-rmsnorm, pass), pass), pass)-rmsnorm` |
| 124 | 4.3436 | 4 | 4 m 20 s | 8×512 | 1/16↘0 | 524288 | fp32 | `bits.1+bp\|rnn.8.tanh-split.cat(rnn.1.tanh, pass)` |
| 125 | 4.3111 | 4 | 3.78 s | 1×256 | 1/16↘0 | 16384 | fp32 | `bits.4.emb.4\|split.add(split.add(conv.2-norm, pass)-norm, pass)-norm-split.add(split.add(norm, pass), pass)-mingru.4` |
| 133 | 4.2742 | 8 | 3.07 s | 4×128 | 1/32↘0 | 32768 | fp32 | `bits.4.emb.4\|rglru.2-split.add(norm, pass)-conv.2-split.mul(pass, dense.4)` |
| 141 | 4.2620 | 9 | 1 m 16 s | 4×512 | 1/16↘0 | 262144 | fp32 | `bits.4.emb.4\|rglru.2-conv.4-split.mul(pass, dense.4)` |
| 145 | 4.2528 | 7 | 5.40 s | 64×32 | 1/16↘0 | 16384 | fp32 | `bits.4.emb.4\|mingru.4-norm-conv.2-split.mul(pass, dense.4)` |
| 149 | 4.2358 | 9 | 1.40 s | 1×256 | 1/32↘0 | 32768 | fp32 | `bits.4.emb.4\|split.add(conv.2-split.add(split.add(rmsnorm, pass), pass)-mingru.4, pass)-split.mul(pass, dense.4)` |
| 153 | 4.2264 | 7 | 3.94 s | 1×1024 | 1/32↘0 | 32768 | fp32 | `bits.4.emb.4\|split.add(dense.4.tanh-split.add(dense.4.gelu-conv.4-split.add(split.add(norm, pass), pass), pass)-split.mul(pass, dense.4), pass)` |
| 159 | 4.2158 | 5 | 4.58 s | 1×1024 | 1/32↘0 | 32768 | fp32 | `bits.4.emb.4\|split.add(dense.2.tanh-split.add(dense.4.gelu-conv.4-split.add(split.add(rmsnorm, pass), pass), pass)-split.mul(pass, dense.4), pass)-split.mul(pass, dense.4)` |
| 161 | 4.1908 | 6 | 32.2 s | 4×128 | 1/32↘0 | 65536 | fp32 | `bits.4.emb.4\|rglru.2-norm-conv.4-split.mul(pass, dense.4)-split.mul(pass, dense.4.tanh)` |
| 169 | 4.1891 | 7 | 2.08 s | 2×256 | 1/32↘0 | 32768 | fp32 | `bits.4.emb.4\|split.add(conv.2-split.add(split.add(dense.4.tanh-rmsnorm, pass), pass)-mingru.4, pass)-split.mul(pass, dense.4)` |
| 176 | 4.1626 | 7 | 1 m 49 s | 16×2048 | 1/16↘0 | 65536 | fp32 | `bits.4.oh+bp\|dense.2.tanh-suffix.2-split.add(mingru.4, pass)-norm-split.add(dense.4.gelu, pass)` |
| 185 | 4.1501 | 5 | 29.4 s | 2×512 | 1/16↘0 | 65536 | fp32 | `bits.4.emb.4\|suffix.2-mingru.4-split.mul(pass, dense.4)-split.mul(pass, dense.4.tanh)` |
| 192 | 4.1418 | 6 | 14.5 s | 4×256 | 1/16↘0 | 131072 | fp32 | `bits.4.oh+bp\|dense.2.tanh-suffix.2-split.add(mingru.4, pass)-split.add(suffix.2-dense.4.gelu, pass)` |
| 193 | 4.1163 | 6 | 21.1 s | 4×512 | 1/32↘0 | 65536 | fp32 | `bits.4.emb.4\|rglru.2-norm-suffix.2-mingru.4-split.add(split.mul(pass, dense.4), pass)` |
| 196 | 4.1106 | 4 | 3 m 11 s | 4×1024 | 1/16↘0 | 524288 | fp32 | `bits.4.oh+bp\|dense.2.tanh-suffix.2-mingru.4-norm-split.add(dense.4.gelu-split.add(dense.4.tanh, pass), pass)` |
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
| 263 | 3.9218 | 5 | 18.3 s | 4×256 | 1/16↘0 | 32768 | fp32 | `tokens.32.shift.emb.4\|split.add(split.add(split.add(conv.2-norm, pass)-split.mul(norm, dense.4), pass)-split.add(split.add(norm, pass)-split.add(rmsnorm, pass), pass)-mingru.4, pass)` |
| 273 | 3.9196 | 5 | 19.3 s | 2×512 | 1/16↘0 | 32768 | fp32 | `tokens.32.fold.emb.4\|split.add(split.add(split.add(conv.2-rmsnorm, pass)-split.mul(norm, dense.4), pass)-split.add(split.add(norm, pass)-rmsnorm, pass)-mingru.4, pass)` |
| 275 | 3.9133 | 6 | 40.9 s | 2×512 | 1/32↘0 | 65536 | fp32 | `tokens.32.shift.emb.4\|split.add(split.add(split.add(conv.4-rmsnorm, pass)-split.mul(norm, dense.4), pass)-split.add(split.add(norm, pass)-rmsnorm, pass)-mingru.4, pass)` |
| 283 | 3.8991 | 5 | 20.0 s | 4×256 | 1/16↘0 | 32768 | fp32 | `tokens.32.shift.emb.4\|split.add(split.add(dense.4.tanh-split.add(conv.2-norm, pass)-split.mul(norm, dense.4), pass)-split.add(split.add(norm, pass)-split.add(rmsnorm, pass), pass)-mingru.4, pass)` |
| 287 | 3.8900 | 6 | 44.8 s | 2×512 | 1/16↘0 | 65536 | fp32 | `tokens.32.shift.emb.4\|split.add(split.add(split.add(conv.2-rmsnorm, pass)-split.mul(norm, dense.4), pass)-split.add(split.add(dense.4.gelu-split.add(norm, pass), pass)-rmsnorm, pass)-mingru.4, pass)` |
| 303 | 3.8718 | 5 | 40.1 s | 2×512 | 1/16↘0 | 65536 | fp32 | `tokens.32.shift.emb.4\|split.add(conv.2-norm-split.mul(dense.4.tanh, dense.4), pass)-split.add(split.add(rmsnorm-split.mul(pass, dense.4.tanh), pass)-mingru.4, pass)` |
| 315 | 3.8637 | 4 | 18.0 s | 32×256 | 1/16↘0 | 16384 | fp32 | `tokens.32.shift.emb.4\|split.cat(rnn.4.tanh-dense.4.tanh, pass)-mingru.4` |
| 323 | 3.8538 | 4 | 44.3 s | 2×512 | 1/16↘0 | 65536 | fp32 | `tokens.32.shift.emb.4\|split.add(conv.2-norm-split.mul(dense.4.tanh, dense.4), pass)-split.add(split.add(split.mul(rmsnorm, dense.4)-split.mul(pass, dense.4.tanh), pass)-mingru.4, pass)` |
| 327 | 3.8508 | 6 | 23.8 s | 1×512 | 1/16↘0 | 32768 | fp32 | `tokens.32.shift.emb.4\|split.add(split.add(split.add(conv.2-rmsnorm, pass)-split.mul(norm, dense.4), pass)-split.mul(pass, dense.4)-split.add(split.add(split.add(norm-split.mul(dense.4, dense.4), pass)-rmsnorm, pass)-mingru.4, pass), pass)` |
| 337 | 3.8479 | 4 | 15.7 s | 32×64 | 1/8↘0 | 16384 | fp32 | `tokens.32.shift.emb.4\|split.add(split.mul(split.add(lrnn.4.2-norm-dense.2, pass)-rmsnorm, dense.4)-rmsnorm-split.add(mingru.4-split.mul(pass, dense.4.tanh), pass), pass)` |
| 339 | 3.8292 | 4 | 1 m 6 s | 32×64 | 1/8↘0 | 32768 | fp32 | `tokens.32.shift.emb.4\|split.mul(split.add(lrnn.4.4-dense.4, pass), dense.4)-split.add(mingru.4-split.mul(pass, dense.4.tanh), pass)` |
| 351 | 3.8289 | 3 | 31.2 s | 1×512 | 1/16↘0 | 32768 | fp32 | `tokens.32.shift.emb.4\|split.add(split.add(dense.4.tanh-split.add(conv.2-rmsnorm, pass)-split.mul(norm, dense.4), pass)-split.mul(pass, dense.4)-split.add(split.add(split.add(norm-split.mul(dense.4, dense.4)-rmsnorm, pass)-rmsnorm, pass)-mingru.4, pass), pass)` |
| 363 | 3.8226 | 4 | 1 m 8 s | 32×64 | 1/8↘0 | 32768 | fp32 | `tokens.32.shift.emb.4\|split.add(split.mul(split.add(lrnn.4.4-dense.4, pass)-split.mul(rmsnorm, dense.4), dense.4)-split.add(mingru.4-split.mul(pass, dense.4.tanh), pass), pass)` |
| 379 | 3.8195 | 3 | 45.8 s | 32×64 | 1/8↘0 | 16384 | fp32 | `tokens.32.shift.emb.4\|split.mul(pass, split.add(lrnn.4.4-dense.4, pass))-split.add(dense.4.tanh-mingru.4-split.mul(pass, dense.4.tanh)-mingru.4, pass)` |
| 380 | 3.8075 | 3 | 5 m 15 s | 8×2048 | 1/16↘0 | 131072 | fp32 | `bits.4.oh+bp\|dense.4.tanh-split.add(suffix.2-rnn.4.tanh-dense.8.gelu-norm, pass)-split.add(dense.8.gelu, pass)` |
| 383 | 3.7984 | 4 | 1 m 28 s | 32×64 | 1/8↘0 | 32768 | fp32 | `tokens.32.shift.emb.4\|split.add(split.mul(split.add(lrnn.4.4-dense.4, pass)-split.mul(split.mul(rmsnorm, dense.4), dense.4), dense.4)-split.add(mingru.4-split.mul(pass, dense.4.tanh), pass), pass)` |
| 395 | 3.7960 | 5 | 10.3 s | 64×128 | 1/8↘0 | 16384 | fp32 | `tokens.32.shift.emb.8\|split.add(split.add(mingru.4, pass), pass)-split.add(split.add(rmsnorm, pass), pass)` |
| 400 | 3.7907 | 5 | 1 m 20 s | 16×1024 | 1/16↘0 | 65536 | fp32 | `bits.4.oh+bp\|dense.4.tanh-split.add(suffix.2-mingru.4-dense.8.gelu-norm, pass)-split.add(dense.8.gelu, pass)` |
| 405 | 3.7907 | 5 | 10.7 s | 4×256 | 1/16↘0 | 65536 | fp32 | `bits.4.emb.8\|split.mul(mingru.4, dense.4.tanh)-suffix.2-dense.8.gelu-split.add(dense.8.gelu-rmsnorm, pass)` |
| 407 | 3.7716 | 5 | 24.8 s | 16×128 | 1/16↘0 | 65536 | fp32 | `tokens.32.shift.emb.8\|mingru.4-split.cat(dense.4.tanh, pass)` |
| 415 | 3.7685 | 5 | 12.5 s | 64×128 | 1/8↘0 | 16384 | fp32 | `tokens.32.shift.emb.8\|split.add(rnn.4.tanh-split.add(norm, pass)-split.add(dense.8.gelu-rmsnorm, pass), pass)` |
| 423 | 3.7311 | 2 | 1 m 47 s | 32×128 | 1/8↘0 | 262144 | fp32 | `tokens.32.shift.emb.8\|split.add(mingru.2, pass)-dense.8.tanh` |
| 451 | 3.7258 | 3 | 55.2 s | 256×64 | 1/8↘0 | 32768 | fp32 | `tokens.32.shift.emb.8\|rnn.8.tanh` |
| 455 | 3.7175 | 5 | 19.1 s | 16×128 | 1/16↘0 | 16384 | fp32 | `tokens.32.shift.emb.8\|split.add(lrnn.4.2, pass)-norm-dense.8.tanh` |
| 459 | 3.7118 | 4 | 25.6 s | 16×256 | 1/8↘0 | 32768 | fp32 | `tokens.32.shift.emb.8\|split.add(mingru.4, pass)-dense.8.tanh` |
| 472 | 3.7077 | 2 | 3 m 30 s | 16×256 | 1/16↘0 | 131072 | fp32 | `bits.4.oh+bp\|lrnn.4.2-suffix.2-mingru.8-rmsnorm-dense.8.gelu-split.add(norm, pass)` |
| 474 | 3.7044 | 3 | 2 m 8 s | 128×128 | 1/8↘0 | 32768 | fp32 | `tokens.32.hexbpe.emb.8\|lrnn.8.2` |
| 479 | 3.7022 | 4 | 36.6 s | 32×128 | 1/8↘0 | 32768 | fp32 | `tokens.32.shift.emb.8\|split.add(mingru.4-split.cat(dense.4, pass), pass)-dense.8.tanh` |
| 487 | 3.6850 | 3 | 1 m 13 s | 32×128 | 1/8↘0 | 65536 | fp32 | `tokens.32.shift.emb.8\|split.add(mingru.2, pass)-rnn.8.tanh` |
| 491 | 3.6776 | 3 | 35.7 s | 16×128 | 1/8↘0 | 32768 | fp32 | `tokens.32.shift.emb.8\|split.add(split.add(mgru.4, pass), pass)-dense.8.tanh` |
| 511 | 3.6641 | 4 | 25.6 s | 16×128 | 1/8↘0 | 32768 | fp32 | `tokens.32.shift.emb.8\|split.add(mgru.4-split.mul(pass, dense.4), pass)-dense.8.tanh` |
| 515 | 3.6466 | 6 | 1 m 7 s | 64×64 | 1/4↘0 | 16384 | fp32 | `tokens.32.shift.emb.8\|lrnn.8.4` |
| 551 | 3.6294 | 4 | 2 m 12 s | 32×64 | 1/16↘0 | 32768 | fp32 | `tokens.32.shift.emb.8\|lrnn.8.4-split.add(split.add(split.add(dense.4.gelu, pass), pass), pass)` |
| 567 | 3.6179 | 5 | 2 m 18 s | 32×64 | 1/16↘0 | 32768 | fp32 | `tokens.32.shift.emb.8\|lrnn.8.4-split.add(split.add(split.add(latent.4.2, pass), pass), pass)` |
| 587 | 3.5904 | 4 | 3 m 5 s | 32×64 | 1/4↘0 | 32768 | fp32 | `tokens.32.shift.emb.8\|lrnn.8.4-split.mul(pass, dense.8)` |
| 623 | 3.5852 | 4 | 2 m 17 s | 32×64 | 1/16↘0 | 32768 | fp32 | `tokens.32.shift.emb.8\|lrnn.8.4-split.add(split.add(split.add(split.add(dense.4.gelu, pass)-dense.8.gelu, pass), pass), pass)` |
| 659 | 3.5808 | 3 | 51.0 s | 64×64 | 1/4↘0 | 8192 | fp32 | `tokens.32.shift.emb.8\|lrnn.8.4-split.add(split.mul(pass, dense.8)-split.mul(pass, dense.8.tanh), pass)` |
| 667 | 3.5734 | 4 | 15.9 s | 16×128 | 1/8↘0 | 32768 | fp32 | `tokens.32.shift.emb.8\|mingru.8-split.mul(latent.8.2, dense.8)` |
| 675 | 3.5686 | 5 | 6.78 s | 64×32 | 1/16↘0 | 8192 | fp32 | `tokens.32.shift.emb.8\|split.add(rnn.4.tanh-dense.4.tanh, pass)-rnn.8.tanh-split.add(split.add(split.mul(dense.8-rmsnorm, dense.8), pass), pass)` |
| 690 | 3.5673 | 5 | 4.64 s | 16×128 | 1/8↘0 | 8192 | fp32 | `tokens.32.hexbpe.emb.8\|mgru.8-split.add(dense.8.tanh, pass)-split.add(split.mul(pass, dense.8), pass)` |
| 695 | 3.5525 | 4 | 1 m 13 s | 32×32 | 1/16↘0 | 32768 | fp32 | `tokens.32.shift.emb.8\|lrnn.8.4-split.add(split.add(split.add(dense.4.tanh, pass)-mingru.8, pass), pass)` |
| 715 | 3.5427 | 5 | 1 m 18 s | 32×32 | 1/16↘0 | 32768 | fp32 | `tokens.32.shift.emb.8\|lrnn.8.4-split.add(split.add(split.add(dense.4.tanh-rglru.4, pass)-mingru.8, pass), pass)` |
| 723 | 3.5402 | 3 | 1 m 6 s | 64×128 | 1/16↘0 | 65536 | fp32 | `tokens.32.shift.emb.8\|mgru.8-split.add(split.add(latent.8.2, pass), pass)` |
| 762 | 3.5127 | 5 | 8.26 s | 16×256 | 1/8↘0 | 8192 | fp32 | `tokens.32.hexbpe.emb.8\|mgru.8-split.add(dense.8.tanh, pass)-split.add(split.mul(dense.8, dense.8), pass)` |
| 826 | 3.5083 | 5 | 11.7 s | 16×128 | 1/8↘0 | 16384 | fp32 | `tokens.32.hexbpe.emb.8\|mgru.8-split.add(split.add(latent.8.2-split.mul(pass, dense.8.tanh), pass)-split.mul(pass, dense.8), pass)` |
| 832 | 3.5041 | 4 | 56.1 s | 128×128 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.16.gelu-dense.8.gelu-rmsnorm` |
| 834 | 3.4818 | 3 | 21.5 s | 16×256 | 1/8↘0 | 16384 | fp32 | `tokens.32.hexbpe.emb.8\|dense.8.gelu-mgru.8-split.add(dense.8.tanh, pass)-split.add(split.mul(dense.8, dense.8), pass)` |
| 875 | 3.4790 | 5 | 21.5 s | 64×64 | 1/16↘0 | 32768 | fp32 | `tokens.32.shift.emb.8\|split.add(mgru.8-split.add(dense.8.tanh-split.add(split.mul(split.add(dense.8, pass), dense.8)-split.mul(pass, dense.8), pass), pass), pass)` |
| 906 | 3.4661 | 5 | 8.58 s | 16×256 | 1/8↘0 | 8192 | fp32 | `tokens.32.hexbpe.emb.8\|dense.8.tanh-mgru.8-split.add(dense.8.gelu-dense.8.tanh, pass)-split.add(split.mul(dense.8, dense.8), pass)` |
| 944 | 3.4507 | 3 | 4 m 29 s | 16×512 | 1/64↘0 | 65536 | fp32 | `bits.4.oh+bp\|lrnn.8.2-norm-rnn.16.gelu` |
| 962 | 3.4455 | 5 | 11.6 s | 16×128 | 1/8↘0 | 16384 | fp32 | `tokens.32.hexbpe.emb.8\|gru.8-split.add(latent.8.2-split.mul(pass, dense.8), pass)-split.mul(pass, dense.8)` |
| 1003 | 3.4448 | 3 | 1 m 1 s | 16×128 | 1/8↘0 | 16384 | fp32 | `tokens.32.shift.emb.8\|gru.8-split.add(latent.8.2-split.mul(pass, dense.8), pass)-split.mul(pass, dense.8)` |
| 1024 | 3.4387 | 4 | 3 m 25 s | 64×128 | 1/16↘0 | 131072 | fp32 | `bits.4.oh+bp\|rnn.8.tanh-mgru.8-suffix.2-dense.16.gelu-norm` |
| 1034 | 3.4350 | 4 | 12.4 s | 16×128 | 1/8↘0 | 16384 | fp32 | `tokens.32.hexbpe.emb.8\|gru.8-split.add(latent.8.2-split.mul(pass, dense.8), pass)-split.mul(dense.8.gelu, dense.8)` |
| 1042 | 3.4017 | 5 | 22.7 s | 16×128 | 1/8↘0 | 32768 | fp32 | `tokens.32.hexbpe.emb.8\|gru.8-split.add(split.add(dense.8.tanh-split.mul(pass, dense.8), pass)-split.mul(split.mul(pass, dense.8.tanh)-dense.8.tanh, dense.8), pass)` |
| 1104 | 3.3984 | 2 | 1 m 32 s | 16×512 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.16.gelu-dense.16.gelu-rmsnorm` |
| 1106 | 3.3883 | 3 | 43.3 s | 16×128 | 1/8↘0 | 32768 | fp32 | `tokens.32.hexbpe.emb.8\|gru.8-split.add(split.add(latent.8.2-split.mul(pass, dense.8), pass)-split.mul(split.mul(pass, dense.8)-dense.8.tanh, dense.8), pass)` |
| 1152 | 3.3729 | 5 | 52.3 s | 64×128 | 1/32↘0 | 32768 | fp32 | `bits.4.oh+bp\|rnn.16.gelu-norm-dense.16.gelu-split.add(conv.2-rmsnorm, pass)` |
| 1216 | 3.3693 | 4 | 2 m 35 s | 16×256 | 1/64↘0 | 65536 | fp32 | `bits.4.oh+bp\|lrnn.8.2-rnn.16.gelu-split.add(dense.16.gelu, pass)` |
| 1225 | 3.3641 | 3 | 1 m 52 s | 64×128 | 1/32↘0 | 65536 | fp32 | `tokens.32.hexbpe.oh\|rnn.16.gelu-dense.8.gelu` |
| 1288 | 3.3611 | 5 | 1 m 56 s | 16×512 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.16.gelu-rnn.8.tanh-suffix.2-dense.16.gelu` |
| 1304 | 3.3547 | 3 | 1 m 16 s | 64×128 | 1/16↘0 | 32768 | fp32 | `tokens.64.shift.emb.8\|split.cat(rglru.1, pass)-dense.16.gelu-mingru.8` |
| 1314 | 3.3418 | 2 | 3 m 35 s | 64×64 | 1/8↘0 | 16384 | fp32 | `tokens.32.hexbpe.emb.16\|lrnn.16.4-split.add(norm, pass)` |
| 1344 | 3.3252 | 3 | 37.6 s | 64×128 | 1/64↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.16.gelu-rnn.16.gelu-norm` |
| 1360 | 3.3001 | 3 | 1 m 16 s | 64×128 | 1/64↘0 | 131072 | fp32 | `bits.4.oh+bp\|rnn.16.gelu-rnn.16.gelu-split.add(rmsnorm, pass)` |
| 1497 | 3.2988 | 4 | 29.7 s | 32×128 | 1/32↘0 | 32768 | fp32 | `tokens.32.hexbpe.oh\|rnn.16.gelu-dense.16.tanh-dense.8.gelu` |
| 1552 | 3.2781 | 3 | 1 m 36 s | 32×512 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.8.tanh-suffix.2-dense.16.gelu-norm-rnn.16.tanh-split.add(dense.16.gelu, pass)` |
| 1616 | 3.2589 | 2 | 3 m 50 s | 128×128 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.16.gelu-rnn.16.gelu-dense.16.gelu` |
| 1696 | 3.2545 | 5 | 1 m 23 s | 32×256 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.16.gelu-rnn.16.gelu-norm-split.add(dense.16.tanh-rglru.16, pass)-norm` |
| 1697 | 3.2539 | 6 | 30.6 s | 32×128 | 1/32↘0 | 32768 | fp32 | `tokens.32.hexbpe.oh\|rnn.8.gelu-norm-rnn.16.gelu-norm-dense.8.tanh-norm-suffix.2-dense.16.gelu` |
| 1753 | 3.2422 | 4 | 24.8 s | 32×128 | 1/32↘0 | 16384 | fp32 | `tokens.32.hexbpe.oh\|rnn.16.gelu-dense.16.gelu-split.add(dense.8.gelu, pass)` |
| 1801 | 3.2288 | 3 | 50.3 s | 32×128 | 1/32↘0 | 32768 | fp32 | `tokens.32.hexbpe.oh\|rnn.8.gelu-norm-rnn.16.gelu-norm-rnn.8.tanh-norm-conv.2-split.mul(dense.16.gelu, dense.16)` |
| 1858 | 3.2252 | 4 | 1 m 60 s | 32×128 | 1/16↘0 | 131072 | fp32 | `tokens.32.hexbpe.emb.16\|mgru.16-dense.16.gelu` |
| 1865 | 3.2194 | 2 | 1 m 31 s | 32×128 | 1/32↘0 | 65536 | fp32 | `tokens.32.hexbpe.oh\|rnn.8.gelu-rnn.16.gelu-split.add(rmsnorm-split.add(mingru.16, pass)-rmsnorm, pass)` |
| 1888 | 3.2186 | 2 | 1 m 47 s | 32×256 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.16.gelu-rnn.16.gelu-norm-split.add(dense.16.gelu-dense.16.tanh, pass)-norm` |
| 1904 | 3.2167 | 5 | 44.3 s | 32×128 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.16.gelu-rnn.16.gelu-rmsnorm-mingru.16` |
| 1920 | 3.2131 | 6 | 46.5 s | 32×128 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.16.gelu-rnn.16.gelu-split.add(rmsnorm-mingru.16-rmsnorm, pass)` |
| 2016 | 3.2063 | 4 | 2 m 12 s | 128×256 | 1/16↘0 | 32768 | fp32 | `bits.4.oh+bp\|dense.8.gelu-mgru.16-dense.32.gelu-norm` |
| 2056 | 3.2018 | 5 | 1 m 57 s | 16×256 | 1/32↘0 | 131072 | fp32 | `bits.4.oh+bp\|dense.4.tanh-rnn.32.gelu-dense.16.gelu` |
| 2080 | 3.1892 | 3 | 30.8 s | 32×256 | 1/32↘0 | 32768 | fp32 | `bits.4.oh+bp\|rnn.8.tanh-norm-rnn.32.gelu-rmsnorm` |
| 2130 | 3.1864 | 4 | 2 m 2 s | 32×128 | 1/16↘0 | 131072 | fp32 | `tokens.32.hexbpe.emb.16\|split.mul(mgru.16, dense.16)-dense.16.gelu` |
| 2144 | 3.1859 | 5 | 1 m 52 s | 64×128 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.16.gelu-mgru.16-norm-dense.16.tanh` |
| 2160 | 3.1778 | 3 | 2 m 29 s | 16×256 | 1/64↘0 | 262144 | fp32 | `bits.4.oh+bp\|rnn.32.gelu-split.add(rmsnorm, pass)` |
| 2245 | 3.1649 | 3 | 1 m 20 s | 64×128 | 1/16↘0 | 32768 | fp32 | `tokens.64.hexbpe.emb.16\|mullstm.8-dense.16.gelu-rmsnorm` |
| 2256 | 3.1615 | 6 | 1 m 47 s | 32×128 | 1/64↘0 | 131072 | fp32 | `bits.4.oh+bp\|dense.8.tanh-rnn.32.gelu-dense.16.gelu` |
| 2272 | 3.1439 | 4 | 55.0 s | 32×128 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|dense.8.tanh-rnn.32.gelu-norm-dense.16.gelu-split.add(rmsnorm, pass)` |
| 2304 | 3.1358 | 2 | 1 m 10 s | 32×128 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|dense.8.tanh-rnn.32.gelu-rmsnorm-dense.16.gelu-split.add(rmsnorm, pass)` |
| 2336 | 3.1347 | 4 | 2 m 21 s | 32×128 | 1/32↘0 | 131072 | fp32 | `bits.4.oh+bp\|latent.8.2-rnn.32.gelu-dense.16.gelu-split.add(rmsnorm, pass)` |
| 2400 | 3.1347 | 6 | 5 m 23 s | 64×64 | 1/32↘0 | 262144 | fp32 | `bits.4.oh+bp\|rnn.32.gelu-dense.16.gelu` |
| 2416 | 3.1004 | 3 | 3 m 28 s | 64×128 | 1/32↘0 | 131072 | fp32 | `bits.4.oh+bp\|rnn.32.gelu-dense.16.gelu-rmsnorm` |
| 2656 | 3.0920 | 5 | 1 m 5 s | 32×256 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.32.gelu-rnn.16.gelu-norm` |
| 2745 | 3.0916 | 4 | 1 m 60 s | 32×128 | 1/32↘0 | 131072 | fp32 | `tokens.32.hexbpe.oh\|latent.8.2-rnn.32.gelu-dense.16.gelu-split.add(rmsnorm, pass)` |
| 2784 | 3.0831 | 5 | 2 m 13 s | 32×128 | 1/32↘0 | 131072 | fp32 | `bits.4.oh+bp\|dense.8.gelu-rnn.32.gelu-rnn.16.gelu-norm-dense.16.tanh` |
| 2928 | 3.0694 | 6 | 59.4 s | 32×128 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.32.gelu-rnn.16.gelu-split.add(dense.16.gelu, pass)` |
| 2944 | 3.0635 | 2 | 2 m 22 s | 64×256 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.32.gelu-dense.16.gelu-mingru.16` |
| 3169 | 3.0618 | 5 | 32.2 s | 16×128 | 1/32↘0 | 65536 | fp32 | `tokens.32.hexbpe.oh\|rnn.32.gelu-dense.16.gelu` |
| 3184 | 3.0410 | 5 | 7 m 8 s | 128×128 | 1/32↘0 | 131072 | fp32 | `bits.4.oh+bp\|rnn.32.gelu-dense.32.gelu` |
| 3185 | 3.0385 | 4 | 18.6 s | 32×64 | 1/32↘0 | 32768 | fp32 | `tokens.32.hexbpe.oh\|rnn.32.gelu-dense.16.gelu-rmsnorm` |
| 3216 | 3.0291 | 3 | 3 m 57 s | 16×512 | 1/32↘0 | 131072 | fp32 | `bits.4.oh+bp\|rnn.32.gelu-dense.32.gelu-rmsnorm` |
| 3441 | 3.0160 | 3 | 1 m 41 s | 64×256 | 1/16↘0 | 32768 | fp32 | `tokens.32.hexbpe.oh\|rnn.32.gelu-dense.16.gelu-dense.16.gelu` |
| 3568 | 3.0160 | 3 | 36.7 s | 64×128 | 1/32↘0 | 32768 | fp32 | `bits.4.oh+bp\|rnn.32.gelu-split.mul(dense.16.gelu-rglru.16-norm, dense.16)-norm-split.add(dense.16.gelu-norm-dense.16.gelu-rmsnorm, pass)` |
| 3600 | 3.0044 | 2 | 2 m 3 s | 64×128 | 1/32↘0 | 32768 | fp32 | `bits.4.oh+bp\|rnn.32.gelu-split.mul(dense.16.gelu-rglru.8-norm, dense.16)-norm-dense.16.gelu-norm-dense.16.gelu-rmsnorm` |
| 3728 | 3.0018 | 7 | 53.4 s | 64×128 | 1/32↘0 | 32768 | fp32 | `bits.4.oh+bp\|rnn.32.gelu-dense.16.gelu-split.add(suffix.2-dense.16.gelu, pass)-split.add(dense.32.gelu, pass)` |
| 3904 | 3.0000 | 4 | 26.7 s | 32×64 | 1/16↘0 | 16384 | fp32 | `tokens.64.shift.emb.16\|gru.16-split.add(split.mul(split.mul(dense.16, dense.16)-rglru.8-dense.16, dense.16), pass)` |
| 3953 | 2.9834 | 2 | 2 m 11 s | 64×256 | 1/32↘0 | 32768 | fp32 | `tokens.32.hexbpe.oh\|rnn.32.gelu-dense.16.gelu-split.cat(dense.16.gelu, pass)` |
| 3984 | 2.9665 | 3 | 45.1 s | 64×256 | 1/32↘0 | 32768 | fp32 | `bits.4.oh+bp\|rnn.32.gelu-rnn.16.gelu-split.add(suffix.2-dense.16.gelu, pass)-dense.32.gelu-norm` |
| 4208 | 2.9614 | 5 | 2 m 7 s | 16×512 | 1/64↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.32.gelu-rnn.32.gelu` |
| 4240 | 2.9573 | 4 | 1 m 13 s | 32×256 | 1/64↘0 | 65536 | fp32 | `bits.4.oh+bp\|rnn.32.gelu-rnn.32.gelu-rmsnorm` |
| 4481 | 2.9371 | 3 | 34.3 s | 64×64 | 1/32↘0 | 32768 | fp32 | `tokens.32.hexbpe.oh\|rnn.32.gelu-rnn.16.gelu-dense.32.gelu` |
| 4513 | 2.9280 | 3 | 38.2 s | 64×64 | 1/32↘0 | 16384 | fp32 | `tokens.32.hexbpe.oh\|rnn.32.gelu-split.cat(dense.16.gelu-mingru.16, pass)-dense.16.gelu-rmsnorm` |
| 4753 | 2.9221 | 3 | 34.6 s | 64×64 | 1/32↘0 | 32768 | fp32 | `tokens.32.hexbpe.oh\|rnn.32.gelu-mingru.16-dense.32.gelu` |
| 4832 | 2.9169 | 5 | 17.3 s | 32×128 | 1/16↘0 | 16384 | fp32 | `tokens.64.shift.emb.16\|mullstm.16-suffix.2-dense.16.gelu-split.mul(rmsnorm, dense.16)-split.add(split.mul(pass, dense.16), pass)` |
| 4964 | 2.9129 | 4 | 32.8 s | 64×128 | 1/32↘0 | 16384 | fp32 | `bits.4.oh+bp\|rnn.32.gelu-split.add(dense.16.gelu-mgru.16, pass)-dense.32.gelu-norm-split.cat(dense.4.gelu, pass)` |
| 5136 | 2.9067 | 5 | 20.0 s | 32×128 | 1/16↘0 | 16384 | fp32 | `tokens.64.shift.emb.16\|mullstm.16-dense.16.gelu-split.mul(rmsnorm-rglru.1, dense.16)-split.mul(pass, dense.16)` |
| 5265 | 2.8977 | 3 | 55.4 s | 128×128 | 1/16↘0 | 16384 | fp32 | `tokens.32.hexbpe.oh\|rnn.32.gelu-dense.32.gelu-split.add(dense.32.gelu, pass)` |
| 5408 | 2.8885 | 5 | 24.6 s | 32×128 | 1/16↘0 | 16384 | fp32 | `tokens.64.shift.emb.16\|mullstm.16-dense.16.gelu-split.add(split.mul(rmsnorm-rglru.1, dense.16), pass)-split.mul(pass, dense.16)-split.mul(pass, dense.16)` |
| 5585 | 2.8805 | 4 | 19.9 s | 64×64 | 1/32↘0 | 8192 | fp32 | `tokens.32.hexbpe.oh\|rnn.32.gelu-split.add(dense.16.gelu-dense.16.gelu-mingru.16, pass)-dense.32.gelu-rmsnorm` |
| 5841 | 2.8499 | 3 | 39.0 s | 64×64 | 1/32↘0 | 16384 | fp32 | `tokens.32.hexbpe.oh\|rnn.32.gelu-split.add(dense.16.gelu-rnn.16.gelu-mingru.16, pass)-dense.32.gelu-rmsnorm` |
| 6081 | 2.8382 | 2 | 45.8 s | 64×128 | 1/32↘0 | 16384 | fp32 | `tokens.32.hexbpe.oh\|rnn.32.gelu-split.add(split.cat(rnn.16.gelu-mingru.16, pass)-dense.32.gelu, pass)-rmsnorm` |
| 6641 | 2.7859 | 2 | 49.6 s | 64×128 | 1/32↘0 | 16384 | fp32 | `tokens.32.hexbpe.oh\|rnn.32.gelu-split.add(split.cat(rnn.16.gelu-mingru.16-rglru.1, pass)-dense.32.gelu, pass)-rmsnorm` |
| 7473 | 2.7786 | 2 | 1 m 10 s | 128×64 | 1/32↘0 | 16384 | fp32 | `tokens.32.hexbpe.oh\|rnn.32.gelu-split.cat(split.add(dense.16.gelu-rglru.16-dense.16.gelu, pass)-mgru.16-split.mul(pass, dense.16.tanh), pass)-dense.32.gelu-rmsnorm` |
| 7713 | 2.7598 | 4 | 1 m 10 s | 256×64 | 1/32↘0 | 16384 | fp32 | `tokens.32.hexbpe.oh\|rnn.32.gelu-split.cat(split.add(split.cat(dense.16.gelu-rglru.16, pass)-dense.16.gelu, pass)-mgru.16, pass)-dense.32.gelu-rmsnorm` |
| 7745 | 2.7557 | 2 | 4 m 39 s | 256×64 | 1/32↘0 | 32768 | fp32 | `tokens.32.hexbpe.oh\|rnn.32.gelu-split.cat(split.add(split.cat(dense.16.gelu-rglru.8, pass)-dense.16.gelu, pass)-mgru.16, pass)-dense.32.gelu-rmsnorm` |
| 8001 | 2.7527 | 3 | 1 m 32 s | 128×128 | 1/32↘0 | 16384 | fp32 | `tokens.32.hexbpe.oh\|rnn.32.gelu-split.cat(dense.32.tanh-split.add(dense.16.gelu-rglru.16-rmsnorm, pass)-mgru.16, pass)-dense.32.gelu-rmsnorm` |
| 8497 | 2.7483 | 2 | 1 m 14 s | 128×64 | 1/32↘0 | 16384 | fp32 | `tokens.32.hexbpe.oh\|rnn.32.gelu-split.cat(split.add(dense.16.gelu-rglru.16-dense.16.gelu, pass)-suffix.2-mgru.16-split.mul(pass, dense.16.tanh), pass)-dense.32.gelu-rmsnorm` |
| 8801 | 2.7097 | 2 | 4 m 42 s | 256×64 | 1/32↘0 | 32768 | fp32 | `tokens.32.hexbpe.oh\|rnn.32.gelu-split.cat(split.add(dense.32.gelu-split.cat(dense.16.gelu-rglru.8, pass)-dense.16.gelu, pass)-mgru.16, pass)-dense.32.gelu-rmsnorm` |
| 9824 | 2.6992 | 3 | 4 m 35 s | 128×128 | 1/32↘0 | 65536 | fp32 | `bits.4.oh+bp\|dense.8.gelu-mullstm.32-dense.64.gelu` |
