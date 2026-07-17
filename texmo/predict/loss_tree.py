"""Structure-mirroring loss predictor (the v2 design).

The predictor's compute graph mirrors the conf's layer DAG. Each conf
compiles to an instruction tape over a small register file of
fixed-width activation vectors: a plain layer reads its register and
overwrites it, a split forks (each branch runs as its own chain from
the fork register), and a merge reads both branch registers. The
whole batch executes as one `jax.lax.scan` over the tape.

Every instruction applies the same step shape:

    p       = Layer_params(features)                  # shared dense
    act_new = Layer_forward(p ++ act_src1 ++ act_src2)

with `act_src2` zeroed for plain layers (a merge flag rides in the
feature vector), so there is no per-type switch in the execution.
`Layer_forward` is either one shared dense or a per-type weight bank
(`per_type_fwd`) gathered per instruction -- the latter is the more
accurate and ~4-5x slower variant; see docs/loss_predictor_v2.md for
the measured trade-offs. The head reads the final activation
concatenated with `act0` (the codec/globals node) -- the codec owns
both ends of the real model, and this skip is load-bearing.

Shares the feature extraction, target clipping, and simple-type
discovery with `loss_rnn`.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..configuration import Configuration
from ..layers.split import SplitDef
from .loss_rnn import (
    N_INIT_GLOBAL,
    _init_global_features,
    _layer_feature_dim,
    _layer_features,
    discover_simple_types,
)
from .predict_common import MAX_LOSS, MIN_LOSS

__all__ = [
    'TreeLossModel', 'compile_conf', 'build_arrays',
    'discover_simple_types', 'fit', 'predict',
]

D_ACT = 32       # activation-vector width (one per register)
D_PARAMS = 16    # digested layer-params width
OUT_HIDDEN = 32  # head hidden width

# One weight-bank slot per simple type, plus one shared slot for the
# extra-dim types (suffix/conv/latent/...) and one for merges.
_EXTRA_SLOT = 0  # index offsets added to n_simple below
_MERGE_SLOT = 1


# -- Compilation -------------------------------------------------------


def _compile_seq(layers, src, dst, alloc, out, type_idx, n_simple):
    """Compile a layer chain reading register `src`, output in `dst`."""
    cur = src
    for layer in layers:
        if isinstance(layer, SplitDef):
            regs = []
            for branch in layer.branches:
                if branch.layers:
                    r = alloc()
                    _compile_seq(branch.layers, cur, r, alloc, out,
                                 type_idx, n_simple)
                    regs.append(r)
                else:
                    regs.append(cur)  # pass branch reads the fork value
            # Merge marker: the split's own features minus the folded
            # gate weights (the gate branch is a real chain here).
            p = _layer_features(layer, type_idx, n_simple).copy()
            p[0] = 0.0
            out.append((p, regs[0], regs[1], dst, True))
        else:
            p = _layer_features(layer, type_idx, n_simple)
            out.append((p, cur, cur, dst, False))
        cur = dst


def compile_conf(conf: Configuration, type_idx: dict[str, int],
                 n_simple: int):
    """Compile a conf into (params [T, P+1], src1, src2, dst, n_regs).

    The last params column is the merge flag; it gates the second
    activation input and selects the merge weight-bank slot."""
    out: list = []
    nregs = [1]

    def alloc():
        nregs[0] += 1
        return nregs[0] - 1

    layers = conf.model.layer_seq.layers
    if layers:
        _compile_seq(layers, 0, 0, alloc, out, type_idx, n_simple)
    else:
        # No hidden layers: one zero-feature instruction keeps the
        # tape non-empty (act[0] flows through a learned identity).
        out.append((np.zeros(_layer_feature_dim(n_simple),
                             dtype=np.float32), 0, 0, 0, False))
    params = np.stack([
        np.concatenate([p, [1.0 if merge else 0.0]])
        for p, _, _, _, merge in out
    ]).astype(np.float32)
    src1 = np.array([i[1] for i in out], dtype=np.int32)
    src2 = np.array([i[2] for i in out], dtype=np.int32)
    dst = np.array([i[3] for i in out], dtype=np.int32)
    return params, src1, src2, dst, nregs[0]


def build_arrays(confs: list[Configuration], type_idx: dict[str, int],
                 n_simple: int):
    """Batch-compile confs into padded tape arrays.

    Returns (globs, params, src1, src2, dst, mask, n_regs)."""
    tapes = [compile_conf(c, type_idx, n_simple) for c in confs]
    T = max(t[0].shape[0] for t in tapes)
    R = max(t[4] for t in tapes)
    P = tapes[0][0].shape[1]
    n = len(confs)
    params = np.zeros((n, T, P), dtype=np.float32)
    src1 = np.zeros((n, T), dtype=np.int32)
    src2 = np.zeros((n, T), dtype=np.int32)
    dst = np.zeros((n, T), dtype=np.int32)
    mask = np.zeros((n, T), dtype=np.float32)
    for i, (p, s1, s2, d, _) in enumerate(tapes):
        t = p.shape[0]
        params[i, :t] = p
        src1[i, :t] = s1
        src2[i, :t] = s2
        dst[i, :t] = d
        mask[i, :t] = 1.0
    globs = np.stack([_init_global_features(c) for c in confs])
    return globs, params, src1, src2, dst, mask, R


def _type_selector(tape_params: np.ndarray, n_simple: int) -> np.ndarray:
    """Weight-bank slot per instruction: the simple-type one-hot's
    argmax, extra-dim types folded to a shared slot, merges to their
    own."""
    b = tape_params[..., 3:3 + n_simple]
    s = np.argmax(b, axis=-1)
    s = np.where(b.max(axis=-1) > 0.5, s, n_simple + _EXTRA_SLOT)
    s = np.where(tape_params[..., -1] > 0.5, n_simple + _MERGE_SLOT, s)
    return s.astype(np.int32)


# -- Model -------------------------------------------------------------


def _init_params(rng, n_feat: int, n_types: int, per_type_fwd: bool):
    keys = jax.random.split(rng, 6)

    def init(key, fi, fo):
        return jax.random.normal(key, (fi, fo)) * jnp.sqrt(1.0 / fi)

    d, d_p = D_ACT, D_PARAMS
    params = {
        'W_glob': init(keys[0], N_INIT_GLOBAL, d),
        'b_glob': jnp.zeros(d),
        'W_p': init(keys[1], n_feat, d_p),
        'b_p': jnp.zeros(d_p),
        'W_h1': init(keys[2], 2 * d, OUT_HIDDEN),
        'b_h1': jnp.zeros(OUT_HIDDEN),
        'W_h2': init(keys[3], OUT_HIDDEN, 1),
        'b_h2': jnp.zeros(1),
    }
    if per_type_fwd:
        # Same shared init replicated per type; diverges in training.
        w0 = init(keys[4], d_p + 2 * d, d)
        params['W_step'] = jnp.broadcast_to(
            w0[None], (n_types,) + w0.shape) * jnp.ones((n_types, 1, 1))
        params['b_step'] = jnp.zeros((n_types, d))
    else:
        params['W_step'] = init(keys[4], d_p + 2 * d, d)
        params['b_step'] = jnp.zeros(d)
    return params


def _forward(params, globs, tape_params, src1, src2, dst, mask,
             n_regs: int, type_sel, per_type_fwd: bool):
    B = globs.shape[0]
    d = params['b_glob'].shape[0]
    act0 = jax.nn.gelu(globs @ params['W_glob'] + params['b_glob'])
    regs = jnp.zeros((B, n_regs, d)).at[:, 0].set(act0)

    feats = tape_params[..., :-1]
    merge_flag = tape_params[..., -1]
    p_all = jax.nn.gelu(feats @ params['W_p'] + params['b_p'])
    bi = jnp.arange(B)

    if per_type_fwd:
        def step(regs, inp):
            p, s1, s2, dt, m, mf, ts = inp
            a1 = regs[bi, s1]
            a2 = regs[bi, s2] * mf[:, None]
            x = jnp.concatenate([p, a1, a2], axis=-1)
            w = params['W_step'][ts]     # [B, d_p+2d, d]
            b = params['b_step'][ts]     # [B, d]
            new = jnp.tanh(jnp.einsum('bf,bfp->bp', x, w) + b)
            upd = jnp.where(m[:, None] > 0.5, new, regs[bi, dt])
            return regs.at[bi, dt].set(upd), None

        xs = (jnp.transpose(p_all, (1, 0, 2)),
              src1.T, src2.T, dst.T, mask.T, merge_flag.T, type_sel.T)
    else:
        def step(regs, inp):
            p, s1, s2, dt, m, mf = inp
            a1 = regs[bi, s1]
            a2 = regs[bi, s2] * mf[:, None]
            x = jnp.concatenate([p, a1, a2], axis=-1)
            new = jnp.tanh(x @ params['W_step'] + params['b_step'])
            upd = jnp.where(m[:, None] > 0.5, new, regs[bi, dt])
            return regs.at[bi, dt].set(upd), None

        xs = (jnp.transpose(p_all, (1, 0, 2)),
              src1.T, src2.T, dst.T, mask.T, merge_flag.T)

    regs, _ = jax.lax.scan(step, regs, xs)
    h = jnp.concatenate([regs[:, 0], act0], axis=-1)
    o = jax.nn.gelu(h @ params['W_h1'] + params['b_h1'])
    return (o @ params['W_h2'] + params['b_h2']).squeeze(-1)


def fit(
    train_data: list[tuple[Configuration, float]],
    simple_types: list[str],
    steps: int = 8000,
    lr: float = 0.02,
    seed: int | None = None,
    batch_size: int = 2048,
    per_type_fwd: bool = False,
) -> tuple[dict, np.ndarray]:
    """Fit tree-predictor params on (conf, loss) pairs.

    Returns (params, loss_trace)."""
    if seed is None:
        import random
        seed = random.randrange(2**31)
    type_idx = {t: i for i, t in enumerate(simple_types)}
    n_simple = len(simple_types)
    confs = [c for c, _ in train_data]
    globs, tape_params, src1, src2, dst, mask, R = build_arrays(
        confs, type_idx, n_simple)
    targets = np.array(
        [float(np.log2(max(min(l, MAX_LOSS), MIN_LOSS)))
         for _, l in train_data], dtype=np.float32)
    type_sel = (_type_selector(tape_params, n_simple)
                if per_type_fwd else None)

    n_feat = tape_params.shape[-1] - 1
    n_types = n_simple + 2
    rng = jax.random.PRNGKey(seed)
    rng, init_key = jax.random.split(rng)
    params = _init_params(init_key, n_feat, n_types, per_type_fwd)
    schedule = optax.cosine_decay_schedule(lr, steps, alpha=0.01)
    optimizer = optax.adamw(schedule, weight_decay=0.0)
    opt_state = optimizer.init(params)
    batch_idx = jax.random.randint(
        rng, (steps, batch_size), 0, globs.shape[0])

    globs_j = jnp.asarray(globs)
    tp_j = jnp.asarray(tape_params)
    s1_j = jnp.asarray(src1)
    s2_j = jnp.asarray(src2)
    d_j = jnp.asarray(dst)
    m_j = jnp.asarray(mask)
    t_j = jnp.asarray(targets)
    ts_j = jnp.asarray(type_sel) if type_sel is not None else None

    def loss_fn(params, idx):
        preds = _forward(
            params, globs_j[idx], tp_j[idx], s1_j[idx], s2_j[idx],
            d_j[idx], m_j[idx], R,
            ts_j[idx] if ts_j is not None else None, per_type_fwd)
        return jnp.mean(jnp.abs(preds - t_j[idx]))

    def step_fn(carry, idx):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params, idx)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    (params, _), trace = jax.lax.scan(
        step_fn, (params, opt_state), batch_idx)
    return params, np.asarray(trace)


def predict(
    params: dict,
    confs: list[Configuration],
    simple_types: list[str],
    per_type_fwd: bool = False,
    chunk: int = 16384,
) -> np.ndarray:
    """Predict log2-loss for confs. Unlike the flat RNN there is no
    max_layers to thread through: tape length is per-batch and the
    register file sizes itself from the confs at hand."""
    type_idx = {t: i for i, t in enumerate(simple_types)}
    n_simple = len(simple_types)
    outs = []
    for i in range(0, len(confs), chunk):
        globs, tp, s1, s2, d, m, R = build_arrays(
            confs[i:i + chunk], type_idx, n_simple)
        ts = (jnp.asarray(_type_selector(tp, n_simple))
              if per_type_fwd else None)
        outs.append(np.asarray(_forward(
            params, jnp.asarray(globs), jnp.asarray(tp),
            jnp.asarray(s1), jnp.asarray(s2), jnp.asarray(d),
            jnp.asarray(m), R, ts, per_type_fwd)))
    return np.concatenate(outs)


@dataclasses.dataclass
class TreeLossModel:
    """All state needed to predict with a trained tree predictor.

    Same no-back-compat policy as `LossModel`: feature-schema changes
    invalidate persisted instances; the loader's probe-and-refit
    handles it."""
    params: dict
    simple_types: list[str]
    per_type_fwd: bool = False

    def predict(self, confs: list[Configuration]) -> np.ndarray:
        return predict(
            self.params, confs, self.simple_types,
            per_type_fwd=self.per_type_fwd)
