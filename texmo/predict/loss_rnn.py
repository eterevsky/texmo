"""Tiny RNN for loss prediction.

A fixed-size hidden state is initialized from the configuration's
global features (output size + batch/length/steps/lr/decay), then
updated once per hidden layer using a per-layer feature vector.
`batch, length, steps, lr, decay` are passed in three places: the
initial dense, every RNN step, and the output dense. A final dense
maps the hidden state (plus globals) to a predicted log-loss.

Variable layer counts are handled by padding each sample to the
max layer count found in training data and using a per-step mask
inside jax.lax.scan: on padded steps the hidden state is carried
through unchanged via jnp.where.
"""

import random

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..configuration import Configuration
from ..layers.skip import SkipDef
from ..layers.suffix import SuffixDef
from .predict_common import MAX_LOSS, MIN_LOSS, layer_type_id

HIDDEN = 8
BATCH_SIZE = 1024


# A layer is "simple" if it's fully described by its input/output sizes
# plus its type (which is captured by the bool slot). The rest
# (suffix, latent, lrnn, skip) carry an extra dimension and get
# dedicated slots; we deliberately don't give them bool slots since
# their extra-dim slot already signals "this is that type" (non-zero
# only for that type, except skip.X=1 which we handle with raw X).
_EXTRA_DIM_TYPES = {'suffix', 'latent', 'lrnn'}


def _is_simple_type(t: str) -> bool:
    return t not in _EXTRA_DIM_TYPES and not t.startswith('skip.')


def _layer_output_size(layer) -> int:
    if isinstance(layer, SuffixDef):
        return layer.input_size * layer.length
    if isinstance(layer, SkipDef):
        return layer.input_size
    return layer.size


# init_globals: output_size + the 5 training knobs. The last 5 (indices
# 1..5) are re-used at every RNN step and at the output dense.
def _init_global_features(conf: Configuration) -> np.ndarray:
    return np.array([
        np.log2(conf.model.output.size),
        np.log2(conf.batch),
        np.log2(conf.length),
        np.log2(conf.steps),
        np.log2(conf.lr),
        np.log2(conf.decay),
    ], dtype=np.float32)


N_INIT_GLOBAL = 6


def discover_simple_types(
    train_data: list[tuple[Configuration, float]],
) -> list[str]:
    """Sorted list of simple layer types (fully described by in/out sizes)."""
    seen: set[str] = set()
    for conf, _ in train_data:
        for layer in conf.model.layers:
            t = layer_type_id(layer)
            if _is_simple_type(t):
                seen.add(t)
    return sorted(seen)


def _layer_feature_dim(n_simple: int) -> int:
    # [log(num_weights), log(in), log(out)] + bool per simple type +
    # [suffix_len, latent_reps, lrnn_reps, skip_add_dist, skip_cat_dist]
    return 3 + n_simple + 5


def _layer_features(
    layer, simple_type_idx: dict[str, int], n_simple: int,
) -> np.ndarray:
    out_size = _layer_output_size(layer)
    feat = np.zeros(_layer_feature_dim(n_simple), dtype=np.float32)
    # log2(max(..., 1)) lets weightless layers (skip, possibly norm)
    # land at 0 rather than -inf.
    feat[0] = np.log2(max(layer.num_weights, 1))
    feat[1] = np.log2(layer.input_size)
    feat[2] = np.log2(out_size)
    t = layer_type_id(layer)
    if _is_simple_type(t):
        if t in simple_type_idx:
            feat[3 + simple_type_idx[t]] = 1.0
    elif t == 'suffix':
        feat[3 + n_simple + 0] = np.log2(layer.length)
    elif t == 'latent':
        feat[3 + n_simple + 1] = np.log2(layer.reps)
    elif t == 'lrnn':
        feat[3 + n_simple + 2] = np.log2(layer.reps)
    elif t == 'skip.add':
        feat[3 + n_simple + 3] = layer.distance
    elif t == 'skip.cat':
        feat[3 + n_simple + 4] = layer.distance
    return feat


def _build_input_arrays(
    confs: list[Configuration],
    simple_type_idx: dict[str, int],
    max_layers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(confs)
    n_simple = len(simple_type_idx)
    feat_dim = _layer_feature_dim(n_simple)
    init_globals = np.zeros((n, N_INIT_GLOBAL), dtype=np.float32)
    layer_feats = np.zeros((n, max_layers, feat_dim), dtype=np.float32)
    masks = np.zeros((n, max_layers), dtype=np.float32)
    for i, conf in enumerate(confs):
        init_globals[i] = _init_global_features(conf)
        for j, layer in enumerate(conf.model.layers[:max_layers]):
            layer_feats[i, j] = _layer_features(
                layer, simple_type_idx, n_simple)
            masks[i, j] = 1.0
    return init_globals, layer_feats, masks


def _build_targets(
    data: list[tuple[Configuration, float]],
) -> np.ndarray:
    return np.array(
        [float(np.log2(max(min(loss, MAX_LOSS), MIN_LOSS)))
         for _, loss in data],
        dtype=np.float32,
    )


def _init_params(
    rng, n_layer_feat: int, hidden: int,
    feat_proj: int, rnn_sub_steps: int, cell_type: str,
) -> dict:
    """Glorot-ish initialization for the dense blocks."""
    gate_mul = 3 if cell_type == 'gru' else 1  # reset, update, candidate
    keys = jax.random.split(rng, 4 + gate_mul * 3 * rnn_sub_steps)

    def init(key, fan_in: int, fan_out: int):
        scale = jnp.sqrt(1.0 / fan_in)
        return jax.random.normal(key, (fan_in, fan_out)) * scale

    params = {
        'W_glob': init(keys[0], N_INIT_GLOBAL, hidden),
        'b_glob': jnp.zeros(hidden),
        'W_out': init(keys[1], hidden, 1),
        'b_out': jnp.zeros(1),
    }
    if feat_proj > 0:
        params['W_proj'] = init(keys[2], n_layer_feat, feat_proj)
        params['b_proj'] = jnp.zeros(feat_proj)
        x_in_dim = feat_proj
    else:
        x_in_dim = n_layer_feat
    key_idx = 4
    for s in range(rnn_sub_steps):
        if cell_type == 'gru':
            for gate in ('r', 'z', 'c'):
                params[f'W_h_{gate}_{s}'] = init(
                    keys[key_idx], hidden, hidden); key_idx += 1
                params[f'W_x_{gate}_{s}'] = init(
                    keys[key_idx], x_in_dim, hidden); key_idx += 1
                params[f'b_{gate}_{s}'] = jnp.zeros(hidden)
        else:
            params[f'W_h_{s}'] = init(
                keys[key_idx], hidden, hidden); key_idx += 1
            params[f'W_x_{s}'] = init(
                keys[key_idx], x_in_dim, hidden); key_idx += 1
            params[f'b_rnn_{s}'] = jnp.zeros(hidden)
    return params


_ACTIVATIONS = {
    'tanh': jnp.tanh,
    'relu': jax.nn.relu,
    'gelu': jax.nn.gelu,
}


def _forward(params, init_globals, layer_feats, masks, cell_activation,
             feat_proj: int, rnn_sub_steps: int, cell_type: str,
             pooling: str):
    """Predict log-loss for each sample in the batch.

    init_globals: [B, N_INIT_GLOBAL]  (output_size + batch/len/steps/lr/decay)
    layer_feats: [B, max_layers, layer_feat_dim]
    masks: [B, max_layers]  (1.0 on real steps, 0.0 on padding)
    returns: [B]
    """
    act = _ACTIVATIONS[cell_activation]
    h = jax.nn.gelu(init_globals @ params['W_glob'] + params['b_glob'])
    if feat_proj > 0:
        # Project layer features once, before the scan. Saves compute and
        # produces the right gradient structure.
        layer_feats = jax.nn.gelu(
            layer_feats @ params['W_proj'] + params['b_proj'])
    # Transpose to scan along the layer axis.
    lf_t = jnp.transpose(layer_feats, (1, 0, 2))  # [max_layers, B, *]
    m_t = jnp.transpose(masks, (1, 0))            # [max_layers, B]

    def step(h, inp):
        lf, m = inp
        new_h = h
        for s in range(rnn_sub_steps):
            if cell_type == 'gru':
                r = jax.nn.sigmoid(
                    new_h @ params[f'W_h_r_{s}']
                    + lf @ params[f'W_x_r_{s}']
                    + params[f'b_r_{s}'])
                z = jax.nn.sigmoid(
                    new_h @ params[f'W_h_z_{s}']
                    + lf @ params[f'W_x_z_{s}']
                    + params[f'b_z_{s}'])
                cand = act(
                    (r * new_h) @ params[f'W_h_c_{s}']
                    + lf @ params[f'W_x_c_{s}']
                    + params[f'b_c_{s}'])
                new_h = (1.0 - z) * new_h + z * cand
            else:
                new_h = act(
                    new_h @ params[f'W_h_{s}']
                    + lf @ params[f'W_x_{s}']
                    + params[f'b_rnn_{s}']
                )
        kept = jnp.where(m[:, None] > 0.5, new_h, h)
        return kept, kept

    h_final, h_stack = jax.lax.scan(step, h, (lf_t, m_t))

    if pooling == 'mean':
        # Masked mean over valid steps of h_stack [max_layers, B, hidden].
        m_expand = m_t[..., None]  # [max_layers, B, 1]
        total = jnp.sum(h_stack * m_expand, axis=0)
        denom = jnp.maximum(jnp.sum(m_expand, axis=0), 1.0)
        h_used = total / denom
    else:  # 'last'
        h_used = h_final

    out = h_used @ params['W_out'] + params['b_out']
    return out.squeeze(-1)


def _loss_fn(params, init_globals, layer_feats, masks, targets,
             cell_activation, feat_proj: int, rnn_sub_steps: int,
             cell_type: str, pooling: str):
    preds = _forward(
        params, init_globals, layer_feats, masks, cell_activation,
        feat_proj, rnn_sub_steps, cell_type, pooling)
    return jnp.mean(jnp.abs(preds - targets))


def fit(
    train_data: list[tuple[Configuration, float]],
    simple_types: list[str],
    steps: int = 2000,
    lr: float = 0.01,
    seed: int | None = None,
    cell_activation: str = 'tanh',
    hidden: int = HIDDEN,
    lr_schedule: str = 'constant',  # 'constant' | 'cosine'
    feat_proj: int = 0,  # 0 = no projection; >0 = pre-RNN dense width
    rnn_sub_steps: int = 1,
    cell_type: str = 'vanilla',  # 'vanilla' | 'gru'
    pooling: str = 'last',  # 'last' | 'mean'
) -> tuple[dict, int, np.ndarray]:
    """Fit RNN params on (conf, loss) pairs.

    Returns (params, max_layers, loss_trace). `max_layers` is the
    scan depth used for training — pass it back to `predict`.
    Seed defaults to random so repeated runs show variance.
    """
    if seed is None:
        seed = random.randrange(2**31)
    type_idx = {t: i for i, t in enumerate(simple_types)}
    confs = [c for c, _ in train_data]
    max_layers = max((len(c.model.layers) for c in confs), default=1)
    max_layers = max(max_layers, 1)
    init_globals_np, layer_feats_np, masks_np = _build_input_arrays(
        confs, type_idx, max_layers)
    targets_np = _build_targets(train_data)
    init_globals = jnp.asarray(init_globals_np)
    layer_feats = jnp.asarray(layer_feats_np)
    masks = jnp.asarray(masks_np)
    targets = jnp.asarray(targets_np)

    n_layer_feat = _layer_feature_dim(len(simple_types))
    rng = jax.random.PRNGKey(seed)
    rng, init_key = jax.random.split(rng)
    params = _init_params(
        init_key, n_layer_feat, hidden, feat_proj, rnn_sub_steps, cell_type)
    if lr_schedule == 'cosine':
        schedule = optax.cosine_decay_schedule(
            init_value=lr, decay_steps=steps, alpha=0.01,
        )
        optimizer = optax.adamw(schedule, weight_decay=0.0)
    else:
        optimizer = optax.adamw(lr, weight_decay=0.0)
    opt_state = optimizer.init(params)

    # Pre-sample mini-batch indices for every training step.
    n_train = init_globals.shape[0]
    batch_idx = jax.random.randint(
        rng, (steps, BATCH_SIZE), 0, n_train)

    def step(carry, idx):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(_loss_fn)(
            params,
            init_globals[idx],
            layer_feats[idx],
            masks[idx],
            targets[idx],
            cell_activation,
            feat_proj,
            rnn_sub_steps,
            cell_type,
            pooling,
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    (params, _), losses = jax.lax.scan(
        step, (params, opt_state), batch_idx)
    return params, max_layers, np.asarray(losses)


def predict(
    params: dict,
    confs: list[Configuration],
    simple_types: list[str],
    max_layers: int,
    cell_activation: str = 'tanh',
    feat_proj: int = 0,
    rnn_sub_steps: int = 1,
    cell_type: str = 'vanilla',
    pooling: str = 'last',
) -> np.ndarray:
    type_idx = {t: i for i, t in enumerate(simple_types)}
    ig_np, lf_np, m_np = _build_input_arrays(
        confs, type_idx, max_layers)
    preds = _forward(
        params, jnp.asarray(ig_np), jnp.asarray(lf_np), jnp.asarray(m_np),
        cell_activation, feat_proj, rnn_sub_steps, cell_type, pooling)
    return np.asarray(preds)
