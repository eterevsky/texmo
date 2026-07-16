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

import dataclasses
import logging
import random
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .. import latency
from ..configuration import Configuration
from ..layers.split import SplitDef
from ..layers.suffix import SuffixDef
from .predict_common import MAX_LOSS, MIN_LOSS, layer_type_id, model_layers

if TYPE_CHECKING:
    from ..db import DbReader

HIDDEN = 8
BATCH_SIZE = 1024

# Log-space value of a clipped (diverged) run -- the upper mode of the
# target distribution and the gated head's switch target.
_LOG_MAX = float(np.log2(MAX_LOSS))


@dataclasses.dataclass
class LossModel:
    """All state needed to predict with a trained loss_rnn.

    No back-compat flags: the loss model is a few-minute refit cycle,
    so when we change the feature schema (new layer, new global, ...)
    we just bump the code, delete the saved row from the DB if any,
    and the next refit produces a fresh model.
    """
    params: dict
    simple_types: list[str]
    max_layers: int
    cell_activation: str = 'tanh'
    feat_proj: int = 0
    rnn_sub_steps: int = 1
    cell_type: str = 'elman'
    pooling: str = 'last'
    out_hidden: int = 0
    out_activation: str = 'gelu'
    gated_head: bool = False

    def predict(self, confs: list[Configuration]) -> np.ndarray:
        return predict(
            self.params, confs, self.simple_types, self.max_layers,
            cell_activation=self.cell_activation,
            feat_proj=self.feat_proj,
            rnn_sub_steps=self.rnn_sub_steps,
            cell_type=self.cell_type,
            pooling=self.pooling,
            out_hidden=self.out_hidden,
            out_activation=self.out_activation,
            gated_head=self.gated_head,
        )


class LossModelHolder:
    """Atomically-swappable holder for the current `LossModel`.

    The Model thread is the sole writer; it publishes new fits via
    `set_model`. Readers (Search threads) call `predict` directly.
    The reference swap is atomic in CPython, so no lock is needed.
    Holders start empty; `is_ready` reports whether any model has been
    published yet.
    """

    def __init__(self, model: LossModel | None = None):
        self._model = model

    def set_model(self, model: LossModel) -> None:
        self._model = model

    def is_ready(self) -> bool:
        return self._model is not None

    def predict(self, confs: list[Configuration]) -> np.ndarray:
        m = self._model
        assert m is not None, "predict called before any model was set"
        return m.predict(confs)


# A layer is "simple" if it's fully described by its input/output sizes
# plus its type (which is captured by the bool slot). The rest
# (suffix, latent, lrnn, msr, lmgu, conv, split) carry an extra
# dimension and get dedicated slots; we deliberately don't give them
# bool slots since their extra-dim slot already signals "this is that
# type" (non-zero only for that type).
_EXTRA_DIM_TYPES = {'suffix', 'latent', 'lrnn', 'msr', 'lmgu', 'conv'}


def _is_simple_type(t: str) -> bool:
    return (
        t not in _EXTRA_DIM_TYPES
        and not t.startswith('split.')
    )


def _layer_output_size(layer) -> int:
    if isinstance(layer, SuffixDef):
        return layer.input_size * layer.length
    if isinstance(layer, SplitDef):
        # add/cat keep the old skip convention: the marker reports the
        # merged-in (source = input) channels, and the merged width
        # flows on as the next layer's input_size -- so data labeled
        # under legacy skip markers matches. mul has no skip analog,
        # so report its true (main-sized) output.
        return (
            layer.input_size if layer.op in ('add', 'cat') else layer.size)
    return layer.size


def _jump_length(layer) -> int:
    """Residual/jump span: a split's main-branch length (equal to the
    distance of the legacy skip.D it translates from)."""
    return len(layer.branches[0].layers)


# init_globals: output_size + 5 log training knobs + cosine flag +
# 2 codec features + the total weight budget. `is_tied` is duck-typed
# off the head: a tied-embedding head has no parameters of its own
# (its matrix is the input table), while the one-hot dense head
# always does. The IO budget is `model.input.num_weights` -- the
# codec's parameters (one-hot: the implicit head; tied: table+scale)
# -- which the per-layer features never see (they only cover the
# hidden chain). log2(num_weights) is also the RF baseline's single
# best feature; the scan can't reconstruct it from the per-layer
# log-weights (sum of logs != log of sum), so it goes in directly
# (2026-07 sweep: -0.0008 val L1; nbits and log-num-mults were nulls).
def _init_global_features(conf: Configuration) -> np.ndarray:
    model = conf.model
    return np.array([
        np.log2(model.output.size),
        np.log2(conf.batch),
        np.log2(conf.length),
        np.log2(conf.steps),
        np.log2(conf.lr),
        np.log2(conf.decay),
        1.0 if conf.cosine else 0.0,
        1.0 if model.output.num_weights == 0 else 0.0,
        np.log2(1 + model.input.num_weights),
        np.log2(max(model.num_weights, 1)),
    ], dtype=np.float32)


N_INIT_GLOBAL = 10


def discover_simple_types(
    train_data: list[tuple[Configuration, float]],
) -> list[str]:
    """Sorted list of simple layer types (fully described by in/out sizes)."""
    seen: set[str] = set()
    for conf, _ in train_data:
        for layer in model_layers(conf):
            t = layer_type_id(layer)
            if _is_simple_type(t):
                seen.add(t)
    return sorted(seen)


def _layer_feature_dim(n_simple: int) -> int:
    # [log(num_weights), log(in), log(out)] + bool per simple type +
    # 11 extra-dim slots:
    #   [suffix_len, latent_reps, lrnn_reps,
    #    split_add_dist, split_cat_dist,
    #    msr_heads, lmgu_reps, conv_kernel,
    #    split_mul_dist, head_block_count, attn_window]
    # head_block_count = log2(rglru blocks / attn heads) -- the shared
    # partition-count slot. attn_window = log2(attn window). Additive --
    # rglru/attn still get their simple-type one-hots for identity (msr
    # keeps its own heads slot).
    return 3 + n_simple + 11


def _layer_features(
    layer, simple_type_idx: dict[str, int], n_simple: int,
) -> np.ndarray:
    out_size = _layer_output_size(layer)
    feat = np.zeros(_layer_feature_dim(n_simple), dtype=np.float32)
    # For a split marker count only the non-main (gate) branch weights:
    # the main branch is inlined as its own steps, so its weights are
    # already represented there. log2(max(..., 1)) lets weightless
    # markers (skip, residual split, norm) land at 0 rather than -inf.
    if isinstance(layer, SplitDef):
        weights = sum(b.num_weights for b in layer.branches[1:])
    else:
        weights = layer.num_weights
    feat[0] = np.log2(max(weights, 1))
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
    # split.add / split.cat live in what were the skip slots: skip.D
    # and its split translation are the same architecture, so all data
    # labeled before the representation switch transferred. split.mul
    # (no skip analog) gets its own slot. The proper per-branch
    # treatment lands with the v2 branching predictor.
    elif t == 'split.add':
        feat[3 + n_simple + 3] = _jump_length(layer)
    elif t == 'split.cat':
        feat[3 + n_simple + 4] = _jump_length(layer)
    elif t == 'msr':
        feat[3 + n_simple + 5] = np.log2(layer.heads)
    elif t == 'lmgu':
        feat[3 + n_simple + 6] = np.log2(layer.reps)
    elif t == 'conv':
        feat[3 + n_simple + 7] = np.log2(layer.kernel)
    elif t == 'split.mul':
        feat[3 + n_simple + 8] = _jump_length(layer)
    # Shared head/block-count slot (additive): rglru's block count and
    # attn's head count partition the channel dim the same way. Both
    # layers keep their simple-type one-hots; attn additionally sets
    # its window slot.
    if t == 'rglru':
        feat[3 + n_simple + 9] = np.log2(layer.blocks)
    elif t == 'attn':
        feat[3 + n_simple + 9] = np.log2(layer.heads)
        feat[3 + n_simple + 10] = np.log2(layer.window)
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
        for j, layer in enumerate(model_layers(conf)[:max_layers]):
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
    out_hidden: int, gated_head: bool = False,
) -> dict:
    """Glorot-ish initialization for the dense blocks."""
    gate_mul = 3 if cell_type == 'gru' else 1  # reset, update, candidate
    # Slots 0..4: W_glob, W_out, W_proj, W_pre_out, W_gate. 5+: per-RNN-step.
    keys = jax.random.split(rng, 5 + gate_mul * 3 * rnn_sub_steps)

    def init(key, fan_in: int, fan_out: int):
        scale = jnp.sqrt(1.0 / fan_in)
        return jax.random.normal(key, (fan_in, fan_out)) * scale

    out_in_dim = out_hidden if out_hidden > 0 else hidden
    params = {
        'W_glob': init(keys[0], N_INIT_GLOBAL, hidden),
        'b_glob': jnp.zeros(hidden),
        'W_out': init(keys[1], out_in_dim, 1),
        'b_out': jnp.zeros(1),
    }
    if out_hidden > 0:
        params['W_pre_out'] = init(keys[3], hidden, out_hidden)
        params['b_pre_out'] = jnp.zeros(out_hidden)
    if gated_head:
        params['W_gate'] = init(keys[4], out_in_dim, 1)
        # Start the gate near the base divergence rate (~2%) so the
        # head begins as the plain regressor and earns its sharpness.
        params['b_gate'] = jnp.full((1,), -4.0)
    if feat_proj > 0:
        params['W_proj'] = init(keys[2], n_layer_feat, feat_proj)
        params['b_proj'] = jnp.zeros(feat_proj)
        x_in_dim = feat_proj
    else:
        x_in_dim = n_layer_feat
    key_idx = 5
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
             pooling: str, out_hidden: int, out_activation: str,
             gated_head: bool = False):
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

    if out_hidden > 0:
        out_act = _ACTIVATIONS[out_activation]
        h_used = out_act(
            h_used @ params['W_pre_out'] + params['b_pre_out'])

    base = (h_used @ params['W_out'] + params['b_out']).squeeze(-1)
    if not gated_head:
        return base
    # Gated head: the median surface is discontinuous along the
    # p(diverge)=0.5 boundary (below: convergent quantile; above: the
    # clip). A sigmoid switch between the regressor and the known clip
    # value expresses that step directly -- the gate is an implicit
    # divergence classifier trained purely by the L1 objective.
    g = (h_used @ params['W_gate'] + params['b_gate']).squeeze(-1)
    return base + jax.nn.sigmoid(g) * (_LOG_MAX - base)


def _loss_fn(params, init_globals, layer_feats, masks, targets,
             cell_activation, feat_proj: int, rnn_sub_steps: int,
             cell_type: str, pooling: str,
             out_hidden: int, out_activation: str,
             gated_head: bool):
    preds = _forward(
        params, init_globals, layer_feats, masks, cell_activation,
        feat_proj, rnn_sub_steps, cell_type, pooling,
        out_hidden, out_activation, gated_head)
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
    cell_type: str = 'elman',  # 'elman' | 'gru'
    pooling: str = 'last',  # 'last' | 'mean'
    out_hidden: int = 0,  # 0 = single dense head; >0 = hidden -> X -> 1
    out_activation: str = 'gelu',
    batch_size: int = BATCH_SIZE,
    gated_head: bool = False,  # sigmoid switch to the divergence clip
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
    max_layers = max((len(model_layers(c)) for c in confs), default=1)
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
        init_key, n_layer_feat, hidden, feat_proj, rnn_sub_steps, cell_type,
        out_hidden, gated_head,
    )
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
        rng, (steps, batch_size), 0, n_train)

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
            out_hidden,
            out_activation,
            gated_head,
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    (params, _), losses = jax.lax.scan(
        step, (params, opt_state), batch_idx)
    return params, max_layers, np.asarray(losses)


def train_loss_model(
    db: 'DbReader',
) -> LossModel | None:
    """Retrain the loss-prediction model on all labeled runs in `db`.

    Applies the best config found in the sweep
    (docs/loss_rnn_experiments.md). Returns None if there are no
    labeled runs.
    """
    with latency.timer('train_loss_model.load'):
        train_data = [
            (conf, loss) for _, conf, loss in db.iter_labeled_runs()
        ]
    if not train_data:
        return None
    simple_types = discover_simple_types(train_data)
    with latency.timer('train_loss_model.fit'):
        params, max_layers, _trace = fit(
            train_data, simple_types,
            hidden=32, lr=0.02, steps=8000, lr_schedule='cosine',
            cell_activation='tanh',
            feat_proj=32, out_hidden=32, out_activation='gelu',
            batch_size=2048,
        )
    logging.info(f"Trained loss model on {len(train_data)} labeled runs")
    return LossModel(
        params=params,
        simple_types=simple_types,
        max_layers=max_layers,
        cell_activation='tanh',
        feat_proj=32,
        out_hidden=32,
        out_activation='gelu',
    )


def predict(
    params: dict,
    confs: list[Configuration],
    simple_types: list[str],
    max_layers: int,
    cell_activation: str = 'tanh',
    feat_proj: int = 0,
    rnn_sub_steps: int = 1,
    cell_type: str = 'elman',
    pooling: str = 'last',
    out_hidden: int = 0,
    out_activation: str = 'gelu',
    gated_head: bool = False,
) -> np.ndarray:
    type_idx = {t: i for i, t in enumerate(simple_types)}
    ig_np, lf_np, m_np = _build_input_arrays(
        confs, type_idx, max_layers)
    preds = _forward(
        params, jnp.asarray(ig_np), jnp.asarray(lf_np), jnp.asarray(m_np),
        cell_activation, feat_proj, rnn_sub_steps, cell_type, pooling,
        out_hidden, out_activation, gated_head)
    return np.asarray(preds)
