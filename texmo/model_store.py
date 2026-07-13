"""JSON model store: a full model as {spec, precision, weights}.

The `weights` value mirrors `Model2Jax.init_weights` EXACTLY:

    [codec_input_weights, [layer_1, layer_2, ...], head_weights]

with each layer's entry in the same shape its Jax implementation
accepts (dicts of named tensors; a Split's weights are the list of
its branches' weight lists, nested recursively). Leaves are one of:

- null            -- a parameter-free slot (one-hot codec input, the
                     tied codec's head, weightless layers);
- a number or nested lists of numbers -- an inline tensor (0-d
                     scalars like the tied codec's `y` are bare
                     numbers);
- {"path": ..., "id": ...} -- a descriptor: tensor `id` inside a
                     safetensors file. Relative paths resolve against
                     the manifest's own directory first, then the
                     working directory. This is how converted
                     checkpoints (RecurrentGemma) reference the
                     original shards without copying gigabytes. An
                     optional "transform": [[op, arg], ...] applies
                     in order after loading; ops: ["reshape", shape],
                     ["transpose", perm], ["flip", axis].
                     (RecurrentGemma's conv1d is stored (D, 1, L) in
                     cross-correlation order; ours is (L, D) with
                     w[0] as the newest tap -- a true convolution --
                     so the converter reshapes, transposes AND flips.)

Structural lists (layer lists, split branches) are distinguished
from inline tensors by content: a list whose elements bottom out in
numbers is a tensor; anything containing dicts/nulls is structure.
Real weight pytrees never produce an ambiguous case (structural list
elements are dicts, lists or null, never bare numbers).

Tensors are cast to the model's precision at load time, so a bf16
checkpoint can be loaded into an fp32 model directly.
"""
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
from safetensors import safe_open

from . import pjson
from .precision import Precision
from .spec_parser import parse_model2


def save_model(path: str, model_def, weights) -> None:
    """Write a model (spec + precision + inline weights) as JSON."""
    doc = {
        'spec': model_def.spec,
        'precision': str(model_def.precision),
        'weights': _serialize(weights),
    }
    with open(path, 'w', encoding='utf-8') as f:
        pjson.save_json(doc, f)


def load_model(
    path: str,
    precision: Precision | None = None,
    check_structure: bool = False,
):
    """Read a model JSON; returns (Model2Def, weights pytree).

    `precision` overrides the stored one (tensors are cast either
    way). `check_structure=True` compares the loaded pytree's
    structure and leaf shapes against a freshly initialized model --
    cheap for search-sized models, expensive for billion-parameter
    ones (it allocates a full init), so it is opt-in.
    """
    with open(path, encoding='utf-8') as f:
        doc = json.load(f)
    if precision is None:
        precision = Precision(doc['precision'])
    md = parse_model2(doc['spec'], precision)
    base = os.path.dirname(os.path.abspath(path))
    files: dict[str, object] = {}
    weights = _deserialize(
        doc['weights'], base, precision.jax_dtype, files)
    if check_structure:
        _check_structure(md, weights)
    return md, weights


def _serialize(node):
    if node is None:
        return None
    if isinstance(node, dict):
        return {k: _serialize(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_serialize(v) for v in node]
    return np.asarray(node).tolist()


def _is_numeric(node) -> bool:
    if isinstance(node, (int, float)):
        return True
    if isinstance(node, list):
        return bool(node) and all(_is_numeric(v) for v in node)
    return False


def _deserialize(node, base: str, dtype, files: dict):
    if node is None:
        return None
    if isinstance(node, dict):
        if 'path' in node and 'id' in node:
            return _load_tensor(node, base, files).astype(dtype)
        return {k: _deserialize(v, base, dtype, files)
                for k, v in node.items()}
    if isinstance(node, (int, float)) or _is_numeric(node):
        return jnp.asarray(node, dtype=dtype)
    return [_deserialize(v, base, dtype, files) for v in node]


def _load_tensor(desc: dict, base: str, files: dict):
    p = desc['path']
    if not os.path.isabs(p):
        candidate = os.path.join(base, p)
        p = candidate if os.path.exists(candidate) else p
    p = os.path.abspath(p)
    if p not in files:
        files[p] = safe_open(p, framework='flax')
    t = files[p].get_tensor(desc['id'])
    for op, arg in desc.get('transform', ()):
        if op == 'reshape':
            t = t.reshape(arg)
        elif op == 'transpose':
            t = t.transpose(arg)
        elif op == 'flip':
            t = jnp.flip(t, axis=arg)
        else:
            raise ValueError(f'unknown transform op: {op!r}')
    return t


def _check_structure(md, weights) -> None:
    ref = md.build_jax().init_weights(jax.random.PRNGKey(0))
    got_leaves, got_def = jax.tree.flatten(weights)
    ref_leaves, ref_def = jax.tree.flatten(ref)
    if got_def != ref_def:
        raise ValueError(
            f'weights structure mismatch for {md.spec!r}:\n'
            f'  file:  {got_def}\n  model: {ref_def}')
    for i, (got, want) in enumerate(zip(got_leaves, ref_leaves)):
        if got.shape != want.shape:
            raise ValueError(
                f'leaf {i} shape mismatch for {md.spec!r}: '
                f'file {got.shape}, model {want.shape}')
