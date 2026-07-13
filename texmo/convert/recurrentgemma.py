"""Generate a texmo model manifest for a RecurrentGemma checkpoint.

Reads the HF checkpoint's config.json + safetensors index, builds the
texmo spec for the Griffin stack, and writes a model_store manifest
whose descriptors point back into the original shards -- no tensor is
copied. The mapping (validated per-layer by the HF numeric
cross-check tests):

    embed_tokens.weight            -> codec {emb};  y = ln(sqrt(H)) inline
    layer temporal (recurrent):       split.add(rmsnorm-split.mul(
        linear_x -> conv_1d -> rg_lru,   dense.H-conv.C-rglru.K,
        gelu(linear_y)                   dense.H.gelu)
      ) * ... -> linear_out            -dense.H, pass)
    layer temporal (attention):       split.add(rmsnorm-attn.H.K.W, pass)
    layer mlp (GeGLU):                split.add(rmsnorm-split.mul(
        gelu(gate_proj) * up_proj        dense.G.gelu, dense.G)
        -> down_proj                     -dense.H, pass)
    final_norm.weight              -> trailing rmsnorm

Direct 1:1 tensors everywhere except conv_1d, stored (D, 1, L)
upstream vs our (L, D) -- handled by a descriptor transform. The
RG-LRU gates share our exact (blocks, in, out) baddbmm orientation.

The generated spec is load-only (non-power-of-2 widths keep it out
of the search) but parses, builds and runs like any model.

After writing the manifest the script verifies every leaf against
`jax.eval_shape` of the model's init -- full structural and shape
validation without materializing a single tensor.

Usage:
    uv run python -m texmo.convert.recurrentgemma \
        --checkpoint gemma/recurrentgemma-2b \
        --out models/recurrentgemma-2b.json
"""
import argparse
import json
import math
import os

import jax
import numpy as np

from .. import pjson
from ..precision import Precision
from ..spec_parser import parse_model2


def build_spec(cfg: dict) -> str:
    h = cfg['hidden_size']
    g = cfg['intermediate_size'] // 2  # gate/up width of the GeGLU
    heads = cfg['num_attention_heads']
    window = cfg['attention_window_size']
    conv = cfg['conv1d_width']
    vocab = cfg['vocab_size']

    recurrent = (f"rmsnorm-split.mul(dense.{h}-conv.{conv}-rglru.{heads}, "
                 f"dense.{h}.gelu)-dense.{h}")
    attention = f"rmsnorm-attn.{h}.{heads}.{window}"
    mlp = f"rmsnorm-split.mul(dense.{g}.gelu, dense.{g})-dense.{h}"

    types = cfg['_block_types']
    blocks = []
    for i in range(cfg['num_hidden_layers']):
        temporal = (recurrent if types[i % len(types)] == 'recurrent'
                    else attention)
        blocks.append(
            f"split.add({temporal}, pass)-split.add({mlp}, pass)")
    chain = "-".join(blocks) + "-rmsnorm"
    return f"tokens.{vocab}.gemma.emb.{h}|{chain}"


class _Mapper:
    """Descriptor factory: HF tensor name -> {path, id[, transform]}."""

    def __init__(self, ckpt_dir: str, out_dir: str, weight_map: dict):
        self._weight_map = weight_map
        # Manifest-relative shard paths, forward slashes for
        # portability (the store resolves them against the manifest's
        # directory).
        self._rel = {
            shard: os.path.relpath(
                os.path.join(ckpt_dir, shard), out_dir).replace('\\', '/')
            for shard in set(weight_map.values())
        }
        self.used: set[str] = set()

    def desc(self, name: str, transform=None) -> dict:
        self.used.add(name)
        d = {'path': self._rel[self._weight_map[name]], 'id': name}
        if transform:
            d['transform'] = transform
        return d


def build_weights(cfg: dict, mapper: _Mapper) -> list:
    h = cfg['hidden_size']
    conv = cfg['conv1d_width']
    types = cfg['_block_types']

    def dense(prefix: str) -> dict:
        return {'w': mapper.desc(f'{prefix}.weight'),
                'b': mapper.desc(f'{prefix}.bias')}

    def rmsnorm(name: str) -> dict:
        return {'gamma': mapper.desc(f'{name}.weight')}

    layers = []
    for i in range(cfg['num_hidden_layers']):
        p = f'model.layers.{i}'
        t = f'{p}.temporal_block'
        if types[i % len(types)] == 'recurrent':
            temporal = [
                rmsnorm(f'{p}.temporal_pre_norm'),
                # split.mul(dense-conv-rglru, dense.gelu)
                [
                    [
                        dense(f'{t}.linear_x'),
                        # HF stores (D, 1, L) in cross-correlation
                        # order (k: old -> new); our conv is (L, D)
                        # with w[0] as the NEWEST tap, hence the flip.
                        {'w': mapper.desc(
                            f'{t}.conv_1d.weight',
                            transform=[['reshape', [h, conv]],
                                       ['transpose', [1, 0]],
                                       ['flip', 0]]),
                         'b': mapper.desc(f'{t}.conv_1d.bias')},
                        {'lam': mapper.desc(f'{t}.rg_lru.recurrent_param'),
                         'w_ig': mapper.desc(
                             f'{t}.rg_lru.input_gate_weight'),
                         'b_ig': mapper.desc(
                             f'{t}.rg_lru.input_gate_bias'),
                         'w_rg': mapper.desc(
                             f'{t}.rg_lru.recurrent_gate_weight'),
                         'b_rg': mapper.desc(
                             f'{t}.rg_lru.recurrent_gate_bias')},
                    ],
                    [dense(f'{t}.linear_y')],
                ],
                dense(f'{t}.linear_out'),
            ]
        else:
            temporal = [
                rmsnorm(f'{p}.temporal_pre_norm'),
                {'w_q': mapper.desc(f'{t}.q_proj.weight'),
                 'w_k': mapper.desc(f'{t}.k_proj.weight'),
                 'w_v': mapper.desc(f'{t}.v_proj.weight'),
                 'w_o': mapper.desc(f'{t}.o_proj.weight'),
                 'b_o': mapper.desc(f'{t}.o_proj.bias')},
            ]
        mlp = [
            rmsnorm(f'{p}.channel_pre_norm'),
            # split.mul(dense.G.gelu, dense.G): gate_proj feeds the
            # activated branch, up_proj the bare one.
            [
                [dense(f'{p}.mlp_block.gate_proj')],
                [dense(f'{p}.mlp_block.up_proj')],
            ],
            dense(f'{p}.mlp_block.down_proj'),
        ]
        # Two residual wrappers per HF layer: temporal, then mlp.
        layers.append([temporal, []])
        layers.append([mlp, []])
    layers.append(rmsnorm('model.final_norm'))

    codec_input = {
        'emb': mapper.desc('model.embed_tokens.weight'),
        # Derived, not stored upstream: the tied codec's input scale,
        # exp(y) = sqrt(H) (embeddings_scale_by_sqrt_dim).
        'y': 0.5 * math.log(h),
    }
    return [codec_input, layers, None]


def _leaf_shape(node, ckpt_dir: str, shapes: dict) -> tuple:
    """Shape of a manifest leaf without loading it."""
    if isinstance(node, dict) and 'path' in node:
        shape = list(shapes[node['id']])
        for op, arg in node.get('transform', ()):
            if op == 'reshape':
                shape = list(arg)
            elif op == 'transpose':
                shape = [shape[a] for a in arg]
            # 'flip' preserves the shape.
        return tuple(shape)
    return np.shape(node)


def _verify(doc: dict, ckpt_dir: str, shapes: dict) -> int:
    """Walk the manifest against jax.eval_shape of the model's init.

    Raises on any structure or shape mismatch; returns the leaf
    count. No tensor is materialized.
    """
    md = parse_model2(doc['spec'], Precision(doc['precision']))
    expected = jax.eval_shape(
        md.build_jax().init_weights, jax.random.PRNGKey(0))

    count = 0

    def walk(got, want, path):
        nonlocal count
        if want is None:
            assert got is None, f'{path}: expected null, got {type(got)}'
            return
        if isinstance(want, dict):
            assert isinstance(got, dict) and set(got) == set(want), (
                f'{path}: keys {sorted(got)} != {sorted(want)}')
            for k in want:
                walk(got[k], want[k], f'{path}.{k}')
            return
        if isinstance(want, list):
            assert isinstance(got, list) and len(got) == len(want), (
                f'{path}: length {len(got)} != {len(want)}')
            for j, (g, w) in enumerate(zip(got, want)):
                walk(g, w, f'{path}[{j}]')
            return
        # want is a ShapeDtypeStruct leaf.
        got_shape = _leaf_shape(got, ckpt_dir, shapes)
        assert got_shape == want.shape, (
            f'{path}: shape {got_shape} != {want.shape}')
        count += 1

    # The expected tree's dicts/lists nest exactly like the manifest;
    # leaves in `expected` are ShapeDtypeStructs (or None slots).
    walk(doc['weights'], expected, 'weights')
    return count


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='gemma/recurrentgemma-2b')
    p.add_argument('--out', default='models/recurrentgemma-2b.json')
    args = p.parse_args()

    with open(os.path.join(args.checkpoint, 'config.json')) as f:
        cfg = json.load(f)
    with open(os.path.join(
            args.checkpoint, 'model.safetensors.index.json')) as f:
        index = json.load(f)
    weight_map = index['weight_map']

    out_dir = os.path.dirname(os.path.abspath(args.out)) or '.'
    os.makedirs(out_dir, exist_ok=True)

    spec = build_spec(cfg)
    mapper = _Mapper(args.checkpoint, out_dir, weight_map)
    doc = {
        'spec': spec,
        'precision': 'bf16',
        'weights': build_weights(cfg, mapper),
    }

    unused = set(weight_map) - mapper.used
    assert not unused, f'unmapped checkpoint tensors: {sorted(unused)[:8]}'

    # Shapes for verification, from the shard headers (no data reads).
    from safetensors import safe_open
    shapes = {}
    for shard in set(weight_map.values()):
        with safe_open(os.path.join(args.checkpoint, shard),
                       framework='flax') as f:
            for name in f.keys():
                shapes[name] = tuple(f.get_slice(name).get_shape())
    n = _verify(doc, args.checkpoint, shapes)

    with open(args.out, 'w', encoding='utf-8') as f:
        pjson.save_json(doc, f)

    md = parse_model2(spec, Precision.BF16)
    print(f'wrote {args.out}')
    print(f'  tensors mapped: {len(mapper.used)} '
          f'(verified {n} leaves against eval_shape)')
    print(f'  num_weights:    {md.num_weights:,}')
    print(f'  spec length:    {len(spec)} chars')


if __name__ == '__main__':
    main()
