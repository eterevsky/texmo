import logging
import os
from unittest import TestCase

import numpy as np
import jax.numpy as jnp

from texmo.model3 import Model3, build_model
from texmo.prng import Rng
from texmo.tokens import set_tokens_dir

set_tokens_dir(os.path.join(os.path.dirname(__file__), "../tokens"))

def _step_and_forward(model: Model3, input: list[int]):
    rng = Rng(seed=0)
    weights = model.init_weights(rng)
    state, out = model.initial_step(weights, dtype=jnp.float32)
    outs = [out]

    for token in input[:-1]:
        state, out = model.step(weights, state, token, dtype=jnp.float32)
        outs.append(out)

    step_outs = jnp.stack(outs)

    batch = jnp.array(input).reshape(1, -1)
    batch_outs = model._forward_batch(weights, batch, dtype=jnp.float32)

    np.testing.assert_array_equal(
        step_outs, batch_outs.reshape(step_outs.shape)
    )

def test_build_model():
    model = build_model("tokens.128.cw.bh-pos.16-emb.64.norm|gru.64-dense.128.relu")
    assert str(model) == "tokens.128.cw.bh-pos.16-emb.64.norm|gru.64-dense.128.relu"

def test_neighbors():
    model = build_model("tokens.128.cw.bh-pos.16-emb.64.norm|gru.64-dense.64.relu")
    neighbors = set(map(str, model.neighbors()))
    assert neighbors == {
            "tokens.128.cw.b2-pos.16-emb.64.norm|gru.64-dense.64.relu",
            "tokens.128.cw.bh-pos.16-emb.128.norm|gru.64-dense.64.relu",
            "tokens.128.cw.bh-pos.16-emb.32.norm|gru.64-dense.64.relu",
            "tokens.128.cw.bh-pos.16-emb.64.norm|dense.64.gelu-dense.64.relu",
            "tokens.128.cw.bh-pos.16-emb.64.norm|dense.64.relu-dense.64.relu",
            "tokens.128.cw.bh-pos.16-emb.64.norm|dense.64.tanh-dense.64.relu",
            "tokens.128.cw.bh-pos.16-emb.64.norm|gru.128-dense.64.relu",
            "tokens.128.cw.bh-pos.16-emb.64.norm|gru.32-dense.64.relu",
            "tokens.128.cw.bh-pos.16-emb.64.norm|gru.64-dense.128.relu",
            "tokens.128.cw.bh-pos.16-emb.64.norm|gru.64-dense.32.relu",
            "tokens.128.cw.bh-pos.16-emb.64.norm|gru.64-dense.64.gelu",
            "tokens.128.cw.bh-pos.16-emb.64.norm|gru.64-dense.64.relu-dense.64.gelu",
            "tokens.128.cw.bh-pos.16-emb.64.norm|gru.64-dense.64.relu-dense.64.tanh",
            "tokens.128.cw.bh-pos.16-emb.64.norm|gru.64-dense.64.relu-gru.64",
            "tokens.128.cw.bh-pos.16-emb.64.norm|gru.64-dense.64.relu-lstm.64",
            "tokens.128.cw.bh-pos.16-emb.64.norm|gru.64-dense.64.relu-mgru.64",
            "tokens.128.cw.bh-pos.16-emb.64.norm|gru.64-dense.64.relu-mingru.64",
            "tokens.128.cw.bh-pos.16-emb.64.norm|gru.64-dense.64.relu-rec.64.gelu",
            "tokens.128.cw.bh-pos.16-emb.64.norm|gru.64-dense.64.relu-rec.64.tanh",
            "tokens.128.cw.bh-pos.16-emb.64.norm|gru.64-dense.64.relu-suffix.2",
            "tokens.128.cw.bh-pos.16-emb.64.norm|gru.64-dense.64.tanh",
            "tokens.128.cw.bh-pos.16-emb.64.norm|gru.64-gru.64",
            "tokens.128.cw.bh-pos.16-emb.64.norm|gru.64-lstm.64",
            "tokens.128.cw.bh-pos.16-emb.64.norm|gru.64-mgru.64",
            "tokens.128.cw.bh-pos.16-emb.64.norm|gru.64-mingru.64",
            "tokens.128.cw.bh-pos.16-emb.64.norm|gru.64-rec.64.gelu",
            "tokens.128.cw.bh-pos.16-emb.64.norm|gru.64-rec.64.relu",
            "tokens.128.cw.bh-pos.16-emb.64.norm|gru.64-rec.64.tanh",
            "tokens.128.cw.bh-pos.16-emb.64.norm|gru.64-suffix.2-dense.64.relu",
            "tokens.128.cw.bh-pos.16-emb.64.norm|gru.64",
            "tokens.128.cw.bh-pos.16-emb.64.norm|lstm.64-dense.64.relu",
            "tokens.128.cw.bh-pos.16-emb.64.norm|mgru.64-dense.64.relu",
            "tokens.128.cw.bh-pos.16-emb.64.norm|mingru.64-dense.64.relu",
            "tokens.128.cw.bh-pos.16-emb.64.norm|rec.64.gelu-dense.64.relu",
            "tokens.128.cw.bh-pos.16-emb.64.norm|rec.64.relu-dense.64.relu",
            "tokens.128.cw.bh-pos.16-emb.64.norm|rec.64.tanh-dense.64.relu",
            "tokens.128.cw.bh-pos.16-emb.64.norm|suffix.2-gru.64-dense.64.relu",
            "tokens.128.cw.bh-pos.16-emb.64|gru.64-dense.64.relu",
            "tokens.128.cw.bh-pos.32-emb.64.norm|gru.64-dense.64.relu",
            "tokens.128.cw.bh-pos.8-emb.64.norm|gru.64-dense.64.relu",
            'tokens.128.cw.bh-pos.16-emb.64.norm|gru.64-norm-dense.64.relu',
            'tokens.128.cw.bh-pos.16-emb.64.norm|gru.64-dense.64.relu-norm',
            "tokens.128.raw.bh-pos.16-emb.64.norm|gru.64-dense.64.relu",
            "tokens.256.cw.bh-pos.16-emb.64.norm|gru.64-dense.64.relu",
            "tokens.64.cw.bh-pos.16-emb.64.norm|gru.64-dense.64.relu",
        }

def test_input():
    model = build_model("tokens.4.cw.bh|")
    _step_and_forward(model, [0, 1, 2, 3])


def test_bits1_neighbors():
    model = build_model('bits.1|')
    neighbors = set(map(str, model.neighbors()))
    assert neighbors == {
        'bits.1+bp|',
        'bits.1+pos|',
        'bits.2|',
        'bits.2.oh|',
        'bits.1|suffix.2',
        'bits.1|dense.1.tanh',
        'bits.1|dense.1.gelu',
        'bits.1|rec.1.tanh',
        'bits.1|rec.1.gelu',
        'bits.1|gru.1',
        'bits.1|mgru.1',
        'bits.1|mingru.1',
        'bits.1|lstm.1',
    }

def test_bits1bp_neighbors():
    model = build_model('bits.1+bp|')
    neighbors = set(map(str, model.neighbors()))
    assert neighbors == {
        'bits.1|',
        'bits.1+pos|',
        'bits.2+bp|',
        'bits.2.oh+bp|',
        'bits.1+bp|suffix.2',
        'bits.1+bp|dense.1.tanh',
        'bits.1+bp|dense.1.gelu',
        'bits.1+bp|rec.1.tanh',
        'bits.1+bp|rec.1.gelu',
        'bits.1+bp|gru.1',
        'bits.1+bp|mgru.1',
        'bits.1+bp|mingru.1',
        'bits.1+bp|lstm.1',
    }


def test_bits4bp_neighbors():
    model = build_model('bits.4+bp|')
    neighbors = set(map(str, model.neighbors()))
    assert neighbors == {
        'bits.2+bp|',
        'bits.2+pos|',
        'bits.4|',
        'bits.4.oh+bp|',
        'bits.8|',
        'bits.4+bp|suffix.2',
        'bits.4+bp|dense.4.tanh',
        'bits.4+bp|dense.4.gelu',
        'bits.4+bp|rec.4.tanh',
        'bits.4+bp|rec.4.gelu',
        'bits.4+bp|gru.4',
        'bits.4+bp|mgru.4',
        'bits.4+bp|mingru.4',
        'bits.4+bp|lstm.4',
    }

def test_suffix_neighbors():
    model = build_model('bits.4+bp|rec.16.tanh-suffix.2-dense.16.tanh')
    neighbors = set(map(str, model.neighbors()))
    assert 'bits.4+bp|rec.16.tanh-dense.16.tanh' in neighbors
    assert 'bits.4+bp|rec.16.tanh-suffix.4-dense.16.tanh' in neighbors
    assert 'bits.4+bp|rec.16.tanh-suffix.2-suffix.2-dense.16.tanh' not in neighbors
