"""Model2 tests: parse + forward/step parity with the existing Model.

Plain-sequence specs only (no skip, no split yet). The main
correctness criterion is that Model2 produces the same logits as
ModelDef on the same spec when given the same weights -- the
recursive wrapper around the layer list shouldn't change anything.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from texmo.model import ModelDef
from texmo.model2 import Model2Def
from texmo.precision import Precision

_1_BY_LOG2 = 1.0 / math.log(2.0)


def _build2(spec, precision=Precision.FP32, seed=42):
    md = Model2Def(spec, precision)
    model = md.build_jax()
    weights = model.init_weights(jax.random.PRNGKey(seed))
    return md, model, weights


# --- LayerSeqDef pass-through wraps the layer chain correctly --------


def test_def_properties():
    md = Model2Def("bytes|dense.32.gelu-dense.16.tanh", Precision.FP32)
    assert md.input.size == 256
    assert md.layer_seq.input_size == 256
    assert md.layer_seq.size == 16
    assert md.layer_seq.length == 1
    assert md.total_padding == 1
    assert md.output.input_size == 16


def test_def_num_weights_matches_model():
    """Model2's weight count should match ModelDef's on the same spec."""
    spec = "bits.1+bp|dense.8.gelu-dense.4.tanh"
    md = ModelDef(spec, Precision.FP32)
    md2 = Model2Def(spec, Precision.FP32)
    assert md.num_weights == md2.num_weights


def test_def_num_mults_matches_model():
    spec = "bits.1+bp|gru.8-dense.4.tanh"
    md = ModelDef(spec, Precision.FP32)
    md2 = Model2Def(spec, Precision.FP32)
    assert md.num_mults == md2.num_mults


def test_def_length_aware_total_padding():
    """suffix.4 contributes 3 to total_padding (length-1)."""
    md = Model2Def(
        "bits.1+bp|dense.4.gelu-suffix.4-dense.4.tanh", Precision.FP32)
    assert md.layer_seq.length == 4  # 1 + (1-1) + (4-1) + (1-1)
    assert md.total_padding == 4


def test_def_no_layers():
    """Empty layer chain still produces a valid model spec."""
    md = Model2Def("bits.1+bp|", Precision.FP32)
    assert md.layer_seq.layers == []
    assert md.layer_seq.size == md.input.size
    assert md.layer_seq.length == 1
    assert md.total_padding == 1


def test_def_rejects_skip_for_now():
    """Model2 won't handle skip syntax until the parser learns to
    translate it into split.op(...) form."""
    with pytest.raises(NotImplementedError):
        Model2Def(
            "bytes|skip.1.add-dense.32.gelu-dense.32.gelu",
            Precision.FP32)


def test_seq_in_seq_invalid():
    """A LayerSeqDef directly containing another LayerSeqDef would
    just be a longer flat sequence -- is_valid rejects it. Splits
    (when they exist) are the only legitimate way to nest one
    LayerSeqDef inside another layer."""
    from texmo.layers.dense import DenseDef
    from texmo.layers.seq import LayerSeqDef
    inner = LayerSeqDef(
        [DenseDef(4, input_size=8, activation='tanh')],
        input_size=8)
    outer = LayerSeqDef([inner], input_size=8)
    assert not outer.is_valid()


# --- Forward / step shape sanity --------------------------------------


def test_forward_shape():
    _, model, weights = _build2("bits.1+bp|dense.8.gelu")
    batch = jax.random.randint(jax.random.PRNGKey(0), (4, 32), 0, 2)
    logits = model.forward(weights, batch)
    assert logits.shape == (4, 32, 2)


def test_forward_shape_bytes():
    _, model, weights = _build2("bytes|dense.64.gelu")
    batch = jax.random.randint(jax.random.PRNGKey(0), (4, 16), 0, 256)
    logits = model.forward(weights, batch)
    assert logits.shape == (4, 16, 256)


def test_loss_batch():
    _, model, weights = _build2("bytes|dense.64.gelu")
    batch = jax.random.randint(jax.random.PRNGKey(0), (4, 16), 0, 256)
    loss = model.loss_batch(weights, batch)
    assert loss.shape == ()
    assert loss > 0
    # Random model on 256 tokens: loss should land near log2(256) = 8.
    assert loss < 12


def test_initial_step_and_step_shapes():
    _, model, weights = _build2("bits.1+bp|dense.8.gelu")
    states, logits0 = model.initial_step(weights)
    assert logits0.shape == (2,)
    # input_state + layer_seq_state
    assert len(states) == 2

    states, logits1 = model.step(weights, states, 0)
    assert logits1.shape == (2,)


def test_no_layers_runs():
    _, model, weights = _build2("bits.1+bp|")
    batch = jax.random.randint(jax.random.PRNGKey(0), (4, 16), 0, 2)
    logits = model.forward(weights, batch)
    assert logits.shape == (4, 16, 2)


# --- step() and forward() agree -------------------------------------


def test_step_matches_forward():
    _, model, weights = _build2("bits.1+bp|dense.8.gelu")
    tokens = [0, 1, 1, 0, 1, 0, 0, 1]
    batch = jnp.array([tokens], dtype=jnp.int32)
    fwd_logits = model.forward(weights, batch)

    states, logits0 = model.initial_step(weights)
    step_logits = [logits0]
    for t in tokens[:-1]:
        states, logits_t = model.step(weights, states, t)
        step_logits.append(logits_t)

    for i in range(len(tokens)):
        np.testing.assert_allclose(
            fwd_logits[0, i], step_logits[i], atol=1e-5)


def test_step_matches_forward_multi_layer():
    _, model, weights = _build2("bits.1+bp|dense.8.gelu-gru.4")
    tokens = [0, 1, 1, 0, 1, 0]
    batch = jnp.array([tokens], dtype=jnp.int32)
    fwd_logits = model.forward(weights, batch)

    states, logits0 = model.initial_step(weights)
    step_logits = [logits0]
    for t in tokens[:-1]:
        states, logits_t = model.step(weights, states, t)
        step_logits.append(logits_t)

    for i in range(len(tokens)):
        np.testing.assert_allclose(
            fwd_logits[0, i], step_logits[i], atol=1e-5)


def test_step_matches_forward_with_suffix():
    """Verify total_padding accounting works through LayerSeq."""
    _, model, weights = _build2(
        "bits.1+bp|dense.4.gelu-suffix.2-dense.4.tanh")
    tokens = [0, 1, 1, 0, 1, 0]
    batch = jnp.array([tokens], dtype=jnp.int32)
    fwd_logits = model.forward(weights, batch)

    states, logits0 = model.initial_step(weights)
    step_logits = [logits0]
    for t in tokens[:-1]:
        states, logits_t = model.step(weights, states, t)
        step_logits.append(logits_t)

    for i in range(len(tokens)):
        np.testing.assert_allclose(
            fwd_logits[0, i], step_logits[i], atol=1e-5)


# --- Training reduces the loss ---------------------------------------


def test_loss_decreases():
    """A few optimizer steps should bring the loss down -- the
    nested layer weights actually receive gradients."""
    import optax
    _, model, weights = _build2("bits.1+bp|dense.8.gelu")
    batch = jax.random.randint(jax.random.PRNGKey(0), (8, 32), 0, 2)

    loss_fn = jax.jit(model.loss_batch)
    grad_fn = jax.jit(jax.grad(model.loss_batch))

    initial_loss = float(loss_fn(weights, batch))

    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(weights)
    for _ in range(20):
        grads = grad_fn(weights, batch)
        updates, opt_state = optimizer.update(grads, opt_state, weights)
        weights = optax.apply_updates(weights, updates)

    final_loss = float(loss_fn(weights, batch))
    assert final_loss < initial_loss


# --- forward_recurrent matches forward ------------------------------


def test_forward_recurrent_matches_forward():
    _, model, weights = _build2("bits.1+bp|dense.4.gelu-gru.4")
    batch = jax.random.randint(
        jax.random.PRNGKey(3), (2, 6), 0, 2).astype(jnp.int32)
    par = model.forward(weights, batch)
    rec = model.forward_recurrent(weights, batch)
    np.testing.assert_allclose(np.asarray(par), np.asarray(rec), atol=1e-5)
