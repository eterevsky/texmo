import math

import jax
import jax.numpy as jnp
import numpy as np

from texmo.model import ModelDef
from texmo.precision import Precision

_1_BY_LOG2 = 1.0 / math.log(2.0)


def _build(spec, precision=Precision.FP32):
    md = ModelDef(spec, precision)
    model = md.build_jax()
    rng = jax.random.PRNGKey(42)
    weights = model.init_weights(rng)
    return model, weights


def test_forward_shape():
    model, weights = _build("bits.1+bp|dense.8.gelu")
    batch = jax.random.randint(jax.random.PRNGKey(0), (4, 32), 0, 2)
    logits = model.forward(weights, batch)
    assert logits.shape == (4, 32, 2)


def test_forward_shape_bytes():
    model, weights = _build("bytes|dense.64.gelu")
    batch = jax.random.randint(jax.random.PRNGKey(0), (4, 16), 0, 256)
    logits = model.forward(weights, batch)
    assert logits.shape == (4, 16, 256)


def test_loss_batch():
    model, weights = _build("bytes|dense.64.gelu")
    batch = jax.random.randint(jax.random.PRNGKey(0), (4, 16), 0, 256)
    loss = model.loss_batch(weights, batch)
    assert loss.shape == ()
    assert loss > 0
    # Random model on 256 tokens: loss near log2(256) = 8
    assert loss < 12


def test_loss_decreases():
    """A few optimization steps should reduce the loss."""
    import optax
    model, weights = _build("bits.1+bp|dense.8.gelu")
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


def test_initial_step_and_step():
    model, weights = _build("bits.1+bp|dense.8.gelu")
    states, logits0 = model.initial_step(weights)
    assert logits0.shape == (2,)
    assert len(states) == 2  # input state + dense layer

    states, logits1 = model.step(weights, states, 0)
    assert logits1.shape == (2,)


def test_step_matches_forward():
    """step() should produce the same logits as forward()."""
    model, weights = _build("bits.1+bp|dense.8.gelu")

    tokens = [0, 1, 1, 0, 1, 0, 0, 1]
    batch = jnp.array([tokens], dtype=jnp.int32)
    fwd_logits = model.forward(weights, batch)  # (1, 8, 2)

    states, step_logits_0 = model.initial_step(weights)
    step_logits = [step_logits_0]
    for t in tokens[:-1]:
        states, logits_t = model.step(weights, states, t)
        step_logits.append(logits_t)

    for i in range(len(tokens)):
        np.testing.assert_allclose(
            fwd_logits[0, i], step_logits[i], atol=1e-5)


def test_multi_layer():
    model, weights = _build("bits.1+bp|dense.8.gelu-dense.4.tanh")
    batch = jax.random.randint(jax.random.PRNGKey(0), (2, 16), 0, 2)
    logits = model.forward(weights, batch)
    assert logits.shape == (2, 16, 2)


def test_multi_layer_step_matches_forward():
    model, weights = _build("bits.1+bp|dense.8.gelu-dense.4.tanh")

    tokens = [0, 1, 1, 0]
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


def test_no_layers():
    """Model with no hidden layers (just input -> output)."""
    model, weights = _build("bits.1+bp|")
    batch = jax.random.randint(jax.random.PRNGKey(0), (4, 16), 0, 2)
    logits = model.forward(weights, batch)
    assert logits.shape == (4, 16, 2)

    states, logits0 = model.initial_step(weights)
    assert logits0.shape == (2,)


def test_weight_count():
    md = ModelDef("bits.1+bp|dense.8.gelu", Precision.FP32)
    model = md.build_jax()
    weights = model.init_weights(jax.random.PRNGKey(0))
    actual = sum(w.size for w in jax.tree.leaves(weights))
    assert actual == md.num_weights


def test_dtype_fp64():
    model, weights = _build("bits.1+bp|dense.8.gelu", Precision.FP64)
    for w in jax.tree.leaves(weights):
        assert w.dtype == jnp.float64

    batch = jax.random.randint(jax.random.PRNGKey(0), (2, 8), 0, 2)
    logits = model.forward(weights, batch)
    assert logits.dtype == jnp.float64
