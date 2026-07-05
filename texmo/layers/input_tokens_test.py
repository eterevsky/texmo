"""Tests for the tokenized input layer (tokens.N.variation.oh|emb.S)."""
import jax
import jax.numpy as jnp
import numpy as np

from texmo.layers.input_tokens import TokensInputDef, TokensInputJax
from texmo.precision import Precision
from texmo.spec_parser import parse_model2


def test_spec_roundtrip():
    d = TokensInputDef.from_spec("tokens.256000.gemma.emb.2560")
    assert (d.ntokens, d.variation, d.emb_size) == (256000, "gemma", 2560)
    assert d.size == 2560
    assert d.tokens_name == "tokens256000_gemma"
    assert d.num_weights == 256000 * 2560
    assert str(d) == "tokens.256000.gemma.emb.2560"

    oh = TokensInputDef.from_spec("tokens.16.test.oh")
    assert (oh.ntokens, oh.emb_size, oh.size) == (16, None, 16)
    assert oh.num_weights == 0
    assert str(oh) == "tokens.16.test.oh"


def test_validity_and_neighbors():
    assert TokensInputDef.from_spec("tokens.16.test.oh").is_valid()
    assert TokensInputDef.from_spec("tokens.256000.gemma.emb.64").is_valid()
    # emb width follows the power-of-2 rule: the real Gemma width is
    # load-only, like rglru.10 / attn.2560.10.2048.
    assert not TokensInputDef.from_spec(
        "tokens.256000.gemma.emb.2560").is_valid()
    d = TokensInputDef.from_spec("tokens.16.test.emb.8")
    assert set(d.neighbors()) == {
        "tokens.16.test.emb.4", "tokens.16.test.emb.16"}
    assert TokensInputDef.from_spec("tokens.16.test.oh").neighbors() == ()


def _emb_layer(ntokens=16, size=8):
    layer = TokensInputJax(ntokens, size, emb=True, dtype=jnp.float32)
    return layer, layer.init_weights(jax.random.PRNGKey(0))


def test_emb_forward_and_padding():
    layer, w = _emb_layer()
    ids = jnp.array([[1, 2, 3], [4, 5, 6]])
    out = layer.forward(w, ids, padding=2)
    assert out.shape == (2, 5, 8)
    # Looked-up rows match the table; pad positions are the mean row.
    assert np.allclose(out[0, 2], w['emb'][1])
    assert np.allclose(out[1, 4], w['emb'][6])
    assert np.allclose(out[0, 0], np.mean(np.asarray(w['emb']), axis=0),
                       atol=1e-6)


def test_oh_forward_and_padding():
    layer = TokensInputJax(16, 16, emb=False, dtype=jnp.float32)
    ids = jnp.array([[3, 0]])
    out = layer.forward(None, ids, padding=1)
    assert out.shape == (1, 3, 16)
    assert np.allclose(out[0, 0], np.full(16, 1 / 16))  # uniform pad
    assert out[0, 1, 3] == 1.0 and out[0, 1].sum() == 1.0


def test_model2_end_to_end_with_embedding():
    md = parse_model2("tokens.16.test.emb.8|dense.8.gelu", Precision.FP32)
    assert md.ntokens == 16
    assert md.num_weights == 16 * 8 + (8 * 8 + 8) + (16 * 8 + 16)
    model = md.build_jax()
    weights = model.init_weights(jax.random.PRNGKey(0))
    assert weights[0]['emb'].shape == (16, 8)  # slot 0 now carries the table

    batch = jax.random.randint(
        jax.random.PRNGKey(1), (2, 12), 0, 16).astype(jnp.int32)
    logits = model.forward(weights, batch)
    assert logits.shape == (2, 12, 16)
    loss = float(model.loss_batch(weights, batch))
    assert np.isfinite(loss) and loss > 0

    # Gradients flow into the embedding table.
    grads = jax.grad(model.loss_batch)(weights, batch)
    assert float(jnp.abs(grads[0]['emb']).sum()) > 0

    # The scanned step path agrees with the parallel forward.
    rec = model.forward_recurrent(weights, batch)
    assert np.allclose(np.asarray(rec), np.asarray(logits), atol=1e-4)
