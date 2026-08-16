"""Tests for the sliding-window MQA layer, including a numeric
cross-check against a numpy reimplementation of the HuggingFace
RecurrentGemmaAttention math (MQA, split-half partial RoPE, banded
causal mask, fp32 softmax, head_dim**-0.5 scaling)."""
import jax
import jax.numpy as jnp
import numpy as np

from texmo.layer import _ATTN_SWAP_WINDOW
from texmo.layers.attn import (
    AttnDef, AttnJax, _ROPE_BASE, _ROTARY_FRACTION)
from texmo.layers.conv import ConvDef
from texmo.layers.msr import MsrDef
from texmo.layers.suffix import SuffixDef
from texmo.spec_parser import _build_layer_def


def _ref_rope(x, positions, rd):
    """Split-half RoPE on the first rd dims (HF rotate_half form)."""
    inv_freq = 1.0 / (_ROPE_BASE ** (np.arange(0, rd, 2) / rd))
    freqs = positions[:, None].astype(np.float64) * inv_freq[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)      # (T, rd)
    cos, sin = np.cos(emb), np.sin(emb)
    while cos.ndim < x.ndim:
        cos, sin = cos[:, None], sin[:, None]          # broadcast heads
    x_rot, x_pass = x[..., :rd], x[..., rd:]
    half = rd // 2
    rotated = np.concatenate(
        [-x_rot[..., half:], x_rot[..., :half]], axis=-1)
    return np.concatenate([x_rot * cos + rotated * sin, x_pass], axis=-1)


def _ref_forward(x, w, heads, window):
    """Mirror of RecurrentGemmaAttention (eager, MQA), batch of one
    handled by looping the batch dim."""
    B, T, _ = x.shape
    size = w['w_q'].shape[0]
    hd = size // heads
    rd = int(hd * _ROTARY_FRACTION)
    positions = np.arange(T)
    out = np.zeros((B, T, size))
    for b in range(B):
        q = (x[b] @ w['w_q'].T).reshape(T, heads, hd)
        k = x[b] @ w['w_k'].T                          # (T, hd)
        v = x[b] @ w['w_v'].T
        q = _ref_rope(q, positions, rd)
        k = _ref_rope(k, positions, rd)
        ctx = np.zeros((T, heads, hd))
        for t in range(T):
            lo = max(0, t - window + 1)
            for h in range(heads):
                scores = (q[t, h] @ k[lo:t + 1].T) / np.sqrt(hd)
                p = np.exp(scores - scores.max())
                p /= p.sum()
                ctx[t, h] = p @ v[lo:t + 1]
        out[b] = ctx.reshape(T, size) @ w['w_o'].T + w['b_o']
    return out


def _make(input_size=6, size=16, heads=2, window=4, seed=0):
    layer = AttnJax(input_size, size, heads, window, dtype=jnp.float32)
    weights = layer.init_weights(jax.random.PRNGKey(seed))
    return layer, weights


def _np_weights(w):
    return {k: np.asarray(v, dtype=np.float64) for k, v in w.items()}


def test_forward_shape():
    layer, w = _make()
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 9, 6))
    assert layer.forward(w, x).shape == (3, 9, 16)


def test_forward_matches_reference():
    layer, w = _make(input_size=6, size=16, heads=2, window=4)
    x = jax.random.normal(jax.random.PRNGKey(2), (2, 9, 6))
    out = np.asarray(layer.forward(w, x), dtype=np.float64)
    ref = _ref_forward(
        np.asarray(x, np.float64), _np_weights(w), heads=2, window=4)
    assert np.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_single_head_full_rotary_fraction_of_window_one_token():
    # T=1: position 0 attends to itself only -- the shift-pad initial
    # vector plays the BOS role. Output must be finite and equal the
    # o-projected value of that single position.
    layer, w = _make(input_size=6, size=16, heads=2, window=4)
    x = jax.random.normal(jax.random.PRNGKey(3), (1, 1, 6))
    out = np.asarray(layer.forward(w, x))
    v = np.asarray(x[0, 0] @ np.asarray(w['w_v']).T)
    expect = np.concatenate([v, v]) @ np.asarray(w['w_o']).T + np.asarray(w['b_o'])
    assert np.allclose(out[0, 0], expect, atol=1e-5)


def test_window_banding():
    # Perturbing a token more than `window` positions before the last
    # position must not change the last position's output.
    layer, w = _make(input_size=6, size=16, heads=2, window=4)
    x = np.asarray(
        jax.random.normal(jax.random.PRNGKey(4), (1, 10, 6)))
    x2 = x.copy()
    x2[0, 2] += 5.0  # last position is 9; 9 - 2 = 7 >= window
    out1 = np.asarray(layer.forward(w, jnp.asarray(x)))
    out2 = np.asarray(layer.forward(w, jnp.asarray(x2)))
    assert np.allclose(out1[0, -1], out2[0, -1], atol=1e-5)
    # ...but a token inside the window does change it.
    x3 = x.copy()
    x3[0, 8] += 5.0
    out3 = np.asarray(layer.forward(w, jnp.asarray(x3)))
    assert not np.allclose(out1[0, -1], out3[0, -1], atol=1e-3)


def test_step_matches_forward():
    layer, w = _make(input_size=6, size=16, heads=2, window=4)
    T = 9  # > window, exercises the ring buffer
    x = jax.random.normal(jax.random.PRNGKey(5), (1, T, 6))
    fwd = np.asarray(layer.forward(w, x))[0]
    state = layer.init_state()
    for t in range(T):
        state, y = layer.step(w, state, x[0, t])
        assert np.allclose(np.asarray(y), fwd[t], atol=1e-4), t


def test_num_weights_matches_init():
    d = AttnDef(16, 2, 4, input_size=6)
    _, w = _make(input_size=6, size=16, heads=2, window=4)
    total = sum(int(np.asarray(v).size) for v in w.values())
    assert d.num_weights == total


def test_validity():
    assert AttnDef(16, 2, 8, input_size=6).is_valid()
    assert AttnDef(16, 1, 2, input_size=6).is_valid()   # single head
    assert not AttnDef(16, 3, 8, input_size=6).is_valid()   # heads not pow2
    assert not AttnDef(16, 8, 8, input_size=6).is_valid()   # head_dim 2 < 4
    assert not AttnDef(16, 2, 3, input_size=6).is_valid()   # window not pow2
    assert not AttnDef(16, 2, 1, input_size=6).is_valid()   # no context
    assert not AttnDef(24, 2, 8, input_size=6).is_valid()   # size not pow2
    # RecurrentGemma shape runs but is load-only (not search-valid).
    rg = AttnDef(2560, 10, 2048, input_size=2560)
    assert rg.head_dim == 256
    assert not rg.is_valid()


def test_parse_and_model2_build():
    from texmo.precision import Precision
    from texmo.spec_parser import parse_model2
    ld = _build_layer_def("attn.16.2.8", input_size=6)
    assert isinstance(ld, AttnDef)
    assert (ld.size, ld.heads, ld.window) == (16, 2, 8)
    assert str(ld) == "attn.16.2.8"
    m = parse_model2("bytes|dense.8.gelu-attn.8.2.8", Precision.FP32)
    assert "attn" in [l.name for l in m.layer_seq.layers]
    m.build_jax()  # constructs without error


def test_neighbors_mutations_and_family_swaps():
    # 2x mutations on all three metaparameters + the msr swap.
    nbs = list(AttnDef(16, 2, 8, input_size=6).neighbors())
    # The msr swap preserves total width, so attn.16.2 (head_dim 8)
    # maps onto msr's *per-head* first field: msr.8.2 is also 16 wide.
    for expect in ("attn.8.2.8", "attn.32.2.8", "attn.16.1.8",
                   "attn.16.4.8", "attn.16.2.4", "attn.16.2.16",
                   "msr.8.2"):
        assert expect in nbs, expect
    # heads != 1 -> no suffix swap.
    assert not any(s.startswith("suffix") for s in nbs)
    # suffix swap only in the exact image of the forward swap:
    # heads == 1 AND size == input width.
    assert "suffix.4" in list(AttnDef(8, 1, 4, input_size=8).neighbors())
    assert "suffix.4" not in list(AttnDef(8, 1, 4, input_size=6).neighbors())


def test_msr_and_suffix_swap_to_attn():
    # msr.8.2 is heads*dim = 16 wide, so its equal-shape attn twin is
    # attn.16.2 (head_dim 8) -- not attn.8.2, which would be 8 wide.
    assert "attn.16.2.16" in list(MsrDef(8, 2, input_size=6).neighbors())
    # suffix.L at input width D -> attn.D.1.L (learned-soft wrap of the
    # same span, sized to the suffix's input).
    assert "attn.8.1.4" in list(SuffixDef(4, input_size=8).neighbors())


def test_msr_attn_swap_preserves_width_and_round_trips():
    # Family swaps land on the equal-shape twin. msr's first field is
    # the per-head dim (width H*D); attn's is the total size (head_dim
    # = size/H). The swap therefore maps H*D <-> size in both
    # directions, keeping both the width and the head count.
    msr = MsrDef(8, 4, input_size=6)
    assert msr.size == 32
    fwd = f"attn.32.4.{_ATTN_SWAP_WINDOW}"
    assert fwd in list(msr.neighbors())

    attn = AttnDef(32, 4, _ATTN_SWAP_WINDOW, input_size=6)
    assert attn.head_dim == 8
    assert attn.size == msr.size
    assert "msr.8.4" in list(attn.neighbors())
    # Round trip: attn -> msr -> attn is the identity at the swap
    # window (the window is the one field msr can't carry).
    back = _build_layer_def("msr.8.4", input_size=6)
    assert fwd in list(back.neighbors())

    # No equal-shape twin below head_dim 4 (attn needs a rotary pair):
    # yield nothing rather than an invalid spec, in both directions.
    assert not any(
        n.startswith("attn.")
        for n in MsrDef(2, 4, input_size=6).neighbors())
    assert not AttnDef(16, 8, 8, input_size=6).is_valid()   # head_dim 2
    assert not any(
        n.startswith("msr.")
        for n in AttnDef(16, 8, 8, input_size=6).neighbors())


def test_conv_attn_swap_is_exact_and_round_trips():
    # Third side of the windowed triangle: conv and attn are the two
    # length-preserving mixers over the same span, so kernel <-> window
    # transfers field for field (both floors are 2, so no valid kernel
    # maps to an invalid window or back).
    src = ConvDef(4, input_size=8)
    twin = "attn.8.1.4"
    assert twin in list(src.neighbors())
    # Round trip is the identity -- both the width and the span survive.
    twin_def = _build_layer_def(twin, input_size=8)
    assert str(twin_def) == twin
    back = list(twin_def.neighbors())
    assert str(src) in back                 # conv.4
    assert "suffix.4" in back               # the other windowed swap
    # Guarded to the exact image of the forward edge: no conv twin at
    # multiple heads, nor when the layer doesn't preserve input width.
    assert not any(
        n.startswith("conv.")
        for n in AttnDef(8, 2, 4, input_size=8).neighbors())
    assert not any(
        n.startswith("conv.")
        for n in AttnDef(8, 1, 4, input_size=6).neighbors())


def test_temporal_wrap_adjacency_invalid():
    from texmo.precision import Precision
    from texmo.spec_parser import parse_model2

    def ok(s):
        return parse_model2(s, Precision.FP32).is_valid()
    assert not ok("bytes|dense.8.gelu-attn.8.1.4-attn.8.1.4")   # attn-attn
    assert not ok("bytes|dense.8.gelu-attn.8.2.4-msr.8.1")      # attn-msr
    assert not ok("bytes|dense.8.gelu-suffix.2-attn.16.1.4")    # suffix-attn
    assert not ok("bytes|dense.8.gelu-attn.8.1.4-conv.4")       # attn-conv
    # Separated by a non-wrap layer -> fine.
    assert ok("bytes|dense.8.gelu-attn.8.2.4-dense.8.gelu-msr.8.1")


def test_append_attn_uses_running_width():
    from texmo.precision import Precision
    from texmo.spec_parser import parse_model2
    # bits.1+bp has output_size 1; the generic appends use
    # min(width, output) = 1, but attn is seeded at the running width 8.
    m = parse_model2("bits.1+bp|dense.8.gelu", Precision.FP32)
    specs = {nb.spec for nb in m.neighbors()}
    assert "bits.1+bp|dense.8.gelu-attn.8.1.16" in specs


def test_predictors_handle_attn():
    from texmo.configuration import Configuration
    from texmo.precision import Precision
    from texmo.predict import timing
    from texmo.predict.loss_rnn import _layer_features
    from texmo.predict.predict_common import model_layers
    from texmo.spec_parser import parse_model2
    model = parse_model2("bytes|dense.8.gelu-attn.8.2.16", Precision.FP32)
    conf = Configuration(model, lr=0.01, length=32, batch=4, steps=1,
                         decay=1.0)
    assert any(c.type_id == "attn" for c in timing.featurize(conf))
    attn_layer = next(l for l in model_layers(conf) if l.name == "attn")
    feat = _layer_features(attn_layer, {"attn": 0}, n_simple=1)
    assert feat[3] == 1.0            # simple-type one-hot
    assert feat[3 + 1 + 9] == 1.0    # shared slot: log2(heads=2)
    assert feat[3 + 1 + 10] == 4.0   # attn window: log2(16)
