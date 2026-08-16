"""Model2 tests: parsing, weights, validity, neighbors, forward/step.

Plain-sequence specs and split specs. Several expectations here were
originally verified as parity against the legacy ModelDef (retired
2026-07) and are now frozen as literal values -- weight counts,
layer counts, validity verdicts, and the classic mutation moves the
legacy neighbor generator defined.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from texmo.precision import Precision
from texmo.spec_parser import parse_model2

_1_BY_LOG2 = 1.0 / math.log(2.0)


def _build2(spec, precision=Precision.FP32, seed=42):
    md = parse_model2(spec, precision)
    model = md.build_jax()
    weights = model.init_weights(jax.random.PRNGKey(seed))
    return md, model, weights


# --- LayerSeqDef pass-through wraps the layer chain correctly --------


def test_def_properties():
    md = parse_model2("bytes|dense.32.gelu-dense.16.tanh", Precision.FP32)
    assert md.input.size == 256
    assert md.layer_seq.input_size == 256
    assert md.layer_seq.size == 16
    assert md.layer_seq.length == 1
    assert md.total_padding == 1
    assert md.output.input_size == 16


def test_def_num_weights():
    # bits.1+bp input width 4 (1 bit + 3 binary-position channels):
    # dense.8: 8*4+8 = 40; dense.4: 4*8+4 = 36; binary head (1 logit):
    # 1*4+1 = 5. Total 81 (matched the legacy ModelDef before its
    # retirement).
    md2 = parse_model2("bits.1+bp|dense.8.gelu-dense.4.tanh", Precision.FP32)
    assert md2.num_weights == 81


def test_def_num_mults():
    md2 = parse_model2("bits.1+bp|gru.8-dense.4.tanh", Precision.FP32)
    assert md2.num_mults == 353


def test_def_length_aware_total_padding():
    """suffix.4 contributes 3 to total_padding (length-1)."""
    md = parse_model2(
        "bits.1+bp|dense.4.gelu-suffix.4-dense.4.tanh", Precision.FP32)
    assert md.layer_seq.length == 4  # 1 + (1-1) + (4-1) + (1-1)
    assert md.total_padding == 4


def test_def_no_layers():
    """Empty layer chain still produces a valid model spec."""
    md = parse_model2("bits.1+bp|", Precision.FP32)
    assert md.layer_seq.layers == []
    assert md.layer_seq.size == md.input.size
    assert md.layer_seq.length == 1
    assert md.total_padding == 1


def test_split_residual_weight_count():
    """The split marker has 0 weights of its own: 3x dense.16 over a
    256-wide input/output. dense.16.gelu from bytes: 16*256+16 = 4112;
    two dense.16 at width 16: 2 * (16*16+16) = 544; head:
    256*16+256 = 4352. Total 9008."""
    spec = "bytes|dense.16.gelu-split.add(dense.16.gelu, pass)-dense.16.gelu"
    m2 = parse_model2(spec, Precision.FP32)
    assert m2.num_weights == 9008
    assert m2.total_padding == 1


# -- num_layers --------------------------------------------------------


@pytest.mark.parametrize("spec, expected", [
    ("bytes|dense.16.gelu", 1),
    ("bytes|dense.16.gelu-dense.16.gelu", 2),
    ("bytes|dense.16.gelu-gru.16-dense.16.gelu", 3),
    # A split counts itself plus the layers in its branches -- the
    # same count the retired flat representation gave the equivalent
    # skip spec (skip marker 1 + each spanned layer 1).
    ("bytes|dense.16.gelu-split.add(dense.16.gelu, pass)"
     "-dense.16.gelu", 4),
    ("bytes|dense.16.gelu-split.add(dense.16.gelu-dense.16.gelu"
     "-dense.16.gelu, pass)-dense.16.gelu", 6),
    ("bytes|split.cat(dense.16.gelu, pass)-dense.16.gelu", 3),
    # Nested residual splits.
    ("bytes|dense.16.gelu-split.add(dense.16.gelu"
     "-split.add(dense.16.gelu, pass), pass)-dense.16.gelu", 6),
    ("bytes|split.add(dense.16.gelu-split.cat(dense.16.gelu, pass),"
     " pass)-dense.16.gelu", 5),
])
def test_num_layers_residual_splits(spec, expected):
    m2 = parse_model2(spec, Precision.FP32)
    assert m2.num_layers == expected


def test_num_layers_split_authored():
    """Hand-authored split spec, no ModelDef analog. Recursive count
    is `1 (outer dense) + 1 (split itself) + 1 (in-branch dense) +
    0 (pass) + 1 (outer dense) = 4`."""
    md = parse_model2(
        "bytes|dense.16.gelu-split.mul(dense.16.gelu, pass)"
        "-dense.16.gelu",
        Precision.FP32)
    assert md.num_layers == 4


def test_num_layers_nested_splits():
    """`split.mul(split.add(dense, pass), pass)` -- the outer split
    counts 1, its main branch contains an inner split which counts
    1 + (1 from dense) + 0 = 2; outer branch_b is pass = 0. Total
    inside outer = 2. Plus outer's own +1 = 3."""
    md = parse_model2(
        "bytes|split.mul(split.add(dense.16.gelu, pass), pass)",
        Precision.FP32)
    assert md.num_layers == 3


def test_num_layers_empty():
    md = parse_model2("bits.1+bp|", Precision.FP32)
    assert md.num_layers == 0


def test_residual_split_spec_runs_forward():
    """Sanity: a residual split spec actually trains."""
    md = parse_model2(
        "bits.1+bp|dense.4.gelu-split.add(dense.4.gelu, pass)-dense.4.gelu",
        Precision.FP32)
    model = md.build_jax()
    weights = model.init_weights(jax.random.PRNGKey(0))
    batch = jax.random.randint(
        jax.random.PRNGKey(1), (2, 8), 0, 2).astype(jnp.int32)
    logits = model.forward(weights, batch)
    assert logits.shape == (2, 8, 2)
    loss = float(model.loss_batch(weights, batch))
    assert loss > 0


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


# -- end-to-end with split syntax --------------------------------------


def test_split_spec_parses_and_runs_forward():
    """Build a Model2 from a split-containing spec, run forward,
    verify shape and that the loss is finite."""
    md = parse_model2(
        "bits.1+bp|split.mul(dense.8.gelu, pass)", Precision.FP32)
    model = md.build_jax()
    weights = model.init_weights(jax.random.PRNGKey(0))
    batch = jax.random.randint(
        jax.random.PRNGKey(1), (2, 16), 0, 2).astype(jnp.int32)
    logits = model.forward(weights, batch)
    assert logits.shape == (2, 16, 2)
    loss = float(model.loss_batch(weights, batch))
    assert loss > 0


def test_split_spec_step_matches_forward():
    """Per-token step path agrees with the full-sequence forward
    for a split-containing model."""
    md = parse_model2(
        "bits.1+bp|split.mul(dense.4.gelu, pass)", Precision.FP32)
    model = md.build_jax()
    weights = model.init_weights(jax.random.PRNGKey(0))
    tokens = [0, 1, 1, 0, 1, 0]
    batch = jnp.array([tokens], dtype=jnp.int32)
    fwd = model.forward(weights, batch)

    states, logits0 = model.initial_step(weights)
    step_logits = [logits0]
    for t in tokens[:-1]:
        states, logits_t = model.step(weights, states, t)
        step_logits.append(logits_t)
    for i in range(len(tokens)):
        np.testing.assert_allclose(
            fwd[0, i], step_logits[i], atol=1e-5)


def test_split_spec_loss_decreases():
    """Training a split-containing model reduces the loss --
    confirms that gradients reach both branches' weights through
    the nested pytree."""
    import optax
    md = parse_model2(
        "bits.1+bp|split.mul(dense.4.gelu, dense.4.gelu)",
        Precision.FP32)
    model = md.build_jax()
    weights = model.init_weights(jax.random.PRNGKey(0))
    batch = jax.random.randint(
        jax.random.PRNGKey(1), (8, 32), 0, 2).astype(jnp.int32)
    loss_fn = jax.jit(model.loss_batch)
    grad_fn = jax.jit(jax.grad(model.loss_batch))
    initial = float(loss_fn(weights, batch))
    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(weights)
    for _ in range(30):
        grads = grad_fn(weights, batch)
        updates, opt_state = optimizer.update(grads, opt_state, weights)
        weights = optax.apply_updates(weights, updates)
    final = float(loss_fn(weights, batch))
    assert final < initial


def test_split_nested_inside_branch():
    """A split nested inside another split's branch parses through
    the `_parse_split` -> `parse_layer_list` recursion."""
    from texmo.layers.split import SplitDef
    md = parse_model2(
        "bits.1+bp|split.mul("
        "dense.4.gelu-split.add(dense.4.gelu, pass)-dense.4.gelu, "
        "pass)",
        Precision.FP32)
    outer = md.layer_seq.layers[0]
    assert isinstance(outer, SplitDef)
    inner_layers = outer.branches[0].layers
    # dense + nested split + dense.
    assert isinstance(inner_layers[1], SplitDef)
    assert inner_layers[1].op == 'add'


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


# --- validity rules: adjacency rules + bare-dense carve-out ----------
#
# The adjacency rules the retired flat representation enforced live
# in LayerSeqDef / SplitDef. These check that the classically-invalid
# shapes stay rejected in split form, that the bare-dense carve-out
# for gated units is accepted, and that adjacency rules apply inside
# branches too.


_INVALID_SPECS = [
    # Norm can't be the first layer (top-level).
    "bits.1+bp|norm-dense.4.gelu",
    # A residual branch can't end in a suffix (the merge point would
    # sit right after it).
    "bytes|split.add(suffix.4, pass)-dense.32.gelu",
    # Two suffix-like layers adjacent.
    "bits.1+bp|suffix.2-suffix.2-dense.4.gelu",
    # Two norm adjacent.
    "bits.1+bp|dense.4.gelu-norm-norm-dense.4.gelu",
    # Norm follows a suffix.
    "bits.1+bp|suffix.2-norm-dense.4.gelu",
]


@pytest.mark.parametrize("spec", _INVALID_SPECS)
def test_invalid_specs(spec):
    assert not parse_model2(spec, Precision.FP32).is_valid()


def test_pre_norm_branch_valid():
    """A residual branch that STARTS with norm -- the pre-norm
    residual pattern. The retired flat representation rejected the
    equivalent skip shape ('skip source before norm'); Model2
    deliberately allows it."""
    spec = "bytes|dense.32.gelu-split.add(norm, pass)-dense.32.gelu"
    assert parse_model2(spec, Precision.FP32).is_valid()


_VALID_SPECS = [
    "bits.1+bp|dense.4.gelu-suffix.2-dense.4.tanh",
    "bytes|dense.32.gelu-split.add(dense.32.gelu, pass)-dense.32.gelu",
    # A suffix at the start of a residual branch is fine.
    "bytes|dense.32.gelu-split.add(suffix.2-dense.32.gelu, pass)"
    "-dense.16.gelu",
]


@pytest.mark.parametrize("spec", _VALID_SPECS)
def test_valid_specs(spec):
    assert parse_model2(spec, Precision.FP32).is_valid()


def test_geglu_bare_dense_branch_valid():
    """A GeGLU -- `split.mul(dense.X.gelu, dense.X)` -- has a bare
    linear value path. The tree accepts it; this shape has no
    ModelDef equivalent (the old model had no gated split)."""
    md = parse_model2(
        "bits.1+bp|split.mul(dense.4.gelu, dense.4)", Precision.FP32)
    assert md.is_valid()


def test_bare_dense_top_level_invalid():
    """The carve-out is branch-only: a bare dense in the top-level
    chain is still rejected."""
    md = parse_model2("bits.1+bp|dense.4-dense.4.gelu", Precision.FP32)
    assert not md.is_valid()


@pytest.mark.parametrize("spec, valid", [
    # add/cat canonical: transform first, identity (pass) second.
    ("bits.1+bp|split.add(dense.4.gelu, pass)", True),
    ("bits.1+bp|split.add(pass, dense.4.gelu)", False),
    ("bits.1+bp|split.cat(dense.4.gelu, pass)", True),
    ("bits.1+bp|split.cat(pass, dense.4.gelu)", False),
    # mul canonical: value first, gate second; the gate is never pass.
    ("bits.1+bp|split.mul(pass, dense.4)", True),
    ("bits.1+bp|split.mul(dense.4, pass)", False),
    # all-pass degenerate splits are rejected for every op.
    ("bits.1+bp|split.add(pass, pass)", False),
    ("bits.1+bp|split.cat(pass, pass)", False),
    ("bits.1+bp|split.mul(pass, pass)", False),
    # mul bilinear dedup: activated value goes first (GeGLU), its swap
    # is rejected, and the symmetric both-bare bilinear is allowed.
    ("bits.1+bp|split.mul(dense.4.gelu, dense.4)", True),
    ("bits.1+bp|split.mul(dense.4, dense.4.gelu)", False),
    ("bits.1+bp|split.mul(dense.4, dense.4)", True),
])
def test_split_pass_position_and_bilinear_rules(spec, valid):
    assert parse_model2(spec, Precision.FP32).is_valid() is valid


# --- neighbor generation ---------------------------------------------


def _arch_neighbor_specs(model) -> set:
    """Specs of same-precision (architecture) neighbors."""
    return {
        n.spec for n in model.neighbors()
        if n.precision == model.precision
    }


_NEIGHBOR_BASES = [
    "bits.1+bp|gru.4-dense.2.tanh",
    "bits.1+bp|dense.4.gelu-suffix.2-dense.4.tanh",
    "bits.1+bp|dense.4.gelu-split.add(dense.4.gelu, pass)-dense.4.gelu",
    "bits.1+bp|dense.4.gelu-split.mul(dense.4.gelu, dense.4)",
]


@pytest.mark.parametrize("spec", _NEIGHBOR_BASES)
def test_neighbors_all_valid_distinct_excludes_self(spec):
    model = parse_model2(spec, Precision.FP32)
    neighbors = list(model.neighbors())
    assert neighbors, "expected at least one neighbor"
    for n in neighbors:
        assert n.is_valid(), f"invalid neighbor: {n.spec}"
    # Architecture neighbors are distinct and never equal to self.
    specs = [n.spec for n in neighbors if n.precision == model.precision]
    assert model.spec not in specs
    assert len(specs) == len(set(specs))


@pytest.mark.parametrize("spec", _NEIGHBOR_BASES)
def test_neighbor_specs_roundtrip(spec):
    """Every neighbor spec is canonical (re-parses to itself)."""
    model = parse_model2(spec, Precision.FP32)
    for n in model.neighbors():
        assert parse_model2(n.spec, n.precision).spec == n.spec


def test_neighbors_cover_classic_moves():
    """The classic mutation moves the legacy ModelDef generator
    defined must stay reachable (frozen from the last parity run
    before its retirement): per-layer size x2 / /2, activation swap,
    dense<->rnn type swap, append, suffix/norm insert, and the
    dense prepend."""
    m2 = parse_model2("bytes|dense.32.gelu-dense.16.gelu", Precision.FP32)
    expected = {
        # size x2 / /2 on each layer
        "bytes|dense.64.gelu-dense.16.gelu",
        "bytes|dense.16.gelu-dense.16.gelu",
        "bytes|dense.32.gelu-dense.32.gelu",
        "bytes|dense.32.gelu-dense.8.gelu",
        # activation swap
        "bytes|dense.32.tanh-dense.16.gelu",
        "bytes|dense.32.gelu-dense.16.tanh",
        # dense <-> rnn type swap
        "bytes|rnn.32.gelu-dense.16.gelu",
        "bytes|dense.32.gelu-rnn.16.gelu",
        # append (recurrent family + conv)
        "bytes|dense.32.gelu-dense.16.gelu-gru.16",
        "bytes|dense.32.gelu-dense.16.gelu-lstm.16",
        "bytes|dense.32.gelu-dense.16.gelu-conv.2",
        # suffix / norm inserts
        "bytes|suffix.2-dense.32.gelu-dense.16.gelu",
        "bytes|dense.32.gelu-suffix.2-dense.16.gelu",
        "bytes|dense.32.gelu-dense.16.gelu-suffix.2",
        "bytes|dense.32.gelu-norm-dense.16.gelu",
        "bytes|dense.32.gelu-dense.16.gelu-norm",
        # dense prepend at the input width
        "bytes|dense.256.tanh-dense.32.gelu-dense.16.gelu",
        "bytes|dense.256.gelu-dense.32.gelu-dense.16.gelu",
    }
    m2_arch = _arch_neighbor_specs(m2)
    missing = expected - m2_arch
    assert not missing, f"Model2 neighbors miss: {sorted(missing)}"


def test_neighbor_introduces_split_families():
    specs = _arch_neighbor_specs(
        parse_model2("bits.1+bp|dense.4.gelu-dense.4.tanh", Precision.FP32))
    # residual (add/cat) and gate (mul) introductions all appear.
    assert any("split.add(" in s for s in specs)
    assert any("split.cat(" in s for s in specs)
    assert any("split.mul(" in s for s in specs)


def test_neighbor_op_swap_add_cat():
    specs = _arch_neighbor_specs(parse_model2(
        "bits.1+bp|dense.4.gelu-split.add(dense.4.gelu, pass)",
        Precision.FP32))
    assert "bits.1+bp|dense.4.gelu-split.cat(dense.4.gelu, pass)" in specs


def test_neighbor_unwrap_split():
    """A split can be removed by inlining its main branch."""
    specs = _arch_neighbor_specs(parse_model2(
        "bits.1+bp|dense.4.gelu-split.add(dense.4.tanh, pass)",
        Precision.FP32))
    assert "bits.1+bp|dense.4.gelu-dense.4.tanh" in specs


def test_neighbor_recurses_into_branch():
    """A layer inside a split branch gets mutated (size 2x here)."""
    specs = _arch_neighbor_specs(parse_model2(
        "bits.1+bp|dense.4.gelu-split.add(dense.4.tanh, pass)",
        Precision.FP32))
    assert "bits.1+bp|dense.4.gelu-split.add(dense.8.tanh, pass)" in specs


def test_neighbor_mul_gate_activation_and_self_gate():
    specs = _arch_neighbor_specs(parse_model2(
        "bits.1+bp|dense.4.gelu-split.mul(dense.4.gelu, dense.4)",
        Precision.FP32))
    # gate activation toggle on the second path
    assert any(
        "split.mul(dense.4.gelu, dense.4.gelu)" in s for s in specs)
    # self-gate: pass is the value (first); the gate slot is never pass
    assert any("split.mul(pass, dense.4.gelu)" in s for s in specs)


def test_neighbor_append_self_gate():
    """split.mul(pass, dense.X.act) is appendable on the running
    activation -- reachable from the empty chain across the GLU
    activations -- and the unwrap removes it (the inverse)."""
    empty = _arch_neighbor_specs(parse_model2("bits.1+bp|", Precision.FP32))
    for g in ("dense.4", "dense.4.gelu", "dense.4.tanh"):
        assert f"bits.1+bp|split.mul(pass, {g})" in empty
    # relu retired: not offered as a gate activation anymore.
    assert "bits.1+bp|split.mul(pass, dense.4.relu)" not in empty
    chain = _arch_neighbor_specs(
        parse_model2("bits.1+bp|dense.4.gelu", Precision.FP32))
    assert "bits.1+bp|dense.4.gelu-split.mul(pass, dense.4)" in chain
    # inverse: unwrapping a trailing self-gate drops it.
    gated = _arch_neighbor_specs(parse_model2(
        "bits.1+bp|dense.4.gelu-split.mul(pass, dense.4)", Precision.FP32))
    assert "bits.1+bp|dense.4.gelu" in gated


def test_neighbor_grows_split_span():
    """The skip distance+1 analog: a split consumes the following
    layer into its main branch."""
    specs = _arch_neighbor_specs(parse_model2(
        "bits.1+bp|split.add(dense.4.gelu, pass)-dense.4.tanh",
        Precision.FP32))
    assert (
        "bits.1+bp|split.add(dense.4.gelu-dense.4.tanh, pass)" in specs)


def test_neighbor_shrinks_split_span():
    """The skip distance-1 analog: a split releases its main branch's
    last layer back out after the split."""
    specs = _arch_neighbor_specs(parse_model2(
        "bits.1+bp|split.add(dense.4.gelu-dense.4.tanh, pass)",
        Precision.FP32))
    assert (
        "bits.1+bp|split.add(dense.4.gelu, pass)-dense.4.tanh" in specs)


def test_grow_shrink_are_inverse():
    grown = "bits.1+bp|split.add(dense.4.gelu-dense.4.tanh, pass)"
    base = "bits.1+bp|split.add(dense.4.gelu, pass)-dense.4.tanh"
    # grow(base) -> grown
    assert grown in _arch_neighbor_specs(parse_model2(base, Precision.FP32))
    # shrink(grown) -> base
    assert base in _arch_neighbor_specs(parse_model2(grown, Precision.FP32))


def test_grow_over_suffix_filtered():
    """Growing the span to end in a suffix-like layer is rejected by
    the branch-can't-end-in-suffix rule (no extra bump needed)."""
    specs = _arch_neighbor_specs(parse_model2(
        "bits.1+bp|split.add(dense.4.gelu, pass)-suffix.2-dense.4.tanh",
        Precision.FP32))
    assert not any(
        "split.add(dense.4.gelu-suffix.2, pass)" in s for s in specs)


def test_neighbors_two_hops_never_raise():
    """Every generated spec must parse -- neighbors() no longer
    swallows parse errors -- including at the second hop, where the
    branch recursion produces the deepest specs."""
    base = parse_model2(
        "bits.1+bp|dense.4.gelu-split.mul(dense.4.gelu, dense.4)",
        Precision.FP32)
    count = 0
    for n in base.neighbors():
        for nn in n.neighbors():  # raises if any 2-hop spec is malformed
            assert nn.is_valid()
            count += 1
    assert count > 0


def test_no_add_cat_to_mul_op_swap():
    """mul is its own family -- a residual split never op-swaps to
    mul, and a gate never op-swaps to add/cat."""
    res = _arch_neighbor_specs(parse_model2(
        "bits.1+bp|dense.4.gelu-split.add(dense.4.gelu, pass)",
        Precision.FP32))
    # The residual's own slot is never rewritten to split.mul(...).
    assert not any(
        s.endswith("split.mul(dense.4.gelu, pass)") for s in res)


def test_hawk_conf_neighbors_no_crash():
    """Neighbor enumeration over a conf containing bare denses (the
    Hawk block's linear front-end and GeGLU path) must not emit
    unparseable specs. Regression: bare DenseDef.neighbors() rendered
    size mutations as 'dense.N.None', which crashed re-parsing."""
    spec = (
        "bits.1+bp|dense.2.gelu"
        "-split.add(rmsnorm-split.mul(dense.2-conv.4-rglru.1,"
        " dense.2.gelu)-dense.2, pass)"
        "-split.add(rmsnorm-split.mul(dense.2.gelu, dense.2)-dense.2,"
        " pass)"
    )
    m = parse_model2(spec, Precision.FP32)
    neighbors = list(m.neighbors())  # raised KeyError('None') before
    assert neighbors
    assert all("None" not in nb.spec for nb in neighbors)


def test_bare_dense_neighbors_well_formed():
    from texmo.layers.dense import DenseDef
    nbs = list(DenseDef(8, input_size=8).neighbors())
    # Size mutations render without an activation suffix...
    assert "dense.4" in nbs and "dense.16" in nbs
    # ...and no cross-type swap or ".None" artifacts appear.
    assert not any("None" in s or s.startswith("rnn") for s in nbs)


# --- codec mode swaps + emb width sync in neighbor generation --------


def test_emb_mode_swap_neighbors_both_ways():
    md = parse_model2("bytes|dense.8.gelu", Precision.FP32)
    specs = {n.spec for n in md.neighbors()}
    assert "bytes.emb.8|dense.8.gelu" in specs

    back = parse_model2("bytes.emb.8|dense.8.gelu", Precision.FP32)
    specs_back = {n.spec for n in back.neighbors()}
    assert "bytes|dense.8.gelu" in specs_back
    # Domain ladder at the same table width.
    assert "bits.4.emb.8|dense.8.gelu" in specs_back


def test_emb_width_syncs_on_last_layer_resize():
    md = parse_model2("bytes.emb.4|dense.4.tanh", Precision.FP32)
    specs = {n.spec for n in md.neighbors()}
    # The dense.4 -> dense.8 resize mutation must arrive with the
    # matching table width (X is slaved to the chain's final width;
    # the sync happens in neighbor generation, not the parser).
    assert "bytes.emb.8|dense.8.tanh" in specs
    assert "bytes.emb.4|dense.8.tanh" not in specs
    # All produced neighbors are valid models (X consistent).
    for n in md.neighbors():
        assert n.is_valid(), n.spec


def test_emb_neighbors_two_hop_wellformed():
    base = parse_model2("bits.2.emb.4|rnn.4.tanh", Precision.FP32)
    for n in base.neighbors():
        for nn in n.neighbors():  # raises if any 2-hop spec is malformed
            assert nn.is_valid(), nn.spec


def test_bare_dense_tail_valid_for_tied_codec_only():
    # The explicit adapter: legal exactly when the head scores against
    # the tied table instead of absorbing the projection.
    assert parse_model2(
        "bytes.emb.4|rnn.8.tanh-dense.4", Precision.FP32).is_valid()
    assert not parse_model2(
        "bytes|rnn.8.tanh-dense.8", Precision.FP32).is_valid()


def test_bare_dense_append_reaches_adapter_form():
    md = parse_model2("bits.4.emb.8|rnn.8.tanh", Precision.FP32)
    specs = {n.spec for n in md.neighbors()}
    assert "bits.4.emb.8|rnn.8.tanh-dense.8" in specs
    # One-hot models don't get the bare append (head absorbs it).
    md = parse_model2("bytes|dense.8.gelu", Precision.FP32)
    assert "bytes|dense.8.gelu-dense.8" not in {
        n.spec for n in md.neighbors()}


def test_appends_are_size_preserving_snapped():
    # Binary one-hot: appends now match the running width instead of
    # collapsing to the (1-logit) head size.
    md = parse_model2("bits.1+bp|rnn.4.tanh", Precision.FP32)
    specs = {n.spec for n in md.neighbors()}
    assert "bits.1+bp|rnn.4.tanh-dense.4.gelu" in specs
    assert "bits.1+bp|rnn.4.tanh-msr.4.1" in specs
    assert "bits.1+bp|rnn.4.tanh-dense.1.gelu" not in specs
    # Odd one-hot input width snaps down to the grid for prepends.
    md = parse_model2("bits.4.oh+bp|rnn.8.tanh", Precision.FP32)
    specs = {n.spec for n in md.neighbors()}
    assert "bits.4.oh+bp|dense.16.tanh-rnn.8.tanh" in specs


# --- step == forward across every layer family -----------------------
#
# The invariant that would have caught the rglru step-reset bug: the
# two execution paths must agree on every spec, whatever the layer's
# internal state machinery. One sweep over all families so a new
# layer type can't land without joining it.

@pytest.mark.parametrize("spec", [
    "bits.1+bp|dense.4.gelu",
    "bits.1+bp|rnn.4.tanh",
    "bits.1+bp|gru.4",
    "bits.1+bp|mgru.4",
    "bits.1+bp|mingru.4",
    "bits.1+bp|lstm.4",
    "bits.1+bp|slstm.4",
    "bits.1+bp|mullstm.4",
    "bits.1+bp|matlstm.4",
    "bits.1+bp|latent.4.2",
    "bits.1+bp|lrnn.4.2",
    "bits.1+bp|lmgu.4.2",
    "bits.1+bp|suffix.2-dense.4.gelu",
    "bits.1+bp|conv.2",
    "bits.1+bp|msr.4.1",
    "bits.1+bp|attn.4.1.4",
    "bits.1+bp|rglru.1",
    "bits.1+bp|rglru.4",
    "bytes.emb.4|rnn.4.tanh",
    "bits.2.emb.4|dense.4.tanh",
    # A consuming layer (conv/suffix, valid-trim in forward) followed
    # by a STATEFUL one: forward drops the consuming layer's transient
    # outputs, so the downstream state must never see them. The old
    # synchronous warm-up ticked the whole chain and fed them in;
    # initial_step now prefills the padding prefix in forward
    # semantics instead (Model2Jax.initial_step -> LayerJax.prefill).
    "bits.4.oh+bp|dense.8-conv.2-rglru.2",
    "bits.1+bp|suffix.2-gru.4",
    # Two consuming layers with a stateful one in between (padding
    # accumulates to 3, and each consumer must trim on its own prefix).
    "bits.1+bp|suffix.2-gru.4-conv.2",
    # Consuming layer upstream of the matrix/KV-state families. These
    # also pin position alignment: msr's and attn's step states carry a
    # position counter, which must advance by the number of prefix
    # positions the layer actually sees, not by total_padding.
    "bits.4.oh+bp|dense.8-conv.2-dense.8.gelu-msr.4.1",
    "bits.1+bp|suffix.2-dense.4.gelu-attn.4.1.4",
    # Stateful branch inside a split fed by a consuming layer: the
    # branch states have to land in the slots step() reads.
    "bits.1+bp|conv.2-split.add(gru.4, pass)",
])
def test_step_matches_forward_all_families(spec):
    _, model, weights = _build2(spec, seed=7)
    ntokens = model.codec.ntokens
    batch = jax.random.randint(
        jax.random.PRNGKey(11), (1, 10), 0, ntokens).astype(jnp.int32)
    fwd = np.asarray(model.forward(weights, batch))
    states, logits0 = model.initial_step(weights)
    np.testing.assert_allclose(
        np.asarray(logits0), fwd[0, 0], atol=2e-5, rtol=1e-4)
    for t in range(batch.shape[1] - 1):
        states, logits = model.step(weights, states, batch[0, t])
        np.testing.assert_allclose(
            np.asarray(logits), fwd[0, t + 1], atol=2e-5, rtol=1e-4,
            err_msg=f"{spec} at t={t}")
