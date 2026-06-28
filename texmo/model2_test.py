"""Model2 tests: parse + forward/step parity with the existing Model.

Plain-sequence specs and split specs. The main correctness
criterion is that Model2 produces the same logits as ModelDef on
the same plain-sequence spec when given the same weights, and that
split specs round-trip through parse + forward/step.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from texmo.model import ModelDef
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


def test_def_num_weights_matches_model():
    """Model2's weight count should match ModelDef's on the same spec."""
    spec = "bits.1+bp|dense.8.gelu-dense.4.tanh"
    md = ModelDef(spec, Precision.FP32)
    md2 = parse_model2(spec, Precision.FP32)
    assert md.num_weights == md2.num_weights


def test_def_num_mults_matches_model():
    spec = "bits.1+bp|gru.8-dense.4.tanh"
    md = ModelDef(spec, Precision.FP32)
    md2 = parse_model2(spec, Precision.FP32)
    assert md.num_mults == md2.num_mults


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


def test_skip_translated_to_split_at_top_level():
    """Legacy skip syntax now parses -- it's translated to the
    equivalent `split.op(...)` form. The runtime sees a skip-free
    tree."""
    from texmo.layers.skip import SkipDef
    from texmo.layers.split import SplitDef
    md = parse_model2(
        "bytes|dense.16.gelu-skip.1.add-dense.16.gelu-dense.16.gelu",
        Precision.FP32)
    layers = md.layer_seq.layers
    # No SkipDef should survive the translation.
    assert not any(isinstance(l, SkipDef) for l in layers)
    # The translated middle layer is a SplitDef with the in-skip
    # layer in branch 0 and a pass branch in branch 1.
    assert isinstance(layers[1], SplitDef)
    assert layers[1].op == 'add'
    assert len(layers[1].branches[0].layers) == 1  # the in-skip dense
    assert layers[1].branches[1].layers == []      # pass


def test_skip_translated_weight_count_matches_modeldef():
    """ModelDef and translated Model2 should agree on num_weights
    for a skip spec -- the skip pseudo-layer has 0 weights in both
    forms (the translated SplitDef's branches hold the same dense
    layers as the original skip's in-skip layers)."""
    spec = "bytes|dense.16.gelu-skip.1.add-dense.16.gelu-dense.16.gelu"
    m1 = ModelDef(spec, Precision.FP32)
    m2 = parse_model2(spec, Precision.FP32)
    assert m1.num_weights == m2.num_weights
    assert m1.total_padding == m2.total_padding


# -- num_layers parity -------------------------------------------------


@pytest.mark.parametrize("spec", [
    "bytes|dense.16.gelu",
    "bytes|dense.16.gelu-dense.16.gelu",
    "bytes|dense.16.gelu-gru.16-dense.16.gelu",
    "bytes|dense.16.gelu-skip.1.add-dense.16.gelu-dense.16.gelu",
    "bytes|dense.16.gelu-skip.3.add"
    "-dense.16.gelu-dense.16.gelu-dense.16.gelu-dense.16.gelu",
    "bytes|skip.1.cat-dense.16.gelu-dense.16.gelu",
    # Nested (laminar) skips: skip.3@1 spans [1,5), skip.1@3 spans
    # [3,5) -- inner nests in the outer's main branch.
    "bytes|dense.16.gelu-skip.3.add-dense.16.gelu-skip.1.add"
    "-dense.16.gelu-dense.16.gelu",
    "bytes|skip.3.add-dense.16.gelu-skip.1.cat"
    "-dense.16.gelu-dense.16.gelu",
])
def test_num_layers_matches_modeldef(spec):
    """For any laminar skip spec (non-overlapping or nested), the
    recursive count on Model2 matches the flat count on ModelDef --
    the property is construction-invariant under skip-to-split
    translation."""
    m1 = ModelDef(spec, Precision.FP32)
    m2 = parse_model2(spec, Precision.FP32)
    assert m1.num_layers == m2.num_layers


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
    m1 = ModelDef("bits.1+bp|", Precision.FP32)
    assert m1.num_layers == 0


def test_skip_translated_spec_runs_forward():
    """Sanity: a translated skip spec actually trains."""
    md = parse_model2(
        "bits.1+bp|dense.4.gelu-skip.1.add-dense.4.gelu-dense.4.gelu",
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


def test_skip_inside_split_branch_translated():
    """Skip inside a split branch translates the same way --
    recursion through `_parse_split` -> `parse_layer_list` ->
    `_translate_skips` covers it without any branch-aware logic."""
    from texmo.layers.skip import SkipDef
    from texmo.layers.split import SplitDef
    md = parse_model2(
        "bits.1+bp|split.mul("
        "dense.4.gelu-skip.1.add-dense.4.gelu-dense.4.gelu, "
        "pass)",
        Precision.FP32)
    outer = md.layer_seq.layers[0]
    assert isinstance(outer, SplitDef)
    # The first branch must be skip-free after translation.
    inner_layers = outer.branches[0].layers
    assert not any(isinstance(l, SkipDef) for l in inner_layers)
    # It should now contain dense + nested split + dense.
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


# --- validity rules: parity with ModelDef + the bare-dense carve-out --
#
# The rules ModelDef enforced on its flat layer list now live in
# LayerSeqDef / SplitDef. These check that a spec the old model
# rejected is still rejected after skip->split translation, that the
# new bare-dense carve-out for gated units is accepted, and that
# adjacency rules apply inside branches too.


# Specs ModelDef rejects that must stay rejected once parsed into the
# tree. Each has a skip form so it round-trips through translation.
_INVALID_SPECS = [
    # Norm can't be the first layer (top-level).
    "bits.1+bp|norm-dense.4.gelu",
    # Skip source right before a norm -> branch starts with norm.
    "bytes|dense.32.gelu-skip.1.add-norm-dense.32.gelu",
    # Merge point right after a suffix -> branch ends in a suffix.
    "bytes|skip.1.add-suffix.4-dense.32.gelu",
    # Two suffix-like layers adjacent.
    "bits.1+bp|suffix.2-suffix.2-dense.4.gelu",
    # Two norm adjacent.
    "bits.1+bp|dense.4.gelu-norm-norm-dense.4.gelu",
    # Norm follows a suffix.
    "bits.1+bp|suffix.2-norm-dense.4.gelu",
]


@pytest.mark.parametrize("spec", _INVALID_SPECS)
def test_invalid_specs_match_modeldef(spec):
    """Both representations agree the spec is invalid."""
    assert not ModelDef(spec, Precision.FP32).is_valid()
    assert not parse_model2(spec, Precision.FP32).is_valid()


_VALID_SPECS = [
    "bits.1+bp|dense.4.gelu-suffix.2-dense.4.tanh",
    "bytes|dense.32.gelu-skip.1.add-dense.32.gelu-dense.32.gelu",
    # Skip source before suffix is fine (start before, skip over).
    "bytes|dense.32.gelu-skip.2.add-suffix.2-dense.32.gelu-dense.16.gelu",
]


@pytest.mark.parametrize("spec", _VALID_SPECS)
def test_valid_specs_match_modeldef(spec):
    assert ModelDef(spec, Precision.FP32).is_valid()
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


def test_neighbors_cover_modeldef_non_skip():
    """Parity: every non-skip ModelDef neighbor is also a Model2
    neighbor (Model2 adds split/gate moves on top)."""
    spec = "bytes|dense.32.gelu-dense.16.gelu"
    md = ModelDef(spec, Precision.FP32)
    m2 = parse_model2(spec, Precision.FP32)
    md_arch = {
        n.spec for n in md.neighbors()
        if n.precision == md.precision and "skip." not in n.spec
    }
    m2_arch = _arch_neighbor_specs(m2)
    missing = md_arch - m2_arch
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
    for g in ("dense.4", "dense.4.gelu", "dense.4.relu", "dense.4.tanh"):
        assert f"bits.1+bp|split.mul(pass, {g})" in empty
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
