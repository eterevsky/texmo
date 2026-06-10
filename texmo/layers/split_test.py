"""SplitDef + SplitJax tests.

Builds SplitDefs directly in Python (the parser doesn't know
`split.op(...)` syntax yet -- that's the next migration item).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from texmo.layers.conv import ConvDef
from texmo.layers.dense import DenseDef
from texmo.layers.seq import LayerSeqDef
from texmo.layers.split import (
    SplitDef, SplitJax, _align_branch_outputs, _merge_all)


def _seq(layers, input_size):
    return LayerSeqDef(layers, input_size=input_size)


def _dense(size, act, input_size):
    return DenseDef(size, input_size=input_size, activation=act)


# -- SplitDef properties ----------------------------------------------


def test_def_str_2_way():
    s = SplitDef(
        'mul',
        [_seq([_dense(8, 'gelu', 4)], 4), _seq([], 4)],
        input_size=4,
    )
    assert str(s) == "split.mul(dense.8.gelu, pass)"


def test_def_str_both_pass():
    s = SplitDef('mul', [_seq([], 4), _seq([], 4)], input_size=4)
    assert str(s) == "split.mul(pass, pass)"


def test_def_size_mul_max():
    s = SplitDef(
        'mul',
        [_seq([_dense(8, 'gelu', 4)], 4),
         _seq([_dense(16, 'gelu', 4)], 4)],
        input_size=4,
    )
    assert s.size == 16  # max(8, 16) -- broadcast pad


def test_def_size_cat_sum():
    s = SplitDef(
        'cat',
        [_seq([_dense(8, 'gelu', 4)], 4),
         _seq([_dense(16, 'gelu', 4)], 4)],
        input_size=4,
    )
    assert s.size == 24  # 8 + 16


def test_def_size_add_max():
    s = SplitDef(
        'add',
        [_seq([_dense(8, 'gelu', 4)], 4),
         _seq([_dense(16, 'gelu', 4)], 4)],
        input_size=4,
    )
    assert s.size == 16


def test_def_size_pass_branch_inherits_input():
    """A `pass` branch outputs the split's input (size = input_size)."""
    s = SplitDef(
        'mul',
        [_seq([_dense(8, 'gelu', 4)], 4), _seq([], 4)],
        input_size=4,
    )
    # mul: max(8, 4) = 8 (broadcast)
    assert s.size == 8


def test_def_length_takes_max():
    """Branches consume independently; the split's length is the
    longest reach so the enclosing total_padding budget is right."""
    s = SplitDef(
        'mul',
        # branch A: dense (no extras consumed)
        [_seq([_dense(8, 'gelu', 4)], 4),
         # branch B: conv.4 (consumes 3) + dense (consumes 0) = 3
         _seq([ConvDef(4, input_size=4),
               _dense(4, 'gelu', 4)], 4)],
        input_size=4,
    )
    # branch A length = 1; branch B length = 1 + 3 = 4.
    assert s.length == 4


def test_def_num_weights_is_sum_over_branches():
    branch_a = _seq([_dense(8, 'gelu', 4)], 4)
    branch_b = _seq([_dense(16, 'gelu', 4)], 4)
    s = SplitDef('mul', [branch_a, branch_b], input_size=4)
    assert s.num_weights == branch_a.num_weights + branch_b.num_weights


def test_def_rejects_unknown_op():
    with pytest.raises(ValueError):
        SplitDef('bogus', [_seq([], 4), _seq([], 4)], input_size=4)


# -- 2-way constraint enforced via is_valid -----------------------------


def test_is_valid_two_way_ok():
    s = SplitDef(
        'mul',
        [_seq([_dense(8, 'gelu', 4)], 4),
         _seq([_dense(8, 'gelu', 4)], 4)],
        input_size=4,
    )
    assert s.is_valid()


def test_is_valid_rejects_three_way():
    """The data structure supports N branches; the validity rule
    is what holds us to 2-way for now."""
    s = SplitDef(
        'mul',
        [_seq([_dense(8, 'gelu', 4)], 4),
         _seq([_dense(8, 'gelu', 4)], 4),
         _seq([_dense(8, 'gelu', 4)], 4)],
        input_size=4,
    )
    assert not s.is_valid()


def test_is_valid_rejects_one_way():
    s = SplitDef('mul', [_seq([_dense(8, 'gelu', 4)], 4)], input_size=4)
    assert not s.is_valid()


def test_is_valid_rejects_branch_input_size_mismatch():
    bad_branch = _seq([_dense(8, 'gelu', 99)], 99)
    good_branch = _seq([_dense(8, 'gelu', 4)], 4)
    s = SplitDef('mul', [good_branch, bad_branch], input_size=4)
    assert not s.is_valid()


# -- SplitJax: shapes, forward / step agree -----------------------------


def _make_jax(split: SplitDef, seed=0, dtype=jnp.float32):
    layer = split.build_jax(dtype)
    weights = layer.init_weights(jax.random.PRNGKey(seed))
    return layer, weights


def test_jax_init_state_per_branch():
    s = SplitDef(
        'mul',
        [_seq([_dense(8, 'gelu', 4)], 4), _seq([], 4)],
        input_size=4,
    )
    layer, _ = _make_jax(s)
    state = layer.init_state()
    assert len(state) == 2


def test_jax_forward_shape_mul_same_size():
    s = SplitDef(
        'mul',
        [_seq([_dense(8, 'gelu', 4)], 4),
         _seq([_dense(8, 'gelu', 4)], 4)],
        input_size=4,
    )
    layer, weights = _make_jax(s)
    inputs = jax.random.normal(
        jax.random.PRNGKey(1), (2, 7, 4), dtype=jnp.float32)
    out = layer.forward(weights, inputs)
    assert out.shape == (2, 7, 8)


def test_jax_forward_shape_cat():
    s = SplitDef(
        'cat',
        [_seq([_dense(8, 'gelu', 4)], 4),
         _seq([_dense(16, 'gelu', 4)], 4)],
        input_size=4,
    )
    layer, weights = _make_jax(s)
    inputs = jax.random.normal(
        jax.random.PRNGKey(1), (2, 5, 4), dtype=jnp.float32)
    out = layer.forward(weights, inputs)
    assert out.shape == (2, 5, 24)


def test_jax_forward_shape_mul_broadcast_pads():
    """mul on branches of different sizes pads the smaller -- legacy
    `_merge_add`-style behaviour, matched by `_merge_mul`."""
    s = SplitDef(
        'mul',
        [_seq([_dense(4, 'gelu', 4)], 4),
         _seq([_dense(16, 'gelu', 4)], 4)],
        input_size=4,
    )
    layer, weights = _make_jax(s)
    inputs = jax.random.normal(
        jax.random.PRNGKey(2), (1, 3, 4), dtype=jnp.float32)
    out = layer.forward(weights, inputs)
    assert out.shape == (1, 3, 16)


def test_jax_forward_aligns_branches_with_different_consumed():
    """Branch A (length-preserving dense) and branch B (conv.4,
    consumes 3) produce different output lengths from the same
    input. The split must trim branch A's leading 3 positions so
    they align with branch B before the merge."""
    s = SplitDef(
        'mul',
        [_seq([_dense(4, 'gelu', 4)], 4),
         _seq([ConvDef(4, input_size=4)], 4)],
        input_size=4,
    )
    layer, weights = _make_jax(s)
    # 10 input positions; branch B consumes 3 -> output 7 positions.
    inputs = jax.random.normal(
        jax.random.PRNGKey(4), (1, 10, 4), dtype=jnp.float32)
    out = layer.forward(weights, inputs)
    assert out.shape == (1, 7, 4)


def test_jax_step_shape():
    s = SplitDef(
        'mul',
        [_seq([_dense(8, 'gelu', 4)], 4), _seq([], 4)],
        input_size=4,
    )
    layer, weights = _make_jax(s)
    state = layer.init_state()
    x = jax.random.normal(jax.random.PRNGKey(5), (4,), dtype=jnp.float32)
    new_state, out = layer.step(weights, state, x)
    assert out.shape == (8,)
    assert len(new_state) == 2


def test_jax_forward_matches_step_no_length_consuming_layers():
    """When no branch consumes positions, every step's output equals
    the corresponding forward output position."""
    s = SplitDef(
        'mul',
        [_seq([_dense(8, 'gelu', 4)], 4),
         _seq([_dense(8, 'gelu', 4)], 4)],
        input_size=4,
    )
    layer, weights = _make_jax(s)
    inputs = jax.random.normal(
        jax.random.PRNGKey(7), (1, 6, 4), dtype=jnp.float32)
    fwd = layer.forward(weights, inputs)

    state = layer.init_state()
    for t in range(inputs.shape[1]):
        state, step_out = layer.step(weights, state, inputs[0, t])
        np.testing.assert_allclose(
            np.asarray(fwd[0, t]), np.asarray(step_out), atol=1e-5)


def test_jax_pass_branch_acts_as_identity_in_mul():
    """split.mul(pass, dense) should equal mul(input, dense(input))
    elementwise on the overlapping channels."""
    s = SplitDef(
        'mul',
        [_seq([], 4),
         _seq([_dense(4, 'gelu', 4)], 4)],
        input_size=4,
    )
    layer, weights = _make_jax(s)
    inputs = jax.random.normal(
        jax.random.PRNGKey(8), (1, 4, 4), dtype=jnp.float32)
    out = layer.forward(weights, inputs)
    # Compute what we expect by hand: dense(input) * input,
    # elementwise on all 4 channels (sizes match).
    dense_layer = layer.branches[1].layers[0]
    dense_out = dense_layer.forward(weights[1][0], inputs)
    expected = inputs * dense_out
    np.testing.assert_allclose(
        np.asarray(out), np.asarray(expected), atol=1e-5)


def test_jax_weight_count_matches_def():
    s = SplitDef(
        'mul',
        [_seq([_dense(8, 'gelu', 4)], 4),
         _seq([_dense(16, 'gelu', 4)], 4)],
        input_size=4,
    )
    _, weights = _make_jax(s)
    actual = sum(w.size for w in jax.tree.leaves(weights))
    assert actual == s.num_weights


# -- Multi-way design check (parser will accept, validity rejects) -----


def test_merge_all_handles_n_branches():
    """The fold infrastructure is multi-way ready; only is_valid
    enforces 2-way for now. This test pokes the fold directly to
    confirm the structure is sound for the day we lift the cap."""
    outs = [
        jnp.array([[1.0, 2.0]]),
        jnp.array([[3.0, 4.0]]),
        jnp.array([[5.0, 6.0]]),
    ]
    np.testing.assert_allclose(
        np.asarray(_merge_all('mul', outs)),
        np.array([[15.0, 48.0]]),  # 1*3*5, 2*4*6
    )
    np.testing.assert_allclose(
        np.asarray(_merge_all('add', outs)),
        np.array([[9.0, 12.0]]),
    )
    np.testing.assert_allclose(
        np.asarray(_merge_all('cat', outs)),
        np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
    )


def test_align_branch_outputs_drops_leading_positions():
    """The alignment helper is what makes asymmetric-consumed branches
    mergeable. Verify directly."""
    # Branch 0 consumed 3, branch 1 consumed 1, branch 2 consumed 0.
    outs = [
        jnp.zeros((1, 7, 2)),    # T=7 from T=10 input, 3 consumed
        jnp.zeros((1, 9, 2)),    # T=9 from T=10, 1 consumed
        jnp.zeros((1, 10, 2)),   # T=10 from T=10, 0 consumed
    ]
    aligned = _align_branch_outputs(outs, [3, 1, 0])
    assert aligned[0].shape == (1, 7, 2)
    assert aligned[1].shape == (1, 7, 2)
    assert aligned[2].shape == (1, 7, 2)
