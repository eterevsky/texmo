"""spec_parser tests.

Verifies the layer-list grammar -- both simple dash-separated
chains (the legacy form) and the new `split.{op}(...)` syntax with
nesting, `pass` branches, and well-formed error messages.
"""
import pytest

from texmo.layers.dense import DenseDef
from texmo.layers.seq import LayerSeqDef
from texmo.layers.split import SplitDef
from texmo.spec_parser import (
    parse_layer_list, _split_at_depth_0)


# -- _split_at_depth_0 -------------------------------------------------


def test_split_at_depth_0_basic():
    assert _split_at_depth_0("a-b-c", "-") == ["a", "b", "c"]


def test_split_at_depth_0_skips_parens():
    """Delimiters inside parens don't count as splits."""
    assert _split_at_depth_0("a-b(c-d)-e", "-") == ["a", "b(c-d)", "e"]


def test_split_at_depth_0_nested_parens():
    assert _split_at_depth_0("a,b(c,d(e,f),g),h", ",") == [
        "a", "b(c,d(e,f),g)", "h"]


def test_split_at_depth_0_rejects_unbalanced_open():
    with pytest.raises(ValueError, match="unbalanced"):
        _split_at_depth_0("a(b", "-")


def test_split_at_depth_0_rejects_unbalanced_close():
    with pytest.raises(ValueError, match="unbalanced"):
        _split_at_depth_0("a)b", "-")


# -- parse_layer_list: simple (legacy) ---------------------------------


def test_parse_empty_returns_empty_list():
    assert parse_layer_list("", input_size=4) == []


def test_parse_whitespace_only_returns_empty():
    assert parse_layer_list("   ", input_size=4) == []


def test_parse_single_simple_layer():
    layers = parse_layer_list("dense.16.gelu", input_size=4)
    assert len(layers) == 1
    assert isinstance(layers[0], DenseDef)
    assert layers[0].size == 16


def test_parse_dash_chain_threads_input_size():
    """input_size of layer N+1 must equal the output size of layer N."""
    layers = parse_layer_list("dense.16.gelu-dense.8.tanh", input_size=4)
    assert layers[0].input_size == 4
    assert layers[0].size == 16
    assert layers[1].input_size == 16  # threaded from layer 0
    assert layers[1].size == 8


def test_parse_rejects_empty_layer_in_chain():
    """`dense.4--dense.4` shouldn't silently swallow the empty
    middle piece."""
    with pytest.raises(ValueError, match="empty layer"):
        parse_layer_list("dense.4.gelu--dense.4.gelu", input_size=4)


# -- parse_layer_list: split ----------------------------------------------


def test_parse_split_mul_with_pass_branch():
    layers = parse_layer_list(
        "split.mul(dense.16.gelu, pass)", input_size=8)
    assert len(layers) == 1
    s = layers[0]
    assert isinstance(s, SplitDef)
    assert s.op == 'mul'
    assert len(s.branches) == 2
    # Branch 0: one dense layer
    assert len(s.branches[0].layers) == 1
    assert isinstance(s.branches[0].layers[0], DenseDef)
    # Branch 1: empty (pass)
    assert s.branches[1].layers == []


def test_parse_split_two_dense_branches():
    layers = parse_layer_list(
        "split.add(dense.16.gelu, dense.32.tanh)", input_size=4)
    assert isinstance(layers[0], SplitDef)
    assert layers[0].op == 'add'
    assert layers[0].branches[0].size == 16
    assert layers[0].branches[1].size == 32


def test_parse_split_multi_layer_branch():
    """A branch that itself is a dash-separated chain."""
    layers = parse_layer_list(
        "split.mul(dense.16.gelu-dense.16.tanh, pass)",
        input_size=4)
    s = layers[0]
    assert len(s.branches[0].layers) == 2
    assert s.branches[0].layers[0].size == 16
    assert s.branches[0].layers[1].size == 16


def test_parse_split_inside_dash_chain():
    """Layers after a split continue with the split's output size."""
    layers = parse_layer_list(
        "dense.8.gelu-split.mul(dense.16.gelu, pass)-dense.4.gelu",
        input_size=4)
    assert len(layers) == 3
    assert isinstance(layers[1], SplitDef)
    # dense.8 -> split (output max(16, 8) = 16) -> dense.4
    assert layers[1].input_size == 8
    assert layers[1].size == 16  # mul broadcast pad
    assert layers[2].input_size == 16


def test_parse_split_nested():
    """split.mul nested inside another split's branch."""
    layers = parse_layer_list(
        "split.mul(split.add(dense.8.gelu, pass), pass)",
        input_size=4)
    outer = layers[0]
    assert isinstance(outer, SplitDef)
    inner = outer.branches[0].layers[0]
    assert isinstance(inner, SplitDef)
    assert inner.op == 'add'


def test_parse_split_three_way_accepted_by_parser():
    """The data structure and parser are multi-way ready; only
    SplitDef.is_valid rejects > 2 branches today."""
    layers = parse_layer_list(
        "split.mul(dense.8.gelu, pass, dense.8.tanh)", input_size=4)
    s = layers[0]
    assert len(s.branches) == 3
    assert not s.is_valid()  # 2-way constraint kicks in here


def test_parse_split_rejects_no_branches():
    with pytest.raises(ValueError, match="no branches"):
        parse_layer_list("split.mul()", input_size=4)


def test_parse_split_rejects_empty_branch_between_commas():
    """Empty branch is a clearer error than silently treating it as
    `pass` -- use `pass` explicitly if you want identity."""
    with pytest.raises(ValueError, match="empty branch"):
        parse_layer_list(
            "split.mul(dense.8.gelu, ,pass)", input_size=4)


def test_parse_split_rejects_unknown_op():
    with pytest.raises(ValueError, match="unknown split op"):
        parse_layer_list("split.bogus(dense.4.gelu, pass)", input_size=4)


def test_parse_split_rejects_missing_open_paren():
    """`split.mul` without parens is invalid -- handle the error in
    the simple-layer path (it'll fail to build_layer_def)."""
    with pytest.raises(Exception):  # ValueError from _build_layer_def
        parse_layer_list("split.mul", input_size=4)


def test_parse_split_rejects_unbalanced_parens():
    with pytest.raises(ValueError, match="unbalanced"):
        parse_layer_list("split.mul(dense.4.gelu, pass", input_size=4)


def test_parse_whitespace_tolerant():
    layers = parse_layer_list(
        "split.mul( dense.16.gelu ,  pass )", input_size=4)
    assert isinstance(layers[0], SplitDef)


def test_parse_both_branches_pass():
    """Edge case: split.mul(pass, pass) parses but is degenerate
    (input * input). Is_valid doesn't reject it today -- the search
    can discover it's useless on its own."""
    layers = parse_layer_list(
        "split.mul(pass, pass)", input_size=4)
    s = layers[0]
    assert s.branches[0].layers == []
    assert s.branches[1].layers == []


# -- Roundtrip --------------------------------------------------------


def test_str_roundtrip_simple():
    layers = parse_layer_list(
        "dense.16.gelu-dense.8.tanh", input_size=4)
    s = LayerSeqDef(layers, input_size=4)
    assert str(s) == "dense.16.gelu-dense.8.tanh"


def test_str_roundtrip_split():
    layers = parse_layer_list(
        "split.mul(dense.16.gelu, pass)", input_size=4)
    assert str(layers[0]) == "split.mul(dense.16.gelu, pass)"


def test_str_roundtrip_nested_split():
    spec = "split.mul(split.add(dense.8.gelu, pass), pass)"
    layers = parse_layer_list(spec, input_size=4)
    assert str(layers[0]) == spec


# -- skip-to-split translation ----------------------------------------


def test_translate_single_skip():
    """skip.1.add over one in-skip layer becomes
    split.add(that_layer, pass)."""
    from texmo.layers.skip import SkipDef
    from texmo.layers.split import SplitDef
    layers = parse_layer_list(
        "dense.4.gelu-skip.1.add-dense.4.gelu-dense.4.gelu",
        input_size=4)
    # 4 layers in, 3 out (the skip pseudo-layer and its 1 in-skip
    # layer fuse into one split).
    assert len(layers) == 3
    assert not any(isinstance(l, SkipDef) for l in layers)
    # Middle layer is the translated split.
    split = layers[1]
    assert isinstance(split, SplitDef)
    assert split.op == 'add'
    assert len(split.branches[0].layers) == 1
    assert split.branches[1].layers == []


def test_translate_skip_distance_3():
    """skip.3.add bundles three in-skip layers into the main branch."""
    from texmo.layers.split import SplitDef
    layers = parse_layer_list(
        "dense.4.gelu-skip.3.add"
        "-dense.4.gelu-dense.4.gelu-dense.4.gelu"
        "-dense.4.gelu",
        input_size=4)
    assert len(layers) == 3
    split = layers[1]
    assert isinstance(split, SplitDef)
    assert len(split.branches[0].layers) == 3


def test_translate_multiple_non_overlapping_skips():
    """Two skips in a row, separated by a non-skip layer, each
    translate independently."""
    from texmo.layers.split import SplitDef
    layers = parse_layer_list(
        "skip.1.add-dense.4.gelu"
        "-dense.4.gelu"
        "-skip.1.cat-dense.4.gelu-dense.4.gelu",
        input_size=4)
    # Input: 6 layers (2 skips + 4 normal). After translation:
    # split + 1 normal + split + 1 normal = 4.
    assert len(layers) == 4
    assert isinstance(layers[0], SplitDef) and layers[0].op == 'add'
    assert isinstance(layers[2], SplitDef) and layers[2].op == 'cat'


def test_translate_skip_at_position_0():
    """A skip as the very first stack layer translates fine -- the
    source comes from the input encoding's output."""
    from texmo.layers.split import SplitDef
    layers = parse_layer_list(
        "skip.1.add-dense.4.gelu-dense.4.gelu",
        input_size=4)
    assert len(layers) == 2
    assert isinstance(layers[0], SplitDef)


def test_translate_skip_at_end_merges_into_output():
    """skip.D where i+D+1 == len(layers) -- the split sits at the
    end of the chain and the model's output dense receives its
    output. Just need to confirm it parses."""
    from texmo.layers.split import SplitDef
    layers = parse_layer_list(
        "dense.4.gelu-skip.2.add-dense.4.gelu-dense.4.gelu",
        input_size=4)
    assert len(layers) == 2
    assert isinstance(layers[1], SplitDef)
    assert len(layers[1].branches[0].layers) == 2


def test_translate_crossing_skips_rejected():
    """Crossing spans (partial overlap, neither contains the other)
    can't be a tree. The parser raises ValueError so the caller can
    decide what to do. Here skip.3@0 spans [0,4) and skip.3@2 spans
    [2,6): they cross."""
    with pytest.raises(ValueError, match="crossing"):
        parse_layer_list(
            "skip.3.add-dense.4.gelu-skip.3.add"
            "-dense.4.gelu-dense.4.gelu-dense.4.gelu-dense.4.gelu",
            input_size=4)


def test_translate_nested_skips():
    """A skip fully inside another skip's span (laminar) becomes a
    nested split. skip.3@0 spans [0,4); skip.1@2 spans [2,4) -- the
    latter nests inside the former's main branch."""
    from texmo.layers.skip import SkipDef
    from texmo.layers.split import SplitDef
    layers = parse_layer_list(
        "skip.3.add-dense.4.gelu-skip.1.add"
        "-dense.4.gelu-dense.4.gelu",
        input_size=4)
    # Outer skip fuses positions 0..3 into one split; trailing dense
    # passes through -> 2 layers, no SkipDef survives anywhere.
    assert len(layers) == 2
    outer = layers[0]
    assert isinstance(outer, SplitDef) and outer.op == 'add'
    assert outer.branches[1].layers == []
    # Main branch holds [dense, inner_split].
    main = outer.branches[0].layers
    assert len(main) == 2
    inner = main[1]
    assert isinstance(inner, SplitDef) and inner.op == 'add'
    assert len(inner.branches[0].layers) == 1
    assert inner.branches[1].layers == []
    assert not any(isinstance(l, SkipDef) for l in main)


def test_translate_overshoot_skip_rejected():
    """A skip whose span runs off the end raises -- the original
    model_def's _skip_overshoots set this to is_valid=False; we
    promote it to a parse error since the translation can't
    proceed."""
    with pytest.raises(ValueError, match="overshoots"):
        parse_layer_list(
            "dense.4.gelu-skip.5.add-dense.4.gelu", input_size=4)


def test_translate_skip_inside_split_branch():
    """Skip inside a split branch is translated by the recursion --
    `_parse_split` calls `parse_layer_list` on each branch, which
    applies the same `_translate_skips` pass."""
    from texmo.layers.skip import SkipDef
    from texmo.layers.split import SplitDef
    layers = parse_layer_list(
        "split.mul("
        "dense.4.gelu-skip.1.add-dense.4.gelu-dense.4.gelu, "
        "pass)",
        input_size=4)
    outer = layers[0]
    inner_layers = outer.branches[0].layers
    assert not any(isinstance(l, SkipDef) for l in inner_layers)
    assert isinstance(inner_layers[1], SplitDef)
    assert inner_layers[1].op == 'add'
