import json
import random

import pytest

from texmo.common import INF
from texmo.configuration import (
    Configuration,
    DecayType,
    Template,
    TemplateEntry,
    TemplateSet,
    conf_neighbors,
    default_from_template,
)
from texmo.precision import Precision
from texmo.spec_parser import parse_model2


def _md(spec):
    return parse_model2(spec, Precision.FP32)


def _conf(spec, lr=0.25, length=128, batch=256, steps=256, decay=1.0):
    return Configuration(
        model=_md(spec), lr=lr, length=length, batch=batch,
        steps=steps, decay=decay,
    )


def test_conf_neighbors():
    template = Template(
        spec=r"bytes\|(dense|rnn)\.\d+\.gelu(-suffix\.\d+)?",
        lr=None,
        length=(128, 128),
        batch=(128, 256),
        steps=(256, 256),
        max_weights=(32, INF),
        precision=[Precision.FP32],
    )

    conf = _conf("bytes|dense.1.gelu", decay=1 / 32)
    neighbors = set(conf_neighbors(conf, template))
    assert neighbors == {
        conf.replace(model=_md("bytes|dense.2.gelu")),
        conf.replace(model=_md("bytes|rnn.1.gelu")),
        conf.replace(model=_md("bytes|dense.1.gelu-suffix.2")),
        conf.replace(lr=0.125),
        conf.replace(lr=0.5),
        conf.replace(decay=1 / 16),
        conf.replace(decay=1 / 64),
        conf.replace(batch=128),
        # Toggle to cosine snaps decay to 1.
        conf.replace(cosine=True, decay=1.0),
    }


def test_dense_layer_neighbor():
    template = Template(
        spec=r"bytes\|(dense|rnn)\.\d+\.\w+",
        lr=(0.25, 0.25),
        length=(128, 128),
        batch=(128, 128),
        steps=(256, 256),
        max_weights=(32, INF),
        precision=[Precision.FP32],
        decay_types=['none', 'cosine'],
    )

    conf = _conf("bytes|dense.1.gelu", batch=128, decay=1)
    neighbors = set(conf_neighbors(conf, template))
    assert neighbors == {
        conf.replace(model=_md("bytes|dense.2.gelu")),
        conf.replace(model=_md("bytes|rnn.1.gelu")),
        # The activation toggle: every other search-valid activation
        # (relu retired). The bare form is a neighbor too but this
        # template's regex requires an activation suffix, so it's
        # filtered out before it gets here.
        conf.replace(model=_md("bytes|dense.1.tanh")),
        conf.replace(model=_md("bytes|dense.1.silu")),
        conf.replace(cosine=True),
    }


def test_conf_neighbors_suffix():
    template = Template(
        spec=r"bytes\|suffix\.\d+",
        lr=None,
        length=(128, 128),
        batch=(128, 256),
        steps=(256, 256),
        max_weights=(32, INF),
        precision=[Precision.FP32],
    )

    conf = _conf("bytes|suffix.2")
    neighbors = set(conf_neighbors(conf, template))
    assert neighbors == {
        conf.replace(model=_md("bytes|suffix.4")),
        conf.replace(lr=0.125),
        conf.replace(lr=0.5),
        conf.replace(batch=128),
        conf.replace(decay=1 / 2),
        conf.replace(cosine=True),
    }


def test_conf_neighbors_suffix_exact():
    template = Template(
        spec='bytes|suffix.2',
        lr=None,
        length=(128, 128),
        batch=(128, 256),
        steps=(256, 256),
        max_weights=(32, INF),
        precision=[Precision.FP32],
    )

    conf = _conf("bytes|suffix.2")
    neighbors = set(conf_neighbors(conf, template))
    assert neighbors == {
        conf.replace(lr=0.125),
        conf.replace(lr=0.5),
        conf.replace(batch=128),
        conf.replace(decay=1 / 2),
        conf.replace(cosine=True),
    }


def test_conf_neighbors_precision():
    template = Template(
        spec=r"bytes\|dense\.\d+\.relu",
        lr=(0.25, 0.25),
        length=(128, 128),
        batch=(128, 128),
        steps=(256, 256),
        max_weights=(32, INF),
        precision=[Precision.FP32, Precision.FP16],
        decay_types=['none', 'cosine'],
    )

    conf = _conf("bytes|dense.1.relu", batch=128, decay=1)
    neighbors = set(conf_neighbors(conf, template))
    # Should include FP16 precision neighbor
    fp16_conf = conf.replace(
        model=parse_model2("bytes|dense.1.relu", Precision.FP16))
    assert fp16_conf in neighbors
    # FP64 not in template precision list, should be excluded
    fp64_conf = conf.replace(
        model=parse_model2("bytes|dense.1.relu", Precision.FP64))
    assert fp64_conf not in neighbors


def test_conf_neighbors_input():
    template = Template(
        spec=r"(bytes|bits\.\d+[\w.+]*)\|dense\.\d+\.gelu",
        lr=(0.25, 0.25),
        length=(128, 128),
        batch=(128, 128),
        steps=(256, 256),
        max_weights=None,
        precision=[Precision.FP32],
        decay_types=['none', 'cosine'],
    )

    conf = _conf("bytes|dense.1.gelu", batch=128, decay=1)
    neighbors = set(conf_neighbors(conf, template))
    specs = {str(n.model) for n in neighbors}
    assert "bits.4.oh+bp|dense.1.gelu" in specs


def test_template_exact_spec_match():
    """When spec is a valid model, Template uses exact match, not regex.

    bits.1+bp contains '+' which is a regex special char — if treated as
    regex it would match wrong specs. Exact match ensures only the
    precise spec is accepted.
    """
    template = Template(
        spec="bits.1+bp|dense.1.relu",
        lr=None,
        length=None,
        batch=None,
        steps=None,
        max_weights=None,
        precision=[Precision.FP32],
    )

    assert template.spec == "bits.1+bp|dense.1.relu"
    assert template.regex is None

    conf_match = _conf("bits.1+bp|dense.1.relu")
    assert template.match(conf_match)

    # Should NOT match a different spec, even if '+' were treated as regex
    conf_no_match = _conf("bits.1|dense.1.relu")
    assert not template.match(conf_no_match)

    conf_bp2 = _conf("bits.2+bp|dense.1.relu")
    assert not template.match(conf_bp2)


def test_conf_is_valid():
    conf = _conf("bytes|rnn.64.tanh-dense.32.gelu", decay=1 / 32)
    assert conf.is_valid()


def test_conf_not_valid_no_activation():
    conf = _conf("bytes|dense.32", decay=1 / 32)
    assert not conf.is_valid()


def test_cosine_requires_decay_one():
    conf_ok = _conf("bytes|dense.32.gelu", decay=1.0).replace(cosine=True)
    assert conf_ok.is_valid()
    conf_bad = _conf("bytes|dense.32.gelu", decay=0.5).replace(cosine=True)
    assert not conf_bad.is_valid()


def test_cosine_neighbor_toggle_from_exp_decay():
    """Toggling cosine on snaps decay to 1, regardless of current decay."""
    template = Template(
        spec="bytes|dense.32.gelu",
        lr=(0.25, 0.25),
        length=(128, 128),
        batch=(128, 128),
        steps=(256, 256),
        max_weights=(32, INF),
        precision=[Precision.FP32],
    )
    conf = _conf("bytes|dense.32.gelu", batch=128, decay=0.5)
    neighbors = set(conf_neighbors(conf, template))
    assert conf.replace(cosine=True, decay=1.0) in neighbors
    # Direct (cosine=True, decay=0.5) is invalid and must NOT be a neighbor.
    assert conf.replace(cosine=True) not in neighbors


def test_cosine_neighbor_toggle_back_to_exp_decay():
    """From cosine=True the toggle goes to (cosine=False, decay=1)."""
    template = Template(
        spec="bytes|dense.32.gelu",
        lr=(0.25, 0.25),
        length=(128, 128),
        batch=(128, 128),
        steps=(256, 256),
        max_weights=(32, INF),
        precision=[Precision.FP32],
    )
    conf = _conf("bytes|dense.32.gelu", batch=128, decay=1.0).replace(
        cosine=True)
    neighbors = set(conf_neighbors(conf, template))
    assert conf.replace(cosine=False) in neighbors
    # exp-decay walk from cosine=True is suppressed; the off-toggle
    # gets us to decay=1, and the regular walk takes over from there.
    assert conf.replace(cosine=False, decay=0.5) not in neighbors


def test_decay_type_property():
    none = _conf("bytes|dense.32.gelu", decay=1.0)
    assert none.decay_type == DecayType.NONE
    exp = _conf("bytes|dense.32.gelu", decay=0.5)
    assert exp.decay_type == DecayType.EXP
    cosine = none.replace(cosine=True)
    assert cosine.decay_type == DecayType.COSINE


def _meta_template(decay_types=None) -> Template:
    return Template(
        spec="bytes|dense.32.gelu",
        lr=(0.25, 0.25),
        length=(128, 128),
        batch=(128, 128),
        steps=(256, 256),
        max_weights=(32, INF),
        precision=[Precision.FP32],
        decay_types=decay_types,
    )


def test_default_from_template_steps_above_default():
    """steps=(4, INF) used to OverflowError in pick_default(2) because
    the geometric-mean fallback tried log2(inf). Now it anchors to min."""
    t = Template(
        spec=None, precision=[Precision.FP32],
        lr=None, length=None, batch=None,
        steps=(4, INF),
        max_weights=(2, INF),
        decay_types=[DecayType.COSINE],
    )
    conf = default_from_template(t, spec="bits.1+bp|dense.4.gelu")
    # When the requested default (2) is below the template's min (4),
    # we anchor to min rather than blowing up on geometric mean.
    assert conf.steps == 4


def test_template_num_layers_default_accepts_any_depth():
    """Without an explicit num_layers, every depth matches."""
    t = Template(
        spec=None, precision=list(Precision),
        lr=None, length=None, batch=None, steps=None,
        max_weights=(2, INF),
    )
    assert t.match_model(_md("bits.1+bp|"))
    assert t.match_model(_md("bits.1+bp|dense.4.gelu"))
    assert t.match_model(
        _md("bits.1+bp|dense.4.gelu-mgru.4-dense.4.gelu"))


def test_template_num_layers_filters_by_depth():
    """num_layers=(1, 2) rejects 0-layer and 3-layer specs."""
    t = Template(
        spec=None, precision=list(Precision),
        lr=None, length=None, batch=None, steps=None,
        max_weights=(2, INF),
        num_layers=(1, 2),
    )
    # 0 layers: rejected.
    assert not t.match_model(_md("bits.1+bp|"))
    # 1 layer: accepted.
    assert t.match_model(_md("bits.1+bp|dense.4.gelu"))
    # 2 layers: accepted.
    assert t.match_model(_md("bits.1+bp|dense.4.gelu-mgru.4"))
    # 3 layers: rejected.
    assert not t.match_model(
        _md("bits.1+bp|dense.4.gelu-mgru.4-dense.4.gelu"))


def test_template_num_layers_counts_all_pipeline_layers():
    """suffix/norm/skip count as layers (they appear in the spec)."""
    t = Template(
        spec=None, precision=list(Precision),
        lr=None, length=None, batch=None, steps=None,
        max_weights=(2, INF),
        num_layers=(2, 2),
    )
    # 1 weighted layer + 1 suffix = 2 layers total: accepted.
    assert t.match_model(_md("bits.1+bp|suffix.2-dense.4.gelu"))
    # 1 weighted layer alone: 1 layer, rejected.
    assert not t.match_model(_md("bits.1+bp|dense.4.gelu"))


def test_template_match_filters_by_decay_type():
    none = _conf("bytes|dense.32.gelu", batch=128, decay=1.0)
    exp = _conf("bytes|dense.32.gelu", batch=128, decay=0.5)
    cosine = none.replace(cosine=True)

    t_all = _meta_template()
    assert t_all.match(none) and t_all.match(exp) and t_all.match(cosine)

    assert _meta_template([DecayType.NONE]).match(none)
    assert not _meta_template([DecayType.NONE]).match(exp)
    assert not _meta_template([DecayType.NONE]).match(cosine)

    assert not _meta_template([DecayType.EXP]).match(none)
    assert _meta_template([DecayType.EXP]).match(exp)
    assert not _meta_template([DecayType.EXP]).match(cosine)

    assert not _meta_template([DecayType.COSINE]).match(none)
    assert not _meta_template([DecayType.COSINE]).match(exp)
    assert _meta_template([DecayType.COSINE]).match(cosine)


def test_template_match_empty_decay_types_matches_nothing():
    t_empty = _meta_template([])
    none = _conf("bytes|dense.32.gelu", batch=128, decay=1.0)
    exp = _conf("bytes|dense.32.gelu", batch=128, decay=0.5)
    cosine = none.replace(cosine=True)
    assert not t_empty.match(none)
    assert not t_empty.match(exp)
    assert not t_empty.match(cosine)


def test_neighbor_walk_excludes_cosine_when_not_in_decay_types():
    """With cosine disabled, the toggle on cosine should not yield."""
    t = _meta_template([DecayType.NONE, DecayType.EXP])
    none = _conf("bytes|dense.32.gelu", batch=128, decay=1.0)
    neighbors = set(conf_neighbors(none, t))
    assert none.replace(cosine=True) not in neighbors


def test_neighbor_walk_excludes_no_decay_when_not_in_decay_types():
    """With 'none' disabled, walking exp decay UP to decay=1.0 stops short."""
    t = _meta_template([DecayType.EXP, DecayType.COSINE])
    exp = _conf("bytes|dense.32.gelu", batch=128, decay=0.5)
    neighbors = set(conf_neighbors(exp, t))
    # decay/2 is still exp -> yielded.
    assert exp.replace(decay=0.25) in neighbors
    # decay*2 = 1.0 is type NONE -> filtered out.
    assert exp.replace(decay=1.0) not in neighbors
    # Cosine toggle still yields (cosine=True, decay=1.0).
    assert exp.replace(cosine=True, decay=1.0) in neighbors


def test_neighbor_walk_excludes_exp_when_not_in_decay_types():
    """With 'exp' disabled, the decay walk from a no-decay conf yields nothing."""
    t = _meta_template([DecayType.NONE, DecayType.COSINE])
    none = _conf("bytes|dense.32.gelu", batch=128, decay=1.0)
    neighbors = set(conf_neighbors(none, t))
    # decay/2 = 0.5 would be exp -> filtered.
    assert none.replace(decay=0.5) not in neighbors
    # Cosine toggle still works.
    assert none.replace(cosine=True) in neighbors


def test_default_from_template_rejects_invalid_spec():
    """An explicitly-provided --default-spec must pass model.is_valid()."""
    t = _meta_template([DecayType.COSINE])
    # msr requires dim >= 2 (RoPE needs at least one pair); msr.1.1 fails.
    with pytest.raises(ValueError, match="not a valid model"):
        default_from_template(t, spec="bits.1+bp|msr.1.1")


def test_default_from_template_decay_types():
    spec = "bytes|dense.32.gelu"
    none_conf = default_from_template(_meta_template([DecayType.NONE]), spec)
    assert none_conf.decay_type == DecayType.NONE

    exp_conf = default_from_template(_meta_template([DecayType.EXP]), spec)
    assert exp_conf.decay_type == DecayType.EXP

    cosine_conf = default_from_template(
        _meta_template([DecayType.COSINE]), spec)
    assert cosine_conf.decay_type == DecayType.COSINE


def test_template_from_form_decay_types_checkboxes():
    """Empty form values mean unchecked; only checked types show up."""
    params = {
        'lr': '', 'length': '', 'batch': '', 'steps': '',
        'weights': '',
        'fp32': 'on',
        'decay_none': 'on',
        'decay_cosine': 'on',
        # decay_exp absent → unchecked
    }
    t = Template.from_form(params)
    assert t.decay_types == [DecayType.NONE, DecayType.COSINE]


def test_template_from_form_ignores_a_spec_in_the_params():
    """The index form has no spec field -- specs live in the sub-search
    rows. A stray `spec` param must not sneak into the base template;
    the compare page passes its per-column spec explicitly instead."""
    params = {
        'lr': '', 'length': '', 'batch': '', 'steps': '', 'weights': '',
        'fp32': 'on', 'decay_none': 'on',
        'spec': r'bytes\|rnn\..*',
    }
    assert Template.from_form(params).spec_pattern is None
    passed = Template.from_form(params, spec=r'bytes\|rnn\..*')
    assert passed.spec_pattern == r'bytes\|rnn\..*'


def test_default_from_template_finds_emb_models():
    """A template that only matches tied-codec specs gets a tied
    default -- the candidate sweep includes emb inputs with chains
    sized to the table width."""
    t = Template(
        spec=r"bits\.\d+\.emb\.\d+\|.*",
        precision=[Precision.FP32],
        lr=None, length=None, batch=None, steps=None,
        max_weights=(2, INF),
    )
    conf = default_from_template(t, spec=None)
    assert str(conf.model).startswith("bits.1.emb.1|")
    assert conf.model.is_valid()

    # Width-specific template: the sweep reaches larger tables too.
    t = Template(
        spec=r"bytes\.emb\.16\|.*",
        precision=[Precision.FP32],
        lr=None, length=None, batch=None, steps=None,
        max_weights=(2, INF),
    )
    conf = default_from_template(t, spec=None)
    assert str(conf.model).startswith("bytes.emb.16|")

    # One-hot templates keep their historical default (one-hot
    # candidates come first).
    t = Template(
        spec=None,
        precision=[Precision.FP32],
        lr=None, length=None, batch=None, steps=None,
        max_weights=(2, INF),
    )
    conf = default_from_template(t, spec=None)
    assert str(conf.model) == "bits.1+bp|"


# --- template sets ---------------------------------------------------

_ATTN_SPEC = "bytes|split.add(norm-attn.8.2.8, pass)"
_ATTN_REGEX = r".*\|split\.add\(.*attn.*"


def _base_template(spec=None, num_layers=None) -> Template:
    return Template(
        spec=spec,
        precision=[Precision.FP32],
        lr=(1 / 256, 1), length=(8, 1024), batch=(1, 256),
        steps=(2, 8192), max_weights=(16, 1 << 20),
        num_layers=num_layers,
    )


def _templates(*entries) -> TemplateSet:
    """Build a set from `(name, share, regex, default_spec)` tuples via
    the JSON form, i.e. the same path the CLI and the web form use."""
    items = []
    for name, share, regex, default_spec in entries:
        item = {'name': name, 'share': share}
        if regex:
            item['regex'] = regex
        if default_spec:
            item['default_spec'] = default_spec
        items.append(item)
    return TemplateSet.from_json(json.dumps(items), _base_template())


def test_with_spec_shares_every_other_bound():
    """Only the spec filter varies per entry: the copy keeps the base
    bounds objects and leaves the base untouched."""
    base = _base_template()
    sub = base.with_spec(_ATTN_REGEX)
    assert sub is not base
    assert sub.regex.pattern == _ATTN_REGEX
    assert base.regex is None and base.spec is None
    for attr in ('lr', 'length', 'batch', 'steps', 'max_weights',
                 'num_layers'):
        assert getattr(sub, attr) is getattr(base, attr)
    assert sub.precision is base.precision
    assert sub.decay_types is base.decay_types
    # An unchanged filter hands back the same object.
    assert base.with_spec(None) is base
    assert sub.with_spec(_ATTN_REGEX) is sub
    # A literal spec is canonicalized, so its round trip is identity too.
    literal = base.with_spec("bytes|dense.4.gelu")
    assert literal.spec == "bytes|dense.4.gelu"
    assert literal.with_spec("bytes|dense.4.gelu") is literal


def test_template_set_single_wraps_the_whole_template():
    """No --templates: one `main` entry over the base template, with
    the base default. Nothing about selection can change."""
    base = _base_template(spec=r"bytes\|.*")
    default = default_from_template(base, spec=None)
    ts = TemplateSet.single(base, spec=base.spec_pattern, default=default)
    assert ts.is_single and ts.names == ['main']
    entry = ts.entries[0]
    assert entry.template is base       # identical, not a copy
    assert entry.default is default
    # Without a spec the entry is unrestricted -- which is what a
    # server with no --spec and no --templates runs.
    plain = TemplateSet.single(_base_template(), default=default)
    assert plain.entries[0].spec is None
    assert plain.entries[0].template.spec_pattern is None
    assert ts.nominal_share(entry) == 100.0
    # A single-entry draw is deterministic AND consumes no randomness,
    # so the RNG stream of a select is exactly what it was before
    # template sets existed.
    state = random.getstate()
    assert ts.draw() is entry
    assert random.getstate() == state


def test_template_set_entries_share_bounds_with_the_base():
    ts = _templates(
        ('main', 60, None, None),
        ('attn', 40, _ATTN_REGEX, _ATTN_SPEC),
    )
    base = _base_template()
    for entry in ts:
        assert entry.template.max_weights.max == base.max_weights.max
        assert entry.template.steps.min == base.steps.min
        assert entry.template.precision == base.precision
        assert entry.template.decay_types == base.decay_types
    # ...and differ only in the spec filter.
    assert ts.by_name('main').template.spec_pattern is None
    assert ts.by_name('attn').template.spec_pattern == _ATTN_REGEX


def test_template_set_default_from_the_sub_template():
    """No default_spec: the entry derives one from its own template."""
    ts = _templates(('emb', 100, r"bits\.\d+\.emb\.\d+\|.*", None))
    entry = ts.entries[0]
    assert str(entry.default.model).startswith('bits.1.emb.1|')
    assert entry.template.match_model(entry.default.model)


def test_template_set_min_layers_from_the_default():
    ts = _templates(
        ('main', 60, None, None),
        ('attn', 40, _ATTN_REGEX, _ATTN_SPEC),
    )
    # 'bits.1+bp|' has no hidden layers; a transformer block has 3.
    assert ts.by_name('main').min_layers == 0
    assert ts.by_name('attn').min_layers == 3


def test_template_set_draw_honors_shares():
    ts = _templates(
        ('main', 60, None, None),
        ('attn', 30, _ATTN_REGEX, _ATTN_SPEC),
        ('emb', 10, r"bits\.\d+\.emb\.\d+\|.*", None),
    )
    random.seed(20260808)
    counts = {name: 0 for name in ts.names}
    n = 30000
    for _ in range(n):
        counts[ts.draw().name] += 1
    for name, expected in (('main', 0.6), ('attn', 0.3), ('emb', 0.1)):
        assert abs(counts[name] / n - expected) < 0.02, counts


def test_template_set_shares_are_normalized():
    """Shares need not sum to 100 -- only their ratios matter."""
    ts = _templates(
        ('main', 3, None, None),
        ('attn', 1, _ATTN_REGEX, _ATTN_SPEC),
    )
    assert ts.nominal_share(ts.by_name('main')) == 75.0
    assert ts.nominal_share(ts.by_name('attn')) == 25.0
    random.seed(7)
    n = 20000
    hits = sum(1 for _ in range(n) if ts.draw().name == 'attn')
    assert abs(hits / n - 0.25) < 0.02


def test_template_set_zero_share_entry_is_never_drawn():
    ts = _templates(
        ('main', 100, None, None),
        ('attn', 0, _ATTN_REGEX, _ATTN_SPEC),
    )
    random.seed(11)
    assert all(ts.draw().name == 'main' for _ in range(500))


def test_template_set_json_round_trip():
    ts = _templates(
        ('main', 60, None, None),
        ('attn', 40, _ATTN_REGEX, _ATTN_SPEC),
    )
    again = TemplateSet.from_json(ts.to_json(), _base_template())
    assert again.names == ts.names
    for a, b in zip(again, ts):
        assert a.share == b.share
        assert a.spec == b.spec
        assert a.default_spec == b.default_spec
        assert a.default == b.default


def test_template_set_rejects_bad_json():
    base = _base_template()
    with pytest.raises(ValueError, match='invalid JSON'):
        TemplateSet.from_json('[{"name": "main",}]', base)
    with pytest.raises(ValueError, match='must be a JSON list'):
        TemplateSet.from_json('{"name": "main"}', base)
    with pytest.raises(ValueError, match='expected an object'):
        TemplateSet.from_json('["main"]', base)
    with pytest.raises(ValueError, match='template set is empty'):
        TemplateSet.from_json('[]', base)
    with pytest.raises(ValueError, match='unknown key'):
        TemplateSet.from_json(
            '[{"name": "main", "share": 1, "spec": ".*"}]', base)
    with pytest.raises(ValueError, match='"name" must be'):
        TemplateSet.from_json('[{"share": 10}]', base)
    with pytest.raises(ValueError, match='"share" must be a number'):
        TemplateSet.from_json('[{"name": "main", "share": "many"}]', base)
    with pytest.raises(ValueError, match='"regex" must be a string'):
        TemplateSet.from_json(
            '[{"name": "main", "share": 1, "regex": 7}]', base)


def test_template_set_rejects_duplicate_names():
    with pytest.raises(ValueError, match='duplicate template names: main'):
        _templates(
            ('main', 60, None, None),
            ('main', 40, _ATTN_REGEX, _ATTN_SPEC),
        )


def test_template_set_rejects_negative_share():
    with pytest.raises(ValueError, match='share must be >= 0'):
        _templates(('main', -1, None, None))


def test_template_set_rejects_all_zero_shares():
    with pytest.raises(ValueError, match='add up to more than 0'):
        _templates(
            ('main', 0, None, None),
            ('attn', 0, _ATTN_REGEX, _ATTN_SPEC),
        )


def test_template_set_rejects_bad_regex():
    # Unbalanced group: not parseable as a model spec either, so it
    # reaches re.compile and blows up there.
    with pytest.raises(ValueError, match='bad spec regex'):
        _templates(('attn', 100, r'.*(attn', None))


def test_template_set_rejects_default_spec_outside_its_own_regex():
    """A default that its own filter rejects can never be selected --
    the sub-space would never bootstrap."""
    with pytest.raises(ValueError, match='does not match the entry spec'):
        _templates(('attn', 100, _ATTN_REGEX, 'bytes|dense.4.gelu'))


def test_template_set_rejects_unreachable_default():
    """Nothing in the auto-pick sweep matches, and no default_spec was
    given: fail loudly at configuration time rather than silently
    running a sub-search that can never hand out a conf."""
    with pytest.raises(ValueError, match="Can't pick up a default model"):
        _templates(('attn', 100, _ATTN_REGEX, None))


def test_template_entry_rejects_empty_name():
    with pytest.raises(ValueError, match='non-empty name'):
        TemplateEntry(_base_template(), name='  ', share=1)


def test_template_set_single_trusts_an_explicit_default():
    """Legacy tolerance: a --default-spec that doesn't match --spec has
    always started fine and simply been skipped at select time.
    Wrapping it in the implicit entry must not turn that into a startup
    error -- only a default the entry resolves ITSELF is validated."""
    base = _base_template(spec=r"bytes\|rnn\..*")
    odd = default_from_template(
        _base_template(), spec='bytes|dense.4.gelu')
    assert not base.matches_spec(str(odd.model))
    ts = TemplateSet.single(base, spec=base.spec_pattern, default=odd)
    assert ts.entries[0].default is odd


# --- template sets from the web form's rows --------------------------


def _row(name='', regex='', default_spec='', share=''):
    return {'name': name, 'regex': regex,
            'default_spec': default_spec, 'share': share}


def test_from_rows_builds_the_entries():
    ts = TemplateSet.from_rows([
        _row('main', share='60'),
        _row('attn', regex=_ATTN_REGEX, default_spec=_ATTN_SPEC,
             share='20'),
    ], _base_template())
    assert ts.names == ['main', 'attn']
    assert ts.by_name('main').spec is None
    assert ts.by_name('attn').spec == _ATTN_REGEX
    assert str(ts.by_name('attn').default.model) == _ATTN_SPEC
    # Shares are normalized, so 60/20 is three quarters to one.
    assert ts.nominal_share(ts.by_name('main')) == 75.0
    assert ts.nominal_share(ts.by_name('attn')) == 25.0


def test_from_rows_drops_all_blank_rows():
    """A stray row from the form's Add button is harmless."""
    ts = TemplateSet.from_rows([
        _row('main', share='60'),
        _row(),
        _row(name='   ', share=' '),
        _row('attn', regex=_ATTN_REGEX, default_spec=_ATTN_SPEC,
             share='40'),
    ], _base_template())
    assert ts.names == ['main', 'attn']


def test_from_rows_with_no_rows_is_one_unrestricted_entry():
    """An emptied table means the plain single search -- and it renders
    back as one blank `main` row, never as no rows at all."""
    base = _base_template()
    default = default_from_template(base, spec=None)
    for rows in ([], [_row()], [_row(), _row()]):
        ts = TemplateSet.from_rows(rows, base, default=default)
        assert ts.is_single
        assert ts.entries[0].name == 'main'
        assert ts.entries[0].spec is None
        assert ts.entries[0].template is base
        assert ts.rows() == [{
            'name': 'main', 'regex': '', 'default_spec': '',
            'share': '100'}]


def test_from_rows_requires_every_field_of_a_touched_row():
    base = _base_template()
    with pytest.raises(ValueError, match='row 1: name is required'):
        TemplateSet.from_rows([_row(share='60')], base)
    with pytest.raises(ValueError, match="entry 'a': share is required"):
        TemplateSet.from_rows([_row('a')], base)
    with pytest.raises(ValueError, match='is not a number'):
        TemplateSet.from_rows([_row('a', share='lots')], base)
    with pytest.raises(ValueError, match='share must be >= 0'):
        TemplateSet.from_rows([_row('a', share='-1')], base)
    # The row number is the position in the SUBMITTED table, so it
    # points at the row the user can see.
    with pytest.raises(ValueError, match='row 3: name is required'):
        TemplateSet.from_rows(
            [_row('a', share='1'), _row(), _row(share='2')], base)


def test_from_rows_applies_the_same_validation_as_the_file():
    base = _base_template()
    with pytest.raises(ValueError, match='duplicate template names: a'):
        TemplateSet.from_rows(
            [_row('a', share='1'), _row('a', share='1')], base)
    with pytest.raises(ValueError, match='add up to more than 0'):
        TemplateSet.from_rows(
            [_row('a', share='0'), _row('b', share='0')], base)
    with pytest.raises(ValueError, match='bad spec regex'):
        TemplateSet.from_rows([_row('a', regex='.*(attn', share='1')], base)
    with pytest.raises(ValueError, match='does not match the entry spec'):
        TemplateSet.from_rows([_row(
            'a', regex=_ATTN_REGEX, default_spec='bytes|dense.4.gelu',
            share='1')], base)


def test_rows_round_trip_through_from_rows():
    base = _base_template()
    ts = TemplateSet.from_rows([
        _row('main', share='60'),
        _row('attn', regex=_ATTN_REGEX, default_spec=_ATTN_SPEC,
             share='20.5'),
    ], base)
    rows = ts.rows()
    assert rows == [
        {'name': 'main', 'regex': '', 'default_spec': '', 'share': '60'},
        {'name': 'attn', 'regex': _ATTN_REGEX,
         'default_spec': _ATTN_SPEC, 'share': '20.5'},
    ]
    again = TemplateSet.from_rows(rows, base)
    assert again.names == ts.names
    for a, b in zip(again, ts):
        assert a.share == b.share
        assert a.spec == b.spec
        assert a.default_spec == b.default_spec
        assert a.default == b.default


def test_rows_and_json_describe_the_same_set():
    """The form and `--templates` are two ways to say one thing."""
    base = _base_template()
    from_file = TemplateSet.from_json(json.dumps([
        {'name': 'main', 'share': 60},
        {'name': 'attn', 'share': 20, 'regex': _ATTN_REGEX,
         'default_spec': _ATTN_SPEC},
    ]), base)
    from_form = TemplateSet.from_rows(from_file.rows(), base)
    assert from_form.to_json() == from_file.to_json()
