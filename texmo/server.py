import base64
import csv
import gzip
import hmac
import io
import logging
import os
import pickle
import threading
import time
import weakref
from datetime import datetime
from itertools import zip_longest
from queue import Queue
from typing import Iterable, Optional

import matplotlib
import matplotlib.pyplot as plt
from flask import (
    Flask,
    Response,
    make_response,
    redirect,
    render_template,
    request,
    send_from_directory,
)
from werkzeug.serving import make_server

from .common import INF, ttoa3
from .configuration import (
    Bounds,
    Configuration,
    DecayType,
    MAIN_ENTRY,
    Precision,
    Template,
    TemplateSet,
    default_from_template,
    format_lr,
)
from . import named_regexes
from .db import ConfScore, DbReader, DbWriter, FrontierVersion
from .db.writer import DbWriterProxy, Stop as WriterStop, WriterThread
from .latency import get_report, report, timer
from .predict.loss_rnn import LossModelHolder
from .predict.persist import load_predictor, save_predictor
from .predict.model_thread import (
    LOSS_REFIT_EVERY,
    BootstrapTiming,
    LossRefit,
    ModelThread,
    RunAdded,
    Stop as ModelStop,
)
from .predict.timing import TrainTimingModel
from .report import (
    build_fastest_loss_graph,
    build_fastest_time_graph,
    build_throughput_graph,
    fastest_near_best,
    per_system_throughput,
)
from .run import Run
from .search import (
    Select,
    SearchThread,
    SetTemplate,
    Stop as SearchStop,
)


# Paths reachable on the external (auth-required) port. Anything else
# 404s — in particular the index page, which is heavy to render and a
# DDoS target.
_EXTERNAL_PATHS = frozenset({'/select', '/add'})

# Internal port (LAN-trusted, no auth required).
_INTERNAL_PORT = 5000
# External port (requires Bearer auth).
_EXTERNAL_PORT = 5001

# How often (seconds) to append a latency-counter snapshot to the dump
# file, so we have a time series of where wall-clock goes (and a record
# that survives a hung shutdown).
_LATENCY_DUMP_INTERVAL = 120


def build_graph(confs: list[Configuration]) -> bytes:
    plt.ioff()
    plt.clf()

    _fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())

    xs = []
    ys = []

    prev_x = None
    prev_y = None

    for conf_score in confs:
        if prev_y is not None:
            xs.append(conf_score.conf.model.num_weights - 0.1)
            ys.append(prev_y)

        xs.append(conf_score.conf.model.num_weights)
        ys.append(conf_score.median_score)

        prev_x = conf_score.conf.model.num_weights
        prev_y = conf_score.median_score

    if prev_x is not None:
        xs.append(prev_x * 2)
        ys.append(prev_y)

    plt.plot(xs, ys)
    plt.xlabel('weights')
    plt.ylabel('entropy, b/B')
    f = io.BytesIO()
    plt.savefig(f, format='png')
    # Every index render lands here; without the close the figures
    # accumulate for the server's lifetime (pyplot keeps a global
    # registry, so `_fig` going out of scope frees nothing).
    plt.close(_fig)
    return f.getvalue()


def _envelope(confs) -> list[tuple[int, float]]:
    """Loss envelope for a list of ConfScore: a list of (weights, score)
    points with strictly decreasing score in weights ascending. Same as
    what `top_confs_global` already yields, but explicitly typed."""
    pts = sorted(
        ((c.conf.num_weights, c.median_score) for c in confs),
        key=lambda p: p[0],
    )
    out: list[tuple[int, float]] = []
    best = float('inf')
    for w, s in pts:
        if s < best:
            best = s
            out.append((w, best))
    return out


def _eval_envelope(env: list[tuple[int, float]], w: int) -> Optional[float]:
    """Step-function value at `w`: the most recent envelope score for
    a Pareto point with weights <= w. None if `w` precedes the envelope."""
    last: Optional[float] = None
    for w_i, s_i in env:
        if w_i > w:
            break
        last = s_i
    return last


def build_diff_graph(
    env1: list[tuple[int, float]],
    env2: list[tuple[int, float]],
    max_weights: int,
) -> bytes:
    """Step plot of (L2 - L1) / min(L1, L2) * 100 over the union of
    weight points in both envelopes. Positive means template 1 is
    better at that weight count."""
    plt.ioff()
    plt.clf()
    _fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # x-grid is the union of weight points from both envelopes that are
    # within the budget; both envelopes must have a value at x for the
    # diff to be defined, so we start at max(min_w1, min_w2).
    ws = sorted({w for w, _ in env1} | {w for w, _ in env2})
    ws = [w for w in ws if w <= max_weights]
    xs: list[float] = []
    ys: list[float] = []
    prev_y: Optional[float] = None
    for w in ws:
        l1 = _eval_envelope(env1, w)
        l2 = _eval_envelope(env2, w)
        if l1 is None or l2 is None:
            continue
        y = (l2 - l1) / min(l1, l2) * 100.0
        if prev_y is not None:
            xs.append(w - 0.1)
            ys.append(prev_y)
        xs.append(w)
        ys.append(y)
        prev_y = y
    if prev_y is not None:
        xs.append(max_weights)
        ys.append(prev_y)

    ax.axhline(0.0, color='gray', linewidth=0.8)
    if xs:
        ax.plot(xs, ys)
    ax.set_xlabel('weights')
    ax.set_ylabel('(L2 - L1) / min(L1, L2), %  '
                  '(positive = T1 better)')

    f = io.BytesIO()
    plt.savefig(f, format='png')
    plt.close(_fig)
    return f.getvalue()


def _maybe_float(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _strip_prefix(form: dict, prefix: str) -> dict:
    """Return a sub-dict with `prefix_` stripped from matching keys."""
    head = prefix + '_'
    return {k[len(head):]: v for k, v in form.items() if k.startswith(head)}


def _template_from_compare_form(
    form: dict, prefix: str, spec: Optional[str],
) -> Template:
    """Build a Template from prefixed compare-form fields, plumbing the
    shared `weights` value through unchanged.

    The spec filter comes from the column's preset (resolved by the
    caller), not from the form fields."""
    sub = _strip_prefix(form, prefix)
    sub['weights'] = form.get('weights', '')
    return Template.from_form(sub, spec=spec)


# The sub-search table posts two parallel repeated fields, one value
# per row per field; rows are reassembled by position. Order matches
# the columns in index.html. The regex and default spec are NOT
# posted: they come from the named preset (see `_resolve_entry_rows`),
# which is the whole point of presets-only sub-searches.
_ENTRY_FORM_FIELDS = (
    ('preset', 'entry_preset'),
    ('share', 'entry_share'),
)

# The Seed checkbox. It cannot join `_ENTRY_FORM_FIELDS`: a browser
# posts nothing at all for an unchecked box, so an unticked row would
# not occupy a slot and every later row's flag would shift up one.
# Each box instead carries its own row index as its value (the page's
# `updateShares` keeps those current across Add/Remove), so the posted
# list simply names the ticked rows.
_ENTRY_SEED_FIELD = 'entry_seed'


def _entry_rows_from_form(params) -> list[dict]:
    """Zip the form's parallel `entry_*` fields into row dicts.

    `zip_longest` rather than `zip`: a hand-built or partially posted
    form can carry unequal counts, and a missing cell should read as
    blank (and so, with the rest of its row, be ignored) instead of
    silently shifting every later row up a slot.
    """
    columns = [params.getlist(field) for _, field in _ENTRY_FORM_FIELDS]
    seeded = set(params.getlist(_ENTRY_SEED_FIELD))
    rows = [
        {key: (value or '').strip()
         for (key, _), value in zip(_ENTRY_FORM_FIELDS, values)}
        for values in zip_longest(*columns, fillvalue='')
    ]
    for i, row in enumerate(rows):
        row['seed'] = str(i) in seeded
    return rows


def _is_unrestricted(preset: str) -> bool:
    """True for the virtual no-filter choice. A blank value counts:
    a hand-built POST that omits the field means the same thing."""
    return (not preset
            or preset.strip().lower() == named_regexes.RESERVED_NAME.lower())


def _lookup_preset(name: str, presets: list[dict]) -> Optional[dict]:
    """The stored preset `name` selects, or None for the unrestricted
    choice.

    Raises ValueError when the name doesn't resolve any more -- the
    file is hand-edited, so both the index form and a bookmarked
    compare URL can outlive the entry they name.
    """
    if _is_unrestricted(name):
        return None
    for preset in presets:
        if preset['name'] == name:
            return preset
    raise ValueError(
        f'preset {name!r} is not in {named_regexes.PATH} any more')


def _resolve_entry_rows(rows: list[dict], presets: list[dict]) -> list[dict]:
    """Turn posted (preset, share) pairs into `TemplateSet.from_rows`
    rows, looking each name up in the presets read for this request.

    The resolved regex and default spec are snapshotted into the live
    template set here, so later edits to the preset file cannot
    quietly change a running search -- adopting them takes another
    Update. The entry name is the preset name, except the
    unrestricted row, which keeps the historical `main` identity so
    its coverage keys and select counters survive the switch.

    Raises ValueError (-> error banner, live template untouched) for a
    name that no longer resolves, or for the same preset twice.
    """
    out: list[dict] = []
    seen: dict[str, tuple[str, int]] = {}
    for i, row in enumerate(rows):
        preset = row['preset']
        share = row['share']
        unrestricted = _is_unrestricted(preset)
        if unrestricted and not share:
            # An untouched row from the Add button: ignore it rather
            # than make the user delete it again.
            continue
        try:
            entry = _lookup_preset(preset, presets)
        except ValueError as exc:
            raise ValueError(
                f'sub-search row {i + 1}: {exc} -- re-add it there, or '
                f'pick another preset for this row') from None
        if entry is None:
            name, regex, default_spec = MAIN_ENTRY, '', ''
            shown = named_regexes.RESERVED_NAME
        else:
            name, shown = preset, preset
            regex, default_spec = entry['regex'], entry['default_spec']
        if name in seen:
            first_shown, first_row = seen[name]
            raise ValueError(
                f'preset {first_shown!r} is selected by both row '
                f'{first_row} and row {i + 1}; each sub-search needs '
                f'its own preset')
        seen[name] = (shown, i + 1)
        out.append({
            'name': name, 'regex': regex,
            'default_spec': default_spec, 'share': share,
            'seed': bool(row.get('seed')),
        })
    return out


def _compare_defaults(live: Template) -> dict:
    """Initial values for the compare form. Both templates start as an
    unfiltered match; weights inherits the live template's upper bound
    so the comparison covers the same range the search is sweeping."""
    weights_default = (
        '' if live.max_weights.max == INF else str(int(live.max_weights.max))
    )
    out = {'weights': weights_default}
    for prefix in ('t1', 't2'):
        out[f'{prefix}_preset'] = named_regexes.RESERVED_NAME
        out[f'{prefix}_lr'] = ''
        out[f'{prefix}_length'] = ''
        out[f'{prefix}_batch'] = ''
        out[f'{prefix}_steps'] = ''
        out[f'{prefix}_time'] = ''
        for p in (Precision.FP32, Precision.FP16, Precision.BF16):
            out[f'{prefix}_{p}'] = True
        out[f'{prefix}_{Precision.FP64}'] = False
        for d in DecayType:
            out[f'{prefix}_decay_{d}'] = True
    return out


def _conf_row(conf_score) -> dict:
    """Build a template row dict for a ConfScore."""
    conf = conf_score.conf
    # --decay and --cosine are mutually exclusive; only emit the
    # flag that's actually in use.
    schedule_flag = (
        ' --cosine' if conf.cosine
        else f' --decay {format_lr(conf.decay)}'
    )
    cmd = (
        f'uv run texmo.py train'
        f" -s '{conf.model}'"
        f' -p {conf.precision}'
        f' -b {conf.batch}'
        f' --lr {format_lr(conf.lr)}'
        f'{schedule_flag}'
        f' -l {conf.length}'
        f' --steps {conf.steps}'
    )
    return {
        'spec': str(conf.model),
        'weights': conf.model.num_weights,
        'precision': str(conf.precision),
        'data': f'{conf.batch}×{conf.length}',
        'lr': conf.learning_str,
        'steps': conf.steps,
        'score': f'{conf_score.median_score:.3f} ({conf_score.num_runs})',
        'time': f'{ttoa3(conf_score.median_time)} on {conf_score.system}',
        'cmd': cmd,
    }


def _render_bounds(b: Bounds):
    if b.min == b.max:
        return str(b.min)
    else:
        return f'{b.min}-{b.max}'


# Where loss-model refits run. 'workers': the server hands ONE job
# per `LOSS_REFIT_EVERY` labeled runs to the next eligible client via
# /select and forgets it -- if the worker dies the next boundary
# issues a fresh job, and overlapping fits resolve by run_count at
# /submit_loss_model. Eligibility is worker-side (config.REFIT /
# --no-refit), so slow machines don't volunteer. 'server': ModelThread
# fits in-process on the same cadence, which needs a fast JAX platform
# in the server process (see docs/loss_prediction.md).
#
# The mode gates only the TRIGGER: the endpoints, the submit
# acceptance path and the client-side job exist in both modes, so a
# stray submission is still accepted when it is valid and newer.
LOSS_REFIT_MODES = ('workers', 'server')


def _probe_timing(model: TrainTimingModel, conf: Configuration) -> None:
    """Exercise a loaded timing model's feature path on one conf, to
    surface a feature-schema mismatch. predict_batch returns early
    (without featurizing) unless a fitted pair matches the conf's
    precision, so probe against such a pair; if none exists there's
    nothing to exercise."""
    for system, precision in model.keys():
        if precision == conf.precision:
            model.predict_batch(system, [conf])
            return


def _load_predictor(reader, name: str, probe):
    """Load the persisted predictor file for `name` (stored next to
    the DB; see predict/persist.py), returning it only if it both
    deserializes and survives `probe` (a trivial use).

    A feature-schema change (e.g. the loss model's new split slot) makes
    an old model unpickle fine but raise a shape mismatch on the first
    predict -- so we don't just guard deserialization, we run it once.
    On any failure we log and return None, leaving the caller to refit.
    """
    try:
        model = load_predictor(reader.path, name)
        if model is None:
            return None
        probe(model)
        return model
    except Exception as e:
        logging.warning(
            f"Ignoring persisted {name!r} model "
            f"(incompatible, will refit): {e!r}")
        return None


class SearchServer(object):
    def __init__(
        self,
        path: str,
        template: Template,
        train_time: tuple[float, float],
        default_spec: str,
        warmup_systems: Optional[Iterable[str]] = None,
        templates_json: Optional[str] = None,
        loss_refit: str = 'workers',
    ):
        assert loss_refit in LOSS_REFIT_MODES, (
            f"loss_refit must be one of {LOSS_REFIT_MODES}, "
            f"got {loss_refit!r}")
        self.loss_refit = loss_refit
        # `path` is also the read-only handle factory for per-request
        # reads in `index` / `compare` / `throughput`; opening a fresh
        # `DbReader` per request avoids contending with the writer.
        self.path: str = path
        # Systems that climb the search's warmup ladder before joining
        # ordinary selection (new machine or new backend).
        self.warmup_systems: set[str] = set(warmup_systems or ())
        # Model specs live in the sub-search entries, never in the base
        # template: the two would otherwise both filter, and the index
        # form would have two boxes meaning the same thing. `--spec`
        # seeds the first entry instead (below).
        cli_spec = template.spec_pattern
        self.template: Template = template.with_spec(None)
        self.train_time: tuple[float, float] = train_time
        # Stored for `update()` so a template change with no literal
        # spec can still fall back to the CLI default.
        self._default_spec = default_spec
        # Derived from the CLI template, spec included, so `--spec` with
        # no `--default-spec` still seeds a matching conf.
        self.default = default_from_template(template, spec=default_spec)
        logging.info(f"Default configuration: {self.default}")

        # Weighted sub-searches (`--templates`). Without a file this is
        # a single `main` entry carrying `--spec` (unrestricted when
        # that wasn't given), which leaves selection exactly as it was.
        if templates_json:
            self.templates = TemplateSet.from_json(
                templates_json, self.template)
            for entry in self.templates:
                logging.info(
                    f'Sub-search {entry.name}: '
                    f'{self.templates.nominal_share(entry):.0f}% '
                    f'spec={entry.spec} default={entry.default}')
        else:
            self.templates = TemplateSet.single(
                self.template, spec=cli_spec, default_spec=default_spec,
                default=self.default)

        self.requests_queue = Queue()
        self.confs_by_system: dict[str, Queue] = {}
        self.confs_by_system_lock = threading.Lock()

        # Single shared instances. The Model thread mutates them
        # (per-(system, precision) writes on the timing model are atomic
        # in CPython; the loss holder swaps its underlying model with a
        # single attribute write); the Search thread only reads.
        self.timing_model = TrainTimingModel()
        self.loss_model = LossModelHolder()

        # Writer -> search signal: bumped whenever an incoming run
        # flips the winning conf at some (system, W, T), which is the
        # server's cheapest "the frontier moved" notification. The
        # search thread's frontier-seed queues invalidate on it.
        self.frontier_version = FrontierVersion()

        # Apply the schema if this is a fresh DB so the read below
        # doesn't trip on a not-yet-created file — `open_connection`
        # only runs the schema bootstrap on a read-write open. The
        # temporary writer lives only on this thread.
        with DbWriter(self.path):
            pass

        # Whatever's missing here gets trained async on the Model
        # thread once the queues are wired up below.
        need_timing_bootstrap = True
        need_loss_refit = True
        with DbReader(self.path) as reader:
            # Load persisted predictors, but only adopt them if they
            # still work with the current feature schema -- an old model
            # from before a schema change deserializes fine yet raises on
            # the first predict. _load_predictor probes each one and
            # returns None (-> refit) on any incompatibility.
            loaded_timing = _load_predictor(
                reader, 'timing',
                lambda m: _probe_timing(m, self.default))
            if loaded_timing is not None:
                # Adopt the persisted instance as the shared one.
                # Safe to swap here: no threads have been started yet.
                self.timing_model = loaded_timing
                need_timing_bootstrap = False
                logging.info(
                    "Loaded persisted timing model from DB "
                    f"({len(loaded_timing.keys())} pairs)")
            loaded_loss_model = _load_predictor(
                reader, 'loss',
                lambda m: m.predict([self.default]))
            if loaded_loss_model is not None:
                self.loss_model.set_model(loaded_loss_model)
                need_loss_refit = False
                logging.info(
                    "Loaded persisted loss model "
                    f"({type(loaded_loss_model).__name__})")

        # Refit bookkeeping. The lock guards the counters below
        # (touched from concurrent request handlers); held only for
        # the few lines of arithmetic, never across any fitting or
        # I/O. Freshness is NOT tracked here: submit_loss_model
        # compares against the run_count of whatever the holder
        # currently publishes, which the model thread's in-process
        # refits keep fresh too. Maintained in both modes -- the
        # submit path stays open even when this server grants no
        # jobs.
        self._refit_lock = threading.Lock()
        self._refit_runs_since_grant = 0
        self._refit_published_at = time.time()
        self._refit_granted_at = 0.0
        logging.info(f"Loss-model refits run on: {self.loss_refit}")

        # The two WSGI listeners, once `serve` binds them. The writer
        # thread's panic path needs them to unbind the ports; empty
        # until then (and in tests, which never call `serve`).
        self._wsgi_servers: list = []

        # Writer thread first: producers below post to its queue and
        # we want a consumer ready before that happens.
        self.write_queue: Queue = Queue()
        # The callback must not hold `self` strongly: a bound method
        # would make SearchServer <-> WriterThread a reference cycle,
        # and `__del__` -- which is what stops these threads when a
        # server is dropped without `join()` -- would then wait on a
        # cyclic collection that may never come, leaving all three
        # threads running against a server nobody can reach.
        server_ref = weakref.ref(self)

        def _writer_fatal():
            server = server_ref()
            if server is not None:
                server._on_writer_fatal()

        self.writer_thread = WriterThread(
            self.path, self.write_queue, on_fatal=_writer_fatal,
            frontier_version=self.frontier_version)
        self.writer_thread.start()

        # Server-side handle the request handlers and `add_run` use to
        # enqueue writes without touching SearchThread.
        self._writer = DbWriterProxy(self.write_queue)

        self.train_queue: Queue = Queue()
        self.model_thread = ModelThread(
            self.path, self.train_queue,
            write_queue=self.write_queue,
            timing_model=self.timing_model,
            loss_model=self.loss_model,
            loss_refit_local=(self.loss_refit == 'server'),
        )
        self.model_thread.start()
        # Whatever wasn't on disk: train it async. Each request gets a
        # falls-through strategy until the Model thread publishes.
        if need_timing_bootstrap:
            self.train_queue.put(BootstrapTiming())
        if need_loss_refit:
            self.train_queue.put(LossRefit())

        # SearchThread opens its own persistent DbReader inside `run()`
        # (so the reader's connection lives on its own thread). Writes
        # happen on the request handler thread directly into
        # `write_queue` — SearchThread is read-only now.
        self.search_thread = SearchThread(
            path=self.path,
            template=template,
            train_time=train_time,
            default=self.default,
            requests_queue=self.requests_queue,
            confs_by_system=self.confs_by_system,
            confs_by_system_lock=self.confs_by_system_lock,
            warmup_systems=self.warmup_systems,
            timing_model=self.timing_model,
            loss_model=self.loss_model,
            templates=self.templates,
            frontier_version=self.frontier_version,
        )
        self.search_thread.start()

    def __del__(self):
        self.requests_queue.put(SearchStop())
        self.train_queue.put(ModelStop())
        self.write_queue.put(WriterStop())

    def index(
        self,
        selected_system: Optional[str] = None,
        selected_entry: Optional[str] = None,
    ):
        return self._render_index(
            template=self.template,
            templates=self.templates,
            default_spec=self._default_spec or '',
            train_time=self.train_time,
            selected_system=selected_system,
            selected_entry=selected_entry,
        )

    def _entry_rows(
        self, templates: TemplateSet, presets: list[dict],
    ) -> list[dict]:
        """One row per live entry: the editable bits (preset, share,
        seed) and the read-only ones (its snapshotted regex/default,
        plus nominal vs realized share and any seed queue) in a single
        table.

        Realized share is `selects / total selects` from the search
        thread's in-memory counters (see `Search.entry_selects`), and
        `selects` is the raw count behind that percentage -- realized
        share means nothing until an entry has a few hundred selects,
        and the count is how you see whether it does.

        Regex and default render from the ENTRY, never from the preset
        file: the entry is what the search is actually running. Where
        the two have since diverged the row carries a `stale` hint,
        which is information, not an error -- nothing changes until
        the user presses Update.
        """
        by_name = {p['name']: p for p in presets}
        preset_names = [p['name'] for p in presets]
        search = self.search_thread.search
        selects = {} if search is None else search.entry_selects
        total = sum(selects.get(e.name, 0) for e in templates)
        rows = []
        for entry in templates:
            unrestricted = not entry.spec
            preset = (named_regexes.RESERVED_NAME if unrestricted
                      else entry.name)
            options = list(preset_names)
            stale = ''
            if not unrestricted:
                stored = by_name.get(entry.name)
                if stored is None:
                    # An entry from --spec or --templates carries a
                    # pattern, not a preset name; when that pattern IS
                    # a preset's, start the row on it.
                    same = next(
                        (p for p in presets if p['regex'] == entry.spec),
                        None)
                    if same is not None:
                        preset = same['name']
                    else:
                        # Keep the row selectable so the form still
                        # round-trips.
                        options.insert(0, entry.name)
                        stale = (
                            f'{entry.name!r} is no longer in '
                            f'{named_regexes.PATH}; Update will reject '
                            f'this row until it comes back or you pick '
                            f'another preset')
                elif (stored['regex'] != (entry.spec or '')
                      or stored['default_spec'] != (entry.default_spec or '')):
                    stale = (
                        f'{named_regexes.PATH} has changed since this was '
                        f'applied; the search is still running the pattern '
                        f'shown here. Press Update to adopt the new one.')
            n = selects.get(entry.name, 0)
            rows.append({
                'preset': preset,
                'options': options,
                'share': f'{entry.share:g}',
                'seed': entry.seed,
                # None until the search thread has built this entry's
                # queue, which first happens on its first seeded
                # select -- so a just-ticked row shows nothing yet.
                'seed_left': (
                    None if search is None
                    else search.seed_queue_size(entry.name)),
                'regex': entry.spec or '',
                'default_spec': entry.default_spec or '',
                'stale': stale,
                'name': entry.name,
                'share_pct': f'{templates.nominal_share(entry):.0f}%',
                'realized': f'{100.0 * n / total:.0f}%' if total else '—',
                'selects': n,
                'default': str(entry.default.model),
            })
        return rows

    def _submitted_entry_rows(
        self, form_rows: list[dict], presets: list[dict],
    ) -> list[dict]:
        """Re-render rows straight from a rejected POST, so the user
        fixes what they submitted instead of retyping it.

        There is no live entry behind these, so the stats cells stay
        empty and regex/default show what the preset file says right
        now -- the best available reading of what Update would apply.
        """
        by_name = {p['name']: p for p in presets}
        preset_names = [p['name'] for p in presets]
        rows = []
        for row in form_rows:
            preset = row['preset'] or named_regexes.RESERVED_NAME
            unrestricted = _is_unrestricted(preset)
            stored = None if unrestricted else by_name.get(preset)
            options = list(preset_names)
            stale = ''
            if not unrestricted and stored is None:
                options.insert(0, preset)
                stale = f'{preset!r} is not in {named_regexes.PATH}'
            rows.append({
                'preset': preset,
                'options': options,
                'share': row['share'],
                'seed': bool(row.get('seed')),
                'seed_left': None,
                'regex': stored['regex'] if stored else '',
                'default_spec': stored['default_spec'] if stored else '',
                'stale': stale,
                'name': '',
                'share_pct': '',
                'realized': '',
                'selects': '',
                'default': '',
            })
        return rows

    def _render_index(
        self,
        *,
        template: Template,
        templates: TemplateSet,
        default_spec: str,
        train_time: tuple[float, float],
        selected_system: Optional[str] = None,
        selected_entry: Optional[str] = None,
        entry_form: Optional[list[dict]] = None,
        error: Optional[str] = None,
    ):
        precision = {}
        for p in Precision:
            precision[p] = p in template.precision

        decay_types = {
            t.value: t in template.decay_types for t in DecayType
        }

        tmin, tmax = train_time
        train_time_str = f'{tmin}-{tmax}'

        # Read per render: the file is hand-edited outside the server,
        # so anything cached would go stale silently.
        presets = named_regexes.load()

        # Frontier filter: with an entry selected, the top list is the
        # one THAT sub-search sees (its spec filter applied at query
        # time -- no schema change, no stored attribution).
        entry = templates.by_name(selected_entry) if selected_entry else None
        query_template = template if entry is None else entry.template

        # Use a dedicated read-only connection to avoid contending with
        # the writer thread for the main connection.
        # `max_time=tmax` limits the top list to confs that at least
        # one system can train within the user's time budget -- confs
        # whose smallest cross-system median_time exceeds tmax aren't
        # actionable from this template.
        with DbReader(self.path) as ro_db:
            systems = ro_db.get_systems()
            top_confs = list(
                ro_db.top_confs_global(
                    query_template, system=selected_system,
                    max_time=tmax))

        graph = build_graph(top_confs)

        top = [_conf_row(tc) for tc in top_confs]

        return render_template(
            "index.html",
            default_spec=default_spec,
            weights=_render_bounds(template.max_weights),
            num_layers=_render_bounds(template.num_layers),
            length=_render_bounds(template.length),
            batch=_render_bounds(template.batch),
            precision=precision,
            lr=_render_bounds(template.lr),
            decay_types=decay_types,
            steps=_render_bounds(template.steps),
            time=train_time_str,
            top=top,
            graph=base64.b64encode(graph).decode('ascii'),
            systems=systems,
            selected_system=selected_system,
            # Always at least one row -- the live set's, or on the
            # rejection path the submitted ones, so the user can fix
            # them in place rather than retype.
            entries=entry_form or self._entry_rows(templates, presets),
            show_entries=not templates.is_single,
            entry_names=templates.names,
            selected_entry=selected_entry or '',
            preset_names=[p['name'] for p in presets],
            reserved_regex_name=named_regexes.RESERVED_NAME,
            presets_path=named_regexes.PATH,
            error=error,
        )

    def update(self, params):
        # The form's default_spec field is authoritative: empty means
        # "no override, let the auto-finder pick". Replaces any prior
        # value (CLI --default-spec on startup, or previous form value).
        new_default_spec = params.get('default_spec', '').strip()
        new_rows = _entry_rows_from_form(params)
        # Resolved against the file as of THIS request; the values go
        # into the live set and stay there until the next Update.
        presets = named_regexes.load()
        new_template = None
        new_default = None
        new_templates = None
        new_train_time = self.train_time
        try:
            # No spec on the base template: the sub-search rows are the
            # only place model specs are defined.
            new_template = Template.from_form(params)
            new_default = default_from_template(
                new_template, spec=new_default_spec or None)
            # The template set is rebuilt against the NEW base template
            # -- entries share every bound but the spec filter, so a
            # bounds edit has to propagate into all of them. Anything
            # malformed (a nameless or shareless row, a non-numeric or
            # negative share, duplicate names, a bad regex, a
            # default_spec outside its own entry) raises here and the
            # whole update is rejected below. All-blank rows are
            # dropped; with no rows left this is one `main` entry
            # covering the whole (unrestricted) template.
            new_templates = TemplateSet.from_rows(
                _resolve_entry_rows(new_rows, presets), new_template,
                default_spec=new_default_spec or None, default=new_default)
            time_str = params.get("time", "")
            if time_str:
                tmin, tmax = map(float, time_str.split("-"))
                new_train_time = (tmin, tmax)
        except Exception as exc:
            # Re-render the index with the (partially) submitted form
            # values and an error banner; leave the live template
            # untouched so the search keeps running on the last known-
            # good configuration.
            logging.warning(f"update() rejected form: {exc}")
            return self._render_index(
                template=new_template or self.template,
                templates=self.templates,
                default_spec=new_default_spec or (
                    self._default_spec or ''),
                train_time=new_train_time,
                entry_form=self._submitted_entry_rows(new_rows, presets),
                error=str(exc),
            )

        # Mutate the search thread's state via a queue message so the
        # update applies atomically between two select_conf calls
        # rather than racing the loop.
        self.requests_queue.put(SetTemplate(
            template=new_template,
            init_conf=new_default,
            train_time=new_train_time,
            templates=new_templates,
        ))
        # SearchServer-side copies are read by Flask handlers (index
        # form rendering, /compare); the search thread's copies may
        # lag by one queue step but converge on the next select.
        self.template = new_template
        self.default = new_default
        self.templates = new_templates
        self.train_time = new_train_time
        self._default_spec = new_default_spec
        logging.info(f'New default configuration: {new_default}')
        logging.info(f'New template: {new_template}')
        logging.info(f'New train time: {new_train_time}')
        if not new_templates.is_single:
            logging.info(f'New sub-searches: {new_templates}')
        return redirect("/")

    def throughput(self, args):
        """Render the per-system throughput chart for the live template.

        For each weight bucket on the global Pareto front, fixes the
        model spec and finds each system's best (batch, length, lr,
        steps) for that spec. Plots `mults * batch * length * steps /
        median_time` per system.

        `args` carries the `system_<name>` checkboxes; on the first
        visit (no args) every system is selected.
        """
        max_time = self.train_time[1]
        with DbReader(self.path) as ro_db:
            all_systems = ro_db.get_systems()
            if args:
                # Submitted form: unchecked boxes are absent. Treat
                # each system as opt-in by `system_<name>` membership.
                selected = [
                    s for s in all_systems if f'system_{s}' in args
                ]
            else:
                selected = list(all_systems)
            per_system = per_system_throughput(
                ro_db, self.template, max_time, systems=selected)
        graph = base64.b64encode(
            build_throughput_graph(per_system)
        ).decode('ascii')
        selected_set = set(selected)
        return render_template(
            "throughput.html",
            graph=graph,
            max_time=max_time,
            num_systems=len(per_system),
            systems=all_systems,
            selected_systems=selected_set,
        )

    def fastest(self, args):
        """Render the cross-system fastest-near-best report.

        For each weight bucket on the global Pareto front, finds the
        (conf, system) pair that reaches within 1% of the best loss
        in the lowest time across all systems. Renders two graphs
        (loss and time per weight) plus a table.
        """
        with DbReader(self.path) as ro_db:
            segments = fastest_near_best(ro_db, self.template)

        # Deduplicate by weight count for the table — two confs at the
        # same weight count shouldn't both appear; keep the faster one.
        by_weights: dict[int, ConfScore] = {}
        for _, _, cs in segments:
            nw = cs.conf.model.num_weights
            existing = by_weights.get(nw)
            if existing is None or cs.median_time < existing.median_time:
                by_weights[nw] = cs
        confs = [
            _conf_row(cs)
            for cs in sorted(
                by_weights.values(),
                key=lambda c: c.conf.model.num_weights,
            )
        ]

        loss_graph = None
        time_graph = None
        if segments:
            max_w = max(cs.conf.model.num_weights for _, _, cs in segments)
            loss_graph = base64.b64encode(
                build_fastest_loss_graph(segments, max_w)
            ).decode('ascii')
            time_graph = base64.b64encode(
                build_fastest_time_graph(segments, max_w)
            ).decode('ascii')

        return render_template(
            "fastest.html",
            confs=confs,
            loss_graph=loss_graph,
            time_graph=time_graph,
        )

    def compare(self, args):
        """Render the template-comparison page.

        With no params -> just the form (plus reasonable defaults).
        With params -> form + diff curve graph + per-template top
        tables.

        Form layout: shared `weights`, then two namespaced columns
        (`t1_*` / `t2_*`) for preset, lr, length, batch, steps, decay
        types, precision, and per-template `max_time`. Each column's
        spec filter is a named preset -- one control, the same
        vocabulary as the index form -- resolved fresh per render.
        """
        presets = named_regexes.load()
        defaults = _compare_defaults(self.template)
        # If the form was submitted, unchecked checkboxes are absent
        # from `args` — falling back to defaults would re-check them.
        # So only inherit defaults when nothing was submitted.
        submitted = bool(args)

        form: dict = {}
        for key, default in defaults.items():
            if isinstance(default, bool):
                form[key] = bool(args.get(key)) if submitted else default
            else:
                form[key] = args.get(key, default)

        graph = None
        top1: list[dict] = []
        top2: list[dict] = []
        max_weights: Optional[int] = None
        error = None
        columns: list[Template] = []

        if args:
            try:
                max_weights = int(form['weights'])
            except (TypeError, ValueError):
                max_weights = None

        if max_weights is not None and max_weights > 0:
            try:
                for prefix in ('t1', 't2'):
                    preset = _lookup_preset(
                        form[f'{prefix}_preset'], presets)
                    columns.append(_template_from_compare_form(
                        form, prefix,
                        preset['regex'] if preset else None))
            except ValueError as exc:
                # A bookmarked URL naming a preset that has since been
                # edited out of the file: say so, render just the form.
                error = str(exc)
                columns = []

        if columns:
            t1, t2 = columns
            t1_max_time = _maybe_float(form.get('t1_time'))
            t2_max_time = _maybe_float(form.get('t2_time'))
            with DbReader(self.path) as ro_db:
                confs1 = list(ro_db.top_confs_global(
                    t1, max_weights=max_weights, max_time=t1_max_time))
                confs2 = list(ro_db.top_confs_global(
                    t2, max_weights=max_weights, max_time=t2_max_time))
            env1 = _envelope(confs1)
            env2 = _envelope(confs2)
            graph = base64.b64encode(
                build_diff_graph(env1, env2, max_weights)
            ).decode('ascii')
            top1 = [_conf_row(c) for c in confs1]
            top2 = [_conf_row(c) for c in confs2]

        # An unresolvable name stays in its dropdown so the page still
        # shows what was asked for and the URL round-trips.
        preset_names = [p['name'] for p in presets]
        preset_options = {}
        for prefix in ('t1', 't2'):
            selected = form[f'{prefix}_preset']
            options = list(preset_names)
            if not _is_unrestricted(selected) and selected not in options:
                options.insert(0, selected)
            preset_options[prefix] = options

        return render_template(
            "compare.html",
            form=form,
            graph=graph,
            top1=top1,
            top2=top2,
            preset_options=preset_options,
            reserved_regex_name=named_regexes.RESERVED_NAME,
            error=error,
        )

    def _maybe_grant_refit(self, system: str) -> Optional[dict]:
        """Grant a loss-refit job to `system` if one is due.

        Fire-and-forget cadence: one job per LOSS_REFIT_EVERY new
        labeled runs since the last GRANT (not publish). A dead
        worker just means the next boundary issues a fresh job; an
        overlap is resolved by the newer run_count at submit time."""
        with self._refit_lock:
            if self._refit_runs_since_grant < LOSS_REFIT_EVERY:
                return None
            self._refit_runs_since_grant = 0
            self._refit_granted_at = time.time()
        logging.info(f"Granting loss-refit job to {system}")
        return {
            "system": system, "conf": None, "strategy": None,
            "refit_loss": {},
        }

    def select(self, args):
        system = args["system"]

        if self.loss_refit == 'workers' and args.get("refit") == "1":
            job = self._maybe_grant_refit(system)
            if job is not None:
                return job

        with self.confs_by_system_lock:
            if system not in self.confs_by_system:
                self.confs_by_system[system] = Queue()
                self.requests_queue.put(Select(system=system))
            response_queue = self.confs_by_system[system]

        self.requests_queue.put(Select(system=system))

        result = response_queue.get()
        if result is None:
            logging.info(f"No conf for system {system}")
            return {"system": system, "conf": None, "strategy": None}
        logging.info(
            f"Sending conf for system {result.system}: {result.conf} "
            f"(strategy={result.strategy})")
        return result.to_dict()

    def add_run(self, params):
        run = Run.from_dict(params["run"])
        if run.loss is None:
            logging.info('run.loss is None!')
            return
        conf = Configuration.from_dict(params["conf"])
        strategy = params.get("strategy")
        logging.info(f"Adding run: {conf} - {run} (strategy={strategy})")
        # Both queues are posted from this same request-handler thread
        # in order; SQLite WAL guarantees that any subsequent refit on
        # ModelThread reading after `AddRun` commits sees the new row.
        self._writer.add_run(
            conf, run, strategy=strategy, track_winner_change=True)
        # Only runs with a loss get here, so both refit cadences --
        # the grant tally below and the Model thread's -- count
        # labeled runs. The tally is kept in either mode; it just
        # goes unread when the Model thread owns the trigger.
        with self._refit_lock:
            self._refit_runs_since_grant += 1
        self.train_queue.put(
            RunAdded(system=run.system, precision=conf.precision))

    def training_data(self) -> tuple[bytes, int]:
        """Gzipped CSV of all labeled runs + the run count.

        Streamed straight off a raw DB cursor -- no Configuration
        parsing here; the client parses, deduplicating by
        (spec, precision)."""
        with timer('SearchServer.training_data'):
            buf = io.BytesIO()
            n = 0
            with DbReader(self.path) as reader, \
                    gzip.GzipFile(fileobj=buf, mode='wb') as gz:
                text = io.TextIOWrapper(gz, encoding='utf-8', newline='')
                w = csv.writer(text)
                w.writerow(['spec', 'precision', 'lr', 'length', 'batch',
                            'steps', 'decay', 'cosine', 'loss'])
                for row in reader.iter_labeled_runs_raw():
                    w.writerow(row)
                    n += 1
                text.flush()
                text.detach()
            logging.info(
                f"Serving training data: {n} runs, "
                f"{len(buf.getvalue()) / 1e6:.1f} MB gzipped")
            return buf.getvalue(), n

    def submit_loss_model(self, args, body: bytes) -> dict:
        """Accept a client-fitted loss model (pickled in `body`).

        The model carries its own freshness measure: `run_count`, the
        number of training rows counted by the worker that fit it
        (monotonic over the DB's life). Accept only if newer than the
        published model, and only if it survives a probe-predict
        (guards against a client running a different feature schema).

        Open in both refit modes: a submission that arrives while the
        model thread owns the cadence is still a valid, newer model."""
        system = args.get('system', '?')
        fit_s = float(args.get('fit_s', 0))
        try:
            model = pickle.loads(body)
            model.predict([self.default])
        except Exception as e:
            logging.warning(
                f"Rejecting loss model from {system}: probe failed: {e!r}")
            return {'accepted': False, 'reason': f'probe failed: {e!r}'}
        run_count = getattr(model, 'run_count', 0)
        now = time.time()
        with self._refit_lock:
            # Compare against the live holder, not a separate tally:
            # in 'server' mode the model thread publishes without
            # touching this handler, and a straggler submission must
            # not overwrite that fresher model.
            published = getattr(
                self.loss_model.current(), 'run_count', 0) or 0
            if run_count <= published:
                logging.info(
                    f"Rejecting stale loss model from {system} "
                    f"(run_count={run_count} <= {published})")
                return {'accepted': False, 'reason': 'stale run_count'}
            turnaround = now - self._refit_granted_at
            interval = now - self._refit_published_at
            self._refit_published_at = now
        save_predictor(self.path, 'loss', model)
        self.loss_model.set_model(model)
        logging.info(
            f"Published client-fitted loss model from {system}: "
            f"run_count={run_count}, fit={ttoa3(fit_s)}, "
            f"turnaround={ttoa3(turnaround)}, "
            f"interval since previous model={ttoa3(interval)}")
        return {'accepted': True}

    def _on_writer_fatal(self) -> None:
        """Bring the server down after an unrecoverable write error.

        The alternative is worse than being down: reads and `/add`
        would keep succeeding while nothing drains the write queue.
        Unbinding both ports makes clients fail their requests and
        fall into their existing retry loops, and `serve` then
        proceeds to its normal shutdown.

        Called ON the writer thread, and the shutdown it starts ends
        in `join()` -- which joins that same thread. So the actual
        `shutdown()` calls go on a one-shot thread (exactly what
        `/stop` does, and for the same reason) and this returns
        immediately, letting the writer finish dying."""
        logging.critical(
            "Writer thread is gone -- shutting the server down so "
            "producers fail fast instead of writing into the void")
        servers = list(self._wsgi_servers)
        if not servers:
            # Nothing bound yet (a failure during startup, or a
            # non-serving construction): the CRITICAL log above is
            # the whole signal.
            return
        def _do_shutdown():
            for srv in servers:
                srv.shutdown()
        threading.Thread(target=_do_shutdown, daemon=True).start()

    def join(self):
        # Producers first so no more writes / refits land on the
        # writer queue, then the writer drains and exits.
        self.requests_queue.put(SearchStop())
        self.search_thread.join()
        logging.info("Search thread joined")
        self.train_queue.put(ModelStop())
        self.model_thread.join()
        logging.info("Model thread joined")
        self.write_queue.put(WriterStop())
        self.writer_thread.join()
        logging.info("Writer thread joined")

    def _dump_latency_loop(self, path: str):
        """Append a timestamped latency snapshot to `path` every
        `_LATENCY_DUMP_INTERVAL` seconds. Daemon loop, so it never blocks
        shutdown -- and the file survives a hung exit."""
        while True:
            time.sleep(_LATENCY_DUMP_INTERVAL)
            try:
                ts = datetime.now().isoformat(timespec='seconds')
                # Queue state: pending Selects on the (serial) search
                # thread, and per-system ready-but-unfetched responses.
                with self.confs_by_system_lock:
                    per_system = {
                        s: q.qsize() for s, q in self.confs_by_system.items()
                    }
                queues = (
                    f"requests_queue: {self.requests_queue.qsize()}  "
                    f"responses: {per_system}\n"
                )
                # utf-8 explicitly: the report has 'us' rendered as the
                # micro sign, which Windows' default cp1252 can't encode.
                with open(path, 'a', encoding='utf-8') as f:
                    f.write(f"\n===== {ts} =====\n{queues}{get_report()}")
                    f.flush()
            except Exception:
                logging.exception("latency dump failed")

    def serve(self, api_key: str):
        """Run the Flask app on the LAN-internal and external ports.

        Blocks until both werkzeug listeners shut down (either via
        SIGINT or the `/stop` route), then joins the search/model
        threads and prints the latency report.
        """
        # Persist latency counters periodically: /latency shows the live
        # snapshot (and answers even while /select is stalled, since it
        # doesn't touch the SearchThread); this gives the time series and
        # a record that outlives a hung shutdown.
        latency_log = os.path.join(
            os.path.dirname(os.path.abspath(self.path)), 'latency.log')
        logging.info(
            f"Appending latency snapshots to {latency_log} every "
            f"{_LATENCY_DUMP_INTERVAL}s")
        threading.Thread(
            target=self._dump_latency_loop, args=(latency_log,),
            daemon=True).start()

        app = Flask("texmo")

        @app.before_request
        def _gate():
            # We bind two sockets: one on the internal port (LAN, no auth),
            # one on the external port (auth required, restricted path
            # set). The handler can tell which socket received the request
            # from SERVER_PORT.
            port = int(request.environ.get('SERVER_PORT', 0))
            if port != _EXTERNAL_PORT:
                return  # internal: no gating
            # 404 first -- gives less info to unauthenticated probes than
            # leaking which paths exist.
            if request.path not in _EXTERNAL_PATHS:
                return ('', 404)
            if not api_key:
                # External port is enabled but no key configured -- refuse
                # rather than serve unauthenticated externally.
                return ('', 401)
            provided = request.headers.get('Authorization', '')
            if not provided.startswith('Bearer '):
                return ('', 401)
            if not hmac.compare_digest(provided[7:], api_key):
                return ('', 401)

        @app.route("/", methods=["GET"])
        def _index():
            selected_system = request.args.get("system") or None
            selected_entry = request.args.get("entry") or None
            return self.index(
                selected_system=selected_system,
                selected_entry=selected_entry)

        @app.route("/update", methods=["POST"])
        def _update():
            with timer("SearchServer.update"):
                return self.update(request.form)

        @app.route("/compare", methods=["GET"])
        def _compare():
            with timer("SearchServer.compare"):
                return self.compare(request.args)

        @app.route("/throughput", methods=["GET"])
        def _throughput():
            with timer("SearchServer.throughput"):
                return self.throughput(request.args)

        @app.route("/fastest", methods=["GET"])
        def _fastest():
            with timer("SearchServer.fastest"):
                return self.fastest(request.args)

        @app.route("/training_data", methods=["GET"])
        def _training_data():
            payload, n = self.training_data()
            resp = Response(payload, mimetype='text/csv')
            resp.headers['Content-Encoding'] = 'gzip'
            resp.headers['X-Run-Count'] = str(n)
            return resp

        @app.route("/submit_loss_model", methods=["POST"])
        def _submit_loss_model():
            return self.submit_loss_model(request.args, request.get_data())

        @app.route("/select", methods=["GET"])
        def _select():
            with timer("SearchServer.select"):
                return self.select(request.args)

        @app.route("/add", methods=["POST"])
        def _add():
            with timer("SearchServer.add_run"):
                self.add_run(request.json)
                return "", 200

        @app.route("/latency", methods=["GET"])
        def _latency():
            response = make_response(get_report(), 200)
            response.mimetype = "text/plain"
            return response

        @app.route('/favicon.ico')
        def _favicon():
            return send_from_directory(
                os.path.join(app.root_path, 'static'),
                'favicon.ico',
                mimetype='image/vnd.microsoft.icon'
            )

        # Bind two listeners on the same WSGI app:
        #   internal: 0.0.0.0:5000 -- LAN-accessible, no auth.
        #   external: 127.0.0.1:5001 -- bound to loopback so only a
        #     local reverse proxy (terminating TLS) can reach it;
        #     auth required.
        internal_srv = make_server(
            '0.0.0.0', _INTERNAL_PORT, app, threaded=True)
        external_srv = make_server(
            '127.0.0.1', _EXTERNAL_PORT, app, threaded=True)
        # Published for `_on_writer_fatal`, which unbinds both ports
        # from the writer thread when a write turns out to be fatal.
        self._wsgi_servers = [internal_srv, external_srv]

        @app.route("/stop", methods=["POST"])
        def _stop():
            # `serve_forever` returns when `shutdown()` is called. Doing
            # that from inside a request thread would deadlock, so spawn
            # a one-shot thread that runs after the response is delivered.
            # The external port `before_request` gate already 404s /stop,
            # so this only fires from the internal (LAN-trusted) listener.
            logging.info("Stop requested via web UI")
            def _do_shutdown():
                internal_srv.shutdown()
                external_srv.shutdown()
            threading.Thread(target=_do_shutdown, daemon=True).start()
            return ("Stopping. Search threads will join and the process "
                    "will exit shortly.", 200)
        logging.info(
            f"Serving internal (no auth) on 0.0.0.0:{_INTERNAL_PORT}, "
            f"external (auth) on 127.0.0.1:{_EXTERNAL_PORT}")
        threads = [
            threading.Thread(target=s.serve_forever, daemon=True)
            for s in (internal_srv, external_srv)
        ]
        for t in threads:
            t.start()
        try:
            # Block forever; KeyboardInterrupt falls through to cleanup.
            for t in threads:
                t.join()
        finally:
            internal_srv.shutdown()
            external_srv.shutdown()
            self.join()
            report()

