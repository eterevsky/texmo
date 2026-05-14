"""Plain-text formatting helpers shared by `texmo/cli/report.py` and
the per-strategy reports logged in `texmo/search.py`, plus the
data-and-rendering helpers for the report charts."""

import io

import matplotlib
import matplotlib.pyplot as plt

from .configuration import Template
from .resultdb import ConfScore, ResultDB


def format_top_conf_row(c: ConfScore, with_system: bool = False) -> str:
    """One row of a top-confs report.

    Layout: score (with run count), time (optionally with the winning
    system), then `conf.aligned_str()` — which puts the model spec
    last so it can run as long as it needs to.
    """
    score_runs = f'{c.median_score:.4f} ({c.num_runs})'
    if c.median_time is None:
        time_str = '   ?    '
    else:
        # `:7.3f` fits values up to 999.999s; system padded to 4 so
        # the next column starts at a fixed offset.
        time_str = f'{c.median_time:7.3f}s'
        if with_system:
            time_str += f' on {c.system:<4}'
    return f'{score_runs:<11} {time_str}  {c.conf.aligned_str()}'


def per_system_throughput(
    db: ResultDB,
    template: Template,
    max_time: float,
    systems: list[str] | None = None,
) -> dict[str, list[tuple[int, float]]]:
    """Compute (weights, throughput) points per system for the
    same-architecture comparison.

    For each Pareto-best conf at weight W (`top_confs_global`, no
    system filter), pin the model spec and pick each system's own
    best (lowest `median_score`) conf with that spec whose median
    time on the system is <= `max_time`. Throughput is the workload
    proxy `num_mults * batch * length * steps` divided by that
    system's median time.

    `systems` restricts the set of systems considered; defaults to
    all systems with runs in the DB.

    Returns {system: [(weights, throughput), ...]} sorted by weights.
    Systems with no qualifying conf are absent.
    """
    out: dict[str, list[tuple[int, float]]] = {}
    if systems is None:
        systems = db.get_systems()
    for cs in db.top_confs_global(template):
        spec = str(cs.conf.model)
        weights = cs.conf.num_weights
        for system in systems:
            best = db.best_conf_for_spec_on_system(spec, system, max_time)
            if best is None or best.median_time is None:
                continue
            mults_per_token = best.conf.model.num_mults
            total_mults = (
                mults_per_token
                * best.conf.batch
                * best.conf.length
                * best.conf.steps
            )
            throughput = total_mults / best.median_time
            out.setdefault(system, []).append((weights, throughput))
    for points in out.values():
        points.sort(key=lambda p: p[0])
    return out


def build_throughput_graph(
    per_system: dict[str, list[tuple[int, float]]],
) -> bytes:
    """Render `per_system_throughput`'s output as a PNG.

    One step-line per system on a log-log chart; X is weights, Y is
    multiplications per second. Same step style as `build_graph`: the
    value at `w_i` extends horizontally up to `w_{i+1}` before
    stepping to `y_{i+1}`.
    """
    plt.ioff()
    plt.clf()
    _fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    for system in sorted(per_system):
        pts = per_system[system]
        if not pts:
            continue
        xs: list[float] = []
        ys: list[float] = []
        prev_y: float | None = None
        for w, y in pts:
            if prev_y is not None:
                xs.append(w - 0.1)
                ys.append(prev_y)
            xs.append(w)
            ys.append(y)
            prev_y = y
        # Trailing horizontal so the last system also extends past its
        # final point, matching the style of `build_graph`.
        xs.append(pts[-1][0] * 2)
        ys.append(prev_y)
        ax.plot(xs, ys, label=system)

    ax.set_xlabel('weights')
    ax.set_ylabel('multiplications / second')
    ax.legend(loc='best', fontsize='small')
    f = io.BytesIO()
    plt.savefig(f, format='png')
    return f.getvalue()
