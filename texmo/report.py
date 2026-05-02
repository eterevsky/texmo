"""Plain-text formatting helpers shared by `texmo/cli/report.py` and
the per-strategy reports logged in `texmo/search2.py`."""

from .resultdb import ConfScore


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
