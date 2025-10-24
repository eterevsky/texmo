import logging
import random
from math import log2, sqrt
from rich.table import Table
from typing import Optional

from . import latency
from .common import INF, itoa3, ttoa3, console
from .configuration2 import Configuration2, Template, conf_neighbors
from .resultdb import ResultDB, ConfScore
from .run import Run


def top_confs_report(
    confs: list[ConfScore], max_weights: int, max_time: float | None, system: str | None
) -> str:
    sys = "" if system is None else f" ({system})"
    t = "" if max_time is None else f" T ≤ {ttoa3(max_time)}"
    lines = [f"Top confs W ≤ {itoa3(max_weights)}{t}{sys}:"]
    for c in confs:
        t = "?       " if c.median_time is None else "{:8}".format(ttoa3(c.median_time))
        lines.append(
            f"{c.median_score:.4f} ({c.num_runs})  {t}  {c.conf.aligned_str()}"
        )
    return "\n".join(lines)


def top_confs_report_rich(
    confs: list[ConfScore], max_weights: int, max_time: float | None, system: str | None
) -> str:
    sys = "" if system is None else f" ({system})"
    t = "" if max_time is None else f" T ≤ {ttoa3(max_time)}"
    table = Table(title=f"Top confs W ≤ {itoa3(max_weights)}{t}{sys}")

    table.add_column('Loss')
    table.add_column('Time')
    table.add_column('Length', justify='right')
    table.add_column('Batch', justify='right')
    table.add_column('LR')
    table.add_column('Steps', justify='right')
    table.add_column('P')
    table.add_column('Model', overflow='fold')


    for c in confs:
        t = '?' if c.median_time is None else ttoa3(c.median_time)
        table.add_row(f'{c.median_score:.4f} ({c.num_runs})', t, str(c.conf.length),
                      str(c.conf.batch), f'{c.conf.lr:.4f}',
                      str(c.conf.steps), str(c.conf.precision),
                      f'{c.conf.model} ({c.conf.model.weights})')

    return table


def _generate_limits():
    """Generate a sequence of expected numbers of completed runs per position
    in the top confs.

    Pairs [x0, x1] mean at least x0 runs for the configuration
    itself, at least x1 runs for the direct neighbors./

    Sequence:

    =0 [1, 0]
    =0 [2, 0]
    =0 [2, 1]
    =0 [2, 1]  <3 [1, 0]
    =0 [3, 1]  <3 [1, 0]
    =0 [3, 2]  <3 [1, 0]
    =0 [3, 2]  <3 [2, 0]
    =0 [3, 2]  <3 [2, 1]  <9 [1, 0]
    =0 [4, 2]  <3 [2, 1]  <9 [1, 0]
    =0 [4, 3]  <3 [2, 1]  <9 [1, 0]
    =0 [4, 3]  <3 [3, 1]  <9 [1, 0]
    =0 [4, 3]  <3 [3, 2]  <9 [1, 0]
    =0 [4, 3]  <3 [3, 2]  <9 [2, 0]
    =0 [4, 3]  <3 [3, 2]  <9 [2, 1]
    =0 [4, 3]  <3 [3, 2]  <9 [2, 1]  <27 [1, 0]
    """
    seq = [[1, 0]]
    while True:
        yield seq

        incremented = False
        for row in seq:
            if row[0] > row[1] + 1:
                row[1] += 1
                incremented = True
                break

        if incremented:
            continue

        for i in range(len(seq) - 1):
            if seq[i][0] > seq[i+1][0] + 1:
                seq[i+1][0] += 1
                incremented = True
                break

        if incremented:
            continue

        if seq[-1][0] > 1:
            seq.append([1, 0])
            continue

        seq[0][0] += 1


class Search(object):
    """Keep track of the configurations and selects what to try next."""

    def __init__(
        self,
        db: ResultDB,
        template: Template,
        init_conf: Configuration2,
        train_time: tuple[float, float],
    ):
        assert isinstance(db, ResultDB)
        self._db = db

        assert isinstance(template, Template)
        self._template = template

        assert isinstance(init_conf, Configuration2)
        self._init_conf = init_conf

        assert isinstance(train_time[0], float)
        assert isinstance(train_time[1], float)
        self._train_time = train_time

    def add_run(
        self,
        conf: Configuration2,
        run: Run,
    ):
        assert isinstance(conf, Configuration2)
        assert isinstance(run, Run)
        # assert self._template.match(conf)

        self._db.add_run(conf, run, update_neighbors=False)

    def _select_time(self) -> float:
        tmin, tmax = self._train_time
        if tmin == tmax:
            return tmin
        assert 0 < tmin < tmax
        return 2 ** random.uniform(log2(tmin), log2(tmax))

    def _select_max_weights(self, t, system: str):
        with latency.timer("Search._select_max_weights"):
            try:
                top_conf = next(
                    self._db.top_confs(max_time=t, system=system, limit=1, template=self._template)
                )
            except StopIteration:
                return self._template.max_weights.min
            maxw = 8 * top_conf.conf.model.weights
            if self._template.max_weights.min >= maxw:
                return self._template.max_weights.min
            if self._template.max_weights.max <= maxw:
                maxw = self._template.max_weights.max
            l = random.uniform(log2(self._template.max_weights.min), log2(maxw))
            return round(2**l)

    def _select_untimed(self, t: float, max_weights: int, system: str):
        with latency.timer("Search._select_untimed"):
            confs = list(
                self._db.top_confs(
                    max_weights=max_weights,
                    system=system,
                    limit=10,
                    template=self._template
                )
            )
            if all(c.median_time is not None for c in confs):
                return None
            selected = None
            for i, c in enumerate(confs):
                # Select a conf if it's a) hasn't been run on the given system,
                # b) there are no similar with fewer steps that took longer
                # than `t` on this system.
                if (c.median_time is None and
                    all(steps > c.conf.steps or median_time < t
                        for steps, median_time
                        in self._db.get_conf_runs_diff_steps(c.conf, system))):
                    selected = i
                    break
            if selected is not None:
                console.log(top_confs_report_rich(confs=confs, max_weights=max_weights, max_time=t, system=system))
                console.log('Selecting conf', i)
                return confs[selected].conf

    def _select_neighbor_fewest_runs(self, conf_id: int, system: str) -> Optional[ConfScore]:
        for neighbor_cs in self._db.get_neighbors_by_runs(conf_id, system):
            if self._template.match(neighbor_cs.conf):
                return neighbor_cs
        return None

    def _select_top_neighbor(
            self, t: float, max_weights: int, system: str
    ) -> Optional[Configuration2]:
        with latency.timer("Search._select_top_neighbor"):
            top_confs = list(
                self._db.top_confs(
                    max_time=t, max_weights=max_weights, system=system, limit=10, template=self._template
                )
            )
            if not top_confs:
                return None
            console.log(top_confs_report_rich(confs=top_confs, max_weights=max_weights, max_time=t, system=system))

            have_confs = 10
            min_runs_neighbor = []

            for seq in _generate_limits():
                for i, (min_self, min_neighbor) in enumerate(seq):
                    end = 3**i
                    start = end // 3

                    if end > have_confs:
                        console.log('Getting top confs up to ', end)
                        top_confs = list(
                            self._db.top_confs(
                                max_time=t, max_weights=max_weights, system=system,
                                limit=end, template=self._template
                            )
                        )
                        have_confs = end

                    for j in range(start, min(end, len(top_confs))):
                        if top_confs[j].num_runs < min_self:
                            console.log('Getting conf', j, 'because it has',
                                        top_confs[j].num_runs, 'runs <', min_self)
                            return top_confs[j].conf

                    if min_neighbor == 0:
                        continue

                    for j in range(start, min(end, len(top_confs))):
                        if len(min_runs_neighbor) < j + 1:
                            top = top_confs[j]
                            neighbor = self._select_neighbor_fewest_runs(
                                top.conf_id, system)
                            min_runs_neighbor.append(None)
                            min_runs_neighbor[j] = neighbor

                        neighbor = min_runs_neighbor[j]
                        if neighbor.num_runs < min_neighbor:
                            console.log('Getting neighbor of conf', j, 'because it has',
                                        neighbor.num_runs, 'runs <', min_neighbor)
                            return neighbor.conf

    def select_conf(self, system: str) -> tuple[Configuration2, float]:
        """Select a configuration to run.

            Returns:
                (configuration to run, soft time limit in seconds)
        """
        with latency.timer("Search.select_conf"):
            t = self._select_time()
            max_weights = self._select_max_weights(t, system)

            conf = self._select_untimed(t, max_weights, system)
            if conf is not None:
                tl = min(4*t, self._train_time[1])
                console.log(f'Conf for {system}: {conf}, TL: {tl}')
                return conf, tl

            conf = self._select_top_neighbor(t, max_weights, system)
            if conf is not None:
                console.log(f'Conf for {system}: {conf}, TL: {ttoa3(t)}')
                return conf, t

            console.log(f'Conf for {system}: {self._init_conf} (default), TL: {ttoa3(tmax)}')
            return self._init_conf, tmax
