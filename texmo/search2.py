import logging
import random
from math import log2, sqrt
from rich.table import Table
from typing import Optional

from . import latency
from .common import INF, itoa3, ttoa3
from .configuration import Configuration, Template, conf_neighbors
from .resultdb import ResultDB, ConfScore
from .run import Run


def top_confs_report(
    confs: list[ConfScore], max_weights: int, max_time: Optional[float], system: Optional[str]
) -> str:
    sys = "" if system is None else f" ({system})"
    t = "" if max_time is None else f" T ≤ {ttoa3(max_time)}"
    lines = [f"Top confs W ≤ {itoa3(max_weights)}{t}{sys}:"]
    for c in confs:
        t = '   ?    ' if c.median_time is None else f'{c.median_time:7.3f}s'
        score_runs = f'{c.median_score:.4f} ({c.num_runs})'
        lines.append(f'    {score_runs:<11} {t}  {c.conf.aligned_str()}')
    return "\n".join(lines)


def top_confs_report_rich(
    confs: list[ConfScore], max_weights: int, max_time: float | None, system: str | None
) -> str:
    sys = "" if system is None else f" ({system})"
    t = "" if max_time is None else f" T ≤ {ttoa3(max_time)}"
    table = Table(title=f"Top confs W ≤ {itoa3(max_weights)}{t}{sys}")

    table.add_column('Model', overflow='fold')
    table.add_column('P')
    table.add_column('Len', justify='right')
    table.add_column('Batch', justify='right')
    table.add_column('Steps', justify='right')
    table.add_column('Loss')
    table.add_column('Time', justify='right')

    for c in confs:
        t = '?' if c.median_time is None else f'{c.median_time:.3f}'
        decay = '' if c.conf.decay == 1 else f'*{c.conf.decay:.4f}'

        table.add_row(
            f'{c.conf.model} ({c.conf.model.num_weights})',
            str(c.conf.precision),
            str(c.conf.length),
            str(c.conf.batch),
            c.conf.learning_str,
            str(c.conf.steps),
            f'{c.median_score:.4f} ({c.num_runs})',
            t)

    return table


def _generate_limits():
    """Generate a sequence of expected numbers of completed runs per position
    in the top confs.

    Pairs [x0, x1] mean at least x0 runs for the configuration
    itself, at least x1 runs for the direct neighbors.

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
    while seq[0][0] <= 7:
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
        init_conf: Configuration,
        train_time: tuple[float, float],
    ):
        assert isinstance(db, ResultDB)
        self._db = db

        assert isinstance(template, Template)
        self.template = template

        assert isinstance(init_conf, Configuration)
        self._init_conf = init_conf

        assert isinstance(train_time[0], float)
        assert isinstance(train_time[1], float)
        self.train_time = train_time

    def add_run(
        self,
        conf: Configuration,
        run: Run,
    ):
        assert isinstance(conf, Configuration)
        assert isinstance(run, Run)

        self._db.add_run(conf, run)

    def _select_time(self) -> float:
        tmin, tmax = self.train_time
        if tmin == tmax:
            return tmin
        assert 0 < tmin < tmax
        return 2 ** random.uniform(log2(tmin), log2(tmax))

    def _select_max_weights(self, t, system: str):
        with latency.timer("Search._select_max_weights"):
            try:
                top_conf = next(
                    self._db.top_confs_for_system(
                        max_time=t, system=system, limit=1, template=self.template)
                )
            except StopIteration:
                return self.template.max_weights.min
            maxw = 8 * top_conf.conf.model.num_weights
            if self.template.max_weights.min >= maxw:
                return self.template.max_weights.min
            if self.template.max_weights.max <= maxw:
                maxw = self.template.max_weights.max
            l = random.uniform(
                log2(self.template.max_weights.min), log2(maxw))
            return round(2**l)

    def _select_neighbor_fewest_runs(
        self, conf: Configuration, system: str
    ) -> Optional[tuple[Configuration, int, int]]:
        """Find the neighbor with fewest runs that matches the template.

        Returns (neighbor_conf, total_runs, system_runs) or None. The
        caller decides whether the candidate is urgent enough to run:
        a neighbor with system_runs == 0 is always eligible to be
        selected; otherwise it's compared against the min-runs threshold
        from _generate_limits.
        """
        neighbors = conf_neighbors(conf, self.template)
        if not neighbors:
            return None

        neighbor_ids = [
            cid for cid in (self._db.get_conf_id(n) for n in neighbors)
            if cid is not None
        ]
        run_counts = self._db.get_run_counts(neighbor_ids, system=system)

        # Prefer neighbors with system_runs == 0 (bootstrap visibility on
        # this system), then among the rest pick the one with the fewest
        # total runs.
        best_conf = None
        best_total = INF
        best_system = INF
        for n in neighbors:
            cid = self._db.get_conf_id(n)
            if cid is None:
                total, sys_runs = 0, 0
            else:
                total, sys_runs = run_counts.get(cid, (0, 0))

            # Tuple ordering: (system_runs == 0 first, then fewest total).
            key = (sys_runs > 0, total)
            best_key = (best_system > 0, best_total)
            if key < best_key:
                best_conf = n
                best_total = total
                best_system = sys_runs

        if best_conf is not None:
            return best_conf, best_total, best_system
        return None

    def _select_top_neighbor(
            self, t: float, max_weights: int, system: str
    ) -> Optional[Configuration]:
        with latency.timer("Search._select_top_neighbor"):
            top_confs = list(
                self._db.top_confs_for_system(
                    max_time=t, max_weights=max_weights, system=system,
                    limit=10, template=self.template
                )
            )
            if not top_confs:
                return None
            logging.info(top_confs_report(confs=top_confs,
                         max_weights=max_weights, max_time=t, system=system))

            have_confs = 10
            min_runs_neighbor = []

            for seq in _generate_limits():
                for i, (min_self, min_neighbor) in enumerate(seq):
                    end = 3**i
                    start = end // 3

                    if end > have_confs:
                        logging.info(f'Getting top confs up to {end}')
                        top_confs = list(
                            self._db.top_confs_for_system(
                                max_time=t, max_weights=max_weights, system=system,
                                limit=end, template=self.template
                            )
                        )
                        have_confs = end

                    for j in range(start, min(end, len(top_confs))):
                        if top_confs[j].num_runs < min_self:
                            logging.info(
                                f'Getting conf {j} because it has {top_confs[j].num_runs} runs < {min_self}')
                            return top_confs[j].conf

                    if min_neighbor == 0:
                        continue

                    for j in range(start, min(end, len(top_confs))):
                        if len(min_runs_neighbor) < j + 1:
                            result = self._select_neighbor_fewest_runs(
                                top_confs[j].conf, system)
                            min_runs_neighbor.append(result)

                        result = min_runs_neighbor[j]
                        if result is not None:
                            neighbor_conf, total_runs, system_runs = result
                            # Pick the neighbor if:
                            # (a) it has no runs on this system yet, or
                            # (b) its total run count is below the threshold.
                            if system_runs == 0:
                                logging.info(
                                    f'Getting neighbor of conf {j}: {system_runs} runs on {system}')
                                return neighbor_conf
                            if total_runs < min_neighbor:
                                logging.info(
                                    f'Getting neighbor of conf {j}: {total_runs} total runs < {min_neighbor}')
                                return neighbor_conf

    def select_conf(self, system: str) -> Optional[Configuration]:
        """Select a configuration to run, or None if nothing matches.

        Returns None when the top-conf search finds nothing and the
        fallback init_conf either doesn't match the current template
        or has already been explored enough (>=7 total runs with at
        least one on this system).
        """
        with latency.timer("Search.select_conf"):
            t = self._select_time()
            max_weights = self._select_max_weights(t, system)

            conf = self._select_top_neighbor(t, max_weights, system)
            if conf is not None:
                logging.info(f'Conf for {system}: {conf}')
                return conf

            # Template can change on the fly — re-check init_conf still
            # matches before falling back to it.
            if not self.template.match(self._init_conf):
                logging.info(
                    f'No conf for {system}: init_conf does not match template')
                return None

            # Skip init_conf if it's already been explored enough.
            conf_id = self._db.get_conf_id(self._init_conf)
            if conf_id is not None:
                counts = self._db.get_run_counts([conf_id], system=system)
                total_runs, system_runs = counts.get(conf_id, (0, 0))
                if total_runs >= 7 and system_runs >= 1:
                    logging.info(
                        f'No conf for {system}: init_conf has {total_runs} '
                        f'total runs ({system_runs} on this system)')
                    return None

            logging.info(
                f'Conf for {system}: {self._init_conf} (default)')
            return self._init_conf
