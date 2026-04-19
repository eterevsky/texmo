import argparse
from io import StringIO

from .common import INF, console, ttoa3
from .configuration import Template
from .resultdb import ResultDB
from .tokens import set_tokens_dir

# def max_points(result_set: ResultSet, t: int, maxx: int):
#     x = []
#     y = []
#     max_loss = 5

#     weight_loss = []
#     max_weights = INF
#     while True:
#         conf_results = result_set.top_conf(t, max_weights)
#         if not conf_results:
#             break
#         conf = conf_results.conf
#         weight_loss.append((conf.model.weights, conf_results.median_score))
#         max_weights = conf.model.weights - 1

#     prev_loss = max_loss

#     for weight, loss in reversed(weight_loss):
#         if loss > max_loss:
#             continue
#         x.append(weight - 1)
#         y.append(prev_loss)

#         x.append(weight)
#         y.append(loss)

#         prev_loss = loss

#     x.append(maxx)
#     y.append(prev_loss)

#     return x, y


# def draw_weight_loss_graph(
#     result_set: ResultSet, template: Template, train_time: tuple[float, float]
# ):
#     fig, ax = plt.subplots()
#     ax.set_ylabel('cross-entropy (bits per byte)')
#     ax.set_xlabel('weights')
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.grid(visible=True)
#     ax.set_yticks([1, 2, 3, 4])
#     ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
#     ax.yaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
#     yticks = [x / 10 for x in range(1, 30)] + [
#         x / 10 for x in range(30, 50, 2)
#     ]
#     ax.set_yticks(yticks, minor=True)

#     t = 1
#     legends = []

#     top_conf_results = result_set.top_conf(train_time[1])

#     while t <= train_time[1]:
#         if t >= train_time[0]:
#             x, y = max_points(
#                 result_set, t, top_conf_results.conf.model.weights * 4
#             )
#             ax.plot(x, y)
#             legends.append(t)
#         t *= 2

#     ax.legend(legends)

#     plt.savefig('results/graph.png')
#     plt.show()


# def draw_loss_by_time(result_set: ResultSet, template: Template):
#     _, ax = plt.subplots()
#     ax.set_xlabel('time (s)')
#     ax.set_ylabel('cross-entropy (bits per byte)')
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.grid(visible=True)
#     ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
#     ax.xaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
#     ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
#     ax.yaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())

#     times = []
#     t = template.t[0]
#     while t <= template.t[1]:
#         times.append(t)
#         t *= 2

#     best = []
#     best_for_bytes = []

#     for t in times:
#         best.append(result_set.top_conf(t).median_score)
#         bytes_top_conf = result_set.top_conf_for_tokenset(
#             t, 'tokens256_raw_all'
#         )
#         best_for_bytes.append(
#             bytes_top_conf.median_score if bytes_top_conf else 4
#         )

#     ax.plot(times, best, label='best')
#     ax.plot(times, best_for_bytes, label='best for bytes')

#     plt.show()


# def get_top_confs(
#     result_set: ResultSet,
#     template: Template,
#     min_max_weights: int,
#     train_time: tuple[float, float],
#     system: str,
# ):
#     top_confs = {}  # (weights_limit, planned_time_s) -> (conf, score)
#     count = {}  # (weights_limit, planned_time_s) -> count
#     tlo, thi = train_time
#     for conf_results in result_set.get_results_by_weights():
#         t = conf_results.median_time(system)
#         if not conf_results.median_score or not t:
#             continue
#         t = 2 ** math.ceil(math.log2(t))
#         if not tlo <= t <= thi:
#             continue
#         conf = conf_results.conf
#         weights = conf.model.weights
#         weights_bucket = max(
#             2 ** math.ceil(math.log2(weights)), min_max_weights
#         )
#         try:
#             count[(weights_bucket, t)] += len(conf_results.runs)
#         except KeyError:
#             count[(weights_bucket, t)] = len(conf_results.runs)

#         while weights_bucket < 2**33:
#             key = (weights_bucket, t)
#             if key in top_confs:
#                 top_conf, top_score = top_confs[key]
#                 if (
#                     conf_results.median_score is not None
#                     and conf_results.median_score < top_score
#                 ):
#                     assert conf_results.median_score is not None
#                     top_confs[key] = (conf, conf_results.median_score)
#             else:
#                 assert conf_results.median_score is not None
#                 top_confs[key] = (conf, conf_results.median_score)
#             weights_bucket *= 2

#     return top_confs, count


def print_top_confs(top_confs, run_count) -> str:
    out = StringIO()
    spec_len = 0
    for conf, score in top_confs.values():
        l = len(str(conf.model))
        if l > spec_len:
            spec_len = l
    for log_weights in range(6, 25):
        weights = 2**log_weights
        first_weight_line = True
        for log_t in range(0, 10):
            t = 2**log_t
            if (weights, t) not in top_confs:
                continue
            conf, score = top_confs[(weights, t)]
            count = run_count.get((weights, t), 0)
            if (
                count == 0
                or log_weights > 10
                and (weights // 2, t) in top_confs
                and top_confs[(weights // 2, t)][0] == conf
            ):
                continue
            if first_weight_line:
                print(
                    '| ------- | ---- | ---- |',
                    '-' * spec_len,
                    '| ------ | --------- |',
                    file=out,
                )
                print(f'| {weights:>7} | ', file=out, end='')
                first_weight_line = False
            else:
                print('|         | ', file=out, end='')
            print(f'{t:>4} | {count:>4} | ', file=out, end='')
            model_str = str(conf.model)
            print(
                model_str,
                ' ' * (spec_len - len(model_str)),
                file=out,
                end='',
            )
            print(f'| {score:.4f} | ', file=out, end='')
            print(f'B{conf.batch} LR{conf.lr:.4f}', file=out, end='')
            print(f' LEN{conf.length}', file=out, end='')

            print(' |', file=out)

    return out.getvalue()


def generate_report_by_weight(
    result_db: ResultDB, template: Template, system: str, max_time: float
) -> str:
    lines = [
        f'Top confs by weights for {system}\n',
        f'Max time: {ttoa3(max_time)}',
        f'Spec: {template.regex.pattern if template.regex else None}',
        f'Optimizer: AdamW',
        f'LR: {template.lr}',
        f'Decay: {template.decay}',
        f'Precision: {', '.join(template.precision)}',
        f'Length: {template.length}',
        f'Batch: {template.batch}',
        f'Steps: {template.steps}',
        f'Weights: {template.max_weights}\n',
    ]
    confs = result_db.top_confs(
        system, sorted='weights', template=template, with_runs=True
    )
    best_score = INF
    for c in confs:
        if c.median_score > best_score:
            continue
        if not template.match(c.conf):
            continue
        if c.median_time is None:
            continue
        if c.median_time > max_time:
            continue
        t = '{:8}'.format(ttoa3(c.median_time))
        lines.append(
            f'{c.median_score:.4f} ({c.num_runs})  {t}  {c.conf.aligned_str()}'
        )
        best_score = c.median_score
    return '\n'.join(lines) + '\n'


def generate_max_report(
    result_db: ResultDB,
    template: Template,
    train_time: tuple[float, float],
    system: str,
) -> str:
    out = StringIO()
    lo, hi = train_time
    assert 1 <= lo <= hi
    # print("generate_max_report", lo, hi)

    # runs_count = result_set.runs_count_per_t()
    t = lo
    while t <= hi:
        print(f'\nT ≤ {t}', file=out)
        print(
            generate_report_by_weight(
                result_db, template, system, max_time=t
            ),
            file=out,
        )

        t *= 2

    return out.getvalue()


def main(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)
    template = Template.from_args(args)
    console.log('Template:', template)

    db = ResultDB.from_args(args.db)
    for conf_score in db.top_confs_global(template):
        console.log(conf_score.conf)
        console.log(f'{conf_score.median_score:.3f} ({conf_score.num_runs})  {ttoa3(conf_score.median_time)} on {conf_score.system}')


def init_args(parser: argparse.ArgumentParser, config):
    parser.add_argument(
        "--db",
        type=str,
        default=config.DB,
        help="path to the SQLite database with the results, or a URL for a PostgreSQL database",
    )
    parser.add_argument(
        '--tokens-dir',
        type=str,
        default=config.TOKENS_DIR,
        help='directory with tokensets',
    )

    # Template args
    parser.add_argument(
        '-s',
        '--spec-regex',
        type=str,
        default=None,
        help='regex covering the acceptable specs',
    )
    parser.add_argument(
        '--length',
        type=str,
        default=None,
        help='range of acceptable training sample lengths',
    )
    parser.add_argument(
        '-b',
        '--batch',
        type=str,
        default=None,
        help='range of acceptable batch sizes, for example "1-256"',
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        metavar="O",
        default="adam,fromage",
        help="the optimizer algorithm",
    )
    parser.add_argument(
        '--decay',
        type=str,
        default="0.0-1.0",
        help="decay of the learning rate over the course of training, i.e. (LR at the last step) / (LR at the first step)",
    )
    parser.add_argument(
        '-l',
        '--lr',
        type=str,
        default=None,
        help='range of acceptable learning rates',
    )
    parser.add_argument(
        '--steps',
        type=str,
        default=None,
        help='range for the number of training steps; lower bound >= 2',
    )
    parser.add_argument(
        '-w',
        '--weights',
        type=str,
        default='2-4294967296',
        help='range for the _maximal_ number of weights in the model',
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        default="bf16,fp16,fp32",
        help="precision"
    )


    parser.set_defaults(func=main)
