import argparse
import logging

from .configuration2 import Configuration2, Template, default_from_template
from .dataset import DataSet
from .manager import Manager
from .model2 import build_model
from .report import (
    draw_weight_loss_graph,
    draw_loss_by_time,
    generate_max_report,
    generate_param_report,
    generate_report_by_weight,
)
from .resultdb import ResultDB
from .search import Search
from .predict import Predictor2
from .tokens import set_tokens_dir
from . import latency


def generate_report(result_set, template, train_time, system: str, draw_weight_loss: bool):
    print(generate_report_by_weight(result_set, template, system))
    print()
    print(generate_max_report(result_set, template, train_time))
    # print("\nLearning Rate")
    # print(generate_param_report(result_set, template, lambda conf: conf.lr))
    # if template.sample_len[0] < template.sample_len[1]:
    #     print("\nSample Length")
    #     print(
    #         generate_param_report(
    #             result_set,
    #             template,
    #             lambda conf: conf.sample_len,
    #             is_float=False,
    #         )
    #     )
    # if template.batch is None or template.batch[0] < template.batch[1]:
    #     print("\nBatch")
    #     print(
    #         generate_param_report(
    #             result_set, template, lambda conf: conf.batch, is_float=False
    #         )
    #     )
    if draw_weight_loss:
        draw_weight_loss_graph(result_set, template, train_time)
    # draw_loss_by_time(result_set, template)


def search_loop(
    dataset: DataSet,
    search: Search,
    system: str,
    sync_tokens: bool,
):
    logging.info("Starting search")
    while True:
        conf = search.select_conf()
        # weights = None
        # if checkpoint is not None:
        #     logging.info(f"Checkpoint: {checkpoint}")
        #     weights = checkpoint.load_weights(checkpoints_path)

        # # TODO: Train from the checkpoint
        # manager = Manager(conf, weights=weights)

        manager = Manager(conf, system, sync_tokens=sync_tokens, dataset=dataset)
        manager.init(quiet=True)

        run, weights = manager.train_and_eval(
            steps=conf.steps,
            time_limit=None,
            temp_steps=None,
            temp_dir=None,
            output_dir=None,
            log=None,
            quiet=True,
        )

        search.add_run(conf, run, weights)


def main(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)
    try:
        dataset = DataSet(path=args.data, in_process=args.sync_tokens)
        template = Template.from_args(args)
        logging.info(f"Template: {template}")
        default = default_from_template(template, spec=args.default_spec)
        logging.info(f"Default configuration: {default}")
        assert template.match(default)

        if template.max_weights.min < default.model.weights:
            logging.info(f"Adjusting minimal number of weights to match the default configuration: {default.model.weights}")
            template.max_weights.min = default.model.weights

        result_db = ResultDB.from_args(args.db)
        # predictr = Predictor2(args.sample_timing, args.train_timing, result_db, extra_dbs)

        train_time = tuple(map(float, args.train_time.split("-")))
        assert len(train_time) in (1, 2)
        if len(train_time) == 1:
            train_time.append(train_time[0])

        search = Search(
            args.system,
            result_db,
            template,
            default,
            checkpoints_path=None,
            predictor=None,
            # predictor=predictor,
            train_time=train_time,
        )

        try:
            search_loop(dataset, search, system=args.system, sync_tokens=args.sync_tokens)
        except KeyboardInterrupt:
            logging.warning("Interrupted\n")

        generate_report(
            search._result_set,
            template,
            train_time=train_time,
            system=args.system,
            draw_weight_loss=args.weight_loss_graph,
        )
        print()
        latency.report()
    finally:
        dataset.join()


def init_args(parser: argparse.ArgumentParser, config):
    # Data
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default=config.DATA,
        help="a file with training data",
    )
    parser.add_argument(
        "--tokens-dir",
        type=str,
        default=config.TOKENS_DIR,
        help="directory with token sets",
    )

    # Template args
    parser.add_argument(
        "-s",
        "--spec-regex",
        type=str,
        default=None,
        help="regex covering the acceptable specs",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=str,
        default=None,
        help='range of acceptable batch sizes, for example "1-256"',
    )
    parser.add_argument(
        "-l",
        "--lr",
        type=str,
        default=None,
        help="range of acceptable learning rates",
    )
    parser.add_argument(
        "--length",
        type=str,
        default=None,
        help="range of acceptable training sample lengths",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="range for the number of training steps; lower bound >= 2",
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default="32-4294967296",
        help="range for the _maximal_ number of weights in the model",
    )
    parser.add_argument(
        "-t",
        "--train-time",
        default="1-16",
        help="range for the training time in seconds",
    )
    parser.add_argument("--default-spec", type=str, default=None, help="default model")

    parser.add_argument(
        "--db",
        type=str,
        default=config.DB,
        help="path to the SQLite database with the results, or a URL for a PostgreSQL database",
    )

    parser.add_argument(
        "--train-timing",
        type=str,
        default=config.TRAIN_TIMING,
        help="a file with measured train timings for each trainined configuration",
    )
    parser.add_argument(
        "--sample-timing",
        type=str,
        default=config.SAMPLE_TIMING,
        help="a file with measured timings of sample preparation",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=config.SYSTEM_NAME,
        help="the name of the system that will be used to identify runs in the DB",
    )
    parser.add_argument(
        "--sync-tokens",
        dest="sync_tokens",
        action="store_true",
        help="Load and toknize training data synchronously, before the training start.",
    )
    parser.add_argument(
        "--no-sync-tokens",
        dest="sync_tokens",
        action="store_false",
        help="Load and tokenize training data concurrently with training in a separate thread.",
    )
    parser.set_defaults(sync_tokens=config.SYNC_TOKENS)
    parser.add_argument(
        "--no-weight-loss-graph",
        dest="weight_loss_graph",
        action="store_false",
        help="Don't show the weight/loss graph after the search has finished",
    )
    parser.set_defaults(weight_loss_graph=True)

    parser.set_defaults(func=main)
