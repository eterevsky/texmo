import argparse
import logging

from .configuration import (
    TOKEN_TYPES,
    Configuration,
    Template,
    conf_to_string,
    conf_tokens_name,
    default_from_template,
)
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


def warmup(dataset):
    conf = Configuration(
        model=build_model(32, "suffix.4-rec.32.relu"),
        ntokens=32,
        token_type="bits4",
        token_processing="capswords",
        lr=0.0625,
        sample_len=128,
        batch=32,
        t=1,
    )
    manager = Manager(conf)
    manager.init(quiet=True)
    manager.train_and_eval(
        steps=None,
        time_limit=1,
        train_set=dataset,
        temp_steps=None,
        temp_dir=None,
        output_dir=None,
        log=None,
        quiet=False,
    )


def generate_report(result_set, template, min_max_weights):
    if template.max_weights is None:
        print(generate_report_by_weight(result_set, template, min_max_weights))
        print()
    print(generate_max_report(result_set, template))
    print("\nLearning Rate")
    print(generate_param_report(result_set, template, lambda conf: conf.lr))
    if template.sample_len[0] < template.sample_len[1]:
        print("\nSample Length")
        print(
            generate_param_report(
                result_set,
                template,
                lambda conf: conf.sample_len,
                is_float=False,
            )
        )
    if template.batch is None or template.batch[0] < template.batch[1]:
        print("\nBatch")
        print(
            generate_param_report(
                result_set, template, lambda conf: conf.batch, is_float=False
            )
        )
    draw_weight_loss_graph(result_set, template)
    draw_loss_by_time(result_set, template)


def search_loop(
    dataset, search: Search, checkpoints_path: str
):
    logging.info("Warming up training")
    warmup(dataset)

    logging.info("Starting search")
    while True:
        conf, checkpoint = search.select_conf()
        weights = None
        if checkpoint is not None:
            logging.info(f"Checkpoint: {checkpoint}")
            weights = checkpoint.load_weights(checkpoints_path)

        # TODO: Train from the checkpoint
        manager = Manager(conf, weights=weights)
        manager.init(quiet=True)

        record, run, weights = manager.train_and_eval(
            steps=None,
            time_limit=conf.t,
            train_set=dataset,
            temp_steps=None,
            temp_dir=None,
            output_dir=None,
            log=None,
            quiet=True,
        )

        search.add_run(record, run, weights, parent_checkpoint=checkpoint)


def main(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)
    try:
        dataset = DataSet(args.data, tokens_dir=args.tokens_dir)
        template = Template.from_args(args)
        default = default_from_template(template, spec=args.default_spec)
        logging.info("Default configuration: " + conf_to_string(default))
        assert template.match_conf(default)

        result_db = ResultDB(args.db)
        extra_dbs = args.extra_db.split(",") if args.extra_db else []
        predictor = Predictor2(args.sample_timing, args.train_timing, result_db, extra_dbs)
        search = Search(
            result_db,
            template,
            default,
            args.min_max_weights,
            checkpoints_path=None,
            predictor=predictor,
            all_neighbors=args.all_neighbors,
        )

        try:
            search_loop(dataset, search, None)
        except KeyboardInterrupt:
            logging.warning("Interrupted\n")

        generate_report(search._result_set, template, args.min_max_weights)
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
        "--token-type",
        type=str,
        default="all,bits1,bits2,bits4",
        help="comma-separate list of allowed tokenset types",
    )
    parser.add_argument(
        "--token-processing",
        type=str,
        default="raw,capswords",
        help="comma-separated list of allowed tokenizer processors",
    )
    parser.add_argument(
        "--ntokens",
        type=str,
        default="2-16384",
        help="number of tokens in a token set",
    )
    parser.add_argument(
        "-s",
        "--spec-regex",
        type=str,
        default=None,
        help="regex covering the acceptable specs (default: unrestricted)",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=str,
        default="1-8192",
        help='range of acceptable batch sizes, for example "1-256" (default: unrestricted)',
    )
    parser.add_argument(
        "-l",
        "--lr",
        type=str,
        default="0.000001-10",
        help="range of acceptable learning rates (default: 0.001-10)",
    )
    parser.add_argument(
        "--sample-len",
        type=str,
        default="2-1024",
        help="range of acceptable sample lens (default: 128)",
    )
    parser.add_argument(
        "-t",
        "--time",
        type=str,
        default="1-256",
        help="range of training time (default: 1-256)",
    )
    parser.add_argument(
        "--max-weights",
        type=int,
        default=None,
        help="max weights. Will vary if left undefined (default: unrestricted)",
    )
    parser.add_argument(
        "--min-max-weights",
        type=int,
        default=32,
        help="minimum max-weights value in search",
    )
    parser.add_argument(
        "--default-spec", type=str, default=None, help="default model"
    )

    parser.add_argument(
        "--db",
        type=str,
        default=config.DB,
        help="path to the SQLite database with the results",
    )
    parser.add_argument(
        "--extra-db",
        type=str,
        default=config.EXTRA_DB,
        help="path to additional result DBs from other machines",
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
        "--all-neighbors",
        action="store_true",
        help="generate all neighbors of the configurations at startup, not just when selecting the neighbor of the top configuration"
    )

    parser.set_defaults(func=main)
