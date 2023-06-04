import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from .configuration import Configuration
from .dataset import DataSet
from .manager import Manager
from .model2 import build_model
from .tokens import TokenSet


def show_loss_graph(manager: Manager, output_dir: str):
    run = manager.run

    steps = run.steps
    xs = np.array(range(steps // 2, steps * 4 + 1))
    ys = run.loss_trend.predict(xs)

    steps2 = steps * 2
    loss2 = run.loss_trend.predict([steps2])[0]
    logging.info(f"Expected loss at {steps2} steps: {loss2:.4f}")

    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(top=8)
    plt.plot(
        range(1, steps + 1),
        list(map(manager.token_set.byte_loss, run.step_loss)),
    )
    plt.plot(xs, ys)
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, manager.name() + ".png"))
    plt.show()


def parse_lr(x: str) -> float:
    if x.startswith("2^"):
        return 2 ** int(x[2:])
    return float(x)


def train(args: argparse.Namespace):
    if args.tokens:
        token_set = TokenSet.from_json_file(args.tokens)
        ntokens = token_set.ntokens
        token_sets = {ntokens: token_set}
        logging.info(
            f"Loaded token set {args.tokens} with {token_set.ntokens} tokens"
        )
    else:
        token_set = None
        ntokens = 256
        token_sets = {}
        logging.info(f"No token set")
    logging.info(f"Training data: {args.data}")
    train_set = DataSet(args.data, token_sets=token_sets)
    lr = parse_lr(args.lr)

    try:
        if args.model_path is not None:
            manager = Manager.load(args.model_path, token_set)
            manager.update_conf(lr, args.sample_len, args.batch, args.time)
            if args.add_layers:
                manager.add_layers(args.add_layers)
            manager.init()
        else:
            conf = Configuration(
                build_model(ntokens, args.spec),
                lr,
                args.sample_len,
                args.batch,
                t=args.time,
            )
            manager = Manager(conf, token_set)
            manager.init()

        manager.train_and_eval(
            args.steps,
            args.time,
            train_set,
            args.temp_steps,
            args.temp_dir,
            args.output_dir,
            args.log,
        )
    finally:
        train_set.join()

    if args.prefix is not None:
        s = manager.continue_prefix(args.prefix, 256)
        print()
        print(s)
        print()

    show_loss_graph(manager, args.output_dir)


def init_args(parser: argparse.ArgumentParser):
    # Data
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        required=True,
        help="a file with",
    )

    # Model
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "-m",
        "--model-path",
        metavar="PATH",
        default=None,
        help="load trained model from file",
    )
    model_group.add_argument(
        "-s",
        "--spec",
        default=None,
        help="layer-by-layer model specification",
    )
    parser.add_argument(
        "-a",
        "--add-layers",
        type=str,
        metavar="SPEC",
        default=None,
        help="add and train layers to a pre-trained model loaded with -m",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        metavar="PATH",
        default=None,
        help="directory to save the trained model",
    )
    parser.add_argument(
        "--steps", type=int, default=None, help="number of training steps"
    )

    # Configuration
    parser.add_argument(
        "--tokens",
        type=str,
        help="path to the token set definition",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        metavar="N",
        default=32,
        help="batch size",
    )
    parser.add_argument(
        "-l",
        "--lr",
        type=str,
        default="0.0625",
        help="learning rate, could be written as a float or as 2^-10",
    )
    parser.add_argument(
        "--sample-len",
        type=int,
        default=128,
        metavar="NTOKENS",
        help="length in tokens of text fragments used for training",
    )
    parser.add_argument(
        "-t",
        "--time",
        type=int,
        metavar="SECONDS",
        help="time limit for training",
        default=None,
    )

    # Intermediate models
    parser.add_argument(
        "--temp-dir",
        default=None,
        metavar="PATH",
        help="directory for intermediate models",
    )
    parser.add_argument(
        "--temp-steps",
        metavar="STEPS",
        type=int,
        default=1000,
        help="save intermediate model every N steps",
    )

    parser.add_argument(
        "--log",
        default="log-mac.csv",
        metavar="PATH",
        help="path to a CSV file for logging",
    )

    # Evaluation
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="Roses are red\nViolets are",
        help="text prefix to be continued",
    )

    parser.set_defaults(func=train)
