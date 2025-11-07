import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from rich import print as pprint

from .configuration2 import Configuration2, Precision
from .dataset import DataSet
from .manager import Manager
from .model3 import build_model
from .tokens import TokenSet, get_tokenizer, set_tokens_dir


def show_loss_graph(manager: Manager, output_dir: str):
    run = manager.run

    scaled_step_loss = np.array(run.step_loss) / manager.tokenizer.tokenset.avg_bytes_per_token


    steps = run.steps
    xs = np.array(range(steps // 2, steps * 4 + 1))
    ys = run.loss_trend.predict(xs) / manager.tokenizer.tokenset.avg_bytes_per_token

    steps2 = steps * 2
    loss2 = run.loss_trend.predict([steps2])[0] / manager.tokenizer.tokenset.avg_bytes_per_token
    # print(loss2)
    logging.info(f"Expected loss at {steps2} steps: {loss2:.4f}")

    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(top=10)
    plt.yticks([1, 1.5, 2, 3, 4, 5, 6, 8], ['1', '1.5', '2', '3', '4', '5', '6', '8'])
    plt.plot(
        range(1, steps + 1),
        scaled_step_loss,
    )

    plt.plot(xs, ys)

    for checkpoint in manager.run.checkpoints.values():
        plt.plot(checkpoint.step + 1, checkpoint.loss, 'ro')
    plt.plot(steps + 1, run.loss, 'ro')

    if output_dir is not None:
        name = manager.name().replace('|', '!')
        plt.savefig(os.path.join(output_dir, name + ".png"))
    plt.show()


def parse_lr(x: str) -> float:
    if x.startswith("2^"):
        return 2 ** int(x[2:])
    return float(x)


def train(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)

    train_set = DataSet(path=args.data, path_processed=args.data_processed)
    lr = parse_lr(args.lr)
    decay = parse_lr(args.decay)

    if args.model_path is not None:
        manager = Manager.load(
            args.model_path,
            test_batch=args.test_batch,
            test_sample_len=args.test_sample_len,
            system=args.system,
        )
        manager.update_conf(lr, args.sample_len, args.batch, args.time, precision=Precision(args.precision))
        if args.add_layers:
            manager.add_layers(args.add_layers)
        manager.init()
    else:
        conf = Configuration2(
            build_model(args.spec),
            precision=args.precision,
            lr=lr,
            length=args.length,
            batch=args.batch,
            steps=args.steps,
            optimizer=args.optimizer,
            decay=decay,
        )
        manager = Manager(
            conf,
            test_batch=args.test_batch,
            test_sample_len=args.test_sample_len,
            system=args.system,
            dataset=train_set,
        )
        manager.init()

    manager.train_and_eval(
        args.steps,
        args.time,
        args.temp_steps,
        args.temp_dir,
        args.output_dir,
        args.log,
    )

    if args.prefix is not None:
        s = manager.continue_prefix(args.prefix, 256, temperature=args.temperature)
        print()
        print(s)
        print()

    if not args.no_graph:
        show_loss_graph(manager, args.output_dir)


def init_args(parser: argparse.ArgumentParser, config):
    # Data
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default=config.DATA,
        help=f"a file with (default: '{config.DATA}')",
    )
    parser.add_argument(
        "--data-processed",
        type=str,
        help=f"training data that has been already processed with capswords filter (default: '{config.DATA_CAPS_WORDS}')",
        default=config.DATA_CAPS_WORDS,
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
        "--tokens-dir",
        type=str,
        default=config.TOKENS_DIR,
        help=f"directory with token sets (default: '{config.TOKENS_DIR}')",
    )
    parser.add_argument(
        "--tokens",
        type=str,
        default=None,
        help="path to the token set definition",
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        choices=["fp64", "fp32", "fp16", "bf16"],
        metavar="P",
        default="fp32",
        help="training precision",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        metavar="N",
        default=32,
        help="batch size (default: 32)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=['adam', 'fromage'],
        metavar="O",
        default="adam",
        help="the optimizer algorithm",
    )
    parser.add_argument(
        "--lr",
        type=str,
        default="0.0078125",
        help="learning rate, could be written as a float or as 2^-10 (default: 1/128)",
    )
    parser.add_argument(
        '--decay',
        type=str,
        default="1.0",
        help="decay of the learning rate over the course of training, i.e. (LR at the last step) / (LR at the first step)  (default: 1)",
    )
    parser.add_argument(
        "-l",
        "--length",
        type=int,
        default=128,
        metavar="NTOKENS",
        help="length in tokens of text fragments used for training (default: 128)",
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
        default=None,
        metavar="PATH",
        help="path to a CSV file for logging",
    )

    # Evaluation
    parser.add_argument(
        "--prefix",
        type=str,
        default="Roses are red\nViolets are",
        help="text prefix to be continued",
    )
    parser.add_argument(
        "--test-batch",
        type=int,
        default=1024,
        help="Size of the test batch, used for final evaluation (default: 1024)",
    )
    parser.add_argument(
        "--test-sample-len",
        type=int,
        default=1024,
        help="Length in bytes of test samples, used for final evaluation",
    )
    parser.add_argument(
        "--no-graph",
        default=False,
        action="store_true",
        help="Don't show trianing loss graph",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="temperature for sampling (default: 0.1)",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=config.SYSTEM_NAME,
        help=f"the name of the system that will be used to identify runs in the DB (default: '{config.SYSTEM_NAME}')",
    )

    parser.set_defaults(func=train)
