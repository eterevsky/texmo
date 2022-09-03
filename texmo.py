import argparse
from collections import namedtuple
import json
import os

from dataset import DataSet
import layered
from manager import Manager
import search
from train import train_and_validate








def benchmark(data, time_limit, log):
    results = []
    for layered in (
        "gru.128-gru.256",
        "gru.128-gru.512",
        "gru.128-gru.1024",
        "gru.128-gru.512.relu",
        "gru.128-gru.1024.relu",
    ):
        for learning_rate in (0.05, 0.02, 0.01):
            for batch_size in (256, 512):
                loss, step, data_size = main(
                    data,
                    1000000,
                    learning_rate,
                    regularization=0.1,
                    output_dir=None,
                    model_path=None,
                    layered_model=layered,
                    temp_dir=None,
                    sample_length=128,
                    batch_size=batch_size,
                    temp_steps=0,
                    time_limit=time_limit,
                    skip_graph=True,
                    log=log,
                )
                results.append(
                    (
                        layered,
                        learning_rate,
                        batch_size,
                        step,
                        data_size,
                        loss,
                    )
                )
    for r in results:
        layered, lr, batch_size, step, data_size, loss = r
        print(
            f"{layered:30} {step:5} {batch_size:3} {data_size:4.1f}M {loss:.4f} L{lr:.3f}"
        )


def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="directory with training data",
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
        "-n",
        "--model-name",
        metavar="NAME",
        default=None,
        help="parse model name",
    )
    model_group.add_argument(
        "-c",
        "--layered-model",
        metavar="SPEC",
        default=None,
        help="layer-by-layer model specification",
    )
    model_group.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help="run a benchmark with various configurations",
    )
    model_group.add_argument(
        "--search",
        metavar="SPEC",
        default=None,
        help="search the best model configuration, starting with a given spec",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        metavar="PATH",
        default=None,
        help="directory for saved model",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="Roses are red\nViolets are blu",
        help="text prefix to be continued",
    )
    parser.add_argument(
        "--max-weights",
        type=int,
        default=None,
        help="for search, the maximum number of weights in the model",
    )
    parser.add_argument(
        "-v",
        "--vary",
        type=str,
        default="struct,lr,batch",
        help="model parameters that can be varied with search. "
        + "A comma-separated list of struct, size, act, lr, batch, len.",
    )

    # Training
    parser.add_argument(
        "-s", "--steps", type=int, default=None, help="number of training steps"
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        metavar="SECONDS",
        help="time limit for training",
        default=None,
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        type=float,
        metavar="RATE",
        help="learning rate",
        default=0.05,
    )
    parser.add_argument(
        "-r",
        "--regularization",
        type=float,
        help="L2 regularization coefficient",
        default=0.1,
    )
    parser.add_argument(
        "--sample-length",
        type=int,
        default=256,
        metavar="LEN",
        help="length of text fragments used for training",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        metavar="BATCH",
        default=256,
        help="batch size",
    )

    # Intermediate models
    parser.add_argument(
        "-t",
        "--temp-dir",
        default=None,
        metavar="PATH",
        help="directory for intermediate models",
    )
    parser.add_argument(
        "--temp-steps",
        metavar="N",
        type=int,
        default=1000,
        help="save intermediate model every N steps",
    )

    parser.add_argument(
        "--skip-graph",
        action="store_true",
        default=False,
        help="do not generate the loss graph",
    )
    parser.add_argument(
        "--log",
        default=None,
        metavar="LOG",
        help="path to a CSV file for logging",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.search:
        search.search(
            args.data,
            args.learning_rate,
            args.sample_length,
            args.batch_size,
            args.regularization,
            args.time_limit,
            args.log,
            start_spec=args.search,
            max_weights=args.max_weights,
            vary=args.vary,
        )
    elif args.benchmark:
        benchmark(args.data, args.time_limit, args.log)
    else:
        main(**vars(args))
