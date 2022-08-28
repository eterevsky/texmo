import argparse
from collections import namedtuple
import json
import matplotlib.pyplot as plt
import os

from dataset import DataSet
import layered
from manager import Manager
import models
from record import TrainingRecord
import search
from train import train_and_validate


def continue_prefix(manager, prefix, length):
    prefix = prefix.encode()
    out = manager.sample(prefix, length)

    try:
        s = (prefix + out).decode("utf-8")
    except UnicodeDecodeError:
        s = repr(prefix + out)
    print()
    print(s)
    print()


def show_loss_graph(manager, output_dir):
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(top=8)
    plt.plot(range(1, len(manager.step_loss) + 1), manager.step_loss)
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, manager.name() + ".png"))
    plt.show()


def create_manager(
    model_name, layered_model, model_path, learning_rate, regularization
):
    if model_name is not None:
        model = models.parse(model_name)
        manager = Manager(model, learning_rate, regularization, 100000)
    elif layered_model is not None:
        model = layered.LayeredModel2.parse(layered_model)
        manager = Manager(model, learning_rate, regularization, 100000)
    else:
        assert model_path is not None
        with open(model_path) as f:
            spec = json.load(f)
        manager = Manager.from_spec(spec)

    manager.init()
    return manager


def main(
    data,
    steps,
    learning_rate,
    regularization,
    output_dir,
    model_name,
    model_path,
    layered_model,
    temp_dir,
    sample_length,
    batch_size,
    temp_steps,
    time_limit,
    log,
    prefix="Roses are red\nViolets are blu",
    skip_graph=False,
    benchmark=False,
):
    # If benchmark is true, benchmark() should be called instead of main().
    assert not benchmark

    if data is not None:
        print(f"Training data: {data}")
        train_set = DataSet(data)
    else:
        train_set = None

    manager = create_manager(
        model_name, layered_model, model_path, learning_rate, regularization
    )

    if train_set is not None:
        report = train_and_validate(
            manager,
            steps,
            time_limit,
            train_set,
            sample_length,
            batch_size,
            temp_steps,
            temp_dir,
            output_dir,
            log,
        )
    else:
        report = None

    if prefix is not None:
        continue_prefix(manager, prefix, 256)

    if (
        steps is not None or time_limit is not None or data is not None
    ) and not skip_graph:
        show_loss_graph(manager, output_dir)

    return report


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
                    model_name=None,
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
        action="store_true",
        default=False,
        help="search the best model configuration",
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
        default=0.02,
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
        )
    elif args.benchmark:
        benchmark(args.data, args.time_limit, args.log)
    else:
        main(**vars(args))
