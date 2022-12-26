import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

import config
from texmo.configuration import Configuration
from texmo.dataset import DataSet
from texmo.manager import Manager
from texmo.model2 import build_model


def show_loss_graph(manager, output_dir):
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
    plt.plot(range(1, steps + 1), run.step_loss)
    plt.plot(xs, ys)
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, manager.name() + ".png"))
    plt.show()


def parse_lr(x: str) -> float:
    if x.startswith("2^"):
        return 2 ** int(x[2:])
    return float(x)


def main(
    data,
    steps,
    lr,
    output_dir,
    model_path,
    spec,
    temp_dir,
    sample_len,
    batch,
    temp_steps,
    time,
    log,
    prefix,
    add_layers,
):
    print(f"Training data: {data}")
    train_set = DataSet(data)
    lr = parse_lr(lr)

    try:
        if model_path is not None:
            manager = Manager.load(model_path)
            manager.update_conf(lr, sample_len, batch, time)
            if add_layers:
                manager.add_layers(add_layers)
            manager.init()
        else:
            conf = Configuration(
                build_model(spec),
                lr,
                sample_len,
                batch,
                t=time,
            )
            manager = Manager(conf)
            manager.init()

        manager.train_and_eval(
            steps,
            time,
            train_set,
            temp_steps,
            temp_dir,
            output_dir,
            log,
        )
    finally:
        train_set.join()

    if prefix is not None:
        s = manager.continue_prefix(prefix, 256)
        print()
        print(s)
        print()

    show_loss_graph(manager, output_dir)


def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default=config.DATA,
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
        "-b",
        "--batch",
        type=int,
        metavar="BATCH",
        default=config.DEFAULT_BATCH,
        help="batch size",
    )
    parser.add_argument(
        "-l",
        "--lr",
        type=str,
        default=str(config.DEFAULT_LR),
        help="learning rate, could be written as a float or as 2^-10",
    )
    parser.add_argument(
        "--sample-len",
        type=int,
        default=128,
        metavar="LEN",
        help="length of text fragments used for training",
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
        metavar="N",
        type=int,
        default=1000,
        help="save intermediate model every N steps",
    )

    parser.add_argument(
        "--log",
        default=config.LOG,
        metavar="LOG",
        help="path to a CSV file for logging",
    )

    # Evaluation
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="Roses are red\nViolets are blu",
        help="text prefix to be continued",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()
    main(**vars(args))
