import argparse
import json
import sys
import time

from texmo.dataset import DataSet
from texmo.manager import Manager

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


def eval(dataset, manager) -> float:
    """Evaluate a model on a random set of training data and/or continue a prefix."""
    batch = dataset.sample(1024, 1024)
    return manager.evaluate(batch)


def main(data, model_path, prefix):
    if data:
        print(f"Loading dataset from {data}")
        dataset = DataSet(data)

    print("Parsing model... ", end='')
    with open(model_path) as f:
        model_json = json.load(f)
    print("done")
    manager = Manager.from_spec(model_json)
    manager.init(training=False)

    if data:
        loss = eval(dataset, manager)
        print(f'Loss: {loss}')

    continue_prefix(manager, prefix, 256)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="directory with training data",
    )
    parser.add_argument(
        "-m",
        "--model-path",
        metavar="PATH",
        required=True,
        help="load trained model from file",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="Roses are red\nViolets are blu",
        help="text prefix to be continued",
    )

    return parser.parse_args()


if __name__ == "__main__":
    print("TexMo model evaluation")
    args = parse_args()
    main(**vars(args))
