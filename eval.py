import argparse
import json

from texmo.dataset import DataSet
from texmo.manager import Manager

def main(data, model_path, prefix):
    manager = Manager.load(model_path, training=False)

    if data:
        try:
            print(f"Loading dataset from {data}")
            dataset = DataSet(data)
            loss = manager.eval(dataset)
            print(f'Loss: {loss}')
        finally:
            dataset.join()

    manager.continue_prefix(prefix, 256)


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
