import argparse

from texmo.configuration import Template, Configuration, conf_neighbors
from texmo.model2 import build_model

def main(spec_regex, spec_default, max_weights):
    template = Template(
        spec_regex=spec_regex,
        batch=(256, 256),
        lr=(1.0, 1.0),
        sample_len=(128, 128),
        init_scale=(1.0, 1.0),
        t=(1.0, 1.0),
        max_weights=max_weights,
    )
    model = build_model(spec_default)
    conf = Configuration(None, model, 1.0, 128, 256, 1.0, 1.0, 1)
    for neighbor in conf_neighbors(conf, template):
        print(neighbor.model)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--spec-regex",
        type=str,
        default=None,
        help="regex convering the acceptable specs (default: unrestricted)",
    )
    parser.add_argument(
        "--spec-default",
        type=str,
        default="dense.1.relu",
        help="initial spec (default: dense.1.relu)",
    )
    parser.add_argument(
        "--max-weights",
        type=int,
        default=None,
        help="max weights. Will vary if left undefined",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
