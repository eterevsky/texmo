import argparse
import logging

import numpy as np

from .. import latency
from ..configuration2 import Template
from ..resultdb import ResultDB
from ..results import ResultSet
from ..tokens import set_tokens_dir
from .loss_predictor_flat import LossPredictorFlat


def main(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)

    np.set_printoptions(linewidth=100, edgeitems=6, precision=3)

    logging.info(f"Loading results from {args.db}")
    result_db = ResultDB(args.db)
    result_set = ResultSet(
        result_db, template=Template(), populate_neighbors=False
    )

    # logging.info("Create legacy predictor")
    # legacy_predictor = LossPredictorV1(result_set, split_test_set=True)

    # logging.info("Training")
    # legacy_predictor.train()
    
    logging.info("Creating new predictor")
    predictor = LossPredictorFlat(result_db)

    logging.info("Training")
    predictor.train()
    
    latency.report()


def init_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--tokens-dir",
        type=str,
        required=True,
        help="directory with token sets",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="path to the SQLite database with the results",
    )
    parser.set_defaults(func=main)
