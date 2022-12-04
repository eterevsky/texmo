import argparse
import logging
import numpy as np
from sklearn.model_selection import train_test_split

from texmo import latency
from texmo.common import NCHAR
from texmo.configuration import Template
from texmo.predict import Predictor, prediction_score
from texmo.resultdb import ResultDB
from texmo.results import ResultSet


def main(db):
    np.set_printoptions(linewidth=100, edgeitems=6, precision=3)

    logging.info(f"Loading results from {db}")
    record_db = ResultDB(args.db)
    result_set = ResultSet(record_db, template=Template(), populate_neighbors=False)

    logging.info("Splitting result set into train & test")
    train_set, test_set = result_set.train_test_split()

    logging.info("Creating Predictor")
    predictor = Predictor(train_set)

    logging.info("Training")
    predictor.train()

    logging.info("Preparing test data")
    losses = []
    confs = []

    for conf_results in test_set._confs.values():
        for run in conf_results.runs:
            losses.append(run.loss)
            confs.append(conf_results.conf)

    logging.info("Predicting")
    predicted_losses = predictor.predict(confs)

    score = prediction_score(losses, predicted_losses)
    logging.info(f"Final score: {score:.5f}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="path to the SQLite database with the results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()
    main(**vars(args))
    latency.report()
