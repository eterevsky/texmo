import argparse
import math
import numpy as np

from texmo.common import INF
from texmo.predict import prediction_score
from texmo.resultdb import ResultDB
from texmo.steploss import StepLossPredictor


def main(db):
    np.set_printoptions(linewidth=100, edgeitems=6, precision=3)
    print(f"Initializing the DB {args.db}")
    record_db = ResultDB(args.db)

    true_losses = []
    predicted_losses = []

    for _, step_loss in record_db.get_runs_with_step_loss():
        true_loss = step_loss[-1]
        if math.isnan(true_loss):
            true_loss = INF
        true_losses.append(true_loss)
        predictor = StepLossPredictor()
        predictor.fit(step_loss[:len(step_loss) // 2])
        predicted_losses.append(predictor.predict(len(step_loss) - 1))

    losses = np.stack((true_losses, predicted_losses))
    print(losses)

    print(prediction_score(true_losses, predicted_losses))


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
    args = parse_args()
    main(args.db)
