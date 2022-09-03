import argparse
import csv
from datetime import datetime
import json
import matplotlib.pyplot as plt
import os
import time

from dataset import DataSet
import eval
import layered
from manager import Manager
from record import TrainingRecord


def train(
    manager,
    steps,
    time_limit,
    train_set,
    sample_length,
    batch_size,
    temp_steps,
    temp_dir,
    quiet=False,
):
    start = time.time()
    finish_time = start + time_limit if time_limit else float("inf")
    if steps is None:
        steps = float("inf")

    last_report = 0

    while time.time() < finish_time and manager.step < steps:
        batch = train_set.sample(length=sample_length, batch_size=batch_size)
        manager.train(batch)
        if not quiet and (
            manager.step < 10
            or (manager.step % 10 == 0 and time.time() - last_report > 3)
            or time.time() - last_report > 10
        ):
            last_report = time.time()
            recent_losses = manager.step_loss[-10:]
            avg_loss = sum(recent_losses) / len(recent_losses)
            print(f"{manager.step} {avg_loss:.4f} {manager.step_loss[-1]:.4f}")

        if (
            temp_steps is not None
            and temp_steps > 0
            and manager.step % temp_steps == 0
            and temp_dir is not None
        ):
            manager.save(temp_dir)

    return time.time() - start


def train_and_eval(
    manager,
    steps,
    time_limit,
    train_set,
    sample_len,
    batch_size,
    temp_steps,
    temp_dir,
    output_dir,
    log,
    quiet=False,
) -> TrainingRecord:
    train_time = train(
        manager,
        steps,
        time_limit,
        train_set,
        sample_len,
        batch_size,
        temp_steps,
        temp_dir,
        quiet=quiet,
    )
    if output_dir is not None:
        manager.save(output_dir)

    batch_loss = eval.eval(train_set, manager)

    report = TrainingRecord(
        timestamp=datetime.now(),
        model_spec=manager.model.full_name,
        weights=manager.model.total_weights(manager.weights),
        steps=manager.step,
        train_time_s=train_time,
        learning_rate=manager.learning_rate,
        regularization=manager.regularization,
        train_sample_len=sample_len,
        train_batch=batch_size,
        total_data=train_set.total_size,
        loss=batch_loss,
        test_sample_len=1024,
        test_batch=1024,
        test_poisoned=True,
    )

    print(report)
    if log is not None:
        with open(log, "a", newline="") as logfile:
            writer = csv.writer(logfile)
            writer.writerow(report.csv_tuple())

    return report


def create_manager(
    model_spec: str,
    model_path: str,
    learning_rate: float,
    regularization: float,
):
    if model_spec is not None:
        model = layered.LayeredModel2.parse(model_spec)
        manager = Manager(model, learning_rate, regularization, 100000)
    else:
        assert model_path is not None
        with open(model_path) as f:
            model_json = json.load(f)
        manager = Manager.from_spec(model_json)

    manager.init()
    return manager


def show_loss_graph(manager, output_dir):
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(top=8)
    plt.plot(range(1, len(manager.step_loss) + 1), manager.step_loss)
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, manager.name() + ".png"))
    plt.show()


def main(
    data,
    steps,
    learning_rate,
    regularization,
    output_dir,
    model_path,
    model_spec,
    temp_dir,
    sample_len,
    batch_size,
    temp_steps,
    time_limit,
    log,
    prefix,
):
    print(f"Training data: {data}")
    train_set = DataSet(data)

    manager = create_manager(
        model_spec, model_path, learning_rate, regularization
    )

    train_and_eval(
        manager,
        steps,
        time_limit,
        train_set,
        sample_len,
        batch_size,
        temp_steps,
        temp_dir,
        output_dir,
        log,
    )

    if prefix is not None:
        eval.continue_prefix(manager, prefix, 256)

    show_loss_graph(manager, output_dir)


def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        required=True,
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
        "-c",
        "--model-spec",
        metavar="SPEC",
        default=None,
        help="layer-by-layer model specification",
    )

    # Training
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        metavar="PATH",
        default=None,
        help="directory to save the trained model",
    )
    parser.add_argument(
        "-s", "--steps", type=int, default=None, help="number of training steps"
    )
    parser.add_argument(
        "-t",
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
        "--sample-len",
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
        default=None,
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
    args = parse_args()
    main(**vars(args))
