import csv
from datetime import datetime
import time

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
):
    start = time.time()
    finish_time = start + time_limit if time_limit else float("inf")
    if steps is None:
        steps = float("inf")

    last_report = 0

    while time.time() < finish_time and manager.step < steps:
        batch = train_set.sample(length=sample_length, batch_size=batch_size)
        manager.train(batch)
        if (
            manager.step < 10
            or (manager.step % 10 == 0 and time.time() - last_report > 3)
            or time.time() - last_report > 10
        ):
            last_report = time.time()
            recent_losses = manager.step_loss[-10:]
            avg_loss = sum(recent_losses) / len(recent_losses)
            print(f"{manager.step} {avg_loss:.4f} {manager.step_loss[-1]:.4f}")

        if (temp_steps is not None
            and temp_steps > 0
            and manager.step % temp_steps == 0
            and temp_dir is not None
        ):
            manager.save(temp_dir)

    return time.time() - start


def train_and_validate(
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
) -> TrainingRecord:
    if time_limit is not None or steps is not None:
        train_time = train(
            manager,
            steps,
            time_limit,
            train_set,
            sample_len,
            batch_size,
            temp_steps,
            temp_dir,
        )
        if output_dir is not None:
            manager.save(output_dir)
    else:
        train_time = 0

    batch = train_set.sample(1024, 1024)
    batch_loss = manager.evaluate(batch)

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
