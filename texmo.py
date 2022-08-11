import argparse
import json
import matplotlib.pyplot as plt
import os
import time

from dataset import DataSet
from manager import Manager
import models


def main(data, steps, learning_rate, regularization, output_dir, model_name, model_path, temp_dir, sample_length, batch_size, temp_steps):
    if data is not None:
        print(f'Training data: {data}')
        train_set = DataSet(data)
    else:
        train_set = None

    if model_name is not None:
        model = models.parse(model_name)
        manager = Manager(model, learning_rate, regularization, steps)
    else:
        assert model_path is not None
        with open(model_path) as f:
            spec = json.load(f)
        manager = Manager.from_spec(spec)
        if learning_rate is not None:
            manager.learning_rate = learning_rate
        if regularization is not None:
            manager.regularization = regularization

    manager.init()

    start = time.time()

    step_array = []
    losses = []
    recent_losses = []
    last_report = 0

    for i in range(steps):
        batch = train_set.sample(length=sample_length, batch_size=batch_size)
        loss = manager.train(batch)
        recent_losses.append(loss)
        if len(recent_losses) > 10: recent_losses.pop(0)
        if manager.step < 10 or manager.step % 10 == 0 or time.time() - last_report > 10:
            last_report = time.time()
            avg_loss = sum(recent_losses) / len(recent_losses)

            if manager.step >= 20:
                step_array.append(manager.step)
                losses.append(avg_loss)

            print(manager.step, avg_loss, loss)

        if temp_steps > 0 and manager.step % temp_steps == 0 and temp_dir is not None:
            manager.save(temp_dir)

    print('Model:', model.name)
    print('Params:', model.total_weights(manager.weights))
    print('Steps:', steps)
    data_size = steps * batch_size * sample_length / 1E6
    print(f'Used {steps:.1f}M data for training')

    if steps > 0:
        t = int(time.time() - start)
        print(f'Training time: {t:.0f} s')

    manager.sample_loss(
        b'Roses are red\nViolets are blue,\nSugar is sweet\nAnd so are you.')

    batch = train_set.sample(1024, 1024)
    print('Batch loss: {:.4f}'.format(manager.evaluate(batch)))

    prefix = b'Roses are red\nViolets are blu'
    out = manager.sample(prefix, 256)

    try:
        s = (prefix + out).decode('utf-8')
    except UnicodeDecodeError:
        s = repr(prefix + out)
    print()
    print(s)
    print()

    if output_dir is not None:
        manager.save(output_dir)

    if steps > 0:
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(top=8)
        plt.plot(step_array, losses)
        plt.savefig(os.path.join(output_dir, manager.name() + '.png'))
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('-d', '--data', type=str, required=True,
                        help='directory with training data')

    # Model
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('-m', '--model-path', metavar='PATH', default=None,
                             help='load trained model from file')
    model_group.add_argument(
        '-n', '--model-name', metavar='NAME', default=None, help='parse model name')

    parser.add_argument('-o', '--output-dir', type=str, metavar='PATH',
                        default=None, help='directory for saved model')

    # Training
    parser.add_argument('-s', '--steps', type=int, default=0,
                        help='number of training steps')
    parser.add_argument('-l', '--learning-rate', type=float, metavar='RATE',
                        help='learning rate', default=0.1)
    parser.add_argument('-r', '--regularization', type=float, help='L2 regularization coefficient',
                        default=0.1)
    parser.add_argument('--sample-length', type=int, default=1024, metavar='LEN',
                        help='length of text fragments used for training')
    parser.add_argument('-b', '--batch-size', type=int, metavar='BATCH',
                        default=256, help='batch size')

    # Intermediate models
    parser.add_argument('-t', '--temp-dir', default=None, metavar='PATH',
                        help='directory for intermediate models')
    parser.add_argument('--temp-steps', metavar='N', type=int,
                        default=1000, help='save intermediate model every N steps')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
