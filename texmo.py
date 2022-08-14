import argparse
from collections import namedtuple
import json
import matplotlib.pyplot as plt
import os
import time

from dataset import DataSet
from manager import Manager
import models


def report(manager, training_time, train_set, sample_length, batch_size, output_dir, skip_graph):
    print('Model:', manager.model.name)
    print('Params:', manager.model.total_weights(manager.weights))
    print('Steps:', manager.step)
    data_size = manager.step * batch_size * sample_length / 1E6
    print(f'Used {data_size:.1f}M data for training')
    print(f'Training time: {training_time:.0f} s')

    manager.sample_loss(
        b'Roses are red\nViolets are blue,\nSugar is sweet\nAnd so are you.')

    batch = train_set.sample(1024, 1024)
    batch_loss = manager.evaluate(batch)
    print(f'Batch loss: {batch_loss:.4f}')

    prefix = b'Roses are red\nViolets are blu'
    out = manager.sample(prefix, 256)

    try:
        s = (prefix + out).decode('utf-8')
    except UnicodeDecodeError:
        s = repr(prefix + out)
    print()
    print(s)
    print()

    if manager.step > 0 and not skip_graph:
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(top=8)
        plt.plot(range(1, len(manager.step_loss) + 1), manager.step_loss)
        plt.savefig(os.path.join(output_dir, manager.name() + '.png'))
        plt.show()

    return batch_loss, data_size


def main(data, steps, learning_rate, regularization, output_dir, model_name, model_path, temp_dir, sample_length, batch_size, temp_steps, time_limit, skip_graph=False):
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
        assert learning_rate is None
        assert regularization is None

    manager.init()

    start = time.time()

    last_report = 0

    for i in range(steps):
        if time_limit is not None and time.time() - start > time_limit:
            break

        batch = train_set.sample(length=sample_length, batch_size=batch_size)
        manager.train(batch)
        if manager.step < 10 or manager.step % 10 == 0 or time.time() - last_report > 10:
            last_report = time.time()
            recent_losses = manager.step_loss[-10:]
            avg_loss = sum(recent_losses) / len(recent_losses)
            print(f'{manager.step} {avg_loss:.4f} {manager.step_loss[-1]:.4f}')

        if temp_steps > 0 and manager.step % temp_steps == 0 and temp_dir is not None:
            manager.save(temp_dir)

    training_time = int(time.time() - start)
    if output_dir is not None:
        manager.save(output_dir)

    loss, data_size = report(manager, training_time, train_set,
                             sample_length, batch_size, output_dir, skip_graph)
    return loss, manager.step, data_size


def benchmark(data, time_limit):
    results = []
    for name in ('recurrent1-512', 'recurrent3-2-128-512-128', 'llstm-512-512', 'grugru-512-512', 'convgru1-128-512'):
        for learning_rate in (0.1, 0.01):
            for sample_length in (128, 256):
                for regularization in (0.1, 0.01):
                    for batch_size in (64, 128, 256):
                        loss, step, data_size = main(data, 1000000, learning_rate, regularization, None, name, None, None,
                                                     sample_length, batch_size, temp_steps=0, time_limit=time_limit, skip_graph=True)
                        results.append(
                            (name, learning_rate, regularization, batch_size, step, data_size, loss))
    for r in results:
        name, lr, reg, batch_size, step, data_size, loss = r
        print(f'{name:30} L{lr:.3f} R{reg:.2f} {batch_size:3} {step:5} {data_size:4.1f}M {loss:.4f}')


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

    model_group.add_argument('--benchmark', action='store_true',
                             default=False, help='run a benchmark with various configurations')

    parser.add_argument('-o', '--output-dir', type=str, metavar='PATH',
                        default=None, help='directory for saved model')

    # Training
    parser.add_argument('-s', '--steps', type=int, default=0,
                        help='number of training steps')
    parser.add_argument('--time-limit', type=int, metavar='SECONDS',
                        help='time limit for training', default=None)
    parser.add_argument('-l', '--learning-rate', type=float, metavar='RATE',
                        help='learning rate', default=0.01)
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

    parser.add_argument('--skip-graph', action='store_true',
                        default=False, help='do not generate the loss graph')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.benchmark:
        benchmark(args.data, args.time_limit)
    else:
        main(**vars(args))
