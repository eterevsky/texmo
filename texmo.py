import argparse
import json
import matplotlib.pyplot as plt
import os
import time

from dataset import DataSet
from manager import Manager
import models


def create_manager(model_name, model_path, learning_rate, regularization, steps):
    if model_name:
        model = models.build_from_name(model_name)
        manager = Manager(model, learning_rate, regularization, steps)
    else:
        with open(model_path) as f:
            spec = json.load(f)
        manager = Manager.from_spec(spec)
        if learning_rate is not None:
            manager.learning_rate = learning_rate
        if regularization is not None:
            manager.regularization = regularization

    manager.init()
    return manager


def train(manager: Manager, steps: int, temp_dir: str, dataset, sample_length, batch_size):
    start = time.time()

    step_array = []
    losses = []
    recent_losses = []

    for i in range(steps):
        batch = dataset.sample(length=sample_length, batch_size=batch_size)
        loss = manager.train(batch)
        recent_losses.append(loss)
        if manager.step < 10 or manager.step % 10 == 0 and recent_losses:
            avg_loss = sum(recent_losses) / len(recent_losses)

            if manager.step >= 20:
                step_array.append(manager.step)
                losses.append(avg_loss)

            recent_losses = []
            print(manager.step, avg_loss)

        if manager.step % 100 == 0 and temp_dir is not None:
            manager.save(temp_dir)

    if steps > 0:
        print(f'Training time: {time.time() - start}')
    
    return step_array, losses


def main(data, steps, learning_rate, regularization, output_dir, model_path, model_name, temp_dir, sample_length, batch_size):
    if data is not None:
        print(f'Training data: {data}')
        dataset = DataSet(data)
    else:
        dataset = None

    manager = create_manager(model_name, model_path, learning_rate, regularization, steps)

    if steps > 0:
        step_array, losses = train(manager, steps, temp_dir, dataset, sample_length, batch_size)

    manager.sample_loss(
        b'Roses are red\nViolets are blue,\nSugar is sweet\nAnd so are you.')

    batch = dataset.sample(1024, 1024)
    print('Batch loss:', manager.evaluate(batch))

    prefix = b'Roses are red\nViolets are blu'
    out = manager.sample(prefix, 256)
    print(prefix + out)

    try:
        s = (prefix + out).decode('utf-8')
    except UnicodeDecodeError:
        s = '<Invalid UTF-8>'
    print()
    print(s)
    print()

    if output_dir is not None:
        manager.save(output_dir)

    if steps > 0:
        plt.xscale('log')
        plt.yscale('log')
        plt.plot(step_array, losses)
        plt.savefig(os.path.join(output_dir, manager.name() + '.png'))
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-m', '--model-path', default=None,
                        help='load trained model')
    group.add_argument('-n', '--model-name', default=None,
                        help='full name of the model to be created')

    parser.add_argument('-d', '--data', type=str, default='data',
                        help='directory with training data')
                        
    parser.add_argument('-o', '--output-dir', type=str,
                        default=None, help='directory for saved model')
    parser.add_argument('-t', '--temp-dir', default=None,
                        help='directory for intermediate models')

    parser.add_argument('-s', '--steps', type=int, default=0,
                        help='number of training steps')
    parser.add_argument('-l', '--learning-rate', type=float,
                        help='learning rate', default=0.1)
    parser.add_argument('-r', '--regularization', type=float, help='L2 regularization coefficient',
                        default=0.1)
    parser.add_argument('--sample-length', type=int, default=128,
                        help='length of text fragments used for training')
    parser.add_argument('-b', '--batch-size', type=int,
                        default=32, help='batch size')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
