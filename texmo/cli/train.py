import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np

from ..configuration import Configuration
from ..dataset import DataSet, DataSetWrapper
from ..manager import Manager, create_manager
from ..model_store import save_model
from ..precision import Precision
from ..spec_parser import parse_model2
from ..tokens import set_tokens_dir


def show_loss_graph(manager: Manager):
    run = manager.run

    # step_loss is already in b/B (converted in Manager.train_step)
    scaled_step_loss = np.array(run.step_loss)

    steps = run.steps
    xs = np.array(range(steps // 2, steps * 4 + 1))
    ys = run.loss_trend.predict(xs)

    steps2 = steps * 2
    loss2 = run.loss_trend.predict([steps2])[0]
    logging.info(f"Expected loss at {steps2} steps: {loss2:.4f}")

    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(top=10)
    plt.yticks([1, 1.5, 2, 3, 4, 5, 6, 8], ['1', '1.5', '2', '3', '4', '5', '6', '8'])
    plt.plot(
        range(1, steps + 1),
        scaled_step_loss,
    )

    plt.plot(xs, ys)

    for checkpoint in manager.run.checkpoints.values():
        plt.plot(checkpoint.step + 1, checkpoint.loss, 'ro')
    plt.plot(steps + 1, run.loss, 'ro')

    plt.show()


def parse_lr(x: str) -> float:
    if x.startswith("2^"):
        return 2 ** int(x[2:])
    if "/" in x:
        # Fraction form: '1/128' -> 0.0078125. Matches the way LRs
        # appear in conf reports (`1/128`, `1/64`, ...).
        num, den = x.split("/", 1)
        return float(num) / float(den)
    return float(x)


def train(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)

    # pread sampling (no mmap readahead amplification) parallelized
    # across worker threads, so input throughput keeps up with the GPU
    # even for tiny models.
    train_set = DataSet(path=args.data, read_mode="pread")
    train_set_wrapper = DataSetWrapper(
        train_set, num_workers=args.sample_threads)
    lr = parse_lr(args.lr)
    decay = parse_lr(args.decay)

    try:
        if args.model_path is not None:
            raise NotImplementedError("Loading pre-trained models is not supported yet")
        else:
            if args.cosine and decay != 1.0:
                raise SystemExit(
                    "--cosine requires --decay 1 (cosine schedule already "
                    "decays LR to 0 over `steps`)")
            conf = Configuration(
                parse_model2(
                    args.spec, precision=Precision(args.precision)),
                lr=lr,
                length=args.length,
                batch=args.batch,
                steps=args.steps,
                decay=decay,
                cosine=args.cosine,
            )
            manager = create_manager(
                args.backend,
                conf=conf,
                system=args.system,
                dataset=train_set_wrapper,
                test_sample_len=args.test_sample_len,
                test_batch=args.test_batch,
            )
            # JAX backend defaults to the chunked-scan trainer; --no-scan
            # forces the per-step path for benchmarking.
            if args.no_scan and hasattr(manager, 'scan_train'):
                manager.scan_train = False

        run, final_conf = manager.train_and_eval(
            args.steps,
            args.time,
        )

        if args.output is not None:
            if args.backend != 'jax':
                raise SystemExit('--output requires the jax backend')
            save_model(args.output, conf.model, manager.weights)
            logging.info(f'Saved model to {args.output}')

        # Skip text sampling if training diverged or produced a nonsensical
        # loss — the model is likely broken and the sampler may crash on
        # NaN probabilities. An empty prefix also skips (there is no last
        # token to continue from).
        if args.prefix and 0 < run.loss < 10:
            temperatures = [
                float(t) for t in args.temperature.split(',') if t.strip()
            ]
            for t in temperatures:
                s = manager.continue_prefix(args.prefix, 256, temperature=t)
                print(f'--- T = {t} ---')
                try:
                    print(s.decode('utf-8'))
                except UnicodeDecodeError:
                    print(s)
            print()
    finally:
        train_set_wrapper.join()

    if not args.no_graph:
        show_loss_graph(manager)


def init_args(parser: argparse.ArgumentParser, config):
    # Data
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default=config.DATA,
        help=f"a file with (default: '{config.DATA}')",
    )
    parser.add_argument(
        "--sample-threads",
        type=int,
        default=getattr(config, "SAMPLE_THREADS", 8),
        help="background worker threads for data sampling "
             "(default: config.SAMPLE_THREADS, else 8)",
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
        "-s",
        "--spec",
        default=None,
        help="layer-by-layer model specification",
    )
    parser.add_argument(
        "-a",
        "--add-layers",
        type=str,
        metavar="SPEC",
        default=None,
        help="add and train layers to a pre-trained model loaded with -m",
    )
    parser.add_argument(
        "--steps", type=int, default=None, help="number of training steps"
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default=None,
        help="export the trained model (spec + inline weights) as a "
             "model_store JSON, loadable by eval / benchmark-model",
    )

    # Configuration
    parser.add_argument(
        "--tokens-dir",
        type=str,
        default=config.TOKENS_DIR,
        help=f"directory with token sets (default: '{config.TOKENS_DIR}')",
    )
    parser.add_argument(
        "--tokens",
        type=str,
        default=None,
        help="path to the token set definition",
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        choices=["fp64", "fp32", "fp16", "bf16"],
        metavar="P",
        default="fp32",
        help="training precision",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        metavar="N",
        default=32,
        help="batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=str,
        default="0.0078125",
        help="learning rate, could be written as a float or as 2^-10 (default: 1/128)",
    )
    parser.add_argument(
        '--decay',
        type=str,
        default="1.0",
        help="decay of the learning rate over the course of training, i.e. (LR at the last step) / (LR at the first step)  (default: 1)",
    )
    parser.add_argument(
        '--cosine',
        default=False,
        action="store_true",
        help="use a cosine LR schedule that decays to 0 over `steps` "
             "(no warmup); requires --decay 1",
    )
    parser.add_argument(
        "-l",
        "--length",
        type=int,
        default=128,
        metavar="NTOKENS",
        help="length in tokens of text fragments used for training (default: 128)",
    )
    parser.add_argument(
        "-t",
        "--time",
        type=int,
        metavar="SECONDS",
        help="time limit for training",
        default=None,
    )

    # Evaluation
    parser.add_argument(
        "--prefix",
        type=str,
        default="Roses are red\nViolets are",
        help="text prefix to be continued",
    )
    parser.add_argument(
        "--test-batch",
        type=int,
        default=1024,
        help="Size of the test batch, used for final evaluation (default: 1024)",
    )
    parser.add_argument(
        "--test-sample-len",
        type=int,
        default=1024,
        help="Length in bytes of test samples, used for final evaluation",
    )
    parser.add_argument(
        "--no-graph",
        default=False,
        action="store_true",
        help="Don't show trianing loss graph",
    )
    parser.add_argument(
        "--no-scan",
        default=False,
        action="store_true",
        help="JAX backend only: disable the chunked lax.scan trainer "
             "and use the per-step loop instead. For A/B benchmarking "
             "and time-limit support.",
    )
    parser.add_argument(
        "--temperature",
        type=str,
        default='0.03,0.1,0.3,1.0',
        help="comma-separated list of sampling temperatures; one "
             "continuation is generated for each "
             "(default: '0.03,0.1,0.3,1.0')",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=config.SYSTEM_NAME,
        help=f"the name of the system that will be used to identify runs in the DB (default: '{config.SYSTEM_NAME}')",
    )

    parser.add_argument(
        "--backend",
        type=str,
        choices=["jax"],
        default=config.BACKEND,
        help=f"training backend (default: '{config.BACKEND}')",
    )

    parser.set_defaults(func=train)
