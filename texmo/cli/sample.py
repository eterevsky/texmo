import argparse
import time

from ..common import itoa3
from ..dataset import DataSet
from ..tokens import get_tokenizer, set_tokens_dir


def sample(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)

    dataset = DataSet(path=args.data, path_processed=args.data_processed)
    tokenizer = get_tokenizer(args.tokens)
    tokenset = tokenizer.tokenset

    if args.length:
        samples, lengths = dataset.sample_bytes(args.length, args.batch, args.tokens)
        print(f"Array:\n{samples}\n")
        print(f"Lengths: {lengths}\n")

        for sample, length in zip(samples, lengths):
            sample = sample[:length]

            print("Sample:\n")
            untokenized = tokenizer.untokenize(sample)

            try:
                untokenized = untokenized.decode("utf-8")
            except UnicodeError:
                pass
            print(untokenized)
            print()

            for token in sample:
                print(tokenset.tokens[token], end="|")
            print()

    else:
        samples = dataset.sample_tokens(args.ntokens, args.batch, args.tokens)
        print(f"Array:\n{samples}\n")

        for sample in samples:
            print("Sample:\n")
            untokenized = tokenizer.untokenize(sample)

            try:
                untokenized = untokenized.decode("utf-8")
            except UnicodeError:
                pass
            print(untokenized)
            print()

            for token in sample:
                print(tokenset.tokens[token], end="|")
            print()
        print()


def sample_init_args(parser: argparse.ArgumentParser, config):
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="training data file",
        default=config.DATA,
    )
    parser.add_argument(
        "--data-processed",
        type=str,
        help="training data that has been already processed with capswords filter",
        default=config.DATA_CAPS_WORDS,
    )
    parser.add_argument(
        "--tokens-dir",
        type=str,
        default=config.TOKENS_DIR,
        help="directory with token sets",
    )
    parser.add_argument("-b", "--batch", type=int, help="batch size", default=1)
    parser.add_argument(
        "-t",
        "--tokens",
        type=str,
        help="name of the token set",
        default=None,
    )
    length_group = parser.add_mutually_exclusive_group(required=True)
    length_group.add_argument(
        "-n",
        "--ntokens",
        type=int,
        help="length of the sample in tokens",
        default=None,
    )
    length_group.add_argument(
        "-l", "--length", type=int, help="length in bytes", default=None
    )
    parser.set_defaults(func=sample)


def benchmark(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)
    dataset = DataSet(path=args.data, path_processed=args.data_processed)

    ntokens = args.ntokens
    batch = args.batch
    token_set = args.token_set

    def run(mode: str, seconds: float = 10.0) -> float:
        dataset.read_mode = mode
        # One warm-up batch (tokenizer init, first faults) outside timing.
        dataset.sample_tokens(ntokens, batch, token_set)
        samples = 0
        start = time.time()
        while time.time() - start < seconds:
            dataset.sample_tokens(ntokens, batch, token_set)
            samples += 1
            if samples % 100 == 0:
                print(".", end="", flush=True)
        return samples / (time.time() - start)

    # TEMPORARY: A/B the current mmap path against the new pread path
    # so we can compare the effect across systems. Drop the mmap run
    # once pread is the default.
    mmap_sps = run("mmap")
    pread_sps = run("pread")
    print()

    for name, sps in (("mmap ", mmap_sps), ("pread", pread_sps)):
        print(f"{name}: {itoa3(sps)} samples/s  {itoa3(sps * ntokens * batch)} tokens/s")
    print(f"pread speedup: {pread_sps / mmap_sps:.2f}x")


def benchmark_init_args(parser: argparse.ArgumentParser, config):
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="a file with training data",
        default=config.DATA,
    )
    parser.add_argument(
        "--data-processed",
        type=str,
        help="training data that has been already processed with capswords filter",
        default=config.DATA_CAPS_WORDS,
    )
    parser.add_argument(
        "--tokens-dir",
        type=str,
        default="tokens",
        help="directory with token sets",
    )
    parser.add_argument(
        "--token-set",
        default="tokens256_capswords_byteshuff",
        type=str,
        help="token set name",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=256,
        help='range of acceptable batch sizes, for example "1-256" (default: unrestricted)',
    )
    parser.add_argument(
        "-n",
        "--ntokens",
        type=int,
        default=256,
        help="range of acceptable sample lens (default: 128)",
    )

    parser.set_defaults(func=benchmark)
