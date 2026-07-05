import argparse
import sys
import time

from ..common import itoa3
from ..dataset import DataSet, DataSetWrapper
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


def tokenize(args: argparse.Namespace):
    """Tokenize a given text (not a corpus sample) and show the ids and
    token strings -- for eyeballing and for comparing against reference
    tokenizers."""
    set_tokens_dir(args.tokens_dir)
    tokenizer = get_tokenizer(args.tokens)
    tokenset = tokenizer.tokenset

    if args.text is not None:
        text = args.text
    elif args.file is not None:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    if hasattr(tokenizer, "encode"):  # BPE sets take str + bos flag
        ids = tokenizer.encode(text, add_bos=args.bos)
    else:
        ids = tokenizer.tokenize(text.encode("utf-8"))

    print(list(ids))
    print()
    for i in ids:
        print(f"{i}\t{ascii(str(tokenset.tokens[i]))}")


def tokenize_init_args(parser: argparse.ArgumentParser, config):
    parser.add_argument(
        "--tokens-dir",
        type=str,
        default=config.TOKENS_DIR,
        help="directory with token sets",
    )
    parser.add_argument(
        "-t",
        "--tokens",
        type=str,
        required=True,
        help="name of the token set",
    )
    parser.add_argument(
        "--text", type=str, default=None, help="text to tokenize")
    parser.add_argument(
        "--file", type=str, default=None,
        help="read the text from a file (default: stdin)")
    parser.add_argument(
        "--bos", action="store_true",
        help="prepend the beginning-of-sequence token (BPE sets)")
    parser.set_defaults(func=tokenize)


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

    def _drive(produce, seconds: float = 10.0) -> float:
        produce()  # warm up (tokenizer init, first faults, queue prime)
        samples = 0
        start = time.time()
        while time.time() - start < seconds:
            produce()
            samples += 1
            if samples % 100 == 0:
                print(".", end="", flush=True)
        return samples / (time.time() - start)

    def run_direct(mode: str) -> float:
        dataset.read_mode = mode
        return _drive(lambda: dataset.sample_tokens(ntokens, batch, token_set))

    def run_parallel(workers: int) -> float:
        dataset.read_mode = "pread"
        wrapper = DataSetWrapper(dataset, num_workers=workers)
        try:
            return _drive(
                lambda: wrapper.sample_tokens(ntokens, batch, token_set))
        finally:
            wrapper.join()

    # TEMPORARY: compare the current mmap path, single-threaded pread,
    # and N-worker pread so we can see per-system how much each helps.
    # Drop the mmap arm once pread is the default.
    results = [
        ("mmap  x1", run_direct("mmap")),
        ("pread x1", run_direct("pread")),
        (f"pread x{args.threads}", run_parallel(args.threads)),
    ]
    print()

    base = results[0][1]
    for name, sps in results:
        print(
            f"{name}: {itoa3(sps)} samples/s  "
            f"{itoa3(sps * ntokens * batch)} tokens/s  "
            f"({sps / base:.1f}x)")


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
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="worker threads for the parallel pread arm (default: 4)",
    )

    parser.set_defaults(func=benchmark)
