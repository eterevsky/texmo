"""Sample continuations of a prefix from a stored model.

The inference-only counterpart of train's post-training sampling
block: load a model JSON manifest, continue a prefix at one or more
temperatures, print each continuation under a `--- T = t ---` rule.

    uv run python texmo.py generate -m models/bits.16-8k-1.json \\
        --prefix 'Roses are red' -n 128 -T 0.3,1.0

A multi-line prefix cannot be passed as a command-line argument, so
`--prefix-file` reads one from a UTF-8 file instead -- verbatim,
including the trailing newline (line endings are normalized to '\\n').

Runs at the manifest's precision.
"""
import argparse
import contextlib
import io
import sys
import time

from ..generate import continue_prefix
from ..model_store import load_model
from ..tokens import set_tokens_dir


def add_prefix_args(
    parser: argparse.ArgumentParser,
    required: bool,
    default: str | None = None,
) -> None:
    """Add the mutually exclusive `--prefix` / `--prefix-file` pair."""
    group = parser.add_mutually_exclusive_group(required=required)
    group.add_argument(
        '--prefix',
        default=default,
        help='text prefix to be continued',
    )
    group.add_argument(
        '--prefix-file',
        metavar='PATH',
        default=None,
        help='read the prefix from a UTF-8 text file, keeping its '
             'newlines (which cannot be passed with --prefix)',
    )


def read_prefix(args: argparse.Namespace) -> str | None:
    """The prefix to continue: the file's text, the string, or None."""
    if args.prefix_file is not None:
        with open(args.prefix_file, encoding='utf-8') as f:
            return f.read()
    return args.prefix


def parse_temperatures(spec: str) -> list[float]:
    return [float(t) for t in spec.split(',') if t.strip()]


def print_continuations(sample, temperatures, write) -> None:
    """One continuation per temperature, each under its own rule.

    `sample(temperature) -> bytes`. A continuation can end mid
    character (byte-level tokenization), so undecodable output falls
    back to the raw bytes repr rather than failing.
    """
    for t in temperatures:
        s = sample(t)
        write(f'--- T = {t} ---')
        try:
            write(s.decode('utf-8'))
        except UnicodeDecodeError:
            write(str(s))
    write('')


@contextlib.contextmanager
def utf8_stdout():
    """stdout as a UTF-8 text stream.

    Sampled text is arbitrary bytes decoded as UTF-8 and the Windows
    console encoding is not UTF-8, so stdout is wrapped rather than
    trusted -- with errors='replace', since losing a character beats
    aborting the whole continuation.
    """
    buffer = getattr(sys.stdout, 'buffer', None)
    if buffer is None:
        # Already a text stream (pytest capture); write straight to it.
        yield sys.stdout
        return
    stream = io.TextIOWrapper(
        buffer, encoding='utf-8', errors='replace', newline='')
    try:
        yield stream
        stream.flush()
    finally:
        # Detach, or closing the wrapper would close stdout itself.
        stream.detach()


def generate(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)
    prefix = read_prefix(args)
    temperatures = parse_temperatures(args.temperature)

    t0 = time.perf_counter()
    md, weights = load_model(args.model)
    model = md.build_jax()
    tokens_name = md.codec.tokens_name

    with utf8_stdout() as out:
        write = lambda line: out.write(line + '\n')
        write(f'model:  {args.model} ({md.num_weights:,} weights, '
              f'{md.precision}), tokens: {tokens_name}')
        write(f'loaded in {time.perf_counter() - t0:.1f}s')
        print_continuations(
            lambda t: continue_prefix(
                model, weights, tokens_name, prefix, args.length,
                temperature=t),
            temperatures,
            write,
        )


def init_args(parser: argparse.ArgumentParser, config):
    parser.add_argument(
        '-m', '--model', required=True,
        help='model JSON manifest')
    add_prefix_args(parser, required=True)
    parser.add_argument(
        '-n', '--length', type=int, default=256,
        help='number of tokens to generate (default: 256)')
    parser.add_argument(
        '-T', '--temperature', default='0.03,0.1,0.3,1.0',
        help='comma-separated list of sampling temperatures; one '
             'continuation is generated for each '
             "(default: '0.03,0.1,0.3,1.0')")
    parser.add_argument(
        '--tokens-dir', default=config.TOKENS_DIR,
        help=f"directory with token sets (default: '{config.TOKENS_DIR}')")
    parser.set_defaults(func=generate)
