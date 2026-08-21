"""Interactive chat with a stored model.

    uv run python texmo.py chat -m models/fold.32-8k-1.json -T 0.5

Reads a line, appends it to the running dialog as a user turn, and
samples the bot's reply up to the next turn boundary (see
`texmo.chat`). The whole dialog is kept as text and re-tokenized each
turn, so the model sees exactly what a corpus sample would look like.

Every session appends to a JSONL transcript under `--log-dir`: one
header record, then one record per exchange written (and flushed) as
soon as the reply lands, so a crash or a Ctrl-C keeps everything up
to that point. JSONL, so one record per line -- not the repo's
pretty-printed `pjson` style.

Ends on EOF (Ctrl-D / Ctrl-Z), Ctrl-C, or two empty lines.
"""
import argparse
import datetime
import json
import os
import time

from ..chat import append_reply, build_prompt, generate_reply
from ..model_store import load_model
from ..tokens import set_tokens_dir
from .generate import utf8_stdout

TURN_SEPARATOR = '\n\n'


def normalize_dialog(text: str) -> str:
    """A dialog history: empty, or ending in exactly one separator."""
    text = text.rstrip('\n')
    return text + TURN_SEPARATOR if text else ''


def read_preamble(path: str | None) -> str:
    """Dialog history to start from -- a file's text, or nothing."""
    if path is None:
        return ''
    with open(path, encoding='utf-8') as f:
        return normalize_dialog(f.read())


def transcript_path(log_dir: str, started: datetime.datetime) -> str:
    return os.path.join(
        log_dir, started.strftime('chat-%Y%m%d-%H%M%S.jsonl'))


def session_record(
    args: argparse.Namespace,
    tokens_name: str,
    started: datetime.datetime,
) -> dict:
    return {
        'model': args.model,
        'tokens': tokens_name,
        'temperature': args.temperature,
        'args': _scalar_args(args),
        'started': started.isoformat(timespec='seconds'),
    }


def exchange_record(
    utterance: str,
    reply: str,
    tokens_generated: int,
    elapsed_s: float,
) -> dict:
    return {
        'user': utterance,
        'reply': reply,
        'tokens_generated': tokens_generated,
        'elapsed_s': round(elapsed_s, 3),
    }


def write_record(log, record: dict) -> None:
    """Append one JSONL record and flush, so a crash keeps it."""
    log.write(json.dumps(record, ensure_ascii=False) + '\n')
    log.flush()


def _scalar_args(args: argparse.Namespace) -> dict:
    # Drops `func` (the subcommand entry point) and anything else
    # argparse carries that json cannot serialize.
    return {
        k: v for k, v in sorted(vars(args).items())
        if isinstance(v, (str, int, float, bool)) or v is None
    }


def chat(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)
    started = datetime.datetime.now()
    dialog = read_preamble(args.preamble_file)

    t0 = time.perf_counter()
    md, weights = load_model(args.model)
    model = md.build_jax()
    tokens_name = md.codec.tokens_name

    os.makedirs(args.log_dir, exist_ok=True)
    path = transcript_path(args.log_dir, started)
    with open(path, 'a', encoding='utf-8') as log, utf8_stdout() as out:
        write_record(log, session_record(args, tokens_name, started))
        out.write(f'model:  {args.model} ({md.num_weights:,} weights, '
                  f'{md.precision}), tokens: {tokens_name}\n')
        out.write(f'loaded in {time.perf_counter() - t0:.1f}s; '
                  f'T = {args.temperature}\n')
        out.write(f'log:    {path}\n')
        out.write('(empty line twice or Ctrl-C to quit)\n')
        _repl(args, model, weights, tokens_name, dialog, out, log)


def _repl(args, model, weights, tokens_name, dialog, out, log) -> None:
    def stream(text):
        # Flushed per token: the point is watching the reply arrive.
        out.write(text)
        out.flush()

    blanks = 0
    while True:
        out.write(f'\n{args.user_name}: ')
        out.flush()
        try:
            utterance = input().strip()
        except (EOFError, KeyboardInterrupt):
            out.write('\n')
            break

        if not utterance:
            blanks += 1
            if blanks >= 2:
                break
            continue
        blanks = 0

        prompt = build_prompt(
            dialog, utterance, args.user_name, args.bot_name)
        # Blank line between the turns, as the dialog format has it.
        out.write(f'\n{args.bot_name}:')
        out.flush()

        t0 = time.perf_counter()
        try:
            reply, ntokens = generate_reply(
                model, weights, tokens_name, prompt,
                args.max_reply_tokens, args.temperature, on_delta=stream)
        except KeyboardInterrupt:
            out.write('\n[interrupted]\n')
            break
        elapsed = time.perf_counter() - t0
        out.write('\n')
        out.flush()

        dialog = append_reply(prompt, reply)
        write_record(
            log, exchange_record(utterance, reply.strip(), ntokens, elapsed))


def init_args(parser: argparse.ArgumentParser, config):
    parser.add_argument(
        '-m', '--model', required=True,
        help='model JSON manifest')
    parser.add_argument(
        '-T', '--temperature', type=float, default=0.7,
        help='sampling temperature (default: 0.7)')
    parser.add_argument(
        '--max-reply-tokens', type=int, default=512,
        help='stop a reply after this many tokens if it never reaches '
             'a turn boundary (default: 512)')
    parser.add_argument(
        '--user-name', default='User',
        help="the user's name in the dialog (default: 'User')")
    parser.add_argument(
        '--bot-name', default='Bot',
        help="the model's name in the dialog (default: 'Bot')")
    parser.add_argument(
        '--preamble-file', metavar='PATH', default=None,
        help='UTF-8 text file with dialog turns to start the '
             'conversation from')
    parser.add_argument(
        '--log-dir', default='data/chat_logs',
        help="directory for JSONL transcripts "
             "(default: 'data/chat_logs')")
    parser.add_argument(
        '--tokens-dir', default=config.TOKENS_DIR,
        help=f"directory with token sets (default: '{config.TOKENS_DIR}')")
    parser.set_defaults(func=chat)
