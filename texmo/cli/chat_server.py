"""Serve a stored model over an OpenAI-compatible endpoint.

    uv run python texmo.py chat-server -m models/fold.32-8k-1.json \\
        [--port 5011] [-T 0.5]

The non-interactive sibling of `texmo.py chat`: the same dialog path
(`texmo.chat`), driven by `POST /v1/chat/completions` instead of a
REPL, so `scripts/dialog_harness.py` can seat this model as a
participant with an `external` (`base_url`) config entry and no
harness changes. See `texmo.chat_server` for the request mapping.

The model is loaded before the listener binds, so `/health` answering
200 means the endpoint can serve.
"""
import argparse
import logging
import time

from ..chat_server import ChatServer
from ..model_store import load_model
from ..tokens import set_tokens_dir


def main(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)

    t0 = time.perf_counter()
    md, weights = load_model(args.model)
    model = md.build_jax()
    tokens_name = md.codec.tokens_name
    logging.info(
        f'Loaded {args.model} ({md.num_weights:,} weights, {md.precision}), '
        f'tokens: {tokens_name}, in {time.perf_counter() - t0:.1f}s')

    server = ChatServer(
        model, weights, tokens_name,
        model_name=args.model,
        temperature=args.temperature,
        max_reply_tokens=args.max_reply_tokens,
        user_name=args.user_name,
        bot_name=args.bot_name,
    )
    server.serve(args.host, args.port)


def init_args(parser: argparse.ArgumentParser, config):
    parser.add_argument(
        '-m', '--model', required=True,
        help='model JSON manifest')
    parser.add_argument(
        '--host', default='127.0.0.1',
        help="interface to bind (default: '127.0.0.1'; use '0.0.0.0' to "
             'accept connections from the LAN)')
    parser.add_argument(
        '--port', type=int, default=5011,
        help='port to bind (default: 5011)')
    parser.add_argument(
        '-T', '--temperature', type=float, default=0.7,
        help='default sampling temperature, overridden by a request'
             " that sets `temperature` (default: 0.7)")
    parser.add_argument(
        '--max-reply-tokens', type=int, default=512,
        help='stop a reply after this many tokens if it never reaches '
             'a turn boundary, overridden by a request that sets '
             '`max_tokens` (default: 512)')
    parser.add_argument(
        '--user-name', default='User',
        help="the requester's name in the dialog (default: 'User')")
    parser.add_argument(
        '--bot-name', default='Bot',
        help="the model's name in the dialog (default: 'Bot')")
    parser.add_argument(
        '--tokens-dir', default=config.TOKENS_DIR,
        help=f"directory with token sets (default: '{config.TOKENS_DIR}')")
    parser.set_defaults(func=main)
