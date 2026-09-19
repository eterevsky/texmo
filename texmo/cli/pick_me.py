"""`texmo.py pick-me` — queue priority runs for one configuration.

Posts a conf to the running search server's `/pick_me` endpoint. The
server flags it with a target run count (`conf.pick_me`), and
`Search.select_conf` hands that conf to the next workers, ahead of
every other strategy, until it has that many runs. The usual reason is
the winner's curse: a conf that reached the top of the frontier on two
runs deserves a third and a fourth before we believe it.
"""

import argparse
import logging

import requests

from ..configuration import Configuration
from ..tokens import set_tokens_dir
from .train import add_conf_args, conf_from_args

# (connect, read) timeout. The server answers once its writer thread
# has committed, which is milliseconds unless the queue is backed up.
_HTTP_TIMEOUT = (10, 120)


def server_url(server_host: str, path: str) -> str:
    """Resolve the client-style `--server` value to an endpoint URL.

    Bare `host:port` is http; an explicit scheme is preserved (same
    rule as `client.worker_loop`).
    """
    if '://' in server_host:
        base = server_host.rstrip('/')
    else:
        base = f"http://{server_host}"
    return f"{base}{path}"


def main(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)
    if args.spec is None:
        raise SystemExit("pick-me requires -s/--spec")
    if args.runs < 1:
        raise SystemExit(f"--runs must be >= 1, got {args.runs}")
    conf: Configuration = conf_from_args(args)
    if not conf.model.is_valid():
        raise SystemExit(
            f"conf is not valid, the search would skip it: {conf}")

    url = server_url(args.server, '/pick_me')
    headers = {}
    if args.api_key:
        headers['Authorization'] = f'Bearer {args.api_key}'
    resp = requests.post(
        url,
        json={'conf': conf.to_dict(), 'runs': args.runs},
        headers=headers,
        timeout=_HTTP_TIMEOUT,
    )
    if resp.status_code != 200:
        raise SystemExit(
            f"{url} returned {resp.status_code}: {resp.text.strip()}")
    body = resp.json()
    logging.info(f"Pick-me accepted: {conf}")
    print(f"conf_id={body['conf_id']} "
          f"({'inserted' if body['inserted'] else 'existing'}) "
          f"runs={body['runs']} target={body['target']}")
    if body['runs'] >= body['target']:
        print("Already at the target — the search won't prioritize it. "
              "Re-run with a higher --runs.")


def init_args(parser: argparse.ArgumentParser, config):
    # The same conf flags as `train`, so a conf can be copied from a
    # train command line verbatim.
    add_conf_args(parser)
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        metavar="N",
        help="keep the conf a priority pick until it has N total runs "
             "(never lowers an existing target; default: 3)",
    )
    parser.add_argument(
        "--tokens-dir",
        type=str,
        default=config.TOKENS_DIR,
        help=f"directory with token sets (default: '{config.TOKENS_DIR}')",
    )
    parser.add_argument(
        "--server",
        default=config.SERVER_HOST,
        help="Server URL or host:port. Bare host:port is treated as "
             "http; pass an explicit https://... for an authenticated "
             "remote endpoint.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=getattr(config, 'API_KEY', '') or '',
        help="Bearer token sent in the Authorization header. Defaults "
             "to config.API_KEY. Empty for LAN clients on port 5000.",
    )
    parser.set_defaults(func=main)
