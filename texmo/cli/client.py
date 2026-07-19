import argparse
import logging

from ..client import worker_loop
from ..dataset import DataSet, DataSetWrapper
from ..latency import report
from ..tokens import set_tokens_dir


def main(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)
    # pread sampling (no mmap readahead amplification) parallelized
    # across worker threads, so input throughput keeps up with the GPU
    # even for tiny models.
    dataset = DataSet(
        path=args.data, path_processed=args.data_processed,
        read_mode="pread")
    dataset_wrapper = DataSetWrapper(
        dataset, num_workers=args.sample_threads)

    try:
        try:
            worker_loop(
                server_host=args.server, system=args.system,
                dataset=dataset_wrapper,
                backend=args.backend, once=args.once,
                api_key=args.api_key or '',
                refit_ok=not args.no_refit,
            )
        except KeyboardInterrupt:
            logging.info("Interrupted")
    finally:
        dataset_wrapper.join()

    report()


def init_args(parser: argparse.ArgumentParser, config):
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default=config.DATA,
        help="a file with training data",
    )
    parser.add_argument(
        "--data-processed",
        type=str,
        help=f"training data that has been already processed with capswords filter (default: '{config.DATA_CAPS_WORDS}')",
        default=config.DATA_CAPS_WORDS,
    )
    parser.add_argument(
        "--tokens-dir",
        type=str,
        default=config.TOKENS_DIR,
        help="directory with token sets",
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
        help="Bearer token sent in the Authorization header on every "
             "request. Defaults to config.API_KEY. Empty for LAN "
             "clients on port 5000.",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=config.SYSTEM_NAME,
        help="the name of the system that will be used to identify runs in the DB",
    )
    parser.add_argument(
        "--sample-threads",
        type=int,
        default=getattr(config, "SAMPLE_THREADS", 4),
        help="background worker threads for data sampling "
             "(default: config.SAMPLE_THREADS, else 4)",
    )
    parser.add_argument(
        '--once',
        action='store_true',
    )
    parser.add_argument(
        '--no-refit',
        action='store_true',
        default=not getattr(config, 'REFIT', True),
        help="never accept loss-model refit jobs from the server "
             "(default from config.REFIT)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["torch", "jax"],
        default=config.BACKEND,
        help=f"training backend (default: '{config.BACKEND}')",
    )
    parser.set_defaults(func=main)
