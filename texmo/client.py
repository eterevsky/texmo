import argparse
import logging
import math
import time

import numpy as np
import requests

from .common import ttoa3
from .configuration import Configuration
from .dataset import DataSet, DataSetWrapper
from .latency import report, timer
from .manager import create_manager
from .tokens import set_tokens_dir


def retry(callable, min_delay=1, exp=1.5, max_delay=120):
    delay = min_delay
    while True:
        try:
            return callable()
        except requests.exceptions.RequestException as e:
            logging.warning(f"Request failed: {e}")
            logging.warning(f"Retrying in {ttoa3(delay)}")
            time.sleep(delay)
            delay = min(delay * exp, max_delay)


def sanitize_float(f: float) -> float | None:
    if math.isnan(f):
        return None
    elif math.isinf(f):
        return 1E12
    return f


def sanitize_json_list(l: list):
    for i, v in enumerate(l):
        if isinstance(v, dict):
            sanitize_json_dict(v)
        elif isinstance(v, list):
            sanitize_json_list(v)
        elif (
            isinstance(v, np.float16)
            or isinstance(v, np.float32)
            or isinstance(v, np.float64)
            or isinstance(v, float)
        ):
            l[i] = sanitize_float(float(v))


def sanitize_json_dict(d: dict):
    for k, v in d.items():
        if isinstance(v, dict):
            sanitize_json_dict(v)
        elif isinstance(v, list):
            sanitize_json_list(v)
        elif (
            isinstance(v, np.float16)
            or isinstance(v, np.float32)
            or isinstance(v, np.float64)
            or isinstance(v, float)
        ):
            d[k] = sanitize_float(float(v))


def sanitize_json(d: dict):
    sanitize_json_dict(d)


def post_result(session, add_url: str, system, run, conf, strategy):
    result = {
        "system": system,
        "run": run.to_dict(),
        "conf": conf.replace(steps=run.steps).to_dict(),
        "strategy": strategy,
    }
    sanitize_json(result)
    try:
        retry(
            lambda: session.post(
                add_url,
                json=result,
            )
        )
    except TypeError as e:
        logging.error(f"Failed to post result: {e}")
        logging.error(f"Result: {result}")
        raise


def worker_loop(server_host: str, system: str, dataset: DataSetWrapper, backend: str, once: bool):
    select_url = f"http://{server_host}/select"
    add_url = f"http://{server_host}/add"
    s = requests.Session()

    delay = 1.0
    max_delay = 120.0

    while True:
        try:
            with timer("get(select)") as t:
                r = s.get(select_url, params={"system": system})
        except requests.exceptions.RequestException as e:
            logging.warning(f'Request failed: {e}, retrying in {ttoa3(delay)}')
            time.sleep(delay)
            delay = min(delay * 1.5, max_delay)
            continue

        d = r.json()
        assert d["system"] == system

        if d["conf"] is None:
            logging.info(f'Server has no conf, sleeping {ttoa3(delay)}')
            time.sleep(delay)
            delay = min(delay * 1.5, max_delay)
            continue
        delay = 1.0

        logging.info(f'Got configuration in {ttoa3(t.value())}')
        conf = Configuration.from_dict(d["conf"])
        strategy = d.get("strategy")

        manager = create_manager(
            backend, conf=conf, system=system, dataset=dataset, verbose=False)

        run, out_conf = manager.train_and_eval(
            steps=conf.steps,
            time_limit=None,
        )

        with timer("post(add)"):
            post_result(s, add_url, system, run, out_conf, strategy)

        if conf.decay == 1:
            for checkpoint in run.checkpoints.values():
                with timer("post(add)"):
                    post_result(s, add_url, system,
                                checkpoint.make_run(run), out_conf, strategy)

        if once:
            break


def main(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)
    dataset = DataSet(path=args.data, path_processed=args.data_processed)
    dataset_wrapper = DataSetWrapper(dataset)

    try:
        try:
            worker_loop(
                server_host=args.server, system=args.system, dataset=dataset_wrapper,
                backend=args.backend, once=args.once,
            )
        except KeyboardInterrupt as e:
            logging.warning("Interrupted at:\n%s", e)
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
        help="Host and port for the texmo server",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=config.SYSTEM_NAME,
        help="the name of the system that will be used to identify runs in the DB",
    )
    parser.add_argument(
        '--once',
        action='store_true',
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["torch", "jax"],
        default=config.BACKEND,
        help=f"training backend (default: '{config.BACKEND}')",
    )
    parser.set_defaults(func=main)
