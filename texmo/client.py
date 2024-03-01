import argparse
import json
import logging
import requests

from .configuration2 import Configuration2
from .dataset import DataSet, DataSetWrapper
from .latency import timer, report
from .manager import Manager, release_device_buffers
from .tokens import set_tokens_dir


def worker_loop(server_host: str, system: str, dataset: DataSetWrapper):
    select_url = f"http://{server_host}/select"
    add_url = f"http://{server_host}/add"

    while True:
        with timer("get(select)"):
            r = requests.get(select_url, params={"system": system})
        d = r.json()
        assert d["system"] == system
        conf = Configuration2.from_dict(d["conf"])

        manager = Manager(conf, system, dataset=dataset)
        manager.init(quiet=True)

        run, _weights = manager.train_and_eval(
            steps=conf.steps,
            time_limit=None,
            temp_steps=None,
            temp_dir=None,
            output_dir=None,
            log=None,
            quiet=True,
        )

        # search.add_run(conf, run)

        with timer("post(add)"):
            requests.post(
                add_url,
                json={"system": system, "run": run.to_dict(), "conf": conf.to_dict()}
            )
        release_device_buffers()


def main(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)
    dataset = DataSet(path=args.data, path_processed=args.data_processed)
    dataset_wrapper = DataSetWrapper(dataset)

    try:
        try:
            worker_loop(
                server_host=args.server, system=args.system, dataset=dataset_wrapper
            )
        except KeyboardInterrupt:
            logging.warning("Interrupted\n")
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
    parser.set_defaults(func=main)
