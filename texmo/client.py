import csv
import io
import logging
import math
import pickle
import time

import numpy as np
import requests

from .common import ttoa3
from .configuration import Configuration
from .dataset import DataSetWrapper
from .latency import timer
from .manager import create_manager
from .precision import Precision
from .predict import loss_tree
from .spec_parser import parse_model2

# (connect, read) timeouts for server requests. Without these a hung
# connection (or a wedged server) blocks the client forever; with them
# the request raises requests.Timeout, which the retry/backoff paths
# already handle. The read budget is generous because select_conf can
# legitimately take tens of seconds on a busy server.
_HTTP_TIMEOUT = (10, 120)

# Timeouts for the loss-refit legs: the training-data download is a
# few tens of MB, and the model POST follows a multi-minute fit.
_REFIT_HTTP_TIMEOUT = (10, 600)


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
                timeout=_HTTP_TIMEOUT,
            )
        )
    except TypeError as e:
        logging.error(f"Failed to post result: {e}")
        logging.error(f"Result: {result}")
        raise


def parse_training_csv(text: str) -> list[tuple[Configuration, float]]:
    """Parse the /training_data CSV into (Configuration, loss) pairs.

    Configurations sharing (spec, precision) reuse one parsed
    Model2Def -- the parse is the expensive part (the server measured
    ~3 min for 900k rows without this dedup), and hyperparameter
    variants of one spec are common.
    """
    models: dict[tuple[str, str], object] = {}
    train_data: list[tuple[Configuration, float]] = []
    rows = csv.reader(io.StringIO(text))
    next(rows)  # header
    for spec, prec, lr, length, batch, steps, decay, cosine, loss in rows:
        key = (spec, prec)
        model = models.get(key)
        if model is None:
            model = parse_model2(spec, Precision(prec))
            models[key] = model
        conf = Configuration(
            model=model, lr=float(lr), length=int(length),
            batch=int(batch), steps=int(steps), decay=float(decay),
            cosine=bool(int(cosine)),
        )
        train_data.append((conf, float(loss)))
    return train_data


def run_refit_job(
    session: requests.Session, base: str, system: str,
) -> None:
    """Execute a loss-refit job: download the training data, fit the
    production tree model, POST the pickled result back. The model
    carries its own freshness measure (run_count = the training rows
    we counted ourselves; see loss_tree.TreeLossModel).

    Any failure is logged and swallowed -- the server issues a fresh
    job at the next refit boundary."""
    logging.info("Starting loss-refit job")
    t0 = time.time()
    with timer("client.refit.download"):
        r = session.get(
            f"{base}/training_data", timeout=_REFIT_HTTP_TIMEOUT)
        r.raise_for_status()
    with timer("client.refit.parse"):
        train_data = parse_training_csv(r.text)
    with timer("client.refit.fit"):
        model = loss_tree.train_from_data(train_data)
    if model is None:
        logging.warning("Refit job got empty training data, skipping")
        return
    fit_s = time.time() - t0
    with timer("client.refit.post"):
        r = session.post(
            f"{base}/submit_loss_model",
            params={
                "system": system,
                "fit_s": f"{fit_s:.1f}",
            },
            data=pickle.dumps(model),
            headers={"Content-Type": "application/octet-stream"},
            timeout=_REFIT_HTTP_TIMEOUT,
        )
        r.raise_for_status()
    d = r.json()
    logging.info(
        f"Refit job done in {ttoa3(fit_s)} "
        f"({len(train_data)} runs): "
        + ("accepted" if d.get("accepted")
           else f"rejected ({d.get('reason')})"))


def worker_loop(
    server_host: str,
    system: str,
    dataset: DataSetWrapper,
    backend: str,
    once: bool,
    api_key: str = '',
    refit_ok: bool = True,
):
    # Treat bare "host:port" as http; an explicit scheme (https://...)
    # is preserved -- used when talking to an authenticated remote
    # endpoint that fronts the server with TLS.
    if '://' in server_host:
        base = server_host.rstrip('/')
    else:
        base = f"http://{server_host}"
    select_url = f"{base}/select"
    add_url = f"{base}/add"
    s = requests.Session()
    if api_key:
        s.headers.update({'Authorization': f'Bearer {api_key}'})

    delay = 1.0
    max_delay = 120.0

    while True:
        try:
            with timer("client.get(select)") as t:
                r = s.get(
                    select_url,
                    params={
                        "system": system,
                        "refit": "1" if refit_ok else "0",
                    },
                    timeout=_HTTP_TIMEOUT)
        except requests.exceptions.RequestException as e:
            logging.warning(f'Request failed: {e}, retrying in {ttoa3(delay)}')
            time.sleep(delay)
            delay = min(delay * 1.5, max_delay)
            continue

        with timer("client.prepare"):
            d = r.json()
            assert d["system"] == system

            refit = d.get("refit_loss")
            if refit is not None:
                try:
                    run_refit_job(s, base, system)
                except Exception as e:
                    # The next refit boundary issues a fresh job.
                    logging.warning(f"Refit job failed: {e!r}")
                if once:
                    break
                continue

            if d["conf"] is None:
                logging.info(f'Server has no conf, sleeping {ttoa3(delay)}')
                time.sleep(delay)
                delay = min(delay * 1.5, max_delay)
                continue
            delay = 1.0

            strategy = d.get("strategy")
            covered, total = d.get("covered"), d.get("total")
            meta = f'strategy={strategy}'
            if covered is not None and total is not None:
                meta += f', coverage={covered}/{total}'
            logging.info(
                f'Got configuration in {ttoa3(t.value())} ({meta})')
            conf = Configuration.from_dict(d["conf"])

            manager = create_manager(
                backend, conf=conf, system=system, dataset=dataset, verbose=False)

        with timer("client.train_and_eval"):
            run, out_conf = manager.train_and_eval(
                steps=conf.steps,
                time_limit=None,
            )

        with timer("client.post(add)"):
            post_result(s, add_url, system, run, out_conf, strategy)

        if once:
            break
