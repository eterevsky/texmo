"""Hyperparameter sweep for the loss-prediction RNN.

Runs each config under N seeds, computes val L1 on the standard 90/10
conf-id split, and prints a markdown row with median + range. Use to
re-tune `loss_rnn.fit` periodically as the dataset grows.

Run: ``uv run python -m texmo.predict.loss_sweep [--seeds N] [--db PATH]``
"""

import argparse
import dataclasses
import logging
from typing import Any

import numpy as np

import config
from ..common import console  # noqa: F401  -- enables rich logging
from ..resultdb import ResultDB
from ..tokens import set_tokens_dir
from . import loss_cli


@dataclasses.dataclass
class _Variant:
    label: str
    factory: Any  # () -> loss_cli.Predictor
    deterministic: bool = False  # if True, only 1 seed ever runs


def _baseline_kwargs() -> dict:
    """Current production config (the 'best' from the prior sweep)."""
    return dict(
        cell_activation='tanh', hidden=32,
        lr=0.02, steps=8000, lr_schedule='cosine',
    )


def _rnn(label: str, **overrides) -> _Variant:
    kwargs = _baseline_kwargs()
    kwargs.update(overrides)

    def factory(seed: int):
        return loss_cli.RnnPredictor(seed=seed, **kwargs)
    return _Variant(label=label, factory=factory)


def _hist_gbr(label: str, with_second_layer: bool) -> _Variant:
    def factory(seed: int):
        return loss_cli.HistGBRPredictor(with_second_layer=with_second_layer)
    return _Variant(label=label, factory=factory, deterministic=True)


def _build_variants() -> list[_Variant]:
    return [
        _rnn(
            'baseline (h=32 lr=0.02 cos 8k bs=1024)',
        ),
        _rnn(
            'head=32.gelu',
            out_hidden=32, out_activation='gelu',
        ),
        _rnn(
            'head=32.gelu + bs=2048',
            out_hidden=32, out_activation='gelu', batch_size=2048,
        ),
        _rnn(
            'head=32.gelu + feat_proj=32',
            out_hidden=32, out_activation='gelu', feat_proj=32,
        ),
        _rnn(
            'head=32.gelu + feat_proj=64',
            out_hidden=32, out_activation='gelu', feat_proj=64,
        ),
        _rnn(
            'head=32.gelu + feat_proj=32 + bs=2048',
            out_hidden=32, out_activation='gelu',
            feat_proj=32, batch_size=2048,
        ),
        _rnn(
            'head=32.gelu + feat_proj=64 + bs=2048',
            out_hidden=32, out_activation='gelu',
            feat_proj=64, batch_size=2048,
        ),
    ]


def _eval_variant(
    variant: _Variant,
    train_data, val_confs, val_targets,
    seeds: list[int],
) -> tuple[float, float, float]:
    """Train+eval `variant` under each seed; return (median, min, max) val L1."""
    seeds_to_run = [seeds[0]] if variant.deterministic else seeds
    losses = []
    for s in seeds_to_run:
        predictor = variant.factory(s)
        predictor.fit(train_data)
        preds = predictor.predict(val_confs)
        loss = loss_cli._eval_log_l1(preds, val_targets)
        losses.append(loss)
        logging.info(f"  {variant.label} seed={s}: val L1 = {loss:.4f}")
    arr = np.array(losses)
    return float(np.median(arr)), float(arr.min()), float(arr.max())


def main(args):
    set_tokens_dir(args.tokens_dir)
    logging.info(f"Loading labeled runs from {args.db}")
    db = ResultDB(args.db, readonly=True)
    labeled = list(db.iter_labeled_runs())
    logging.info(f"Loaded {len(labeled)} labeled runs")

    train_data, val_data = loss_cli._train_val_split(labeled)
    logging.info(f"Split: train={len(train_data)}, val={len(val_data)}")

    val_confs = [c for c, _ in val_data]
    val_targets = np.array([loss for _, loss in val_data])

    seeds = list(range(1, args.seeds + 1))
    variants = _build_variants()
    rows: list[tuple[str, float, float, float, float]] = []

    for v in variants:
        logging.info(f"\n=== {v.label} ===")
        median, lo, hi = _eval_variant(
            v, train_data, val_confs, val_targets, seeds)
        typ = (2.0 ** median - 1.0) * 100.0
        rows.append((v.label, median, typ, lo, hi))
        logging.info(
            f"{v.label}: median val L1 = {median:.4f} "
            f"(~{typ:.1f}% typical), range [{lo:.4f}, {hi:.4f}]"
        )

    print()
    print('| Config | val L1 (med) | ~typ | range |')
    print('|---|---|---|---|')
    for label, median, typ, lo, hi in rows:
        print(
            f'| {label} | {median:.4f} | {typ:.1f}% | '
            f'{hi - lo:.4f} |'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, default=config.DB)
    parser.add_argument('--tokens-dir', type=str, default=config.TOKENS_DIR)
    parser.add_argument(
        '--seeds', type=int, default=5,
        help='Number of seeds per RNN config (HistGBR runs once)',
    )
    main(parser.parse_args())
