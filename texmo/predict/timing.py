"""Training-run time prediction model for the JAX chunked-scan trainer.

For a given (system, precision) pair, predicts total wall-clock time
of a training run, decomposed into four positive linear blocks:

    T_total = T_init  +  N_full * T_scan_full
              +  has_short * T_scan_short
              +  total_steps * T_step

with

    T_init       = sum_c <w_init[t_c],       f_c>      # per-layer features
    T_step       = sum_c <w_step[t_c],       f_c>      # per-layer features
    T_scan_full  = <w_scan_full[t_in],       g_full>   # input-only features
    T_scan_short = <w_scan_short[t_in],      g_short>  # input-only + N_short

The chunk structure mirrors `ManagerJax.train`: training is batched
into `lax.scan` calls of size `CHUNK_SIZE`. A run with `S` total steps
runs `S // CHUNK_SIZE` full-size scans plus, if `S % CHUNK_SIZE > 0`,
a final short scan of `S % CHUNK_SIZE` steps. JIT cost paid once
per unique scan shape becomes the `T_init`/`T_scan_short` constants;
the per-step cost inside a scan becomes `T_step`.

`T_scan_full` / `T_scan_short` are keyed by the input-layer type
because the scan only carries the int8 token tensor through it; the
hidden layers contribute to `T_step`/`T_init` via per-layer features.

Fit is a single joint NNLS solve over all four blocks of weights. We
prefer absolute MSE in seconds over log-space relative error: search
makes time-budget decisions where a 10× miss on a 1000s run is a much
bigger problem than a 10× miss on a 1s run.
"""

import functools
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

from ..configuration import Configuration
from ..layers.attn import AttnDef
from ..layers.conv import ConvDef
from ..layers.dense import DenseDef
from ..layers.gru import GruDef, MgruDef, MinGruDef
from ..layers.latent import LatentDef, LrnnDef
from ..layers.lmgu import LmguDef
from ..layers.lstm import LstmDef
from ..layers.matlstm import MatLstmDef
from ..layers.msr import MsrDef
from ..layers.mullstm import MulLstmDef
from ..layers.norm import NormDef
from ..layers.rglru import RglruDef
from ..layers.rmsnorm import RmsNormDef
from ..layers.rnn import RnnDef
from ..layers.slstm import SLstmDef
from ..layers.split import SplitDef
from ..layers.suffix import SuffixDef
from ..precision import Precision
from .predict_common import layer_type_id

# Must match `_CHUNK_SIZE` in manager_jax.py. Defined here too so that
# importing the timing model doesn't pull in JAX (search runs in a
# CPU-only thread and shouldn't drag the accelerator runtime in).
CHUNK_SIZE = 256


# -- Per-layer feature extractors ----------------------------------------


def _features_base(isize: int, osize: int, batch: int, length: int) -> list[float]:
    return [
        1.0,
        length,
        length * batch,
        isize,
        isize * length,
        isize * length * batch,
        osize,
        osize * length,
        osize * length * batch,
    ]


def _features_dense(isize: int, osize: int, batch: int, length: int) -> list[float]:
    return _features_base(isize, osize, batch, length) + [
        isize * osize,
        isize * osize * length,
        isize * osize * length * batch,
    ]


def _features_suffix(
    isize: int, batch: int, length: int, suffix_length: int
) -> list[float]:
    osize = isize * suffix_length
    return _features_base(isize, osize, batch, length) + [suffix_length]


def _features_conv(
    isize: int, batch: int, length: int, kernel: int
) -> list[float]:
    # Depthwise causal conv: each output channel reads `kernel` past
    # positions of the same input channel, so cost scales with
    # isize * kernel per output position (no I*O cross term like dense).
    # Output channel count == input channel count.
    return _features_base(isize, isize, batch, length) + [
        isize * kernel,
        isize * kernel * length,
        isize * kernel * length * batch,
    ]


def _features_rglru(
    isize: int, block_width: int, batch: int, length: int
) -> list[float]:
    # Size-preserving. Two block-diagonal gates: per position the matmul
    # cost is 2 * blocks * block_width^2 = 2 * isize * block_width
    # (block_width = isize / blocks), scaling from isize^2 (blocks=1, a
    # full gate) down to isize (per-channel, block_width=1). The input
    # gating + diagonal recurrence are elementwise (the base terms);
    # NNLS picks the coefficient (absorbs the factor 2). Gates are
    # hoisted out of the scan, so they carry the length factor.
    return _features_base(isize, isize, batch, length) + [
        isize * block_width,
        isize * block_width * length,
        isize * block_width * length * batch,
    ]


def _features_attn(
    isize: int, osize: int, batch: int, length: int, window: int,
) -> list[float]:
    # Dense matmul features cover the q/k/v/o projections (NNLS picks
    # the coefficient). The window terms cover the banded score and
    # value-mix matmuls: per token ~ window * size each, over the
    # sequence ~ length * window * size.
    return _features_dense(isize, osize, batch, length) + [
        osize * window,
        osize * window * length,
        osize * window * length * batch,
    ]


def _features_matlstm(
    isize: int, osize: int, batch: int, length: int
) -> list[float]:
    # Dense matmul features cover Q/K/V/IFO projections. Add three
    # OS^2 terms for the per-step matrix-state ops (outer product
    # v*k^T, scaled add into C, C@q readout). No L^2 -- matLSTM is
    # scan-based, not parallel-form.
    return _features_dense(isize, osize, batch, length) + [
        osize * osize,
        osize * osize * length,
        osize * osize * batch * length,
    ]


def _features_msr(
    isize: int, osize: int, batch: int, length: int
) -> list[float]:
    # Dense matmul features cover the stacked Q/K/V projection (NNLS
    # picks a ~3x larger coefficient). The new L^2 terms cover the two
    # T x T einsums (Q K^T and decayed scores @ V) and the decay
    # elementwise multiply — being defensive with multiple shapes so
    # NNLS can pick whichever fits the measured data.
    return _features_dense(isize, osize, batch, length) + [
        length * length,
        length * length * osize,
        length * length * osize * batch,
    ]


def _features_reps_recurrent(
    isize: int, osize: int, batch: int, length: int, reps: int
) -> list[float]:
    # latent/lrnn/lmgu iterate their internal (size x size) recurrence
    # `reps` times per token while the input projection fires once.
    # Base block: dense features + the OS^2 triple (the matlstm shape)
    # covering the one-shot work. The same block reps-scaled lets NNLS
    # apportion everything that repeats per iteration -- the OS^2
    # matmul FLOPs *and* the per-iteration dispatch/elementwise
    # overheads (dominant at small sizes), at every shape. Non-work
    # members (reps-scaled IS terms, one-shot OS^2) get ~0 weights;
    # lmgu's 4 inner matrices vs latent's 1 are absorbed by the
    # per-type coefficient.
    base = _features_dense(isize, osize, batch, length) + [
        float(osize * osize),
        float(osize * osize * length),
        float(osize * osize * batch * length),
    ]
    return base + [reps * f for f in base]


def _features_split(osize: int, batch: int, length: int) -> list[float]:
    # The merge itself: an elementwise op (add/mul) or a concat copy
    # over the merged channels per position. Same cheap shape as the
    # old skip merge; keyed per op (split.add / split.mul / split.cat)
    # so each learns its own coefficient. The branches' compute shows
    # up as their own per-layer components (sum-over-branches), so this
    # term is only the combine cost. osize is the merged output width.
    return [1.0, osize, osize * length, osize * batch * length]


def _features_input(batch: int, length: int) -> list[float]:
    return [1.0, length, length * batch]


# -- Scan dispatch features (input-only). Cost is dominated by JAX
#    bookkeeping over the scan tensor, which scales with (batch,
#    length); the model architecture inside the scan is captured by
#    `T_step`/`T_init`, not here.

_SCAN_FULL_FEATURES = 3   # [1, batch, batch*length]
_SCAN_SHORT_FEATURES = 4  # full features + N_short


def _features_scan_full(batch: int, length: int) -> list[float]:
    return [1.0, float(batch), float(batch * length)]


def _features_scan_short(
    batch: int, length: int, short_steps: int
) -> list[float]:
    return [1.0, float(batch), float(batch * length), float(short_steps)]


# -- Layer-type wiring ----------------------------------------------------


_MATMUL_LAYER_TYPES = (
    DenseDef,
    RnnDef,
    GruDef,
    MgruDef,
    MinGruDef,
    LstmDef,
    SLstmDef,
    MulLstmDef,
)

# Layers whose inner recurrence fires `reps` times per token.
_REPS_LAYER_TYPES = (LatentDef, LrnnDef, LmguDef)


@dataclass
class Component:
    """One component contributing to per-layer features."""

    type_id: str  # e.g. "dense.gelu", "rnn.tanh", "bits.1+bp", "output"
    features: np.ndarray


def _input_component(input_def, batch: int, length: int) -> Component:
    type_id = str(input_def)
    # One key per token domain for embedded inputs: 'bytes.emb.4' and
    # 'bytes.emb.8' are the same lookup cost-wise, and per-width keys
    # would each need their own runs to fit.
    if '.emb.' in type_id:
        type_id = type_id[:type_id.index('.emb.') + len('.emb')]
    return Component(
        type_id=type_id,
        features=np.array(_features_input(batch, length)),
    )


def _layer_component(layer, batch: int, length: int) -> Component:
    type_id = layer_type_id(layer)

    if isinstance(layer, SuffixDef):
        features = _features_suffix(layer.input_size, batch, length, layer.length)
    elif isinstance(layer, ConvDef):
        features = _features_conv(
            layer.input_size, batch, length, layer.kernel)
    elif isinstance(layer, MatLstmDef):
        features = _features_matlstm(
            layer.input_size, layer.size, batch, length)
    elif isinstance(layer, MsrDef):
        features = _features_msr(layer.input_size, layer.size, batch, length)
    elif isinstance(layer, RglruDef):
        features = _features_rglru(
            layer.input_size, layer.block_width, batch, length)
    elif isinstance(layer, AttnDef):
        features = _features_attn(
            layer.input_size, layer.size, batch, length, layer.window)
    elif isinstance(layer, _REPS_LAYER_TYPES):
        features = _features_reps_recurrent(
            layer.input_size, layer.size, batch, length, layer.reps)
    elif isinstance(layer, _MATMUL_LAYER_TYPES):
        features = _features_dense(layer.input_size, layer.size, batch, length)
    elif isinstance(layer, (NormDef, RmsNormDef)):
        # rmsnorm adds a per-channel (1+gamma) multiply on top of norm's
        # reduce -- still O(size) elementwise, so the base features cover
        # both (keyed separately by type_id).
        features = _features_base(layer.input_size, layer.size, batch, length)
    else:
        raise ValueError(f"unknown layer type for timing model: {layer}")

    return Component(type_id=type_id, features=np.array(features))


def _output_component(output_def, batch: int, length: int) -> Component:
    return Component(
        type_id="output",
        features=np.array(
            _features_dense(output_def.input_size, output_def.size, batch, length)
        ),
    )


def _split_component(split, batch: int, length: int) -> Component:
    return Component(
        type_id=layer_type_id(split),  # split.add / split.mul / split.cat
        features=np.array(_features_split(split.size, batch, length)),
    )


def _collect_layer_components(
    layers: list, batch: int, length: int, out: list[Component]
) -> None:
    """Flatten a layer sequence into feature components, recursing into
    split branches. A split adds a merge component plus the components
    of every layer in its branches -- so the branches' compute is
    summed in (and multi-way generalizes for free)."""
    for layer in layers:
        if isinstance(layer, SplitDef):
            out.append(_split_component(layer, batch, length))
            for branch in layer.branches:
                _collect_layer_components(branch.layers, batch, length, out)
        else:
            out.append(_layer_component(layer, batch, length))


@functools.cache
def featurize(conf: Configuration) -> list[Component]:
    """Per-layer feature components. components[0] is the input layer.

    Cached because Configuration is immutable and the same confs get
    re-featurized on every refit / batched prediction.
    """
    model_def = conf.model
    batch = conf.batch
    length = conf.length
    components: list[Component] = [
        _input_component(model_def.input, batch, length)
    ]
    _collect_layer_components(
        model_def.layer_seq.layers, batch, length, components)
    components.append(_output_component(model_def.output, batch, length))
    return components


def chunk_structure(steps: int) -> tuple[int, int]:
    """Return (num_full_scans, short_steps) for a chunked-scan run."""
    return steps // CHUNK_SIZE, steps % CHUNK_SIZE


# -- Weights container ----------------------------------------------------


@dataclass
class Weights:
    """Fitted weights for one (system, precision) pair.

    All four dicts hold non-negative arrays; missing keys are silently
    treated as zero contributions (so configs containing a layer type
    unseen in training underestimate but don't blow up).
    """

    init: dict[str, np.ndarray]
    step: dict[str, np.ndarray]
    scan_full: dict[str, np.ndarray]
    scan_short: dict[str, np.ndarray]


def _dot(weights: dict[str, np.ndarray], type_id: str, features) -> float:
    w = weights.get(type_id)
    if w is None or len(w) != len(features):
        # A length mismatch means the weights predate a feature-schema
        # change for this type; treat like an unseen type (zero
        # contribution) until the next refit rather than crashing.
        return 0.0
    return float(np.dot(w, features))


def _predict_init(weights: Weights, components: list[Component]) -> float:
    return sum(_dot(weights.init, c.type_id, c.features) for c in components)


def _predict_step(weights: Weights, components: list[Component]) -> float:
    return sum(_dot(weights.step, c.type_id, c.features) for c in components)


def _predict_scan_full(
    weights: Weights, input_type_id: str, batch: int, length: int
) -> float:
    return _dot(
        weights.scan_full,
        input_type_id,
        _features_scan_full(batch, length),
    )


def _predict_scan_short(
    weights: Weights,
    input_type_id: str,
    batch: int,
    length: int,
    short_steps: int,
) -> float:
    return _dot(
        weights.scan_short,
        input_type_id,
        _features_scan_short(batch, length, short_steps),
    )


def predict_total_time(weights: Weights, conf: Configuration) -> float:
    """Predict total run time for `conf`."""
    components = featurize(conf)
    input_type_id = components[0].type_id
    num_full, short_steps = chunk_structure(conf.steps)

    init = _predict_init(weights, components)
    step = _predict_step(weights, components)
    scan_full = _predict_scan_full(
        weights, input_type_id, conf.batch, conf.length)
    scan_short = (
        _predict_scan_short(
            weights, input_type_id, conf.batch, conf.length, short_steps)
        if short_steps > 0 else 0.0
    )
    return (
        init
        + num_full * scan_full
        + scan_short
        + conf.steps * step
    )


def predict_step_time(weights: Weights, conf: Configuration) -> float:
    """Predict steady-state per-step cost (T_step component only)."""
    return _predict_step(weights, featurize(conf))


def _predict_total_time_batch(
    weights: Weights, confs: list[Configuration]
) -> np.ndarray:
    """Vectorized total-time prediction for many confs.

    Groups per-layer features by type into (n_confs, feat_size)
    matrices and runs one matmul per type for each of the init/step
    blocks, so per-conf Python overhead disappears.
    """
    n = len(confs)
    if n == 0:
        return np.zeros(0)

    n_full = np.zeros(n)
    n_short = np.zeros(n)
    steps_arr = np.zeros(n)
    batches = np.zeros(n, dtype=np.int64)
    lengths = np.zeros(n, dtype=np.int64)
    input_types: list[str] = []

    feats_per_type: dict[str, np.ndarray] = {}
    for i, conf in enumerate(confs):
        nf, ns = chunk_structure(conf.steps)
        n_full[i] = nf
        n_short[i] = ns
        steps_arr[i] = conf.steps
        batches[i] = conf.batch
        lengths[i] = conf.length
        comps = featurize(conf)
        input_types.append(comps[0].type_id)
        for c in comps:
            arr = feats_per_type.get(c.type_id)
            if arr is None:
                arr = np.zeros((n, len(c.features)))
                feats_per_type[c.type_id] = arr
            arr[i] += c.features

    init = np.zeros(n)
    step = np.zeros(n)
    for t, feats in feats_per_type.items():
        wi = weights.init.get(t)
        if wi is not None:
            init += feats @ wi
        ws = weights.step.get(t)
        if ws is not None:
            step += feats @ ws

    scan_full = np.zeros(n)
    scan_short = np.zeros(n)
    for i, t in enumerate(input_types):
        b, ll = int(batches[i]), int(lengths[i])
        wsf = weights.scan_full.get(t)
        if wsf is not None:
            scan_full[i] = float(np.dot(wsf, _features_scan_full(b, ll)))
        if n_short[i] > 0:
            wss = weights.scan_short.get(t)
            if wss is not None:
                scan_short[i] = float(np.dot(
                    wss,
                    _features_scan_short(b, ll, int(n_short[i])),
                ))

    return init + n_full * scan_full + scan_short + steps_arr * step


# -- Fitting --------------------------------------------------------------


def _build_design_matrix(
    samples: list[tuple[Configuration, float]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pack samples into the (X, y) matrix for joint NNLS.

    Each row corresponds to one (conf, total_time) sample. Columns are
    the concatenation of four blocks (`init`, `step`, `scan_full`,
    `scan_short`), with each block split per type. Returns (X, y, layout)
    where `layout` maps each block/type to its (offset, size) so the
    fitted theta can be unflattened.
    """
    # First pass: discover per-type feature widths.
    layer_sizes: dict[str, int] = {}
    input_type_ids: set[str] = set()
    for conf, _ in samples:
        comps = featurize(conf)
        input_type_ids.add(comps[0].type_id)
        for c in comps:
            if c.type_id not in layer_sizes:
                layer_sizes[c.type_id] = len(c.features)

    layer_keys = sorted(layer_sizes.keys())
    input_keys = sorted(input_type_ids)

    layout = {
        "init": {},        # type_id -> (offset, size)
        "step": {},
        "scan_full": {},
        "scan_short": {},
    }
    pos = 0
    for k in layer_keys:
        layout["init"][k] = (pos, layer_sizes[k])
        pos += layer_sizes[k]
    for k in layer_keys:
        layout["step"][k] = (pos, layer_sizes[k])
        pos += layer_sizes[k]
    for k in input_keys:
        layout["scan_full"][k] = (pos, _SCAN_FULL_FEATURES)
        pos += _SCAN_FULL_FEATURES
    for k in input_keys:
        layout["scan_short"][k] = (pos, _SCAN_SHORT_FEATURES)
        pos += _SCAN_SHORT_FEATURES

    n = len(samples)
    X = np.zeros((n, pos), dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)

    for i, (conf, total_time) in enumerate(samples):
        y[i] = total_time
        comps = featurize(conf)
        in_t = comps[0].type_id
        nf, ns = chunk_structure(conf.steps)

        for c in comps:
            io, isz = layout["init"][c.type_id]
            X[i, io:io + isz] += c.features
            so, ssz = layout["step"][c.type_id]
            X[i, so:so + ssz] += conf.steps * c.features

        if nf > 0:
            off, sz = layout["scan_full"][in_t]
            X[i, off:off + sz] += nf * np.array(
                _features_scan_full(conf.batch, conf.length))
        if ns > 0:
            off, sz = layout["scan_short"][in_t]
            X[i, off:off + sz] += np.array(
                _features_scan_short(conf.batch, conf.length, ns))

    return X, y, layout


def _fit(
    samples: list[tuple[Configuration, float]],
) -> tuple[Weights, float]:
    """Fit weights to total-time samples via NNLS. Returns (weights, mse).

    mse is the mean squared residual in seconds².
    """
    X, y, layout = _build_design_matrix(samples)
    # NNLS default is maxiter = 3*n_features, which can be too tight
    # for the wide design matrices we build here. Bump it generously.
    theta, _ = nnls(X, y, maxiter=10 * X.shape[1])
    pred = X @ theta
    mse = float(np.mean((pred - y) ** 2))

    def _slice(block: str) -> dict[str, np.ndarray]:
        return {
            k: theta[off:off + sz]
            for k, (off, sz) in layout[block].items()
        }

    weights = Weights(
        init=_slice("init"),
        step=_slice("step"),
        scan_full=_slice("scan_full"),
        scan_short=_slice("scan_short"),
    )
    return weights, mse


# -- Model class ----------------------------------------------------------


class TrainTimingModel:
    """Per-(system, precision) total-time predictor for chunked-scan runs."""

    MIN_TIME = 0.1     # skip runs faster than 100ms total (noise floor).
    MIN_STEPS = 4
    MIN_SAMPLES = 20

    # Cap on runs read for a fit: the most recent N per (system,
    # precision). Bounds the read/parse cost as the DB grows, and
    # recent runs reflect the machine's current performance anyway
    # (code and drivers change). The least-squares is long saturated
    # at this sample size.
    FIT_RUNS_CAP = 20_000

    def __init__(self):
        self._weights: dict[tuple[str, Precision], Weights] = {}
        self._losses: dict[tuple[str, Precision], float] = {}

    @staticmethod
    def _run_total_time(run, steps: int) -> float | None:
        """Extract usable total time from a Run, or None if it should be skipped."""
        if run.train_time is None or run.train_time <= 0:
            return None
        if steps < TrainTimingModel.MIN_STEPS:
            return None
        if run.train_time < TrainTimingModel.MIN_TIME:
            return None
        return run.train_time

    def fit(self, runs: list[tuple[Configuration, object]], verbose: bool = True):
        buckets: dict[
            tuple[str, Precision], list[tuple[Configuration, float]]
        ] = {}
        for conf, run in runs:
            total_time = self._run_total_time(run, conf.steps)
            if total_time is None:
                continue
            key = (run.system, conf.precision)
            buckets.setdefault(key, []).append((conf, total_time))

        for key, samples in buckets.items():
            if len(samples) < self.MIN_SAMPLES:
                if verbose:
                    print(f"skipping {key}: only {len(samples)} samples")
                continue
            weights, mse = _fit(samples)
            self._weights[key] = weights
            self._losses[key] = mse
            if verbose:
                rmse = math.sqrt(mse)
                print(
                    f"fit {key}: {len(samples)} samples, "
                    f"RMSE={rmse:.3f}s"
                )

    def predict(
        self, system: str, conf: Configuration
    ) -> float | None:
        """Predict total run time, or None if no model is fit for this pair."""
        weights = self._weights.get((system, conf.precision))
        if weights is None:
            return None
        return predict_total_time(weights, conf)

    def has_weights(self, system: str, precision: Precision) -> bool:
        """Whether a fitted model exists for this (system, precision)."""
        return (system, precision) in self._weights

    def predict_batch(
        self, system: str, confs: list[Configuration]
    ) -> np.ndarray | None:
        """Predict total run time for many confs of the same precision."""
        if not confs:
            return np.zeros(0)
        weights = self._weights.get((system, confs[0].precision))
        if weights is None:
            return None
        return _predict_total_time_batch(weights, confs)

    def predict_step_time(
        self, system: str, conf: Configuration
    ) -> float | None:
        """Predict steady-state per-step cost only (no init/scan terms)."""
        weights = self._weights.get((system, conf.precision))
        if weights is None:
            return None
        return predict_step_time(weights, conf)

    def predict_max_steps(
        self, system: str, conf: Configuration, time_budget: float,
    ) -> int | None:
        """Largest power-of-2 `steps` whose predicted total time fits the budget.

        Returns None if no model is fit for this (system, precision),
        and 0 if even `steps=2` exceeds the budget.
        """
        weights = self._weights.get((system, conf.precision))
        if weights is None:
            return None
        components = featurize(conf)
        input_type_id = components[0].type_id
        init = _predict_init(weights, components)
        scan_full = _predict_scan_full(
            weights, input_type_id, conf.batch, conf.length)
        step = _predict_step(weights, components)

        # Coarse upper bound from per-step + amortized full-scan dispatch.
        per_step_amortized = step + scan_full / CHUNK_SIZE
        if per_step_amortized <= 0:
            # Degenerate fit (no signal at all): can't bound, skip.
            return 0
        upper = int((time_budget - init) / per_step_amortized) + 1
        if upper < 2:
            return 0
        candidate = 1 << int(math.floor(math.log2(upper)))
        while candidate >= 2:
            num_full = candidate // CHUNK_SIZE
            short = candidate % CHUNK_SIZE
            scan_short = (
                _predict_scan_short(
                    weights, input_type_id,
                    conf.batch, conf.length, short)
                if short > 0 else 0.0
            )
            total = (
                init
                + num_full * scan_full
                + scan_short
                + candidate * step
            )
            if total <= time_budget:
                return candidate
            candidate >>= 1
        return 0

    def keys(self) -> list[tuple[str, Precision]]:
        return list(self._weights.keys())

    def loss(self, system: str, precision: Precision) -> float:
        return self._losses[(system, precision)]

    def snapshot(self) -> 'TrainTimingModel':
        """Return a new model sharing none of the caller's mutable state.

        `_weights` values are fitted dataclasses we never mutate in place,
        so a shallow dict copy suffices to insulate readers from future
        refits.
        """
        s = TrainTimingModel()
        s._weights = dict(self._weights)
        s._losses = dict(self._losses)
        return s
