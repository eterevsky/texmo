"""Training-step time prediction model.

For a given (system, precision) pair, predicts the time of a single
training step after warm-up, based on the model spec, sample length,
and batch size. See docs for the design.

The model is an additive sum of per-component terms:

    T_pred = sum_c <exp(θ_c), f_c>

where `c` ranges over the components of the model (input, each hidden
layer, output), `θ_c` is a per-component-type weight vector, and
`f_c` is a feature vector computed from the component's dimensions.

Feature vector depends on the layer type. A base set (length, batch,
input size, output size and a few products) applies to most types; layers
with a dense matmul get three extra features (in*os products);
suffix gets an extra `suffix_length` feature; skip is minimal.

Positivity is enforced via `exp(θ)`. Fitting minimizes mean squared
log-relative error.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..configuration import Configuration
from ..layers.dense import DenseDef
from ..layers.gru import GruDef, MgruDef, MinGruDef
from ..layers.latent import LatentDef, LrnnDef
from ..layers.lstm import LstmDef
from ..layers.norm import NormDef
from ..layers.rnn import RnnDef
from ..layers.skip import SkipDef
from ..layers.suffix import SuffixDef
from ..precision import Precision


# -- Feature extractors. Each returns a numpy array of features for
#    one component. Different component types have different feature
#    sets (and thus different weight-vector sizes).


def _features_base(isize: int, osize: int, batch: int, length: int) -> list[float]:
    """Main features covering length/batch/in/oout interactions (no matmul)."""
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
    """Base features + features for input size × output size.

    For layers with in→out dense projection."""
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


def _features_skip(isize: int, batch: int, length: int) -> list[float]:
    return [1.0, isize, isize * length, isize * batch * length]


def _features_input(batch: int, length: int) -> list[float]:
    """Features for input-encoding layers.

    Since every input layer has fixed output size, no output size features are
    included.
    """
    return [1.0, length, length * batch]


# -- Per-type wiring: map a layer / component to (type_id, features).


_MATMUL_LAYER_TYPES = (
    DenseDef,
    RnnDef,
    GruDef,
    MgruDef,
    MinGruDef,
    LstmDef,
    LatentDef,
    LrnnDef,
)


@dataclass
class Component:
    """One component contributing to training-step time."""

    type_id: str  # e.g. "dense.gelu", "rnn.tanh", "bits.1+bp", "output"
    features: np.ndarray


def _input_component(input_def, batch: int, length: int) -> Component:
    return Component(
        type_id=str(input_def),
        features=np.array(_features_input(batch, length)),
    )


def _layer_component(layer, batch: int, length: int) -> Component:
    if isinstance(layer, DenseDef):
        type_id = f"dense.{layer._activation}"
    elif isinstance(layer, RnnDef):
        type_id = f"rnn.{layer._activation}"
    elif isinstance(layer, SkipDef):
        type_id = f"skip.{layer.op}"
    else:
        type_id = layer.name

    if isinstance(layer, SuffixDef):
        features = _features_suffix(layer.input_size, batch, length, layer.length)
    elif isinstance(layer, SkipDef):
        features = _features_skip(layer.input_size, batch, length)
    elif isinstance(layer, _MATMUL_LAYER_TYPES):
        features = _features_dense(layer.input_size, layer.size, batch, length)
    elif isinstance(layer, NormDef):
        features = _features_base(layer.input_size, layer.size, batch, length)
    else:
        raise ValueError(f"unknown layer type for timing model: {layer}")

    features = np.array(features)

    return Component(type_id=type_id, features=features)


def _output_component(output_def, batch: int, length: int) -> Component:
    # Output is a dense projection (no activation) — same shape as a
    # dense-matmul layer. Use a dedicated type so it gets its own fit.
    return Component(
        type_id="output",
        features=np.array(
            _features_dense(output_def.input_size, output_def.size, batch, length)
        ),
    )


def featurize(conf: Configuration, batch: int, length: int) -> list[Component]:
    """Turn a configuration + runtime shape into a list of components."""
    model_def = conf.model
    components: list[Component] = []
    components.append(_input_component(model_def.input, batch, length))
    for layer in model_def.layers:
        components.append(_layer_component(layer, batch, length))
    components.append(_output_component(model_def.output, batch, length))
    return components


# -- Prediction -----------------------------------------------------------


def predict_time(
    weights: dict[str, np.ndarray],
    components: list[Component],
) -> float:
    """Predict step time from weights and component features."""
    total = 0.0
    for c in components:
        theta = weights.get(c.type_id)
        if theta is None:
            raise KeyError(
                f"no weights for component type '{c.type_id}' — "
                f"(system, precision) pair may not have been fit with "
                f"data covering this layer type"
            )
        total += float(np.dot(np.exp(theta), c.features))
    return total


# -- Fitting --------------------------------------------------------------


def _prepare_training_arrays(
    samples: list[tuple[list[Component], float]],
) -> tuple[dict[str, int], dict[str, jnp.ndarray], jnp.ndarray]:
    """Pack samples into per-type feature arrays.

    Returns:
        type_sizes: dict from type_id to feature-vector length.
        feat_per_type: dict from type_id to (n_samples, feat_size) array.
            Rows are zero for samples that don't contain that type.
        times: (n_samples,) array of per-step times.
    """
    # Determine feature size for each type from first appearance.
    type_sizes: dict[str, int] = {}
    for comps, _ in samples:
        for c in comps:
            if c.type_id not in type_sizes:
                type_sizes[c.type_id] = len(c.features)

    n = len(samples)
    feat_per_type = {
        t: np.zeros((n, sz), dtype=np.float64) for t, sz in type_sizes.items()
    }
    times = np.zeros(n, dtype=np.float64)
    for i, (comps, t) in enumerate(samples):
        times[i] = t
        for c in comps:
            # Same-type components at the same sample sum their features.
            feat_per_type[c.type_id][i] += c.features

    return (
        type_sizes,
        {t: jnp.asarray(a) for t, a in feat_per_type.items()},
        jnp.asarray(times),
    )


def _predicted_times(
    theta: dict[str, jnp.ndarray], feats: dict[str, jnp.ndarray]
) -> jnp.ndarray:
    """Vectorized prediction for all samples."""
    total = jnp.zeros(next(iter(feats.values())).shape[0])
    for type_id, features in feats.items():
        w = jnp.exp(theta[type_id])
        total = total + features @ w
    return total


def _loss(
    theta: dict[str, jnp.ndarray],
    feats: dict[str, jnp.ndarray],
    times: jnp.ndarray,
) -> jnp.ndarray:
    """Mean squared log-relative error."""
    pred = _predicted_times(theta, feats)
    pred = jnp.maximum(pred, 1e-6)
    return jnp.mean((jnp.log(pred) - jnp.log(times)) ** 2)


def _fit(
    samples: list[tuple[list[Component], float]],
    steps: int = 2000,
    lr: float = 0.05,
) -> tuple[dict[str, np.ndarray], float]:
    """Fit θ to minimize log-MSE on `samples`. Returns (weights, final_loss)."""
    type_sizes, feats, times = _prepare_training_arrays(samples)

    # Initialize θ small so that exp(θ) starts near zero.
    theta = {t: -14.0 * jnp.ones(sz, dtype=jnp.float64) for t, sz in type_sizes.items()}

    loss_grad = jax.jit(jax.value_and_grad(_loss))
    optimizer = optax.adamw(lr, weight_decay=0.0)
    opt_state = optimizer.init(theta)

    final_loss = float("inf")
    for _ in range(steps):
        final_loss, grads = loss_grad(theta, feats, times)
        updates, opt_state = optimizer.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)

    return (
        {t: np.asarray(theta[t]) for t in type_sizes},
        float(final_loss),
    )


# -- Model class ----------------------------------------------------------


class TrainTimingModel:
    """Per-(system, precision) training-step time predictor."""

    MIN_TIME = 1e-3  # skip runs faster than 1ms (noise floor).
    MIN_STEPS = 4
    MIN_SAMPLES = 20  # don't fit pairs with fewer samples.

    def __init__(self):
        self._weights: dict[tuple[str, Precision], dict[str, np.ndarray]] = {}
        self._losses: dict[tuple[str, Precision], float] = {}

    @staticmethod
    def _run_per_step_time(run, steps: int) -> float | None:
        """Extract per-step time from a Run, or None if unusable."""
        if run.train_time is None or run.train_time <= 0:
            return None
        if steps < TrainTimingModel.MIN_STEPS:
            return None
        per_step = run.train_time / (steps - 1)
        if per_step < TrainTimingModel.MIN_TIME:
            return None
        return per_step

    def fit(self, runs: list[tuple[Configuration, object]], verbose: bool = True):
        buckets: dict[tuple[str, Precision], list[tuple[list[Component], float]]] = {}
        for conf, run in runs:
            per_step = self._run_per_step_time(run, conf.steps)
            if per_step is None:
                continue
            key = (run.system, conf.precision)
            comps = featurize(conf, conf.batch, conf.length)
            buckets.setdefault(key, []).append((comps, per_step))

        for key, samples in buckets.items():
            if len(samples) < self.MIN_SAMPLES:
                if verbose:
                    print(f"skipping {key}: only {len(samples)} samples")
                continue
            weights, loss = _fit(samples)
            self._weights[key] = weights
            self._losses[key] = loss
            if verbose:
                pct = (np.exp(np.sqrt(loss)) - 1) * 100
                print(
                    f"fit {key}: {len(samples)} samples, "
                    f"log-MSE={loss:.4f} (~{pct:.1f}% relative error)"
                )

    def predict(
        self,
        system: str,
        conf: Configuration,
        batch: int,
        length: int,
    ) -> float:
        key = (system, conf.precision)
        weights = self._weights.get(key)
        if weights is None:
            raise KeyError(
                f"no timing model fit for (system={system!r}, "
                f"precision={conf.precision}). "
                f"Need at least {self.MIN_SAMPLES} runs to fit."
            )
        components = featurize(conf, batch, length)
        return predict_time(weights, components)

    def keys(self) -> list[tuple[str, Precision]]:
        return list(self._weights.keys())

    def loss(self, system: str, precision: Precision) -> float:
        return self._losses[(system, precision)]
