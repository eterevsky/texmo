import jax
import jax.numpy as jnp

from ..common import is_power2_int
from ..layer import LayerDef, LayerJax
from ..layer_jax import LayerWeights, xavier_uniform


def _normalize_jax(x: jax.Array, eps: float = 1e-5) -> jax.Array:
    """L2-normalize along the last axis, gradient-safe at x=0.

    Uses `sqrt(||x||^2 + eps)` rather than `max(||x||, eps)` because
    jnp.linalg.norm at the zero vector has a NaN derivative
    (x_i / ||x|| = 0/0), which the max() form would propagate
    through autodiff even when the value path takes the eps branch.
    The additive smoothing inside sqrt makes the gradient well-
    defined everywhere.

    Adds `eps` (not `eps*eps`) inside sqrt: `eps*eps = 1e-10` would
    be a denormal in fp16 (loses precision below ~6e-5) and would
    fall under bf16's coarse mantissa quantum, so the additive
    smoothing would effectively vanish. `eps = 1e-5` is well within
    range for all three dtypes we train in.
    """
    norm_sq = jnp.sum(x * x, axis=-1, keepdims=True)
    return x / jnp.sqrt(norm_sq + eps)


class LatentJax(LayerJax):
    """Depth-recurrent dense layer (latent reasoning).

    Inspired by https://arxiv.org/abs/2502.05171.

    Each call iterates a shared block `reps` times on a fresh latent
    state:

        e = Wi @ x + b
        s_0 = 0
        s_i = tanh(Wr @ normalize(s_{i-1}) + e)   for i = 1..reps
        out = s_reps

    The input contribution `e` is re-injected at every iteration; this
    is what makes the recurrence stable and gives the layer its
    "iterative refinement" character.

    Latent state is always zero-initialized (no RNG needed) -- the
    paper initializes it randomly to encourage path independence, but
    a deterministic layer is what the search and the scan trainer
    want. Raw `x` is not normalized before the Wi projection; prepend
    a `norm` layer in the spec to recover that.
    """

    def __init__(self, input_size: int, size: int, reps: int, dtype):
        super().__init__(input_size, size, dtype)
        self.reps = reps

    def init_weights(self, rng: jax.Array) -> LayerWeights:
        k_i, k_r = jax.random.split(rng)
        return {
            'w_i': xavier_uniform(
                k_i, (self.size, self.input_size), dtype=self.dtype),
            'w_r': xavier_uniform(
                k_r, (self.size, self.size), dtype=self.dtype),
            'b': jnp.zeros(self.size, dtype=self.dtype),
        }

    def step(
        self, weights: LayerWeights, state, x: jax.Array
    ) -> tuple[None, jax.Array]:
        # x: (input_size,)
        e = weights['w_i'] @ x + weights['b']
        w_r = weights['w_r']

        def body(s, _):
            return jnp.tanh(w_r @ _normalize_jax(s) + e), None

        s, _ = jax.lax.scan(
            body, jnp.zeros(self.size, dtype=self.dtype), length=self.reps)
        return None, s

    def forward(
        self, weights: LayerWeights, inputs: jax.Array
    ) -> jax.Array:
        # inputs: (batch, seq_len, input_size)
        # No time recurrence — each position is independent.
        e = inputs @ weights['w_i'].T + weights['b']
        w_r = weights['w_r']

        def body(s, _):
            return jnp.tanh(_normalize_jax(s) @ w_r.T + e), None

        s, _ = jax.lax.scan(body, jnp.zeros_like(e), length=self.reps)
        return s


class LrnnJax(LayerJax):
    """Depth-recurrent RNN combining time recurrence with latent reasoning.

    At each timestep, runs `reps` refinement iterations using the
    previous timestep's hidden state as additional context:

        e_t = Wi @ x_t + Wh @ normalize(h_{t-1}) + b
        s_0 = 0
        s_i = tanh(Wr @ normalize(s_{i-1}) + e_t)
        h_t = s_reps

    Latent state is always zero-initialized. Raw `x_t` is not
    normalized before the Wi projection; prepend a `norm` layer in the
    spec to recover that.
    """

    def __init__(self, input_size: int, size: int, reps: int, dtype):
        super().__init__(input_size, size, dtype)
        self.reps = reps

    def init_weights(self, rng: jax.Array) -> LayerWeights:
        k_i, k_h, k_r = jax.random.split(rng, 3)
        return {
            'w_i': xavier_uniform(
                k_i, (self.size, self.input_size), dtype=self.dtype),
            'w_h': xavier_uniform(
                k_h, (self.size, self.size), dtype=self.dtype),
            'w_r': xavier_uniform(
                k_r, (self.size, self.size), dtype=self.dtype),
            'b': jnp.zeros(self.size, dtype=self.dtype),
        }

    def init_state(self):
        return jnp.zeros(self.size, dtype=self.dtype)

    def step(
        self, weights: LayerWeights, state, x: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        # x: (input_size,), state: (size,)
        e = (weights['w_i'] @ x
             + weights['w_h'] @ _normalize_jax(state)
             + weights['b'])
        w_r = weights['w_r']

        def body(s, _):
            return jnp.tanh(w_r @ _normalize_jax(s) + e), None

        s, _ = jax.lax.scan(
            body, jnp.zeros(self.size, dtype=self.dtype), length=self.reps)
        return s, s

    def forward(
        self, weights: LayerWeights, inputs: jax.Array
    ) -> jax.Array:
        # inputs: (batch, seq_len, input_size)
        # Hoist the input projection out of the scan.
        wi_x = inputs @ weights['w_i'].T + weights['b']
        wi_x_t = jnp.transpose(wi_x, (1, 0, 2))

        w_h = weights['w_h']
        w_r = weights['w_r']
        reps = self.reps

        def time_step(h, wi):
            # wi: (batch, size), h: (batch, size)
            e = wi + _normalize_jax(h) @ w_h.T

            def reps_body(s, _):
                return jnp.tanh(_normalize_jax(s) @ w_r.T + e), None

            s, _ = jax.lax.scan(reps_body, jnp.zeros_like(e), length=reps)
            return s, s

        batch = inputs.shape[0]
        init = jnp.zeros((batch, self.size), dtype=self.dtype)
        _, outputs_t = jax.lax.scan(time_step, init, wi_x_t)
        return jnp.transpose(outputs_t, (1, 0, 2))


class LatentDef(LayerDef):
    name = "latent"

    def __init__(self, size: int, reps: int, input_size: int):
        super().__init__(input_size=input_size)
        self.size = size
        self.reps = reps

    def __str__(self) -> str:
        return f"latent.{self.size}.{self.reps}"

    def is_valid(self) -> bool:
        return (
            is_power2_int(self.size) and self.size > 1
            and is_power2_int(self.reps) and self.reps >= 2
        )

    @property
    def num_weights(self) -> int:
        # Wi(input, size) + bias + Wr(size, size), no bias on Wr.
        return self.size * self.input_size + self.size + self.size * self.size

    @property
    def num_mults(self) -> int:
        # The inner w_r matmul fires `reps` times per token; treat the
        # whole layer as a weight-count×reps workload. Slightly
        # overcounts the input projection (which only fires once), but
        # that's a small term once size >= input_size.
        return self.num_weights * self.reps

    def build_jax(self, dtype) -> LatentJax:
        return LatentJax(self.input_size, self.size, self.reps, dtype)


class LrnnDef(LayerDef):
    name = "lrnn"

    def __init__(self, size: int, reps: int, input_size: int):
        super().__init__(input_size=input_size)
        self.size = size
        self.reps = reps

    def __str__(self) -> str:
        return f"lrnn.{self.size}.{self.reps}"

    def is_valid(self) -> bool:
        return (
            is_power2_int(self.size) and self.size > 1
            and is_power2_int(self.reps) and self.reps >= 2
        )

    @property
    def num_weights(self) -> int:
        # Wi(input,size)+bias + Wh(size,size) + Wr(size,size)
        return (
            self.size * self.input_size + self.size
            + self.size * self.size + self.size * self.size
        )

    @property
    def num_mults(self) -> int:
        # Same approximation as LatentDef: w_r fires `reps` times per
        # token, so scale the whole weight count by `reps`.
        return self.num_weights * self.reps

    def build_jax(self, dtype) -> LrnnJax:
        return LrnnJax(self.input_size, self.size, self.reps, dtype)
