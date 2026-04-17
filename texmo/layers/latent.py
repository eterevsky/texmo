import jax
import jax.numpy as jnp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..common import is_power2_int
from ..layer import LayerDef, LayerJax, LayerModule, LayerState
from ..layer_jax import LayerWeights, xavier_uniform


def _normalize_jax(x: jax.Array, eps: float = 1e-5) -> jax.Array:
    """L2-normalize along the last axis. Matches PyTorch's F.normalize."""
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / jnp.maximum(norm, eps)


def _normalize(x: Tensor) -> Tensor:
    return F.normalize(x, dim=-1, eps=1e-5)


class LatentModule(LayerModule):
    """Depth-recurrent dense layer (latent reasoning).

    Inspired by https://arxiv.org/abs/2502.05171.

    Each call iterates a shared block `reps` times on a fresh latent state:

        e = Wi @ normalize(x) + b
        s_0 = 0
        s_i = tanh(Wr @ normalize(s_{i-1}) + e)   for i = 1..reps
        out = s_reps

    The input contribution `e` is re-injected at every iteration; this is
    what makes the recurrence stable and gives the layer its "iterative
    refinement" character.
    """

    def __init__(self, input_size: int, size: int, reps: int):
        super().__init__()
        self.size = size
        self.reps = reps
        self.wi = nn.Linear(input_size, size)
        self.wr = nn.Linear(size, size, bias=False)
        # Set to True in tests for deterministic output. The paper inits
        # the latent state randomly to encourage path independence; we
        # follow that in both train and eval modes.
        self._deterministic_init = False

    def _init_latent(self, e: Tensor) -> Tensor:
        if self._deterministic_init:
            return torch.zeros_like(e)
        return torch.randn_like(e)

    def step(
        self, state: LayerState, input: Tensor
    ) -> tuple[LayerState, Tensor]:
        e = self.wi(_normalize(input))
        s = self._init_latent(e)
        for _ in range(self.reps):
            s = torch.tanh(self.wr(_normalize(s)) + e)
        return None, s

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: (batch, seq_len, input_size)
        e = self.wi(_normalize(inputs))  # (batch, seq_len, size)
        s = self._init_latent(e)
        for _ in range(self.reps):
            s = torch.tanh(self.wr(_normalize(s)) + e)
        return s


class LrnnModule(LayerModule):
    """Depth-recurrent RNN: combines time-recurrence with latent reasoning.

    At each timestep, runs `reps` refinement iterations using the previous
    timestep's hidden state as additional context:

        e_t = Wi @ normalize(x_t) + Wh @ normalize(h_{t-1}) + b
        s_0 = 0
        s_i = tanh(Wr @ normalize(s_{i-1}) + e_t)
        h_t = s_reps
    """

    def __init__(self, input_size: int, size: int, reps: int):
        super().__init__()
        self.size = size
        self.reps = reps
        self.wi = nn.Linear(input_size, size)
        self.wh = nn.Linear(size, size, bias=False)
        self.wr = nn.Linear(size, size, bias=False)
        # Set to True in tests for deterministic output. The paper inits
        # the latent state randomly to encourage path independence; we
        # follow that in both train and eval modes.
        self._deterministic_init = False

    def init_state(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> LayerState:
        return torch.zeros(self.size, device=device, dtype=dtype)

    def _init_latent(self, e: Tensor) -> Tensor:
        if self._deterministic_init:
            return torch.zeros_like(e)
        return torch.randn_like(e)

    def step(
        self, state: LayerState, input: Tensor
    ) -> tuple[LayerState, Tensor]:
        e = self.wi(_normalize(input)) + self.wh(_normalize(state))
        s = self._init_latent(e)
        for _ in range(self.reps):
            s = torch.tanh(self.wr(_normalize(s)) + e)
        return s, s

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: (batch, seq_len, input_size)
        # Precompute the input contribution for all timesteps at once.
        wi_x = self.wi(_normalize(inputs))  # (batch, seq_len, size)

        batch, seq_len, _ = inputs.shape
        h = torch.zeros(
            batch, self.size, device=inputs.device, dtype=inputs.dtype)
        outputs = []
        for t in range(seq_len):
            e = wi_x[:, t] + self.wh(_normalize(h))
            s = self._init_latent(e)
            for _ in range(self.reps):
                s = torch.tanh(self.wr(_normalize(s)) + e)
            h = s
            outputs.append(h)
        return torch.stack(outputs, dim=1)


class LatentJax(LayerJax):
    """Depth-recurrent dense layer (see LatentModule).

    Latent state is always zero-initialized (no RNG needed).
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
        e = weights['w_i'] @ _normalize_jax(x) + weights['b']
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
        e = _normalize_jax(inputs) @ weights['w_i'].T + weights['b']
        w_r = weights['w_r']

        def body(s, _):
            return jnp.tanh(_normalize_jax(s) @ w_r.T + e), None

        s, _ = jax.lax.scan(body, jnp.zeros_like(e), length=self.reps)
        return s


class LrnnJax(LayerJax):
    """Depth-recurrent RNN combining time recurrence with latent reasoning.

    See LrnnModule. Latent state is always zero-initialized.
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
        e = (weights['w_i'] @ _normalize_jax(x)
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
        # Hoist the input projection (with normalization) out of the scan.
        wi_x = (_normalize_jax(inputs) @ weights['w_i'].T + weights['b'])
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
        # Linear(input, size) + bias + Linear(size, size, bias=False)
        return self.size * self.input_size + self.size + self.size * self.size

    def build_module(
        self, state_dict: dict[str, Tensor] | None = None
    ) -> LatentModule:
        module = LatentModule(self.input_size, self.size, self.reps)
        if state_dict is not None:
            module.load_state_dict(state_dict)
        return module

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

    def build_module(
        self, state_dict: dict[str, Tensor] | None = None
    ) -> LrnnModule:
        module = LrnnModule(self.input_size, self.size, self.reps)
        if state_dict is not None:
            module.load_state_dict(state_dict)
        return module

    def build_jax(self, dtype) -> LrnnJax:
        return LrnnJax(self.input_size, self.size, self.reps, dtype)
