import jax
import jax.nn as jax_nn
import jax.numpy as jnp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..common import is_power2_int
from ..layer import LayerDef, LayerJax, LayerModule, LayerState
from ..layer_jax import LayerWeights, xavier_uniform

_JAX_ACTIVATIONS = {
    "relu": jax_nn.relu,
    "tanh": jnp.tanh,
    "gelu": jax_nn.gelu,
}


class RnnModule(LayerModule):
    """Simple Elman RNN backed by nn.RNN (cuDNN-optimized)."""

    def __init__(self, input_size: int, size: int, nonlinearity: str):
        super().__init__()
        self.size = size
        self.rnn = nn.RNN(input_size, size, nonlinearity=nonlinearity, batch_first=True)

    def init_state(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> LayerState:
        return torch.zeros(self.size, device=device, dtype=dtype)

    def step(
        self, state: LayerState, input: Tensor
    ) -> tuple[LayerState, Tensor]:
        # input: (input_size,), state: (size,)
        _, h = self.rnn(
            input.flatten().unsqueeze(0).unsqueeze(0),
            state.unsqueeze(0).unsqueeze(0),
        )
        new_state = h.squeeze(0).squeeze(0)
        return new_state, new_state

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: (batch, seq_len, input_size)
        output, _ = self.rnn(inputs)
        return output


class RnnGELUModule(LayerModule):
    """Elman RNN with GELU activation (not supported by nn.RNN)."""

    def __init__(self, input_size: int, size: int):
        super().__init__()
        self.size = size
        self.linear = nn.Linear(input_size + size, size)

    def init_state(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> LayerState:
        return torch.zeros(self.size, device=device, dtype=dtype)

    def step(
        self, state: LayerState, input: Tensor
    ) -> tuple[LayerState, Tensor]:
        input_state = torch.cat([input.flatten(), state])
        new_state = F.gelu(self.linear(input_state))
        return new_state, new_state

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: (batch, seq_len, input_size)
        batch, seq_len, _ = inputs.shape
        state = torch.zeros(batch, self.size, device=inputs.device, dtype=inputs.dtype)
        outputs = []
        for t in range(seq_len):
            state = F.gelu(self.linear(torch.cat([inputs[:, t, :], state], dim=-1)))
            outputs.append(state)
        return torch.stack(outputs, dim=1)


class RnnJax(LayerJax):
    """Elman RNN with separate input and hidden projections.

    Splits the weight matrix so forward() can hoist the input
    projection out of the scan loop (one batched matmul over the full
    sequence instead of one per timestep). Each matrix is Xavier-init
    based on its own fan_in/fan_out. Uses a single bias.

    Weights:
        w_ih: (size, input_size)
        w_hh: (size, size)
        b: (size,)

    Note: unlike Torch's nn.RNN, this uses one bias vector instead of
    two (bias_ih + bias_hh). The JAX param count is thus `size` less
    than RnnDef.num_weights for tanh/relu activations.
    """

    def __init__(self, input_size: int, size: int, activation_name: str, dtype):
        super().__init__(input_size, size, dtype)
        self.activation = _JAX_ACTIVATIONS[activation_name]

    def init_weights(self, rng: jax.Array) -> LayerWeights:
        k_ih, k_hh = jax.random.split(rng)
        return {
            'w_ih': xavier_uniform(
                k_ih, (self.size, self.input_size), dtype=self.dtype),
            'w_hh': xavier_uniform(
                k_hh, (self.size, self.size), dtype=self.dtype),
            'b': jnp.zeros(self.size, dtype=self.dtype),
        }

    def init_state(self):
        return jnp.zeros(self.size, dtype=self.dtype)

    def step(
        self, weights: LayerWeights, state, x: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        # x: (input_size,), state: (size,)
        new_state = self.activation(
            weights['w_ih'] @ x + weights['w_hh'] @ state + weights['b'])
        return new_state, new_state

    def forward(
        self, weights: LayerWeights, inputs: jax.Array
    ) -> jax.Array:
        # inputs: (batch, seq_len, input_size)
        w_hh = weights['w_hh']
        # Hoist input projection out of the scan: one batched matmul
        # over the whole sequence.
        Wx = inputs @ weights['w_ih'].T + weights['b']
        # (batch, seq_len, size) → (seq_len, batch, size) for scan
        Wx_t = jnp.transpose(Wx, (1, 0, 2))

        def scan_step(state, wx_batch):
            # state: (batch, size), wx_batch: (batch, size)
            new_state = self.activation(wx_batch + state @ w_hh.T)
            return new_state, new_state

        batch = inputs.shape[0]
        init = jnp.zeros((batch, self.size), dtype=self.dtype)
        _, outputs_t = jax.lax.scan(scan_step, init, Wx_t)
        return jnp.transpose(outputs_t, (1, 0, 2))


class RnnDef(LayerDef):
    name = "rnn"

    def __init__(self, size: int, input_size: int, activation: str | None = None):
        super().__init__(input_size=input_size)
        self.size = size
        self._activation = activation

    def __str__(self) -> str:
        if self._activation:
            return f"rnn.{self.size}.{self._activation}"
        return f"rnn.{self.size}"

    def is_valid(self) -> bool:
        # relu is retired (see DenseDef.is_valid): parses and runs,
        # but no longer search-valid.
        return (is_power2_int(self.size)
                and self._activation in ("tanh", "gelu"))

    @property
    def num_weights(self) -> int:
        # Single-bias count (matches RnnJax). Torch's nn.RNN uses two
        # biases for tanh/relu, so its actual param count will be
        # `size` larger than this.
        s = self.size
        return s * self.input_size + s * s + s

    def build_module(self, state_dict: dict[str, Tensor] | None = None) -> LayerModule:
        if self._activation == "gelu":
            module = RnnGELUModule(self.input_size, self.size)
        else:
            module = RnnModule(self.input_size, self.size, self._activation)
        if state_dict is not None:
            module.load_state_dict(state_dict)
        return module

    def build_jax(self, dtype) -> RnnJax:
        return RnnJax(self.input_size, self.size, self._activation, dtype)
