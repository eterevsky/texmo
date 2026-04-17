import jax
import jax.numpy as jnp
import torch
import torch.nn as nn
from torch import Tensor

from ..common import is_power2_int
from ..layer import LayerDef, LayerJax, LayerModule, LayerState
from ..layer_jax import LayerWeights, xavier_uniform


class GruModule(LayerModule):
    """Standard GRU backed by nn.GRU (cuDNN-optimized)."""

    def __init__(self, input_size: int, size: int):
        super().__init__()
        self.size = size
        self.gru = nn.GRU(input_size, size, batch_first=True)

    def init_state(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> LayerState:
        return torch.zeros(self.size, device=device, dtype=dtype)

    def step(
        self, state: LayerState, input: Tensor
    ) -> tuple[LayerState, Tensor]:
        _, h = self.gru(
            input.flatten().unsqueeze(0).unsqueeze(0),
            state.unsqueeze(0).unsqueeze(0),
        )
        new_state = h.squeeze(0).squeeze(0)
        return new_state, new_state

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: (batch, seq_len, input_size)
        output, _ = self.gru(inputs)
        return output


class MgruModule(LayerModule):
    """GRU variant with a single gate (update = 1 - forget).

        f = sigmoid(Wf @ [x, h] + bf)
        hc = tanh(Wh @ [x, f*h] + bh)
        h = (1-f) * h + f * hc
    """

    def __init__(self, input_size: int, size: int):
        super().__init__()
        self.size = size
        # Split the input and hidden contributions so we can precompute the
        # input part for all timesteps at once in forward(). Biases live on
        # the input-side linears to avoid double-counting.
        self.wf_x = nn.Linear(input_size, size)
        self.wf_h = nn.Linear(size, size, bias=False)
        self.wh_x = nn.Linear(input_size, size)
        self.wh_h = nn.Linear(size, size, bias=False)

    def init_state(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> LayerState:
        return torch.zeros(self.size, device=device, dtype=dtype)

    def step(
        self, state: LayerState, input: Tensor
    ) -> tuple[LayerState, Tensor]:
        x = input.flatten()
        f = torch.sigmoid(self.wf_x(x) + self.wf_h(state))
        hc = torch.tanh(self.wh_x(x) + self.wh_h(f * state))
        new_state = (1 - f) * state + f * hc
        return new_state, new_state

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: (batch, seq_len, input_size)
        # Precompute input contributions for all timesteps at once. Only the
        # hidden-state-dependent part stays inside the loop.
        f_x = self.wf_x(inputs)  # (batch, seq_len, size)
        h_x = self.wh_x(inputs)  # (batch, seq_len, size)

        batch, seq_len, _ = inputs.shape
        state = torch.zeros(
            batch, self.size, device=inputs.device, dtype=inputs.dtype)
        outputs = []
        for t in range(seq_len):
            f = torch.sigmoid(f_x[:, t] + self.wf_h(state))
            hc = torch.tanh(h_x[:, t] + self.wh_h(f * state))
            state = (1 - f) * state + f * hc
            outputs.append(state)
        return torch.stack(outputs, dim=1)


class MinGruModule(LayerModule):
    """Minimal GRU where the gate and candidate depend only on the input.

    From https://arxiv.org/abs/2305.17473

        zh = W @ x + b      (shape: 2*size)
        z = sigmoid(zh[:size])
        h_new = zh[size:]
        h = (1-z) * h + z * h_new
    """

    def __init__(self, input_size: int, size: int):
        super().__init__()
        self.size = size
        self.linear = nn.Linear(input_size, 2 * size)

    def init_state(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> LayerState:
        return torch.zeros(self.size, device=device, dtype=dtype)

    def step(
        self, state: LayerState, input: Tensor
    ) -> tuple[LayerState, Tensor]:
        zh = self.linear(input.flatten())
        z = torch.sigmoid(zh[:self.size])
        h_new = zh[self.size:]
        new_state = (1 - z) * state + z * h_new
        return new_state, new_state

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: (batch, seq_len, input_size)
        batch, seq_len, _ = inputs.shape
        # Compute z and h_new for all timesteps at once
        zh = self.linear(inputs)  # (batch, seq_len, 2*size)
        z = torch.sigmoid(zh[..., :self.size])
        h_new = zh[..., self.size:]

        state = torch.zeros(
            batch, self.size, device=inputs.device, dtype=inputs.dtype)
        outputs = []
        for t in range(seq_len):
            state = (1 - z[:, t]) * state + z[:, t] * h_new[:, t]
            outputs.append(state)
        return torch.stack(outputs, dim=1)


class GruJax(LayerJax):
    """Standard GRU with stacked gate projections.

    Matches nn.GRU equations with a single bias per gate:
        r = sigmoid(W_ir x + W_hr h + b_r)       # reset gate
        z = sigmoid(W_iz x + W_hz h + b_z)       # update gate
        n = tanh(W_in x + b_n + r * (W_hn h))    # candidate
        h_new = (1 - z) * n + z * h

    Weight layout (ordered as [r, z, n]):
        w_ih: (3*size, input_size)   — stacked input projections
        w_hrz: (2*size, size)        — stacked hidden proj for r, z
        w_hn: (size, size)           — hidden proj for n (gated by r)
        b: (3*size,)                 — stacked biases

    The n gate's hidden projection stays separate because it's
    multiplied by r before adding. The r and z gates are stacked so
    sigmoid can be applied to both at once.

    Note: unlike nn.GRU, uses one bias per gate (not two). The JAX
    param count is thus 3*size less than GruDef.num_weights.
    """

    def init_weights(self, rng: jax.Array) -> LayerWeights:
        k_ih, k_hrz, k_hn = jax.random.split(rng, 3)
        return {
            'w_ih': xavier_uniform(
                k_ih, (3 * self.size, self.input_size), dtype=self.dtype),
            'w_hrz': xavier_uniform(
                k_hrz, (2 * self.size, self.size), dtype=self.dtype),
            'w_hn': xavier_uniform(
                k_hn, (self.size, self.size), dtype=self.dtype),
            'b': jnp.zeros(3 * self.size, dtype=self.dtype),
        }

    def init_state(self):
        return jnp.zeros(self.size, dtype=self.dtype)

    def step(
        self, weights: LayerWeights, state, x: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        # x: (input_size,), state: (size,)
        s = self.size
        gates_i = weights['w_ih'] @ x + weights['b']  # (3*size,)
        r = jax.nn.sigmoid(gates_i[:s] + weights['w_hrz'][:s] @ state)
        z = jax.nn.sigmoid(gates_i[s:2 * s] + weights['w_hrz'][s:] @ state)
        n = jnp.tanh(
            gates_i[2 * s:] + r * (weights['w_hn'] @ state))
        new_state = (1 - z) * n + z * state
        return new_state, new_state

    def forward(
        self, weights: LayerWeights, inputs: jax.Array
    ) -> jax.Array:
        # inputs: (batch, seq_len, input_size)
        s = self.size
        # Hoist input projections for all 3 gates as one batched matmul.
        Wx = inputs @ weights['w_ih'].T + weights['b']  # (batch, seq_len, 3*size)
        Wx_t = jnp.transpose(Wx, (1, 0, 2))  # (seq_len, batch, 3*size)

        w_hrz = weights['w_hrz']
        w_hn = weights['w_hn']

        def scan_step(state, wx):
            # One matmul for r, z hidden projections.
            h_rz = state @ w_hrz.T  # (batch, 2*size)
            rz = jax.nn.sigmoid(wx[:, :2 * s] + h_rz)
            r = rz[:, :s]
            z = rz[:, s:]
            n = jnp.tanh(wx[:, 2 * s:] + r * (state @ w_hn.T))
            new_state = (1 - z) * n + z * state
            return new_state, new_state

        batch = inputs.shape[0]
        init = jnp.zeros((batch, self.size), dtype=self.dtype)
        _, outputs_t = jax.lax.scan(scan_step, init, Wx_t)
        return jnp.transpose(outputs_t, (1, 0, 2))


class MgruJax(LayerJax):
    """GRU variant with a single gate (update = 1 - forget).

        f = sigmoid(w_fx x + w_fh h + b_f)
        hc = tanh(w_hx x + w_hh (f*h) + b_h)
        h_new = (1-f) * h + f * hc

    Stacked input projection for both gates; hidden projections kept
    separate because hc's hidden proj is gated by f.

    Weight layout (ordered as [f, h]):
        w_ih: (2*size, input_size)   — stacked input projections
        w_fh: (size, size)           — hidden proj for f
        w_hh: (size, size)           — hidden proj for h (gated by f)
        b: (2*size,)                 — stacked biases
    """

    def init_weights(self, rng: jax.Array) -> LayerWeights:
        k_ih, k_fh, k_hh = jax.random.split(rng, 3)
        return {
            'w_ih': xavier_uniform(
                k_ih, (2 * self.size, self.input_size), dtype=self.dtype),
            'w_fh': xavier_uniform(
                k_fh, (self.size, self.size), dtype=self.dtype),
            'w_hh': xavier_uniform(
                k_hh, (self.size, self.size), dtype=self.dtype),
            'b': jnp.zeros(2 * self.size, dtype=self.dtype),
        }

    def init_state(self):
        return jnp.zeros(self.size, dtype=self.dtype)

    def step(
        self, weights: LayerWeights, state, x: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        # x: (input_size,), state: (size,)
        s = self.size
        gates_i = weights['w_ih'] @ x + weights['b']  # (2*size,)
        f = jax.nn.sigmoid(gates_i[:s] + weights['w_fh'] @ state)
        hc = jnp.tanh(gates_i[s:] + weights['w_hh'] @ (f * state))
        new_state = (1 - f) * state + f * hc
        return new_state, new_state

    def forward(
        self, weights: LayerWeights, inputs: jax.Array
    ) -> jax.Array:
        s = self.size
        # Hoist input projections for both gates as one batched matmul.
        Wx = inputs @ weights['w_ih'].T + weights['b']  # (batch, seq_len, 2*size)
        Wx_t = jnp.transpose(Wx, (1, 0, 2))

        w_fh = weights['w_fh']
        w_hh = weights['w_hh']

        def scan_step(state, wx):
            f = jax.nn.sigmoid(wx[:, :s] + state @ w_fh.T)
            hc = jnp.tanh(wx[:, s:] + (f * state) @ w_hh.T)
            new_state = (1 - f) * state + f * hc
            return new_state, new_state

        batch = inputs.shape[0]
        init = jnp.zeros((batch, self.size), dtype=self.dtype)
        _, outputs_t = jax.lax.scan(scan_step, init, Wx_t)
        return jnp.transpose(outputs_t, (1, 0, 2))


class MinGruJax(LayerJax):
    """Minimal GRU where the gate and candidate depend only on the input.

        z = sigmoid(w_z x + b_z)
        h_new = w_h x + b_h
        h = (1-z) * h + z * h_new

    Weight layout (ordered as [z, h]):
        w_ih: (2*size, input_size)   — stacked input projections
        b: (2*size,)                 — stacked biases
    """

    def init_weights(self, rng: jax.Array) -> LayerWeights:
        return {
            'w_ih': xavier_uniform(
                rng, (2 * self.size, self.input_size), dtype=self.dtype),
            'b': jnp.zeros(2 * self.size, dtype=self.dtype),
        }

    def init_state(self):
        return jnp.zeros(self.size, dtype=self.dtype)

    def step(
        self, weights: LayerWeights, state, x: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        s = self.size
        gates = weights['w_ih'] @ x + weights['b']  # (2*size,)
        z = jax.nn.sigmoid(gates[:s])
        h_new = gates[s:]
        new_state = (1 - z) * state + z * h_new
        return new_state, new_state

    def forward(
        self, weights: LayerWeights, inputs: jax.Array
    ) -> jax.Array:
        s = self.size
        # z and h_new depend only on x — compute for all timesteps at once.
        gates = inputs @ weights['w_ih'].T + weights['b']  # (batch, seq_len, 2*size)
        z = jax.nn.sigmoid(gates[..., :s])
        h_new = gates[..., s:]

        z_t = jnp.transpose(z, (1, 0, 2))
        h_new_t = jnp.transpose(h_new, (1, 0, 2))

        def scan_step(state, zh):
            zt, ht = zh
            new_state = (1 - zt) * state + zt * ht
            return new_state, new_state

        batch = inputs.shape[0]
        init = jnp.zeros((batch, self.size), dtype=self.dtype)
        _, outputs_t = jax.lax.scan(scan_step, init, (z_t, h_new_t))
        return jnp.transpose(outputs_t, (1, 0, 2))


class GruDef(LayerDef):
    name = "gru"

    def __init__(self, size: int, input_size: int):
        super().__init__(input_size=input_size)
        self.size = size

    def __str__(self) -> str:
        return f"gru.{self.size}"

    def is_valid(self) -> bool:
        return is_power2_int(self.size)

    @property
    def num_weights(self) -> int:
        # Single-bias per gate (matches GruJax). Torch's nn.GRU uses two
        # biases per gate, so its actual param count is 3*size larger.
        return 3 * self.size * (self.input_size + self.size) + 3 * self.size

    def build_module(
        self, state_dict: dict[str, Tensor] | None = None
    ) -> GruModule:
        module = GruModule(self.input_size, self.size)
        if state_dict is not None:
            module.load_state_dict(state_dict)
        return module

    def build_jax(self, dtype) -> GruJax:
        return GruJax(self.input_size, self.size, dtype)


class MgruDef(LayerDef):
    name = "mgru"

    def __init__(self, size: int, input_size: int):
        super().__init__(input_size=input_size)
        self.size = size

    def __str__(self) -> str:
        return f"mgru.{self.size}"

    def is_valid(self) -> bool:
        return is_power2_int(self.size)

    @property
    def num_weights(self) -> int:
        # 2 linear layers, each with (input+size) inputs and size outputs
        return 2 * (self.size * (self.input_size + self.size) + self.size)

    def build_module(
        self, state_dict: dict[str, Tensor] | None = None
    ) -> MgruModule:
        module = MgruModule(self.input_size, self.size)
        if state_dict is not None:
            module.load_state_dict(state_dict)
        return module

    def build_jax(self, dtype) -> MgruJax:
        return MgruJax(self.input_size, self.size, dtype)


class MinGruDef(LayerDef):
    name = "mingru"

    def __init__(self, size: int, input_size: int):
        super().__init__(input_size=input_size)
        self.size = size

    def __str__(self) -> str:
        return f"mingru.{self.size}"

    def is_valid(self) -> bool:
        return is_power2_int(self.size)

    @property
    def num_weights(self) -> int:
        # nn.Linear(input_size, 2*size): 2*size*input_size + 2*size
        return 2 * self.size * self.input_size + 2 * self.size

    def build_module(
        self, state_dict: dict[str, Tensor] | None = None
    ) -> MinGruModule:
        module = MinGruModule(self.input_size, self.size)
        if state_dict is not None:
            module.load_state_dict(state_dict)
        return module

    def build_jax(self, dtype) -> MinGruJax:
        return MinGruJax(self.input_size, self.size, dtype)
