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
    """Standard GRU with separate input and hidden projections per gate.

    Matches nn.GRU equations with a single bias per gate:
        r = sigmoid(W_ir x + W_hr h + b_r)       # reset gate
        z = sigmoid(W_iz x + W_hz h + b_z)       # update gate
        n = tanh(W_in x + b_n + r * (W_hn h))    # candidate
        h_new = (1 - z) * n + z * h

    Note: unlike nn.GRU, uses one bias per gate (not two). The JAX
    param count is thus 3*size less than GruDef.num_weights.
    """

    def __init__(self, input_size: int, size: int, dtype):
        super().__init__(input_size, size, dtype)

    def init_weights(self, rng: jax.Array) -> LayerWeights:
        keys = jax.random.split(rng, 6)
        w = {}
        for i, gate in enumerate(('r', 'z', 'n')):
            w[f'w_i{gate}'] = xavier_uniform(
                keys[2 * i], (self.size, self.input_size), dtype=self.dtype)
            w[f'w_h{gate}'] = xavier_uniform(
                keys[2 * i + 1], (self.size, self.size), dtype=self.dtype)
            w[f'b_{gate}'] = jnp.zeros(self.size, dtype=self.dtype)
        return w

    def init_state(self):
        return jnp.zeros(self.size, dtype=self.dtype)

    def step(
        self, weights: LayerWeights, state, x: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        # x: (input_size,), state: (size,)
        r = jax.nn.sigmoid(
            weights['w_ir'] @ x + weights['w_hr'] @ state + weights['b_r'])
        z = jax.nn.sigmoid(
            weights['w_iz'] @ x + weights['w_hz'] @ state + weights['b_z'])
        n = jnp.tanh(
            weights['w_in'] @ x + weights['b_n']
            + r * (weights['w_hn'] @ state))
        new_state = (1 - z) * n + z * state
        return new_state, new_state

    def forward(
        self, weights: LayerWeights, inputs: jax.Array
    ) -> jax.Array:
        # inputs: (batch, seq_len, input_size)
        # Hoist input projections (and biases) for all three gates.
        Wx_r = inputs @ weights['w_ir'].T + weights['b_r']
        Wx_z = inputs @ weights['w_iz'].T + weights['b_z']
        Wx_n = inputs @ weights['w_in'].T + weights['b_n']

        w_hr = weights['w_hr']
        w_hz = weights['w_hz']
        w_hn = weights['w_hn']

        def scan_step(state, wx):
            wx_r, wx_z, wx_n = wx
            r = jax.nn.sigmoid(wx_r + state @ w_hr.T)
            z = jax.nn.sigmoid(wx_z + state @ w_hz.T)
            n = jnp.tanh(wx_n + r * (state @ w_hn.T))
            new_state = (1 - z) * n + z * state
            return new_state, new_state

        # (batch, seq_len, size) → (seq_len, batch, size)
        Wx_r_t = jnp.transpose(Wx_r, (1, 0, 2))
        Wx_z_t = jnp.transpose(Wx_z, (1, 0, 2))
        Wx_n_t = jnp.transpose(Wx_n, (1, 0, 2))

        batch = inputs.shape[0]
        init = jnp.zeros((batch, self.size), dtype=self.dtype)
        _, outputs_t = jax.lax.scan(
            scan_step, init, (Wx_r_t, Wx_z_t, Wx_n_t))
        return jnp.transpose(outputs_t, (1, 0, 2))


class MgruJax(LayerJax):
    """GRU variant with a single gate (update = 1 - forget).

        f = sigmoid(w_fx x + w_fh h + b_f)
        hc = tanh(w_hx x + w_hh (f*h) + b_h)
        h_new = (1-f) * h + f * hc

    Matches MgruModule structure: biases live on the input-side
    projections to avoid double-counting.
    """

    def init_weights(self, rng: jax.Array) -> LayerWeights:
        k_fx, k_fh, k_hx, k_hh = jax.random.split(rng, 4)
        return {
            'w_fx': xavier_uniform(k_fx, (self.size, self.input_size), dtype=self.dtype),
            'w_fh': xavier_uniform(k_fh, (self.size, self.size), dtype=self.dtype),
            'b_f': jnp.zeros(self.size, dtype=self.dtype),
            'w_hx': xavier_uniform(k_hx, (self.size, self.input_size), dtype=self.dtype),
            'w_hh': xavier_uniform(k_hh, (self.size, self.size), dtype=self.dtype),
            'b_h': jnp.zeros(self.size, dtype=self.dtype),
        }

    def init_state(self):
        return jnp.zeros(self.size, dtype=self.dtype)

    def step(
        self, weights: LayerWeights, state, x: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        # x: (input_size,), state: (size,)
        f = jax.nn.sigmoid(
            weights['w_fx'] @ x + weights['w_fh'] @ state + weights['b_f'])
        hc = jnp.tanh(
            weights['w_hx'] @ x + weights['w_hh'] @ (f * state) + weights['b_h'])
        new_state = (1 - f) * state + f * hc
        return new_state, new_state

    def forward(
        self, weights: LayerWeights, inputs: jax.Array
    ) -> jax.Array:
        # Hoist input projections for both gates.
        Fx = inputs @ weights['w_fx'].T + weights['b_f']
        Hx = inputs @ weights['w_hx'].T + weights['b_h']

        w_fh = weights['w_fh']
        w_hh = weights['w_hh']

        def scan_step(state, xs):
            fx, hx = xs
            f = jax.nn.sigmoid(fx + state @ w_fh.T)
            hc = jnp.tanh(hx + (f * state) @ w_hh.T)
            new_state = (1 - f) * state + f * hc
            return new_state, new_state

        Fx_t = jnp.transpose(Fx, (1, 0, 2))
        Hx_t = jnp.transpose(Hx, (1, 0, 2))

        batch = inputs.shape[0]
        init = jnp.zeros((batch, self.size), dtype=self.dtype)
        _, outputs_t = jax.lax.scan(scan_step, init, (Fx_t, Hx_t))
        return jnp.transpose(outputs_t, (1, 0, 2))


class MinGruJax(LayerJax):
    """Minimal GRU where the gate and candidate depend only on the input.

        z = sigmoid(w_z x + b_z)
        h_new = w_h x + b_h
        h = (1-z) * h + z * h_new
    """

    def init_weights(self, rng: jax.Array) -> LayerWeights:
        k_z, k_h = jax.random.split(rng)
        return {
            'w_z': xavier_uniform(k_z, (self.size, self.input_size), dtype=self.dtype),
            'b_z': jnp.zeros(self.size, dtype=self.dtype),
            'w_h': xavier_uniform(k_h, (self.size, self.input_size), dtype=self.dtype),
            'b_h': jnp.zeros(self.size, dtype=self.dtype),
        }

    def init_state(self):
        return jnp.zeros(self.size, dtype=self.dtype)

    def step(
        self, weights: LayerWeights, state, x: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        z = jax.nn.sigmoid(weights['w_z'] @ x + weights['b_z'])
        h_new = weights['w_h'] @ x + weights['b_h']
        new_state = (1 - z) * state + z * h_new
        return new_state, new_state

    def forward(
        self, weights: LayerWeights, inputs: jax.Array
    ) -> jax.Array:
        # z and h_new depend only on x — compute for all timesteps at once.
        z = jax.nn.sigmoid(inputs @ weights['w_z'].T + weights['b_z'])
        h_new = inputs @ weights['w_h'].T + weights['b_h']

        def scan_step(state, zh):
            zt, ht = zh
            new_state = (1 - zt) * state + zt * ht
            return new_state, new_state

        z_t = jnp.transpose(z, (1, 0, 2))
        h_new_t = jnp.transpose(h_new, (1, 0, 2))

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
