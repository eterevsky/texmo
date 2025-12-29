import jax
import jax.numpy as jnp
from math import log2

from ..common import is_power2_int
from ..layer import Layer, LayerState, LayerWeights
from ..prng import Rng
from .registry import layer_cls


_E_INV_SQRT = jnp.e ** -0.5

group_norm = lambda x, w, b : ((x - x.mean(axis=1,keepdims=1)) / (x.var(axis=1,keepdims=1) + 64e-5)**0.5).flatten() * w + b


@layer_cls
class Rwkv(Layer):
    name = 'rwkv'

    def __init__(self, size: int, heads: int, input_shape=None):
        super().__init__(input_shape=input_shape)
        self.size = size
        self.heads = heads
        self.head_size = size // heads
        self.output_shape = (size,)

    def __str__(self) -> str:
        return f'rwkv.{self.size}.{self.heads}'

    def is_valid(self) -> bool:
        return (
            self.heads >= 1 and
            self.size >= self.heads * 2 and
            is_power2_int(self.size) and
            is_power2_int(self.heads))

    def neighbors(self):
        yield f'rwkv.{self.size}.{self.heads*2}'
        yield f'rwkv.{self.size}.{self.heads//2}'
        yield f'rwkv.{self.size*2}.{self.heads}'
        yield f'rwkv.{self.size//2}.{self.heads}'

        n = int(log2(self.size))
        default_heads = 2 ** ((n + 2) // 3)
        if self.heads == default_heads:
            yield f'lstm.{self.size}'

    @property
    def weights(self) -> int:
        return (
            self.input_size * self.size +
            self.size * 20 +
            self.heads * self.head_size +
            self.size * self.head_size * 6 +
            self.size * self.size * 12)

    def init_weights(self, rng: Rng, _init_scale: float, dtype) -> LayerWeights:
        return {
            'inputW': rng.he((self.size, self.input_size), dtype=dtype),
            'preW': rng.normal(self.size, dtype=dtype),
            'preB': rng.normal(self.size, dtype=dtype),
            'midW': rng.normal(self.size, dtype=dtype),
            'midB': rng.normal(self.size, dtype=dtype),
            'att': {
                'mr': rng.normal(self.size, dtype=dtype),
                'mw': rng.normal(self.size, dtype=dtype),
                'mk': rng.normal(self.size, dtype=dtype),
                'mv': rng.normal(self.size, dtype=dtype),
                'ma': rng.normal(self.size, dtype=dtype),
                'mg': rng.normal(self.size, dtype=dtype),
                'w_bias': jnp.zeros(self.size, dtype=dtype),
                'r_k': rng.he((self.heads, self.head_size, 1), dtype=dtype),
                'Ww1': jnp.zeros((self.size, self.head_size), dtype=dtype),
                'Ww2': rng.he((self.head_size, self.size), dtype=dtype),
                'Wa1': jnp.zeros((self.size, self.head_size), dtype=dtype),
                'Wa2': rng.he((self.head_size, self.size), dtype=dtype),
                'Wg1': jnp.zeros((self.size, self.head_size), dtype=dtype),
                'Wg2': rng.he((self.head_size, self.size), dtype=dtype),
                'a_bias': rng.normal(self.size, dtype=dtype),
                'k_k': rng.normal(self.size, dtype=dtype),
                'k_a': rng.normal(self.size, dtype=dtype),
                'Wr': rng.he((self.size, self.size), dtype=dtype),
                'Wk': rng.he((self.size, self.size), dtype=dtype),
                'Wv': rng.he((self.size, self.size), dtype=dtype),
                'Wo': rng.he((self.size, self.size), dtype=dtype),
                'ln_w': rng.normal(self.size, dtype=dtype),
                'ln_b': jnp.zeros(self.size, dtype=dtype),
            },
            'ffn': {
                'mix': rng.normal(self.size, dtype=dtype),
                'Wk': rng.he((4*self.size, self.size), dtype=dtype),
                'Wv': jnp.zeros((self.size, 4*self.size), dtype=dtype),
            },
        }

    def init_state(self, _weights, dtype):
        return {
            'x': jnp.zeros((self.size,), dtype=dtype),
            'xx': jnp.zeros((self.size,), dtype=dtype),
            'a': jnp.zeros((self.heads, self.head_size, self.head_size), dtype=dtype)
        }

    def step(
            self,
            weights: LayerWeights,
            state: LayerState,
            input: jax.Array,
            dtype) -> tuple[LayerState, jax.Array]:
        input = input.flatten()
        new_state = {}

        x = weights['inputW'] @ input

        x = x / (jnp.linalg.norm(input) + 1E-5)
        x_ = x * weights['preW'] + weights['preB']

        dx, new_state['x'], new_state['a'] = self._time_mixing(
            x_, state['x'], state['a'], weights['att'], dtype)

        x = x + dx

        x_ = x / (jnp.linalg.norm(x) + 1E-5)
        x_ = x_ * weights['midW'] + weights['midB']

        dx, new_state['xx'] = self._channel_mixing(
            x_, state['xx'], weights['ffn']['mix'], weights['ffn']['Wk'],
            weights['ffn']['Wv'])
        x = x + dx

        return new_state, x

    def _time_mixing(self, x, last_x, S, weights, dtype):
        dx = last_x - x

        xr = x + weights['mr'] * dx
        xw = x + weights['mw'] * dx
        xk = x + weights['mk'] * dx
        xv = x + weights['mv'] * dx
        xa = x + weights['ma'] * dx
        xg = x + weights['mg'] * dx

        r = weights['Wr'] @ xr
        w = jnp.exp(-jax.nn.sigmoid(jnp.tanh(xw @ weights['Ww1']) @ weights['Ww2'] + weights['w_bias']) * _E_INV_SQRT)
        k = weights['Wk'] @ xk
        v = weights['Wv'] @ xv
        a = jax.nn.sigmoid(
            xa @ weights['Wa1'] @ weights['Wa2'] + weights['a_bias'])
        g = jax.nn.sigmoid(xg @ weights['Wg1']) @ weights['Wg2']
        kk = k * weights['k_k']
        k += k * (a-1) * weights['k_a']

        r,w,k,v,kk,a = [
            i.reshape(self.heads, self.head_size, 1) for i in [r,w,k,v,kk,a]
        ]
        kk /= jnp.maximum(jnp.linalg.norm(kk, axis=1, keepdims=1), 1e-12)

        S = S * w.mT - S @ kk * (kk*a).mT + v * k.mT
        y = S @ r

        y = group_norm(y, weights['ln_w'], weights['ln_b'])
        y += ((r * k * weights['r_k']).sum(axis=1,keepdims=1) * v).flatten()
        return weights['Wo'] @ (y * g), x, S

    def _channel_mixing(self, x, last_x, mix, Wk, Wv):
        k = Wk @ ( x + mix * (last_x - x) )
        v = Wv @ jnp.maximum(k, 0)**2
        return v, x


