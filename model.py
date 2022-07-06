import jax
import jax.numpy as jnp
import optax


NCHAR = 256  # Considering characters with codes 0..NCHAR


class Model(object):
    def __init__(self, **kwargs):
        self._params = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def full_name(self):
        """A name of the model, including its parameters (i.e. sizes of layers)"""
        values = '-'.join(map(str, self._params.values()))
        if values:
            return f'{self.name}-{values}'
        else:
            return self.name

    def serialize(self):
        spec = self._params.copy()
        spec['name'] = self.name
        return spec

    def init_weights(self, key):
        """Initializes the weights object.

        Args:
            key: JAX pseudorandom key, PRNGKey
        """
        return {}

    def init_state(self):
        """Initialize the state object."""
        return None

    def step(self, weights, state, c):
        """One forward step in the recurrent network.

        Args:
            weights: dictionary with the model parameters
            state: array with the internal state (initialized by init_state or returned from the
                previous step)
            c: a single character represented as a one-of. shape = (256,)

        Returns:
            (new state, an output 1D vector of size NCHAR that after softmax will produce character
            probabilities.)
        """
        pass

    def step_prob(self, weights, state, c, temperature=1.0):
        """Runs one step of the model and calculate the final probabilities.

        Arguments are the same as in step().

        Returns:
            (new state, a 1D vector of size NCHAR with probabilities of characters)
        """
        state, out = self.step(weights, state, c)
        out_softmax = jax.nn.softmax(out / temperature)
        return state, out_softmax

    def loss(self, weights, x):
        _, y = jax.lax.scan(lambda s, c: self.step(weights, s, c), self.init_state(), x)

        return optax.softmax_cross_entropy(y[:-1,:], x[1:,:])

    def loss_batch(self, weights, xbatch):
        """Implements loss for a batch provided the model implements step_batch."""
        sbatch = jnp.zeros((xbatch.shape[0], self._hidden))
        xbatch = jnp.swapaxes(xbatch, 0, 1)  # (element, batch, one-hot)
        _, ybatch = jax.lax.scan(lambda s, c: self._step_batch(weights, s, c), sbatch, xbatch)
        entropy = optax.softmax_cross_entropy(ybatch[:-1,:,:], xbatch[1:,:,:])
        return jnp.average(entropy)

    def weights_loss(self, weights):
        if not weights: return 0
        s = 0
        for v in weights.values():
            s += jnp.average(v * v)

        return s

    def total_weights(self, weights):
        n = 0
        for p in weights.values():
            if type(p) is dict:
                n += self.total_weights(p)
            else:
                n += p.flatten().shape[0]
        return n


def init_suffix(suffix_len):
    return jnp.ones((suffix_len - 1, NCHAR)) / NCHAR


def stack_suffix(prev_suffix, c):
    return jnp.vstack((prev_suffix, c.reshape((1, -1))))


