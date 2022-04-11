import jax
import jax.numpy as jnp
import optax


NCHAR = 256  # Considering characters with codes 0..NCHAR


class Model(object):
    def init_params(self, key):
        """Initializes the params object.

        Args:
            key: JAX pseudorandom key, PRNGKey
        """
        return {}

    def init_state(self):
        """Initialize the state object."""
        return None

    def step(self, params, state, c):
        """One forward step in the recurrent network.

        Args:
            params: dictionary with the model parameters
            state: array with the internal state (initialized by init_state or returned from the
                previous step)
            x: a single character represented as a one-of. shape = (256,)

        Returns:
            (new state, an output 1D vector of size NCHAR that after softmax will produce character
            probabilities.)
        """
        pass

    def step_prob(self, params, state, c, temperature=1.0):
        """Runs one step of the model and calculate the final probabilities.

        Arguments are the same as in step().

        Returns:
            (new state, a 1D vector of size NCHAR with probabilities of characters)
        """
        state, out = self.step(params, state, c)
        out_softmax = jax.nn.softmax(out / temperature)
        return state, out_softmax

    def loss(self, params, x):
        _, y = jax.lax.scan(lambda s, c: self.step(params, s, c), self.init_state(), x)

        return optax.softmax_cross_entropy(y[:-1,:], x[1:,:])

    def loss_batch(self, params, xbatch):
        """Implements loss for a batch provided the model implements step_batch."""
        sbatch = jnp.zeros((xbatch.shape[0], self._hidden))
        xbatch = jnp.swapaxes(xbatch, 0, 1)  # (element, batch, one-hot)
        _, ybatch = jax.lax.scan(lambda s, c: self._step_batch(params, s, c), sbatch, xbatch)
        entropy = optax.softmax_cross_entropy(ybatch[:-1,:,:], xbatch[1:,:,:])
        return jnp.average(entropy)

    def params_loss(self, params):
        if not params: return 0
        s = 0
        for v in params.values():
            s += jnp.average(v * v)

        return s

