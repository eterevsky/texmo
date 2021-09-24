import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, device_put
from timeit import timeit
import matplotlib.pyplot as plt


def selu(x, alpha=5/3, l=1.05):
    return l * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)


key = jax.random.PRNGKey(0)
print(key)
# x = jax.random.normal(key, (10,))
# print(x)

# size = 3000
# x = jax.random.normal(key, (size, size), dtype=jnp.float32)
# x = device_put(x)
# print(timeit(lambda:jnp.dot(x, x.T).block_until_ready(), number=100))

x = jax.random.normal(key, (1000000,))
print(x)
print(timeit(lambda: selu(x).block_until_ready(), number=100))

selu_jit = jit(selu)
print(timeit(lambda: selu_jit(x).block_until_ready(), number=100))
