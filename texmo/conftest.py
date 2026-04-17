import jax

# Tests always run on CPU.
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platforms', 'cpu')
