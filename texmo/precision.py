import enum

import jax.numpy as jnp


class Precision(enum.StrEnum):
    FP64 = enum.auto()
    FP32 = enum.auto()
    FP16 = enum.auto()
    BF16 = enum.auto()

    @property
    def jax_dtype(self) -> jnp.dtype:
        match self:
            case Precision.FP64:
                return jnp.float64
            case Precision.FP32:
                return jnp.float32
            case Precision.FP16:
                return jnp.float16
            case Precision.BF16:
                return jnp.bfloat16

    @property
    def neighbors(self):
        match self:
            case Precision.FP64:
                return (Precision.FP32,)
            case Precision.FP32:
                return (Precision.FP64, Precision.FP16, Precision.BF16)
            case Precision.FP16:
                return (Precision.FP32, Precision.BF16)
            case Precision.BF16:
                return (Precision.FP32, Precision.FP16)
