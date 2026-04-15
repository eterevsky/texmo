from typing import Optional

from .configuration import Configuration
from .dataset import DataSet, DataSetWrapper
from .run import Run


class JaxManager:
    """JAX training backend (stub — not yet implemented)."""

    def __init__(
        self,
        conf: Configuration,
        system: str,
        dataset: DataSet,
        device: str = 'auto',
        test_sample_len: int = 1024,
        test_batch: int = 1024,
    ):
        raise NotImplementedError(
            "JAX backend is not yet implemented. Use --backend torch.")
