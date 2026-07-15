from .manager import Manager


class ManagerTorch(Manager):
    """PyTorch training backend -- parked.

    The legacy full-model torch runtime (`Model`) was retired with
    `ModelDef`; Model2 has a JAX runtime only. The per-layer
    `*Module` classes are still maintained and tested, so reviving
    this backend means writing a Model2-shaped torch runtime on top
    of them, not resurrecting the old one.
    """

    def __init__(self, device: str = 'auto', **kwargs):
        raise NotImplementedError(
            "The torch backend is parked: Model2 has no torch "
            "runtime (the legacy Model was retired). Use "
            "backend='jax'.")
